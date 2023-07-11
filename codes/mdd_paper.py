import random
import time
import warnings
import sys
import argparse
import copy
import os
from typing import Optional, List, Dict

import torch
import torch.nn.parallel
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('.')
from dalib.adaptation.mdd import MarginDisparityDiscrepancy, ImageClassifier
import dalib.vision.datasets as datasets
import dalib.vision.models as models
from tools.utils import AverageMeter, ProgressMeter, accuracy, ForeverDataIterator
from tools.transforms import ResizeImage
from tools.lr_scheduler import StepwiseLR

from breeds import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np

class Wrapped_Encoder(nn.Module):
    def __init__(self, encoder, out_features):
        super().__init__()
        self.encoder = encoder
        self.out_features = out_features

    def forward(self, x):
        return self.encoder(x)

def main(args: argparse.Namespace):
    seed = args.seed
    args.seed = 0
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True
    args.seed = seed 

    # Setup wandb
    if args.wandb:
        global wandb
        import wandb
        wandb.init(entity="caitianle1998", project=args.project,
                    config=args, reinit=True, settings=wandb.Settings(symlink=False))

    # Data loading code
    train_source_dataset, pseudo_val_dataset, train_target_dataset, val_dataset, subclass_to_ratio= get_breeds_loaders(ratio=args.ratio, data_dir=args.root)
    print(len(train_source_dataset))
    
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    pseudo_val_loader = DataLoader(pseudo_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    if args.arch == 'swav':
        model = models.resnet50(pretrained=False)#torch.hub.load('facebookresearch/swav', 'resnet50')
        state = torch.load('swav_800ep_pretrain.pth.tar', map_location='cpu')
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        state_dict = torch.load('swav_800ep_eval_linear.pth.tar', map_location='cpu')['state_dict']
        encoder = nn.Sequential(
                *(list(model.children())[:-1]), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
        print(encoder)
        backbone = Wrapped_Encoder(encoder, 2048)
    else:
        backbone = models.__dict__[args.arch](pretrained=True).to(device)
    num_classes = 30
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 width=args.bottleneck_dim).to(device)
    mdd = MarginDisparityDiscrepancy(args.margin).to(device)

    # define optimizer and lr_scheduler
    # The learning rate of the classiﬁers are set 10 times to that of the feature extractor by default.
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = StepwiseLR(optimizer, init_lr=args.lr, gamma=args.lr_gamma, decay_rate=0.75)

    #classifier = nn.DataParallel(classifier)

    # start training
    best_acc1 = 0.
    source_best_acc1 = 0.
    best_model = classifier.state_dict()
    for epoch in range(args.epochs):
        if args.wandb:
            log = {'optim/lr': lr_scheduler.get_lr()}
            wandb.log(log, step=epoch)
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, mdd, optimizer,
              lr_scheduler, epoch, args)

        # evaluate on validation set
        source_acc1 = validate(pseudo_val_loader, classifier, epoch, args, 'source_')
        acc1 = validate(val_loader, classifier, epoch, args)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())
        best_acc1 = max(acc1, best_acc1)
        source_best_acc1 = max(source_acc1, source_best_acc1)
        if args.wandb:
            wandb.log({'test/best_acc': best_acc1, 'test/source_best_acc': source_best_acc1}, step=epoch)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(best_model)
    acc1 = validate(test_loader, classifier, epoch, args)
    print("test_acc1 = {:3.1f}".format(acc1))


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          classifier: ImageClassifier, mdd: MarginDisparityDiscrepancy, optimizer: SGD,
          lr_scheduler: StepwiseLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    classifier.train()
    mdd.train()

    criterion = nn.CrossEntropyLoss().to(device)

    end = time.time()
    for i in range(args.iters_per_epoch):
        lr_scheduler.step()
        optimizer.zero_grad()

        # measure data loading time
        data_time.update(time.time() - end)

        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        outputs, outputs_adv = classifier(x)
        y_s, y_t = outputs.chunk(2, dim=0)
        y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

        # compute cross entropy loss on source domain
        cls_loss = criterion(y_s, labels_s)
        # compute margin disparity discrepancy between domains
        transfer_loss = mdd(y_s, y_s_adv, y_t, y_t_adv)
        loss = cls_loss + transfer_loss * args.trade_off
        classifier.step()

        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_t.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    if args.wandb:
        log = {'train/train_loss': losses.avg, 'train/transfer_loss': trans_losses.avg,
            'train/source_acc': cls_accs.avg, 'train/target_acc': tgt_accs.avg}
        wandb.log(log, step=epoch)


def validate(val_loader: DataLoader, model: ImageClassifier, epoch: int, args: argparse.Namespace, prefix='') -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    if args.wandb:
        log = {'test/loss': losses.avg, 'test/'+prefix+'_acc': top1.avg, 'test/'+prefix+'_acc5':top5.avg}
        wandb.log(log, step=epoch)

    return top1.avg


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='swav',help='backbone architecture')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.004, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--margin', type=float, default=4., help="margin gamma")
    parser.add_argument('--bottleneck-dim', default=2048, type=int)
    parser.add_argument('--center-crop', default=False, action='store_true')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--lr-gamma', default=0.0002, type=float)
    parser.add_argument('--wandb', action='store_true', help='use wandb')
    parser.add_argument('--project', default='da', type=str, help='project name')
    parser.add_argument('--note', default='', type=str, help='note')
    parser.add_argument('--ratio', default=1, type=float)
    args = parser.parse_args()
    args.root = '/shared/Imagenet'
    print(args)
    main(args)

