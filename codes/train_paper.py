import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.models as models
from tqdm import tqdm

from dataset.randaugment import RandAugmentMC
from dataset.breeds import get_breeds_loaders
from robustness.data_augmentation import *
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0
source_best_acc = 0


class TransformFixMatch(object):
    def __init__(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
        )
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='entity30', type=str,
                        help='dataset name in breeds')
    parser.add_argument('--breeds-ratio', default=1, type=float,
                        help='initial learning rate')
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of epochs to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                        help='initial learning rate')
    parser.add_argument('--cosine', action='store_true',
                        help='cosine learning rate')
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="lr decay factor")
    parser.add_argument('--warmup-epochs', default=0, type=int,
                        help='warmup epochs')
    parser.add_argument('--warmup-factor', default=4, type=int,
                        help='use larger labeled batch size')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--weak', action='store_true',
                        help='pseudo label on weak augmentation')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=3, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=10, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--end-lambda-u', default=0, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--unlabel-data', default='target', type=str,
                        help='use unlabeled data from (source, target or both)')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=0, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--wandb', action='store_true', help='use wandb')
    parser.add_argument('--project', default='da',
                        type=str, help='project name')
    parser.add_argument('--note', default='', type=str, help='note')
    parser.add_argument('--prob-decay', default=0.9, type=float, help='moving average of marginal distribution')
    parser.add_argument('--da', action='store_true', help='use distribution alignment')

    args = parser.parse_args()
    args.lr_last_layer = args.lr * 20
    if args.end_lambda_u == 0:
        args.end_lambda_u = args.lambda_u
    
    global best_acc

    #args.num_classes = 

    def create_model(args):
        model = models.resnet50(pretrained=False)
        state = torch.load('swav_800ep_pretrain.pth.tar', map_location='cpu')
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.fc = nn.Linear(2048, 30)
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    if args.wandb and args.local_rank in [-1, 0]:
        global wandb
        import wandb
        wandb.init(entity="caitianle1998", project=args.project,
                   config=args, reinit=True, settings=wandb.Settings(symlink=False))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)

    # Fix dataset
    seed = args.seed
    args.seed = 0
    set_seed(args)
    args.seed = seed

    source_train_set, source_test_set, target_train_set, target_test_set, subclass_to_ratio = get_breeds_loaders(
        source_train_transform=TransformFixMatch(), target_train_transform=TransformFixMatch(), 
        ratio=args.breeds_ratio, dataset='make_'+args.dataset)

    if args.seed is not None:
        set_seed(args)

    logger.info(f"  Number of train data: {len(source_train_set)}")

    labeled_dataset = source_train_set

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    warmup_labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size*args.warmup_factor,
        num_workers=args.num_workers,
        drop_last=True, pin_memory=True)

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True, pin_memory=True)

    if args.unlabel_data == 'source':
        unlabeled_dataset = source_train_set
    elif args.unlabel_data == 'target':
        unlabeled_dataset = target_train_set
    else:
        unlabeled_dataset = torch.utils.data.ConcatDataset([source_train_set]+[target_train_set]*int(args.unlabel_data))

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True, pin_memory=True)

    source_test_loader = DataLoader(
        source_test_set,
        sampler=SequentialSampler(source_test_set),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    target_test_loader = DataLoader(
        target_test_set,
        sampler=SequentialSampler(target_test_set),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if 'fc' in n],
         'lr': args.lr_last_layer},
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n]},
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [int(args.epochs*0.5), int(args.epochs*0.75)], gamma=args.gamma
        )

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        #args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location=args.device)
        best_acc = checkpoint['best_acc']
        #args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        if args.amp and checkpoint.get('amp') is not None:
            amp.load_state_dict(checkpoint['amp'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint['scheduler'])
        
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.breeds_ratio}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.epochs * args.eval_step}")

    model.zero_grad()
    train(args, warmup_labeled_trainloader, labeled_trainloader, unlabeled_trainloader, source_test_loader, target_test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, warmup_labeled_trainloader, labeled_trainloader, unlabeled_trainloader, source_test_loader, target_test_loader,
          model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    global best_acc, source_best_acc
    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    mask_probs = AverageMeter()
    end = time.time()

    labeled_iter = iter(warmup_labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    for group in optimizer.param_groups:
        group['lr'] = group['lr'] * args.warmup_factor

    eval_factor = args.warmup_factor

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        if epoch == args.warmup_epochs:
            labeled_iter = iter(labeled_trainloader)
            for group in optimizer.param_groups:
                group['lr'] = group['lr'] / args.warmup_factor
            eval_factor = 1
            prob_avg = torch.ones(30).to(args.device) / 30.
            
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step//eval_factor),
                         disable=args.local_rank not in [-1, 0])

        for batch_idx in range(args.eval_step//eval_factor):
            try:
                (inputs_x, _), targets_x = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                (inputs_x, _), targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            if epoch < args.warmup_epochs:
                inputs = inputs_x.to(args.device, non_blocking=True)
                targets_x = targets_x.to(args.device, non_blocking=True)
                logits_x = model(inputs)
                loss = F.cross_entropy(logits_x, targets_x, reduction='mean')
                lambda_u = 0
            else:
                lambda_u = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * (
                    args.end_lambda_u - args.lambda_u) + args.lambda_u
                inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(args.device, non_blocking=True)
                targets_x = targets_x.to(args.device, non_blocking=True)
                logits = model(inputs)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                pseudo_label = torch.softmax(
                    logits_u_w.detach()/args.T, dim=-1)
                if args.da:
                    pseudo_label = pseudo_label / prob_avg.detach().clone().unsqueeze(0)
                    pseudo_label = pseudo_label / pseudo_label.sum(dim=-1, keepdim=True)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()
                if args.weak:
                    Lu = (F.cross_entropy(logits_u_w, targets_u,
                                      reduction='none') * mask).mean()
                else:
                    Lu = (F.cross_entropy(logits_u_s, targets_u,
                                      reduction='none') * mask).mean()

                prob_u_w = torch.softmax(logits_u_w/args.T, dim=-1)
                prob_u_w_avg = prob_u_w.mean(dim=0)
                prob_avg = args.prob_decay * prob_avg + (1 - args.prob_decay) * prob_u_w_avg
                
                loss = Lx + lambda_u * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            if epoch >= args.warmup_epochs:
                losses_x.update(Lx.item())
                losses_u.update(Lu.item())
                mask_probs.update(mask.mean().item())
                
            optimizer.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            if epoch >= args.warmup_epochs:
                prob_avg = prob_avg.detach()
                if args.local_rank != -1:
                    torch.distributed.all_reduce(prob_avg)
                    prob_avg = prob_avg / prob_avg.sum()
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step//eval_factor,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(
                args, target_test_loader, test_model, epoch, 'target_')
            source_test_loss, source_test_acc = test(
                args, source_test_loader, test_model, epoch, 'source_')

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            source_best_acc = max(source_test_acc, source_best_acc)

            if args.wandb:
                log = {'train/train_loss': losses.avg, 'train/train_loss_source': losses_x.avg, 'train/train_loss_target': losses_u.avg, 'train/mask': mask_probs.avg, 'test/source_loss': source_test_loss, 'test/source_acc': source_test_acc, 'test/target_loss': test_loss, 'test/target_acc': test_acc, 'test/best_acc': best_acc, 'test/source_best_acc': source_best_acc, 'optim/lr': scheduler.get_last_lr()[0], 'optim/lambda_u': lambda_u}
                wandb.log(log, step=epoch)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'amp': amp.state_dict() if args.amp else None
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))
        scheduler.step()


def test(args, test_loader, model, epoch, prefix=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
