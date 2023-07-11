from robustness import datasets
from robustness.tools import folder
from robustness.tools.breeds_helpers import *
import torch.utils.data as data
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch

def get_label_mapping(ranges, label):
    '''map subclasses in ranges to label'''
    def custom_label_mapping(classes, class_to_idx):
        mapping = {}
        for class_name, idx in class_to_idx.items():
            for new_idx, range_set in enumerate(ranges):
                if idx in range_set:
                    mapping[class_name] = label

        filtered_classes = list(mapping.keys()).sort()
        return filtered_classes, mapping
    return custom_label_mapping

class CustomImageNetDataset(datasets.CustomImageNet):
    def __init__(self, data_path, custom_grouping, label=None, **kwargs):
        super(CustomImageNetDataset, self).__init__(data_path, custom_grouping, **kwargs)
        if label is not None:
            self.label_mapping = get_label_mapping(custom_grouping, label)
            
    def make_datasets(self, transform_train=None, transform_test=None):
        if transform_train is None:
            transform_train = self.transform_train
        if transform_test is None:
            transform_test = self.transform_test
        data_path = self.data_path
        train_path = os.path.join(data_path, 'train')
        test_path = os.path.join(data_path, 'val')
        train_set = folder.ImageFolder(root=train_path, transform=transform_train,
                                    label_mapping=self.label_mapping)
        test_set = folder.ImageFolder(root=test_path, transform=transform_test,
                                    label_mapping=self.label_mapping)        
        return train_set, test_set

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = len(dataset)

    def __getitem__(self, index):
        return self.dataset[index], index

    def __len__(self):
        return self.len

def get_breeds_loaders(dataset='make_entity30', ratio=0, data_dir='/shared/Imagenet', info_dir='/com_space/caitianle/BREEDS-Benchmarks/imagenet_class_hierarchy/modified', source_train_transform=None, source_test_transform=None, target_train_transform=None, target_test_transform=None, index=False, subclass_label=False):

    ret = globals()[dataset](info_dir, split=None)
    superclasses, subclass_split, label_map = ret

    source_train_set = []
    target_train_set = []
    source_test_set = []
    target_test_set = []
    subclass_to_ratio = {}
    SELECTED_RANGES = subclass_split[0]

    for label, superclass_range in enumerate(SELECTED_RANGES):
        subclass_list = superclass_range
        for subclass_idx in subclass_list:
            subclass = [[subclass_idx]]
            if subclass_label:
                class_label = (label, subclass_idx)
            else:
                class_label = label
            dataset = CustomImageNetDataset(data_dir, subclass, label=class_label)
            subclass_source_train_set, subclass_source_test_set = dataset.make_datasets(transform_train=source_train_transform)
            subclass_target_train_set, subclass_target_test_set = dataset.make_datasets(transform_train=target_train_transform)
            
            rand = int(np.random.random()>0.5)
            length = len(subclass_source_train_set)
            train_set_size = int((0.5+ratio*(rand-0.5))*length)
            source_train_set.append(Subset(subclass_source_train_set, list(range(train_set_size))))
            target_train_set.append(Subset(subclass_target_train_set, list(range(train_set_size, length))))
            
            length = len(subclass_source_test_set)
            test_set_size = int((0.5-ratio*(rand-0.5))* length)
            target_test_set.append(Subset(subclass_target_test_set, list(range(test_set_size))))
            source_test_set.append(Subset(subclass_source_test_set, list(range(test_set_size, length))))
            
            if 0.5+ratio*(rand-0.5) == 0:
                subclass_to_ratio[subclass_idx] = 1
            else:
                subclass_to_ratio[subclass_idx] = (0.5-ratio*(rand-0.5)) / (0.5+ratio*(rand-0.5))

    source_train_set = ConcatDataset(source_train_set)
    source_test_set = ConcatDataset(source_test_set)
    target_train_set = ConcatDataset(target_train_set)
    target_test_set = ConcatDataset(target_test_set)

    if index:
        source_train_set = IndexedDataset(source_train_set)
        
    return source_train_set, source_test_set, target_train_set, target_test_set, subclass_to_ratio