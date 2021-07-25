import torch as t
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import random

import os

# transfrom 
transform_train = transforms.Compose([
    transforms.Resize(size = 256),
    transforms.RandomCrop(size = 224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = (.5, .5, .5),
        std =  (.5, .5, .5),
    ),
])
transform_valid = transforms.Compose([
    transforms.Resize(size = 256),
    transforms.CenterCrop(size = 224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = (.5, .5, .5),
        std =  (.5, .5, .5),
    ),
])

CLASS_NUM = 14

# dataset 
class Cloth1M(Dataset):
    def __init__(self, root, mode, transform = None, part = 1, size = None):
        if mode not in ['train', 'trainc', 'val', 'test', 'valid']:
            raise RuntimeError("invalid mode")        
        mode = 'val' if mode == 'valid' else mode        
        root = root + '/' if root[-1] != '/' else root
        
        self.part = part
        self.transform = transform

        prefix1 = 'clean_' if (mode == 'val' or mode == 'trainc' or mode == 'test') else 'noisy_'
        mode = 'train' if mode == 'trainc' else mode
        prefix2 = prefix1 + mode + '_'
        
        imglocat_list = root + 'images/' + prefix2 + 'key_list.txt' # train/valid/test part of clean/noisy img path
        imglabel_list = root + 'images/' + prefix1 + 'label_kv.txt' # all clean/all noisy img path & label

        labels = dict() # all (dict project image name to its label)
        for data in open(imglabel_list):
            locat, label = data.rstrip().split(' ')
            labels[root + locat] = int(label)        
        
        self.data = list() # train/valid/test part
        for data in open(imglocat_list):     
            p = root + data.rstrip()
            self.data.append((p, labels[p]))  
        random.shuffle(self.data)
        self.data = self.data[:int(len(self.data) * self.part)]
        self.size = size
        self.psi = [0] * len(self.data)

    def __len__(self):
        if self.size is None:
            return len(self.data)
        else:
            return self.size
    
    def update_psi(self, idx, value):
        self.psi[idx] = value

    def __getitem__(self, index):

        if self.size is not None:
            index = random.randint(0, len(self.data) - 1)
        img_path, label = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index, self.psi[index]

    def url(self, index):
        return (self.data[index][0])

   

    