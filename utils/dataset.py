import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

import random
from torchvision.transforms import Compose, CenterCrop, Normalize, RandomCrop
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms.functional as F
from utils.transform import Crop, ToLabel, Relabel

EXTENSIONS = ['.jpg', '.png']

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def image_path(root, basename, extension):
    return os.path.join(root, basename + extension)

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

class VOC12(Dataset):
    def __init__(self, root, img_list, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        with open(img_list, 'r') as f:
            filenames = f.readlines()
        
        self.filenames = [x.strip() for x in filenames]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')
        
        if self.input_transform is None and self.target_transform is None:
            tw, th = 256, 256
            # tw = random.randint(image.size[0]//2, image.size[0])
            # th = random.randint(image.size[1]//2, image.size[1])
            
            padding = (max(0, tw - image.size[0]), max(0, th - image.size[1]))
            image = F.pad(image, padding)

            iw, ih = image.size[0], image.size[1]

            if iw == tw and tw == th:
                bi, bj = 0, 0
            else:
                bi = random.randint(0, ih - th)
                bj = random.randint(0, iw - tw)
            
            self.input_transform = Compose([
                Crop(bi, bj, th, tw),
                ToTensor(),
                Normalize([.485, .456, .406], [.229, .224, .225]),
            ])
            self.target_transform = Compose([
                Crop(bi, bj, th, tw),
                ToLabel(),
                Relabel(255, 0),
            ])

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return image, label
    
    def __len__(self):
        return len(self.filenames)


