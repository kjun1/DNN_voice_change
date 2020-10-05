import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import os
from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision

import torch.nn as nn
import torch

from PIL import Image

transform_dict = {
        'train': transforms.Compose(
            [transforms.Resize((256,256)),
             transforms.CenterCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ]),
        'test': transforms.Compose(
            [transforms.Resize((256,256)),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])}

data_folder = "../../dataset/moeimouto-faces"
phase = "train"

data = torchvision.datasets.ImageFolder(root=data_folder, transform=transform_dict[phase])



train_ratio = 0.8

train_size = int(train_ratio * len(data))
val_size  = len(data) - train_size
data_size  = {"train":train_size, "val":val_size}
print(data_size)
data_train, data_val = torch.utils.data.random_split(data, [train_size, val_size])



batch_size = 1
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_loader   = torch.utils.data.DataLoader(data_val,   batch_size=batch_size, shuffle=False)
dataloaders  = {"train":train_loader, "val":val_loader}
