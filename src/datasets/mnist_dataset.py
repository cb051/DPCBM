import torch
import numpy as np
import torch
import torchvision.transforms as t

from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import DataLoader, Subset


class MNIST():
    def __init__(self):
        PATH = 'datasets/MNIST/' # path where MNIST dataset should be stored
        self.train_data = datasets.MNIST(root=PATH, train=True, download=True, transform=t.ToTensor())
        self.val_data = datasets.MNIST(root=PATH, train=False, download=True, transform=t.ToTensor())
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.repeat(3,1,1)), # 3 channels
            transforms.Resize(32, antialias=True),
        ])
    def create_train_xyc(self):
        train_data = self.train_data
        x_train = []
        y_train = []
        c_train = pd.DataFrame({})
        for img, target in train_data:
            x_train.append(img)
            y_train.append(target)
            concept = pd.DataFrame({'has vertical line': 0, 'has horizontal line': 0,
                                    'has loop': 0, 'has curve': 0,
                                    'has right diagonal': 0, 'has left diagonal': 0,
                                    'has hard corner':0
                                    }, index=[0])
            if target in [1,4,7]: # vertical
                concept['has vertical line'] = 1
            if target in [4,5,7]: # horizontal line
                concept['has horizontal line'] = 1
            if target in [0,6,8]: # loop
                concept['has loop'] = 1
            if target in [2,3,5,6,9]: # curve
                concept['has curve'] = 1
            if target in [2,4,7,8]: # right slant diagonal
                concept['has left diagonal'] = 1
            if target in [8]: # left slant diagonal
                concept['has right diagonal'] = 1
            if target in [5,7]: # vertical
                concept['has hard corner'] = 1
            c_train = pd.concat([c_train,concept], ignore_index=True)

        return x_train, y_train, c_train 
    def create_val_xyc(self):
        val_data = self.val_data

        x_val = []
        y_val = []
        c_val = pd.DataFrame({})
        for img, target in val_data:
            x_val.append(img)
            y_val.append(target)
            concept = pd.DataFrame({'has vertical line': 0, 'has horizontal line': 0,
                                'has loop': 0, 'has curve': 0,
                                    'has right diagonal': 0, 'has left diagonal': 0,
                                    'has hard corner':0
                                }, index=[0])
            if target in [1,4,7]: # vertical
                concept['has vertical line'] = 1
            if target in [4,5,7]: # horizontal line
                concept['has horizontal line'] = 1
            if target in [0,6,8]: # loop
                concept['has loop'] = 1
            if target in [2,3,5,6,9]: # curve
                concept['has curve'] = 1
            if target in [2,4,7,8]: # left slant diagonal
                concept['has left diagonal'] = 1
            if target in [8]: # right slant diagonal
                concept['has right diagonal'] = 1
            if target in [5,7]: # vertical
                concept['has hard corner'] = 1
            c_val = pd.concat([c_val,concept], ignore_index=True)
        return x_val, y_val, c_val
    
    def get_dataloaders(self, batch_size=256, NDW=16):
        """
        batch_size: batch size
        NDW: num dataworkers in torch.dataloader

        returns
        train dataloader: torch dataloader of form x,y,c (input, target, concepts)
        val dataloader: torch dataloader of form x,y,c (input, target, concepts)
        """
        #  train dl
        x_train, y_train, c_train = self.create_train_xyc()
        train_imgs = list()
        # apply transform to train set

        for img in x_train:
            img = self.transform(img)
            train_imgs.append(img)
        train_imgs = torch.stack(train_imgs)
        y_train = torch.tensor(np.array(y_train), dtype=torch.long)
        c_train = torch.tensor(c_train.to_numpy(), dtype=torch.float32)
        train_data = torch.utils.data.TensorDataset(train_imgs,y_train,c_train)
        train_dl = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=NDW)
        
        # val dl
        x_val, y_val, c_val = self.create_val_xyc()
        val_imgs = list()
        for img in x_val:
            img = self.transform(img)
            val_imgs.append(img)
        val_imgs = torch.stack(val_imgs)
        y_val = torch.tensor(np.array(y_val), dtype=torch.long)
        c_val = torch.tensor(c_val.to_numpy(), dtype=torch.float32)
        val_data = torch.utils.data.TensorDataset(val_imgs,y_val,c_val)
        val_dl = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False,  num_workers=NDW)

        return train_dl, val_dl