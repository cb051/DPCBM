import torch
import numpy as np
import torch
import torchvision.transforms as t
import torchvision.datasets as tv_data

import pandas as pd
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import os
import urllib
import zipfile
from PIL import Image


# adapted from Sheth et al. (2023) - Auxiliary Losses for Learning Generalizable Concept-based Models 
# AWA2 Dataset introduced by Xian et al. (2018) - Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly
class AWA2(tv_data.ImageFolder):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=tv_data.folder.default_loader,
                 is_valid_file=None,
                 train=True):
        img_root = os.path.join(root, 'JPEGImages')
        # check if dataset is downloaded
        self._download_data()

        super(AWA2, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        
        # obtain labels/targets
        self.class_to_idx = dict()
        with open(f'{root}/classes.txt', 'r') as f:
            for idx, line in enumerate(f):
                class_name = line.split('\t')[1].strip()
                self.class_to_idx[class_name] = idx

        # obtain associated attributes
        self.C_A = np.loadtxt(f'{root}/predicate-matrix-binary.txt', dtype=np.float32)

        # Collect all image paths and their corresponding class indices
        self.img_paths = []
        class_dirs = os.listdir(img_root)
        for class_name in class_dirs:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(img_root, class_name)
            for img_name in os.listdir(class_dir):
                self.img_paths.append((os.path.join(class_dir, img_name), class_idx))
    
    def __getitem__(self, index):
        # generate one sample
        img_path, target = self.img_paths[index]

        # Load image
        sample = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        # Retrieve attributes for the target class
        attribute = self.C_A[target]
        return sample, target, attribute
    
    def _download_data(self):
            img_url = "https://cvml.ista.ac.at/AwA2/AwA2-data.zip" # imgs
            attribute_url = "https://cvml.ista.ac.at/AwA2/AwA2-base.zip" # labels (concepts and labels)
            
            img_filename = "datasets/AWA2/AWA2-imgs.zip"
            attribute_filename = "datasets/AWA2/AWA2-labels.zip" # change accordingly
            

            parent_dir_path = "datasets/AWA2"  # change accordingly
            ds_path = "datasets/AWA2/Animals_with_Attributes2" # extracted file destination
            
            if os.path.exists(ds_path): # if path exists, skip
                print("AWA2 already downloaded. Skipping...")
                return
            else: # download AWA2 dataset
                print("Downloading AWA2 dataset...")
                urllib.request.urlretrieve(img_url, img_filename)
                urllib.request.urlretrieve(attribute_url, attribute_filename)

                with zipfile.ZipFile(img_filename, 'r') as zip: # extract imgs
                    print("Downloading imgs (13 GB)...")
                    zip.extractall(parent_dir_path)

                with zipfile.ZipFile(attribute_filename, 'r') as zip: # extract file
                    print("Downloading labels (32 KB)...")
                    zip.extractall(parent_dir_path)
                
                # Clean up
                os.remove(attribute_filename)  
                os.remove(img_filename)

def get_dataloaders(batch_size, NDW=16):
    # set root
    root = "datasets/AWA2/Animals_with_Attributes2" # change accordingly
    concept_transform = t.Compose([
                                    t.CenterCrop((320, 320)),
                                    t.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
                                    t.RandomHorizontalFlip(),
                                    t.ToTensor(), #implicitly divides by 255
                                    t.Normalize(mean = [0.5, 0.5, 0.5], std = [.2, .2, .2])
                        ])
    


    ds = AWA2(
                    root,
                    transform=concept_transform,
                    target_transform=None,
                    loader=tv_data.folder.default_loader,
                    is_valid_file=None,
                )
    train_ds, val_ds = random_split(ds,[0.8,0.2])

    # random sampler
    labels = [train_ds.dataset.img_paths[idx][1] for idx in train_ds.indices]
    class_counts = np.bincount(np.array(labels))
    probs = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = torch.tensor([probs[label] for label in labels], dtype=torch.float)
    train_sampler = WeightedRandomSampler(weights,num_samples=len(train_ds),replacement=True)
    
    train_dl = DataLoader(train_ds,
                        batch_size,
                        num_workers=NDW,
                        shuffle=False,
                        sampler=train_sampler)
    val_dl = DataLoader(val_ds,
                        batch_size,
                        num_workers=NDW,
                        shuffle=False)
    return train_dl, val_dl



