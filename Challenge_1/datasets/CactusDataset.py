import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

class CactusDataset(Dataset):
    def __init__(self, root_dir, labels_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels_path = labels_path
        if labels_path is not None:
            self.df = pd.read_csv(labels_path)
        else:
            self.df = None

    def __len__(self):
        if self.labels_path is not None:
            return len(self.df)
        else:
            return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if self.labels_path is not None:
            img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        else:
            img_name = os.path.join(self.root_dir, os.listdir(self.root_dir)[idx])
        image = Image.open(img_name)
        if self.labels_path is not None:
            label = self.df.iloc[idx, 1]
        else:
            label = -1
        if self.transform:
            image = self.transform(image)
        return img_name, image, label

    def filter(self, label):
        new_dataset = CactusDataset(self.root_dir, self.labels_path, self.transform)
        new_dataset.df = self.df[self.df['has_cactus'] == label].reset_index(drop=True)
        return new_dataset