# Cactus Dataset model
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np

class CactusDataset(Dataset):
    def __init__(self, train='./data/train/train', labels='./data/train.csv', transform=None, target_transform=None):
        if labels is None:
            self.df = None
        else:
            self.df = pd.read_csv(labels)
        self.imgPath = train
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # here you just need to return a single integer number as the length of your dataset, in your
        #  case, number of images in your train folder or lines in csv file
        if self.df is not None:
            return len(self.df)
        else:
            return len(os.listdir(self.imgPath))

    def __getitem__(self, idx):
        # if we already have the labels we can use them to find the nth element otherwise we use the folder order. This is used for
        # example when we have only a partial number of labels compared to the number of images or when we don't have them at all
        img_path = os.path.join(self.imgPath,
                                self.df['id'][idx] if self.df is not None else os.listdir(self.imgPath)[idx])
        image = Image.open(img_path).convert('RGB')
        if self.df is not None:
            label = self.df['has_cactus'][idx]
        else:
            label = -1
        # we perform transformations if they are not None and the labels are not None
        if self.transform:
            image = self.transform(image)
        if self.target_transform is not None and label is not None:
            label = self.target_transform(label)
        return image, label

    # this function is used to get the distribution of the labels in the dataset
    def label_distribution(self):
        if self.df is not None:
            return self.df['has_cactus'].value_counts(ascending=True).values
        else:
            return np.array([0, 0])


class ConcatTransformDataset(Dataset):
    def __init__(self, datasets, transforms=None):
        self.datasets = datasets
        self.transforms = transforms

    def __getitem__(self, index):
        dataset_idx, sample_idx = self._get_dataset_index(index)
        image, label = self.datasets[dataset_idx][sample_idx]
        if self.transforms:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def _get_dataset_index(self, index):
        for i, dataset in enumerate(self.datasets):
            if index < len(dataset):
                return i, index
            index -= len(dataset)
        raise IndexError('Index out of range')