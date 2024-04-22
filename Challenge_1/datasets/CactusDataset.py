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
        self.labels=labels

    def __len__(self):
        # here you just need to return a single integer number as the length of your dataset, in your
        #  case, number of images in your train folder or lines in csv file
        if self.df is not None:
            return len(self.df)
        else:
            return len(os.listdir(self.imgPath))

    def filter(self, label):
        # create a new dataset with the filtered data
        self_copy = CactusDataset(self.imgPath, self.labels, self.transform, self.target_transform)
        if self_copy.df is not None:
            self_copy.df = self.df[self.df['has_cactus'] == label].reset_index(drop=True)
        return self_copy

    def __getitem__(self, idx):
        # if we already have the labels we can use them to find the nth element otherwise we use the folder order. This is used for
        # example when we have only a partial number of labels compared to the number of images or when we don't have them at all
        img_path = os.path.join(self.imgPath,self.df['id'][idx] if self.df is not None else os.listdir(self.imgPath)[idx])
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
            # return a series with the label distribution. It enforce the order of the labels and 0 if they are not present
            return self.df['has_cactus'].value_counts().reindex([0, 1], fill_value=0)
        else:
            return np.array([0, 0])

