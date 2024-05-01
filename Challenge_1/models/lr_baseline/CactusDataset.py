import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class CactusDataset(Dataset):
    def __init__(self, csv_file, data_folder, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.df.id[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.df.has_cactus[idx]
        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_image_id(self, idx):
        return self.df.id[idx]
    
    def oversample(self):
        """
        Double the minority class by oversampling
        """
        cactus = self.df[self.df.has_cactus == 1]
        non_cactus = self.df[self.df.has_cactus == 0]
        self.df = pd.concat([cactus, non_cactus, non_cactus])
        self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def get_class_distribution(self):
        return self.df.has_cactus.value_counts().to_dict()