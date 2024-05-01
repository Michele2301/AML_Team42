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
            tensor_image = self.transform(image)

            return tensor_image, label

        return image, label
    
    def get_image_id(self, idx):
        return self.df.id[idx]