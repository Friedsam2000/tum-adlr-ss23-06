import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torch

class load_data(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.images_dir = images_dir
        # Convert the obstacle labels to binary
        self.labels.replace({"no_obstacle": 0, "obstacle": 1}, inplace=True)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.images_dir, 'test_images', self.labels.iloc[idx, 0])
            image = Image.open(img_name)
            label = self.labels.iloc[idx, 1:].values.astype(float)
            sample = {'image': image, 'label': torch.tensor(label, dtype=torch.float32)}
            if self.transform:
                sample['image'] = self.transform(sample['image'])
            return sample
        except ValueError as e:
            print(f"Error in row {idx}: {self.labels.iloc[idx]}")
            raise e
