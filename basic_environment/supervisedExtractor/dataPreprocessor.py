import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch

class load_data(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.image_paths = [os.path.join(images_dir, 'test_images', img_name) for img_name in self.labels.iloc[:, 0]]
        # Convert the obstacle labels to binary
        self.labels.replace({"no_obstacle": 0, "obstacle": 1}, inplace=True)
        self.labels_values = self.labels.iloc[:, 1:].values.astype(float) # Cached as numpy array
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name)
        label = torch.tensor(self.labels_values[idx], dtype=torch.float32)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample
