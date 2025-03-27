import torch
from torch.utils.data import Dataset
import os
import numpy as np
import gzip

# Custom MNIST-like dataset loader
class CustomMNISTDataset(Dataset):
    """Custom dataset loader for MNIST-like datasets in .idx or .gz format."""
    
    def __init__(self, image_path, label_path, transform=None):
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform

        # Load images and labels
        self.images = self._load_images(image_path)
        self.labels = self._load_labels(label_path)

    def _load_images(self, path):
        with gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb') as f:
            f.read(16)  # Skip the header
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(-1, 28, 28)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(1) / 255.0

    def _load_labels(self, path):
        with gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb') as f:
            f.read(8)  # Skip the header
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
