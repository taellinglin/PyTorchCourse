import os
import numpy as np
import gzip
from PIL import Image
from torchvision import transforms

class CustomMNISTDataset:
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.images, self.labels = self.load_dataset()

    def load_dataset(self):
        image_paths = []
        label_paths = []

        # Assuming the dataset consists of images and labels in the dataset path
        for file in os.listdir(self.dataset_path):
            if 'train-images-idx3-ubyte.gz' in file:
                image_paths.append(os.path.join(self.dataset_path, file))
            elif 'train-labels-idx1-ubyte.gz' in file:
                label_paths.append(os.path.join(self.dataset_path, file))

        if not image_paths or not label_paths:
            raise ValueError(f"❌ Missing image or label files in {self.dataset_path}")

        images = []
        labels = []

        # Assuming one image file and one label file
        for img_path, label_path in zip(image_paths, label_paths):
            images_data, labels_data = self.load_mnist_data(img_path, label_path)
            images.extend(images_data)
            labels.extend(labels_data)

        return images, labels

    def load_mnist_data(self, img_path, label_path):
        """Load MNIST data from .gz files."""
        with gzip.open(img_path, 'rb') as f:
            # Skip the magic number and metadata
            f.read(16)
            # Read the image data
            img_data = np.frombuffer(f.read(), dtype=np.uint8)
            img_data = img_data.reshape(-1, 28, 28)  # Reshape to 28x28 images

        with gzip.open(label_path, 'rb') as f:
            # Skip the magic number and metadata
            f.read(8)
            # Read the label data
            label_data = np.frombuffer(f.read(), dtype=np.uint8)

        images = [Image.fromarray(img) for img in img_data]  # Convert each image to a PIL Image

        # If you have any transformation, apply it here
        if self.transform:
            images = [self.transform(img) for img in images]

        return images, label_data

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Return a single image and its label at the given index."""
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
