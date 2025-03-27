import os
import struct
import numpy as np
import torch
import gzip
from PIL import Image, ImageFont, ImageDraw
import cv2
import random
import string

# 📝 Define the HandwrittenFontDataset class
class HandwrittenFontDataset(torch.utils.data.Dataset):
    def __init__(self, font_path, num_samples):
        self.font_path = font_path
        self.num_samples = num_samples
        self.font = ImageFont.truetype(self.font_path, 32)  # Font size
        self.characters = string.digits + string.ascii_uppercase + string.ascii_lowercase

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Randomly choose a character
        char = random.choice(self.characters)
        # Proceed with image creation and processing...

        # Create image with that character
        img = Image.new('L', (64, 64), color=255)  # Create a blank image (grayscale)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), char, font=self.font, fill=0)  # Draw the character

        # Convert image to numpy array (resize to 28x28 for MNIST format)
        img = np.array(img)
        img = preprocess_for_mnist(img)

        # Convert character to label (integer)
        label = self.characters.index(char)

        return torch.tensor(img, dtype=torch.uint8), label

# 📄 Resize and preprocess images for MNIST format
def preprocess_for_mnist(img):
    """Resize image to 28x28 and normalize to 0-255 range."""
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype(np.uint8)  # Convert to unsigned byte
    return img

# 📄 Write images to idx3-ubyte format
def write_idx3_ubyte(images, file_path):
    """Write images to idx3-ubyte format."""
    with open(file_path, 'wb') as f:
        # Magic number (0x00000801 for image files)
        f.write(struct.pack(">IIII", 2051, len(images), 28, 28))

        # Write image data as unsigned bytes (each pixel in range [0, 255])
        for image in images:
            f.write(image.tobytes())

# 📄 Write labels to idx1-ubyte format
def write_idx1_ubyte(labels, file_path):
    """Write labels to idx1-ubyte format."""
    with open(file_path, 'wb') as f:
        # Magic number (0x00000801 for label files)
        f.write(struct.pack(">II", 2049, len(labels)))

        # Write each label as a byte
        for label in labels:
            f.write(struct.pack("B", label))

# 📄 Compress file to .gz format
def compress_file(input_path, output_path):
    """Compress the idx3 and idx1 files to .gz format."""
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            f_out.writelines(f_in)

# 📊 Save dataset in MNIST format
def save_mnist_format(images, labels, output_dir):
    """Save the dataset in MNIST format to raw/ directory."""
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Prepare file paths
    train_images_path = os.path.join(raw_dir, "train-images-idx3-ubyte")
    train_labels_path = os.path.join(raw_dir, "train-labels-idx1-ubyte")

    # Write uncompressed idx3 and idx1 files
    write_idx3_ubyte(images, train_images_path)
    write_idx1_ubyte(labels, train_labels_path)

    # Compress idx3 and idx1 files into .gz format
    compress_file(train_images_path, f"{train_images_path}.gz")
    compress_file(train_labels_path, f"{train_labels_path}.gz")

    print(f"Dataset saved in MNIST format at {raw_dir}")

# ✅ Generate and save the dataset
def create_mnist_dataset(font_path, num_samples=4096):
    """Generate dataset and save in MNIST format."""
    # Get font name without extension
    font_name = os.path.splitext(os.path.basename(font_path))[0]
    output_dir = os.path.join("./data", font_name)
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    dataset = HandwrittenFontDataset(font_path, num_samples)
    
    images = []
    labels = []
    
    for i in range(num_samples):
        img, label = dataset[i]
        images.append(img.numpy())
        labels.append(label)
    
    # Save in MNIST format
    save_mnist_format(images, labels, output_dir)

# 🔥 Example usage
def choose_font_and_create_dataset():
    # List all TTF and OTF files in the root directory
    font_files = [f for f in os.listdir("./") if f.endswith(".ttf") or f.endswith(".otf")]
    
    # Display available fonts for user to choose
    print("Available fonts:")
    for i, font_file in enumerate(font_files):
        print(f"{i+1}. {font_file}")
    
    # Get user's choice
    choice = int(input(f"Choose a font (1-{len(font_files)}): "))
    chosen_font = font_files[choice - 1]
    
    print(f"Creating dataset using font: {chosen_font}")
    create_mnist_dataset(chosen_font)

# Run the font selection and dataset creation
choose_font_and_create_dataset()
