import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


# CNN Model
class FinalCNN(nn.Module):
    def __init__(self):
        super(FinalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)  # (28x28) → (24x24)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)  # (12x12) → (8x8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling by 2x2
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Print activation details once
def print_activation_details(model, sample_batch):
    """Print activation map sizes once before training."""
    with torch.no_grad():
        x = sample_batch
        print("\n--- CNN Activation Details (One-time) ---")
        
        x = model.conv1(x)
        print(f"Conv1: {x.shape}")

        x = model.pool(x)
        print(f"Pool1: {x.shape}")

        x = model.conv2(x)
        print(f"Conv2: {x.shape}")

        x = model.pool(x)
        print(f"Pool2: {x.shape}")

        x = x.view(-1, 32 * 4 * 4)
        print(f"Flattened: {x.shape}")

        x = model.fc1(x)
        print(f"FC1: {x.shape}")

        x = model.fc2(x)
        print(f"FC2: {x.shape}")

        x = model.fc3(x)
        print(f"Output (Logits): {x.shape}\n")





# Display sample predictions
def display_predictions(model, data_loader, num_samples=6):
    """Displays sample images with predicted labels"""
    model.eval()
    
    images, labels = next(iter(data_loader))
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

    # Displaying 6 samples
    plt.figure(figsize=(12, 6))
    
    for i in range(num_samples):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Predicted: {predictions[i].item()} | Actual: {labels[i].item()}')
        plt.axis('off')
    
    plt.show()


# MNIST Dataset and Loader
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model, Criterion, and Optimizer
model = FinalCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the Model
losses, accuracies = train_model(model, criterion, optimizer, train_loader, epochs=5)

# Display sample predictions
display_predictions(model, train_loader)
