import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset_loader import CustomMNISTDataset
import os
import matplotlib.font_manager as fm
# CNN Model
# CNN Model with output layer for 62 categories
class FinalCNN(nn.Module):
    def __init__(self):
        super(FinalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)  # Output layer with 62 units for (0-9, a-z, A-Z)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Final output
        return x
def plot_loss_accuracy(losses, accuracies):
    """Plots Loss vs Accuracy on the same graph."""
    plt.figure(figsize=(10, 6))
    
    # Plot Loss
    plt.plot(losses, color='red', label='Loss (Cost)', linestyle='-', marker='o')
    
    # Plot Accuracy
    plt.plot(accuracies, color='blue', label='Accuracy', linestyle='-', marker='x')
    
    plt.title('Training Loss and Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    
    # Show the plot
    plt.show()

# 🔥 Function to choose the dataset dynamically
def choose_dataset(dataset_name):
    """Choose and load a custom dataset dynamically."""
    
    # ✅ Dynamic path generation
    base_path = './data'
    dataset_path = os.path.join(base_path, dataset_name, 'raw')

    # Validate dataset path
    if not os.path.exists(dataset_path):
        raise ValueError(f"❌ Dataset {dataset_name} not found at {dataset_path}")

    # ✅ Locate image and label files dynamically
    image_file = None
    label_file = None
    
    for file in os.listdir(dataset_path):
        if 'images' in file:
            image_file = os.path.join(dataset_path, file)
        elif 'labels' in file:
            label_file = os.path.join(dataset_path, file)

    # Ensure both image and label files are found
    if not image_file or not label_file:
        raise ValueError(f"❌ Missing image or label files in {dataset_path}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize between -1 and 1
    ])

    # ✅ Load the custom dataset with file paths
    dataset = CustomMNISTDataset(dataset_path=dataset_path, transform=transform)

    return dataset


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


# Training Function
def train_model(model, criterion, optimizer, train_loader, epochs=256):
    losses = []
    accuracies = []

    # Print activation details once before training
    sample_batch, _ = next(iter(train_loader))
    print_activation_details(model, sample_batch)

    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct, total = 0, 0

        # tqdm progress bar
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as t:
            for images, labels in t:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Update metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                t.set_postfix(loss=loss.item())

        # Store epoch loss and accuracy
        losses.append(epoch_loss / len(train_loader))
        accuracy = 100 * correct / total
        accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    # After training, plot the loss and accuracy
    plot_loss_accuracy(losses, accuracies)

    return losses, accuracies


# Display sample predictions

def get_dataset_options(base_path='./data'):
    """List all subdirectories in the data directory."""
    try:
        # List all subdirectories in the base_path (data folder)
        options = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
        return options
    except FileNotFoundError:
        print(f"❌ Directory {base_path} not found!")
        return []

# Mapping for numbers to characters (0-9, a-z, A-Z)
def number_to_char(number):
    if 0 <= number <= 9:
        return str(number)  # 0-9
    elif 10 <= number <= 35:
        return chr(number + 87)  # a-z (10 -> 'a', 35 -> 'z')
    elif 36 <= number <= 61:
        return chr(number + 29)  # A-Z (36 -> 'A', 61 -> 'Z')
    else:
        return ''

def display_predictions(model, data_loader, num_samples=6, font_path='./Daemon.otf'):
    """Displays sample images with predicted labels"""
    model.eval()
    
    # Load custom font
    prop = fm.FontProperties(fname=font_path)
    
    images, labels = next(iter(data_loader))
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

    # Displaying 6 samples
    plt.figure(figsize=(12, 6))
    
    for i in range(num_samples):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        
        # Convert predicted number to corresponding character
        predicted_char = number_to_char(predictions[i].item())
        actual_char = number_to_char(labels[i].item())

        # Title with 'Predicted' and 'Actual' both in custom font
        plt.title(f'Predicted: {predicted_char} | Actual: {actual_char}', fontsize=84, fontproperties=prop)

        plt.axis('off')
    
    plt.show()


if __name__ == "__main__":
    # Choose Dataset
    dataset_options = get_dataset_options()

    if dataset_options:
        # Dynamically display dataset options
        print("Available datasets:")
        for i, option in enumerate(dataset_options, 1):
            print(f"{i}. {option}")

        # User input to choose a dataset
        dataset_index = int(input(f"Enter the number corresponding to the dataset (1-{len(dataset_options)}): ")) - 1

        # Ensure valid selection
        if 0 <= dataset_index < len(dataset_options):
            dataset_name = dataset_options[dataset_index]
            print(f"You selected: {dataset_name}")
        else:
            print("❌ Invalid selection.")
            dataset_name = None
    else:
        print("❌ No datasets found in the data folder.")
        dataset_name = None

        train_dataset = choose_dataset(dataset_name)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Model, Criterion, and Optimizer
        model = FinalCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.005)

        display_predictions(model, train_loader)

        # Train the Model
        losses, accuracies = train_model(model, criterion, optimizer, train_loader, epochs=256)

        # Display sample predictions
        display_predictions(model, train_loader)
