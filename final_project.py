import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # Added import for plotting

# Final Project: Convolutional Neural Network for MNIST Classification
class FinalCNN(nn.Module):
    def __init__(self):
        super(FinalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training Function
def train_model(model, criterion, optimizer, train_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Visualizing a few test images and their predicted labels
    model.eval()
    with torch.no_grad():
        test_images, test_labels = next(iter(train_loader))
        test_outputs = model(test_images)
        _, predicted = torch.max(test_outputs, 1)

        # Plotting the images and their predicted labels
        plt.figure(figsize=(12, 6))
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(test_images[i].squeeze(), cmap='gray')
            plt.title(f'Predicted: {predicted[i].item()}')
            plt.axis('off')
        plt.show()

# Example Usage
if __name__ == "__main__":
    # MNIST Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    model = FinalCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_model(model, criterion, optimizer, train_loader)  # Call the function
