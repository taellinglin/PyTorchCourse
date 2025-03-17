import torch
import torch.nn as nn
import torch.optim as optim

# Convolutional Neural Network Model
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
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
def train_model(model, criterion, optimizer, x_train, y_train, epochs=100):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

import matplotlib.pyplot as plt  # Added import for plotting

# Example Usage
if __name__ == "__main__":
    # Sample Data
    x_train = torch.randn(100, 1, 28, 28)  # Example for MNIST
    y_train = torch.randint(0, 10, (100,))

    # Plotting the input data
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(x_train[i].squeeze(), cmap='gray')
        plt.title(f'Label: {y_train[i].item()}')
        plt.axis('off')
    plt.show()


    model = ConvolutionalNeuralNetwork()

    # Plotting the predictions
    y_pred = model(x_train).detach().numpy()
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(x_train[i].squeeze(), cmap='gray')
        plt.title(f'Predicted: {np.argmax(y_pred[i])}')
        plt.axis('off')
    plt.show()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_model(model, criterion, optimizer, x_train, y_train)
