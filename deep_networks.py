import torch
import torch.nn as nn
import torch.optim as optim

# Deep Neural Network Model
class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepNeuralNetwork, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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
    x_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))

    # Plotting the input data
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.numpy(), cmap='viridis')
    plt.title('Deep Neural Network Input Data')
    plt.xlabel('Input Feature 1')
    plt.ylabel('Input Feature 2')
    plt.colorbar(label='Output Class')
    plt.show()


    model = DeepNeuralNetwork(input_size=10, hidden_sizes=[20, 10], output_size=2)

    # Plotting the predictions
    y_pred = model(x_train).detach().numpy()
    plt.scatter(x_train[:, 0], x_train[:, 1], c=np.argmax(y_pred, axis=1), cmap='viridis')
    plt.title('Deep Neural Network Predictions')
    plt.xlabel('Input Feature 1')
    plt.ylabel('Input Feature 2')
    plt.colorbar(label='Predicted Class')
    plt.show()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_model(model, criterion, optimizer, x_train, y_train)
