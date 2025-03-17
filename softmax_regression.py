import torch
import torch.nn as nn
import torch.optim as optim

# Softmax Regression Model
class SoftmaxRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)

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
    x_train = torch.randn(100, 2)
    y_train = torch.randint(0, 2, (100,))

    # Plotting the input data
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.numpy(), cmap='viridis')
    plt.title('Softmax Regression Input Data')
    plt.xlabel('Input Feature 1')
    plt.ylabel('Input Feature 2')
    plt.colorbar(label='Output Class')
    plt.show()


    model = SoftmaxRegressionModel(input_size=2, num_classes=2)

    # Plotting the predictions
    y_pred = model(x_train).detach().numpy()
    plt.scatter(x_train[:, 0], x_train[:, 1], c=np.argmax(y_pred, axis=1), cmap='viridis')
    plt.title('Softmax Regression Predictions')
    plt.xlabel('Input Feature 1')
    plt.ylabel('Input Feature 2')
    plt.colorbar(label='Predicted Class')
    plt.show()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_model(model, criterion, optimizer, x_train, y_train)
