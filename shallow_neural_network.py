import torch
import torch.nn as nn
import torch.optim as optim

# Shallow Neural Network Model
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShallowNeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)

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

# Example Usage
if __name__ == "__main__":
    # Sample Data
    x_train = torch.randn(100, 2)
    y_train = torch.randint(0, 2, (100,))

    model = ShallowNeuralNetwork(input_size=2, hidden_size=5, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_model(model, criterion, optimizer, x_train, y_train)
