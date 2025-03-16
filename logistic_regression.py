import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Cross-Entropy Loss Function
def cross_entropy_loss(y_pred, y_true):
    return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

# Training Function
def train_model(model, criterion, optimizer, x_train, y_train, epochs=100):
    print("Initial model parameters:", list(model.parameters()))
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        # Removed cross_entropy calculation as it is redundant




        print(f'Loss: {loss.item():.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

        if (epoch+1) % 10 == 0:

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Example Usage
if __name__ == "__main__":
    # Sample Data
    x_data = torch.tensor(np.random.rand(100, 2), dtype=torch.float32)
    y_data = torch.tensor(np.random.randint(0, 2, (100, 1)), dtype=torch.float32)

    # Create Dataset and DataLoader
    dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    model = LogisticRegressionModel(input_size=2)

    # Initialize criterion and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Call the training function
    train_model(model, criterion, optimizer, x_data, y_data, epochs=100)

    with torch.no_grad():
        sample_predictions = model(x_data)
        print("Sample predictions after training:", sample_predictions)
