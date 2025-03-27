import torch
import numpy as np
from torch import nn
from torch import optim
from torch import autograd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from final_project import FinalCNN
def main():
    import logistic_regression  # Import the module at the beginning
    print("Select a module to run (1-6):")

    # Function to plot data
    def plot_data(x, y, title):
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
        plt.title(title)
        plt.xlabel('Input Feature 1')
        plt.ylabel('Input Feature 2')
        plt.colorbar(label='Output Class')
        plt.show()

    print("1. Logistic Regression")
    print("2. Softmax Regression")
    print("3. Shallow Neural Networks")
    print("4. Deep Networks")
    print("5. Convolutional Neural Networks")
    print("6. Final Project")
    choice = input("Enter the module number (1-6): ")
    


    if choice == '1':
        logistic_regression.train_model(model, criterion, optimizer, x_data, y_data, epochs=1000)  # Call the function
        # Plotting the predictions
        y_pred = model(x_data).detach().numpy()
        plot_data(x_data.numpy(), np.argmax(y_pred, axis=1), 'Logistic Regression Predictions')


    elif choice == '2':
        import softmax_regression
        # Create sample data for softmax regression
        x_train = torch.tensor(np.random.rand(100, 2), dtype=torch.float32)
        y_train = torch.tensor(np.random.randint(0, 2, (100,)), dtype=torch.long)

        # Plotting the softmax regression data
        plot_data(x_train.numpy(), y_train.numpy(), 'Softmax Regression Input Data')

        x_train = torch.tensor(np.random.rand(100, 2), dtype=torch.float32)
        y_train = torch.tensor(np.random.randint(0, 2, (100,)), dtype=torch.long)

        # Initialize model, criterion, and optimizer for softmax regression
        model = softmax_regression.SoftmaxRegressionModel(input_size=2, num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        softmax_regression.train_model(model, criterion, optimizer, x_train, y_train, epochs=100)  # Call the function
        # Plotting the predictions
        y_pred = model(x_train).detach().numpy()
        plot_data(x_train.numpy(), np.argmax(y_pred, axis=1), 'Softmax Regression Predictions')


    elif choice == '3':
        import shallow_neural_network
        # Create sample data for shallow neural network
        x_train = torch.randn(100, 2)
        y_train = torch.randint(0, 2, (100,))

        # Plotting the shallow neural network data
        plot_data(x_train.numpy(), y_train.numpy(), 'Shallow Neural Network Input Data')

        x_train = torch.randn(100, 2)
        y_train = torch.randint(0, 2, (100,))

        # Initialize model, criterion, and optimizer for shallow neural network
        model = shallow_neural_network.ShallowNeuralNetwork(input_size=2, hidden_size=5, output_size=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        shallow_neural_network.train_model(model, criterion, optimizer, x_train, y_train, epochs=100)  # Call the function
        # Plotting the predictions
        y_pred = model(x_train).detach().numpy()
        plot_data(x_train.numpy(), np.argmax(y_pred, axis=1), 'Shallow Neural Network Predictions')


    elif choice == '4':
        import deep_networks
        # Create sample data for deep neural network
        x_train = torch.randn(100, 10)
        y_train = torch.randint(0, 2, (100,))

        # Plotting the deep neural network data
        plot_data(x_train.numpy(), y_train.numpy(), 'Deep Neural Network Input Data')

        x_train = torch.randn(100, 10)
        y_train = torch.randint(0, 2, (100,))

        # Initialize model, criterion, and optimizer for deep neural network
        model = deep_networks.DeepNeuralNetwork(input_size=10, hidden_sizes=[20, 10], output_size=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        deep_networks.train_model(model, criterion, optimizer, x_train, y_train, epochs=100)  # Call the function
        # Plotting the predictions
        y_pred = model(x_train).detach().numpy()
        plot_data(x_train.numpy(), np.argmax(y_pred, axis=1), 'Deep Neural Network Predictions')


    elif choice == '5':
        import convolutional_neural_networks
        # Create sample data for convolutional neural network
        x_train = torch.randn(100, 1, 28, 28)  # Example for MNIST
        y_train = torch.randint(0, 10, (100,))

        # Plotting the convolutional neural network data
        plot_data(x_train.view(100, -1).numpy(), y_train.numpy(), 'Convolutional Neural Network Input Data')

        x_train = torch.randn(100, 1, 28, 28)  # Example for MNIST
        y_train = torch.randint(0, 10, (100,))

        # Initialize model, criterion, and optimizer for convolutional neural network
        model = convolutional_neural_networks.ConvolutionalNeuralNetwork()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        convolutional_neural_networks.train_model(model, criterion, optimizer, x_train, y_train, epochs=100)  # Call the function
        # Plotting the predictions
        y_pred = model(x_train).detach().numpy()
        plot_data(x_train.view(100, -1).numpy(), np.argmax(y_pred, axis=1), 'Convolutional Neural Network Predictions')


    elif choice == '6':
        print("Choose dataset option:")
        print("1. MNIST Dataset")
        print("2. Custom Dataset")
        dataset_choice = input("Enter 1 or 2: ")

        if dataset_choice == '1':
            # MNIST Dataset
            transform = transforms.Compose([transforms.ToTensor()])
            train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
            
        elif dataset_choice == '2':
            # Custom Dataset Example
            class CustomDataset(Dataset):
                def __init__(self, data, labels, transform=None):
                    self.data = data
                    self.labels = labels
                    self.transform = transform
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    sample = self.data[idx]
                    label = self.labels[idx]
                    if self.transform:
                        sample = self.transform(sample)
                    return sample, label
            
            
        else:
            print("Invalid choice. Please select 1 or 2.")
            return

        model = FinalCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        losses, accuracies = model.train_model(model, criterion, optimizer, train_loader)  # Call the function
        
        # Plotting the loss and accuracy
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(losses) + 1), losses, color='red', label='Cost (Loss)')
        plt.title('Cost (Loss) vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Cost (Loss)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(accuracies) + 1), accuracies, color='blue', label='Accuracy')
        plt.title('Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.show()


    else:
        print("Invalid choice. Please select a number between 1 and 6.")

if __name__ == "__main__":
    main()
