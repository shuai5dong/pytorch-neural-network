import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import random

# Hyperparameters
learning_rate = 0.01
num_epochs = 10
batch_size = 1
K = 6  # Number of bits in the hash function
L = 5  # Number of hash tables


# Create a simple feedforward neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear( 784,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# A function to create hash tables based on the weights
def create_hash_tables(model):
    hash_tables = []
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            weights = layer.weight.data.numpy()
            hash_table = {}

            # Create hash tables for weights
            for i in range(weights.shape[0]):
                # Generate a hash fingerprint for each weight vector
                hash_fingerprint = tuple(
                    np.sign(np.dot(weights[i], np.random.randn(weights.shape[1], K))))  # Simple hash function
                if hash_fingerprint in hash_table:
                    hash_table[hash_fingerprint].append(i)
                else:
                    hash_table[hash_fingerprint] = [i]

            hash_tables.append(hash_table)
    return hash_tables


# Function to retrieve active set based on hashing
def get_active_set(hash_tables, input_tensor):
    active_set = []
    for i, hash_table in enumerate(hash_tables):
        # This simulates input processing to get a hash fingerprint
        hash_fingerprint = tuple(
            np.sign(np.dot(input_tensor, np.random.randn(input_tensor.size(1), K))))  # Hashing input
        if hash_fingerprint in hash_table:
            active_set.append(hash_table[hash_fingerprint])
        else:
            active_set.append([])  # No active nodes
    return active_set


# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize neural network and optimizer
model = NeuralNetwork()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()

        # Create hash tables for the model's weights
        hash_tables = create_hash_tables(model)

        # Get the active set based on the input data
        active_set = get_active_set(hash_tables, inputs)

        # Forward propagation only through the active nodes (simplified)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")