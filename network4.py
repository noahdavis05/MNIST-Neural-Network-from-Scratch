import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a transformation to normalize the dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to mean=0.5 and std=0.5
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()  # Flatten the input
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer to hidden layer
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(128, 10)  # Hidden layer to output layer
        self.softmax = nn.LogSoftmax(dim=1)  # Softmax for classification

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        x = self.fc1(x)  # First linear layer
        x = self.relu(x)  # Activation function
        x = self.fc2(x)  # Second linear layer
        return self.softmax(x)  # Output layer

# Instantiate the model
model = SimpleNN()

# Define the loss function and optimizer
criterion = nn.NLLLoss()  # Negative log likelihood loss
optimizer = optim.Adam(model.parameters())  # Adam optimizer

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

# Test the model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # No need to track gradients during evaluation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted labels
        total += labels.size(0)  # Total number of labels
        correct += (predicted == labels).sum().item()  # Count correct predictions

test_acc = correct / total  # Calculate accuracy
print(f'\nTest accuracy: {test_acc:.4f}')
