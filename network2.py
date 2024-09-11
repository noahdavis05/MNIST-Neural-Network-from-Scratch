import numpy as np
import os

# Load the previously saved MNIST data
x_train = np.load('./mnist/x_train.npy')
y_train = np.load('./mnist/y_train.npy')
x_test = np.load('./mnist/x_test.npy')
y_test = np.load('./mnist/y_test.npy')

# Step one - Prepare the data

# Flatten the images 28X28 --> 784
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Normalize the pixel values
# Scaling down the pixel values speeds up training and makes the model more stable because it ensures that large values like 255 don't dominate the learning process.
x_train = x_train / 255.0
x_test = x_test / 255.0

# temp fix
# One-hot encode labels
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

# Define the neural network parameters
num_classes = 10

# One-hot encode labels
y_train_one_hot = one_hot_encode(y_train, num_classes)
y_test_one_hot = one_hot_encode(y_test, num_classes)

# Step two - Design thre neural network architecture.
"""
1.In this step we must decide on the number of layers.
 We must have an input layer of 784 neurons (one for each pixel)
 Just one hidden layer (for now), with 128 neurons (as it's a common choice)
 The output layer will have 10 neurons as there are 10 outputs to classify.
2.Initialize weights and biases. The connections between neurons have associated
  weights, which are initially random. Each neuron has a bias term which helps
  the neuron to learn independently of the input.
3.Activation functions. Sigmoid: often used in hidden layers, squashes the output
  to be between o and 1. Softmax: Typically used in the output layer for classification
  problems to convert raw scores into probability.
"""

# Number of neurons in each layer
input_size = 784
hidden_size1 = 128
hidden_size2 = 64
output_size = 10

# Initialize weights and biases
np.random.seed(42)
w1 = np.random.randn(input_size, hidden_size1) * 0.01 # weights for input to hidden
b1 = np.zeros((1,hidden_size1)) # biases for hidden layer 1
w2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
b2 = np.zeros((1,hidden_size2)) # biases for the hidden layer 2
w3 = np.random.randn(hidden_size2, output_size) * 0.01 # weights for hiden layer to output layer
b3 = np.zeros((1,output_size)) # biases for output layer



"""
Initializing weights randomly helps break symmetry. If all weights started the same,
the neurones would all leanr the same thing during training, which is undersireable.
The small random vales ensure that different neurons learn differently.
"""

# Step 3 - Forward Propogation
"""
In this step we will
1.Pass the input data through the network: This involves multiplying the inputs
  by the weights, adding biases, and applying an activation function at each layer.
2.Compute the output: For the hidden layer, we'll apply the sigmoid activation
  function to introduce non-linearity. For the output layer, we'll use the softmax
  activation function to convert the raw output into probabilities.
"""

# Sigmoid activation function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Softmax activation function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Shift values to avoid overflow
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ReLU activation function
def relu(z):
    return np.maximum(0, z)

# Derivative of ReLU activation function
def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# Forward pass
# This is the process where all neurons in each layer are involved. Each neurons output is computed and passed to to the next layer.
# Forward pass
def forward_propogation(x, w1, b1, w2, b2, w3, b3):
    # Input to hidden layer 1
    z1 = np.dot(x, w1) + b1 
    a1 = relu(z1)  # Use ReLU for hidden layer 1

    # hidden layer 1 to hidden layer 2
    z2 = np.dot(a1, w2) + b2
    a2 = relu(z2)  # Use ReLU for hidden layer 2

    # hidden layer to output layer
    z3 = np.dot(a2, w3) + b3
    y_hat = softmax(z3) # final output probabilities

    return a1, a2, y_hat


# Step 4 - Loss function and back propogation
"""
1.Define the loss function. For a classification problem like MNIST, the cross-entropy loss]
  function is commonly used. Cross-entropy measures the difference between the predicted
  probability distribution and the true distribution.
2.Implement backpropogation. Backpropogation is the process of computing gradients of the loss
  function with respect to weight and bias, and then updating them to minimize the loss.
"""

# Cross-entropy loss
# This function penalizes incorrect predictions more heavily, guiding the network to correct its errors.
def cross_entropy_loss(y, y_hat):
    m = y.shape[0] # Number of examples
    loss = -np.sum(y * np.log(y_hat + 1e-8)) / m
    return loss 

# Compute gradients and update weights using backpropogation
# Steps in backpropogation
# 1. Compute the gradient of the loss with respect to the output layer's activation.
# 2. Compute the gradient of the loss with respect to the hidden layer's activation
# 3. Update the weights and biases using the gradients.
# Backpropagation
def backPropogation(x, y, a1, a2, y_hat, w1, b1, w2, b2, w3, b3, learning_rate=0.01):
    m = x.shape[0] # number of examples

    # Compute gradients for output layer
    dz3 = y_hat - y
    dw3 = np.dot(a2.T, dz3) / m
    db3 = np.sum(dz3, axis=0, keepdims=True) / m

    # Compute gradients for hidden layer 2
    da2 = np.dot(dz3, w3.T)
    dz2 = da2 * relu_derivative(a2)  # Use ReLU derivative
    dw2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m 

    # Compute gradients for hidden layer 1
    da1 = np.dot(dz2, w2.T)  # Gradient wrt hidden layer 1 activation
    dz1 = da1 * relu_derivative(a1)  # Use ReLU derivative
    dw1 = np.dot(x.T, dz1) / m  # Gradient wrt weights of hidden layer 1
    db1 = np.sum(dz1, axis=0, keepdims=True) / m  # Gradient wrt biases of hidden layer 1

    # Update weights and biases
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2 
    b2 -= learning_rate * db2
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3

    return w1, b1, w2, b2, w3, b3



# Step 5 - Training the neural network
"""
1. Initialize the weights and biases.
2. Loop over multiple epochs.
3. In each epoch, perform forward propogation to compute predictions and loss.
4. Perform backpropogation to update weights and biases.
5. Monitor training progress by evaluating the loss and accuracy.
"""

def train(x_train, y_train, x_test, y_test, w1, b1, w2, b2, w3, b3, epochs=1000, learning_rate=0.1):
    for epoch in range(epochs):
        # Forward propagation
        a1, a2, y_hat = forward_propogation(x_train, w1, b1, w2, b2, w3, b3)

        # Compute loss
        loss = cross_entropy_loss(y_train, y_hat)

        # Backpropagation
        w1, b1, w2, b2, w3, b3 = backPropogation(x_train, y_train, a1, a2, y_hat, w1, b1, w2, b2, w3, b3, learning_rate)

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Evaluate on test data
    a1_test, a2_test, y_hat_test = forward_propogation(x_test, w1, b1, w2, b2, w3, b3)
    test_loss = cross_entropy_loss(y_test, y_hat_test)
    accuracy = np.mean(np.argmax(y_hat_test, axis=1) == np.argmax(y_test, axis=1))

    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {accuracy}')

    return w1, b1, w2, b2, w3, b3

    

# train the network!!
w1, b1, w2, b2, w3, b3 = train(x_train, y_train_one_hot, x_test, y_test_one_hot, w1, b1, w2, b2, w3, b3)

# give the user an option to save results if happy with them
save = input("Do you want to save the results? (y/n): ")
if save == 'y':
    # Create a folder if it doesn't exist
    folder_path = './n2Results'
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)
    np.save('./n1Results/w1.npy', w1)
    np.save('./n1Results/b1.npy', b1)
    np.save('./n1Results/w2.npy', w2)
    np.save('./n1Results/b2.npy', b2)
    np.save('./n1Results/w3.npy', w3)
    np.save('./n1Results/b3.npy', b3)
    

