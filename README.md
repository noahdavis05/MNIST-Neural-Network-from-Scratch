# MNIST-Intro

## How a Neural Network works
A neural network is designed to mimic how the human brain processes information. It learns patterns from data by adjusting the connections (weights) between neurons (nodes) across different layers.

At a high level, a neural network has:

Input Layer: Receives the data.
Hidden Layers: Process the data through weights, biases, and activation functions.
Output Layer: Produces the final result (e.g., classifying a digit in MNIST).
The goal of the network is to take input data, process it, and give a meaningful output, like classifying an image as a digit.

Detailed Breakdown:
Input Layer:

This is where you feed in your data. For MNIST, each image is 28x28 pixels, so the input layer will have 784 neurons (28*28 = 784), where each neuron corresponds to a pixel.
Hidden Layers:

These layers perform most of the processing. Each neuron in a hidden layer is connected to all the neurons from the previous layer. The connections have weights which are adjusted during training.
Hidden layers also have biases to adjust the output before it's passed to the next layer.
An activation function (e.g., sigmoid, ReLU) is applied to introduce non-linearity, allowing the network to solve complex problems.
Output Layer:

This layer produces the final output. For MNIST, there are 10 output neurons, one for each digit (0-9).
The output is typically a probability distribution showing the likelihood of each class (e.g., 85% chance it’s a 5).
How it Learns:
Forward Propagation:

Data flows through the network from input to output. At each layer, the inputs are multiplied by the weights, summed with the biases, and passed through the activation function.
Loss Function:

The network makes predictions, but initially, they will be incorrect. The loss function measures how far the predictions are from the actual values. For classification tasks like MNIST, cross-entropy is commonly used as the loss function.
Backpropagation:

This is where learning happens. The network calculates the gradient (derivative) of the loss with respect to each weight using backpropagation. These gradients tell the network how to adjust the weights to minimize the loss.
Weight Update:

The weights are adjusted using an optimizer (e.g., gradient descent) based on the calculated gradients. Over many iterations, the network learns to minimize the loss, improving its predictions.
Goal:
The goal is to train the network to adjust its weights and biases so that it can make accurate predictions when given new input data.