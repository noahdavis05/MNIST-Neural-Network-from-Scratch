# What I've Learnt about Neural Networks
This covers the basics as to what I've learnt throughout coding my neural networks. 
## The Basics
Neural Networks are a type of machine learning model inspired by the structure and function of the brain. They consist of layers of interconnected nodes (or neurons) that work together to learn patterns from data.
## The structure
A typical neural network is made up of three types of layers:
- Input Layer - This layer recieves the input data. Each node in this layer represents one featrue or input value. (e.g. each node is a pixel of an image).
- Hidden Layers - These are intermediate layers between the input and outpout layers. They consist of neurons that apply transformations to the data using weights and biases, and activation functions.
- Output layer - This layer produces the final predictions (e.g. class labels or values). Each node in the output layer corresponds to a possible output.
## How data moves through the network
This is known as the process of forward propogation. In this process:
### Input Layer
- The network takes in raw input (e.g. an image or data points). Each feature (or pixel value, etc) is fed into a neuron in the input layer.
### Weight Multiplication and Bias addition
- Each neuron in the next layer (hidden or output) recieves inputs from all neurons in the previous layer.
- These inputs are multiplied by weights, which controls how strongly the input influences the neuron.
- The sum of these weighted inputs is calculated, and then the bias is added. 
- The value of the current node can be calculated as such. Z = SUM OF (Every weight between prev layer and this node * the activation (output) of each corresponding neuron) + the bias of the current neuron.
### What is an activation?
- The activation is the output of a neuron. It's what the neuron sends to the next layer after applying the activation function.
- When the network starts, the input layer just passes the raw data as activations. For example, if the input is an image, the pixel values are the activations.
- Once the neurons in the hidden layers recieve their input (the weughted sum of activations + biases), they apply an activation function like ReLU or sigmoid to that value.
- For ReLU: If the weighted sum is positive, the neuron sends that value to the next layer. If it's negative it sends zero to the next layer.
- For sigmoid: The output is squashed between 0 and 1.
### Activation function (non-linearity)
- Once a neuron has computed its weighted sum (the z-value), it passes this sum through an activation function.
## Learning from Data: Backpropogation and optimization
To make the neural network leanr from the data, they need to adjust their weights and biases based on the error (or "loss") between the predicted output and the true output.
<br>
The process is:
- Loss function: A loss function (e.g., Mean Squared Error for regression, Cross-Entropy for classification) measures how far the network's predictions are from the true values. The goal of training is to minimize the loss.
- Backpropagation: This is the process of calculating the gradients of the loss with respect to the network's parameters (weights and biases). The gradients indicate how much each parameter contributed to the error, allowing the network to adjust the parameters accordingly. The gradients indicate how much each parameter contributed to the error, allowing the network to adjust the parameters accordingly. <br>
Backpropogation uses the chain rule of calculus to compute these gradients layer by layer, from the output back to the input.
- Gradient Descent: After calculating the gradients, the weights and biases are updated using an optimization algorithm like gradient descent. This process involves moving in a direction that reduces loss. The sizes of the step taken in this direction is controlled by the learning rate.
<br>
To do this the weight = the weight - (the learning rate * gradient of the loss with respect to weight)
## Training and Generalization
- Epochs: During training, the network processes the entire dataset multiple times (each complete pass through the data is called an epoch).
- Overfitting: If the network becomes too good at fitting the training data, it might memorize it rather than learn general patterns. To combat overfitting, techniques like regularization, dropout, and early stopping are used.
- Generalization: The goal is for the network to generalize well, meaning it should perform well on unseesn data (the test set), rather than just the training data.