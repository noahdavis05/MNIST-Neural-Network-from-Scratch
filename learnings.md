# Neural Networks to solve MNIST dataset
## The basics
At a high level, a neural network has:
- Input layer: This is the layer of nodes which recives the input.
- Hidden layers: The layer/layers which process the data through weights, biases, and activation functions.
- Output layer: Produces the final result.

## How it works
### The Basic Structure of a Neural Network
- Input layer: The network starts with the input layer, which, in this case takes in the flattened 28x28 pixel images from the MNIST dataset (784 input nodes).
- Hidden layer: The hidden layer(s) introduce nin-linearity, allowing the network to learn complex patterns.
- Output layer: The output layer has 10 neurons, corresponding to the 10 possible digit classes (0-9).

### Forward Propogation
- Weights and Biases: You initialize weights and biases for each layer. These parameters are used to perform weighted sums on the inputs. You apply random initialization with small values to prevent over saturization of activations initially.
- Linear Transformation: The input is passed through the network by applying a linear transformation for each layer. Z = X*W + b. Where X is the input, w is the weights and b is the biases.
- Activation Function: After computing Z, an activation function is applied (Like the ReLu function for the hidden layer and softmax for the output). The activation function help intoduce non-linearity and allow the model to approximate complex relationships.

### Loss Function (Cross-Entropy)
A loss function measures how well your predicted porbabilities match the true labels. A lower loss indicates that the model is making better predictions.

### Training the Network
- Backpropogation: This is the process of calculating the gradients of the loss function with respect to the network's weight using the chain rule. You calculate how much each weight contributes to the error and update the weights accordingly.
- Gradient descent: Gradient descent is used to optimize the weights by adjusting them to minimize the loss functino. This involves updating the weights using the gradients computed from backPropogation. W = W - alpha * ∂L/∂W

### Convergence and Accuracy
- As the number of epochs increases, you expect the loss to decrease, and accuracy to improve. Convergence occurs when the loss no longer decreases significantly.

## My Results from my first network
with 1000 epochs 20% accuracy
with 2000 epochs 45.05%
