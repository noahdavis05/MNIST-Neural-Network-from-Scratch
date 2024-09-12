import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the MNIST dataset (replace with the path where your dataset is saved)
x_test = np.load('./mnist/x_test.npy')
y_test = np.load('./mnist/y_test.npy')

# Step 2: Select 10 random examples from the test set
num_samples = 10
sample_indices = np.random.choice(x_test.shape[0], num_samples, replace=False)
x_test_samples = x_test[sample_indices]
y_test_samples = y_test[sample_indices]

# Step 3: Preprocess the selected images
# Flatten the images (28x28 -> 784)
x_test_samples_flattened = x_test_samples.reshape(num_samples, 28*28)

# Normalize the pixel values
x_test_samples_normalized = x_test_samples_flattened / 255.0

# Step 4: Display the images (Optional - just to see what we're testing)
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(x_test_samples[i], cmap='gray')
    plt.axis('off')
plt.show()

# Step 5: Use the neural network to predict on these samples
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Softmax activation function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Shift values to avoid overflow
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward_propogation(x, w1, b1, w2, b2):
    # Input to hidden layer
    z1 = np.dot(x, w1) + b1 
    a1 = sigmoid(z1)

    # hidden layer to output layer
    z2 = np.dot(a1, w2) + b2
    y_hat = softmax(z2) # final output probabilities

    return a1, y_hat

# Load the weights and biases
w1 = np.load('./n1Results/w1.npy')
b1 = np.load('./n1Results/b1.npy')
w2 = np.load('./n1Results/w2.npy')
b2 = np.load('./n1Results/b2.npy')


a1, y_hat = forward_propogation(x_test_samples_normalized, w1, b1, w2, b2)

# Step 6: Print the predicted labels
predicted_labels = np.argmax(y_hat, axis=1)
print("Predicted labels:", predicted_labels)
print("True labels:", y_test_samples)







