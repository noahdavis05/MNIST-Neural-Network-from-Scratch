import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as numpy

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # Normalize the values to be between 0 and 1

# Build the model, 1 input layer, 1 hidden layer, and 1 output layer.
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Input layer: flattens 28x28 image into a 784-dimensional vector
    layers.Dense(128, activation='relu'),  # Hidden layer: 128 neurons, ReLU activation
    layers.Dense(10, activation='softmax') # Output layer: 10 neurons (one for each digit), softmax for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Test the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
