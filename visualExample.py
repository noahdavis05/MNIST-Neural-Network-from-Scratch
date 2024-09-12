import pygame
import sys
import numpy as np
from skimage.transform import resize
import math

# functions
# function to turn the drawn_image into a 2d array
def coordinates_to_image(drawn_image, size):
    # Initialize a 2D array of zeros with dimensions size x size
    image = np.zeros((size, size), dtype=int)
    
    # Loop through the coordinates in drawn_image
    for x, y in drawn_image:
        # Ensure coordinates are within bounds
        if 0 <= x < size and 0 <= y < size:
            image[y, x] = 255  # Set the pixel to 1
    
    return image


def resize_image(image, new_size):
    # Resize the image to 28x28
    image_28x28 = resize(image, (28, 28), mode='reflect', anti_aliasing=True)

    # Round to the nearest integer and clip values to the range 0-255
    #image_28x28 = np.round(image_28x28 * 255).astype(np.uint8)

    # Ensure values are within the range 0-255
    #image_28x28 = np.clip(image_28x28, 0, 255).astype(np.uint8)

    # Check the range of values
    print(f"Min value: {image_28x28.min()}, Max value: {image_28x28.max()}")

    return image_28x28

def normalize_image(image):
    """Normalize the image so that the largest value is 255."""
    max_value = np.max(image)
    if max_value > 0:  # Prevent division by zero
        image_normalized = (image / max_value) * 255
    else:
        image_normalized = image
    return image_normalized.astype(np.uint8)


# setup the ai predictions
# Load the weights and biases
w1 = np.load(r'C:\Users\noah\OneDrive\Documents\Python\MNISTnetwork\Clone\MNIST-Intro\n1Results\w1.npy')
b1 = np.load(r'C:\Users\noah\OneDrive\Documents\Python\MNISTnetwork\Clone\MNIST-Intro\n1Results\b1.npy')
w2 = np.load(r'C:\Users\noah\OneDrive\Documents\Python\MNISTnetwork\Clone\MNIST-Intro\n1Results\w2.npy')
b2 = np.load(r'C:\Users\noah\OneDrive\Documents\Python\MNISTnetwork\Clone\MNIST-Intro\n1Results\b2.npy')
w3 = np.load(r'C:\Users\noah\OneDrive\Documents\Python\MNISTnetwork\Clone\MNIST-Intro\n1Results\w3.npy')
b3 = np.load(r'C:\Users\noah\OneDrive\Documents\Python\MNISTnetwork\Clone\MNIST-Intro\n1Results\b3.npy')


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

def predict(drawn_image, w1, b1, w2, b2, w3, b3):
    image = coordinates_to_image(drawn_image, RECT_SIZE)
    image = resize_image(image, SMALLEST_SIZE)
    image = normalize_image(image)
    
    #flatten image into a 1d array
    
    a1, a2, y_hat = forward_propogation(image.flatten()/255, w1, b1, w2, b2, w3, b3)
                                        
    # create a list of the porbabilities for all inputs
    # show the users their results
    text_list = []
    temp_string = f"Predicted Number: {np.argmax(y_hat)}"
    text_surface = render_text(temp_string, BLACK, font, 30)
    text_list.append(text_surface)
    for i in range(10):
        temp_string = f"Probability of {i}: {y_hat[0][i]}"
        text_surface = render_text(temp_string, BLACK, font, 15)
        text_list.append(text_surface)
    

    
    return image, text_list


# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 600, 200
SMALLEST_SIZE = 28
RECT_SIZE = SMALLEST_SIZE*5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0) 

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Drawing in Rectangle")

# set up text
font_size = 36
font = pygame.font.SysFont(None, font_size)
text_list = []

def render_text(text, color, font, size):
    # Create a font object with the specified size
    font = pygame.font.SysFont(None, size)
    # Render the text
    text_surface = font.render(text, True, color)
    return text_surface


# Main loop
running = True
drawing = False
shape_color = RED
shape_start = (0, 0)

# array which stores the pixel values of the drawn image
drawn_image = []

# Create a rect object to draw in
rect_x = 0
rect_y = 0
rect_width = RECT_SIZE
rect_height = RECT_SIZE
drawing_rect = pygame.Rect(rect_x, rect_y, rect_width, rect_height)

# create a rect object to show the shrunk image
shrink_x = 150
shrink_y = 50
shrink_width = SMALLEST_SIZE
shrink_height = SMALLEST_SIZE
shrink_rect = pygame.Rect(shrink_x, shrink_y, shrink_width, shrink_height)

# create two rect objects to be buttons
clear_button = pygame.Rect(50, 150, 100, 50)
predict_button = pygame.Rect(250, 150, 100, 50)

# add a variable to store if mouse is up or down
mouse_down = False
predicted = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                if drawing_rect.collidepoint(event.pos):
                    mouse_down = True
                if clear_button.collidepoint(event.pos):
                    drawn_image = []
                if predict_button.collidepoint(event.pos):
                    predicted = True
                    # convert the drawn image to a 2d array
                    image, text_list = predict(drawn_image, w1, b1, w2, b2, w3, b3)
                    # Iterate through each number in the image array and write each number to a text file
                    

                    #print(image.shape)
                    #print(image)
                    #print(image.flatten())
                    # flatten the image and normalize
                    
                    #print(image_normalized)
                    #print(image_normalized.shape

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if drawing_rect.collidepoint(event.pos):
                    mouse_down = False

    # Define the radius of the circle
    circle_radius = 6  # You can adjust this value

    # This is called when the mouse is down
    if mouse_down:
        mouse_pos = pygame.mouse.get_pos()
        
        # Get the coordinates of the circle's bounding box
        for i in range(-circle_radius, circle_radius + 1):
            for j in range(-circle_radius, circle_radius + 1):
                x = mouse_pos[0] + i
                y = mouse_pos[1] + j
                
                # Calculate distance from the center (mouse_pos) to (x, y)
                distance = math.sqrt(i**2 + j**2)
                
                # If the distance is less than or equal to the radius, draw the pixel
                if distance <= circle_radius:
                    if x >= rect_x and x <= rect_x + rect_width and y >= rect_y and y <= rect_y + rect_height:
                        temp_coord = (x, y)
                        if temp_coord not in drawn_image:
                            drawn_image.append(temp_coord)
    


    # Clear the screen
    screen.fill(WHITE)

    # iterate through drawn image and draw the pixels
    for coord in drawn_image:
        pygame.draw.rect(screen, BLACK, pygame.Rect(coord[0], coord[1], 1, 1))

    # iterate through the 28x28 image and draw the pixels
    if predicted:
        for i in range(SMALLEST_SIZE):
            for j in range(SMALLEST_SIZE):
                if image[i, j] > 0:
                    val = 256 - 1 * image[i, j]
                    pygame.draw.rect(screen, ((val,val,val)), pygame.Rect(shrink_x + j, shrink_y + i, 1, 1))

        for i, text_surface in enumerate(text_list):
            if i == 0:
                screen.blit(text_surface, (150, 0))
            else:
                screen.blit(text_surface, (350, 0 + (i-1)*20))

        
    
    # Draw the rectangle
    pygame.draw.rect(screen, BLACK, drawing_rect, 2)
    

    # Draw the buttons
    pygame.draw.rect(screen, RED, clear_button, 0)
    pygame.draw.rect(screen, GREEN, predict_button, 0)

    # Update the display
    pygame.display.flip()
