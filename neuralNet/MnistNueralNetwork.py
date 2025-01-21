#Nerual Network Mnist dataset plus pygame drawing numbers and pressing enter for the models prediction of the drawn number.
#Model is saved as model_weights.npz --- current model weight stats are 98% accuracy on test data and 100% accuracy on training data
# This model was trained with input layer of 28*28, 1 hidden layer of 256 neurons, and output layer of 10 neurons
# Mnist data set consists of 60,000 training images and 10,000 test images of handwritten digits
# The model is trained with 10000 iterations and a learning rate of 0.01 

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist #only used to load MNIST dataset
import pygame
import os
from scipy.ndimage import gaussian_filter



def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)


# to obtain output prediction we must forward propagate the input data through the network
def predict(x, w1, b1, W2, b2):
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True)) 
    a2 = a2 / np.sum(a2, axis=1, keepdims=True)
    return a2 # returns the output layer prediction


# def draw_network(screen, w1, b1, W2, b2, a1, a2, iteration, loss):
#     screen.fill((0, 0, 0))
#     input_nodes = 784  # Corrected to 784
#     hidden_nodes = len(b1[0])
#     output_nodes = len(b2[0])

#     # Positions for nodes
#     input_positions = [(50, 50 + i * 2) for i in range(input_nodes)]
#     hidden_positions = [(200, 50 + i * 20) for i in range(hidden_nodes)]
#     output_positions = [(350, 50 + i * 20) for i in range(output_nodes)]

#     # Draw input layer
#     for i, pos in enumerate(input_positions):
#         pygame.draw.circle(screen, (255, 255, 255), pos, 2)

#     # Draw hidden layer
#     for i, pos in enumerate(hidden_positions):
#         color_value = int(np.clip(a1[0][i] * 255, 0, 255))
#         color = (color_value, 0, 0)
#         pygame.draw.circle(screen, color, pos, 5)

#     # Draw output layer
#     for i, pos in enumerate(output_positions):
#         color_value = int(np.clip(a2[0][i] * 255, 0, 255))
#         color = (color_value, 0, 0)
#         pygame.draw.circle(screen, color, pos, 5)

#     # Draw lines between input and hidden layer
#     for i, input_pos in enumerate(input_positions):
#         for j, hidden_pos in enumerate(hidden_positions):
#             weight = w1[i, j]
#             color_value = int(np.clip(weight * 255, 0, 255))
#             color = (color_value, color_value, color_value)
#             pygame.draw.line(screen, color, input_pos, hidden_pos)

#     # Draw lines between hidden and output layer
#     for i, hidden_pos in enumerate(hidden_positions):
#         for j, output_pos in enumerate(output_positions):
#             weight = W2[i, j]
#             color_value = int(np.clip(weight * 255, 0, 255))
#             color = (color_value, color_value, color_value)
#             pygame.draw.line(screen, color, hidden_pos, output_pos)

#     # Display iteration and loss
#     font = pygame.font.SysFont(None, 36)
#     text = font.render(f"Iteration: {iteration}, Loss: {loss:.4f}", True, (255, 255, 255))
#     screen.blit(text, (10, 10))

#     pygame.display.flip()


def model(x_train, y_train, x_test, y_test, size=2048, learning_rate=0.015, iterations=1000):
    input_size = x_train.shape[1]
    output_size = 10

    # Initialize random weights and biases 
  
    np.random.seed(0)
    w1 = np.random.randn(input_size, size) * 0.01 # Weights between input layer and hidden layer 
    b1 = np.zeros((1, size)) # Biases for hidden layer
    W2 = np.random.randn(size, output_size) * 0.01 # Weights between hidden layer and output layer
    b2 = np.zeros((1, output_size)) # Biases for output layer

    epsilon = 1e-8  # Small value to prevent log(0) which caused NaN values


    #visualation code taken out bc it was extermely slow... it looked cool
    #pygame.init()
    #screen = pygame.display.set_mode((1920, 1080))
    #pygame.display.set_caption("Neural Network Training Visualization")

    for i in range(iterations):
        # Forward propagation
        z1 = np.dot(x_train, w1) + b1 # z1 = x*w1 + b1 transfrom the data linearly to the hidden layer
        a1 = relu(z1) # Apply relu activation function to introduce non-linearity to learn more complex patterns
        z2 = np.dot(a1, W2) + b2 # z2 = a1*W2 + b2 transfrom the data linearly to the output layer
        a2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))  # Subtract max to prevent overflow
        a2 = a2 / np.sum(a2, axis=1, keepdims=True) # Softmax activation function applied to z2 to get the probabilities 

        # Loss function measures how well the neural network's predictions match the actual target values
        log_prob = -np.log(a2[range(len(y_train)), np.argmax(y_train, axis=1)] + epsilon) # Cross entropy loss function computes the negative log of the predicted probabilities for the true classes
        loss = np.sum(log_prob) / len(y_train) # average of the log probabilities over all training examples

        # Backward propagation to calculate the gradients of the loss with respect to the weights and biases
        dz2 = a2 - y_train # Error in the output layer
        dW2 = np.dot(a1.T, dz2) # Gradient of the loss with respect to the weights in the output layer
        db2 = np.sum(dz2, axis=0, keepdims=True) # Calculate the gradiant of loss with respect to biases in the output layer
        da1 = np.dot(dz2, W2.T) # Gradiant of loss with repsect to the activations in the hidden layer
        dz1 = da1 * relu_derivative(z1) # Gradiant of loss with respect to the linear transformation in the hidden layer
        dW1 = np.dot(x_train.T, dz1) # Gradiant of loss with respect to the weights in the hidden layer
        db1 = np.sum(dz1, axis=0, keepdims=True) # Gradiant of loss with respect to the biases in the hidden layer

        # Gradient clipping to prevent values from becoming to large which was an issue 
        dW2 = np.clip(dW2, -1, 1) 
        db2 = np.clip(db2, -1, 1)
        dW1 = np.clip(dW1, -1, 1)
        db1 = np.clip(db1, -1, 1)

        # Update weights and biases
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        w1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        # Print loss every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}, Loss: {loss}")

        # Draw the network
        #draw_network(screen, w1, b1, W2, b2, a1, a2, i + 1, loss)

    y_pred_train = predict(x_train, w1, b1, W2, b2)
    y_pred_test = predict(x_test, w1, b1, W2, b2)
    # 
    train_accuracy = np.mean(np.argmax(y_pred_train, axis=1) == np.argmax(y_train, axis=1)) * 100
    test_accuracy = np.mean(np.argmax(y_pred_test, axis=1) == y_test) * 100

    # Calculate training accuracy
    train_pred_labels = np.argmax(y_pred_train, axis=1) # Predicted labels
    train_true_labels = np.argmax(y_train, axis=1) # One-hot to categorical
    train_correct_predictions = train_pred_labels == train_true_labels # Correct predictions labeled true and incorrect labeled false
    train_accuracy = np.mean(train_correct_predictions) * 100 # Accuracy of correct predictions

    # Calculate test accuracy
    test_pred_labels = np.argmax(y_pred_test, axis=1) # Predicted labels
    test_correct_predictions = test_pred_labels == y_test # Correct predictions labeled true and incorrect labeled false
    test_accuracy = np.mean(test_correct_predictions) * 100 # Accuracy of correct predictions

    print(f"Train Accuracy: {train_accuracy}%")
    print(f"Test Accuracy: {test_accuracy}%")

    # Save weights and biases
    np.savez('model_weights.npz', w1=w1, b1=b1, W2=W2, b2=b2)

    return w1, b1, W2, b2

def load_model():
    if os.path.exists('model_weights.npz'):
        data = np.load('model_weights.npz')
        return data['w1'], data['b1'], data['W2'], data['b2']
    else:
        return None, None, None, None

def draw_and_predict(w1, b1, W2, b2):
    pygame.init()
    screen = pygame.display.set_mode((280, 280))
    pygame.display.set_caption("Draw a digit")
    clock = pygame.time.Clock()
    canvas = np.zeros((28, 28))

    drawing = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    x = canvas.reshape(1, -1)
                    x = x / 255.0  # Normalize the input
                    #print(x)
                    x = gaussian_filter(x, sigma=1)
                    probabilities = predict(x, w1, b1, W2, b2)
                    prediction = np.argmax(probabilities)
                    confidence = np.max(probabilities)
                    print(f"Predicted Digit: {prediction} with confidence: {confidence:.2f}")
                    #prediction = np.argmax(predict(x, w1, b1, W2, b2))
                    #print(f"Predicted Digit: {prediction}")
                elif event.key == pygame.K_c:
                    canvas.fill(0)
                    screen.fill((0, 0, 0))

        if drawing:
            x, y = pygame.mouse.get_pos()
            if 0 <= x < 280 and 0 <= y < 280:
                canvas[y // 10, x // 10] = 255
                pygame.draw.rect(screen, (255, 255, 255), (x // 10 * 10, y // 10 * 10, 10, 10))

        pygame.display.flip()
        clock.tick(60)

def main():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    # One-hot encode the labels to represent the digits 0-9 as a 10-dimensional vector
    num_classes = 10
    y_train_one_hot = np.zeros((y_train.shape[0], num_classes))
    y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1

    # Load model if exists, otherwise train
    w1, b1, W2, b2 = load_model()
    if w1 is None:
        w1, b1, W2, b2 = model(x_train, y_train_one_hot, x_test, y_test)


    # Uncomment to see test example pictures
    # plt.figure(figsize=(10, 5))
    # for i in range(20):
    #     plt.subplot(1, 20, i+1)
    #     plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    #     plt.title(f"Label: {y_test[i]}")
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    draw_and_predict(w1, b1, W2, b2)
    
  
if __name__ == '__main__':
    main()