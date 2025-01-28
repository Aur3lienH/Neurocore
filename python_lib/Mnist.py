from src.network.Network import Network
from src.network.Layers import FCL, InputLayer
from src.network.Activation import ReLU
from src.network.Loss import MSE
from src.network.Matrix import Matrix, NumpyToMatrixArray
import numpy as np
from keras.datasets import mnist

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
# Normalize pixel values to range [0,1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape images to 1D arrays (flatten from 28x28 to 784)
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

# Convert labels to one-hot encoding
def to_one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train = to_one_hot(y_train)
y_test = to_one_hot(y_test)

# Create and configure the network
net = Network()
net.AddLayer(InputLayer(784))  # 28x28 = 784 input neurons
net.AddLayer(FCL(128, ReLU()))  # Hidden layer with 128 neurons
net.AddLayer(FCL(10, ReLU()))   # Output layer with 10 neurons (one per digit)

net.Compile(MSE())
net.Print()

# Train the network
# Note: You might want to use a smaller subset of data initially for testing
batch_size = 32
num_epochs = 10
learning_rate = 0.01

# Take a small subset for initial testing
train_subset = 1000
X_train_small = X_train[:train_subset]
y_train_small = y_train[:train_subset]

net.Learn(X_train_small, y_train_small, batch_size, num_epochs, learning_rate)

# Test the network with a single image
test_image = X_test[0]
prediction = net.FeedForward(test_image)
print(f"Predicted digit: {np.argmax(prediction)}")
print(f"Actual digit: {np.argmax(y_test[0])}")
