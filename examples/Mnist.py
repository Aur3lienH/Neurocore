from Neurocore.network.Network import Network
from Neurocore.network.Layers import FCL, InputLayer
from Neurocore.network.Activation import ReLU
from Neurocore.network.Loss import MSE
from Neurocore.network.Matrix import Matrix, NumpyToMatrixArray
from Neurocore.network.Config import Config
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

#Set the verbose
Config.VERBOSE = True
Config.USE_GPU = True

# Load MNIST dataset from scikit-learn
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False,parser='auto')

# Split into train and validation/test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Split temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Preprocess the data
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

def to_one_hot(y, num_classes=10):
    y = y.astype(int)
    return np.eye(num_classes)[y]

y_train = to_one_hot(y_train)
y_val = to_one_hot(y_val)
y_test = to_one_hot(y_test)

# Create and configure the network
net = Network()
net.AddLayer(InputLayer(784))
net.AddLayer(FCL(128, ReLU()))
net.AddLayer(FCL(10, ReLU()))
net.Compile(MSE())
net.Print()

# Train the network
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Train on training set
print(f"\nTraining on {len(X_train)} samples...")
net.Learn(X_train, y_train, batch_size, num_epochs, learning_rate)

# Now evaluate on validation set first
print("\nEvaluating on validation set...")
correct_val = 0
total_val = len(X_val)

for i in range(total_val):
    
    prediction = net.FeedForward(X_val[i])
    if np.argmax(prediction) == np.argmax(y_val[i]):
        correct_val += 1

val_accuracy = (correct_val / total_val) * 100
print(f"\nValidation Accuracy: {val_accuracy:.2f}%")
print(f"Correct predictions: {correct_val}/{total_val}")

# Finally evaluate on test set
print("\nEvaluating on test set...")
correct_test = 0
total_test = len(X_test)

for i in range(total_test):
    
    prediction = net.FeedForward(X_test[i])
    if np.argmax(prediction) == np.argmax(y_test[i]):
        correct_test += 1

test_accuracy = (correct_test / total_test) * 100
print(f"\nTest Accuracy: {test_accuracy:.2f}%")
print(f"Correct predictions: {correct_test}/{total_test}")
