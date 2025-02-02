<h2> N.E.U.R.O.C.O.R.E (NEUROCORE Engine Using Recursive Operations for Computational Optimization and Research Excellence)</h2>

## 1. Introduction 🧠

Neurocore aims to create a deep learning library from scratch using CRTP (Curiously Recursive Template Pattern). The main goal of this project is to optimize small networks using JIT(Just In Time) compilation so each network is efficiently optimized by the compiler (GNU compiler and CUDA). 


## 2. Installation to Neurocore👷

### 2.1. Prerequisites

- Make sure you have installed the following packages to use Neurocore:
  - **c++ Compiler**: g++ (version 10.1 or higher)
  - **python** (version 3.8 or higher)
  - **pip** (latest version recommended)
  - **CUDA Toolkit** (optional, version 12.2 or higher)
  - **CUDNN** (optional, version 8.9 or higher)


### 2.2. Installation

- From the Neurocore repository:
```bash
git clone git@github.com:Aur3lienH/Neurocore.git
cd Neurocore
git submodule init
git submodule update
pip install .
```
-From the Neurocore release:
```bash
sudo pip install 
```

## 3.Neurocore Tests 🧪✅

```bash
./run_tests
```
If you see something which is not green, you may be missing packages or the library can't be installed on your computer

## 4.Neurocore Example 📝

### 4.1. Train Mnist on 10 epochs 


```bash
python Mnist.py
```

### 4.2 Explanation of the small example 😎

**Firstly import all of the Neurocore utils**

| Module | Description |
|--------|-------------|
| `Network` | Main class managing the neural network architecture. Handles model construction, training, and inference. Provides the backbone for building and operating the neural network. |
| `FCL` (Fully Connected Layer) | Implements dense layers where each neuron connects to all neurons in the previous layer. Used for creating hidden and output layers, enabling complex pattern recognition. |
| `InputLayer` | Defines the network's first layer that receives data. Specifies input dimensions and initializes data flow through the network. Essential for establishing the network's input structure. |
| `ReLU` | Rectified Linear Unit activation function that introduces non-linearity in the network. Transforms negative values to zero while preserving positive values, helping the network learn complex patterns. |
| `MSE` | Mean Squared Error loss function that calculates the average squared difference between network predictions and target values during training. Guides the network's learning process by quantifying prediction errors. |
| `Matrix` | Core module handling optimized matrix operations for deep learning. Provides efficient implementation of network computations, leveraging hardware-specific optimizations through CRTP. |
| `NumpyToMatrixArray` | Conversion utility that transforms NumPy arrays into Neurocore's internal Matrix format. Ensures seamless compatibility with external data structures while maintaining optimization benefits. |
| `Config` | Configuration module for adjusting verbosity of logs and progress messages during network execution. Controls debugging output and training progress information. |

```python
from Neurocore.network.Network import Network
from Neurocore.network.Layers import FCL, InputLayer
from Neurocore.network.Activation import ReLU
from Neurocore.network.Loss import MSE
from Neurocore.network.Matrix import Matrix, NumpyToMatrixArray
from Neurocore.network.Config import Config
```



How to create the neural network using Neurocore.

**Network for the small example**

```python
net = Network()
net.AddLayer(InputLayer(784))
net.AddLayer(FCL(128, ReLU()))
net.AddLayer(FCL(10, ReLU()))
net.Compile(MSE())
net.Print()
```
!!! Network should always **start** with an **InputLayer** and be **compiled** before usage !!!

| Parameter | Description |
|-----------|-------------|
| X_train | Input data as numpy array (N+1 dimensional, where N is input dimension) |
| Y_train | Expected output as numpy array (same format as input) |
| batch_size | Training batch size - affects learning stability and memory usage |
| num_epochs | Number of complete passes through the training dataset |
| learning_rate | Learning step size (too high may cause instability, too low may cause slow convergence) |


```python
net.Learn(X_train, y_train, batch_size, num_epochs, learning_rate)
```

For inference FeedForward (Just go threw the network)<br>
X_val = input (numpy array)<br>
Y_val = ouptut (numpy array)<br>

```python
Y_val = net.FeedForward(X_val)
```
## 5.Neurocore Objectives 🎯

The main of this repo is to make a deep learning library accessible to everybody which is a little bit aware of the subject.<br>
The library achieves *efficiency* and lightweight networks by:
- Generating **computer-specific** compiled code
- **Optimizing** for each **individual** neural network

Neurocore can have has side effect slightly *better performance* of small networks because of it's easier to retrieve the instructions

### 5.1 Neurocore Challenges and Roadmap 🛣️

- **Performance Optimization**
  - Dual-mode efficiency for CPU and GPU compilation
  - Block matrix operations for FCL networks
  - Optimal matrix layouts for convolution operations
  
- **Feature Development**
  - Implementation of advanced layers (SMorph, LSTM)
  - Extended network architectures
  - Additional optimization strategies


