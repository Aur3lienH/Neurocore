<h2> Neurocore (NEUROCORE Engine Using Recursive Operations for Computational Optimization and Research Excellence)</h2>

## 1. Introduction

This repository aims to create a deep learning library from scratch using CRTP (Curiously Recursive Template Pattern). The main goal of this project is to optimize small networks using JIT(Just In Time) compilation so each network is efficiently optimized by the compiler (GNU compiler and CUDA). 



## 2. Installation 👷

### 2.1. Prerequisites

- Make sure you have installed the following packages:
  - c++ Compiler: g++ (version 10.1 or higher)
  - CUDA Toolkit (optional, version 12.2 or higher)
  - CUDNN (optional, version 8.9 or higher)
  - python (version 3.8 or higher)
  - pip (latest version recommended)


### 2.2. Installation

- From the repository:
```bash
git clone git@github.com:Aur3lienH/Neurocore.git
cd Neurocore
git submodule init
git submodule update
pip install .
```
-From the release:
```bash
sudo pip install 
```

## 3.Tests

```bash
./run_tests
```
If you see something which is not green, you may be missing packages or the library can't be installed on your computer

## 4. Example

### 4.1. Train Mnist on 10 epochs

How to create a neural network.

```bash
python Mnist.py
```

**Network for the small example**

```python
net = Network()
net.AddLayer(InputLayer(784))
net.AddLayer(FCL(128, ReLU()))
net.AddLayer(FCL(10, ReLU()))
net.Compile(MSE())
net.Print()
```
!!! Network should always start with an InputLayer and be compiled before used !!!

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
## 5. Objective of this repo

The main of this repo is to make a deep learning library accessible to everybody which is a little bit aware of the subject.<br>
Making it efficient under the hood and lightweight during inference by compiling specically for the computer the library is running on and the specific network.
This can have has side effect slightly better performance because of it's easier to retrieve the instructions

## Challenges and Roadmap

- **Performance Optimization**
  - Dual-mode efficiency for CPU and GPU compilation
  - Block matrix operations for FCL networks
  - Optimal matrix layouts for convolution operations
  
- **Feature Development**
  - Implementation of advanced layers (SMorph, LSTM)
  - Extended network architectures
  - Additional optimization strategies


