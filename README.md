# Neurocore (NEUROCORE Engine Using Recursive Operations for Computational Optimization and Research Excellence)

## 1. Introduction

This repository aims to create a deep learning library from scratch using CRTP (Curiously Recursive Template Pattern). The main goal of this project is to optimize small networks using JIT(Just In Time) compilation so each network is efficiently optimized by the compiler (GNU compiler and CUDA). 



## 2. Installation

### 2.1. Prerequisites

- Make sure you have installed the following packages:
  - g++
  - CUDA toolkit (not mandatory)
  - CUDA (Not mandatory)
  - CUDNN (Not mandatory)
  - >= Python3.8



### 2.2. Installation

- Clone the repository:
```bash
git clone git@github.com:Aur3lienH/Neurocore.git
cd Neurocore

```

- Go to the repository:
```bash
cd DeepLearning/python_lib
```

- Compile the library:
```bash

```

## 3. Usage

### 3.1. Create a neural network

How to create a neural network.

```c++
#include "Network.h"
#include "InputLayer.h"
#include "FCL.h"
#include "ReLU.h"
// Create the neural network.
Network* network = new Network();
//Must start with one Input Layer.
network->AddLayer(new InputLayer(784));
//Hidden layer with 128.
network->AddLayer(new FCL(128, new ReLU()));
//Output layer with 10 neurons.
network->AddLayer(new FCL(10, new Softmax()));
//Compilte the network, with the optimizer and the loss function.
network->Compile(Opti::Adam, new CrossEntropy());
```

### 3.2. Train the neural network

How to train the neural network.

```c++
#include "DataLoader"

// Load the data.
//dataset : dataset[0] = input, dataset[1] = output
DataLoader* data = new DataLoader(dataset, size);

// Train the network.
//epochs : number of epochs (int)
//learningRate : learning rate (float)
//batchSize : size of the batch (int)
//threadCount : number of threads (int)

network->Learn(epochs, learningRate, data, batchSize, threadCount);

```

### 3.3 Predict with the neural network

How to predict with the neural network.

```c++
//input : Matrix of input (MAT)
//out : Matrix of output (MAT)
MAT out = network->Process(input);
```

look at matrix source file for more details [here](./include/matrix/Matrix.cuh) 


### 3.4. Save and load the neural network


```c++
// Save the network.

network->Save("path/to/save");

// Load the network.
Network* network = Network::Load("path/to/load");

```

## 4.Tests

```bash
git clone git@github.com:Aur3lienH/Neurocore.git
cd Neurocore

```

## 5. Challenges

Chanllenges are making this library efficient in both CPU and GPU mode using two different compilers
