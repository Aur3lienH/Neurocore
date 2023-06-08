#pragma once
#include "Network.h"
#include "Matrix.h"
#include <iostream>
#include <mutex>
#include <condition_variable>

class ThreadArg
{
public:
    ThreadArg(Network* network, Matrix*** inputs, Matrix*** outputs, std::mutex* mutex, std::condition_variable* cv, int dataLength);
    Network* network;
    Matrix*** inputs;
    Matrix*** outputs;
    std::condition_variable* cv;
    std::mutex* mutex;
    int dataLength;
};