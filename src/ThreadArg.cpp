#include "ThreadArg.h"

ThreadArg::ThreadArg(Network* network, Matrix*** inputs, Matrix*** outputs, std::mutex* mutex, std::condition_variable* cv, int dataLength)
{
    this->network = network;
    this->inputs = inputs;
    this->outputs = outputs;
    this->mutex = mutex;
    this->cv = cv;
    this->dataLength = dataLength;
}