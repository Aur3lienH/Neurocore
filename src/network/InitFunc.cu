#include "network/InitFunc.cuh"

std::mt19937 WeightsInit::rng = std::mt19937(std::random_device{}());