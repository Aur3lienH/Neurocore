#include "DataLoader.cuh"
#include <algorithm>
#include <random>

#if USE_GPU
DataLoader::DataLoader(Matrix_GPU*** _data, int _dataLength)
#else
DataLoader::DataLoader(Matrix*** _data, int _dataLength)
#endif
{
    data = _data;
    dataLength = _dataLength;
    rng = std::mt19937(rd());
}


void DataLoader::Shuffle()
{
    std::shuffle(data, data + dataLength, rng);
}


