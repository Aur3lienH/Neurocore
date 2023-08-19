#include "DataLoader.cuh"
#include <algorithm>
#include <random>

DataLoader::DataLoader(MAT*** _data, int _dataLength)
{
    data = _data;
    dataLength = _dataLength;
    rng = std::mt19937(rd());
}


void DataLoader::Shuffle()
{
    std::shuffle(data, data + dataLength, rng);
}


