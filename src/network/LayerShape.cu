#include "network/LayerShape.cuh"
#include "matrix/Matrix.cuh"
#include <iostream>


/*

template<int rows, int cols, int dims, int size>
LayerShape<rows, cols, dims, size>* LayerShape<rows, cols, dims, size>::Load(std::ifstream& reader)
{
    int rows;
    int cols;
    int dims;
    int size;
    reader.read(reinterpret_cast<char*>(&rows), sizeof(int));
    reader.read(reinterpret_cast<char*>(&cols), sizeof(int));
    reader.read(reinterpret_cast<char*>(&dims), sizeof(int));
    reader.read(reinterpret_cast<char*>(&size), sizeof(int));
    return new LayerShape(rows, cols, dims, size);
}

template<int rows, int cols, int dims, int size>
void LayerShape<rows, cols, dims, size>::Save(std::ofstream& writer)
{
    writer.write(reinterpret_cast<char*>(rows), sizeof(int));
    writer.write(reinterpret_cast<char*>(cols), sizeof(int));
    writer.write(reinterpret_cast<char*>(dims), sizeof(int));
    writer.write(reinterpret_cast<char*>(size), sizeof(int));
}

*/

