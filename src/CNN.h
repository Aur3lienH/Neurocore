#pragma once
#include "vector.h"
#include "Matrix.h"



class CNN
{
public:
    CNN(Vector2<double> input);
private:
    Matrix* filtre;
};
