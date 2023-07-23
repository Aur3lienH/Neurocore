#pragma once
#include "Matrix.h"
#include <cmath>
#include <random>


class ImagePreProcessing
{
public:
    static Matrix* ImageWithNoise(const Matrix* original);
    static Matrix* ToBlackAndWhite(Matrix** original, bool alpha);
    static Matrix* DownSize(Matrix* input, int rows, int cols);
    static Matrix* Rotate(const Matrix* input, double angle);
    static Matrix* Stretch(const Matrix* input, double x, double y);
private:
    static double Random(double mean, double stddev);
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution;
};  