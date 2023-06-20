//
// Created by matmu on 20/06/2023.
//

#ifndef DEEPLEARNING_CONVLAYER_H
#define DEEPLEARNING_CONVLAYER_H


#include "Matrix.h"

class ConvLayer{
    MatrixCarre* filter;

public:
    explicit ConvLayer(MatrixCarre* _filter);
    void Convolve(Matrix* input, Matrix* output);
    void FullConvolve(Matrix* input, Matrix* output);
};
#endif //DEEPLEARNING_CONVLAYER_H
