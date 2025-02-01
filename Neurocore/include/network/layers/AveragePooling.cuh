#pragma once

#include "Layer.cuh"

template<typename LayerShape,typename PrevLayerShape,int filterSize, int stride>
class AveragePoolLayer final
{
public:
    AveragePoolLayer(): fs_2(filterSize * filterSize)
    {
        output = new LMAT<LayerShape>();
        newDelta = new LMAT<PrevLayerShape>();
    }

    //static Layer* Load(std::ifstream& reader);

    const LMAT<LayerShape>* FeedForward(const LMAT<PrevLayerShape>* input)
    {
#if USE_GPU
        checkCUDNN(
                cudnnPoolingForward(Matrix_GPU::cuda->cudnnHandle,
                                    poolingDescriptor,
                                    &Matrix_GPU::cuda->one,
                                    forwardInputDesc,
                                    input->GetData(),
                                    &Matrix_GPU::cuda->zero,
                                    forwardOutputDesc,
                                    result->GetData()));
#else
        LMAT<PrevLayerShape>::template AveragePool<filterSize,stride>(input, output);
#endif

        return output;
    }

    LMAT<PrevLayerShape>* BackPropagate(const LMAT<LayerShape>* delta, const LMAT<PrevLayerShape>* previousActivation) {
#if USE_GPU
        cudnnPoolingBackward(Matrix_GPU::cuda->cudnnHandle,
                             poolingDescriptor,
                             &Matrix_GPU::cuda->one,
                             forwardOutputDesc,
                             result->GetData(),
                             forwardOutputDesc,
                             delta->GetData(),
                             forwardInputDesc,
                             previousActivation->GetData(),
                             &Matrix_GPU::cuda->zero,
                             forwardInputDesc,
                             newDelta->GetData());
#else
        // All elements in the pooling window have the same delta which is delta / (filterSize * filterSize)
        for (int d = 0; d < LayerShape::z; ++d)
        {
            for (int i = 0; i < LayerShape::x; ++i)
            {
                for (int j = 0; j < LayerShape::y; ++j)
                {
                    for (int k = 0; k < filterSize; ++k)
                    {
                        for (int l = 0; l < filterSize; ++l)
                        {
                            (*newDelta)(i * stride + k, j * stride + l) = (*delta)(i, j) / fs_2;
                        }
                    }
                }
            }
            previousActivation->GoToNextMatrix();
            output->GoToNextMatrix();
            newDelta->GoToNextMatrix();
            delta->GoToNextMatrix();
        }

        previousActivation->ResetOffset();
        output->ResetOffset();
        newDelta->ResetOffset();
        delta->ResetOffset();
#endif

        return newDelta;
    }

    std::string getLayerTitle()
    {
        std::string buf;
        buf += "Layer: AveragePool\n";
        buf += "Size: " + std::to_string(filterSize) + "\n";
        buf += "Stride: " + std::to_string(stride) + "\n";

        return buf;
    }

    void Compile() {

    }

    void SpecificSave(std::ofstream& writer);

#if USE_GPU

    void Compile(LayerShape* previousActivation) override;

#endif

private:
    LMAT<LayerShape>* output = nullptr;
    LMAT<PrevLayerShape>* newDelta = nullptr;
    const int fs_2;
};



#if USE_GPU
void AveragePoolLayer::Compile(LayerShape* previousActivation)
{
    PoolingLayer::Compile(previousActivation);
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDescriptor));
    checkCUDNN(cudnnSetPooling2dDescriptor(poolingDescriptor,
                                           CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                           CUDNN_NOT_PROPAGATE_NAN,
                                           filterSize,
                                           filterSize,
                                           0,
                                           0,
                                           stride,
                                           stride));
}
#endif