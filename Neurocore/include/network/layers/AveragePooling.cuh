#pragma once

#include "Layer.cuh"

template<typename LayerShape,typename PrevLayerShape,int filterSize, int stride, bool GPU = GPU_DEFAULT>
class AveragePoolLayer final
{
    cudnnPoolingDescriptor_t poolingDescriptor;
    cudnnTensorDescriptor_t forwardInputDesc, forwardOutputDesc;
public:

    using Shape = LayerShape;
    
    AveragePoolLayer(): fs_2(filterSize * filterSize)
    {
        output = new LMAT<LayerShape>();
        newDelta = new LMAT<PrevLayerShape>();
    }

    ~AveragePoolLayer() {
        delete output;
        delete newDelta;
    }

    //static Layer* Load(std::ifstream& reader);

    const LMAT<LayerShape>* FeedForward(const LMAT<PrevLayerShape>* input)
    {
        if constexpr (GPU)
        {
            checkCUDNN(
                   cudnnPoolingForward(cuda->cudnnHandle,
                                       poolingDescriptor,
                                       &cuda->one,
                                       forwardInputDesc,
                                       input->GetData(),
                                       &cuda->zero,
                                       forwardOutputDesc,
                                       output->GetData()));
        }
        else
        {
            LMAT<PrevLayerShape>::template AveragePool<filterSize,stride>(input, output);
        }


        return output;
    }

    LMAT<PrevLayerShape>* BackPropagate(const LMAT<LayerShape>* delta, const LMAT<PrevLayerShape>* previousActivation)
    {
        if constexpr (GPU)
        {
            cudnnPoolingBackward(cuda->cudnnHandle,
                                poolingDescriptor,
                                &cuda->one,
                                forwardOutputDesc,
                                output->GetData(),
                                forwardOutputDesc,
                                delta->GetData(),
                                forwardInputDesc,
                                previousActivation->GetData(),
                                &cuda->zero,
                                forwardInputDesc,
                                newDelta->GetData());
        }
        else
        {
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
                                newDelta->set(i * stride + k, j * stride + l, delta->get(i, j) / fs_2);
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
        }

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
        if constexpr (GPU)
        {
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

            checkCUDNN(cudnnCreateTensorDescriptor(&forwardInputDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(forwardInputDesc,
                                                  CUDNN_TENSOR_NCHW,
                                                  CUDNN_DATA_FLOAT,
                                                  1,
                                                  PrevLayerShape::z,
                                                  PrevLayerShape::x,
                                                  PrevLayerShape::y));
            checkCUDNN(cudnnCreateTensorDescriptor(&forwardOutputDesc));
            checkCUDNN(cudnnSetTensor4dDescriptor(forwardOutputDesc,
                                                  CUDNN_TENSOR_NCHW,
                                                  CUDNN_DATA_FLOAT,
                                                  1,
                                                  LayerShape::z,
                                                  LayerShape::x,
                                                  LayerShape::y));
        }
    }

    void SpecificSave(std::ofstream& writer);

private:
    LMAT<LayerShape>* output = nullptr;
    LMAT<PrevLayerShape>* newDelta = nullptr;
    const int fs_2;
};