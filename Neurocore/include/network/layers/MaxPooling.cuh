#pragma once

#include "matrix/Matrix.cuh"


template<typename LayerShape,typename PrevLayerShape,int filterSize, int stride, bool GPU = GPU_DEFAULT>
class MaxPoolLayer final
{
    cudnnPoolingDescriptor_t poolingDescriptor;
    cudnnTensorDescriptor_t forwardInputDesc, forwardOutputDesc;
public:
    MaxPoolLayer()
    {
        output = new LMAT<LayerShape>();
        newDelta = new LMAT<PrevLayerShape>();
    }

    void Compile() {
        if constexpr (GPU)
        {
            checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDescriptor));
            checkCUDNN(cudnnSetPooling2dDescriptor(poolingDescriptor,
                                                   CUDNN_POOLING_MAX,
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
            //result->Reshape(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);
            LMAT<PrevLayerShape>::template MaxPool<filterSize,stride>(input, output);
        }

		return output;
    }

    LMAT<PrevLayerShape>* BackPropagate(const LMAT<LayerShape>* delta, const LMAT<PrevLayerShape>* previousActivation)
    {
        if constexpr (GPU)
        {
            checkCUDNN(
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
                                        newDelta->GetData()));
        }
        else
        {
            // The idea is that if an element is the maximum than maxPool has selected, then the delta is
            // the same as the previous delta, because the current element is the only one affecting the result.

            for (int m = 0; m < LayerShape::z; m++)
            {
                for (int i = 0; i < LayerShape::x; ++i)
                {
                    for (int j = 0; j < LayerShape::y; ++j)
                    {
                        for (int k = 0; k < filterSize; ++k)
                        {
                            for (int l = 0; l < filterSize; ++l)
                            {
                                const int r = i * stride + k;
                                //if (r >= previousActivation->GetRows())
                                //    continue;
                                const int c = j * stride + l;
                                //if (c >= previousActivation->GetCols())
                                //    continue;
                                //std::cout << m  << "  " << i << "  " << j << "  " << k << "  " << l << "\n";
                                //std::cout << r << " : x y : " << c << "\n";
                                //std::cout << (*previousActivation)(r,c) << "\n";

                                if (r >= previousActivation->GetRows())
                                    continue;
                                if (c >= previousActivation->GetCols())
                                    continue;


                                if (previousActivation->get(r, c) == output->get(i, j))
                                    newDelta->set(r, c, delta->get(i, j));
                                // Should already be 0
                                //else
                                //    (*newDelta)(r,c) = 0.0;
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


        //std::cout << *delta;

        return newDelta;
    }

    std::string getLayerTitle()
    {
        std::string buf;
        buf += "Layer : MaxPool\n";
        buf += "Size: " + std::to_string(filterSize) + "\n";
        buf += "Stride: " + std::to_string(stride) + "\n";
        buf += "Output : " + LayerShape::GetDimensions() + "\n";
        return buf;
    }
private:
    LMAT<LayerShape>* output = nullptr;
    LMAT<PrevLayerShape>* newDelta = nullptr;

    //void SpecificSave(std::ofstream& writer);
};






/*
Layer* MaxPoolLayer::Load(std::ifstream& reader)
{
    int _filterSize;
    int _tempStride;
    reader.read(reinterpret_cast<char*>(&_filterSize), sizeof(int));
    reader.read(reinterpret_cast<char*>(&_tempStride), sizeof(int));
    return new MaxPoolLayer(_filterSize, _tempStride);
}
*/

/*
void MaxPoolLayer::SpecificSave(std::ofstream& writer)
{
    int tempFilterSize = filterSize;
    int tempStride = stride;
    writer.write(reinterpret_cast<char*>(&tempFilterSize), sizeof(int));
    writer.write(reinterpret_cast<char*>(&tempStride), sizeof(int));
}
*/