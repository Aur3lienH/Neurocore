#pragma once

#include "matrix/Matrix.cuh"


template<typename LayerShape,typename PrevLayerShape,int filterSize, int stride>
class MaxPoolLayer final
{

public:
    MaxPoolLayer()
    {
        output = new LMAT<LayerShape>();
        newDelta = new LMAT<PrevLayerShape>();
    }

    void Compile() {

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

		//result->Reshape(layerShape->dimensions[0], layerShape->dimensions[1], layerShape->dimensions[2]);
		LMAT<PrevLayerShape>::template MaxPool<filterSize,stride>(input, output);
#endif

		return output;
    }

    LMAT<PrevLayerShape>* BackPropagate(const LMAT<LayerShape>* delta, const LMAT<PrevLayerShape>* previousActivation)
    {
        #if USE_GPU
    checkCUDNN(
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
                                 newDelta->GetData()));
#else
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


                            if ((*previousActivation)(r, c) == (*output)(i, j))
                                (*newDelta)(r, c) = (*delta)(i, j);
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
    #endif


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

#if USE_GPU
    void Compile(LayerShape* previousActivation);
#endif
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
#if USE_GPU

void MaxPoolLayer::Compile(LayerShape* previousActivation)
{
    PoolingLayer::Compile(previousActivation);
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
}

#endif
