#pragma once

#include "tools/Serializer.h"
#include "matrix/Matrix.cuh"
#include "network/LayerShape.cuh"
#include "network/optimizers/Optimizer.h"

template<typename Derived>
class Layer final
{
public:

	using Shape = typename Derived::Shape;

    Layer();

    virtual ~Layer();

	template<int rows_in, int cols_in, int dims_in, int rows_out, int cols_out, int dims_out>
    const MAT<rows_out,cols_out,dims_out>* FeedForward(const MAT<rows_in,cols_in,dims_in>* input)
	{
	    return static_cast<Derived*>(this)->FeedForward(input);
	}

	template<int rows_in, int cols_in, int dims_in, int rows_out, int cols_out, int dims_out, int rows_in2, int cols_in2, int dims_in2>
    const MAT<rows_out,cols_out,dims_out>* BackPropagate(const MAT<rows_in,cols_in,dims_in>* delta, const MAT<rows_in2,cols_in2,dims_in2>* previousActivation)
    {
	    return static_cast<Derived*>(this)->BackPropagate(delta,previousActivation);
    }

	template<int rows, int cols, int dims>
    [[nodiscard]] const MAT<rows,cols,dims>* getResult() const
    {
	    return static_cast<Derived*>(this)->getResultImpl();
    }

    void ClearDelta()
    {
	    static_cast<Derived*>(this)->ClearDelta();
    }

    void UpdateWeights(double learningRate, int batchSize)
    {
	    static_cast<Derived*>(this)->UpdateWeights(learningRate, batchSize);	    
    }

    void AddDeltaFrom(Layer* layer)
    {
	    static_cast<Derived*>(this)->AddDeltaFrom(layer);
    }

    void AverageGradients(int batchSize)
    {
	    static_cast<Derived*>(this)->AverageGradients(batchSize);
    }

    //Must define the layerShape !
	template<int x, int y, int z, int size>
    void Compile(LayerShape<x,y,z,size>* previousOutput, Opti opti)
    {
	    static_cast<Derived*>(this)->Compile(previousOutput,opti);
    }

	template<int x, int y, int z, int size>
    void Compile(LayerShape<x,y,z,size>* previousOutput)
    {
	    static_cast<Derived*>(this)->Compile(previousOutput);
    }

	template<int x, int y, int z, int size>
    LayerShape<x,y,z,size>* GetLayerShape()
    {
	    return static_cast<Derived*>(this)->GetLayerShape();
    }

    std::string getLayerTitle();

    Layer* Clone()
    {
	    return static_cast<Derived*>(this)->Clone();
    }

	//Disable because of templates

    //static Layer* Load(std::ifstream& reader);

    //void SpecificSave(std::ofstream& writer);

    //void Save(std::ofstream& writer);

};

template<typename Derived>
Layer<Derived>::Layer()
{

}
/*
template<typename Derived>
void Layer<Derived>::Compile(LayerShape<>* previousLayer, Opti opti)
{

	switch (opti)
	{
	case Opti::Constant :
		optimizer = new Constant();
		break;
	case Opti::Adam :
		optimizer = new Adam();
		break;
	default:
		throw std::invalid_argument("Layer Constructor : Invalid Optimizer ! ");
	}

	std::cout << "here here \n";
	Compile(previousLayer);
	std::cout << "bruh bruh \n";
}
*/


/*
template<typename Derived>
Layer* Layer::Load(std::ifstream& reader)
{
	//Load layerID
	int layerID;
	reader.read(reinterpret_cast<char*>(&layerID), sizeof(int));
	switch (layerID)
	{
	case 0:
		{
			return FCL::Load(reader);
		}
	case 1:
		{
			return InputLayer::Load(reader);
		}
	case 2:
		{
			return ConvLayer::Load(reader);
		}
	case 3:
		{
			return Flatten::Load(reader);
		}
	case 4:
		{
			return MaxPoolLayer::Load(reader);
		}
	case 5:
		{
			return AveragePoolLayer::Load(reader);
		}

	default:
		throw std::invalid_argument("Invalid ID for loading layers !");
	}

}

template<typename Derived>
void Layer::Save(std::ofstream& writer)
{
	//Save layer ID
	writer.write(reinterpret_cast<char*>(&LayerID), sizeof(int));
	SpecificSave(writer);
}
*/

template<typename Derived>
Layer<Derived>::~Layer()
{

}





