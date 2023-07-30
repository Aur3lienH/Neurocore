#include "Layer.h"
#include "Matrix.h"
#include "FCL.h"
#include "InputLayer.h"
#include "ConvLayer.h"
#include "LayerShape.h"
#include "MaxPooling.h"
#include "AveragePooling.h"
#include "Flatten.h"


Layer::Layer()
{

}

void Layer::Compile(LayerShape* previousLayer, Opti opti)
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

    Compile(previousLayer);
}

LayerShape* Layer::GetLayerShape()
{
    return layerShape;
}

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

void Layer::Save(std::ofstream& writer)
{
    //Save layer ID
    writer.write(reinterpret_cast<char*>(&LayerID), sizeof(int));
    SpecificSave(writer);
}

Layer::~Layer()
{
    delete layerShape;
    delete optimizer;
}



