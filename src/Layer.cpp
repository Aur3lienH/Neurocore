#include "Layer.h"
#include "Matrix.h"
#include "FCL.h"
#include "InputLayer.h"
#include "LastLayer.h"


Layer::Layer(int* NeuronsCount, int NeuronsCountSize)
{
    this->NeuronsCount = NeuronsCount;
    this->NeuronsCountSize = NeuronsCountSize;
}

int Layer::getNeuronsCount(int index)
{
    return NeuronsCount[index];
}

Layer* Layer::Load(std::ifstream& reader)
{
    //Load layerID
    int layerID;
    reader.read(reinterpret_cast<char*>(&layerID),sizeof(int));
    switch (layerID)
    {
        case 0:
        {
            return FCL::Load(reader);
            break;
        }
        
        case 1:
        {
            return InputLayer::Load(reader);
            break;
        }

        case 2:
        {
            return LastLayer::Load(reader);
            break;
        }

        default:
            throw std::invalid_argument("Invalid ID for loading layers !");
            break;
    } 

}

void Layer::Save(std::ofstream& writer)
{
    //Save layer ID
    writer.write(reinterpret_cast<char*>(&LayerID),sizeof(int));
    SpecificSave(writer);
}



