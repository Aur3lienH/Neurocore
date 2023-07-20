#include "Quickdraw.h"
#include <filesystem>
#include "Mnist.h"
#include "../Network.h"
#include "../Matrix.h"
#include "../Layer.h"
#include "../InputLayer.h"
#include "../FCL.h"
#include "../ConvLayer.h"
#include "../Flatten.h"
#include "../MaxPooling.h"
#include "../Optimizers.h"
#include "Tools.h"
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <sstream>

std::pair<int, int> GetDataLengthAndNumCategories(const std::string& path){
    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;

    int numInputs = 0, numCategories = 0;

    for (const auto& dirEntry : recursive_directory_iterator(path))
    {
        if (dirEntry.path().extension() != ".bin")
            continue;

        std::ifstream in(dirEntry.path(), std::ios::binary | std::ios::in);
        int numDrawingsInFile = 20000;
        //in.read((char*)(&numDrawingsInFile), sizeof(numDrawingsInFile));
        numInputs += numDrawingsInFile;
        numCategories++;
        in.close();
    }

    return {numInputs, numCategories};
}

void QuickDraw1(){
    std::cout << "quickdraw 1\n";
    std::pair<int, int> dataInfo = GetDataLengthAndNumCategories("datasets/Quickdraw");
    std::cout<<"info"<<std::endl;
    const int dataLength = dataInfo.first;
    const int numCategories = dataInfo.second;
    Matrix*** data = GetQuickdrawDataset("datasets/Quickdraw", dataLength, numCategories);
    std::cout<<"loaded"<<std::endl;

    

    Network* network = new Network();
    network->AddLayer(new InputLayer(784));
    network->AddLayer(new FCL(128, new ReLU()));
    network->AddLayer(new FCL(10, new Softmax()));
    std::cout << "before compiling !\n";
    network->Compile(Opti::Adam,new CrossEntropy());
    std::cout << "compiled ! \n";
    int trainLength = dataLength * 0.8;
    int testLength = dataLength - trainLength;
    network->Learn(20,0.01,new DataLoader(data,trainLength), 64,16);

    double trainingAccuracy = TestAccuracy(network,data, 1000);
    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";


    double testingAccuracy = TestAccuracy(network,data + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";
}

Matrix*** GetQuickdrawDataset(const std::string& path, const int dataLength, const int numCategories){
    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;

    Matrix*** dataset = new Matrix**[dataLength];

    int inputIndex = 0, categoryIndex = 0;

    for (const auto& dirEntry : recursive_directory_iterator(path))
    {
        if (dirEntry.path().extension() != ".bin")
            continue;

        std::ifstream in(dirEntry.path(), std::ios::binary | std::ios::in);
        int numDrawingsInFile = 20000;
        //in.read((char*)(&numDrawingsInFile), sizeof(numDrawingsInFile));
        
        for (int i = 0; i < numDrawingsInFile; i++){
            dataset[inputIndex] = new Matrix*[2];
            dataset[inputIndex][0] = LabelToMatrix(categoryIndex, numCategories);
            dataset[inputIndex][1] = new Matrix(28*28,1);
            for (int p = 0; p < 28 * 28; p++){
                unsigned char pixVal;
                in.read((char*)&pixVal, 1);
                (*dataset[inputIndex][1])[p] = static_cast<double>(pixVal) / 255.0;
            }

            inputIndex++;
        }

        categoryIndex++;

        in.close();
    }

    return dataset;
}

Matrix* LabelToMatrix(const int label, const int numLabels){
    Matrix* res = new Matrix(numLabels, 1);
    (*res)[label] = 1;

    return res;
}