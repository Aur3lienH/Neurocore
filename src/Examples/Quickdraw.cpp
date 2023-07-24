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

std::pair<int, int> GetDataLengthAndNumCategories(const std::string& path, const int numDrawingsPerCategory)
{
    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;

    int numInputs = 0, numCategories = 0;

    for (const auto& dirEntry : recursive_directory_iterator(path))
    {
        if (dirEntry.path().extension() != ".bin")
            continue;

        std::ifstream in(dirEntry.path(), std::ios::binary | std::ios::in);
        //int numDrawingsInFile;
        //in.read((char*)(&numDrawingsInFile), sizeof(numDrawingsInFile));
        //numInputs += numDrawingsInFile;
        numInputs += numDrawingsPerCategory;
        numCategories++;
        in.close();
    }

    return {numInputs, numCategories};
}

void QuickDraw1(const int numDrawingsPerCategory)
{
    std::cout << "quickdraw 1\n";
    std::pair<int, int> dataInfo = GetDataLengthAndNumCategories("datasets/Quickdraw", numDrawingsPerCategory);
    const int dataLength = dataInfo.first;
    const int numCategories = dataInfo.second;
    Matrix*** data = GetQuickdrawDataset("datasets/Quickdraw", dataLength, numCategories, numDrawingsPerCategory,
                                         false);
    std::cout << "loaded" << std::endl;

    auto* network = new Network();
    network->AddLayer(new InputLayer(784));
    network->AddLayer(new FCL(128, new ReLU()));
    network->AddLayer(new FCL(numCategories, new Softmax()));
    std::cout << "before compiling !\n";
    network->Compile(Opti::Adam, new CrossEntropy());
    std::cout << "compiled ! \n";

    int trainLength = dataLength * 0.8;
    int testLength = dataLength - trainLength;
    auto* dataLoader = new DataLoader(data, trainLength);
    network->Learn(20, 0.01, dataLoader, 64, 16);

    double trainingAccuracy = TestAccuracy(network, data, 1000);
    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";


    double testingAccuracy = TestAccuracy(network, data + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";
}

void QuickDraw2(const int numDrawingsPerCategory)
{
    std::cout << "quickdraw 1\n";
    std::pair<int, int> dataInfo = GetDataLengthAndNumCategories("datasets/Quickdraw", numDrawingsPerCategory);
    const int dataLength = dataInfo.first;
    const int numCategories = dataInfo.second;
    Matrix*** data = GetQuickdrawDataset("datasets/Quickdraw", dataLength, numCategories, numDrawingsPerCategory, true);
    std::cout << "loaded" << std::endl;

    auto* network = new Network();
    network->AddLayer(new InputLayer(28, 28, 1));
    network->AddLayer(new ConvLayer(new LayerShape(3, 3, 32), new ReLU()));
    network->AddLayer(new MaxPoolLayer(2, 2));
    network->AddLayer(new Flatten());
    network->AddLayer(new FCL(numCategories, new Softmax()));

    network->Compile(Opti::Adam, new CrossEntropy());

    network->PrintNetwork();
    int trainLength = dataLength * 0.8;
    int testLength = dataLength - trainLength;
    auto* dataLoader = new DataLoader(data, trainLength);
    dataLoader->Shuffle();
    network->Learn(10, 0.01, dataLoader, 96, 16);

    double trainingAccuracy = TestAccuracy(network, data, 1000);
    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";


    double testingAccuracy = TestAccuracy(network, data + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";
}

Matrix***
GetQuickdrawDataset(const std::string& path, int dataLength, int numCategories, const int numDrawingsPerCategory,
                    bool format2D)
{
    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;

    auto*** dataset = new Matrix** [dataLength];

    int inputIndex = 0, categoryIndex = 0;
    const int rows = format2D ? 28 : 28 * 28;
    const int cols = format2D ? 28 : 1;

    for (const auto& dirEntry : recursive_directory_iterator(path))
    {
        if (dirEntry.path().extension() != ".bin")
            continue;

        std::ifstream in(dirEntry.path(), std::ios::binary | std::ios::in);
        int numDrawingsInFile = numDrawingsPerCategory;
        //in.read((char*)(&numDrawingsInFile), sizeof(numDrawingsInFile));
        in.seekg(sizeof(int), std::ios::beg);

        for (int i = 0; i < numDrawingsInFile; i++)
        {
            /*if (inputIndex % (2500) == 0)
            {
                const int pos = in.tellg();
                std::cout << dirEntry.path().filename() << std::endl;
                for (int y = 0; y < 28; ++y)
                {
                    for (int x = 0; x < 28; ++x)
                    {
                        auto* a = new unsigned char(0);
                        in.read((char*) a, 1);
                        std::cout << std::setw(3) << (int) *a << " ";
                    }
                    std::cout << std::endl;
                }
                in.seekg(pos);
            }*/
            dataset[inputIndex] = new Matrix* [2];
            dataset[inputIndex][1] = LabelToMatrix(categoryIndex, numCategories);
            dataset[inputIndex][0] = new Matrix(rows, cols);
            for (int p = 0; p < 28 * 28; p++)
            {
                unsigned char pixVal;
                in.read((char*) &pixVal, 1);
                (*dataset[inputIndex][0])[p] = static_cast<double>(pixVal) / 255.0;
            }

            inputIndex++;
        }

        categoryIndex++;

        in.close();
    }

    // Shuffle the dataset
    std::random_device rd;
    auto rng = std::mt19937(rd());
    std::shuffle(dataset, dataset + dataLength, rng);

    return dataset;
}

Matrix* LabelToMatrix(const int label, const int numLabels)
{
    auto* res = new Matrix(numLabels, 1);
    (*res)[label] = 1;

    return res;
}