#include "Quickdraw.h"
#include <filesystem>
#include "Mnist.h"
#include "../InputLayer.cuh"
#include "../FCL.cuh"
#include "../ConvLayer.cuh"
#include "../Flatten.cuh"
#include "../MaxPooling.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#if USE_GPU
#else

#include <thread>

#endif

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
    MAT*** data = GetQuickdrawDataset("datasets/Quickdraw", dataLength, numCategories, numDrawingsPerCategory,
                                      false);

    std::cout << "loaded" << std::endl;

    auto* network = new Network();
    network->AddLayer(new InputLayer(784));
    network->AddLayer(new FCL(128, new ReLU()));
    network->AddLayer(new FCL(numCategories, new Softmax()));
    std::cout << "before compiling !\n";
    network->Compile(Opti::Adam, new CrossEntropy());
    std::cout << "compiled ! \n";

    const int trainLength = dataLength * 0.8;
    //const int testLength = dataLength - trainLength;
    auto* dataLoader = new DataLoader(data, trainLength);

#if USE_GPU
    network->Learn(20, 0.01, dataLoader, 64, 1);
#else
    const int numThreads = std::thread::hardware_concurrency();
    network->Learn(20, 0.01, dataLoader, 64, numThreads);
#endif

    const double trainingAccuracy = TestAccuracy(network, data, 1000);
    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";


    const double testingAccuracy = TestAccuracy(network, data + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";
}

void QuickDraw2(const int numDrawingsPerCategory)
{
    std::cout << "quickdraw 1\n";
    std::pair<int, int> dataInfo = GetDataLengthAndNumCategories("../datasets/Quickdraw", numDrawingsPerCategory);
    const int dataLength = dataInfo.first;
    const int numCategories = dataInfo.second;

    MAT*** data = GetQuickdrawDataset("../datasets/Quickdraw", dataLength, numCategories, numDrawingsPerCategory,
                                      true);
    std::cout << "loaded" << std::endl;

    auto* network = new Network();
    network->AddLayer(new InputLayer(28, 28, 1));
    network->AddLayer(new ConvLayer(new LayerShape(3, 3, 32), new ReLU()));
    network->AddLayer(new MaxPoolLayer(2, 2));
    network->AddLayer(new Flatten());
    network->AddLayer(new FCL(numCategories, new Softmax()));

    network->Compile(Opti::Adam, new CrossEntropy());

    network->PrintNetwork();
    const int trainLength = dataLength * 0.8;
    //const int testLength = dataLength - trainLength;
    auto* dataLoader = new DataLoader(data, trainLength);
    dataLoader->Shuffle();
#if USE_GPU
    network->Learn(10, 0.01, dataLoader, 96, 1);
#else
    const int numThreads = std::thread::hardware_concurrency();
    network->Learn(10, 0.01, dataLoader, 96, numThreads);
#endif

    double trainingAccuracy = TestAccuracy(network, data, 1000);
    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";


    double testingAccuracy = TestAccuracy(network, data + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";
}

MAT*** GetQuickdrawDataset(const std::string& path, int dataLength, int numCategories,
                           int numDrawingsPerCategory, bool format2D)
{
    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;

#if USE_GPU
    auto*** dataset = new Matrix_GPU** [dataLength];
#else
    auto*** dataset = new Matrix** [dataLength];
#endif

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

            dataset[inputIndex] = new MAT* [2];
            dataset[inputIndex][1] = LabelToMatrix(categoryIndex, numCategories);
#if USE_GPU
            Matrix m(rows, cols);
#else
            dataset[inputIndex][0] = new Matrix(rows, cols);
#endif
            for (int p = 0; p < 28 * 28; p++)
            {
                unsigned char pixVal;
                in.read((char*) &pixVal, 1);
#if USE_GPU
                m[p] = static_cast<double>(pixVal) / 255.0;
#else
                (*dataset[inputIndex][0])[p] = static_cast<double>(pixVal) / 255.0;
#endif
            }
#if USE_GPU
            dataset[inputIndex][0] = new Matrix_GPU(m);
#endif

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

#if USE_GPU

Matrix_GPU* LabelToMatrix(const int label, const int numLabels)
{
    auto* res = new Matrix_GPU(numLabels, 1);
    res->SetAt(label, 1);

    return res;
}

#else

MAT* LabelToMatrix(int label, int numLabels)
{
    auto* res = new MAT(numLabels, 1);
    (*res)[label] = 1;

    return res;
}

#endif