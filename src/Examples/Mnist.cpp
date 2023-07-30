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
#include "../DropoutFCL.h"
#include "Tools.h"
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <sstream>
#include <thread>


const std::string MNIST_DATA_PATH = "./datasets/mnist/mnist_train.csv";

const std::string MNIST_FASHION_DATA_PATH = "./datasets/mnist_fashion/train-images-idx3-ubyte";
const std::string MNIST_FASHIOIN_LABEL_PATH = "./datasets/mnist_fashion/train-labels-idx1-ubyte";


Matrix* LabelToMatrix(int label)
{
    auto* matrix = new Matrix(10, 1, 0.0);
    matrix->operator[](label) = 1;
    return matrix;
}

int MatrixToLabel(const Matrix* matrix)
{
    int label = 0;
    double max = 0;
    for (int i = 0; i < matrix->getRows(); i++)
    {
        if (matrix->operator[](i) > max)
        {
            max = matrix->operator[](i);
            label = i;
        }
    }
    return label;
}


Matrix*** GetDataset(std::string path, int dataLength, bool format2D)
{
    int cols = 0;
    int rows = 0;
    if (format2D)
    {
        cols = 28;
        rows = 28;
    }
    else
    {
        cols = 1;
        rows = 784;
    }

    std::ifstream file(path);


    Matrix*** dataset = new Matrix** [dataLength];

    std::string line;
    std::string value;
    if (file.is_open())
    {
        int j = 0;
        while (getline(file, line))
        {
            dataset[j] = new Matrix* [2];

            std::stringstream s(line);
            int i = -1;
            dataset[j][0] = new Matrix(rows, cols);

            while (getline(s, value, ','))
            {

                if (i == -1)
                {
                    dataset[j][1] = LabelToMatrix(std::stoi(value));
                }
                else
                {
                    dataset[j][0][0][i] = std::stod(value);
                }
                i++;
            }
            j++;
        }
    }
    else
    {
        std::cout << "File not found" << std::endl;
    }
    std::cout << "Dataset loaded" << std::endl;
    return dataset;

}


void Mnist1()
{
    std::cout << "mnist 1\n";
    int dataLength = CSVTools::CsvLength(MNIST_DATA_PATH);
    Matrix*** data = GetDataset(MNIST_DATA_PATH, dataLength, false);

    std::cout << "Data length: " << dataLength << std::endl;

    for (int i = 0; i < dataLength; i++)
    {
        data[i][0]->operator*=(1.0 / 255.0);
    }


    Network* network = new Network();
    network->AddLayer(new InputLayer(784));
    network->AddLayer(new DropoutFCL(128, new ReLU(), 0.5));
    network->AddLayer(new FCL(10, new Softmax()));
    std::cout << "before compiling !\n";
    network->Compile(Opti::Adam, new CrossEntropy());
    std::cout << "compiled ! \n";
    int trainLength = dataLength * 0.8;
    int testLength = dataLength - trainLength;
    network->Learn(20, 0.01, new DataLoader(data, trainLength), 64, 16);

    double trainingAccuracy = TestAccuracy(network, data, 1000);
    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";


    double testingAccuracy = TestAccuracy(network, data + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";
}

void Mnist2()
{

    int dataLength = CSVTools::CsvLength(MNIST_DATA_PATH);
    Matrix*** data = GetDataset(MNIST_DATA_PATH, dataLength, true);

    std::cout << "Data length: " << dataLength << std::endl;

    for (int i = 0; i < dataLength; i++)
    {
        data[i][0]->operator*=(1.0 / 255.0);
    }


    Network* network = new Network();
    network->AddLayer(new InputLayer(28, 28, 1));
    network->AddLayer(new ConvLayer(new LayerShape(3, 3, 32), new ReLU()));
    network->AddLayer(new MaxPoolLayer(2, 2));
    network->AddLayer(new Flatten());
    network->AddLayer(new FCL(10, new Softmax()));

    network->Compile(Opti::Adam, new CrossEntropy());

    network->PrintNetwork();

    int trainLength = dataLength * 0.8;
    int testLength = dataLength - trainLength;

    const int numThreads = static_cast<int>(std::thread::hardware_concurrency());
    network->Learn(3, 0.1, new DataLoader(data, trainLength), 96, numThreads);

    network->Save("./Models/MNIST_11.net");


    double trainingAccuracy = TestAccuracy(network, data, 1000);

    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";


    double testingAccuracy = TestAccuracy(network, data + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";
}


void FashionMnist1()
{
    int dataLength;
    Matrix*** data = GetFashionDataset(MNIST_FASHION_DATA_PATH, MNIST_FASHIOIN_LABEL_PATH, dataLength, false);

    std::cout << "Data length: " << dataLength << std::endl;


    for (int i = 0; i < dataLength; i++)
    {
        data[i][0]->operator*=(1.0 / 255.0);
    }


    Network* network = new Network();
    network->AddLayer(new InputLayer(784));
    network->AddLayer(new FCL(512, new Tanh()));
    network->AddLayer(new FCL(10, new Softmax()));

    network->Compile(Opti::Adam, new CrossEntropy());

    network->PrintNetwork();

    int trainLength = dataLength * 0.8;
    int testLength = dataLength - trainLength;

    network->Learn(10, 0.1, new DataLoader(data, trainLength), 64, 4);

    network->Save("./Models/MNIST_11.net");


    double trainingAccuracy = TestAccuracy(network, data, 1000);

    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";


    double testingAccuracy = TestAccuracy(network, data + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";
}


void FashionMnist2()
{
    int dataLength;
    Matrix*** data = GetFashionDataset(MNIST_FASHION_DATA_PATH, MNIST_FASHIOIN_LABEL_PATH, dataLength, true);

    std::cout << "Data length: " << dataLength << std::endl;


    for (int i = 0; i < dataLength; i++)
    {
        data[i][0]->operator*=(1.0 / 255.0);
    }


    Network* network = new Network();
    network->AddLayer(new InputLayer(28, 28, 1));
    network->AddLayer(new ConvLayer(new LayerShape(3, 3, 32), new ReLU()));
    network->AddLayer(new MaxPoolLayer(2, 2));
    network->AddLayer(new ConvLayer(new LayerShape(2, 2, 32), new ReLU()));
    network->AddLayer(new MaxPoolLayer(2, 2));
    network->AddLayer(new Flatten());
    network->AddLayer(new FCL(128, new ReLU()));
    network->AddLayer(new FCL(10, new Softmax()));

    network->Compile(Opti::Adam, new CrossEntropy());

    network->PrintNetwork();

    int trainLength = dataLength * 0.8;
    int testLength = dataLength - trainLength;

    network->Learn(5, 0.1, new DataLoader(data, trainLength), 64, 2);
    network->Save("./Models/MNIST_11.net");


    double trainingAccuracy = TestAccuracy(network, data, 1000);

    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";


    double testingAccuracy = TestAccuracy(network, data + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";
}


double TestAccuracy(Network* network, Matrix*** data, int dataLength)
{
    int correct = 0;
    for (int i = 0; i < dataLength; i++)
    {
        Matrix* prediction = network->Process(data[i][0]);
        if (MatrixToLabel(prediction) == MatrixToLabel(data[i][1]))
        {
            correct++;
        }
    }
    return (double) correct / (double) dataLength;
}

void LoadAndTest(std::string filename, bool is2D)
{
    Network* network = Network::Load(filename);

    //network->Compile();

    network->PrintNetwork();

    int dataLength = CSVTools::CsvLength(MNIST_DATA_PATH);
    Matrix*** data = GetDataset(MNIST_DATA_PATH, dataLength, is2D);

    int trainLength = dataLength * 0.8;
    int testLength = dataLength - trainLength;


    double trainingAccuracy = TestAccuracy(network, data, 1000);
    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";


    double testingAccuracy = TestAccuracy(network, data + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";
}

Matrix*** GetFashionDataset(std::string data, std::string label, int& dataLength, bool format2D)
{
    int labelLength;
    int cols = 0;
    int rows = 0;
    if (format2D)
    {
        cols = 28;
        rows = 28;
    }
    else
    {
        cols = 1;
        rows = 784;
    }
    std::ifstream dataFile(data);
    std::ifstream labelFile(label);

    if (!dataFile.is_open() || !labelFile.is_open())
    {
        throw std::runtime_error("File not found");
        return nullptr;
    }


    int magicNumber = 0;
    dataFile.read((char*) &magicNumber, sizeof(int));
    if (ReverseInt(magicNumber) != 2051)
    {
        throw std::runtime_error("Invalid magic number");
        return nullptr;
    }


    labelFile.read((char*) &magicNumber, sizeof(int));
    if (ReverseInt(magicNumber) != 2049)
    {
        throw std::runtime_error("Invalid magic number");
        return nullptr;
    }


    dataFile.read((char*) &dataLength, sizeof(int));
    labelFile.read((char*) &labelLength, sizeof(int));
    int n_rows;
    int n_cols;

    dataLength = ReverseInt(dataLength);
    dataFile.read((char*) &n_rows, sizeof(n_rows));
    n_rows = ReverseInt(n_rows);
    dataFile.read((char*) &n_cols, sizeof(n_cols));
    n_cols = ReverseInt(n_cols);

    Matrix*** dataset = new Matrix** [dataLength];

    for (int i = 0; i < dataLength; i++)
    {
        dataset[i] = new Matrix* [2];

        unsigned char label;

        labelFile.read((char*) &label, sizeof(label));
        dataset[i][1] = LabelToMatrix(label);
        dataset[i][0] = new Matrix(rows, cols);
        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < cols; k++)
            {
                unsigned char value;
                dataFile.read((char*) &value, sizeof(value));
                (*dataset[i][0])(j, k) = (double) value;
            }
        }
    }

    return dataset;

}

int ReverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

