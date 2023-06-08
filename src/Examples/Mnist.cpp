#include "Mnist.h"
#include "../Network.h"
#include "../Matrix.h"
#include "../Layer.h"
#include "../InputLayer.h"
#include "../FCL.h"
#include "../LastLayer.h"
#include "Tools.h"
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <sstream>



const std::string MNIST_DATA_PATH = "/home/aure/Projects/DeepLearning/datasets/mnist/mnist_train.csv";


Matrix* LabelToMatrix(int label)
{
    Matrix* matrix = new Matrix(10,1,0.0);
    matrix->operator[](label) = 1;
    return matrix;
}

int MatrixToLabel(const Matrix* matrix)
{
    int label = 0;
    double max = 0;
    for(int i = 0; i < matrix->getRows(); i++)
    {
        if(matrix->operator[](i) > max)
        {
            max = matrix->operator[](i);
            label = i;
        }
    }
    return label;
}


Matrix*** GetDataset(std::string path, int dataLength)
{

    std::ifstream file(path);


    Matrix*** dataset = new Matrix**[2];
    dataset[0] = new Matrix*[dataLength];
    dataset[1] = new Matrix*[dataLength];

    std::string line;
    std::string value;
    if(file.is_open())
    {
        int j = 0;
        while(getline(file, line))
        {

            std::stringstream s(line);
            int i = -1;
            dataset[0][j] = new Matrix(784, 1);
            
            while (getline(s, value, ','))
            {

                if(i == -1)
                {
                    dataset[1][j] = LabelToMatrix(std::stoi(value));
                }
                else
                {
                    dataset[0][j][0][i] = std::stod(value);
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


void Mnist()
{
    int dataLength = Tools::CsvLength(MNIST_DATA_PATH);
    Matrix*** data = GetDataset(MNIST_DATA_PATH, dataLength);
    
    std::cout << "Data length: " << dataLength << std::endl;

    for (int i = 0; i < dataLength; i++)
    {
        data[0][i]->operator*=(1.0/255.0);
    }
    

    Network* network = new Network();
    network->AddLayer(new InputLayer(784));
    network->AddLayer(new FCL(16, new SigmoidPrime()));
    network->AddLayer(new LastLayer(10, new Softmax(), new CrossEntropy()));

    network->Compile();

    network->Learn(10,0.01,data[0], data[1], 1, dataLength, 1);
    Matrix* ok = network->FeedForward(data[0][50]);
    std::cout << "Label: " << MatrixToLabel(data[1][50]) << std::endl;
    std::cout << "Prediction: " << MatrixToLabel(ok) << std::endl;

    std::cout << *ok << std::endl;

    std::cout << "Test accuracy: " << TestAccuracy(network, data, dataLength) << std::endl;
}

double TestAccuracy(Network* network, Matrix*** dataset, int dataLength)
{
    int correct = 0;
    for(int i = 0; i < dataLength; i++)
    {
        Matrix* prediction = network->FeedForward(dataset[0][i]);
        if(MatrixToLabel(prediction) == MatrixToLabel(dataset[1][i]))
        {
            correct++;
        }
    }
    return (double)correct/(double)dataLength;
}

