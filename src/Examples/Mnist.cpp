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



const std::string MNIST_DATA_PATH = "./datasets/mnist/mnist_train.csv";


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


Matrix*** GetDataset(std::string path, int dataLength, bool format2D)
{
    int cols = 0;
    int rows = 0;
    if(format2D)
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


    Matrix*** dataset = new Matrix**[dataLength];

    std::string line;
    std::string value;
    if(file.is_open())
    {
        int j = 0;
        while(getline(file, line))
        {
            dataset[j] = new Matrix*[2];

            std::stringstream s(line);
            int i = -1;
            dataset[j][0] = new Matrix(rows, cols);
            
            while (getline(s, value, ','))
            {

                if(i == -1)
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
        data[i][0]->operator*=(1.0/255.0);
    }

    

    Network* network = new Network();
    network->AddLayer(new InputLayer(784));
    network->AddLayer(new FCL(128, new ReLU()));
    network->AddLayer(new FCL(10, new Softmax()));
    std::cout << "before compiling !\n";
    network->Compile(Opti::Adam,new CrossEntropy());

    int trainLength = dataLength * 0.8;
    int testLength = dataLength - trainLength;
    network->Learn(5,0.01,new DataLoader(data,1,trainLength), 1,1);


    double accuracy = TestAccuracy(network,data[0] + trainLength,data[1] + trainLength, testLength);
    std::cout << "Accurcy : " << accuracy * 100 << "% \n";
}

void Mnist2()
{
    
    int dataLength = CSVTools::CsvLength(MNIST_DATA_PATH);
    Matrix*** data = GetDataset(MNIST_DATA_PATH, dataLength, true);
    
    std::cout << "Data length: " << dataLength << std::endl;

    for (int i = 0; i < dataLength; i++)
    {
        data[0][i]->operator*=(1.0/255.0);
    }
    
    

    Network* network = new Network();
    network->AddLayer(new InputLayer(28,28,1));
    network->AddLayer(new ConvLayer(new LayerShape(5,5,32),new ReLU()));
    network->AddLayer(new MaxPoolLayer(2,2));   
    network->AddLayer(new ConvLayer(new LayerShape(5,5,16),new ReLU()));
    network->AddLayer(new MaxPoolLayer(2,2));
    network->AddLayer(new Flatten());
    network->AddLayer(new FCL(10, new Softmax()));

    network->Compile(Opti::Adam,new CrossEntropy());

    network->PrintNetwork();
    
    int trainLength = dataLength * 0.8;
    int testLength = dataLength - trainLength;

    network->Learn(3,0.1,new DataLoader(data,30,trainLength),1,1);

    network->Save("./Models/MNIST_11.net");


    double trainingAccuracy = TestAccuracy(network,data[0],data[1], 1000);

    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";


    double testingAccuracy = TestAccuracy(network,data[0] + trainLength,data[1] + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";
}



double TestAccuracy(Network* network, Matrix** inputs, Matrix** ouputs, int dataLength)
{
    int correct = 0;
    for(int i = 0; i < dataLength; i++)
    {
        Matrix* prediction = network->Process(inputs[i]);
        if(MatrixToLabel(prediction) == MatrixToLabel(ouputs[i]))
        {
            correct++;
        }
    }
    return (double)correct/(double)dataLength;
}

void LoadAndTest(std::string filename, bool is2D)
{
    Network* network = Network::Load(filename);

    //network->Compile();

    network->PrintNetwork();

    int dataLength = CSVTools::CsvLength(MNIST_DATA_PATH);
    Matrix*** data = GetDataset(MNIST_DATA_PATH, dataLength,is2D);

    int trainLength = dataLength * 0.8;
    int testLength = dataLength - trainLength;

    
    double trainingAccuracy = TestAccuracy(network,data[0],data[1], 1000);
    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";


    double testingAccuracy = TestAccuracy(network,data[0] + trainLength,data[1] + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";
}

