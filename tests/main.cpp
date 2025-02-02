#include <iostream>
#include "tests/Tests.h"
int main(int argc, char** argv)
{
    Tests::ExecuteTests();
    //Mnist2();
    //QuickDraw2(10000);

#if USE_GPU
    delete Matrix_GPU::cuda;
#endif
    return 0;
    //LoadAndTest("./Models/MNIST_11.net",true);
}

/*
#include "GUI.cuh"

int main()
{
    return GUI();
}*/

/*
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "Matrix.cuh"
#include <SFML/Graphics.hpp>
#include <vector>
#include "Network.cuh"
#include "InputLayer.cuh"
#include "FCL.cuh"
#include "Tools.cuh"
#include "Mnist.cuh"
#include "Quickdraw.cuh"
#include "MaxPooling.cuh"
#include "ConvLayer.cuh"
#include "Flatten.cuh"
#include <thread>
#include "CUDA.cuh"

using namespace cv;

//Todo: Learn from https://data-flair.training/blogs/opencv-sudoku-solver/
int main(int argc, char** argv)
{
    Mat image;
    image = imread("../Sudoku/sudoku.jpg", IMREAD_COLOR);
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }

    Mat proc = image.clone();
    Mat gray = image.clone();

    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, proc, Size(9, 9), 0);
    adaptiveThreshold(proc, proc, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    bitwise_not(proc, proc);
    Mat kernel = (Mat_<uchar>(3, 3) << 0, 1, 0,
            1, 1, 1,
            0, 1, 0);
    dilate(proc, proc, kernel);

    std::vector<std::vector<Point>> contours;
    findContours(proc, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    drawContours(proc, contours, -1, Scalar(0, 255, 0), 3);
    std::sort(contours.begin(), contours.end(), [](const std::vector<Point>& a, const std::vector<Point>& b)
    {
        return contourArea(a) > contourArea(b);
    });
    Point topLeft, topRight, bottomLeft, bottomRight;
    bottomRight = std::max_element(contours[0].begin(), contours[0].end(), [](const Point& a, const Point& b)
    {
        return a.x + a.y < b.x + b.y;
    })[0];
    topLeft = std::min_element(contours[0].begin(), contours[0].end(), [](const Point& a, const Point& b)
    {
        return a.x + a.y < b.x + b.y;
    })[0];
    bottomLeft = std::min_element(contours[0].begin(), contours[0].end(), [](const Point& a, const Point& b)
    {
        return a.x - a.y < b.x - b.y;
    })[0];
    topRight = std::max_element(contours[0].begin(), contours[0].end(), [](const Point& a, const Point& b)
    {
        return a.x - a.y < b.x - b.y;
    })[0];
    std::vector<Point> cont{topLeft, topRight, bottomRight, bottomLeft};
    contours.clear();
    contours.emplace_back(cont);
    drawContours(image, contours, 0, Scalar(0, 0, 255), 3);
    imwrite("../Sudoku/proc.jpg", image);

    */
/*
     top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
cont = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
side = max([  distance_between(bottom_right, top_right),
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),
            distance_between(top_left, top_right) ])
     * *//*


    const float side = static_cast<float>(std::max({norm(bottomRight - topRight), norm(topLeft - bottomLeft),
                                                    norm(bottomRight - bottomLeft), norm(topLeft - topRight)}));
    const Point2f dst[4] = {Point2f(0, 0), Point2f(side - 1, 0), Point2f(side - 1, side - 1), Point2f(0, side - 1)};
    const Point2f src[4] = {topLeft, topRight, bottomRight, bottomLeft};
    const Mat transform = getPerspectiveTransform(src, dst);
    Mat warp;
    warpPerspective(gray, warp, transform, Size(side, side));
    imwrite("../Sudoku/warp.jpg", warp);

    */
/*squares = []
side = img.shape[:1]
side = side[0] / 9
for j in range(9):
    for i in range(9):
        p1 = (i * side, j * side)  #Top left corner of a box
        p2 = ((i + 1) * side, (j + 1) * side)  #Bottom right corner
        squares.append((p1, p2)) return squares*//*

    std::vector<std::vector<Point>> squares;
    const float side2 = warp.rows / 9;
    const int margin = static_cast<int>(side2 * .25f);
    for (int j = 0; j < 9; j++)
    {
        for (int i = 0; i < 9; ++i)
        {
            const Point p1(i * side2 + margin, j * side2 + margin);
            const Point p2((i + 1) * side2 - margin, (j + 1) * side2 - margin);
            squares.emplace_back(std::vector<Point>{p1, p2});
        }
    }

    */
/*
     * digits = []
img = pre_process_image(img.copy(), skip_dilate=True)
for square in squares:
    digits.append(extract_digit(img, square, size))*//*

    std::vector<Mat> digits;
    Mat img = warp.clone();
    for (const auto& square : squares)
    {
        Mat digit = img(Rect(square[0], square[1]));
        Mat scaled = Mat::zeros(28, 28, CV_8UC1);
        resize(digit, scaled, Size(28, 28));
        digits.emplace_back(scaled);
    }

    Matrix_GPU** input = new Matrix_GPU* [81];
    bool emptyCells[81];
    float max;
    for (int i = 0; i < 81; i++)
    {
        max = 0;
        input[i] = new Matrix_GPU(784, 1);
        float* data = new float[784];
        for (int j = 0; j < 784; j++)
        {
            data[j] = 1 - digits[i].at<uchar>(j / 28, j % 28) /
                          255.0f; // 1- because the background is white and the number is black
            // Cleaning the image
            if (data[j] < 0.6f)
                data[j] = 0;
            else if (max < data[j])
                max = data[j];
        }
        emptyCells[i] = max < 0.8f;
        checkCUDA(cudaMemcpy(input[i]->GetData(), data, 784 * sizeof(float), cudaMemcpyHostToDevice));
        delete[] data;
    }
    const int dataLength = CSVTools::CsvLength("../datasets/mnist/mnist_train.csv");

    MAT*** data = GetDataset("../datasets/mnist/mnist_train.csv", dataLength, false);

    std::cout << "Data length: " << dataLength << std::endl;

    const float scale = 1.0f / 255.0f;
    for (int i = 0; i < dataLength; i++)
        *data[i][0] *= scale;

    Network* network = new Network();
    network->AddLayer(new InputLayer(784));
    network->AddLayer(new FCL(512, new ReLU()));
    network->AddLayer(new FCL(10, new Softmax()));
    std::cout << "before compiling !\n";
    network->Compile(Opti::Adam, new CrossEntropy());
    std::cout << "compiled ! \n";
    const int trainLength = dataLength * .4f;

#if USE_GPU
    network->Learn(1, 0.01, new DataLoader(data, trainLength), 128, 1);
#else
    const int numThreads = std::thread::hardware_concurrency();
    network->Learn(1, 0.01, new DataLoader(data, trainLength), 128, numThreads);
#endif

    double trainingAccuracy = TestAccuracy(network, data, 1000);
    std::cout << "Training Accuracy : " << trainingAccuracy * 100 << "% \n";
    double testingAccuracy = TestAccuracy(network, data + trainLength, 1000);
    std::cout << "Testing Accuracy : " << testingAccuracy * 100 << "% \n";

    Matrix_GPU output(9, 9);
    Matrix board(9, 9);
    for (int i = 0; i < 81; i++)
    {
        if (emptyCells[i])
        {
            board[i] = 0;
            continue;
        }
        output = *network->FeedForward(input[i]);
        const int label = MatrixToLabel(&output);
        board[i] = static_cast<float>(label);

        //input[9]->Reshape(28, 28, 1);
        //*input[9] *= 255;
        //std::cout << *input[i] << std::endl;
        //std::cout << digits[9] << std::endl;
        //std::cout << output << std::endl;
        //std::cout << label << std::endl;
        float* data = input[i]->GetData_CPU();
        Mat mat(28, 28, CV_8UC1);
        for (int j = 0; j < 784; j++)
            mat.at<uchar>(j / 28, j % 28) = static_cast<uchar>(255 * data[j]);
        imwrite("../Sudoku/digits/" + std::to_string(i) + ".jpg", mat);
    }

    std::cout << board << std::endl;

    */
/*Mat gray = image.clone();
    cvtColor(image, gray, COLOR_BGR2GRAY);
    //namedWindow("Display Image", WINDOW_AUTOSIZE);
    //imshow("Display Image", image);
    //waitKey(0);
    imwrite("../Sudoku/gray.jpg", gray);
    Mat edges = image.clone();
    Canny(gray, edges, 90, 150, 3);
    Mat kernel = Mat::ones(3, 3, CV_8UC1);
    dilate(edges, edges, kernel, Point(-1, -1), 1);
    Mat kernel2 = Mat::ones(5, 5, CV_8UC1);
    erode(edges, edges, kernel2, Point(-1, -1), 1);
    imwrite("../Sudoku/canny.jpg", edges);

    std::vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(edges, lines, 1, CV_PI / 180, 150); // runs the actual detection

    if (lines.empty())
    {
        printf("No lines detected\n");
        return -1;
    }

    const float rho_threshold = 15;
    const float theta_threshold = 0.1;

    std::vector<int>* similar_lines[lines.size()];
    for (int i = 0; i < lines.size(); i++)
        similar_lines[i] = new std::vector<int>();

    for (int i = 0; i < lines.size(); i++)
    {
        for (int j = 0; j < lines.size(); j++)
        {
            if (i == j)
                continue;

            float rho_i, theta_i, rho_j, theta_j;
            rho_i = lines[i][0];
            theta_i = lines[i][1];
            rho_j = lines[j][0];
            theta_j = lines[j][1];
            if (abs(rho_i - rho_j) < rho_threshold && abs(theta_i - theta_j) < theta_threshold)
                similar_lines[i]->push_back(j);
        }
    }


    int* indices = new int[lines.size()];
    for (int i = 0; i < lines.size(); i++)
        indices[i] = i;
    std::sort(indices, indices + lines.size(), [&similar_lines](const int& lhs, const int& rhs)
    {
        return similar_lines[lhs]->size() > similar_lines[rhs]->size();
    });

    auto line_flags = new bool[lines.size()];
    for (int i = 0; i < lines.size(); i++)
        line_flags[i] = true;

    for (int i = 0; i < lines.size() - 1; i++)
    {
        if (!line_flags[indices[i]])
            continue;

        for (int j = i + 1; j < lines.size(); j++)
        {
            if (!line_flags[indices[j]])
                continue;

            float rho_i, theta_i, rho_j, theta_j;
            rho_i = lines[indices[i]][0];
            theta_i = lines[indices[i]][1];
            rho_j = lines[indices[j]][0];
            theta_j = lines[indices[j]][1];
            if (abs(rho_i - rho_j) < rho_threshold && abs(theta_i - theta_j) < theta_threshold)
                line_flags[indices[j]] = false;
        }
    }

    std::cout << "number of Hough lines:" << lines.size() << std::endl;

    std::vector<Vec2f> filtered_lines{};

    for (int i = 0; i < lines.size(); i++)
        if (line_flags[i])
            filtered_lines.push_back(lines[i]);

    std::cout << "number of filtered lines:" << filtered_lines.size() << std::endl;

    for (int i = 0; i < lines.size(); i++)
        delete similar_lines[i];

    for (const auto& filteredLine : filtered_lines)
    {
        const float rho = filteredLine[0], theta = filteredLine[1];
        const float a = cos(theta), b = sin(theta);
        const float x0 = a * rho, y0 = b * rho;
        const int x1 = cvRound(x0 + 1000 * (-b)), y1 = cvRound(y0 + 1000 * (a));
        const int x2 = cvRound(x0 - 1000 * (-b)), y2 = cvRound(y0 - 1000 * (a));
        line(image, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255), 2);
    }

    imwrite("../Sudoku/hough.jpg", image);

    return 0;*//*

}*/
