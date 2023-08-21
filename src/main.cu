#include "Examples/Mnist.cuh"

int main()
{
    Mnist1();
    //QuickDraw2(10000);

#if USE_GPU
    delete Matrix_GPU::cuda;
#endif
    return 0;
    //LoadAndTest("./Models/MNIST_11.net",true);
}