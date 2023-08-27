#include "Examples/Mnist.cuh"

//Todo:Au tout début de FCL::UpdateWeights on a
// Diff at 401405: 0.0123963!=0
// Diff at 401406: 0.0200166!=0
// Diff at 401407: -0.0121016!=0
// Diff at *index*: *CPU*!=*GPU* | Matrix = Weights
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