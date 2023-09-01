cmake_minimum_required(VERSION 3.26)
project(DeepLearning CXX CUDA)

set(CMAKE_CXX_STANDARD 17)


#add_compile_options(-std=c++17 -pthread -O3 -mavx)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthreadnimportequoi -O3 -mavx")

include_directories(src)
include_directories(src/Examples)
include_directories(src/Tools)

add_executable(DeepLearning
        src/Examples/Mnist.cu
        src/Examples/Mnist.cuh
        src/Examples/Quickdraw.cu
        src/Examples/Quickdraw.cuh
        src/Examples/Tests.cu
        src/Examples/Tests.cuh
        src/Examples/Tools.cu
        src/Examples/Tools.cuh
        src/Tools/Bytes.cu
        src/Tools/Bytes.cuh
        src/Tools/ManagerIO.cu
        src/Tools/ManagerIO.cuh
        src/Tools/ProgressBar.cu
        src/Tools/ProgressBar.cuh
        src/Tools/Seiralizer.cu
        src/Tools/Serializer.cuh
        src/Tools/Unit.cu
        src/Tools/Unit.cuh
        src/Activation.cu
        src/Activation.cuh
        src/AveragePooling.cu
        src/AveragePooling.cuh
        src/ConvLayer.cu
        src/ConvLayer.cuh
        src/DataLoader.cu
        src/DataLoader.cuh
        src/DropoutFCL.cu
        src/DropoutFCL.cuh
        src/FCL.cu
        src/FCL.cuh
        src/Flatten.cu
        src/Flatten.cuh
        src/ImagePreProcessing.cu
        src/ImagePreProcessing.cuh
        src/InitFunc.cu
        src/InitFunc.cuh
        src/InputLayer.cu
        src/InputLayer.cuh
        src/Layer.cu
        src/Layer.cuh
        src/LayerShape.cu
        src/LayerShape.cuh
        src/Loss.cu
        src/Loss.cuh
        src/main.cu
        src/Matrix.cu
        src/Matrix.cuh
        src/MaxPooling.cu
        src/MaxPooling.cuh
        src/Network.cu
        src/Network.cuh
        src/Optimizers.cu
        src/Optimizers.cuh
        src/PoolingLayer.cu
        src/PoolingLayer.cuh
        src/ThreadArg.cu
        src/ThreadArg.cuh
        src/vector.cuh
        src/CUDA.cuh)

set_target_properties(
        DeepLearning
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(DeepLearning ${CUDA_LIBRARIES} cudnn)
target_link_libraries(DeepLearning -lcublas)