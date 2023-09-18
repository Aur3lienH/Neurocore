cmake_minimum_required(VERSION 3.26)
project("h")
set(CMAKE_CXX_STANDARD 17)
set_target_properties(src/main.cu LINKER_LANGUAGE CXX)

add_compile_definitions(USE_GPU)


if (USE_GPU)
    message("Using GPU")
else ()
    message("Using CPU")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -O3 -mavx -mavx512bw")

include_directories(src)
include_directories(src/Examples)
include_directories(src/Tools)

add_executable(DeepLearning
        src/main.cu)


set_target_properties(
        DeepLearning
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(DeepLearning ${CUDA_LIBRARIES} cudnn)
target_link_libraries(DeepLearning ${OpenCV_LIBS})