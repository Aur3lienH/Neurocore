cmake_minimum_required(VERSION 3.16)

# Set the compilers BEFORE project()
#set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-12" CACHE STRING "CUDA host compiler")
project(deeplib_tests VERSION 1.0 LANGUAGES CUDA CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Language standards

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_PARALLEL_LEVEL 0)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 0")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compile-threads 0")
set(CMAKE_CUDA_ARCHITECTURES "86")
#set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)

# Compiler flags
set(CMAKE_C_FLAGS "-fPIC -O1 -ffast-math -Wno-deprecated-gpu-targets")
set(CMAKE_CXX_FLAGS "-fPIC -O1 -mavx -march=native ")
#set(CMAKE_C_FLAGS "-fPIC -O1 -Wno-deprecated-gpu-targets")
#set(CMAKE_CXX_FLAGS "-fPIC -O1")


# Find packages
#find_package(PkgConfig REQUIRED)
#pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
#pkg_check_modules(SDL2 REQUIRED sdl2)
#pkg_check_modules(SDL2IMAGE REQUIRED SDL2_image)
#pkg_check_modules(CAIRO REQUIRED cairo)
message("${CMAKE_SOURCE_DIR}/tests/template_implementations/*.cpp")
# Collect all source files for tests
file(GLOB_RECURSE TEST_SOURCES
        "${CMAKE_SOURCE_DIR}/tests/*.cpp"
        "${CMAKE_SOURCE_DIR}/template_implementations/*.cu"
        "${CMAKE_SOURCE_DIR}/network/*.cpp"
        #"${CMAKE_SOURCE_DIR}/*.cu"
        "${CMAKE_SOURCE_DIR}/main.cpp"
        "${CMAKE_SOURCE_DIR}/matrix/*.cu"
)
# Create the test executable
add_executable(${PROJECT_NAME} ${TEST_SOURCES})

file(GLOB_RECURSE HEADERS
        "${CMAKE_SOURCE_DIR}/tests/*.h"
        "${CMAKE_SOURCE_DIR}/template_implementations/*.h"
        "${CMAKE_SOURCE_DIR}/network/*.h"
        "${CMAKE_SOURCE_DIR}/matrix/*.h"

        # Because this project is a dirty shit with different extensions
        "${CMAKE_SOURCE_DIR}/tests/*.cuh"
        "${CMAKE_SOURCE_DIR}/template_implementations/*.cuh"
        "${CMAKE_SOURCE_DIR}/network/*.cuh"
        "${CMAKE_SOURCE_DIR}/matrix/*.cuh"
)

target_precompile_headers(${PROJECT_NAME} PRIVATE
    # Standard library headers
    <iostream>
    <vector>
    <memory>
    <string>
    <unordered_map>
    <algorithm>
    
    # CUDA headers
    <cuda_runtime.h>
    <device_launch_parameters.h>
    <cublas_v2.h>
    
    # Your commonly used headers (adjust paths as needed)
    #"gpuComputation/CUDA.cuh"
    #"gpuComputation/CUDALink.hxx"
    ${HEADERS}
)

set(CMAKE_VERBOSE_MAKEFILE ON)

set (USE_GPU true)

if (USE_GPU)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_GPU)
endif()
target_compile_definitions(${PROJECT_NAME} PRIVATE TEST)
set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


# set_source_files_properties(
#         "${CMAKE_SOURCE_DIR}/tests/ActivationTests.cpp"
#         "${CMAKE_SOURCE_DIR}/tests/LayerTests.cpp"
#         "${CMAKE_SOURCE_DIR}/tests/NetworkTests.cpp"
#         "${CMAKE_SOURCE_DIR}/tests/LossTests.cpp"
#         "${CMAKE_SOURCE_DIR}/tests/MatrixTests.cpp"
#         "${CMAKE_SOURCE_DIR}/tests/Tests.cpp"
#         "${CMAKE_SOURCE_DIR}/main.cpp"
#         PROPERTIES LANGUAGE CUDA
# )
target_link_libraries(deeplib_tests PRIVATE stdc++) # does not compile if it's not there

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas)
#find_library(CUBLAS_LIBRARY cublas PATHS /opt/cuda/lib64)
#if(CUBLAS_LIBRARY)
#    message(STATUS "cuBLAS found: ${CUBLAS_LIBRARY}")
#    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUBLAS_LIBRARY})
#else()
#    message(FATAL_ERROR "cuBLAS not found!")
#endif()

#find_library(CUDART_LIBRARY cudart PATHS /opt/cuda/lib64)
#if (CUDART_LIBRARY)
#    message(STATUS "CUDA Runtime Library found: ${CUDART_LIBRARY}")
#    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDART_LIBRARY})
#else()
#    message(FATAL_ERROR "CUDA Runtime Library not found!")
#endif()



# Specify include directories
target_include_directories(${PROJECT_NAME} PRIVATE
        ${GTK3_INCLUDE_DIRS}
        ${SDL2_INCLUDE_DIRS}
        ${SDL2IMAGE_INCLUDE_DIRS}
        ${CAIRO_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../Neurocore/include
        ${CMAKE_SOURCE_DIR}/../Neurocore/dependencies/pybind11/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/dependencies/pybind11/include
        ${CMAKE_SOURCE_DIR}/../Neurocore/src
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${Python3_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${GTK3_LIBRARIES}
        ${SDL2_LIBRARIES}
        ${SDL2IMAGE_LIBRARIES}
        ${CAIRO_LIBRARIES}
        ${Python3_LIBRARIES}
        m
        SDL2
        cudnn
)
add_compile_definitions(SAFE=1)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CUDA)


# Enable testing
enable_testing()