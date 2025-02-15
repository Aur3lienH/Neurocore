cmake_minimum_required(VERSION 3.13)

# Set the compilers BEFORE project()
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-12" CACHE STRING "CUDA host compiler")
project(deeplib_tests VERSION 1.0 LANGUAGES CUDA CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Language standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES "75;86")


# Compiler flags
set(CMAKE_C_FLAGS "-fPIC -O0 -ffast-math")
set(CMAKE_CXX_FLAGS "-fPIC -O0 -mavx -march=native")


# Find packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
pkg_check_modules(SDL2 REQUIRED sdl2)
pkg_check_modules(SDL2IMAGE REQUIRED SDL2_image)
pkg_check_modules(CAIRO REQUIRED cairo)

# Collect all source files for tests
file(GLOB_RECURSE TEST_SOURCES
        "${CMAKE_SOURCE_DIR}/src/network/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/tests/ActivationTests.cpp"
        "${CMAKE_SOURCE_DIR}/src/tests/Tests.cpp"
        "${CMAKE_SOURCE_DIR}/src/main.cpp"
        "${CMAKE_SOURCE_DIR}/include/*.h"
        "${CMAKE_SOURCE_DIR}/include/*.hpp"
        "${CMAKE_SOURCE_DIR}/include/*.cuh"
        "${CMAKE_SOURCE_DIR}/src/GPUmain.cpp"
)

# Create the test executable
add_executable(${PROJECT_NAME} ${TEST_SOURCES})

set(CMAKE_VERBOSE_MAKEFILE ON)

set (USE_GPU true)

if (USE_GPU)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_GPU)
    set_target_properties(
            ${PROJECT_NAME}
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
endif()
set_source_files_properties(
        "${CMAKE_SOURCE_DIR}/src/GPUmain.cpp"
        "${CMAKE_SOURCE_DIR}/src/tests/ActivationTests.cpp"
        "${CMAKE_SOURCE_DIR}/src/tests/Tests.cpp"
        "${CMAKE_SOURCE_DIR}/src/main.cpp"
        PROPERTIES LANGUAGE CUDA
)
target_link_libraries(deeplib_tests PRIVATE stdc++) # does not compile if it's not there

find_library(CUBLAS_LIBRARY cublas PATHS /usr/local/cuda-12.2/lib64 /usr/local/cuda/targets/x86_64-linux/lib)
if(CUBLAS_LIBRARY)
    message(STATUS "cuBLAS found: ${CUBLAS_LIBRARY}")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUBLAS_LIBRARY})
else()
    message(FATAL_ERROR "cuBLAS not found!")
endif()

find_library(CUDART_LIBRARY cudart PATHS /usr/local/cuda-12.2/lib64)
if (CUDART_LIBRARY)
    message(STATUS "CUDA Runtime Library found: ${CUDART_LIBRARY}")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDART_LIBRARY})
else()
    message(FATAL_ERROR "CUDA Runtime Library not found!")
endif()

# Specify include directories
target_include_directories(${PROJECT_NAME} PRIVATE
        ${GTK3_INCLUDE_DIRS}
        ${SDL2_INCLUDE_DIRS}
        ${SDL2IMAGE_INCLUDE_DIRS}
        ${CAIRO_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/dependencies/pybind11/include
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

# Enable testing
enable_testing()