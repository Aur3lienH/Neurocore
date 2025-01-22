cmake_minimum_required(VERSION 3.27)
project(DeepLearningLibrary)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto=auto -I../include")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and include directories
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${PYTHON_INCLUDE_DIRS})

# Add pybind11 as a subdirectory
set(PYBIND11_SOURCE_DIR "../../dependencies/pybind11")
set(PYBIND11_BINARY_DIR "${CMAKE_BINARY_DIR}/pybind11")
add_subdirectory(${PYBIND11_SOURCE_DIR} ${PYBIND11_BINARY_DIR})

# Create the Python module
pybind11_add_module(deep_learning_py network.cpp)
target_link_libraries(deep_learning_py PRIVATE ${PYTHON_LIBRARIES})