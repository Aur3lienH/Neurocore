cmake_minimum_required(VERSION 3.27)
project(DeepLearningLibrary)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto=auto -fPIC -I${INCLUDE_DIR}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

# Add pybind11 as a subdirectory
set(PYBIND11_SOURCE_DIR ${PYBIND11_ROOT})
set(PYBIND11_BINARY_DIR ${PYBIND11_ROOT})
add_subdirectory(${PYBIND11_SOURCE_DIR} ${PYBIND11_BINARY_DIR})



#target_include_directories(deep_learning_py 
#    PRIVATE
#        ${Python3_INCLUDE_DIRS}
#        ${INCLUDE_DIR}  
#        ${PYBIND11_ROOT}/include
#)

# Create the Python module
pybind11_add_module(deep_learning_py ${NETWORK_FILE})
target_link_libraries(deep_learning_py PRIVATE ${PYTHON3_LIBRARIES})

set_target_properties(deep_learning_py PROPERTIES
    OUTPUT_NAME "neurocore"  
    PREFIX ""  # Remove 'lib' prefix 
    SUFFIX ".so"
)