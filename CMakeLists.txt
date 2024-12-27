cmake_minimum_required(VERSION 3.13)

# Set the compilers BEFORE project()
set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-12" CACHE STRING "CUDA host compiler")
project(deeplib VERSION 1.0 LANGUAGES CUDA CXX)

# Language standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES "75;86")

# Compiler flags
set(CMAKE_C_FLAGS "-fPIC -O3 -ffast-math")
set(CMAKE_CXX_FLAGS "-fPIC -O3 -mavx -march=native")



# Find Python and setup pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
add_subdirectory(dependencies/pybind11)

# Find packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
pkg_check_modules(SDL2 REQUIRED sdl2)
pkg_check_modules(SDL2IMAGE REQUIRED SDL2_image)
pkg_check_modules(CAIRO REQUIRED cairo)

# Collect all source files for the main executable
file(GLOB_RECURSE SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.c"
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
)

# Create the main library target
add_library(${PROJECT_NAME}_core SHARED ${SOURCES})

# Specify include directories for the core target
target_include_directories(${PROJECT_NAME}_core PRIVATE
    ${GTK3_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
    ${SDL2IMAGE_INCLUDE_DIRS}
    ${CAIRO_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/dependencies/pybind11/include  # Add pybind11 headers
    ${Python3_INCLUDE_DIRS}
)

# Link libraries to the main library
target_link_libraries(${PROJECT_NAME}_core
    ${GTK3_LIBRARIES}
    ${SDL2_LIBRARIES}
    ${SDL2IMAGE_LIBRARIES}
    ${CAIRO_LIBRARIES}
    m
    SDL2
)

# Define version info for the Python module
set(DEEP_VERSION_INFO "0.0.1")

# Define the Python module target using pybind11
pybind11_add_module(${PROJECT_NAME} src/pythonWrapper/Network.cpp)

# Add include directories for the Python module
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/dependencies/pybind11/include  # Add pybind11 headers
    ${Python3_INCLUDE_DIRS}
)

# Link the core library and Python to the Python module
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PROJECT_NAME}_core
    Python3::Python
)
target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${DEEP_VERSION_INFO})
set_target_properties(${PROJECT_NAME}_core PROPERTIES
    OUTPUT_NAME "deep"  # This will make it output as libdeep.so
)
