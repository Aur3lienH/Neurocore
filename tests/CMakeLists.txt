cmake_minimum_required(VERSION 3.13)



# Set the compilers BEFORE project()
set(CMAKE_CXX_COMPILER "g++")
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
set(CMAKE_CXX_FLAGS "-fPIC -O0 -mavx -march=native -fsanitize=address -g -fno-omit-frame-pointer -rdynamic")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -g -fno-omit-frame-pointer -rdynamic")

# Find packages
#find_package(PkgConfig REQUIRED)
#pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
#pkg_check_modules(SDL2 REQUIRED sdl2)
#pkg_check_modules(SDL2IMAGE REQUIRED SDL2_image)
#pkg_check_modules(CAIRO REQUIRED cairo)

# Collect all source files for tests
file(GLOB_RECURSE TEST_SOURCES
        "${CMAKE_SOURCE_DIR}/tests/*.cpp"
        "${CMAKE_SOURCE_DIR}/network/*.cpp"
        #"${CMAKE_SOURCE_DIR}/*.cu"
        "${CMAKE_SOURCE_DIR}/main.cpp"
        "${CMAKE_SOURCE_DIR}/../Neurocore/include/*.h"
        "${CMAKE_SOURCE_DIR}/../Neurocore/include/*.hpp"
        "${CMAKE_SOURCE_DIR}/../Neurocore/include/*.cuh"
)
# Create the test executable
add_executable(${PROJECT_NAME} ${TEST_SOURCES})


# Specify include directories
target_include_directories(${PROJECT_NAME} PRIVATE
        ${GTK3_INCLUDE_DIRS}
        ${SDL2_INCLUDE_DIRS}
        ${SDL2IMAGE_INCLUDE_DIRS}
        ${CAIRO_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../Neurocore/include
        ${CMAKE_SOURCE_DIR}/../Neurocore/dependencies/pybind11/include
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
)
add_compile_definitions(SAFE=1)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CUDA)


# Enable testing
enable_testing()