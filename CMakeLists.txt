cmake_minimum_required(VERSION 3.13)

# Set the compilers BEFORE project()
set(CMAKE_CXX_COMPILER "g++")
project(deeplib_tests VERSION 1.0 LANGUAGES CUDA CXX)

# Language standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES "75;86")

# Compiler flags
set(CMAKE_C_FLAGS "-fPIC -O3 -ffast-math")
set(CMAKE_CXX_FLAGS "-fPIC -O3 -mavx -march=native")

# Find packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
pkg_check_modules(SDL2 REQUIRED sdl2)
pkg_check_modules(SDL2IMAGE REQUIRED SDL2_image)
pkg_check_modules(CAIRO REQUIRED cairo)

# Collect all source files for tests
file(GLOB_RECURSE TEST_SOURCES
        "${CMAKE_SOURCE_DIR}/src/tests/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/network/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/tests/*.cu"
        "${CMAKE_SOURCE_DIR}/src/main.cpp"
)

# Create the test executable
add_executable(${PROJECT_NAME} ${TEST_SOURCES})

# Specify include directories
target_include_directories(${PROJECT_NAME} PRIVATE
        ${GTK3_INCLUDE_DIRS}
        ${SDL2_INCLUDE_DIRS}
        ${SDL2IMAGE_INCLUDE_DIRS}
        ${CAIRO_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/dependencies/pybind11/include
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${GTK3_LIBRARIES}
        ${SDL2_LIBRARIES}
        ${SDL2IMAGE_LIBRARIES}
        ${CAIRO_LIBRARIES}
        m
        SDL2
)

# Enable testing
enable_testing()