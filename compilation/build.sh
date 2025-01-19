#!/bin/bash

cmake -S . -B build -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-12
cmake --build build
