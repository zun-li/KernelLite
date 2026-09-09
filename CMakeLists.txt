cmake_minimum_required(VERSION 3.18)

project(cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")

find_package(CUDAToolkit REQUIRED)

add_subdirectory(sgemm)
add_subdirectory(histogram)
