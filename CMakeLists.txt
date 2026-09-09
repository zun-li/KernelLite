cmake_minimum_required(VERSION 3.18)

project(cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")

find_package(CUDAToolkit REQUIRED)

add_executable(main main.cu)

add_executable(sgemm0 sgemm/sgemm0.cu)
add_executable(sgemm1 sgemm/sgemm1.cu)
add_executable(sgemm2 sgemm/sgemm2.cu)
add_executable(sgemm3 sgemm/sgemm3.cu)
add_executable(sgemm4 sgemm/sgemm4.cu)

target_link_libraries(sgemm0 PRIVATE CUDA::cublas)
target_link_libraries(sgemm1 PRIVATE CUDA::cublas)
target_link_libraries(sgemm2 PRIVATE CUDA::cublas)
target_link_libraries(sgemm3 PRIVATE CUDA::cublas)
target_link_libraries(sgemm4 PRIVATE CUDA::cublas)
