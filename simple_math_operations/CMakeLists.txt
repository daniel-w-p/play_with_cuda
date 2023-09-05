cmake_minimum_required(VERSION 3.26)
project(simple LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 17)

add_executable(vec_add vec_add.cu)
set_property(TARGET vec_add PROPERTY CUDA_ARCHITECTURES OFF)

add_executable(vec_add_mm vec_add_mm.cu)
set_property(TARGET vec_add_mm PROPERTY CUDA_ARCHITECTURES OFF)

add_executable(vec_add_gpu_init vec_add_gpu_init.cu)
set_property(TARGET vec_add_gpu_init PROPERTY CUDA_ARCHITECTURES OFF)

add_executable(vec_add_gpu_init_stream vec_add_gpu_init_stream.cu)
set_property(TARGET vec_add_gpu_init_stream PROPERTY CUDA_ARCHITECTURES OFF)

add_executable(vec_add_streams_best vec_add_streams_best.cu)
set_property(TARGET vec_add_streams_best PROPERTY CUDA_ARCHITECTURES OFF)
