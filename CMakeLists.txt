cmake_minimum_required(VERSION 2.8)
Project(cuda_experiments)

INCLUDE_DIRECTORIES(include)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CUDA_NVCC_FLAGS "-arch=sm_52 -std=c++11")
#ADD_SUBDIRECTORY(matmul_naive)
#ADD_SUBDIRECTORY(matmul_shmem)
ADD_SUBDIRECTORY(shffl)
ADD_SUBDIRECTORY(ballot)
ADD_SUBDIRECTORY(histogram)
