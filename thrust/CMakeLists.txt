cmake_minimum_required(VERSION 2.8)
Project(matmul_shmem)
SET(CMAKE_CXX_COMPILER g++)
SET(CMAKE_VERBOSE_MAKEFILE TRUE)
#CUDA_VERBOSE_BUILD(Default ON)

#ADD_EXECUTABLE(matmul
#  matmul.cu)

CUDA_ADD_EXECUTABLE(remove_points2d remove_points2d.cu)


