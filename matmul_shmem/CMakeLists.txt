cmake_minimum_required(VERSION 2.8)
Project(matmul_shmem)
SET(CMAKE_VERBOSE_MAKEFILE TRUE)


#CUDA_VERBOSE_BUILD(Default ON)

#ADD_EXECUTABLE(matmul
#  matmul.cu)

CUDA_ADD_EXECUTABLE(matmul_shmem matmul_shmem.cu)


