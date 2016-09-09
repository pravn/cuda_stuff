cmake_minimum_required(VERSION 2.8)
Project(matmul_shmem)

SET(CMAKE_CXX_COMPILER g++)
FIND_PACKAGE(CUDA REQUIRED)
SET(CMAKE_VERBOSE_MAKEFILE TRUE)


#CUDA_VERBOSE_BUILD(Default ON)

#ADD_EXECUTABLE(matmul
#  matmul.cu)

CUDA_ADD_EXECUTABLE(shffl shffl.cu)


