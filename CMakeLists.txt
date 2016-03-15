cmake_minimum_required(VERSION 2.8)
Project(cuda_experiments)

#SET(CMAKE_CXX_COMPILER g++)
#FIND_PACKAGE(CUDA REQUIRED)
#SET(CMAKE_VERBOSE_MAKEFILE TRUE)


#CUDA_VERBOSE_BUILD(Default ON)

#ADD_EXECUTABLE(matmul
#  matmul.cu)

ADD_SUBDIRECTORY(matmul)
ADD_SUBDIRECTORY(matmul_shmem)
#CUDA_ADD_EXECUTABLE(matmul_shmem matmul_shmem.cu)
