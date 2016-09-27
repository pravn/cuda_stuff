
#include <iostream>
#include <cuda.h>
#include <Timer.h>
#define N 2048
#define BLOCK_SIZE 128

__global__ void vectorized_load_atomics_kernel(int *data, int *count){
  uint tid = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ int block_reduced[NUM_THREADS_PER_BLOCK];
  block_reduced[threadIdx.x] = 0;

  __syncthreads();

  int4 r_data = reinterpret_cast<int4*>(data)[tid];

    atomicAdd(&block_reduced[r_data.x],1);
    atomicAdd(&block_reduced[r_data.y],1);
    atomicAdd(&block_reduced[r_data.z],1);
    atomicAdd(&block_reduced[r_data.w],1);

  __syncthreads();

  for(int i=threadIdx.x; i<NUM_BINS; i+=NUM_BINS)
    atomicAdd(&count[i],block_reduced[i]);
}
