#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
__global__ void cub_block_histogram(int *data, int *result){
  typedef cub::BlockHistogram<int, NUM_THREADS_PER_BLOCK, 1, NUM_BINS,cub::BLOCK_HISTO_SORT> BlockHistogram;

  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ typename BlockHistogram::TempStorage temp_storage;

  __shared__ int smem_histogram[NUM_BINS];

  int tmp_data[1];

  tmp_data[1] = data[tid];

  BlockHistogram(temp_storage).Histogram(tmp_data, smem_histogram);

    __syncthreads();

    for(int tid = threadIdx.x; tid<NUM_BINS; tid+=blockDim.x)
      atomicAdd(&result[tid], smem_histogram[tid]);
}
  
