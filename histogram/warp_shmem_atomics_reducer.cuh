__global__ void warp_shmem_atomics_reducer(int *data, int *count){
  uint tid = blockIdx.x*blockDim.x + threadIdx.x;

  #define num_warps  NUM_THREADS_PER_BLOCK/WARP_SIZE
  int warp_id   = threadIdx.x/WARP_SIZE;
  int lane_id   = threadIdx.x%WARP_SIZE;

  __shared__ int block_reduced[NUM_THREADS_PER_BLOCK];
  block_reduced[threadIdx.x] = 0;

  __shared__ int warp_reduced[num_warps*NUM_BINS];

  for(int i = lane_id; i<WARP_SIZE; i+=blockDim.x){
    warp_reduced[warp_id*NUM_BINS+i]  = 0;
  }

  __syncthreads();

  atomicAdd(&warp_reduced[warp_id*NUM_BINS+data[tid]],1);

  __syncthreads();

  //sum warp subhistograms
  //only the zeroth warp adds
  if(warp_id==0){
    for(int i=0; i<num_warps; i++){
      block_reduced[threadIdx.x] += warp_reduced[i*NUM_BINS+threadIdx.x];
    }
    __syncthreads();
  }

  for(int i=threadIdx.x; i<NUM_BINS; i+=NUM_BINS)
    atomicAdd(&count[i],block_reduced[i]);
  
}
