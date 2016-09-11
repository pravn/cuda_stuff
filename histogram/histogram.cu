#include <iostream>
#include <cuda.h>

//initial attempt - probably not very performant
//histogram with N bins in several blocks 
//using ballot to set bits 
//then reduce in warp using popc
//reduce warp sums to block level sum using standard parallel reduction in shared memory

//ballot and popc borrowed from the approach in 
//Shane Cook's CUDA Programming book
//single block 

#define NUM_BLOCKS 1
#define NUM_THREADS_PER_BLOCK 64
#define WARP_SIZE 32 
#define NUM_BINS 16

__global__ void reducer(int *data, int *count ){
  uint tid = blockIdx.x*blockDim.x + threadIdx.x;
  uint warp_id = threadIdx.x >> 5;

  uint lane_id = threadIdx.x% 32;

  uint warp_set_bits[NUM_BINS];

  __shared__ uint warp_reduced_count[WARP_SIZE][NUM_BINS];


  for(int i=0; i<NUM_BINS; i++){
    warp_set_bits[i] = __ballot(data[tid]==i);
  }


  if(lane_id==0){
    for(int i=0; i<NUM_BINS; i++){
      warp_reduced_count[warp_id][i] = __popc(warp_set_bits[i]);
    }
  }

  __syncthreads();
  //reduce to single value 
  if(warp_id==0){
    for(int j = WARP_SIZE/2; j>0; j>>=1){
      for(int i = 0; i< NUM_BINS; i++){
	if(tid<j) warp_reduced_count[tid][i] += warp_reduced_count[tid+j][i];
      __syncthreads();
      }
    }
  }

  __syncthreads();
      
  
  if(threadIdx.x==0){
    for(int i=0; i<NUM_BINS; i++){
      count[i] = warp_reduced_count[0][i];
    }
  }
  
}  



int main()
{
  int *h_data; 
  int *d_data;

  int *h_count;
  int *d_count;
  int *result;
  
  h_data = new int[NUM_THREADS_PER_BLOCK];
  cudaMalloc((void **) &d_data, sizeof(int)*NUM_THREADS_PER_BLOCK);

  for(int i=0; i<NUM_THREADS_PER_BLOCK; i++){
    h_data[i] = (NUM_BINS) * ((float) rand()/RAND_MAX);
  }

  h_count = new int[NUM_BINS];

  for(int j=0; j<NUM_BINS; j++){
    h_count[j] = 0;
  }

  for(int i=0; i<NUM_THREADS_PER_BLOCK; i++){
    for(int j=0; j<NUM_BINS; j++){
      if(h_data[i]==j){
	h_count[j]++;
      }
    }
  }


  cudaMemcpy(d_data, h_data, NUM_THREADS_PER_BLOCK*sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void **) &d_count, NUM_BINS*sizeof(int));
  cudaMemcpy(d_count, &h_count, NUM_BINS*sizeof(int), cudaMemcpyHostToDevice);

  reducer<<<1, NUM_THREADS_PER_BLOCK>>> (d_data, d_count);

    result = new int[NUM_BINS];
    cudaMemcpy(result, d_count, NUM_BINS*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<NUM_BINS; i++){
      std::cout << h_count[i] << " " << result[i] << std::endl;
    }

  
  //cleanup
  delete[] h_data;
  delete[] result;
  cudaFree(d_data);
  cudaFree(d_count);
  

}
