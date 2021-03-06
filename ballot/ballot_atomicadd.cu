#include <iostream>
#include <cuda.h>


//find number of points above a certain threshold
//using atomicOr and popc
//ballot and popc, atomicOr borrowed from the approach in 
//Shane Cook's CUDA Programming book
//single block 

//pretty trivial setting 
#define NUM_WARPS_PER_BLOCK 2
#define NUM_BLOCKS 1
#define NUM_THREADS_PER_BLOCK 64

__global__ void reducer(int *data, int *count ){
  uint tid = blockIdx.x*blockDim.x + threadIdx.x;
  uint warp_id = threadIdx.x >> 5;
  int threshold = 0;

  uint lane_id = threadIdx.x% 32;

  __shared__ uint warp_set_bits[NUM_WARPS_PER_BLOCK];
  __shared__ uint block_reduced_count;

  block_reduced_count = 0;

  //set the bits for each warp and copy to warp shared data
  atomicOr(&warp_set_bits[warp_id], __ballot(data[tid]>threshold));

  if(threadIdx.x == 0){
    *count += __popc(warp_set_bits[0]) + __popc(warp_set_bits[1]);
    printf("reduced count %d\n", *count);
  }

  
}  



int main(){
    int *h_data; 
  int *d_data;
  
  int h_count = 0;
  int *d_count;


  
  h_data = new int[NUM_THREADS_PER_BLOCK];
  cudaMalloc((void **) &d_data, sizeof(int)*NUM_THREADS_PER_BLOCK);

  for(int i=0; i<NUM_THREADS_PER_BLOCK; i++){
    h_data[i] = i;
    //    std::cout << h_data[i] << std::endl;
  }


  cudaMemcpy(d_data, h_data, NUM_THREADS_PER_BLOCK*sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void **) &d_count, sizeof(int));
  cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

  
  reducer<<<1, NUM_THREADS_PER_BLOCK>>> (d_data, d_count);

  cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "(count > 0) " << h_count << std::endl;
  

  //cleanup
  delete[] h_data;
  cudaFree(d_data);
  cudaFree(d_count);

}
