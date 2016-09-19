#include <stdio.h>
#include <cuda.h>

#define cucheck_dev(call)                                   \
  {                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
  const char *err_str = cudaGetErrorString(cucheck_err);  \
  printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
  assert(0);                                              \
  }                                                         \
  }



__global__ void junk_kernel(){}  

__global__ void parallel_popc_histogram_caller(int *data, int *count){

  int tid_x = threadIdx.x;
  int tid_y = blockIdx.y*NUM_THREADS_PER_BLOCK+threadIdx.y;

  int lane_x = threadIdx.x%32;
  int warp_x = threadIdx.x/32;

  int lane_y = threadIdx.y%32;
  int warp_y = threadIdx.y/32;

  //  __shared__ uint s_data[NUM_THREADS_PER_BLOCK]; //now in y direction
  __shared__ uint s_count[NUM_BINS];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  uint warp_set_bits;



  //we load 32 data items in coalesced fashion 
  //naturally, this only makes sense in the zeroth warp
  //we MUST use the same data in the entire x-y block (!)


  uint r = data[blockIdx.x*NUM_THREADS_PER_BLOCK+threadIdx.x];

  warp_set_bits = __ballot(lane_y==r);

  uint warp_count_bits;


  warp_count_bits = __popc(warp_set_bits);
  


  if(lane_x==0){
    atomicAdd(&s_count[lane_y],warp_count_bits);
  }

  
  __syncthreads();

  

  //    printf("blockDim.y %d\n", s_count[0]);


  if(lane_y==0){
    for(int i=threadIdx.x; i<NUM_BINS; i+=NUM_BINS){
      atomicAdd(&count[i], s_count[i]);
    }
    }
  


  /*  if((warp_x==0)&&(warp_y==0)&&(threadIdx.x<NUM_BINS)){
    atomicAdd(&count[threadIdx.x],s_count[threadIdx.x]);
    }*/


  /*
  if(threadIdx.x==0){
    printf("scont [0] %d\n", s_count[0]);
    printf("scount [1] %d\n", s_count[1]);
    }*/
  
  


  //    printf("%d \n", s_data[lane_x + warp_x*WARP_SIZE]);

  ///  printf("%d\n", threadIdx.x);

  

  //  atomicAdd(count, 2);


  //  warp_set_bits = __ballot(warp_y==s_data[ty]);

  //    printf("Hello from tid %d\n", lane_x+NUM_BINS*tid_y);

    /*
  int tmp_count = __popc(warp_set_bits);*/


  //make each warp search through bins and accumulate in shared mem
   //first copy global data into shared memory 
  
}

  
