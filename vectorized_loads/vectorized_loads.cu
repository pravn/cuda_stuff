#include <iostream>
#include <cuda.h>
#include <Timer.h>
#define N 2048
#define BLOCK_SIZE 128

inline double calc_bw(double time){
  //megabytes per second
  double bw;
  return bw = N*BLOCK_SIZE*sizeof(int)/(time/1e3)*1e-6;
}
  

//I am planning to use this to improve performance in atomics kernels

//from Justin Luitjens' blog post in parallelforall (CUDA Pro Tip: Increase performance with vectorized loads)
//https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
__global__ void device_copy_vec4_kernel(int *d_in, int *d_out){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  //works best if N is a multiple of 4
  //assume that N is a multiple of 4 for now
    for(int i=idx; i<N*BLOCK_SIZE/4; i += blockDim.x*gridDim.x){
    reinterpret_cast<int4*> (d_out)[i] = reinterpret_cast<int4*>(d_in)[i];
    }

}


void device_copy_vec4(int *d_in, int *d_out){
   int threads = BLOCK_SIZE;
   int blocks = N/4;
   device_copy_vec4_kernel<<<blocks,threads>>>(d_in, d_out);
}



__global__ void device_copy_vec1_kernel(int *d_in, int *d_out){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  for(int i=idx; i<N*BLOCK_SIZE; i+=blockDim.x*gridDim.x)
    d_out[i] = d_in[i];
}
						

void device_copy_vec1(int *d_in, int *d_out){
  int threads = BLOCK_SIZE;
  int blocks  = N;
  device_copy_vec1_kernel<<<blocks, threads>>> (d_in, d_out);
}

int main(){

  int *h_a;
  int *d_a;
  int *d_a_vec4;
  int *d_a_vec1;

  h_a = new int[N*BLOCK_SIZE];
  cudaMalloc((void **) &d_a,N*BLOCK_SIZE*sizeof(int));
  cudaMalloc((void **) &d_a_vec4,N*BLOCK_SIZE*sizeof(int));
  cudaMalloc((void **) &d_a_vec1,N*BLOCK_SIZE*sizeof(int));
  


  for(int i=0; i<N*BLOCK_SIZE; i++){
    h_a[i]=i;
  }




  cudaMemcpy(d_a, h_a, N*BLOCK_SIZE*sizeof(int), cudaMemcpyHostToDevice);




  CPUTimer timer;

  timer.startTimer();
  device_copy_vec4(d_a, d_a_vec4);

  timer.stopTimer();

#ifdef VALIDATE
  int *tmp = new int[N*BLOCK_SIZE];
  memset(tmp,0,N*BLOCK_SIZE);
  
  cudaMemcpy(tmp, d_a_vec4, N*BLOCK_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  
  
  std::cout << "Validating vec4 " << std::endl;
  
  for(int i=0; i<N*BLOCK_SIZE; i++){
    std::cout << i << " " << tmp[i] << std::endl;
    }


  delete[] tmp;
#endif

  std::cout << "vec4 bandwidth (MB/s) = " << calc_bw(timer.getElapsedTime()) << std::endl;

  timer.startTimer();
  device_copy_vec1(d_a, d_a_vec1);
  timer.stopTimer();

#ifdef VALIDATE
  tmp = new int[N*BLOCK_SIZE];
  memset(tmp,0,N*BLOCK_SIZE);
  
  cudaMemcpy(tmp, d_a_vec1, N*BLOCK_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  

  std::cout << "Validating vec1 " << std::endl;
  
  for(int i=0; i<N*BLOCK_SIZE; i++){
    std::cout << i << " " << tmp[i] << std::endl;
  }

  delete[] tmp;
#endif

  

  std::cout << "vec1 bandwidth (MB/s) = " << calc_bw(timer.getElapsedTime()) << std::endl;
    


  

  delete [] h_a;
  cudaFree(d_a);
  cudaFree(d_a_vec4);
  cudaFree(d_a_vec1);
  
  
}    


  
  
  



  



