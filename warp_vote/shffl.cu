#include <stdio.h>
#include <iostream>
#define warpSize 32

inline __device__
int warpReduceSum(int value){
  int laneId = threadIdx.x %32;

  for(int i=warpSize/2; i>0; i/=2){
    value += __shfl_down(value, i);
  }

  return value;
  //  printf("New value %d %d\n", threadIdx.x, value );
  
}

__inline__ __device__
int blockReduceSum(int val){
  static __shared__ int shared[32];
  int lane = threadIdx.x % warpSize;
  int wid  = threadIdx.x / warpSize;

  //  int val = threadIdx.x;

  val = warpReduceSum(val);



  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] :0;

  //    printf("val %d\n", val);

  //  val = shared[lane];

  if(wid == 0) val = warpReduceSum(val);
  return val;
}


__global__ void reducer(int *h_reduced, int *d_reduced){
  int redval = h_reduced[blockIdx.x*blockDim.x + threadIdx.x];
  redval=  blockReduceSum(redval);

  if(threadIdx.x==0) *d_reduced = redval;

}
  


int main(){
  int *h_val;
  int *d_val;
  int h_reduced;
  int *d_reduced;

  const int N = 128;
  h_val = new int[N];

  for(int i=0; i<N; i++){
    h_val[i] = i;
  }
  

  cudaMalloc((void **) &d_val, sizeof(int)*N);
  cudaMemcpy(d_val, h_val, sizeof(int)*N, cudaMemcpyHostToDevice);

 cudaMalloc((void **) &d_reduced, sizeof(int));
  
  reducer<<<1, 128>>> (d_val, d_reduced );
  cudaDeviceSynchronize();

  cudaMemcpy(&h_reduced, d_reduced, sizeof(int), cudaMemcpyDeviceToHost);
 
  std::cout << "Reduced val " << h_reduced << std::endl;
  std::cout << "N*(N-1)/2 = " << N*(N-1)/2 << std::endl;

  delete[] h_val;
  cudaFree(d_val);
  return 0;
}

    		   
