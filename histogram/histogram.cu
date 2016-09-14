#include <iostream>
#include <cuda.h>
#include <Timer.h>

//initial attempt - probably not very performant
//histogram with N bins in several blocks 
//using ballot to set bits 
//then reduce in warp using popc
//reduce warp sums to block level sum using standard parallel reduction in shared memory

//ballot and popc borrowed from the approach in 
//Shane Cook's CUDA Programming book
//single block 

const long int NUM_BLOCKS=2<<8;
#define NUM_THREADS_PER_BLOCK 32
#define WARP_SIZE 32 
#define NUM_BINS 32
#define NUM_WARPS_PER_BLOCK NUM_THREADS_PER_BLOCK/WARP_SIZE
#define BIN_UNROLL 8

__global__ void shmem_atomics_reducer(int *data, int *count){
  uint tid = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ int block_reduced[NUM_THREADS_PER_BLOCK];
  block_reduced[threadIdx.x] = 0;

  __syncthreads();

  

    atomicAdd(&block_reduced[data[tid]],1);
  __syncthreads();

  for(int i=threadIdx.x; i<NUM_BINS; i+=NUM_BINS)
    atomicAdd(&count[i],block_reduced[i]);
  
}
	  

__global__ void ballot_popc_reducer(int *data, int *count ){
  uint tid = blockIdx.x*blockDim.x + threadIdx.x;
  uint warp_id = threadIdx.x >> 5;

  uint lane_id = threadIdx.x%32;

  uint warp_set_bits=0;

  __shared__ uint warp_reduced_count[NUM_WARPS_PER_BLOCK];


  for(int i=0; i<NUM_BINS; i++){
    if(blockIdx.x%NUM_BINS==i){
    //  printf("w %d\n", data[tid]);

      int did = blockIdx.x/NUM_BINS*blockDim.x + threadIdx.x;

      warp_set_bits = __ballot(data[did]==i);
      //      warp_set_bits = __ballot(s_data[threadIdx.x]==i);

      if(lane_id==0){
	warp_reduced_count[warp_id] = __popc(warp_set_bits);
      }
  
      //    printf("warp_reduced_count %d\n", warp_reduced_count[warp_id][i]);
  
      __syncthreads();
      //reduce to single value 


      if(warp_id==0){
	int t = threadIdx.x;
	for(int j = NUM_WARPS_PER_BLOCK/2; j>0; j>>=1){
	  if(t<j) warp_reduced_count[t] += warp_reduced_count[t+j];
	  __syncthreads();
	}
      }//warp id


      __syncthreads();
      

      if(threadIdx.x==0){
	atomicAdd(&count[i],warp_reduced_count[0]);
	}


    } // if(blockIdx.x%NUM_BINS==i)
  
  }//for 
}



__global__ void ballot_popc_reducer2(int *data, int *count ){
  uint tid = blockIdx.x*blockDim.x + threadIdx.x;
  uint warp_id = threadIdx.x >> 5;

  uint lane_id = threadIdx.x%32;

  uint warp_set_bits=0;

  __shared__ uint s_data[WARP_SIZE];

  __shared__ uint warp_reduced_count[NUM_WARPS_PER_BLOCK];

  uint did = blockIdx.x*WARP_SIZE + lane_id;

  if(warp_id==0){
    s_data[lane_id] = data[did];
  }
  __syncthreads();


  for(int i=0; i<NUM_BINS; i++){
    if(warp_id==i){
    //  printf("w %d\n", data[tid]);
      warp_set_bits = __ballot(s_data[lane_id]==i);
      //      warp_set_bits = __ballot(s_data[threadIdx.x]==i);

      if(lane_id==0){
	warp_reduced_count[warp_id] = __popc(warp_set_bits);
      }
      //    printf("warp_reduced_count %d\n", warp_reduced_count[warp_id][i]);
      __syncthreads();
      //reduce to single value 

    if(lane_id==0){
      atomicAdd(&count[i], warp_reduced_count[i]);
    }

    } // if(blockIdx.x%NUM_BINS==i)



  
  }//for 


}


__global__ void ballot_popc_reducer3(int *data, int *count ){
  uint tid = blockIdx.x*blockDim.x + threadIdx.x;
  uint warp_id = threadIdx.x >> 5;

  uint lane_id = threadIdx.x%32;

  uint warp_set_bits[BIN_UNROLL];

  __shared__ uint s_data[WARP_SIZE];

  __shared__ uint warp_reduced_count[NUM_WARPS_PER_BLOCK];

  uint did = blockIdx.x*WARP_SIZE + lane_id;

  if(warp_id==0){
    s_data[lane_id] = data[did];
  }
  __syncthreads();


  for(int i=0; i<NUM_BINS/BIN_UNROLL; i++){
    if(warp_id==i){
    //  printf("w %d\n", data[tid]);
#pragma unroll 
      for(int j=0; j<BIN_UNROLL;j++){
	warp_set_bits[j] = __ballot(s_data[lane_id]==BIN_UNROLL*i+j);


	if(lane_id==0){
	  warp_reduced_count[i*BIN_UNROLL+j] = __popc(warp_set_bits[j]);
	}

	//    printf("warp_reduced_count %d\n", warp_reduced_count[warp_id][i]);
	__syncthreads();
	//reduce to single value 

	if(lane_id==0){
	  atomicAdd(&count[i*BIN_UNROLL+j], warp_reduced_count[i*BIN_UNROLL+j]);
	}
      }
    } // if(blockIdx.x%NUM_BINS==i)
  
  }//for 


}




__global__ void ballot_popc_reducer4(int *data, int *count ){
  uint tid = blockIdx.x*blockDim.x + threadIdx.x;
  uint warp_id = threadIdx.x >> 5;

  uint lane_id = threadIdx.x%32;

  uint warp_set_bits[BIN_UNROLL];

  __shared__ uint s_data[WARP_SIZE];

  __shared__ uint warp_reduced_count[NUM_WARPS_PER_BLOCK];

  uint did = blockIdx.x*WARP_SIZE + lane_id;

  if(warp_id==0){
    s_data[lane_id] = data[did];
  }
  __syncthreads();


    for(int i=0; i<NUM_BINS/BIN_UNROLL; i++){
    if(warp_id==i){
    //  printf("w %d\n", data[tid]);
#pragma unroll 
      for(int j=0; j<BIN_UNROLL;j++){
	warp_set_bits[j] = __ballot(s_data[lane_id]==BIN_UNROLL*i+j);
	

	if(lane_id==0){
	  warp_reduced_count[i*BIN_UNROLL+j] = __popc(warp_set_bits[j]);
	}

	//    printf("warp_reduced_count %d\n", warp_reduced_count[warp_id][i]);
	__syncthreads();
	//reduce to single value 

	if(lane_id==0){
	  atomicAdd(&count[i*BIN_UNROLL+j], warp_reduced_count[i*BIN_UNROLL+j]);
	}
      }
    } // if(blockIdx.x%NUM_BINS==i)
  
  }//for 


    
}


void run_atomics_reducer(int *h_data){
  int *d_data;
  int *h_result_atomics;
  int *d_result_atomics;
  int *h_result;

  cudaMalloc((void **) &d_data, NUM_THREADS_PER_BLOCK*NUM_BLOCKS*sizeof(int));
  cudaMemcpy(d_data, h_data, NUM_THREADS_PER_BLOCK*NUM_BLOCKS*sizeof(int), cudaMemcpyHostToDevice);

  h_result = new int[NUM_BINS];
  memset(h_result, 0, NUM_BINS*sizeof(int));

  cudaMalloc((void **) &d_result_atomics, NUM_BINS*sizeof(int));
  cudaMemset(d_result_atomics, 0, NUM_BINS*sizeof(int));


  //  CUDATimer atomics_timer;

  //    CUDATimer atomics_timer;
  CUDATimer atomics_timer;

  atomics_timer.startTimer();

  //  cudaEventRecord(start, 0);
  shmem_atomics_reducer<<< NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>> (d_data, d_result_atomics);

  cudaDeviceSynchronize();
  atomics_timer.stopTimer();

  //  cudaEventRecord(stop, 0);


  //  float milliseconds;
  //  cudaEventSynchronize(stop);
  //  cudaEventElapsedTime(&milliseconds, start, stop);

  for(int i=0; i<NUM_THREADS_PER_BLOCK*NUM_BLOCKS; i++){
    for(int j=0; j<NUM_BINS; j++){
      if(h_data[i]==j)
	h_result[j]++;
    }
  }

  h_result_atomics = new int[NUM_BINS];
  cudaMemcpy(h_result_atomics, d_result_atomics, NUM_BINS*sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "======================================" << std::endl;
    std::cout << "atomics time " << atomics_timer.getElapsedTime() << std::endl;

  //  std::cout << "atomics time " << milliseconds << std::endl;

  //  float mbytes = NUM_THREADS_PER_BLOCK*NUM_BLOCKS*sizeof(int)*1e-6; //MB
  float mbytes = NUM_THREADS_PER_BLOCK*NUM_BLOCKS*sizeof(int)*1e-6;

  std::cout << "MB = " << mbytes << std::endl;

   float bandwidth = mbytes/atomics_timer.getElapsedTime()*1e3;
   //  float bandwidth = mbytes/milliseconds*1e3;

  std::cout << "atomics bandwidth " << bandwidth << std::endl;


  /*  
  for(int i=0; i<NUM_BINS; i++){
    std::cout << h_result[i] << " " << h_result_atomics[i] << std::endl;
    }*/
  

  
  //  cudaEventDestroy(start);
  //  cudaEventDestroy(stop);

  cudaFree(d_data);
  delete[] h_result_atomics;
  cudaFree(d_result_atomics);
  delete[] h_result;

}
  

//assume that we have sizes coded correctly
void run_ballot_popc_reducer(int *h_data){
  int *d_data;
  int *h_result_ballot_popc;
  int *d_result_ballot_popc;
  int *h_result;
  
  cudaMalloc((void **) &d_data, NUM_BLOCKS*NUM_THREADS_PER_BLOCK*sizeof(int));
  cudaMemcpy(d_data, h_data, NUM_BLOCKS*NUM_THREADS_PER_BLOCK*sizeof(int), cudaMemcpyHostToDevice);

  h_result = new int[NUM_BINS];
  memset(h_result, 0, NUM_BINS*sizeof(int));
  
  cudaMalloc((void **) &d_result_ballot_popc, NUM_BINS*sizeof(int));
  cudaMemset(d_result_ballot_popc, 0, NUM_BINS*sizeof(int));

  CUDATimer popc_timer;


  //  cudaEventRecord(start);
    popc_timer.startTimer();
    //  ballot_popc_reducer<<<NUM_BINS*NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>> (d_data, d_result_ballot_popc);
  //ballot_popc_reducer2<<<NUM_BLOCKS, WARP_SIZE*NUM_BINS>>> (d_data, d_result_ballot_popc);
  //  ballot_popc_reducer3<<<NUM_BLOCKS, WARP_SIZE*NUM_BINS>>> (d_data, d_result_ballot_popc);
  const int NITER = 1000;
  for (int iter = 0; iter < NITER; iter++)
    ballot_popc_reducer4<<<NUM_BLOCKS, WARP_SIZE*NUM_BINS>>> (d_data, d_result_ballot_popc);
    popc_timer.stopTimer();

    //  cudaEventRecord(stop);

    //  cudaEventSynchronize(stop);
  
    //  float milliseconds = 0;
  //  cudaEventElapsedTime(&milliseconds, start, stop);
  

  for(int i=0; i<NUM_THREADS_PER_BLOCK*NUM_BLOCKS; i++){
    for(int j=0; j<NUM_BINS; j++){
      if(h_data[i]==j)
	h_result[j]++;
    }
  }
	
  h_result_ballot_popc = new int[NUM_BINS];
  cudaMemcpy(h_result_ballot_popc, d_result_ballot_popc, NUM_BINS*sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "===================================" << std::endl;
  std::cout << "popc time " << popc_timer.getElapsedTime() << std::endl;
									   //  std::cout << "popc time " << milliseconds << std::endl;

  float bytes = NUM_THREADS_PER_BLOCK*NUM_BLOCKS*sizeof(int);

  //  std::cout << "MB = " << mbytes << std::endl;

  float bandwidth = NITER*bytes/popc_timer.getElapsedTime()/1e6;

  //  float bandwidth = mbytes/milliseconds*1e3;

  std::cout << "popc bandwidth " << bandwidth << std::endl;
  

  /*
  for(int i=0; i<NUM_BINS; i++){
    std::cout << h_result[i] << " " << h_result_ballot_popc[i] << std::endl;
    }*/


  cudaFree(d_data);
  delete[] h_result_ballot_popc;
  cudaFree(d_result_ballot_popc);
  delete[] h_result;
  
}
  
  
  
  


int main()
{
  int *h_data; 
  h_data = new int[NUM_THREADS_PER_BLOCK*NUM_BLOCKS];

  std::cout << "dimensions " << NUM_THREADS_PER_BLOCK * NUM_BLOCKS << std::endl;

  for(int i=0; i<NUM_THREADS_PER_BLOCK*NUM_BLOCKS; i++){
    h_data[i] = (NUM_BINS) * ((float) rand()/RAND_MAX);
    //    printf("data[%d] %d\n", i, h_data[i]);
  }



  std::cout << "NUM_WARPS_PER_BLOCK " << NUM_WARPS_PER_BLOCK << std::endl;
  
  run_ballot_popc_reducer(h_data);
  //  cudaDeviceSynchronize();
    run_atomics_reducer(h_data);



  //cleanup
  delete[] h_data;

}
