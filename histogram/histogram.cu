#include <iostream>
#include <cuda.h>
#include <Timer.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/sort.h>

//initial attempt - probably not very performant
//histogram with N bins in several blocks 
//using ballot to set bits 
//then reduce in warp using popc
//reduce warp sums to block level sum using standard parallel reduction in shared memory

//ballot and popc borrowed from the approach in 
//Shane Cook's CUDA Programming book
//single block 

const long int NUM_BLOCKS=1024;
#define WARP_SIZE 32 
#define NUM_BINS 32
#define NUM_THREADS_PER_BLOCK 128
#define NUM_WARPS_PER_BLOCK NUM_THREADS_PER_BLOCK/WARP_SIZE
#define BIN_UNROLL 8

#include "shmem_atomics_reducer.cuh"
#include "ballot_popc_reducer.cuh"
//this one seems to be bogus
//#include "reducer.cuh"
//another bogus test
//#include "test_reducer.cuh"
#include "vectorized_load_atomics.cuh"

inline double calc_bandwidth(double time_ms){
  int megabyte =  1<<20;
  return  (double) NUM_BLOCKS*NUM_THREADS_PER_BLOCK*sizeof(int)/megabyte /(time_ms*1e-3);
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

  CUDATimer atomics_timer;

  atomics_timer.startTimer();
  shmem_atomics_reducer<<< NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>> (d_data, d_result_atomics);

  cudaDeviceSynchronize();
  atomics_timer.stopTimer();

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

  float mbytes = NUM_THREADS_PER_BLOCK*NUM_BLOCKS*sizeof(int)*1e-6;

  std::cout << "MB = " << mbytes << std::endl;

  float bandwidth = mbytes/atomics_timer.getElapsedTime()*1e3;
  std::cout << "atomics bandwidth for scalar loads (MB/s) " << bandwidth << std::endl;

  for(int i=0; i<NUM_BINS; i++){
    std::cout << h_result[i] << " " << h_result_atomics[i] << std::endl;
    }

  cudaFree(d_data);
  delete[] h_result_atomics;
  cudaFree(d_result_atomics);
  delete[] h_result;

}

void run_vectorized_load_atomics(int *h_data){
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

  CUDATimer atomics_timer;

  
  atomics_timer.startTimer();
  vectorized_load_atomics_kernel<<< NUM_BLOCKS/4, NUM_THREADS_PER_BLOCK>>> (d_data, d_result_atomics);
  atomics_timer.stopTimer();

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

  float mbytes = NUM_THREADS_PER_BLOCK*NUM_BLOCKS*sizeof(int)*1e-6;
  std::cout << "MB = " << mbytes << std::endl;

  float bandwidth = mbytes/atomics_timer.getElapsedTime()*1e3;
  std::cout << "atomics bandwidth with vec4 loads (MB/s) " << bandwidth << std::endl;
  for(int i=0; i<NUM_BINS; i++){
    std::cout << h_result[i] << " " << h_result_atomics[i] << std::endl;
    }


  cudaFree(d_data);
  delete[] h_result_atomics;
  cudaFree(d_result_atomics);
  delete[] h_result;

}

  

//assume that we have sizes coded correctly
#ifdef TEST_REDUCER
void run_test_reducer(int *h_data){
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


    popc_timer.startTimer();

#ifdef DISCARD
     ballot_popc_reducer<<<NUM_BINS*NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>> (d_data, d_result_ballot_popc);
  ballot_popc_reducer2<<<NUM_BLOCKS, WARP_SIZE*NUM_BINS>>> (d_data, d_result_ballot_popc);
    ballot_popc_reducer3<<<NUM_BLOCKS, WARP_SIZE*NUM_BINS>>> (d_data, d_result_ballot_popc);
    ballot_popc_reducer4<<<NUM_BLOCKS, WARP_SIZE*NUM_BINS>>> (d_data, d_result_ballot_popc);
#endif

    //    dim3 dimBlock(NUM_THREADS_PER_BLOCK, NUM_BINS);

    

        const int NITER = 1;
    for (int iter = 0; iter < NITER; iter++)
        cudaMemset(d_result_ballot_popc, 0,  NUM_BINS*sizeof(int));
    //      parallel_popc_histogram_caller<<<NUM_BLOCKS, dimBlock>>> (d_data, d_result_ballot_popc);
    //     ballot_popc_reducer<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>> (d_data, d_result_ballot_popc);
    test_reducer<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>> (d_data, d_result_ballot_popc);
    //  shmem_atomics_reducer<<< NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>> (d_data, d_result_ballot_popc);
    popc_timer.stopTimer();



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

  float mbytes = NUM_THREADS_PER_BLOCK*NUM_BLOCKS*sizeof(int)*1e-6;

  //  std::cout << "MB = " << mbytes << std::endl;

  float bandwidth = NITER*mbytes/popc_timer.getElapsedTime()*1e3;

  //  float bandwidth = mbytes/milliseconds*1e3;

  std::cout << "popc bandwidth " << bandwidth << std::endl;



  for(int i=0; i<NUM_BINS; i++){
    std::cout << h_result[i] << " " << h_result_ballot_popc[i] << std::endl;
    }


  cudaFree(d_data);
  delete[] h_result_ballot_popc;
  cudaFree(d_result_ballot_popc);
  delete[] h_result;
  
}
#endif


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

  popc_timer.startTimer();

#ifdef DISCARD
     ballot_popc_reducer<<<NUM_BINS*NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>> (d_data, d_result_ballot_popc);
  ballot_popc_reducer2<<<NUM_BLOCKS, WARP_SIZE*NUM_BINS>>> (d_data, d_result_ballot_popc);
    ballot_popc_reducer3<<<NUM_BLOCKS, WARP_SIZE*NUM_BINS>>> (d_data, d_result_ballot_popc);
    ballot_popc_reducer4<<<NUM_BLOCKS, WARP_SIZE*NUM_BINS>>> (d_data, d_result_ballot_popc);
#endif

    

        const int NITER = 1;
    for (int iter = 0; iter < NITER; iter++)
      cudaMemset(d_result_ballot_popc, 0,  NUM_BINS*sizeof(int));
    //      parallel_popc_histogram_caller<<<NUM_BLOCKS, dimBlock>>> (d_data, d_result_ballot_popc);
    //     ballot_popc_reducer<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>> (d_data, d_result_ballot_popc);
    ballot_popc_reducer<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>> (d_data, d_result_ballot_popc);
    //  shmem_atomics_reducer<<< NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>> (d_data, d_result_ballot_popc);
    popc_timer.stopTimer();



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

  float mbytes = NUM_THREADS_PER_BLOCK*NUM_BLOCKS*sizeof(int)*1e-6;

  //  std::cout << "MB = " << mbytes << std::endl;

  float bandwidth = NITER*mbytes/popc_timer.getElapsedTime()*1e3;

  //  float bandwidth = mbytes/milliseconds*1e3;

  std::cout << "popc bandwidth " << bandwidth << std::endl;



  for(int i=0; i<NUM_BINS; i++){
    std::cout << h_result[i] << " " << h_result_ballot_popc[i] << std::endl;
    }


  cudaFree(d_data);
  delete[] h_result_ballot_popc;
  cudaFree(d_result_ballot_popc);
  delete[] h_result;
  
}
  
  
void run_thrust_sort_testing(int *h_data){
  int *d_data;

  cudaMalloc((void **) &d_data, NUM_BLOCKS*NUM_THREADS_PER_BLOCK*sizeof(int));
  cudaMemcpy(d_data, h_data, NUM_BLOCKS*NUM_THREADS_PER_BLOCK*sizeof(int), cudaMemcpyHostToDevice);

  thrust::device_ptr<int> t_data(d_data);

  CUDATimer sort_time;

  sort_time.startTimer();
  thrust::sort(t_data, t_data+NUM_BLOCKS*NUM_THREADS_PER_BLOCK);
  sort_time.stopTimer();

  double time_ms = sort_time.getElapsedTime();

  std::cout << "thrust time in ms " << time_ms << std::endl;
  std::cout << "thrust sort bandwidth in megabytes per second " << calc_bandwidth(time_ms) << std::endl;
    
  cudaFree(d_data);
}
  


int main()
{
  int *h_data; 
  h_data = new int[NUM_THREADS_PER_BLOCK*NUM_BLOCKS];

  std::cout << "dimensions " << NUM_THREADS_PER_BLOCK * NUM_BLOCKS << std::endl;

  for(int i=0; i<NUM_THREADS_PER_BLOCK*NUM_BLOCKS; i++){
    h_data[i] = (NUM_BINS) * ((float) rand()/RAND_MAX);

    #ifdef WORST_CASE
    h_data[i] = 0;
    #endif

	    

  }



  std::cout << "NUM_WARPS_PER_BLOCK " << NUM_WARPS_PER_BLOCK << std::endl;
  
  //run_ballot_popc_reducer(h_data);
  //run_test_reducer(h_data);
  run_atomics_reducer(h_data);
  run_vectorized_load_atomics(h_data);
  run_thrust_sort_testing(h_data);

  //cleanup
  delete[] h_data;

}
