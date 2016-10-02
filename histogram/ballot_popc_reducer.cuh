__global__ void ballot_popc_reducer(int *data, int *count ){
  uint tid = blockIdx.x*blockDim.x + threadIdx.x;
  uint warp_id = threadIdx.x >> 5;

  uint lane_id = threadIdx.x%32;

  uint warp_set_bits=0;

  __shared__ uint warp_reduced_count[NUM_WARPS_PER_BLOCK];
  __shared__ uint s_data[NUM_THREADS_PER_BLOCK];


  s_data[threadIdx.x] = data[tid];

  __syncthreads();
  


  for(int i=0; i<NUM_BINS; i++){
      warp_set_bits = __ballot(s_data[threadIdx.x]==i);

      if(lane_id==0){
	warp_reduced_count[warp_id] = __popc(warp_set_bits);
      }
  
      __syncthreads();


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


