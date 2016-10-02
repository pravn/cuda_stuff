#ifdef JUNK
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

	__syncthreads();
	//reduce to single value 

	if(lane_id==0){
	  atomicAdd(&count[i*BIN_UNROLL+j], warp_reduced_count[i*BIN_UNROLL+j]);
	}
      }
    } // if(blockIdx.x%NUM_BINS==i)
  
  }//for 


    
}
#endif

