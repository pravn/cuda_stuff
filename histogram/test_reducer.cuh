__global__ void test_reducer(int *data, int *count){
  uint tid = blockIdx.x*blockDim.x + threadIdx.x;
  uint warp_id = threadIdx.x >> 5;
  uint lane_id = threadIdx.x%32;
 
  __shared__ uint s_data[NUM_THREADS_PER_BLOCK];
  __shared__ uint s_count[NUM_BINS];
  __shared__ uint warp_set_bits[NUM_BINS];

  //    s_data[threadIdx.x] = data[blockIdx.x*NUM_THREADS_PER_BLOCK+threadIdx.x];

  s_data[lane_id+warp_id*WARP_SIZE] = data[NUM_THREADS_PER_BLOCK*blockIdx.x+warp_id*WARP_SIZE+lane_id];

  __syncthreads();


  if(warp_id==0) warp_set_bits[lane_id]=0;
  if(warp_id==0) s_count[lane_id]=0;

  __syncthreads();


  //  atomicAdd(&s_count[s_data[lane_id+warp_id*WARP_SIZE]],1);


  /*  for(int i=0; i<NUM_BINS; i++){
    atomicAdd(&warp_set_bits[i], __ballot(s_data[i*WARP_SIZE+lane_id]==i));
    __syncthreads();
    }*/

  for(int i=0; i<NUM_BINS; i++){
        atomicOr(&warp_set_bits[warp_id],__ballot(warp_id==s_data[i*WARP_SIZE+lane_id]));
    //    warp_set_bits[warp_id] += __ballot(warp_id==s_data[i*WARP_SIZE+lane_id]);

    //  __syncthreads();
  //    atomicAdd(&s_count[warp_id], __popc(warp_set_bits[i]));
  //__syncthreads();

  }

  
    //  s_count[warp_id] += __popc(warp_set_bits[warp_id]);
  __syncthreads();


  if(warp_id==0){
    atomicAdd(&count[lane_id], s_count[lane_id]);
  }

}

	  
