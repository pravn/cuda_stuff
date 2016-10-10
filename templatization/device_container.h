#ifndef DEVICE_CONTAINER_H
#define DEVICE_CONTAINER_H
template  <typename T>
class device_container{
public:
  device_container(int N=1){
   _size = N;
   cudaError_t cuda_status = cudaMalloc((void **) &x, sizeof(T)*_size);

  if(cuda_status!=cudaSuccess){
    std::cout << "Failed to do cudaMalloc " << cudaGetErrorString(cuda_status) << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Deleting device_container " << std::endl;
}

  ~device_container(){
    cudaError_t cuda_status = cudaFree(x);
    if(cuda_status!=cudaSuccess){
      std::cout << "Failed to do cudaFree " << cudaGetErrorString(cuda_status) << std::endl;
      exit(EXIT_FAILURE);
    }
  }
    
  inline  size_t size(){
    return _size;
  }

  T return_raw_pointer(){
    return &x[0];
  }

  void H2D(T *h_ptr){
    cudaError_t cuda_status = cudaMemcpy(x, h_ptr, _size*sizeof(T), cudaMemcpyHostToDevice);
    if(cuda_status!=cudaSuccess){
      std::cout << "Failed to copy to device " << cudaGetErrorString(cuda_status) << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  void D2H(T *h_ptr){
    cudaError_t cuda_status = cudaMemcpy(h_ptr, x, _size*sizeof(T), cudaMemcpyDeviceToHost);
    if(cuda_status!=cudaSuccess){
      std::cout << "Failed to copy from device " << cudaGetErrorString(cuda_status) << std::endl;
      exit(EXIT_FAILURE);
    }
  }
    
private:
T *x;
size_t _size;
};
#endif    



    
  
