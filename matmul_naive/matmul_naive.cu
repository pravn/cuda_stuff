#define SIZE 1024
#define TILE_WIDTH 64
#include <iostream>
#include <cassert>
#include <stdio.h>

template <typename T> 
struct Mat{
public:
  T *h_x;
  T *d_x;
  int n;

  Mat(size_t n_): n(n_)
  {
    //    assert(typename(T)==int or double or float)
    //    figure this out
    h_x = (T *) malloc(sizeof(T)*n*n);
    cudaMalloc((void**)&d_x, sizeof(T)*n*n);
  }

  void cudaMemcpyH2D(){
    cudaMemcpy(d_x, h_x, sizeof(T)*n*n, cudaMemcpyHostToDevice);
  }
    

  void cudaMemcpyD2H(){
    cudaMemcpy(h_x, d_x, sizeof(T)*n*n, cudaMemcpyDeviceToHost);
  }


  ~Mat(){
    free(h_x);
    cudaFree(d_x);
  }

};
  
  
void initialize(float *a, float *b, float *c, int size){
  assert((a!=NULL)&&(b!=NULL)&&(c!=NULL));
  assert((a+size*size-1!=NULL)&&(b+size*size-1!=NULL)&&(c+size*size-1!=NULL));
  
  for(int rows=0; rows<size; rows++){
    for(int cols = 0; cols <size; cols++){
      a[size*rows+cols]=((float)rand())/RAND_MAX;
      b[size*rows+cols]=((float)rand())/RAND_MAX;
      c[size*rows+cols]=0.0f;
    }
  }
}


void matmul_host(float *a, float *b, float *c, int size){
  assert((a!=NULL)&&(b!=NULL)&&(c!=NULL));
  assert((a+size*size-1!=NULL)&&(b+size*size-1!=NULL)&&(c+size*size-1!=NULL));
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      for(int k=0; k<size; k++){
	c[size*i+j] += a[size*i+k]*b[size*k+j];
      }
    }
  }
}  

void print(float *A, int size){
  assert((A!=NULL) && (A+size*size-1)!=NULL);
  
  for (int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      std::cout << A[size*i+j];

      if(j!=size-1){
	std::cout << " ";
      }
    }
    std::cout << std::endl;
  }
}
  

void print(float *a, float *b, float *c, int size){
  assert((a!=NULL)&&(b!=NULL)&&(c!=NULL));
  assert((a+size*size-1!=NULL)&&(b+size*size-1!=NULL)&&(c+size*size-1!=NULL));
  
  std::cout << "A=" << std::endl;
    print(a, size);

  std::cout << "B=" << std::endl;
  print(b, size);

  std::cout << "C=" << std::endl;
  print(c, size);

}
 
//naive
__global__ void matmul_device(float *a, float *b, float *c, int size){
  assert((a!=NULL)&&(b!=NULL)&&(c!=NULL));
  assert((a+size*size-1!=NULL)&&(b+size*size-1!=NULL)&&(c+size*size-1!=NULL));

  int tx = blockIdx.x*TILE_WIDTH + threadIdx.x;
  int ty = blockIdx.y*TILE_WIDTH + threadIdx.y;

  float p = 0;
  
  while((tx<size)&&(ty<size)){
  for(int k=0; k<size; k++){
    p += a[size*ty+k]*b[size*k+tx];
  }


  c[ty+size*tx]=p;

  }
}


int main(int argc, char **argv){
  
  float *a, *b, *c;
  float *a_d, *b_d, *c_d;
  

  a = (float *) malloc(SIZE*SIZE*sizeof(float));
  b = (float *) malloc(SIZE*SIZE*sizeof(float));
  c = (float *) malloc(SIZE*SIZE*sizeof(float));


  initialize(a,b,c,SIZE);
  //    matmul_host(a, b, c, SIZE);

    std::cout << "HOST SUCCESS " << std::endl;
    //    print(c, SIZE);

  cudaMalloc((void **)&a_d,SIZE*SIZE*sizeof(float));
  cudaMalloc((void **)&b_d,SIZE*SIZE*sizeof(float));
  cudaMalloc((void **)&c_d,SIZE*SIZE*sizeof(float));

  cudaMemcpy(a_d, a, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);


  dim3 dimGrid(SIZE/TILE_WIDTH, SIZE/TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

  
  
  matmul_device<<<dimGrid, dimBlock>>>(a_d, b_d, c_d, SIZE);

  std::cout << "DEVICE SUCCESS " << std::endl;

  unsigned int data_size;
  data_size = sizeof(float)*SIZE*SIZE/pow(2,20);

  std::cout << "Size of matrix in MB " << data_size  << std::endl;
  
  cudaMemcpy(c, c_d, SIZE*SIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //  cudaMemcpy(a, a_d, SIZE*SIZE*sizeof(float), cudaMemcpyDeviceToHost);

  //   print(c,SIZE);

  free(a);
  free(b);
  free(c);

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  return 0;

  }  
  
