#define N 4
#include <iostream>
#include <cassert>

void initialize(float *a, float *b, float *c, int size){
  assert((a!=NULL)&&(b!=NULL)&&(c!=NULL));
  assert((a+size*size-1!=NULL)&&(b+size*size-1!=NULL)&&(c+size*size-1!=NULL));
  
  for(int rows=0; rows<size; rows++){
    for(int cols = 0; cols <size; cols++){
      a[N*rows+cols]=((float)rand())/RAND_MAX;
      b[N*rows+cols]=((float)rand())/RAND_MAX;
      c[N*rows+cols]=0.0f;
    }
  }
}


void matmul_host(float *a, float *b, float *c, int size){
  assert((a!=NULL)&&(b!=NULL)&&(c!=NULL));
  assert((a+size*size-1!=NULL)&&(b+size*size-1!=NULL)&&(c+size*size-1!=NULL));
  for(int i=0; i<size; i++){
    for(int j=0; j<size; j++){
      for(int k=0; k<size; k++){
	c[N*i+j] += a[N*i+k]*b[N*k+j];}
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
 



int main(int argc, char **argv){
  
  float *a, *b, *c;
  

  a = (float *) malloc(N*N*sizeof(float));
  b = (float *) malloc(N*N*sizeof(float));
  c = (float *) malloc(N*N*sizeof(float));


  initialize(a,b,c,N);
  matmul_host(a, b, c, N);

  unsigned int data_size;
  data_size = sizeof(float)*N*N/pow(2,20);

  std::cout << "Size of matrix in MB " << data_size  << std::endl;
  
    print(a,b,c,N);

    free(a);
    free(b);
    free(c);

  return 0;

  }  
  
