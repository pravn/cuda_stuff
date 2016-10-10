#include <iostream>
#include "device_container.h"
int main(){
  const int N=4;
  device_container<int>a(N);
  int *x = new int[N];

  for(int i=0; i<N; i++){
    x[i] = 2*i;
  }


  a.H2D(x);


  int *y = new int[N];
  a.D2H(y);

  std::cout << "Print output of cudaMemcpyDeviceToHost " << std::endl;
  for(int i=0; i<N; i++){
    std::cout << i << " " << y[i] << std::endl;
  }

  delete[] x;
  delete[] y;
  
}
  

  
  
  

 
    
