ALL:histogram

NVCC:=/usr/local/cuda-7.5/bin/nvcc
histogram: histogram.o
	$(NVCC) -o histogram histogram.o -L/usr/local/cuda-7.5/lib64 -lcudart_static -lcudadevrt

histogram.o: histogram.cu
	$(NVCC) -c histogram.cu -I../include -std=c++11 -rdc=true -arch=sm_52

clean:
	rm -f histogram.o histogram
