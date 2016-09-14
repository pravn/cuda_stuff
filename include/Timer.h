#ifndef TIMER_H
#define TIMER_H

class Timer{

  virtual void startTimer()=0;
  virtual void stopTimer()=0;
  virtual double getElapsedTime()=0;

};



#include <chrono>
#include <ctime>


class CPUTimer: public Timer{
 public:
  CPUTimer(){
    startTime = std::chrono::system_clock::now();
    elapsedTime = startTime - startTime;
  }
    

  void startTimer() {
    startTime = std::chrono::system_clock::now();
  }


  void stopTimer() {
    stopTime = std::chrono::system_clock::now();
    elapsedTime += stopTime-startTime;
  }

  double getElapsedTime(){
    return 1000.0f*elapsedTime.count();
  }


 private:
  std::chrono::time_point<std::chrono::system_clock> startTime;
  std::chrono::time_point<std::chrono::system_clock> stopTime;
  std::chrono::duration<double> elapsedTime;
};



class CUDATimer: public Timer{
 public:
  CUDATimer(){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~CUDATimer(){
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void startTimer(){
    cudaEventRecord(start,0);
  }

  void stopTimer(){
    cudaEventRecord(stop,0);
  }

  double getElapsedTime(){
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    return static_cast<double> (milliseconds);
  }


 private:
  cudaEvent_t start, stop;
  float milliseconds = 0;
};
  
    
#endif    
  
