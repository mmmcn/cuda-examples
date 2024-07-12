#ifndef UTILS_UTILS_H_
#define UTILS_UTILS_H_

#include <iostream>
#include <memory>
#include <optional>

#include "cuda_runtime.h"


struct GPUTimer {
public:
  GPUTimer() : _stream(std::nullopt) {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
  }

  GPUTimer(cudaStream_t& stream) : _stream(stream) {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
  }

  void start() {
    if (!_stream.has_value()) {
      cudaStreamSynchronize(0);
      cudaEventRecord(_start, 0);
    } else {
       cudaStreamSynchronize(*_stream);
       cudaEventRecord(_start, *_stream);
    }
  }

  void stop() {
    if (!_stream.has_value()) {
      cudaEventRecord(_stop, 0);
      cudaEventSynchronize(_stop);
    } else {
      cudaEventRecord(_stop, *_stream);
      cudaEventSynchronize(_stop);
    }
  }

  float elapsed() {
    float elapsed = 0.0;
    cudaEventElapsedTime(&elapsed, _start, _stop);
    return elapsed;
  }

  ~GPUTimer() {
    cudaEventDestroy(_start);
    cudaEventDestroy(_stop);
  }
private:
  cudaEvent_t _start;
  cudaEvent_t _stop;
  std::optional<cudaStream_t> _stream;
};

#endif  // UTILS_UTILS_H_
