#ifndef UTILS_UTILS_H_
#define UTILS_UTILS_H_

#include <iostream>
#include <memory>
#include <optional>
#include <random>

#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

typedef __nv_half float16_t;
typedef __nv_half2 float162_t;
typedef __nv_bfloat16 bfloat16_t;
typedef __nv_bfloat162 bfloat162_t;

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t error = call;                                                   \
    if (error != cudaSuccess) {                                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;  \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
  } while (0)


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

template <typename T>
inline void randomInitFloat(T* data, int64_t numel, int low=-1, int high=1) {
    std::mt19937 gen_f(114514);
    std::uniform_real_distribution<> dis_f(low, high);
    for (int i = 0; i < numel; i++) {
        data[i] = static_cast<T>(dis_f(gen_f));
    }
}

template <typename T>
__host__ __device__ T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

#endif  // UTILS_UTILS_H_
