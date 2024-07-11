#include <iostream>
#include <stdio.h>
#include <random>
#include "cuda_runtime.h"


#define TILE_DIM   32
#define BLOCK_ROWS 32
#define NUM_REPEAT 5
#define NUM_WARMUP 2

void checkResultAndShowGBPerSeconds(float* actual, float* desired, int N, float ms) {
  bool passed = true;
  for (int i = 0; i < N; i++) {
    if (actual[i] != desired[i]) {
      passed = false;
      printf("%25s\n", "*** FAILED ***");
      break;
    }
  }

  if (passed) {
    printf("%20.2f\n", 2 * N * sizeof(float) * 1e-6 * NUM_REPEAT / ms);
  }

}

__global__ void copy(float* odata, float* idata) {
  uint32_t x = blockIdx.x * TILE_DIM + threadIdx.x;
  uint32_t y = blockIdx.y * TILE_DIM + threadIdx.y;
  
  uint32_t width = gridDim.x * TILE_DIM;

  for (uint32_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    odata[(y + j) * width + x] = idata[(y + j) * width + x];
  }
}

__global__ void transposeNaive(float* odata, float* idata) {
  uint32_t x = blockIdx.x * TILE_DIM + threadIdx.x;
  uint32_t y = blockIdx.y * TILE_DIM + threadIdx.y;
  uint32_t width = gridDim.x * TILE_DIM;

  for (uint32_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    odata[x * width + (y + j)] = idata[(y + j) * width + x];
  }
}

__global__ void transposeSharedMem(float* odata, float* idata) {
  __shared__ float tile_shm[TILE_DIM][TILE_DIM];

  uint32_t x = blockIdx.x * TILE_DIM + threadIdx.x;
  uint32_t y = blockIdx.y * TILE_DIM + threadIdx.y;
  uint32_t width = gridDim.x * TILE_DIM;
  
  for (uint32_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    tile_shm[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  for (uint32_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    odata[(y + j) * width + x] = tile_shm[threadIdx.x][threadIdx.y + j];
  }
}

__global__ void transposeSharedMemNoBankConflicts(float* odata, float* idata) {
  __shared__ float tile_shm[TILE_DIM][TILE_DIM + 1];  // avoid 32-way bank conflicts

  uint32_t x = blockIdx.x * TILE_DIM + threadIdx.x;
  uint32_t y = blockIdx.y * TILE_DIM + threadIdx.y;
  uint32_t width = gridDim.x * TILE_DIM;
  
  for (uint32_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    tile_shm[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  for (uint32_t j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    odata[(y + j) * width + x] = tile_shm[threadIdx.x][threadIdx.y + j];
  }
}


int main() {
  const int n_rows = 1024;
  const int n_cols = 1024;
  const int mem_size = n_rows * n_cols * sizeof(float);

  dim3 block(TILE_DIM, BLOCK_ROWS, 1);
  dim3 grid((n_cols + TILE_DIM - 1) / TILE_DIM, (n_rows + TILE_DIM - 1) / TILE_DIM, 1);

  float* host_x = (float*)malloc(mem_size);
  float* host_y = (float*)malloc(mem_size);
  float* host_transpose_golden = (float*)malloc(mem_size);

  std::mt19937 gen_f(114514);
  std::uniform_real_distribution<> dis_f(-1, 1);
  for (int i = 0; i < n_rows * n_cols; i++) {
    host_x[i] = static_cast<float>(dis_f(gen_f));
  }

  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_cols; j++) {
      host_transpose_golden[j * n_cols + i] = host_x[i * n_cols + j];
    }
  }

  printf("%35s%25s\n", "Routine", "Bandwidth (GB/s)");

  float* device_x;
  float* device_y;
  cudaMalloc(&device_x, mem_size);
  cudaMalloc(&device_y, mem_size);
  cudaMemcpy(device_x, host_x, mem_size, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed = 0.0;

  
  // copy kernel
  printf("%35s", "copy");
  for (int i = 0; i < NUM_WARMUP; i++) {
    copy<<<grid, block>>>(device_y, device_x);
  }

  cudaEventRecord(start, 0);
  for (int i = 0; i < NUM_REPEAT; i++) {
    copy<<<grid, block>>>(device_y, device_x);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);

  cudaMemcpy(host_y, device_y, mem_size, cudaMemcpyDeviceToHost);
  checkResultAndShowGBPerSeconds(host_y, host_x, n_rows * n_cols, elapsed);
 
 
  // transposeNaive kernel
  printf("%35s", "transposeNaive");
  cudaMemset(device_y, 0, mem_size);
  for (int i = 0; i < NUM_WARMUP; i++) {
    transposeNaive<<<grid, block>>>(device_y, device_x);
  }
  
  cudaStreamSynchronize(0);
  cudaEventRecord(start, 0);
  for (int i = 0; i < NUM_REPEAT; i++) {
    transposeNaive<<<grid, block>>>(device_y, device_x);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);

  cudaMemcpy(host_y, device_y, mem_size, cudaMemcpyDeviceToHost);
  checkResultAndShowGBPerSeconds(host_y, host_transpose_golden, n_rows * n_cols, elapsed);
  

  // transposeSharedMem kernel
  printf("%35s", "transposeSharedMem");
  cudaMemset(device_y, 0, mem_size);
  for (int i = 0; i < NUM_WARMUP; i++) {
    transposeSharedMem<<<grid, block>>>(device_y, device_x);
  }
  
  cudaStreamSynchronize(0);
  cudaEventRecord(start, 0);
  for (int i = 0; i < NUM_REPEAT; i++) {
    transposeSharedMem<<<grid, block>>>(device_y, device_x);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);

  cudaMemcpy(host_y, device_y, mem_size, cudaMemcpyDeviceToHost);
  checkResultAndShowGBPerSeconds(host_y, host_transpose_golden, n_rows * n_cols, elapsed);
  
  
  // transposeSharedMemNoBankConflicts kernel
  printf("%35s", "transposeSharedMemNoBankConflicts");
  cudaMemset(device_y, 0, mem_size);
  for (int i = 0; i < NUM_WARMUP; i++) {
    transposeSharedMemNoBankConflicts<<<grid, block>>>(device_y, device_x);
  }
  
  cudaStreamSynchronize(0);
  cudaEventRecord(start, 0);
  for (int i = 0; i < NUM_REPEAT; i++) {
    transposeSharedMemNoBankConflicts<<<grid, block>>>(device_y, device_x);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);

  cudaMemcpy(host_y, device_y, mem_size, cudaMemcpyDeviceToHost);
  checkResultAndShowGBPerSeconds(host_y, host_transpose_golden, n_rows * n_cols, elapsed);

  cudaFree(device_x);
  cudaFree(device_y);
  free(host_x);
  free(host_y);
  free(host_transpose_golden);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
