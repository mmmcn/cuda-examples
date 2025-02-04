#pragma once
#include <iostream>

template <int BLOCK_SIZE>
__global__ void sgemm_v2_kernel(float* output_data, const float* A_data, const float* B_data, int M, int N, int K, float alpha, float beta) {
    uint32_t row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col_idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float shm_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shm_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;
    for (int outer_k = 0; outer_k < K; outer_k += BLOCK_SIZE) {
        shm_A[threadIdx.y][threadIdx.x] = A_data[row_idx * K + outer_k + threadIdx.x];
        shm_B[threadIdx.y][threadIdx.x] = B_data[(outer_k + threadIdx.y) * N + col_idx];

        __syncthreads();

        #pragma unroll
        for (int inner_k = 0; inner_k < BLOCK_SIZE; inner_k++) {
            sum += shm_A[threadIdx.y][inner_k] * shm_B[inner_k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row_idx < M && col_idx < N) {
        output_data[row_idx * N + col_idx] = alpha * sum + beta * output_data[row_idx * N + col_idx];
    }
}
