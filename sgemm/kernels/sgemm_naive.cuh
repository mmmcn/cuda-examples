#pragma once
#include <iostream>

__global__ void sgemm_naive_kernel(float* output_data, const float* A_data, const float* B_data, int M, int N, int K, float alpha, float beta) {
    uint32_t row_idx = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (row_idx >= M || col_idx >= N) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += A_data[row_idx * K + i] * B_data[i * N + col_idx];
    }

    output_data[row_idx * N + col_idx] = alpha * sum + beta * output_data[row_idx * N + col_idx];
}