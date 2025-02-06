#pragma once
#include <iostream>
#include <assert.h>


template <int TILE_M, int TILE_N, int TILE_K, int TM, int TN, int BLOCK_SIZE>
__global__ void sgemm_v3_kernel(float* output_data, const float* A_data, const float* B_data, int M, int N, int K, float alpha, float beta) {
    
    // // each thread load one element from global memory to shared memory
    // static_assert(TILE_M * TILE_K == BLOCK_SIZE);    
    // static_assert(TILE_N * TILE_K == BLOCK_SIZE);

    __shared__ float smem_A[TILE_M][TILE_K];
    __shared__ float smem_B[TILE_K][TILE_N];
    
    uint32_t block_col_idx = threadIdx.x % TILE_N;
    uint32_t block_row_idx = threadIdx.x / TILE_N;
    uint32_t block_col_idx_a = threadIdx.x % TILE_K;
    uint32_t block_row_idx_a = threadIdx.x / TILE_K;

    constexpr int row_offset_A = BLOCK_SIZE / TILE_K;
    constexpr int row_offset_B = BLOCK_SIZE / TILE_N;

    uint32_t row_idx = threadIdx.x / (TILE_N / TN);
    uint32_t col_idx = threadIdx.x % (TILE_N / TN);

    A_data += blockIdx.y * TILE_M * K;
    B_data += blockIdx.x * TILE_N;
    output_data += blockIdx.y * TILE_M * N + blockIdx.x * TILE_N;

    float sums[TM][TN] = {{0.0f}};
    float regs_M[TM] = {0.0f};
    float regs_N[TN] = {0.0f};

    for (int tile_k_idx = 0; tile_k_idx < K; tile_k_idx += TILE_K) {
        #pragma unroll
        for (int i = 0; i < TILE_M; i += row_offset_A) {
            smem_A[i + block_row_idx_a][block_col_idx_a] = A_data[(block_row_idx_a + i) * K + block_col_idx_a];
        }

        #pragma unroll
        for (int i = 0; i < TILE_K; i += row_offset_B) {
            smem_B[i + block_row_idx][block_col_idx] = B_data[(block_row_idx + i) * N + block_col_idx];
        }
        // smem_A[block_row_idx_a][block_col_idx_a] = A_data[block_row_idx_a * K + block_col_idx_a];
        // smem_B[block_row_idx][block_col_idx] = B_data[block_row_idx * N + block_col_idx];
        __syncthreads();

        A_data += TILE_K;
        B_data += TILE_K * N;

        for (int k = 0; k < TILE_K; k++) {

            for (int tm_idx = 0; tm_idx < TM; tm_idx++) {
                regs_M[tm_idx] = smem_A[row_idx * TM + tm_idx][k];
            }
            for (int tn_idx = 0; tn_idx < TN; tn_idx++) {
                regs_N[tn_idx] = smem_B[k][col_idx * TN + tn_idx];
            }

            for (int tm_idx = 0; tm_idx < TM; tm_idx++) {
                for (int tn_idx = 0; tn_idx < TN; tn_idx++) {
                    // sums[tm_idx][tn_idx] += smem_A[row_idx * TM + tm_idx][k] * smem_B[k][col_idx * TN + tn_idx];
                    sums[tm_idx][tn_idx] += regs_M[tm_idx] * regs_N[tn_idx];
                }
            }
        }
        __syncthreads();
    }

    for (int tm_idx = 0; tm_idx < TM; tm_idx++) {
        for (int tn_idx = 0; tn_idx < TN; tn_idx++) {
            uint32_t offset = (row_idx * TM + tm_idx) * N + col_idx * TN + tn_idx;
            output_data[offset] = alpha * sums[tm_idx][tn_idx] + beta * output_data[offset];
        }
    }
}
