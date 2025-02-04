#include <iostream>
#include <vector>
#include <optional>
#include <algorithm>
#include <functional>
#include <assert.h>

#include "cuda_runtime.h"

#include "kernels/sgemm_naive.cuh"
#include "kernels/sgemm_v2.cuh"
#include "../utils/utils.h"

#define M 2048
#define N 2048
#define K 2048


// for profiling
#define NUM_WARMUP 2
#define NUM_REPEAT 2


void run_sgemm_cpu(float* A, float* B, float* C, int m, int n, int k, float alpha=1.0, float beta=0.0) {
    (void)alpha;
    (void)beta;
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float tmp_a = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += tmp_a * B[k * N + j];
            }
        }
    }
}


inline float gemm_tflops(int m, int n, int k, float time_ms) {
    return (2.0f * m * n * k) / (time_ms * 1e9);
}


void perf(const std::function<void()>& f, const char* caller) {
    for (int i = 0; i < NUM_WARMUP; i++) {
        f();
    }

    GPUTimer timer;
    std::vector<float> cost_times(NUM_REPEAT);
    for (int i = 0; i < NUM_REPEAT; i++) {
        // beta always set to 0.0, so do not need to reset matrix C here.
        timer.start();
        f();
        timer.stop();
        cost_times[i] = timer.elapsed();
    }

    std::cout << caller << " 's tflops is " << gemm_tflops(M, N, K, std::accumulate(cost_times.cbegin(), cost_times.cend(), 0.0f) / (1.0 * NUM_REPEAT)) << std::endl;
}


void run_sgemm_naive(float* A, float* B, float* C, int m, int n, int k, float alpha=1.0, float beta=0.0) {
    dim3 block = {32, 32, 1};
    dim3 grid = {ceil_div((uint32_t)m, block.y), ceil_div((uint32_t)n, block.x), 1};

    perf([&]() { sgemm_naive_kernel<<<grid, block>>>(C, A, B, m, n, k, alpha, beta); }, "sgemm_naive_kernel");
}


void run_sgemm_v2(float* A, float* B, float* C, int m, int n, int k, float alpha=1.0, float beta=0.0) {
    constexpr int BLOCK_SIZE = 32;
    dim3 block = {32, 32, 1};
    dim3 grid = {ceil_div((uint32_t)m, block.y), ceil_div((uint32_t)n, block.x), 1};

    perf([&]() { sgemm_v2_kernel<BLOCK_SIZE><<<grid, block>>>(C, A, B, m, n, k, alpha, beta); }, "sgemm_v2_kernel");
}


inline bool check_result(float* actual, float* desired, int64_t numel){
    constexpr float abs_diff = 1e-2;
    bool passed = true;
    for (int i = 0; i < numel; i++) {
        float diff = static_cast<float>(actual[i] - desired[i]);
        if (diff > abs_diff || diff < -abs_diff) {
            std::cout << "Check failed, desired is " << static_cast<float>(desired[i])
                      << " but actual is " << static_cast<float>(actual[i]) << "\n";
            passed = false;
            break;
        }
    }
    return passed;
}


int main(int argc, char* argv[]) {
    assert(argc == 2);
    float* host_A = (float*)malloc(sizeof(float) * M * K);
    float* host_B = (float*)malloc(sizeof(float) * N * K);
    float* host_C = (float*)malloc(sizeof(float) * M * N);

    randomInitFloat<float>(host_A, M * K);
    randomInitFloat<float>(host_B, N * K);

    float* device_A;
    float* device_B;
    float* device_C;
    CHECK_CUDA(cudaMalloc(&device_A, sizeof(float) * M * K));
    CHECK_CUDA(cudaMalloc(&device_B, sizeof(float) * N * K));
    CHECK_CUDA(cudaMalloc(&device_C, sizeof(float) * M * N));
    CHECK_CUDA(cudaMemcpy((void*)device_A, (void*)host_A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy((void*)device_B, (void*)host_B, sizeof(float) * N * K, cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy((void*)device_C, (void*)host_C, sizeof(float) * M * N, cudaMemcpyHostToDevice));


    switch (std::atoi(argv[1])) {
        case 1:
            // naive sgemm kernel
            run_sgemm_naive(device_A, device_B, device_C, M, N, K);
            break;
        case 2:
            run_sgemm_v2(device_A, device_B, device_C, M, N, K);
            break;
        default:
            throw std::invalid_argument("invalid kernel number");
    }

    // check the correctness of our kernel
    run_sgemm_cpu(host_A, host_B, host_C, M, N, K);
    float* host_result = (float*)malloc(sizeof(float) * M * N);
    CHECK_CUDA(cudaMemcpy(host_result, device_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    if (check_result(host_result, host_C, M * N)) {
        std::cout << "Precision check passed" << std::endl;
    }


    free(host_A);
    free(host_B);
    free(host_C);
    CHECK_CUDA(cudaFree(device_A));
    CHECK_CUDA(cudaFree(device_B));
    CHECK_CUDA(cudaFree(device_C));

    free(host_result);

    return 0;
}
