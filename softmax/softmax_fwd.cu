#include <iostream>
#include <assert.h>
#include <cmath>
#include "cuda_runtime.h"
#include <chrono>

#include "../utils/utils.h"

// 512M
#define N 128 * 1024
#define C 1024
#define BLOCK_SIZE 256
#define NUM_REPEAT 4

// assume the input memory format always contiguous, and inner size is 1

inline void checkResult(const float* actual, const float* desired, int64_t numel){
    constexpr float abs_diff = 1e-5;
    for (int i = 0; i < numel; i++) {
        float diff = static_cast<float>(actual[i] - desired[i]);
        if (diff > abs_diff || diff < -abs_diff) {
            std::cout << "Check failed, desired is " << static_cast<float>(desired[i])
                      << " but actual is " << static_cast<float>(actual[i]) << "\n";
            return;
        }
    }
    std::cout << "Check passed\n";
}

void softmaxForwardCPU(float* output, const float* input, int n, int c) {
    for (int i = 0; i < n; i++) {
        const float* cur_input = input + i * c;
        float* cur_output = output + i * c;

        float maxval = -INFINITY;
        for (int j = 0; j < c; j++) {
            if (cur_input[j] > maxval) {
                maxval = cur_input[j];
            } 
        }

        float sum = 0.0;
        for (int j = 0; j < c; j++) {
            cur_output[j] = expf(cur_input[j] - maxval);
            sum += cur_output[j];
        }

        float rsum = 1.0f / sum;
        for (int j = 0; j < c; j++) {
            cur_output[j] *= rsum;
        }
    }
}


// "Online normalizer calculation for softmax": https://arxiv.org/abs/1805.02867
void softmaxForwardOnlineCPU(float* output, const float* input, int n, int c) {
    for (int i = 0; i < n; i++) {
        float* cur_output = output + i * c;
        const float* cur_input = input + i * c;

        float maxval = -INFINITY;
        float sum = 0.0f;

        for (int j = 0; j < c; j++) {
            float maxval_pre = maxval;
            if (cur_input[j] > maxval) {
                maxval = cur_input[j];
                sum = sum * expf(maxval_pre - maxval) + expf(cur_input[j] - maxval);
            } else {
                sum += expf(cur_input[j] - maxval);
            }
        }

        for (int j = 0; j < c; j++) {
            cur_output[j] = expf(cur_input[j] - maxval) / sum;
        }
    }
}

__global__ void softmaxForwardKernel1(float* output_data, const float* input_data, int n, int c) {
    // each thread process one row
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float* cur_output_data = output_data + tid * c;
        const float* cur_input_data = input_data + tid * c;

        float maxval = -INFINITY;
        for (int i = 0; i < c; i++) {
            if (cur_input_data[i] > maxval) {
                maxval = cur_input_data[i];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < c; i++) {
            cur_output_data[i] = expf(cur_input_data[i] - maxval);
            sum += cur_output_data[i];
        }
        for (int i = 0; i < c; i++) {
            cur_output_data[i] /= sum;
        }
    }
}

__global__ void softmaxForwardKernel2(float* output_data, const float* input_data, int n, int c) {
    // parallelizes over n and c
    // block level reduce
    extern __shared__ float shm[];
    uint32_t bid = blockIdx.x;
    uint32_t tid = threadIdx.x;
    
    float maxval = -INFINITY;
    for (int i = tid; i < c; i += blockDim.x) {
        maxval = fmaxf(maxval, input_data[bid * c + i]);
    }
    shm[tid] = maxval;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset >= 1; offset >>= 1) {
        if (tid < offset) {
            shm[tid] = fmaxf(shm[tid], shm[tid + offset]);
        }
        __syncthreads();
    }
    maxval = shm[0];
    float sum = 0.0f;
    for (int i = tid; i < c; i += blockDim.x) {
        output_data[bid * c + i] = expf(input_data[bid * c + i] - maxval);
        sum += output_data[bid * c + i];
    }
    shm[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset >= 1; offset >>= 1) {
        if (tid < offset) {
            shm[tid] += shm[tid + offset];
        }
        __syncthreads();
    }
    sum = shm[0];
    for (int i = tid; i < c; i += blockDim.x) {
        output_data[bid * c + i] /= sum;
    }
}

template <typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template <typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

// warp-level reduction
template <template <typename> class ReduceOp>
__device__ float warpReduce(float val) {
    ReduceOp<float> r;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float b = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = r(val, b);
    }
    return val;
}

__global__ void softmaxForwardKernel3(float* output_data, const float* input_data, int n, int c) {
    // warp level reduce
    uint32_t bid = blockIdx.x;
    uint32_t tid = threadIdx.x;

    float maxval = -INFINITY;
    // calculate max
    for (int i = tid; i < c; i += blockDim.x) {
        maxval = fmaxf(maxval, input_data[bid * c + i]);
    }
    // we're using __shfl_xor_sync, so no need to broadcast maxval
    maxval = warpReduce<Max>(maxval);

    for (int i = tid; i < c; i += blockDim.x) {
        output_data[bid * c + i] = expf(input_data[bid * c + i] - maxval);
    }

    // calculate sum
    float sum = 0.0f;
    for (int i = tid; i < c; i += blockDim.x) {
        sum += output_data[bid * c + i];
    }
    sum = warpReduce<Add>(sum);

    for (int i = tid; i < c; i += blockDim.x) {
        output_data[bid * c + i] /= sum;
    }
}

__global__ void softmaxForwardKernel4(float* output_data, const float* input_data, int n, int c) {
    // warp level reduce
    // different from Kernel3, we may have more than one warps in each block, each warp process one row
    uint32_t bid = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t tid = threadIdx.x;

    float maxval = -INFINITY;
    // calculate max
    for (int i = tid; i < c; i += blockDim.x) {
        maxval = fmaxf(maxval, input_data[bid * c + i]);
    }
    // we're using __shfl_xor_sync, so no need to broadcast maxval 
    maxval = warpReduce<Max>(maxval);

    for (int i = tid; i < c; i += blockDim.x) {
        output_data[bid * c + i] = expf(input_data[bid * c + i] - maxval);
    }

    // calculate sum
    float sum = 0.0f;
    for (int i = tid; i < c; i += blockDim.x) {
        sum += output_data[bid * c + i];
    }
    sum = warpReduce<Add>(sum);

    for (int i = tid; i < c; i += blockDim.x) {
        output_data[bid * c + i] /= sum;
    }
}

__global__ void softmaxForwardKernel5(float* output_data, const float* input_data, int n, int c) {
    // warp level reduce, using online softmax
    uint32_t bid = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t tid = threadIdx.x;

    float maxval = -INFINITY;
    float sumval = 0.0f;
    // calculate max and sum, assign to each Lane
    for (int i = tid; i < c; i += blockDim.x) {
        if (input_data[bid * c + i] > maxval) {
            float pre_maxval = maxval;
            maxval = input_data[bid * c + i];
            sumval = sumval * expf(pre_maxval - maxval) + 1.0f;
        } else {
            sumval += expf(input_data[bid * c + i] - maxval);
        }
    }

    // now, do warp reduce
    float offset_sum;
    float offset_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        // __shfl_xor_sync(0xFFFFFFFF, val, offset);
       offset_max =  __shfl_xor_sync(0xFFFFFFFF, maxval, offset);
       offset_sum = __shfl_xor_sync(0xFFFFFFFF, sumval, offset);

       if (offset_max > maxval) {
          sumval *= expf(maxval - offset_max);
          maxval = offset_max;
       } else {
          offset_sum *= expf(offset_max - maxval);
       }
       sumval += offset_sum;
    }

    // no need to broadcast
    for (int i = tid; i < c; i += blockDim.x) {
        output_data[bid * c + i] = expf(input_data[bid * c + i] - maxval) / sumval;
    }
}

__global__ void softmaxForwardKernel6(float* output_data, const float* input_data, int n, int c) {
    // this kernel should be more efficient when c is larger
    extern __shared__ float shm[];
    uint32_t bid = blockIdx.x;
    uint32_t tid = threadIdx.x;

    uint32_t warpId = tid / 32;
    uint32_t laneId = tid % 32;

    float maxval = -INFINITY;
    for (int i = tid; i < c; i += blockDim.x) {
        maxval = fmaxf(maxval, input_data[bid * c + tid]);
    }
    // reduce-max in each warp
    maxval = warpReduce<Max>(maxval);

    if (laneId == 0) {
        shm[warpId] = maxval;
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < blockDim.x / 32; i++) {
            maxval = fmaxf(maxval, shm[i]);
        }
        shm[0] = maxval;
    }
    __syncthreads();

    // broadcast maxval to all threads
    float maxval_bd = shm[0];

    for (int i = tid; i < c; i += blockDim.x) {
        output_data[bid * c + i] = expf(input_data[bid * c + i] - maxval_bd);
    }

    float sum = 0.0f;
    for (int i = tid; i < c; i += blockDim.x) {
        sum += output_data[bid * c + i];
    }
    sum = warpReduce<Add>(sum);

    if (laneId == 0) {
        shm[warpId] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        float val = shm[0];
        for (int i = 1; i < blockDim.x / 32; i++) {
            val += shm[i];
        }
        shm[0] = val;
    }
    __syncthreads();

    // broadcast global sum into all threads
    float sum_bd = shm[0];

    for (int i = tid; i < c; i += blockDim.x) {
        output_data[bid * c + i] /= sum_bd;
    }
}

void benchmark_cpu() {
    // naive version vs online version
    float* input_data = (float*)malloc(N * C * sizeof(float));
    float* output_data1 = (float*)malloc(N * C * sizeof(float));
    float* output_data2 = (float*)malloc(N * C * sizeof(float));
    randomInitFloat(input_data, N * C, -2, 2);

    constexpr int repeat = NUM_REPEAT;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) softmaxForwardCPU(output_data1, input_data, N, C);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "naive version cost " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1.f * repeat) << " us\n";

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) softmaxForwardOnlineCPU(output_data2, input_data, N, C);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "online version cost " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1.f * repeat) << " us\n";

    checkResult(output_data1, output_data2, N * C);
    
    free(input_data);
    free(output_data1);
    free(output_data2);
}

void run_softmax_kernel1(float* output, const float* input) {
    const int block_num = ceil_div(N, BLOCK_SIZE);
    for (int i = 0; i < NUM_REPEAT; i++) {
        softmaxForwardKernel1<<<block_num, BLOCK_SIZE>>>(output, input, N, C);
    }
}

void run_softmax_kernel2(float* output, const float* input) {
    for (int i = 0; i < NUM_REPEAT; i++) {
        softmaxForwardKernel2<<<N, BLOCK_SIZE, sizeof(float) * BLOCK_SIZE>>>(output, input, N, C);
    }
}

void run_softmax_kernel3(float* output, const float* input) {
    constexpr int block_size = 32;  // we use warp reduce in this kernel, so block_size should be 32
    for (int i = 0; i < NUM_REPEAT; i++) {
        softmaxForwardKernel3<<<N, block_size>>>(output, input, N, C);
    }
}

void run_softmax_kernel4(float* output, const float* input) {
    constexpr int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;

    dim3 block(32, warps_per_block, 1);
    dim3 grid(ceil_div(N, warps_per_block), 1, 1);
    for (int i = 0; i < NUM_REPEAT; i++) {
        softmaxForwardKernel4<<<grid, block>>>(output, input, N, C);
    }
}

void run_softmax_kernel5(float* output, const float* input) {
    constexpr int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;

    dim3 block(32, warps_per_block, 1);
    dim3 grid(ceil_div(N, warps_per_block), 1, 1);
    for (int i = 0; i < NUM_REPEAT; i++) {
        softmaxForwardKernel5<<<grid, block>>>(output, input, N, C);
    }
}

void run_softmax_kernel6(float* output, const float* input) {
    constexpr int block_size = 256;
    const int shm_size = block_size / 32 * sizeof(float);
    for (int i = 0; i < NUM_REPEAT; i++) {
        softmaxForwardKernel6<<<N, block_size, shm_size>>>(output, input, N, C);
    }
}

int main(int argc, char* argv[]) {
    if (argc == 1) {
        benchmark_cpu();
    } else {
        assert(argc == 2);
        float* host_input_data = (float*)malloc(N * C * sizeof(float));
        float* output_data_golden = (float*)malloc(N * C * sizeof(float));
        float* output_data_result = (float*)malloc(N * C * sizeof(float));
        randomInitFloat(host_input_data, N * C, -2, 2);

        float* device_input_data;
        float* device_output_data;
        CHECK_CUDA(cudaMalloc((void**)&device_input_data, N * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&device_output_data, N * C * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(device_input_data, host_input_data, N * C * sizeof(float), cudaMemcpyHostToDevice));

        // run on cpu
        softmaxForwardCPU(output_data_golden, host_input_data, N, C);
        switch (std::atoi(argv[1])) {
            case 1:
                run_softmax_kernel1(device_output_data, device_input_data);
                break;
            case 2:
                run_softmax_kernel2(device_output_data, device_input_data);
                break;
            case 3:
                run_softmax_kernel3(device_output_data, device_input_data);
                break;
            case 4:
                run_softmax_kernel4(device_output_data, device_input_data);
                break;
            case 5:
                run_softmax_kernel5(device_output_data, device_input_data);
                break;
            case 6:
                run_softmax_kernel6(device_output_data, device_input_data);
                break;
        }
        // check result
        CHECK_CUDA(cudaMemcpy(output_data_result, device_output_data, N * C * sizeof(float), cudaMemcpyDeviceToHost));
        checkResult(output_data_golden, output_data_result, N * C);

        free(host_input_data);
        free(output_data_golden);
        free(output_data_result);
        CHECK_CUDA(cudaFree(device_input_data));
        CHECK_CUDA(cudaFree(device_output_data));
    }
    return 0;
}
