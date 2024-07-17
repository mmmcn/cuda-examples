// ncu --launch-skip 2 ./binary/reduce_sum
#include <iostream>
#include <random>
#include <assert.h>

#include "cuda_runtime.h"

#include "../utils/utils.h"


#define NUM_WARMUP 2
#define NUM_REPEAT 3
#define NUM_THREADS 256

template <typename T>
inline void checkResult(T actual, T desired) {
    constexpr float abs_diff = 1e-1;  // smooth check
    float diff = static_cast<float>(actual - desired);
    if (diff > abs_diff || diff < -abs_diff) {
        std::cout << "Check failed, desired is " << static_cast<float>(desired)
                  << " but actual is " << static_cast<float>(actual) << "\n";
    }
}

template <typename T>
inline void checkResult(T* actual, T* desired, int64_t numel){
    constexpr float abs_diff = 1e-1;  // smooth check for low percision dtype
    for (int i = 0; i < numel; i++) {
        float diff = static_cast<float>(actual[i] - desired[i]);
        if (diff > abs_diff || diff < -abs_diff) {
            std::cout << "Check failed, desired is " << static_cast<float>(desired[i])
                      << " but actual is " << static_cast<float>(actual[i]) << "\n";
            break;
        }
    }
}

template <typename T>
T reduceSumNaiveOnCPU(T* d_in, int64_t numel) {
    T sum = static_cast<T>(0.0);
    for (int i = 0; i < numel; i++) {
        sum += d_in[i];
    }
    return sum;
}


template <typename T>
__global__ void reduceSumGlobalMemKernel(T* d_out, T* d_in, int64_t numel) {
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < numel) {
        // for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        //     if (tid < stride && (gid + stride) < numel) {  // we'd better check boundary
        //         d_in[gid] += d_in[gid + stride];
        //     }
        //     __syncthreads();
        // }
        for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
            if (tid % (2 * stride) == 0 && (gid + stride) < numel) {
                d_in[gid] += d_in[gid + stride];
            }
            __syncthreads();
        }
    }

    // thread 0 stores the sum of each block, so write it back to d_out
    if (tid == 0) {
        d_out[blockIdx.x] = d_in[gid];
    }
}

template <typename T>
__global__ void reduceSumSharedMemKernel(T* d_out, T* d_in, int64_t numel) {
    // blockDim.x
    extern __shared__ T shm[];

    uint32_t tid = threadIdx.x;
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < numel) {
        shm[tid] = d_in[gid];
    }
    __syncthreads();

    if (gid < numel) {
        for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
            if (tid % (2 * stride) == 0 && (gid + stride) < numel) {
                shm[tid] += shm[tid + stride];
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        d_out[blockIdx.x] = shm[tid];
    }
}

template <typename T>
__global__ void reduceSumSharedMemLessWarpDivergenceKernel(T* d_out, T* d_in, int64_t numel) {
    // NOTE: this kernel assume that the numel is divisible by blockDim.x
    extern __shared__ T shm[];

    uint32_t tid = threadIdx.x;
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    shm[tid] = d_in[gid];
    __syncthreads();

    if (gid < numel) {
        for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
            uint32_t index = 2 * stride * tid;
            if (index < blockDim.x) {
                shm[index] += shm[index + stride];
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        d_out[blockIdx.x] = shm[tid];
    }
}

template <typename T>
__global__ void reduceSumSharedMemNoBankConflictKernel(T* d_out, T* d_in, int64_t numel) {
    extern __shared__ T shm[];
    uint32_t tid = threadIdx.x;
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    shm[tid] = d_in[gid];
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // when stride <= 16, warp divergence still exists.
        if (tid < stride) {
            shm[tid] += shm[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_out[blockIdx.x] = shm[tid];
    }
}

template <bool IS_LAST = false, typename T>
__global__ void reduceSumAddDuringLoadKernel(T* d_out, T* d_in, int64_t numel) {
    extern __shared__ T shm[];
    uint32_t tid = threadIdx.x;
    uint32_t gid;
    if constexpr (IS_LAST) {
        gid = threadIdx.x + blockIdx.x * blockDim.x;
        shm[tid] = d_in[gid];
    } else {
        gid = threadIdx.x + blockIdx.x * (blockDim.x * 2);
        shm[tid] = d_in[gid] + d_in[gid + blockDim.x];
    }
    
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // when stride <= 16, warp divergence still exists.
        if (tid < stride) {
            shm[tid] += shm[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_out[blockIdx.x] = shm[tid];
    }
}

template <typename T>
__device__ void warpReduce(volatile T* cache, int tid){
    cache[tid] += cache[tid+32];
    cache[tid] += cache[tid+16];
    cache[tid] += cache[tid+8];
    cache[tid] += cache[tid+4];
    cache[tid] += cache[tid+2];
    cache[tid] += cache[tid+1];
}

template <bool IS_LAST = false, typename T>
__global__ void reduceSumAddDuringLoadOptKernel(T* d_out, T* d_in, int64_t numel) {
    extern __shared__ T shm[];
    uint32_t tid = threadIdx.x;
    uint32_t gid;
    if constexpr (IS_LAST) {
        gid = threadIdx.x + blockIdx.x * blockDim.x;
        shm[tid] = d_in[gid];
    } else {
        gid = threadIdx.x + blockIdx.x * (blockDim.x * 2);
        shm[tid] = d_in[gid] + d_in[gid + blockDim.x];
    }
    
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shm[tid] += shm[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) warpReduce(shm, tid);
    if (tid == 0) {
        d_out[blockIdx.x] = shm[tid];
    }
}

template <typename T>
inline void randomInit(T* data, int64_t numel) {
    std::mt19937 gen_f(114514);
    std::uniform_real_distribution<> dis_f(-1, 1);
    for (int i = 0; i < numel; i++) {
        data[i] = static_cast<T>(dis_f(gen_f));
    }
}

template <typename T, int64_t NUMEL>
void runReduceSumV1() {
    constexpr uint32_t num_threads = NUM_THREADS;
    constexpr uint32_t num_blocks = (NUMEL + num_threads - 1) / num_threads;
    static_assert(NUMEL % num_threads == 0, "For current implementation, the numel should be divisible by num_threads");

    T* host_in = (T*)malloc(sizeof(T) * NUMEL);
    T* host_golden_result = (T*)malloc(sizeof(T) * num_blocks);
    T* host_actual_result = (T*)malloc(sizeof(T) * num_blocks);

    randomInit(host_in, NUMEL);
    for (int i = 0; i < num_blocks; i++) {
        T sum = static_cast<T>(0.0);
        for (int j = 0; j < num_threads; j++) {
            sum += host_in[i * num_threads + j];
        }
        host_golden_result[i] = sum;
    }

    T* device_in;
    T* device_result;
    CHECK_CUDA(cudaMalloc((void **)&device_in, sizeof(T) * NUMEL));
    CHECK_CUDA(cudaMalloc((void **)&device_result, num_blocks * sizeof(T)));

    for (int i = 0; i < NUM_REPEAT; i++) {
        CHECK_CUDA(cudaMemcpy(device_in, host_in, sizeof(T) * NUMEL, cudaMemcpyHostToDevice));
        reduceSumGlobalMemKernel<<<num_blocks, num_threads>>>(device_result, device_in, NUMEL);
    }
    CHECK_CUDA(cudaMemcpy(host_actual_result, device_result, num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
    checkResult(host_actual_result, host_golden_result, num_blocks);

    free(host_in);
    free(host_actual_result);
    free(host_golden_result);
    cudaFree(device_in);
    cudaFree(device_result);
}

template <typename T, int64_t NUMEL>
void runReduceSumV2() {
    constexpr uint32_t num_threads = NUM_THREADS;
    constexpr uint32_t num_blocks = (NUMEL + num_threads - 1) / num_threads;
    static_assert(NUMEL % num_threads == 0, "For current implementation, the numel should be divisible by num_threads");

    T* host_in = (T*)malloc(sizeof(T) * NUMEL);
    T* host_golden_result = (T*)malloc(sizeof(T) * num_blocks);
    T* host_actual_result = (T*)malloc(sizeof(T) * num_blocks);

    randomInit(host_in, NUMEL);
    for (int i = 0; i < num_blocks; i++) {
        T sum = static_cast<T>(0.0);
        for (int j = 0; j < num_threads; j++) {
            sum += host_in[i * num_threads + j];
        }
        host_golden_result[i] = sum;
    }

    T* device_in;
    T* device_result;
    CHECK_CUDA(cudaMalloc((void **)&device_in, sizeof(T) * NUMEL));
    CHECK_CUDA(cudaMalloc((void **)&device_result, num_blocks * sizeof(T)));
    uint32_t shm_size = sizeof(T) * num_threads;

    for (int i = 0; i < NUM_REPEAT; i++) {
        CHECK_CUDA(cudaMemcpy(device_in, host_in, sizeof(T) * NUMEL, cudaMemcpyHostToDevice));
        reduceSumSharedMemKernel<<<num_blocks, num_threads, shm_size>>>(device_result, device_in, NUMEL);
    }
    CHECK_CUDA(cudaMemcpy(host_actual_result, device_result, num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
    checkResult(host_actual_result, host_golden_result, num_blocks);

    free(host_in);
    free(host_actual_result);
    free(host_golden_result);
    cudaFree(device_in);
    cudaFree(device_result);
}

template <typename T, int64_t NUMEL>
void runReduceSumV3() {
    constexpr uint32_t num_threads = NUM_THREADS;
    constexpr uint32_t num_blocks = (NUMEL + num_threads - 1) / num_threads;
    static_assert(NUMEL % num_threads == 0, "For current implementation, the numel should be divisible by num_threads");

    T* host_in = (T*)malloc(sizeof(T) * NUMEL);
    T* host_golden_result = (T*)malloc(sizeof(T) * num_blocks);
    T* host_actual_result = (T*)malloc(sizeof(T) * num_blocks);

    randomInit(host_in, NUMEL);
    for (int i = 0; i < num_blocks; i++) {
        T sum = static_cast<T>(0.0);
        for (int j = 0; j < num_threads; j++) {
            sum += host_in[i * num_threads + j];
        }
        host_golden_result[i] = sum;
    }

    T* device_in;
    T* device_result;
    CHECK_CUDA(cudaMalloc((void **)&device_in, sizeof(T) * NUMEL));
    CHECK_CUDA(cudaMalloc((void **)&device_result, num_blocks * sizeof(T)));
    uint32_t shm_size = sizeof(T) * num_threads;

    for (int i = 0; i < NUM_REPEAT; i++) {
        CHECK_CUDA(cudaMemcpy(device_in, host_in, sizeof(T) * NUMEL, cudaMemcpyHostToDevice));
        reduceSumSharedMemLessWarpDivergenceKernel<<<num_blocks, num_threads, shm_size>>>(device_result, device_in, NUMEL);
    }
    CHECK_CUDA(cudaMemcpy(host_actual_result, device_result, num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
    checkResult(host_actual_result, host_golden_result, num_blocks);

    free(host_in);
    free(host_actual_result);
    free(host_golden_result);
    cudaFree(device_in);
    cudaFree(device_result);
}

template <typename T, int64_t NUMEL>
void runReduceSumV4() {
    constexpr uint32_t num_threads = NUM_THREADS;
    constexpr uint32_t num_blocks = (NUMEL + num_threads - 1) / num_threads;
    static_assert(NUMEL % num_threads == 0, "For current implementation, the numel should be divisible by num_threads");

    T* host_in = (T*)malloc(sizeof(T) * NUMEL);
    T* host_golden_result = (T*)malloc(sizeof(T) * num_blocks);
    T* host_actual_result = (T*)malloc(sizeof(T) * num_blocks);

    randomInit(host_in, NUMEL);
    for (int i = 0; i < num_blocks; i++) {
        T sum = static_cast<T>(0.0);
        for (int j = 0; j < num_threads; j++) {
            sum += host_in[i * num_threads + j];
        }
        host_golden_result[i] = sum;
    }

    T* device_in;
    T* device_result;
    CHECK_CUDA(cudaMalloc((void **)&device_in, sizeof(T) * NUMEL));
    CHECK_CUDA(cudaMalloc((void **)&device_result, num_blocks * sizeof(T)));
    uint32_t shm_size = sizeof(T) * num_threads;

    for (int i = 0; i < NUM_REPEAT; i++) {
        CHECK_CUDA(cudaMemcpy(device_in, host_in, sizeof(T) * NUMEL, cudaMemcpyHostToDevice));
        reduceSumSharedMemNoBankConflictKernel<<<num_blocks, num_threads, shm_size>>>(device_result, device_in, NUMEL);
    }
    CHECK_CUDA(cudaMemcpy(host_actual_result, device_result, num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
    checkResult(host_actual_result, host_golden_result, num_blocks);

    free(host_in);
    free(host_actual_result);
    free(host_golden_result);
    cudaFree(device_in);
    cudaFree(device_result);
}

template <typename T, int64_t NUMEL>
void runReduceSumV5() {
    constexpr uint32_t num_threads = NUM_THREADS;
    constexpr uint32_t num_blocks = (NUMEL + 2 * num_threads - 1) / (2 * num_threads);
    static_assert(NUMEL % num_threads == 0, "For current implementation, the numel should be divisible by num_threads");

    T* host_in = (T*)malloc(sizeof(T) * NUMEL);
    T* host_golden_result = (T*)malloc(sizeof(T) * num_blocks);
    T* host_actual_result = (T*)malloc(sizeof(T) * num_blocks);

    randomInit(host_in, NUMEL);
    for (int i = 0; i < num_blocks; i++) {
        T sum = static_cast<T>(0.0);
        for (int j = 0; j < num_threads * 2; j++) {
            sum += host_in[i * num_threads * 2 + j];
        }
        host_golden_result[i] = sum;
    }

    T* device_in;
    T* device_result;
    CHECK_CUDA(cudaMalloc((void **)&device_in, sizeof(T) * NUMEL));
    CHECK_CUDA(cudaMalloc((void **)&device_result, num_blocks * sizeof(T)));
    uint32_t shm_size = sizeof(T) * num_threads;

    for (int i = 0; i < NUM_REPEAT; i++) {
        CHECK_CUDA(cudaMemcpy(device_in, host_in, sizeof(T) * NUMEL, cudaMemcpyHostToDevice));
        reduceSumAddDuringLoadKernel<<<num_blocks, num_threads, shm_size>>>(device_result, device_in, NUMEL);
    }
    CHECK_CUDA(cudaMemcpy(host_actual_result, device_result, num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
    checkResult(host_actual_result, host_golden_result, num_blocks);

    free(host_in);
    free(host_actual_result);
    free(host_golden_result);
    cudaFree(device_in);
    cudaFree(device_result);
}

template <typename T, int64_t NUMEL>
void runReduceSumV6() {
    constexpr uint32_t num_threads = NUM_THREADS;
    constexpr uint32_t num_blocks = (NUMEL + 2 * num_threads - 1) / (2 * num_threads);
    static_assert(NUMEL % num_threads == 0, "For current implementation, the numel should be divisible by num_threads");

    T* host_in = (T*)malloc(sizeof(T) * NUMEL);
    T* host_golden_result = (T*)malloc(sizeof(T) * num_blocks);
    T* host_actual_result = (T*)malloc(sizeof(T) * num_blocks);

    randomInit(host_in, NUMEL);
    for (int i = 0; i < num_blocks; i++) {
        T sum = static_cast<T>(0.0);
        for (int j = 0; j < num_threads * 2; j++) {
            sum += host_in[i * num_threads * 2 + j];
        }
        host_golden_result[i] = sum;
    }

    T* device_in;
    T* device_result;
    CHECK_CUDA(cudaMalloc((void **)&device_in, sizeof(T) * NUMEL));
    CHECK_CUDA(cudaMalloc((void **)&device_result, num_blocks * sizeof(T)));
    uint32_t shm_size = sizeof(T) * num_threads;

    for (int i = 0; i < NUM_REPEAT; i++) {
        CHECK_CUDA(cudaMemcpy(device_in, host_in, sizeof(T) * NUMEL, cudaMemcpyHostToDevice));
        reduceSumAddDuringLoadOptKernel<<<num_blocks, num_threads, shm_size>>>(device_result, device_in, NUMEL);
    }
    CHECK_CUDA(cudaMemcpy(host_actual_result, device_result, num_blocks * sizeof(T), cudaMemcpyDeviceToHost));
    checkResult(host_actual_result, host_golden_result, num_blocks);

    free(host_in);
    free(host_actual_result);
    free(host_golden_result);
    cudaFree(device_in);
    cudaFree(device_result);
}

int main(int argc, char* argv[]) {
    constexpr int64_t numel = 32 * 1024 * 1024;
    using scalar_t = float;

    if (argc == 1) {
        runReduceSumV1<scalar_t, numel>();
    } else {
        assert(argc == 2);
        switch (std::atoi(argv[1])) {
            case 1:
                runReduceSumV1<scalar_t, numel>();  // baseline, global memory
                break;
            case 2:
                runReduceSumV2<scalar_t, numel>();  // add shared memory, no additional optimization
                break;
            case 3:
                runReduceSumV3<scalar_t, numel>();  // reduce warp divergence
                break;
            case 4:
                runReduceSumV4<scalar_t, numel>();  // avoid bank conflict
                break;
            case 5:
                runReduceSumV5<scalar_t, numel>();  // avoid idle threads, compute once during load into shared mem
                break;
            case 6:
                runReduceSumV6<scalar_t, numel>();  // avoid explict synchronization in warp0
                break;
                
        }
    }
    return 0;
}
