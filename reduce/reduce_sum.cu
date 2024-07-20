// ncu --launch-skip 2 ./binary/reduce_sum
#include <iostream>
#include <random>
#include <assert.h>

#include "cuda_runtime.h"

#include "../utils/utils.h"


#define NUM_WARMUP 2
#define NUM_REPEAT 3
#define NUM_THREADS 256
#define WARP_SIZE 32

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

template <typename T, uint32_t BLOCK_SIZE>
__device__ void warpReduceUnroll(volatile T* cache, uint32_t tid) {
    if constexpr (BLOCK_SIZE >= 64) {
        cache[tid] += cache[tid + 32];
    }
    if constexpr (BLOCK_SIZE >= 32) {
        cache[tid] += cache[tid + 16];
    }
    if constexpr (BLOCK_SIZE >= 16) {
        cache[tid] += cache[tid + 8];
    }
    if constexpr (BLOCK_SIZE >= 8) {
        cache[tid] += cache[tid + 4];
    }
    if constexpr (BLOCK_SIZE >= 4) {
        cache[tid] += cache[tid + 2];
    }
    if constexpr (BLOCK_SIZE >= 2) {
        cache[tid] += cache[tid + 1];
    }
}

template <typename T>
__device__ void warpReduce(volatile T* cache, uint32_t tid) {
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}

template <bool IS_LAST = false, typename T>
__global__ void reduceSumAddDuringLoadWarpReduceKernel(T* d_out, T* d_in, int64_t numel) {
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

template <uint32_t BLOCK_SIZE = 256, bool IS_LAST = false, typename T>
__global__ void reduceSumAddDuringLoadWarpReduceUnrollLoopKernel(T* d_out, T* d_in, int64_t numel) {
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

    if constexpr (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            shm[tid] += shm[tid + 256];
        }
        __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            shm[tid] += shm[tid + 128];
        }
        __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            shm[tid] += shm[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) warpReduceUnroll<T, BLOCK_SIZE>(shm, tid);
    if (tid == 0) {
        d_out[blockIdx.x] = shm[tid];
    }
}

template <uint32_t BLOCK_SIZE, uint32_t NUM_PER_THREAD, typename T>
__global__ void reduceSumMultiAddKernel(T* d_out, T* d_in, int64_t numel) {
    extern __shared__ T shm[];
    uint32_t tid = threadIdx.x;
    uint32_t gid = threadIdx.x + blockIdx.x * (BLOCK_SIZE * NUM_PER_THREAD);

    // each threads load and accumulate NUM_PER_THREAD element from global mem to shared mem
    shm[tid] = 0;
    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        shm[tid] += d_in[i * BLOCK_SIZE + gid];
    }
    __syncthreads();

    
    // BLOCK_SIZE wont greater than 512 actually
    if constexpr (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            shm[tid] += shm[tid + 256];
        }
        __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            shm[tid] += shm[tid + 128];
        }
        __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            shm[tid] += shm[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduceUnroll<T, BLOCK_SIZE>(shm, tid);
    }
    if (tid == 0) {
        d_out[blockIdx.x] = shm[tid];
    }
}


template <typename T, uint32_t BLOCK_SIZE>
__device__ T warpReduceSHFL(T val) {
    // the BLOCK_SIZE alway even number, do not warry
    if constexpr (BLOCK_SIZE >= 32) {
        val += __shfl_down_sync(0xffffffff, val, 16);
    }
    if constexpr (BLOCK_SIZE >= 16) {
       val += __shfl_down_sync(0xffffffff, val, 8);
    }
    if constexpr (BLOCK_SIZE >= 8) {
        val += __shfl_down_sync(0xffffffff, val, 4);
    }
    if constexpr (BLOCK_SIZE >= 4) {
        val += __shfl_down_sync(0xffffffff, val, 2);
    }
    if constexpr (BLOCK_SIZE >= 2) {
        val += __shfl_down_sync(0xffffffff, val, 1);
    }
    return val;
}

template <uint32_t BLOCK_SIZE, uint32_t NUM_PER_THREAD, typename T>
__global__ void blockReduceSumKernel(T* d_out, T* d_in, int64_t numel) {
    T sum = static_cast<T>(0.0);
    uint32_t tid = threadIdx.x;
    uint32_t gid = threadIdx.x + blockIdx.x * (BLOCK_SIZE * NUM_PER_THREAD);

    // each threads load and accumulate NUM_PER_THREAD element from global mem to register
    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        sum += d_in[i * BLOCK_SIZE + gid];
    }
    
    static __shared__ T shm[WARP_SIZE];  // warp level shared memory
    uint32_t lid = tid % WARP_SIZE;      // Lane index
    uint32_t wid = tid / WARP_SIZE;      // Warp index

    sum = warpReduceSHFL<T, BLOCK_SIZE>(sum);

    if (lid == 0) {
        shm[wid] = sum;
    }
    __syncthreads();

    // read from shared memory only if that warp exists
    sum = (tid < BLOCK_SIZE / WARP_SIZE) ? shm[lid] : static_cast<T>(0.0);
    // do last reduce using 1-th warp
    if (wid == 0) {
        sum = warpReduceSHFL<T, BLOCK_SIZE / WARP_SIZE>(sum);
    }
    if (tid == 0) {
        d_out[blockIdx.x] = sum;
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
        reduceSumAddDuringLoadWarpReduceKernel<<<num_blocks, num_threads, shm_size>>>(device_result, device_in, NUMEL);
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
void runReduceSumV7() {
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
        reduceSumAddDuringLoadWarpReduceUnrollLoopKernel<<<num_blocks, num_threads, shm_size>>>(device_result, device_in, NUMEL);
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
void runReduceSumV8() {
    constexpr uint32_t num_threads = NUM_THREADS;
    constexpr uint32_t num_blocks = 1024;
    constexpr uint32_t num_per_block = NUMEL / num_blocks;
    constexpr uint32_t num_per_thread = num_per_block / num_threads;
    static_assert(NUMEL % num_threads == 0, "For current implementation, the numel should be divisible by num_threads");

    T* host_in = (T*)malloc(sizeof(T) * NUMEL);
    T* host_golden_result = (T*)malloc(sizeof(T) * num_blocks);
    T* host_actual_result = (T*)malloc(sizeof(T) * num_blocks);

    randomInit(host_in, NUMEL);
    for (int i = 0; i < num_blocks; i++) {
        T sum = static_cast<T>(0.0);
        for (int j = 0; j < num_per_block; j++) {
            sum += host_in[i * num_per_block + j];
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
        reduceSumMultiAddKernel<num_threads, num_per_thread><<<num_blocks, num_threads, shm_size>>>(device_result, device_in, NUMEL);
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
void runReduceSumV9() {
    constexpr uint32_t num_threads = NUM_THREADS;
    constexpr uint32_t num_blocks = 1024;
    constexpr uint32_t num_per_block = NUMEL / num_blocks;
    constexpr uint32_t num_per_thread = num_per_block / num_threads;
    static_assert(NUMEL % num_threads == 0, "For current implementation, the numel should be divisible by num_threads");

    T* host_in = (T*)malloc(sizeof(T) * NUMEL);
    T* host_golden_result = (T*)malloc(sizeof(T) * num_blocks);
    T* host_actual_result = (T*)malloc(sizeof(T) * num_blocks);

    randomInit(host_in, NUMEL);
    for (int i = 0; i < num_blocks; i++) {
        T sum = static_cast<T>(0.0);
        for (int j = 0; j < num_per_block; j++) {
            sum += host_in[i * num_per_block + j];
        }
        host_golden_result[i] = sum;
    }

    T* device_in;
    T* device_result;
    CHECK_CUDA(cudaMalloc((void **)&device_in, sizeof(T) * NUMEL));
    CHECK_CUDA(cudaMalloc((void **)&device_result, num_blocks * sizeof(T)));

    for (int i = 0; i < NUM_REPEAT; i++) {
        CHECK_CUDA(cudaMemcpy(device_in, host_in, sizeof(T) * NUMEL, cudaMemcpyHostToDevice));
        blockReduceSumKernel<num_threads, num_per_thread><<<num_blocks, num_threads>>>(device_result, device_in, NUMEL);
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
    constexpr int64_t numel = 64 * 1024 * 1024;
    static_assert(numel % NUM_THREADS == 0, "For current implementation, the numel should be divisible by num_threads");
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
            case 7:
                runReduceSumV7<scalar_t, numel>();  // avoid explict synchronization in warp0 && unroll loop
                break;
            case 8:
                runReduceSumV8<scalar_t, numel>();  // avoid explict synchronization in warp0 && unroll loop && multi add
                break;
            case 9:
                runReduceSumV9<scalar_t, numel>();  // block reduce
                break;
        }
    }
    return 0;
}
