#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

#include <unistd.h>
#include <assert.h>

#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

#include "../utils/utils.h"


#define NUM_EMBEDDING 21128
#define EMBEDDING_DIM 768
#define NUM_INDICES 256 * 256

#define NUM_WARMUP 5
#define NUM_REPEAT 5


typedef __nv_half float16_t;
typedef __nv_half2 float162_t;
typedef __nv_bfloat16 bfloat16_t;
typedef __nv_bfloat162 bfloat162_t;

template <typename T>
struct vec2_type_traits;

template <>
struct vec2_type_traits<float> {
    using type = float2;
};

template <>
struct vec2_type_traits<float16_t> {
    using type = float162_t;
};

template <>
struct vec2_type_traits<bfloat16_t> {
    using type = bfloat162_t;
};

namespace custom {
// Implement atomicAdd by atomicCAS primitive

__device__ void atomicAdd(float16_t* address, float16_t val) {
    unsigned int* address_as_ui = (unsigned int*)((size_t)address & ~2);  // ensure aligned with 4 bytes
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        uint16_t sum = ((size_t)address & 2) ? (old >> 16) : (old & 0xffff);  // little endian
        float new_val = __half2float(__ushort_as_half(sum)) + static_cast<float>(val);
        uint16_t newval_u = __half_as_ushort(__float2half(new_val));
        old = ((size_t)address & 2) ? (old & 0xffff) | (newval_u << 16) : (old & 0xffff0000) | newval_u;
        old = atomicCAS(address_as_ui, assumed, old);
    } while(assumed != old);
}

__device__ void atomicAdd(bfloat16_t* address, bfloat16_t val) {
    unsigned int* address_as_ui = (unsigned int*)((size_t)address & ~2);  // ensure aligned with 4 bytes
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        uint16_t sum = ((size_t)address & 2) ? (old >> 16) : (old & 0xffff);  // little endian
        float new_val = __bfloat162float(__ushort_as_bfloat16(sum)) + static_cast<float>(val);
        uint16_t newval_u = __bfloat16_as_ushort(__float2bfloat16(new_val));
        old = ((size_t)address & 2) ? (old & 0xffff) | (newval_u << 16) : (old & 0xffff0000) | newval_u;
        old = atomicCAS(address_as_ui, assumed, old);
    } while(assumed != old);
}

__device__ void atomicAdd(float162_t* address, float162_t val) {
    // mimic vectorized atomicAdd, though dont know how cuda implements internally
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        uint16_t sum = ((size_t)address & 2) ? (old >> 16) : (old & 0xffff);  // little endian
        float new_val = __half2float(__ushort_as_half(sum)) + static_cast<float>(val.x);
        uint16_t newval_u = __half_as_ushort(__float2half(new_val));
        old = ((size_t)address & 2) ? (old & 0xffff) | (newval_u << 16) : (old & 0xffff0000) | newval_u;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);

    address_as_ui = ((size_t)address & 2) ? (unsigned int*)((char*)address + 4): address_as_ui;
    old = *address_as_ui;
    do {
        assumed = old;
        uint16_t sum = ((size_t)address & 2) ? (old & 0xffff) : (old >> 16);
        float new_val = __half2float(__ushort_as_half(sum)) + static_cast<float>(val.y);
        uint16_t newval_u = __half_as_ushort(__float2half(new_val));
        old = ((size_t)address & 2) ? (old & 0xffff0000) | newval_u : (old & 0xffff) | (newval_u << 16);
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}

__device__ void atomicAdd(bfloat162_t* address, bfloat162_t val) {
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        uint16_t sum = ((size_t)address & 2) ? (old >> 16) : (old & 0xffff);  // little endian
        float new_val = __bfloat162float(__ushort_as_bfloat16(sum)) + static_cast<float>(val.x);
        uint16_t newval_u = __bfloat16_as_ushort(__float2bfloat16(new_val));
        old = ((size_t)address & 2) ? (old & 0xffff) | (newval_u << 16) : (old & 0xffff0000) | newval_u;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);

    address_as_ui = ((size_t)address & 2) ? (unsigned int*)((char*)address + 4): address_as_ui;
    old = *address_as_ui;
    do {
        assumed = old;
        uint16_t sum = ((size_t)address & 2) ? (old & 0xffff) : (old >> 16);
        float new_val = __bfloat162float(__ushort_as_bfloat16(sum)) + static_cast<float>(val.y);
        uint16_t newval_u = __bfloat16_as_ushort(__float2bfloat16(new_val));
        old = ((size_t)address & 2) ? (old & 0xffff0000) | newval_u : (old & 0xffff) | (newval_u << 16);
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}

}  // namespace custom

template <typename scalar_t>
void checkEmbeddingBackwardResult(const scalar_t* actual, const scalar_t* desired, int64_t numel) {
    constexpr float abs_diff = 1e-1;  // smooth check
    for (int64_t i = 0; i < numel; i++) {
        float diff = static_cast<float>(actual[i] - desired[i]);
        if (diff > abs_diff || diff < -abs_diff) {
            std::cout << "Check failed, current diff: " << diff << "\n";
            break;
        }
    }
}

template <typename scalar_t, typename index_t>
void embeddingBackwardCPU(const scalar_t* grad, const index_t* indices, scalar_t* out, int64_t num_indices, int64_t num_embedding, int64_t embedding_dim, int64_t padding_idx) {
    for (int64_t i = 0; i < num_indices; i++) {
        index_t origin_id = indices[i];
        scalar_t* cur_out_ptr = out + origin_id * embedding_dim;
        const scalar_t* cur_grad_ptr = grad + i * embedding_dim;
        bool valid_id = (origin_id >= 0 && origin_id < num_embedding && origin_id != padding_idx);
        if (valid_id) {
            for (int64_t j = 0; j < embedding_dim; j++) {
                cur_out_ptr[j] += cur_grad_ptr[j];
            }
        }
    }
}
 
template <typename scalar_t, typename index_t>
__global__ void embeddingBackwardKernel(const scalar_t* grad, const index_t* indices, scalar_t* out, int64_t num_indices, int64_t num_embedding, int64_t embedding_dim, int64_t padding_idx) {
    uint32_t ox = threadIdx.x;
    uint32_t oy = blockIdx.x + threadIdx.y * gridDim.x;

    for (; oy < num_indices; oy += blockDim.y * gridDim.x) {
        index_t id = indices[oy];
        scalar_t* cur_out_ptr = out + id * embedding_dim;
        const scalar_t* cur_grad_ptr = grad + oy * embedding_dim;
        bool valid_id = (id >= 0 && id < num_embedding && id != padding_idx);
        if (valid_id) {
            for (int64_t i = ox; i < embedding_dim; i += blockDim.x) {
                // custom::atomicAdd(cur_out_ptr + i, cur_grad_ptr[i]);
                atomicAdd(cur_out_ptr + i, cur_grad_ptr[i]);
            }
        }
    }
}

template <typename scalar_t, typename index_t>
__global__ void embeddingBackwardNoWarpDivergenceKernel(const scalar_t* grad, const index_t* indices, scalar_t* out, int64_t num_indices, int64_t num_embedding, int64_t embedding_dim, int64_t padding_idx) {
    uint32_t ox = threadIdx.x;
    uint32_t oy = blockIdx.x + threadIdx.y * gridDim.x;

    for (; oy < num_indices; oy += blockDim.y * gridDim.x) {
        index_t id = indices[oy];
        scalar_t* cur_out_ptr = out + id * embedding_dim;
        const scalar_t* cur_grad_ptr = grad + oy * embedding_dim;
        bool valid_id = (id >= 0 && id < num_embedding && id != padding_idx);
        if (valid_id) {
            for (int64_t i = ox * 2; i < embedding_dim; i += (blockDim.x * 2)) {
                // custom::atomicAdd(cur_out_ptr + i, cur_grad_ptr[i]);
                atomicAdd(cur_out_ptr + i, cur_grad_ptr[i]);
            }
            for (int64_t i = ox * 2 + 1; i < embedding_dim; i += (blockDim.x * 2)) {
                // custom::atomicAdd(cur_out_ptr + i, cur_grad_ptr[i]);
                atomicAdd(cur_out_ptr + i, cur_grad_ptr[i]);
            }
        }
    }
}

template <typename scalar_t, typename index_t>
__global__ void embeddingBackwardAtomic2Kernel(const scalar_t* grad, const index_t* indices, scalar_t* out, int64_t num_indices, int64_t num_embedding, int64_t embedding_dim, int64_t padding_idx) {
    using vec2_t = typename vec2_type_traits<scalar_t>::type;
    uint32_t ox = threadIdx.x;
    uint32_t oy = blockIdx.x + threadIdx.y * gridDim.x;

    for (; oy < num_indices; oy += blockDim.y * gridDim.x) {
        index_t id = indices[oy];
        scalar_t* cur_out_ptr = out + id * embedding_dim;
        const scalar_t* cur_grad_ptr = grad + oy * embedding_dim;
        bool valid_id = (id >= 0 && id < num_embedding && id != padding_idx);
        if (valid_id) {
            for (int64_t i = ox; i * 2 + 2 <= embedding_dim; i += blockDim.x) {
                atomicAdd(reinterpret_cast<vec2_t*>(cur_out_ptr) + i,
                          *(reinterpret_cast<vec2_t*>(const_cast<scalar_t*>(cur_grad_ptr)) + i));
            }
        }
    }
}

template <typename scalar_t, typename index_t>
void run() {
    constexpr int warp_size = 32;
    dim3 block(warp_size, 16, 1);
    dim3 grid((EMBEDDING_DIM + warp_size - 1) / warp_size, 1, 1);

    // host memory allocation
    scalar_t* host_grad = (scalar_t*)malloc(sizeof(scalar_t) * NUM_INDICES * EMBEDDING_DIM);
    scalar_t* host_grad_out = (scalar_t*)malloc(sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM);
    index_t* host_indices = (index_t*)malloc(sizeof(index_t) * NUM_INDICES);
    scalar_t* device_result = (scalar_t*)malloc(sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM);

    memset(host_grad_out, 0, sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM);
    std::mt19937 gen_f(114514);
    std::uniform_real_distribution<> dis_f(-1, 1);
    for (int i = 0; i < NUM_INDICES * EMBEDDING_DIM; i++) {
        host_grad[i] = static_cast<scalar_t>(dis_f(gen_f));
    }

    std::mt19937 gen_i(42);
    std::uniform_int_distribution<> dis_i(0, NUM_EMBEDDING - 1);
    for (int i = 0; i < NUM_INDICES; i++) {
        host_indices[i] = static_cast<index_t>(dis_i(gen_i));
    }

    embeddingBackwardCPU(host_grad, host_indices, host_grad_out, NUM_INDICES, NUM_EMBEDDING, EMBEDDING_DIM, -1);

    // device memory allocation
    scalar_t* device_grad;
    scalar_t* device_grad_out;
    index_t* device_indices;
    cudaMalloc((void **)&device_grad, sizeof(scalar_t) * NUM_INDICES * EMBEDDING_DIM);
    cudaMalloc((void **)&device_grad_out, sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM);
    cudaMalloc((void **)&device_indices, sizeof(index_t) * NUM_INDICES);
    cudaMemset(device_grad_out, 0, sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM);
    cudaMemcpy(device_grad, host_grad, sizeof(scalar_t) * NUM_INDICES * EMBEDDING_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(device_indices, host_indices, sizeof(index_t) * NUM_INDICES, cudaMemcpyHostToDevice);

    // 
#define LAUNCH_NAIVE_KERNEL()                                   \
  embeddingBackwardKernel<scalar_t, index_t><<<grid, block>>>(  \
        device_grad,                                            \
        device_indices,                                         \
        device_grad_out,                                        \
        static_cast<int64_t>(NUM_INDICES),                      \
        static_cast<int64_t>(NUM_EMBEDDING),                    \
        static_cast<int64_t>(EMBEDDING_DIM),                    \
        static_cast<int64_t>(-1));

#define LAUNCH_NO_WARP_DIVERGENCE_KERNEL()                      \
  embeddingBackwardNoWarpDivergenceKernel<scalar_t, index_t><<<grid, block>>>(  \
        device_grad,                                            \
        device_indices,                                         \
        device_grad_out,                                        \
        static_cast<int64_t>(NUM_INDICES),                      \
        static_cast<int64_t>(NUM_EMBEDDING),                    \
        static_cast<int64_t>(EMBEDDING_DIM),                    \
        static_cast<int64_t>(-1));

#define LAUNCH_ATOMIC_VEC2_KERNEL()                                    \
  embeddingBackwardAtomic2Kernel<scalar_t, index_t><<<grid, block>>>(  \
        device_grad,                                                   \
        device_indices,                                                \
        device_grad_out,                                               \
        static_cast<int64_t>(NUM_INDICES),                             \
        static_cast<int64_t>(NUM_EMBEDDING),                           \
        static_cast<int64_t>(EMBEDDING_DIM),                           \
        static_cast<int64_t>(-1));

    GPUTimer timer;
    std::vector<float> cost_times(NUM_REPEAT);

    for (int i = 0; i < NUM_WARMUP; i++) {
        cudaMemset(device_grad_out, 0, sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM);
        LAUNCH_NAIVE_KERNEL();
    }

    for (int i = 0; i < NUM_REPEAT; i++) {
        cudaMemset(device_grad_out, 0, sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM);
        timer.start();
        LAUNCH_NAIVE_KERNEL();
        timer.stop();
        cost_times[i] = timer.elapsed();
    }

    std::cout << "NaiveKernel cost " << std::accumulate(cost_times.cbegin(), cost_times.cend(), 0.0f) / (1.0 * NUM_REPEAT) << " ms\n";
    cudaMemcpy(device_result, device_grad_out, sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM, cudaMemcpyDeviceToHost);
    checkEmbeddingBackwardResult(device_result, host_grad_out, NUM_EMBEDDING * EMBEDDING_DIM);

    for (int i = 0; i < NUM_WARMUP; i++) {
        cudaMemset(device_grad_out, 0, sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM);
        LAUNCH_NO_WARP_DIVERGENCE_KERNEL();
    }

    for (int i = 0; i < NUM_REPEAT; i++) {
        cudaMemset(device_grad_out, 0, sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM);
        timer.start();
        LAUNCH_NO_WARP_DIVERGENCE_KERNEL();
        timer.stop();
        cost_times[i] = timer.elapsed();
    }
    std::cout << "NoWarpDivergenceKernel cost " << std::accumulate(cost_times.cbegin(), cost_times.cend(), 0.0f) / (1.0 * NUM_REPEAT) << " ms\n";
    cudaMemcpy(device_result, device_grad_out, sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM, cudaMemcpyDeviceToHost);
    checkEmbeddingBackwardResult(device_result, host_grad_out, NUM_EMBEDDING * EMBEDDING_DIM);

    for (int i = 0; i < NUM_WARMUP; i++) {
        cudaMemset(device_grad_out, 0, sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM);
        LAUNCH_ATOMIC_VEC2_KERNEL();
    }

    for (int i = 0; i < NUM_REPEAT; i++) {
        cudaMemset(device_grad_out, 0, sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM);
        timer.start();
        LAUNCH_ATOMIC_VEC2_KERNEL();
        timer.stop();
        cost_times[i] = timer.elapsed();
    }

    std::cout << "Atomic2Kernel cost " << std::accumulate(cost_times.cbegin(), cost_times.cend(), 0.0f) / (1.0 * NUM_REPEAT) << " ms\n";
    cudaMemcpy(device_result, device_grad_out, sizeof(scalar_t) * NUM_EMBEDDING * EMBEDDING_DIM, cudaMemcpyDeviceToHost);
    checkEmbeddingBackwardResult(device_result, host_grad_out, NUM_EMBEDDING * EMBEDDING_DIM);

    free(host_grad);
    free(host_grad_out);
    free(host_indices);
    free(device_result);
    cudaFree(device_grad);
    cudaFree(device_grad_out);
    cudaFree(device_indices);
}

int main() {
    assert(!(EMBEDDING_DIM & 1));  // simplify vectorized case

    // run<bfloat16_t, int64_t>();
    run<float16_t, int64_t>();

    return 0;
}
