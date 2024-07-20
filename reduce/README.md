- reduce v1(reduceSumGlobalMemKernel): baseline，直接基于global memory
- reduce v2(reduceSumSharedMemKernel): 在shared memory中做reduce
- reduce v3(reduceSumSharedMemLessWarpDivergenceKernel): 减少warp divergence情况
- reduce v4(reduceSumSharedMemNoBankConflictKernel): 优化shared memory方式，避免产生bank conflict
- reduce v5(reduceSumAddDuringLoadKernel): 每个block会把相邻blockDim.x个元素加到当前第threadIdx.x个shared memory上，即所有的线程其实都参与了Add计算
- reduce v6(reduceSumAddDuringLoadWarpReduceKernel): 最后一个warp避免`__syncthreads`，
- reduce v7(reduceSumAddDuringLoadWarpReduceUnrollLoopKernel): 把kernel中的循环全部展开，减少一些指令
- reduce v8(reduceSumMultiAddKernel): 每个block会把相邻NUM_PER_THREAD * blockDim.x个元素加到当前第threadIdx.x个shared memory上 （固定了block数，方便调整block来查看不同block对性能的影响，但实验中取256、512、1024并没发现明显区别）
- reduce v9(blockReduceSumKernel): 利用了`__shfl_down_sync`，由于只有每个block中的第一个warp会用到shared memory，不用考虑bank conflict

在4060显卡上基于Nsight Compute测试得到的性能如下：
|reduce kernel|cost(us)| Bandwidth [%] | DRAM Throughput [%] |
| --- | --- | --- | --- |
|reduce v1| 6196.16 | 15.93% | 33.37% |
|reduce v2| 5476.86 | 18.02% | 19.89% |
|reduce v3| 3610.98 | 27.33% | 29.18% |
|reduce v4| 3467.17 | 28.46% | 31.28% |
|reduce v5| 1792.64 | 55.05% | 58.68% |
|reduce v6| 1098.59 | 89.83% | 96.56% |
|reduce v7| 1086.59 | 90.82% | 97.01% |
|reduce v8| 1088.26 | 90.69% | 97.22% |
|reduce v9| 1128.10 | 87.48% | 97.40% |

发现最终kernel带宽利用率能到97.40%，还是挺惊讶的，不过这个数据是直接看的Nsight Compute的DRAM Throughput，如果按照`64 * 1024 * 1024 * 4 / 1128100 / 272`来算的话是87.48%

v6~v9性能可能存在波动，以后有机会再测试下

--------------
相关资料
- [深入浅出GPU优化系列：reduce优化](https://zhuanlan.zhihu.com/p/426978026)
- [reduce优化学习笔记](https://zhuanlan.zhihu.com/p/596012674)
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [PyTorch-BlockReduceSum](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/block_reduce.cuh#L58)
