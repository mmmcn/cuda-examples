embedding_backward算子实现  
--------------------------
基于atomicAdd完成，实现了三种版本的kernel，分别针对warp divergence和memory access两个点做了优化。  

非atomic版本可参考[PyTorch实现](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu#L212)  
  
在4060上`-arch=sm_89`貌似不支持float2类型的atomicAdd，因此只展示`float16`和`bfloat16`性能。
`float16`和`bfloat16`在三种kernel下的性能如表所示：
|dtype|NaiveKernel cost(ms)|NoWarpDivergenceKernel cost(ms)|Atomic2Kernel cost(ms)|
| --- | --- | --- | --- |
|float16|3.14ms| 2.27ms | 1.77ms |
|bfloat16|3.38ms | 2.54ms | 1.62ms |
  
可以看到，`NoWarpDivergenceKernel`实现能够带来25%~30%的性能提升，`Atomic2Kernel`实现能够带来40%+甚至50%的性能提升。