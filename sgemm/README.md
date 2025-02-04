**WIP**

## Naive implementation
每个线程负责计算一个输出矩阵中的结果，需要读取矩阵A的一行和矩阵B的一列

## Shared Memory Caching
每个线程负责计算一个输出矩阵中的结果，首先将当前thread_block计算时所需的矩阵A和矩阵B中数据加载到shared memory中，可以避免减少对global memory的读取次数