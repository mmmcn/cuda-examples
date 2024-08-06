**Benchmark**
|kernel|cost(us)| Bandwidth [%] | DRAM Throughput [%] |
| --- | --- | --- | --- |
|naive cpu| 1198600 |  |  |
|online cpu| 888204 |  |  |
|kernel v1| 44250 |  | 24.49% |
|kernel v2| 5290 |  | 78.61% |
|kernel v3| 7440 |  | 63.42% |
|kernel v4| 4750 |  | 87.76% |
|kernel v5| 4490 |  | 93.04% |
|kernel v6| 4600 |  | 90.89% |

------------
相关资料
- [Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867)
- [llm.c](https://github.com/karpathy/llm.c/blob/master/dev/cuda/softmax_forward.cu)
- [pytorch-softmax](https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/PersistentSoftmax.cuh#L303)