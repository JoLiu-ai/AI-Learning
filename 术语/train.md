ZeRO-1/2/3 and FSDP

ZeRO（Zero Redundancy Optimizer）是由微软提出的一种分布式训练优化技术，旨在通过减少数据冗余来提高大规模深度学习模型的训练效率，特别是在多节点分布式训练的情况下。ZeRO 包括多个不同的优化阶段，分别为 ZeRO-1、ZeRO-2、ZeRO-3。FSDP（Fully Sharded Data Parallel）是 PyTorch 的一种实现方法，可以与 ZeRO 技术相结合，提高训练效率和内存利用率。

## 参数分布
![image](https://github.com/user-attachments/assets/a14f2aac-8500-4f49-be94-10fe913969e6)

**Memory Consumption**
{2bytes(weights) + 2bytes(gradients) + 12bytes(optim states)}𝝍

A100/H100 with 80GB memory, the largest trainable model is
```80GB/16Bytes=5.0B```

## 总览
![image](https://github.com/user-attachments/assets/2516b01a-69ca-4f51-a31f-5984e4e254a1)

### ZeRO-1

optimizer states

**Memory Consumption**

{2bytes(weights) + 2bytes(gradients) + **12/N** bytes(optim states)}𝝍

WHEN N=64:
```80GB/4.2Bytes=19B```

### ZeRO-2

optimizer states and gradients

**Memory Consumption**

{2bytes(weights) + **2/N** bytes(gradients) + **12/N** bytes(optim states)}𝝍

WHEN N=64:
```80GB/2.2Bytes=36B```

### ZeRO-3


**Memory Consumption**
weights & gradients & optim states
{**2/N** bytes(weights) + **2/N** bytes(gradients) + **12/N** bytes(optim states)}𝝍

WHEN N=64:
```80GB/0.25Bytes=320B```


