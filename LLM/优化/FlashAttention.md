# 概述
FlashAttention - `Fast and Memory-Efficient Exact Attention with I0-Awareness`

对**Memory-Bound**的优化
- 目标：减少IO，即尽可能访问GPU内缓存（即**SRAM**）
  - `分块计算`，`fusion融合`, `减少中间结果缓存`
  -  反向传播时，重新计算中间结果
 
- 结果：
  - 无精度损失 
  - 2-4x speedup, 10-20x memory reduction
  - 显存O(N^2) -> O(N)

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/fc4177b7-df8e-40d1-921c-a39a73421435)

# 预备知识
## 大模型参数
- GPU memory:
  - P: Parameters
  - G: Gradients
  - OS: Optimizer states
> 三者占比：1:1:6

优化器的构造会封装 parameters：
- optimizer = optim.Adam(model.parameters(), lr=0.001)
- loss.backward() => parameters.grad
- optimizer.step() => optimizer states
  - momentum：gradient 的指数平均 【Adam】
  - variance：gradient square 的指数平均【Adam】
 
## GPU Memory
- SRAM[GPU内]  > HBM[GPU外] > DRAM
  ![image](https://github.com/user-attachments/assets/fb1f825e-939d-446f-aef6-33298b162c74)

>  **优化目标**： 尽可能访问GPU内缓存，即**SRAM**

- SM（Stream multiproecssors，流多处理器 
  - L1 cache - SRAM
  - register file -  SRAM

- SRAM：Static RAM（Random Access Memory） 192KB per（A100 108个，4090 128个）
 -  108*192/1024 = 20MB
- HBM：high bandwidth memory（4090 24GB，A100 80GB）

## compute-bound vs. memory-bound
- compute-bound ：
  -  多维度的矩阵相乘或是高 channel 数的 convolution
- memory-bound：
  -  element-wise （e.g.， activation， dropout） & reduction （e.g.， sum， softmax， batch norm， layer norm）

从这个图可以看出GPT-2是memory-bound
![image](https://github.com/user-attachments/assets/a3cd9e40-a734-4fb8-8867-fae8e57a39e5)

# **具体流程**

## 标准attention计算


![image](https://github.com/hinswhale/AI-Learning/assets/22999866/6b85c509-7b16-4454-8b86-fd83d8d2c0b6)

符号说明: Q — queries, K — keys, V — values, S — scores, P — probabilities, O — outputs.

**瓶颈**：

- 每次操作都要在HBM和SRAM直接移动数据 [ NxN矩阵（S，P）， N >> D, O(N^2)]


![image](https://github.com/user-attachments/assets/7a7eba39-acb2-44bf-9cdd-9aefd45d6a55)


## **Softmax**

$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$ 

## **Safe Softmax**

**如果 𝑥𝑖 过大，可能出现数据上溢的情况**

$\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}}$ 

## 核心思想

- **分块** +   **kernel fusion** 减少在HBM和SRAM之间数据传输次数

### Challenges:

- (1) Compute **softmax normalization** without access to full input.
- (2) Backward without the large attention matrix from forward.

### **Softmax normalization in blocks**
![image](https://github.com/user-attachments/assets/3dbc4f25-3e42-4925-a1b6-5df1bfdba894)


![image](https://github.com/user-attachments/assets/cd1a8f8b-9894-4890-8e1c-8d1e15bd7dbf)


> 图片来源：https://www.bilibili.com/video/BV1UT421k7rA/

### two main ideas

- **Tiling**（在前向和后向传递中使用）
    - Restructure algorithm to load block by block from HBM to SRAM to compute attention.
- **Recomputation**（仅在后向传递中使用 - 如果您熟悉activation/gradient checkpointing，这将很容易理解）。
    - Don't store attn. matrix from forward, recompute it in the backward.

### 算法流程

![image](https://github.com/user-attachments/assets/24ca3d36-d1b6-46ee-87c4-b9387e11905b)


### tiling

- 分块

![image](https://github.com/user-attachments/assets/431eaffd-d6a4-4a7b-99a5-ca3bf7c7e10d)


> 图片来源：https://zhuanlan.zhihu.com/p/669926191

- **Softmax normalization**
![image](https://github.com/user-attachments/assets/f3348e86-5ef2-4a52-b8a6-5ff0be195590)

![image](https://github.com/user-attachments/assets/372f2fcf-e6d7-436f-8c42-d2fdcd76fd1c)


### Recomputation(backward pass)【todo】

![image](../images/Inference_regular_attn.gif)

![image](../images/inference_splitkv.gif)


### flash & flash2


![image](https://github.com/user-attachments/assets/fdc50cbb-db70-407a-9740-9569195932cc)


# 参考

1. [Hardware-aware Algorithms for Sequence Modeling - Tri Dao](https://www.youtube.com/watch?v=foG0ebzuw34)
2. [flash-atttention-2](https://princeton-nlp.github.io/flash-atttention-2/)
3. [ELI5: FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
4. [Flash Attention 为什么那么快？](https://www.bilibili.com/video/BV1UT421k7rA/)
5. [图解大模型计算加速系列：FlashAttention V1，从硬件到计算逻辑](https://zhuanlan.zhihu.com/p/669926191)
