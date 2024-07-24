# 主要思想
对于Memory-Bound的优化
 - `fusion融合`, 不对中间结果缓存，减少HBM的访问
 -  模型训练时需要保留中间结果，反向传播时使用。


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
- SRAM > HBM > DRAM

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

- operations fused：将好几个 operations fuse 成一个 operation 进而减轻 memory 存取的 loading
attention QKV 计算
分块矩阵，然后是 loop（outer loop，inner loop，对应的是 gpu cuda 的 kernel 优化）；

## 原始算法

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/6b85c509-7b16-4454-8b86-fd83d8d2c0b6)

符号说明: Q — queries, K — keys, V — values, S — scores, P — probabilities, O — outputs.

瓶颈：每次操作都要在HBM和SRAM直接移动数据 [主要是 NxN矩阵（S，P）， N >> D]
优化：kernel fusion - 在HBM和SRAM直接来回移动一次数据
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/96c6531b-3149-4a96-ab82-ccd6f608f47e)

### 术语
`kernel`：GPU的一次操作  
`Fusion`：把多个操作合并成一个  
`kernel fusion`


`materialization`:

### Challenges:
- (1) Compute softmax normalization without access to full input.
- (2) Backward without the large attention matrix from forward.

## two main ideas
- Tiling（在前向和后向传递中使用）- 简单讲就是将NxN的softmax/分数矩阵划分为块。
- Recomputation（仅在后向传递中使用 - 如果您熟悉activation/gradient checkpointing，这将很容易理解）。

(1) Tiling: Restructure algorithm to load block by block from HBM to SRAM to compute attention.
(2) Recomputation: Don't store attn. matrix from forward, recompute it in the backward.

### tiling
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/e5989edb-da91-4700-b5d9-ee8844afc8a2)

### Recomputation(backward pass)

![image](./images/Inference_regular_attn.gif)

![image](./images/inference_splitkv.gif)

![Inference regular attention](./images/Inference_regular_attn.gif)

# 参考
1. [Hardware-aware Algorithms for Sequence Modeling - Tri Dao](https://www.youtube.com/watch?v=foG0ebzuw34)
2. [flash-atttention-2](https://princeton-nlp.github.io/flash-atttention-2/)
3. [ELI5: FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
