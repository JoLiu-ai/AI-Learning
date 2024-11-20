## QLORA

- NF4 Quantization
- Double Quantization
- Paged optimizers


### 0.block-wise quantization
#### 问题背景:
- 整个权重矩阵使用相同的量化参数会导致精度损失
- 权重分布在不同区域可能差异很大
- 单一量化参数无法很好地表达这种差异

#### 解决方案:
- 将权重矩阵分成小块(Blocks)
- 每个Block使用独立的量化参数
- 更好地保持局部数值特征

#### 带来的新问题
- 量化参数膨胀:每个Block都需要scale和zero_point，这些参数以FP16存储
- Block越小，参数数量越多

### 1. NF4 Quantization
4- bit -> 2^4=16
bucket number

### 2. Double Quantization
#### 流程
原始: 16/32-bit权重
第一次量化 --> 4-bit权重 + 16-bit常量
第二次量化 --> 4-bit权重 + 2-bit常量

### 3. Paged optimizers

