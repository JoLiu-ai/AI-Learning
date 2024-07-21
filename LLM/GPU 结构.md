## GPU 结构
- SM (Streaming Multiproces) : 每个GPU包含多个SM, 每个SM独立执行计算任务
- L2 Cache
- HBM（High Bandwidth Memory）- 与memory controller连接, 在GPU外部
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/4d6c59d6-ebbd-4857-9c0c-b0c6311b3576)


### A100 整体结构

<img width="600" length="500" alt="2" src="https://github.com/hinswhale/AI-Learning/assets/22999866/eddd26a2-2a4d-4fd9-b7d4-12a60482d95c">

1. **PCI Express 4.0 Host Interface**：
    - 连接GPU与**主机系统**，负责数据交换。
2. **GigaThread Engine with MIG Control**：
    - 管理线程和内存，支持多实例GPU（MIG）功能。
    - 允许将GPU划分为多个逻辑GPU实例，提高资源利用率。
3. **GPC（Graphics Processing Clusters）**：
    - 每个GPC包含多个TPC（Texture Processing Clusters）。
    - TPC由两个SM（Streaming Multiprocessors）组成。
4. **SM（Streaming Multiprocessor）**：
    - 每个SM包含多个CUDA核心、张量核心、纹理单元、寄存器文件和共享内存等。
    - SM是执行并行计算的基本单元。
5. **L2 Cache（二级缓存）**：
    - L2缓存位于多个GPC之间，共享访问。
    - 提供对GPU内存的数据缓存，减少延迟，提高性能。
6. **HBM2（High Bandwidth Memory 2）**：
    - GPU外部的高速内存，通过内存控制器连接到GPU核心。
    - 提供高带宽和低延迟的存储访问，适用于大数据集和复杂计算任务。
7. **Memory Controller（内存控制器）**：
    - 管理HBM2内存与GPU核心之间的数据传输。
    - 每个内存控制器与一个HBM2堆叠连接。
8. **NVLink**：
    - **GPU之间**的高速互联接口，允许多个GPU高效通信。
    - 提高多GPU系统的带宽，降低延迟。

--- 

### SM (Streaming Multiproces）

<img width="600" length="500" alt="2" src="https://github.com/hinswhale/AI-Learning/assets/22999866/b0818c7d-9f7e-4b8a-bd0b-53798d60864b">


#### 功能
  - 单一指令，多线程执行。比如矩阵乘法里结果里的每个元素可以分配一个线程。

#### 每个SM 结构
- 4 processing block/SM
- 每个**processing block**
    - 16 `INT32`, 16`FP32`, 8`FP64` CUDA cores
   - 1 Tensor cores
   - 1 Warp scheduler/processing block
   - L1 cache
   - so on(参考👇)

**L1 Instruction Cache**
- 靠近计算核心的指令缓存，但容量比L0缓存大，存储更多指令，提供次快的指令访问。

**L0 Instruction Cache**：
- 通常是最靠近计算核心的**指令缓存**，存储最近执行的指令，提供最快的指令访问。


**Warp Scheduler**

- 每个Warp Scheduler一次调度32个线程（称为一个warp）。**Warp**是GPU里调度任务的最小单元。

**Dispatch Unit**

- 负责指令的分发

**Register File（由SRAM构成）**

- 用于存储线程的**临时数据**和**中间计算结果**
- “16,384 x 32-bit”表示有16384个32位的寄存器

**INT32 / FP32 / FP64**

- INT32 core / FP32 core / FP64 core

**Tensor Core**

- 专为深度学习优化：短阵运算，混合精度计算。

**SFU(Special Function Units)**

- 例如： 三角函数(sin. cos) ，指数函数(exp），对数函数（log），平方根 (sqrt)

**LD/ST (Load/Store Unit)**

- 负责从内存加载数据（Load）和将数据存储到内存（Store）

**Tex (Texture Unit)**

- 负责处理纹理映射相关的任务，从内存中读取纹理数据，并进行相应的过滤和处理 （Texture Data）【二维图像（纹理）应用到三维模型表面的一种技术】

**192KB L1 Data Cache / Shared Memory**

- **L1 Data Cache**（L1缓存）， `SM内共享`，用于**存储数据**，减小内存访问延迟。（由SRAM构成）
- **Shared Memory**（共享内存）**Thread Block**（同一个线程块）中不同线程之间的数据共享
>  区别 **Thread Block** vs **warp**： ```Thread Block```是程序员定义的执行单位，可以包含多个```warp```，并且线程块内的线程可以共享内存和进行同步。

> L1指令缓存通常容量较小，但速度极快; L2缓存、L3缓存等。它们容量较大，但访问速度相对较慢，逐步靠近主存，层次越高，缓存容量越大，访问延迟也相应增加

--- 

## 存储对比

<img width="600" length="500" alt="2" src="https://github.com/user-attachments/assets/1ceda0c1-6ab8-4245-a5d4-f6791db15ad2">


- **Register File**  [GPU - SM内部] (SRAM)
- **L1 Data Cache** / **Shared Memory** [GPU - SM内部]  (SRAM)
- **L2  Cache** ：所有SM 共享 [GPU - SM外部,across all SMs] （SRAM）
- **HBM(high bandwidth Memory)** - 显存，不在GPU上,，支持多个SM并行访问
    
### SRAM HBM  DRAM
- SRAM（Static Random Access Memory） - SM里， `L1 cache` &  `Register File `
- DRAM（Dynamic Random Access Memory）- 在没有HBM的系统中，DRAM作为GPU的全局内存 & CPU里
- **SRAM > HBM > DRAM**

<img width="400" length="500" alt="2" src="https://github.com/user-attachments/assets/eb16782d-147e-4ae2-a189-d9a588dbfb9b">

---

### Tensor Core：
- 张量核心专门设计用于深度学习任务，显著提高矩阵运算的效率。
- 每个张量核心可以在一个时钟周期内执行一个矩阵乘法和累加运算（例如4x4矩阵的乘加）。
- FP16的范围较小，直接进行大量累加可能会导致数值溢出或精度不足。在累加过程中将FP16数据转换为FP32（单精度浮点数），然后进行累加操作，以减少误差。累加完成后，结果可以再转换回FP16。

![image](https://github.com/user-attachments/assets/eada45b0-3f90-422d-9bd4-9ae659803d12)

**Overflow**
对于FP16（半精度浮点数），其表示的最大值约为65504。当累加结果超过这个值时，就会发生溢出，结果变为无穷大（Infinity）
    
**Precision Loss**

**大数吃小数**


CUDA Core：

每个CUDA核心可以在一个时钟周期内执行一个浮点或整数运算。
CUDA核心的设计使其能够在并行环境下高效地执行简单算术运算。
Tensor Core：

每个张量核心可以在一个时钟周期内执行一个矩阵乘法和累加运算（例如4x4矩阵的乘加）。
张量核心专门设计用于深度学习任务，显著提高矩阵运算的效率。

## 性能相关
### 术语
1. **Peak FP64 GigaFLOPs (19500)**: 
    
    - A100 GPU在双精度浮点运算（FP64）下的理论最大性能。以“GigaFLOPs”表示，即每秒可进行19500亿次双精度浮点运算。
    
2. **Memory B/W (1555 GB/sec)**: 
    
   - 指A100 GPU的内存带宽，表示**每秒钟GPU可以从内存中读取或写入的数据量**，单位是GB/sec（千兆字节每秒）。
    
3. **Compute Intensity (100)**: 
    
    - 计算强度（Compute Intensity）是计算密集度的量度，通常表示为**计算量与内存访问量的比率， 即 计算量 / 数据传输量**。
    
    - 描述了执行一定数量的计算所需的数据传输量

```
NVIDIA A100的参数：
Peak FP64 GigaFLOPs: 19500 GFlops
Memory B/W: 1555 GB/sec
Compute Intensity: 100 = 19500 / 1555
```

### 性能计算
以 N * N 矩阵(C=A*B)为例， 

**计算量(FLOPs)**： $2N^3$

**数据传输量(Bytes Accessed**)： $3N^2 * sizeof$

**计算量** 是 **传输量** 的 Nx

**FLOPS**
- 每个元素的计算需要N次乘法和N-1次加法，共2N次计算
- 总共有N<sup>2</sup>个元素需要计算
- 总计算次数为2N<sup>3</sup>

**传输的数据量**
- 需要读取两个N x N的输入矩阵
- 需要写入一个N x N的输出矩阵
- 总计算次数为 $2N^2$
- 假设每个元素是4字节(32位浮点数),  $Bytes Accessed = (2 * N^2 + N^2) * 4 = 3N^2 * 4 = 12N^2 $

#### 如何充分利用GPU

1. **传输慢，数据大**
  - Memory Bound（Compute Intensity ⬇️， 内存带宽需求很高）
  - 计算强度较低，数据传输时间占主导

2. **传输快，数据小**
  - Compute Bound （Compute Intensity ⬆️）

按`Peak FP64 GigaFLOPs` 计算公式，这两种情况都能达到性能最大化

### compute vs memory Intensity
**compute-bound** 计算密集型
  - 在每单位内存传输量中包含大量计算
  - 示例：复杂的数学运算、深度学习中的前向和反向传播计算等， 如 FFN
**memory-bound** 内存带宽密集型
   - 每单位计算需要大量的数据传输，频繁地从内存中读取或写入数据，计算量相对较小。
   - 示例：如需要频繁访问大量数据的任务， 如分词、嵌入向量化，Self-Attention Mechanism



# 参考
1.[AI 工程师都应该知道的GPU工作原理](https://www.bilibili.com/video/BV1rH4y1c7Zs/)
