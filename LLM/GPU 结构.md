## GPU 结构
- SM (Streaming Multiproces) : 每个GPU包含多个SM, 每个SM独立执行计算任务
- L2 Cache
- HBM（High Bandwidth Memory）- 与memory controller连接, 在GPU外部
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/4d6c59d6-ebbd-4857-9c0c-b0c6311b3576)


### A100

<img width="600" length="500" alt="2" src="https://github.com/hinswhale/AI-Learning/assets/22999866/eddd26a2-2a4d-4fd9-b7d4-12a60482d95c">

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

## SRAM HBM  DRAM
- SRAM（Static Random Access Memory） - SM里， `L1 cache` &  `Register File `
- HBM（High Bandwidth Memory）- 支持多个SM并行访问
- DRAM（Dynamic Random Access Memory）- 在没有HBM的系统中，DRAM作为GPU的全局内存 & CPU里
- **SRAM > HBM > DRAM**

<img width="400" length="500" alt="2" src="https://github.com/hinswhale/AI-Learning/assets/22999866/fe65144a-7810-4eca-91ed-39235c902369">

# 参考
1.[AI 工程师都应该知道的GPU工作原理](https://www.bilibili.com/video/BV1rH4y1c7Zs/)
