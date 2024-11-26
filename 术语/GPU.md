关键词：
`NVLINK`, `EDR`,
`Summit nodes`

-- 来源：chatgpt/ claude/internet

teraflops - 每秒**万亿**次浮点运算
Petaflops - 每秒**200千万亿**次浮点运算

### PB (Petabyte):
PB 是数据存储容量的单位，代表 千兆字节（GB） 的千倍，即 1 PB = 1024 TB (terabytes)，或者 1,048,576 GB。通常用于描述大规模数据中心或超算的存储能力。
举例：1 PB 相当于大约 250,000 张 4K 视频。

### Gbit (Gigabit):
Gbit 是数据传输速度的单位，代表 十亿比特（1 Gbit = 1,000,000,000 bit）。它用于衡量网络带宽或数据传输速率。
Gbit 和 GB 的区别：1 字节（Byte） = 8 比特（bit），因此 1 GB = 8 Gbit。
举例：如果网络带宽是 10 Gbit/s，那么它每秒传输 10 亿比特的数据。

### NVLINK
NVLINK 是由 NVIDIA 开发的高速互联技术，用于连接多个 GPU 和 CPU，以提高数据传输带宽和减少延迟。NVLINK 提供比传统的 PCIe（Peripheral Component Interconnect Express）更高的带宽，通常用于高性能计算（HPC）和深度学习系统中。

- 带宽: NVLINK 可提供每条链接高达 25 GB/s 的数据传输速度，远高于 PCIe 3.0 和 4.0 的带宽。
- 用途: 它允许多个 GPU 之间实现高效的数据共享，适用于大规模并行处理任务，如深度学习训练、科学模拟和大数据分析

### InfiniBand 
- InfiniBand 一种计算机网络互联技术，提供高带宽和低延迟，广泛应用于超级计算机和分布式存储
- EDR（Enhanced Data Rate）是其一个版本，支持高达 100 Gbit/s 的单链路带宽。在双轨配置下，可以实现 200 Gbit/s。

### DDR
DDR（Double Data Rate）是指 双倍数据速率 内存。它是目前最常见的内存技术之一，广泛应用于个人计算机、服务器、游戏主机等设备中。以下是一些主要的 DDR 标准：

- DDR1（第一代）：最初的 DDR 内存，速度相对较慢。
- DDR2（第二代）：相较 DDR1 提供更高的传输速率和更低的功耗。
- DDR3（第三代）：带宽进一步提升，功耗进一步降低。
- DDR4（第四代）：相较 DDR3，DDR4 提供更高的速度（高达 3200 MT/s）和更低的电压（1.2V）。
- DDR5（第五代）：最新一代 DDR 内存，带宽更高，支持更大的内存容量和更低的功耗。
每一代 DDR 内存的 数据传输速率 都比前一代更高，并且支持更高的带宽，使得系统可以更快地访问数据。

### Summit nodes
Summit nodes 美国橡树岭国家实验室（Oak Ridge National Laboratory，ORNL）开发的超级计算机 Summit 中的计算单元。每个计算节点被称为一个 Summit node
- 每个 Summit node 配备了：
- 2个IBM POWER9处理器。
- 6个NVIDIA Tesla V100 GPU，专为加速深度学习和其他高性能计算任务设计。
- 每个节点的总内存为 512 GB DDR4内存。

### 架构 
计算能力/CUDA 核心/Tensor 核心/内存
以下是 NVIDIA GPU 架构的表格，包括常见设备、特点/引入特性、是否可用于深度学习（炼丹）以及其他相关信息。

| 架构         | 发布时间   | 常见设备                                      | 特点/引入特性                                           | 其他                      |
|--------------|------------|-----------------------------------------------|--------------------------------------------------------|---------------------------|
| Tesla        | 2006年     | Tesla C1060, Tesla M1060                       | 首个支持 CUDA 的 GPU，通用数据并行计算                | 开创了通用 GPU 计算的先河 |
| Fermi        | 2010年     | GeForce GTX 480, Quadro 6000                  | 第一个完整的 GPU 计算架构，支持 ECC 内存，改进的内存管理| 支持 CUDA 3.2 至 CUDA 8   |
| Kepler       | 2012年     | GeForce GTX 680, Tesla K40                     | 优化了功耗和性能，支持 NVLink 和 HBM2                 | 支持 CUDA 5 至 CUDA 10    |
| Maxwell      | 2014年     | GeForce GTX 980, Quadro M6000                 | 提高能效比，支持更高的图形质量                         | 支持 CUDA 6 至 CUDA 11    |
| Pascal       | 2016年     | GeForce GTX 1080, Tesla P100                   | 引入 NVLink 和 HBM2，显著提升计算能力                 | 支持 CUDA 8 至今          |
| Volta        | 2017年     | Tesla V100, Titan V                            | 引入 Tensor Cores，优化深度学习性能                   | 支持 CUDA 9 至今          |
| Turing       | 2018年     | GeForce RTX 2080, Quadro RTX 8000             | 第二代 Tensor Core， 实时光线追踪技术，支持 DLSS                         | 支持 CUDA 10 至今         |
| Ampere       | 2020年     | A100/GeForce RTX 30 系列                        | 第三代 Tensor Core                             | 支持 CUDA 11 至今         |
| Hopper       | 2022年     | H100                                          | 针对 HPC 和 AI 性能优化，支持 MIG 功能                 | 最新架构，适合复杂计算    |
| Blackwell    | 2024年（预计）| 待发布                                       | 针对生成式 AI 工作流优化，能效大幅提升                 | 新一代架构                |

以下是包含 NVIDIA 常用设备、架构、特点/用途以及其他相关信息的表格

| 设备型号         | 类型                     | 架构          | 特点/用途                                              | 是否可炼丹 | 
|------------------|--------------------------|---------------|-------------------------------------------------------|------------|
| **NVIDIA A100**  | 企业级 GPU               | Ampere        | 高性能计算，支持混合精度训练，适合大规模深度学习和 HPC 工作负载。| 40GB HBM2，支持多实例GPU技术 |
| **NVIDIA V100**  | 企业级 GPU               | Volta         | 适合高性能计算和深度学习，提供强大的计算能力和内存带宽。        | 32GB HBM2，支持 NVLink    |
| **NVIDIA T4**    | 中端 GPU                 | Turing        | 性价比高，适合推理任务和入门级训练。                          | 16GB GDDR6，低功耗设计    |
| **NVIDIA RTX 4090** | 消费级 GPU              | Ada Lovelace   | 强大的计算能力，适合高端游戏和深度学习推理。                    | 24GB GDDR6X，支持光追技术  |

 

