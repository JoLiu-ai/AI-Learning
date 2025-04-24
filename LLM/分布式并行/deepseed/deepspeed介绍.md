
DeepSpeed 是一个深度学习优化库，旨在通过一系列系统优化策略提高训练大型模型的效率和可扩展性。以下是 DeepSpeed 的详细介绍：

## 🧠 核心训练优化策略
### 1. ZeRO (Zero Redundancy Optimizer)
ZeRO 是 DeepSpeed 的核心技术，通过分片存储模型状态（参数、梯度、优化器状态）来降低单卡显存占用：​
- ZeRO-1：​分片优化器状态。

- ZeRO-2：​进一步分片梯度。

- ZeRO-3：​连模型参数也进行分片，实现最大化的内存节省

此外，**ZeRO-Offload** 和 **ZeRO-Infinity** 扩展了 ZeRO-3，将部分状态卸载到 CPU 或 NVMe 存储，进一步减少 GPU 内存需求。

---

### 2. 3D 并行（3D Parallelism）
  - **数据并行（Data Parallelism）**：将**训练数据**分批分配到多个 GPU 并行计算。
  - **张量并行（tensor Parallelism）**：	将**单层模型参数**拆分到多设备（如矩阵乘法分块）
  - **流水线并行（Pipeline Parallelism）**：将**模型按层切分**到多设备，通过流水线调度 micro-batches

关键区别：
- 模型并行针对 单层内部 的并行计算，适合**参数密集型**的层（如大矩阵运算）；
- 流水线并行针对 层间 的流水线调度，适合**层数多**的模型（如深度神经网络）。

---

### 3. 混合精度训练（Mixed Precision Training）
通过使用半精度（fp16）计算，减少内存占用和加速训练过程，同时保持模型精度

- **动态损失缩放（Dynamic Loss Scaling）** 自动调整损失缩放因子，避免梯度下溢，提高训练稳定性
- **混合精度优化器（Mixed Precision Optimizer）** 支持在混合精度训练中使用高效的优化器，如 fused Adam，提升训练性能

-  稀疏注意力（Sparse Attention）
  DeepSpeed 支持稀疏注意力机制，通过降低注意力计算复杂度，减少长序列模型的计算和内存需求。这对处理长文本或长序列输入的模型尤为重要，同时保持模型性能。
- 模型压缩
  DeepSpeed 提供模型压缩技术，如 Progressive Layer Dropping，通过逐步丢弃不重要层减少计算和内存需求，同时保持模型准确性，从而加速训练。

---


## DeepSpeed 分布式启动器命令说明

| 参数 | 含义 | 示例 |
|------|------|------|
| `--master_port` | 主节点端口号 | `--master_port 29500` |
| `--master_addr` | 主节点 IP | `--master_addr=10.51.97.28`（查看方式：`ifconfig` -> `eth0` -> `inet`） |
| `--nnodes` | 节点数 | 两台机器：`--nnodes=2` |
| `--node_rank` | 节点 rank，从主节点 0 开始递增 | `--node_rank=0` 为主节点 |
| `--nproc_per_node` | 每个节点的进程数 | 使用 8 张 GPU 卡：`--nproc_per_node=8` |

---

## NCCL（NVIDIA Collective Communications Library）通信参数说明

| 参数 | 含义 | 说明 |
|------|------|------|
| `NCCL_IB_DISABLE` | 禁用 IB 网卡传输端口 | IB（InfiniBand）是用于高性能计算的通信标准 |
| `NCCL_SHM_DISABLE` | 禁用共享内存传输 | 共享内存用于同一节点进程间的快速通信 |
| `NCCL_P2P_DISABLE` | 禁用 GPU 之间的点对点通信 | P2P 通信通过 CUDA 和 NVLink 实现 GPU 直连 |

---

## 检查 GPU 是否支持 NVLink

使用以下命令：

```bash
nvidia-smi topo -p2p n
```
