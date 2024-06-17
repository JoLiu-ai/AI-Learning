### 简介

`torchrun`
     - torchrun可以自动分配worker的`RANK`和`WORLD_SIZE`

### 流程
1. **初始化进程组**：
    - 使用 **`dist.init_process_group`** 初始化进程组，指定后端、端口、世界大小和当前进程的排名。
2. **设置分布式采样器**：
    - 使用 **`DistributedSampler`** 设置分布式采样器，用于分配数据到多个进程中。
3. **使用 `DistributedDataParallel` 封装模型**：
    - 使用 **`DistributedDataParallel`** 封装模型，实现模型的分布式训练。
4. **使用 `torchrun` 启动分布式训练**：
    - 使用 **`torchrun`** 命令启动分布式训练，指定参数如 **`-nproc_per_node`**、**`-master_addr`**、**`-master_port`**、**`-nnodes`** 等。
        - **--nproc_per_node**：指定每个节点（机器）上的进程数
        - **--nnodes**：指定总共的节点数
        - **--node_rank**：指定当前节点（机器）的排名
        - **--rdzv_backend**：指定分布式后端的类型，可以是nccl、mpi等

### 代码

```python
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

# 初始化进程组
dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=ngpus_per_node, rank=gpu)

# 设置分布式采样器
sampler = DistributedSampler(data, num_replicas=ngpus_per_node, rank=gpu)

# 使用 DistributedDataParallel 封装模型
model = DistributedDataParallel(model, device_ids=[gpu])

# 销毁进程组
dist.destroy_process_group()
```
### 使用 torchrun 启动分布式训练
```sh
torchrun --nproc-per-node=2 ddp_gpus_torchrun.py --max_epochs 5 --batch_size 32

python -m torch.distributed.launch --use-env --nproc-per-node=2 ddp_gpus_torchrun.py --max_epochs 5 --batch_size 32
```
