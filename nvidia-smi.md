
`nvidia-smi` 是 NVIDIA 提供的一种工具，用于监控和管理 NVIDIA GPU。它的全称是 "NVIDIA System Management Interface"。通过 `nvidia-smi`，用户可以查看 GPU 的运行状况、性能指标以及进行一些管理操作。以下是 `nvidia-smi` 输出的一些关键部分的解析：

### 1. 简单执行 `nvidia-smi` 的输出
当你在命令行中运行 `nvidia-smi` 时，输出通常类似如下：

```plaintext
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |             MIG M.   |
|===============================+======================+======================|
|   0  Tesla K80           On   | 00000000:00:1E.0 Off |                    0 |
| N/A   41C    P8    27W / 149W |    193MiB / 11441MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     29683      C   python                                       181MiB |
+-----------------------------------------------------------------------------+
```

### 2. 输出解析

- **NVIDIA-SMI 460.32.03**: NVIDIA-SMI 工具的版本号。
- **Driver Version: 460.32.03**: 安装的 NVIDIA 驱动程序版本号。
- **CUDA Version: 11.2**: 支持的 CUDA 版本。

#### GPU 信息部分

| 列名 | 描述 |
|---|---|
| GPU | GPU 的索引号，从 0 开始。 |
| Name | GPU 的型号名称，例如 Tesla K80。 |
| Persistence-M | 持久模式状态，On 或 Off。持久模式减少了在 GPU 空闲时启动 GPU 的延迟。 |
| Bus-Id | GPU 的总线 ID，表示 GPU 在 PCIe 总线上的位置。 |
| Disp.A | 显示属性，表示 GPU 是否连接到显示器。 |
| Volatile Uncorr. ECC | 易失性不可纠正的 ECC 错误数。 |

#### 性能和温度部分

| 列名 | 描述 |
|---|---|
| Fan | 风扇速度，百分比表示。 |
| Temp | GPU 温度，摄氏度。 |
| Perf | 性能状态，从 P0（最高性能）到 P12（最低性能）。 |
| Pwr:Usage/Cap | GPU 当前功耗和功耗上限，单位是瓦特。 |
| Memory-Usage | GPU 内存使用情况，显示已用内存和总内存。 |
| GPU-Util | GPU 利用率，百分比表示。 |
| Compute M. | 计算模式，表示 GPU 是否被配置用于计算任务。Default/Exclusive-Process/Exclusive-Thread.Prohibited |
| MIG M. | 多实例 GPU (MIG) 模式，表示是否启用和配置。 |

#### 进程信息部分

| 列名 | 描述 |
|---|---|
| GPU | 使用 GPU 的索引号。 |
| PID | 进程 ID。 |
| Type | 进程类型，C 表示计算进程（CUDA 应用），G 表示图形进程。 |
| Process name | 进程名称。 |
| GPU Memory Usage | 进程使用的 GPU 内存，单位是 MiB。 |

### 常见命令

- `nvidia-smi`: 显示 GPU 的总体信息和当前活动的进程。
- `nvidia-smi -l 1`: 每隔 1 秒刷新一次输出，持续监控 GPU 状态。
- `nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.total,memory.used --format=csv`: 查询并以 CSV 格式显示特定的 GPU 信息。

### 进阶用法

- **管理 GPU**: `nvidia-smi` 还可以用于管理 GPU，例如设置持久模式、调整风扇速度、设定功耗限制等。
- **监控多 GPU 系统**: 如果系统中有多个 GPU，可以使用 `nvidia-smi` 来查看每个 GPU 的详细状态。


### 重要指标
- 利用率 GPU-Util
- 内存使用 Memory-Usage 和 GPU Memory Usage
- 监控 GPU 的功耗  Pwr/Cap
- 使用 dmesg 命令查看系统日志

### 使用
- CUDA_VISIBLE_DEVICES  限制每个进程可见的 GPU
- 
