
# basic
**CUDA核心**: 单精度(FP32)和整数(INT32)运算，核心是算术逻辑单元(ALU)
 - **FP32单元**  CUDA核心的一部分，在NVIDIA的架构中,FP32能力通常等同于CUDA核心的数量
**Tensor核心**: 为深度学习等AI工作负载优化的矩阵乘法加速单元，可以处理混合精度运算,包括FP16、FP32，TF32和INT8等。


# 计算流程
1. 数据从**HBM**加载数据：（主要全局内存）
2. **HBM** -> **L2缓存**：
3. **进入 SRAM** :  **L2缓存** ->  **SM L1缓存或共享内存(`基于SRAM技术`)**：
   - **L1缓存**：为每个线程提供快速的数据访问，但其容量较小。
   - **共享内存**：在同一个SM内的所有线程之间共享，允许线程间的高效数据交换。

4. **执行计算**：
   - 从L1缓存或共享内存读取数据。
   - 在ALU（算术逻辑单元）中执行实际的计算操作。- (每个SM包含多个CUDA核心，可以并行执行多个线程)

5. **结果存回**：-> L1缓存或共享内存 -> L2缓存 -> HBM

```
CPU (Host)
    ↓
准备数据并传输到GPU
    ↓
PCIe/NVLink
    ↓
HBM (High Bandwidth Memory)
    ↓
  L2 Cache
    ↓
L1 Cache / Shared Memory (in SM)
    ↓
CUDA Cores (perform calculations)
    ↓
L1 Cache / Shared Memory (write back)
    ↓
  L2 Cache
    ↓
HBM (write back)
    ↓
PCIe/NVLink
    ↓
传输回CPU
    ↓
CPU (处理结果)

```
