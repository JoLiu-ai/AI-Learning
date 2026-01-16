## pd 分离（Prefill/Decode Separation）
### **Prefill**（预填充）：
  - 输入完整Prompt，负责一次性对所有输入 token 做前向计算，生成并缓存各层的 Key/Value 向量，供后续 decode 阶段重复利用；该阶段不进行任何自回归生成
  - 计算密集型（compute‑bound），需高算力
  - 预填充所需要的时间 = number of tokens * ( number of parameters / accelerator compute bandwidth) 
  **GPU并行计算** & **GPU 显存**
###  **Decode**（解码）：
  - decode 阶段读取 prefill 构建的 KV Cache，以自回归方式生成第一个及后续 Token，每次生成一个 token，需将其添加到已生成的序列中，需频繁访问/更新 KV Cache。
  - 访存密集型（memory‑bound），需高频显存访问(主要瓶颈在于对KV Cache的读写带宽)
  - time/token = total number of bytes moved (the model weights) / accelerator memory bandwidth
**GPU 的显存带宽和容量**
### pd 分离原理
pd 分离将推理过程拆分为计算密集型的 **预填充（prefill）** 与内存密集型的 **解码（decode）** 两阶段，在硬件或实例层面分离，以消除两阶段间的资源竞争并分别优化算力与带宽，
从而显著提升吞吐（rps）并降低首 token 延迟（TTFT）与 TPOT

优化点：根据不同阶段的资源消耗特性，合理分配硬件资源。

优化策略包括：​
- Prefill 优化：
  - 采用分块处理（Chunked Prefill）技术，减少一次性计算量。
  - 利用更高效的矩阵运算库，提升计算速度。​

- Decode 优化：
  - 优化 KV 缓存的存储结构，减少内存访问延迟。
  - 采用更高效的解码算法，提升生成速度。
 
总推理时间 = Prefill 阶段时间 + Decode 阶段时间 = TTFT + (TPOT * 生成 token 的总数)

TTFT ≈ Prefill 时间 + Decode 首 Token 时间  
TPOT ≈ 后续 Decode 时间  
Prefill 加权资源可显著降低 **TTFT**，Decode 加权资源可显著降低 **TPOT**  ，
Prefill 卡上尽可能做**大批量并行计算**，Decode 卡上尽量**减少批次大小**以低延迟。 


### 优势
- rps (吞吐量)倍增：一般可实现 2–4.5× 的整体 rps 提升​
- 延迟可控：可独立按需扩缩 prefill 节点以降低 TTFT，或扩缩 decode 节点以降低 TPOT，灵活满足不同 SLO。

资源隔离：避免 compute‑bound 与 memory‑bound 在同实例上相互抢占，提升硬件利用率与成本效率。

### 缺点与挑战
- **网络/通信开销**：拆分部署后，KV Cache （可达数十 GB）需在 Prefill 与 Decode 之间传输；跨机部署则加剧网络带宽压力。 
- **调度复杂度**：需动态调度请求到不同实例，保证负载均衡与 QoS


### 相关性能指标
- rps（吞吐量）
- Time To First Token (TTFT): 从输入到输出第一个 token 的时延，**实时聊天场景优先**
- Time Per Output Token (TPOT): 在首个 Token 之后，每生成一个新 Token 所需的平均时延，**实时聊天场景优先**
- latency = (TTFT) + (TPOT) * (the number of tokens to be generated).
   > Tokens Per Second (TPS)：TPS = (the number of tokens to be generated) / Latency
- Throughput：吞吐量，即每秒针对所有请求生成的 token 数
  ![image](https://github.com/user-attachments/assets/614fcdf4-d95f-4b65-a0b1-c0d2cf195030)
- SLO（Service Level Objective）：如首 Token 延迟（TTFT）、每 Token 延迟（TPOT）及吞吐等指标
- QoS（Quality of Service，服务质量）
  - 带宽：​分配给特定应用的数据传输速率。
  - 延迟（Latency）：​数据从源头到目的地的传输时间。
  - 抖动（Jitter）：​数据包到达时间的变化，影响实时应用的稳定性。
  - 丢包率：​在传输过程中丢失的数据包比例。​

## 资料
- https://www.cnblogs.com/menkeyi/p/18778750
- https://www.cnblogs.com/menkeyi/p/18767869


 ## ep 分离（Expert Parallelism Separation）
 - 在模型内部专注于  **MoE（Mixture‑of‑Experts）**架构，将多个专家子网络分布到不同设备，借助门控网络动态路由输入至少量专家，降低单请求的计算与通信开销，同时支持极大参数量的模型扩展​
