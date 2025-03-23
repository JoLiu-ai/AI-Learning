## 什么是 `KL散度`
KL散度（Kullback-Leibler Divergence）是衡量两个概率分布 P和 Q差异的非对称性指标

给定两个概率分布P和Q，从P到Q的KL散度定义为：
> KL散度衡量的是：当我们用分布Q来近似真实分布P时，所产生的信息损失

离散形式：

$D_{KL}(P \parallel Q) = \sum_{i} P(i) \ln \left( \frac{P(i)}{Q(i)} \right)$

连续形式：  
$D_{KL}(P \parallel Q) = \int_{-\infty}^{\infty} p(x) \ln \left( \frac{p(x)}{q(x)} \right) \, dx$
## 性质：
- 非对称性 $D_{KL}(P∥Q)≠D_{KL}(Q∥P)$，KL散度不满足交换律。
- 非负性 $D_{KL}(P∥Q)≥0$，当且仅当 P=Q 时取零。
- 与交叉熵的关系 $D_{KL}(P∥Q)=H(P,Q)−H(P)$，其中：
  - $H(P)=−∑P(i)ln⁡P(i)$ 是分布 P 的熵；
  - $H(P,Q)=−∑P(i)ln⁡Q(i)$ 是交叉熵。

## 正向KL vs 反向KL
正向KL: $D_{KL}(P||Q)$    
反向KL: $D_{KL}(Q||P)$   

| **特性** | **正向KL：DKL(P∥Q)** | **反向KL：DKL(Q∥P)** |
| --- | --- | --- |
| 期望来源 | 基于P | 基于Q |
| 行为倾向 | 零回避(Zero-Avoiding)、平均寻找(Mean-seeking) | 零强制(Zero-Forcing)、模式寻找(Mode-seeking) |
| 多峰分布时 | Q覆盖P的所有模式（避免Q=0） | Q专注于单一主要峰 |
| 错误惩罚 | P高Q低时惩罚大 | Q高P低时惩罚大 |
| 近似效果 | 更"平滑"、覆盖广 | 更"锐利"、专注窄 |
| 应用场景 | 知识蒸馏（传统分类任务） | 变分推断、RLHF、模型压缩 |
| 生成结果 | 分布宽泛（Mean-Seeking） | 分布集中（Mode-Seeking） |

![image](https://github.com/user-attachments/assets/ca47a7b3-a082-49d4-9d41-4bf0d8665ace)
