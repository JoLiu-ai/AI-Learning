# 什么是Rejection Sampling
拒绝采样（Rejection Sampling）是一种用于从复杂概率分布中抽取样本的统计方法，尤其在直接从目标分布中采样不易时非常有用。该方法依赖于一个易于采样的简单分布（称为提议分布或proposal distribution），通过构造接受和拒绝的区域来实现。

# 解决什么问题

# 怎么做的

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义目标分布 P(x)
def target_distribution(x):
    return 0.3 * np.exp(-(x - 0.3)**2) + 0.7 * np.exp(-(x - 2)**2 / 0.3)

# 定义提议分布 Q(x)
def proposal_distribution(x):
    return (1 / np.sqrt(2 * np.pi * 1.2**2)) * np.exp(-0.5 * ((x - 1.4)**2 / 1.2**2))

# 拒绝采样过程
def rejection_sampling(num_samples):
    samples = []
    k = 2.5  # 常数 k 的选择
    while len(samples) < num_samples:
        x0 = np.random.normal(1.4, 1.2)  # 从提议分布中抽样
        u = np.random.uniform(0, k * proposal_distribution(x0))  # 从均匀分布中抽样
        if u <= target_distribution(x0):  # 检查接受条件
            samples.append(x0)
    return samples

# 执行拒绝采样
samples = rejection_sampling(10000)

# 绘制结果
plt.hist(samples, bins=50, density=True, alpha=0.5, label='Sampled Distribution')
x = np.linspace(-4, 6, 100)
plt.plot(x, target_distribution(x), color='red', label='Target Distribution')
plt.plot(x, proposal_distribution(x), color='blue', label='Proposal Distribution')
plt.legend()
plt.show()
```

![image](https://github.com/user-attachments/assets/e654b8ec-486c-4774-8446-ffd6c960b631)

