# 什么是量化

# 量化原理
## 对称量化和非对称量化（absolute maximum quantization和zero-point quantization）。

# 量化分类
* 根据量化后的`目标区间`，
  ** 二值量化（1, -1）
  ** 三值量化（-1, 0, 1）
  ** 定点数量化（INT4, INT8）【最常见】
  ** 2 的指数量化。
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/968b6ef7-def1-464a-87ce-718440ba3696)
* 根据量化节点的`分布`，可以分为均匀量化和非均匀量化。
  ![image](https://github.com/hinswhale/AI-Learning/assets/22999866/8034d4d1-ce60-4577-9221-9c9498786b87)
非均匀量化:根据待量化参数的概率分布计算量化节点。如果某一个区域参数取值较为密集，就多分配一些量化节点，其余部分少一些。这样量化精度较高，但计算复杂度也高
均匀量化: LLM 常用
* 均匀量化
** 对称量化
** 非对称量化
# 量化方法

# 工具
