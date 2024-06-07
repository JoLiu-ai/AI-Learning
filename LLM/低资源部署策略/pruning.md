![image](https://github.com/hinswhale/AI-Learning/assets/22999866/e1e45194-b2c2-4b70-9ad3-cfe5416d7302)![image](https://github.com/hinswhale/AI-Learning/assets/22999866/f3b65872-9c41-4cf0-b020-b4b3040df6c4)![image](https://github.com/hinswhale/AI-Learning/assets/22999866/35f5b8e0-1369-4b00-bbed-5c2de20dcead)![image](https://github.com/hinswhale/AI-Learning/assets/22999866/b13add41-a599-4bc9-aea4-4463b22ef22b)
## Pruning vs Quantization
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/952d87bf-39c9-4eb1-ba44-2490a9e14c34)


1. 在内存占用相同情况下，大稀疏模型比小密集模型实现了更高的精度。
2. 经过剪枝之后稀疏模型要优于同体积非稀疏模型。
3. 资源有限的情况下，剪枝是比较有效的模型压缩策略。
4. 优化点还可以往硬件稀疏矩阵储存方向发展

## 剪枝算法分类
### Unstructured Pruning（非结构化剪枝）
### Structured Pruning（结构化剪枝）
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/626909fd-3c7c-49e5-b205-6f9877cd1a6c)

| Unstructured Pruning（非结构化剪枝） | Structured Pruning（结构化剪枝） |
|--------------------------------------|------------------------------------|
|随机对独立的权重或者神经元链接进行剪枝   | 大部分算法在 filter/channel/layer 上进行剪枝      |
| 算法简单, 精度不可控, 模型压缩比高   | 剪枝算法相对复杂                                   |
| 剪枝后权重矩阵稀疏，没有专用硬件难以实现压缩和加速的效果 | 保留原始卷积结构，不需要专用硬件来实现 |

## 剪枝流程
### 方法
🍀 训练一个模型 -> 对模型进行剪枝 -> 对剪枝后模型进行微调  
🍀 在模型训练过程中进行剪枝 -> 对剪枝后模型进行微调  
🍀 进行剪枝 -> 从头训练剪枝后模型（应用少, 消耗大） 


### 流程三大件
🍄 训练 Training：训练过参数化模型，得到最佳网络性能，以此为基准；   
🍄 剪枝 Pruning：根据算法对模型剪枝，调整网络结构中通道或层数，得到剪枝后的网络结构；  
🍄 微调 Finetune：在原数据集上进行微调，用于重新弥补因为剪枝后的稀疏模型丢失的精度性能。  

<img width="817" alt="image" src="https://github.com/hinswhale/AI-Learning/assets/22999866/b306832d-b448-44b0-8cfe-8de49cdf997f)">

<img width="817" alt="image" src="https://github.com/hinswhale/AI-Learning/assets/22999866/139dcd8d-3104-4e25-ae4d-596aa00b87b9)">

## 算法
### L1-norm based Channel Pruning
使用 L1-norm 标准来衡量卷积核的重要性

<img width="847" alt="image" src="https://github.com/hinswhale/AI-Learning/assets/22999866/704d8681-6d93-41eb-adfc-fb255c905420">




# 资料
1. [模型剪枝核心原理](https://www.bilibili.com/video/BV1y34y1Z7KQ/)
