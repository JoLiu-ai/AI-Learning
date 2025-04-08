|  |  | 特点 | 论文 |
| --- | --- | --- | --- |
| multi-head attention (MHA)  | 标准的多头注意力机制 | 所有注意力头的 Key 和 Value 矩阵权重不共享 |  |
| Grouped Query Attention (GQA) | 分组查询注意力，折中 | GQA将查询头分成G组，每个组共享一个Key 和 Value 矩阵 | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints |
| multi-query attention (MQA)  |  | 所有的头之间共享同一份 Key 和 Value 矩阵，每个头只单独保留了一份 Query 参数 | Fast Transformer Decoding: One Write-Head is All You Need |

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/4d7a246c-8241-45b8-ab02-9865a2afe392)


MHA
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/e5f5158b-3bfa-4454-af0d-31088be436e9)


MQA
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/2fc0fe7a-ad58-4a7b-a523-2cbe298e01fe)


GQA
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/8066df05-3354-4fa8-a204-20326c2c2f39)

## todo :code
