LoRA, DoRA, AdaLoRA, Delta-LoRA

![image](https://github.com/user-attachments/assets/93e4d43e-882f-4f8b-aade-dd40288f6c9b)
> from: https://www.dailydoseofds.com/understanding-lora-derived-techniques-for-optimal-llm-fine-tuning/

![image](https://github.com/user-attachments/assets/2f490fd8-09f7-4264-b928-fefd937b865d)


### LoRA(Low-Rank Adaption)
![image](https://github.com/user-attachments/assets/6ffeaf11-17ec-4bb1-9aa7-d5de44d3c3a5)
LoRA的一个技术细节是:在开始时，矩阵A被初始化为均值为零的随机值，但在均值周围有一些方差。矩阵B初始化为完全零矩阵
> [https://mp.weixin.qq.com/s/-_JqRklaRI9bD_6QQGKrjg](https://towardsdatascience.com/an-overview-of-the-lora-family-515d81134725)

### LoRA+
![image](https://github.com/user-attachments/assets/2868b12e-397a-4ae2-be4a-0d67c60bf39f)
为矩阵a和b引入不同的学习率

### VeRA(Vector-based Random Matrix Adaptation)
![image](https://github.com/user-attachments/assets/b4788af2-8779-495e-914e-667aa5a12c2c)


### LoRA-FA
![image](https://github.com/user-attachments/assets/5f8b6ae8-b21d-4c83-af5e-b06e61ccd6d9)

### LoRa-drop
![image](https://github.com/user-attachments/assets/96e7bc77-5501-4aa0-a05f-758ab33927e9)

### AdaLoRA
在 LoRA 的基础上加入了自适应机制，使得低秩矩阵的秩大小能够在训练过程中进行动态调整
低秩矩阵的秩并不是预先设定的固定值，而是在训练过程中根据模型表现动态调整

### DoRA
![image](https://github.com/user-attachments/assets/6d129f2d-4a32-4a9f-abe5-c369284cb5ee)
在LoRA方法的基础上，通过引入“增量矩阵”（Delta matrix）来进一步减少训练参数数量
不直接优化低秩矩阵的所有元素，而是仅对低秩矩阵中的“增量部分”进行微调。它将低秩矩阵分为一个初始矩阵和一个增量矩阵，并只训练增量部分，从而减少训练的参数数量

### Delta-LoRA

### QLoRA
![image](https://github.com/user-attachments/assets/b692e016-ac30-4e67-b934-4027f096b58f)


资料
https://www.dailydoseofds.com/understanding-lora-derived-techniques-for-optimal-llm-fine-tuning/





