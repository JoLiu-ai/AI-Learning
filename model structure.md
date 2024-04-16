Masked Self-attention

# **归一化函数（Normalizing Function）**

## LN

### Layer Normalization 



<img src="https://github.com/hinswhale/AI-Learning/assets/22999866/9c42ce87-20e9-4b48-80d2-04764e6d787f" alt="image" width="500" />

### RMSNorm

Root Mean Square Layer Normalization  均方根层归一化

<img src="https://github.com/hinswhale/AI-Learning/assets/22999866/a7b8d0e2-c2ec-4f9a-b8a2-8af9efae3a33" alt="image" width="500" />
```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """ 
        LlamaRMSNorm is equivalent to T5LayerNorm 
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps  # eps 防止取倒数之后分母为 0 

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # weight 是末尾乘的可训练参数, 即 g_i 

        return (self.weight * hidden_states).to(input_dtype)
```

### DeepNorm

通过这一简单的操作， Transformer 的层数可以被成功地扩展至 1,000 层，进而有效提升了模型性能与训练稳定性

- GLM-130B
- ![image](https://github.com/hinswhale/AI-Learning/assets/22999866/4ca0a201-a1c7-4f34-8515-a291cae7e7a3)

## 归一化模块位置

### 层后归一化（Post-Layer Normalization, Post-Norm）
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/765e80fa-d21e-43b0-a5ae-170a2700fd22)


### 层前归一化（Pre-Layer Normalization, Pre-Norm）
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/3defdee6-52d1-4680-97ea-89a9ebc65651)

### 夹心归一化（Sandwich-Layer Normalization, SandwichNorm）
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/0bdb2087-d50e-4d9b-ac44-5fc417beb0e0)


## **激活函数**

### ReLU
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/6ede9ab8-2cd6-4781-bce1-ddf78ec87375)


ReLU 可能会产生神经元失效的问题，被置为 0 的神经元将学习不到有用的信息。

### **SwiGLU**
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/d80cf296-9188-45b4-b97d-09f97224a632)


### GELU
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/a7d8762d-66a4-44bc-9694-cc6ef7847dab)


### GLU(Gated Linear Unit）

变种 SwiGLU 和 GeGLU
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/31ce5496-73a9-4522-8279-cb96df949119)


### **位置嵌入**

### 绝对位置编码

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/7d581b0d-eb04-44e5-a0c9-ea7c4022042f)


### 相对位置编码

RoPE

ALiBi
