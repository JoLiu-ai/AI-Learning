Masked Self-attention

# **归一化函数（Normalizing Function）**

## 方法

### LN

Layer Normalization 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/5b7b25d3-7b74-41be-b1ea-0c7fc2bc95be/Untitled.png)

### RMSNorm

Root Mean Square Layer Normalization  均方根层归一化

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/20b38e1f-e166-4af9-8b38-744987ff248a/Untitled.png)

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

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/7c906cc9-2e61-4696-9cf7-a158e424c73c/Untitled.png)

## 归一化模块位置

### 层后归一化（Post-Layer Normalization, Post-Norm）

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/7d4608aa-4448-4a0e-8d7e-44dd8cc40913/Untitled.png)

### 层前归一化（Pre-Layer Normalization, Pre-Norm）

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/b6bb1444-c1c7-4f35-a75e-acb77f064d88/Untitled.png)

### 夹心归一化（Sandwich-Layer Normalization, SandwichNorm）

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/cfb0d416-7773-4520-bf28-70647a743ba7/Untitled.png)

## **激活函数**

### ReLU

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/8a6ffff3-8de8-4860-9be0-9fcf2c77c8d6/Untitled.png)

ReLU 可能会产生神经元失效的问题，被置为 0 的神经元将学习不到有用的信息。

### **SwiGLU**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/a775e921-3ba8-44e2-b09c-2e993a1c7545/Untitled.png)

### GELU

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/d039e34c-f784-42c9-9d9b-216a960340c1/Untitled.png)

### GLU(Gated Linear Unit）

变种 SwiGLU 和 GeGLU

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/d4cf5046-bf97-4e99-94df-243f286e4391/Untitled.png)

### **位置嵌入**

### 绝对位置编码

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/a5ba1f75-5499-4e97-9d46-e34b33fd91b7/Untitled.png)

### 相对位置编码

RoPE

ALiBi
