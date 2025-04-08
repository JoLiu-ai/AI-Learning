processing：未完成

### Prompt Tuning
- add **soft prompt**：使用一组 手动设计的提示 来帮助模型更好地理解任务
- 通过引入 虚拟 token（通常是“嵌入的提示”）到输入序列中，这些虚拟 token 用于引导模型进行更合适的生成或推理任务。整个模型的其余部分（如 transformer 层的权重）保持不变，仅通过学习这些虚拟 token 的嵌入来调整模型的行为  
```[虚拟 token 1] + [虚拟 token 2] + ... + "文本内容" ```

- 学习结构化的提示 token 嵌入
- 输入的提示固定长度：Prompt Tuning 在输入文本中加入一个固定长度的提示（虚拟 token 的嵌入），通过微调这些嵌入来优化模型性能。
- 嵌入的位置：通常是输入文本的前或后部分
- 学习固定长度的软提示标记

伪代码
```python
# 学习可训练的软提示
soft_prompt = nn.Parameter(torch.randn(num_tokens, embedding_dim))
input_embeddings = torch.cat([soft_prompt, original_embeddings], dim=0)
```

#### 代码
配置
```python
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=model_name_or_path,
)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

**reasoning**  
```python
from peft import PeftModel, PeftConfig

peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

# 加载PEFT配置
config = PeftConfig.from_pretrained(peft_model_id)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
# 加载PEFT模型
model = PeftModel.from_pretrained(model, peft_model_id)

# Tokenizer编码
inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', return_tensors="pt")

# 模型推理
outputs = model.generate(
        input_ids=inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        max_new_tokens=10, 
        eos_token_id=3
    )

# Tokenizer 解码
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
```


#### 源码

```python
class PromptEmbedding(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        super().__init__()

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)
        
        if config.prompt_tuning_init == PromptTuningInit.TEXT:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
            init_text = config.prompt_tuning_init_text
            init_token_ids = tokenizer(init_text)["input_ids"]
            
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]

            word_embedding_weights = word_embeddings(torch.LongTensor(init_token_ids)).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

    def forward(self, indices):
        # 这里的 indices 是虚拟 token 的索引
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings
```

### P-Tuning
- 可微的virtual token，但仅限于输入层，没有在每一层都加
- 位置也不一定是前缀，插入的位置是可选的
- 使用提示编码器增强提示的语义表达

伪代码
```python
# 使用LSTM编码提示
prompt_encoder = nn.LSTM(...)
soft_prompt = prompt_encoder(learnable_tokens)
```

```python
peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20, encoder_hidden_size=128)
```


#### 源码
```python
class PromptEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_dim = config.token_dim
        self.input_size = self.token_dim
        self.output_size = self.token_dim
        self.hidden_size = config.encoder_hidden_size
        self.total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.encoder_type = config.encoder_reparameterization_type

        # 初始化 embedding 层
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.token_dim)
        if not config.inference_mode:
            # 根据PromptEncoder重参数化类型初始化相应的lstm和mlp
            if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
                lstm_dropout = config.encoder_dropout
                num_layers = config.encoder_num_layers
                # LSTM
                self.lstm_head = torch.nn.LSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=num_layers,
                    dropout=lstm_dropout,
                    bidirectional=True,
                    batch_first=True,
                )

                self.mlp_head = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size * 2, self.output_size),
                )

            elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
                warnings.warn(
                    f"for {self.encoder_type}, the `encoder_num_layers` is ignored. Exactly 2 MLP layers are used."
                )
                layers = [
                    torch.nn.Linear(self.input_size, self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.output_size),
                ]
                self.mlp_head = torch.nn.Sequential(*layers)

            else:
                raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")
    def forward(self, indices):
        input_embeds = self.embedding(indices)
        if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
            output_embeds = self.mlp_head(input_embeds)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

        return output_embeds
```

### Prefix Tuning
关键特征是：
- 在每一层transformer层的输入前添加可学习的前缀（prefix）
- 前缀是连续的可学习向量
- 前缀被添加到所有注意力层的键（key）和值（value）向量中
- 模型参数仍然保持冻结

伪代码
```python
# 为每一层学习前缀
prefix_keys = nn.Parameter(torch.randn(num_layers, num_heads, prefix_length, hidden_size))
prefix_values = nn.Parameter(torch.randn(num_layers, num_heads, prefix_length, hidden_size))
```

### Adapter Tuning
插入少量可训练的小型神经网络模块

### LoRA (Low-Rank Adaptation)
- 在原始模型参数旁边添加低秩矩阵
- 仅训练这些低秩矩阵
- 参数量极少，计算开销小

**公式**：

LoRA（Low-Rank Adaptation）通过将模型权重 W 分解为两个低秩矩阵 A 和 B ，并在每次前向传播时计算一个低秩调整项 $$\Delta$$ W ，从而对原始权重进行调整。

调整后的权重计算公式为：

$$
W' = W + \Delta W = W + \frac{\alpha}{r} \cdot A \cdot B
$$

其中：
- W 是原始的模型权重矩阵。
- A 和 B 是低秩矩阵，表示对权重矩阵 W 的调整。
- r 是秩（rank），控制低秩矩阵的大小。
- $$\alpha$$ 是 LoRA 缩放因子，用于调整低秩矩阵的影响力。
- A $$\cdot$$ B 是低秩矩阵的乘积，表示对原始权重矩阵 W 的增量。
其中：
$𝐴 \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{r \times d}$是低秩矩阵,r 是秩（rank），通常设置为较小的值。
训练时，仅优化𝐴和 𝐵，而不更新 𝑊

> - Transformer的权重矩阵包括Attention模块里用于计算query, key, value的 $$W_q$$ ， $$W_k$$ ， $$W_v$$ 以及多头attention的 $$W_o$$ ,以及MLP层的权重矩阵
> - LoRA只应用于Attention模块中的4种权重矩阵，而且通过消融实验发现同时调整 $$W_q$$ 和 $$W_v$$会产生最佳结果。默认的模块名基本都为 $$W_q$$ 和 $$W_v$$ 权重矩阵。


![image](https://github.com/user-attachments/assets/4bdfcf8c-c1c8-4e75-8a8c-0858d91f15ab)

```python
from peft import LoraConfig
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)
```
> - task_type：指定任务类型。如： Causal Language Modeling（因果语言建模）, SEQ2SEQ_LM（序列到序列语言模型，如 T5）、TOKEN_CLASSIFICATION（标注任务)等。  
> - inference_mode：是否在推理模式下使用Peft模型。  
> - r： 低秩分解中矩阵𝐴和𝐵的秩,常用值：4、8、16,越大性能越好，但参数量增加
> - lora_alpha:LoRA 的缩放因子，用于调节 LoRA 权重的影响力,典型范围：1-32.在应用 𝐴 和 𝐵 时，权重调整为：Δ𝑊=𝛼/𝑟⋅(𝐴⋅𝐵),其中 𝛼=lora_alpha,这可以放大或缩小 LoRA 的调整幅度
> - lora_dropout：LoRA 层的丢弃（dropout）率，取值范围为[0, 1)。  
> - target_modules：要替换为 LoRA 的模块名称列表或模块名称的正则表达式。针对不同类型的模型，模块名称不一样，因此，我们需要根据具体的模型进行设置，比如，LLaMa的默认模块名为[q_proj, v_proj]，我们也可以自行指定为：[q_proj,k_proj,v_proj,o_proj]。 在 PEFT 中支持的模型默认的模块名如下所示：

源码
```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # 冻结原始权重
        self.linear.weight.requires_grad = False
        
        # 低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 缩放因子
        self.scaling = alpha / rank

    def forward(self, x):
        # 原始线性变换
        base_output = self.linear(x)
        
        # LoRA更新
        lora_output = self.scaling * (x @ self.lora_A.T) @ self.lora_B.T
        
        return base_output + lora_output
```
变形：
LoRA VS AdaLoRA vs QLoRA

| 特征     | LoRA | AdaLoRA | QLoRA |
|----------|------|---------|-------|
| 参数适配 | 固定秩 | 动态秩 | 量化 + 低秩 |
| 计算效率 | 中等 | 较高   | 最高  |
| 参数量   | 少   | 更少   | 最少  |
| 复杂度   | 低   | 中等   | 较高  |

- 计算资源充足：
选择AdaLoRA
性能最佳

- 显存受限
选择QLoRA
极低参数量

- 快速原型
选择标准LoRA
实现最简单

### **IA3**
- 原理：（通过抑制和放大内部激活注入适配器）使用学习向量重新调整内部激活
- 激活模块：基于transformer的架构中的attention和feedforward模块

### **P-Tuning 与 Prompt Tuning / Prefix Tuning 的区别**

| 特征                  | P-Tuning                           | Prompt Tuning                      | Prefix Tuning                    |
|-----------------------|------------------------------------|------------------------------------|----------------------------------|
| **核心思想**           | 学习虚拟 token 的嵌入            | 学习结构化的提示 token 嵌入      | 向输入序列添加可学习的前缀嵌入 |
| **嵌入的位置**         | 输入文本的任意位置（前、后）     | 通常是输入文本的前或后部分      | 主要插入在输入序列的前缀部分 |
| **微调内容**           | 只微调虚拟 token 的嵌入         | 微调提示 token 嵌入             | 只微调前缀嵌入，不调整模型其他部分 |
| **与输入的关系**       | 虚拟 token 嵌入与文本拼接       | 提示 token 与文本拼接            | 前缀嵌入与文本拼接             |
| **训练成本**           | 低（只调整少量参数）            | 低（只调整提示 token 的嵌入）  | 低（只调整前缀嵌入）           |
| **应用场景**           | 生成任务、少数据微调任务        | 生成任务、文本分类、问答等任务 | 生成任务、对话生成等任务       |

### 参考资料
1. 吃果冻不吐果冻皮 https://zhuanlan.zhihu.com/p/649315197
2. 
