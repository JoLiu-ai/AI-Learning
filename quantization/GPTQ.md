Post Training Dynamic Quantization，PTDQ

Quantization

## 三种方法：

### 量化感知训练（Quantization Aware Training, QAT
[Tailor et al., 2021; Kim et al., 2022;Ding et al., 2022]）：
* during the model’s training / fine-tuning process
在模型训练过程中加入伪量化算子，通过训练时统计输入输出的数据范围可以提升量化后模型的精度，适用于对模型精度要求较高的场景；其量化目标无缝地集成到模型的训练过程中。这种方法使LLM在训练过程中适应低精度表示，增强其处理由量化引起的精度损失的能力。这种适应旨在量化过程之后保持更高性能。
### 量化感知微调（Quantization-Aware Fine-tuning，QAF）：
在微调过程中对LLM进行量化。主要目标是确保经过微调的LLM在量化为较低位宽后仍保持性能。通过将量化感知整合到微调中，以在模型压缩和保持性能之间取得平衡。
### 训练后量化（Post Training Quantization, PTQ）
[Liu et al., 2021b; Nagel et al., 2020; Fang et al., 2020]：
* after it has completed its training
在LLM训练完成后对其参数进行量化，只需要少量校准数据，适用于追求高易用性和缺乏训练资源的场景。主要目标是减少LLM的存储和计算复杂性，而无需对LLM架构进行修改或进行重新训练。PTQ的主要优势在于其简单性和高效性。但PTQ可能会在量化过程中引入一定程度的精度损失。


量化感知训练（QAT）：在模型的训练或微调过程中应用量化。这种方法使LLM能够在训练期间适应低精度表示，从而在量化后保持更高的性能。

后训练量化（PTQ）：在模型训练完成后对其参数进行量化。PTQ的主要目标是减少LLM的存储和计算复杂性，而不需要对模型架构进行修改或重新训练。

LLM-QAT、PEQA、QLORA、GPTQ、AWQ、SpQR、RPTQ、OliVe、ZeroQuant、SmoothQuant、ZeroQuant-V2、Outlier Suppression+、MoFQ、ZeroQuant-FP、FPTQ、QuantEase、Norm Tweaking、OmniQuant等。

Post Training Dynamic Quantization（PTDQ）
* 对训练后的模型权重执行动态量化，将浮点模型转换为动态量化模型，
* 仅对模型权重进行量化，偏置不会量化。
* 默认情况下，仅对Linear和RNN变体量化 (因为这些layer的参数量很大，收益更高)。
```python
torch.quantization.quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False)
```
[^]

<!-- 参数解释：

model：模型（默认为FP32）

qconfig_spec：

集合：比如： qconfig_spec={nn.LSTM, nn.Linear} 。列出要量化的神经网络模块。]

字典： qconfig_spec = {nn.Linear: default_dynamic_qconfig, nn.LSTM: default_dynamic_qconfig}

dtype： float16 或 qint8

mapping：就地执行模型转换，原始模块发生变异

inplace：将子模块的类型映射到需要替换子模块的相应动态量化版本的类型 -->

```python
import torch
from torch import nn


class DemoModel(torch.nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model_fp32 = DemoModel()
    # 创建一个量化的模型实例
    model_int8 = torch.quantization.quantize_dynamic(model=model_fp32,  # 原始模型
                                                     qconfig_spec={torch.nn.Linear},  # 要动态量化的算子
                                                     dtype=torch.qint8)  # 将权重量化为：qint8

    print(model_fp32)
    print(model_int8)

    # 运行模型
    input_fp32 = torch.randn(1, 1, 2, 2)
    output_fp32 = model_fp32(input_fp32)
    print(output_fp32)

    output_int8 = model_int8(input_fp32)
    print(output_int8)

```
输出结果

```python
DemoModel(
  (conv): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
  (relu): ReLU()
  (fc): Linear(in_features=2, out_features=2, bias=True)
)
DemoModel(
  (conv): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
  (relu): ReLU()
  (fc): DynamicQuantizedLinear(in_features=2, out_features=2, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
)
tensor([[[[0.3120, 0.3042],
          [0.3120, 0.3042]]]], grad_fn=<AddBackward0>)
tensor([[[[0.3120, 0.3042],
          [0.3120, 0.3042]]]])
```
