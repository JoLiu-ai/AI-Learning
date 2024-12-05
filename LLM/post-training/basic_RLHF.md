🎯目标：训练一个模型来模拟人类偏好，将人类的主观偏好判断转化为可量化的奖励信号  
🔧方法：收集人类对输出质量的偏好判断，如"A比B好"这样的成对比较数据  
✅涉及到的3个model：Language Model(LM), Tuned Language Model (RL Policy),Reward (Preference) Model   
  Language Model(LM) - 生成内容；  
  Reward Model（RM）- 进行评估并提供反馈；  
  Tuned Language Model (RL Policy) - 从人类的偏好中学习，辅助奖励模型进行优化，从而提升语言模型的输出质量。  

✅ 传统RLHF方法: Involves three model (reward model, reference model, and the fine-tuned model)
✅ DPO: Only involves the fine-tuned model.(Direct Preference Optimization: Your Language Model is Secretly a Reward Model)

![image](https://github.com/user-attachments/assets/81ba0cd8-b043-4868-b43b-1561c2d4c07b)


✅ two loss items


### 📌传统RLHF方法：
<img width="571" alt="截屏2024-12-04 18 03 41" src="https://github.com/user-attachments/assets/92a035cf-19fb-4fc0-8d1c-c1272befed63">

- 流程： 
  - 预训练语言模型 π_ref 
  - 收集偏好数据 D = {(x, y_w, y_l)} 
  - 训练奖励模型 r_θ(x,y) 
  - 通过强化学习优化策略模型 

- 目标函数：
  最大化期望奖励 E[R(x,y)]
- 损失函数：
  策略梯度损失： L(π) = E[(r_θ(x,y) - β * KL(π(·|x) || π_ref(·|x))))]


### 📌DPO
DPO converts the problem into a single-phase SFT task.
![image](https://github.com/user-attachments/assets/c2261119-1073-4082-adec-8ab11b3550e9)

主要区别：

RLHF需要复杂的奖励建模和策略梯度
DPO直接通过偏好数据优化模型
计算更简单、效率更高
避免了传统方法的不稳定性

##
### Prompt Tuning
- From discrete prompt to continuous prompt
The Power of Scale for Parameter-Efficient Prompt Tuning [Lester, ACL 2021]
![image](https://github.com/user-attachments/assets/046dda28-2711-4540-9771-c808a96bd933)


### Prefix-Tuning
- Prepend prefixes for each input.
- Prefix-Tuning: Optimizing Continuous Prompts for Generation [Li et al, ACL 2021]

- Prompt-Tuning only adds learnable prompts to the first layer.
- Prefix-Tuning adds tunable prompts to each layer.

  downsides:
  increase input length & the KV cache

  ## LoRA family - without introudcing extra inference latency

##  Bit-Delta
tune one bit.


### QLoRA
- quantize the base model to 4-bits, like a 7-billion parameter model.
- page the optimizer with the CPU offloading
  

![image](https://github.com/user-attachments/assets/c45c5520-a043-45e7-9625-9595433f365c)
(from: https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part28.html)



1/ https://huggingface.co/blog/rlhf  
2/ [Umar Jamil](https://www.youtube.com/watch?v=hvGa5Mba4c8)
