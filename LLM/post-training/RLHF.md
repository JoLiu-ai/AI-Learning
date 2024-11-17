🎯目标：训练一个模型来模拟人类偏好，将人类的主观偏好判断转化为可量化的奖励信号  
🔧方法：收集人类对输出质量的偏好判断，如"A比B好"这样的成对比较数据  

### RLHF整体流程：
![image](https://github.com/user-attachments/assets/1ff6026a-21db-4a55-8ec8-28947a95ea75)

- 伪代码
```python
class RLHF_Pipeline:
    def __init__(self):
        self.base_llm = None        # 基础语言模型
        self.sft_model = None       # SFT模型
        self.reward_model = None    # 奖励模型
        self.rl_model = None        # RL最终模型
        
    def training_flow(self):
        # 1. 预训练基础模型
        self.pretrain_base_model()

        # 2. SFT阶段
        self.sft_model = self.supervised_finetuning(self.base_llm)
        
        # 2. Reward Model阶段
        self.reward_model = self.train_reward_model()
        
        # 3. RL阶段
        self.rl_model = self.rl_training(self.sft_model, self.reward_model)
```
#### SFT (Supervised Fine-Tuning)：
- 目的: 使模型初步对齐人类意图
- 数据: 高质量人工编写的问答对
- 方法: Supervised
- 特点:作为RM和RL的基础模型
```python

class SFT:
    def __init__(self):
        self.model = None
        
    def training_step(self, prompt, response):
        """标准的有监督学习"""
        output = self.model(prompt)
        loss = self.compute_loss(output, response)
        return loss
```
#### B. Reward Model详解：
流程：
- 数据构建
  - 收集多样化的提示
  - response_generation：使用SFT模型生成多个回答
- 训练策略
  - loss_type：Bradley-Terry模型/交叉熵
  - sampling：重要性采样
  - regularization：避免过拟合的策略
- 评估指标
  -  preference_accuracy 偏好预测准确率
  -  ranking_correlation排序相关性
  -  human_alignment 与人类判断的一致性

```python
# 奖励模型的训练
class RewardModel:
    def __init__(self):
        self.model = None

    def train_step(self, batch):
        # 1. 获取一对比较数据
        prompt, better_resp, worse_resp = batch
        
        # 2. 计算奖励分数
        score_better = self.model(prompt, better_resp)
        score_worse = self.model(prompt, worse_resp)
        
        # 3. 计算损失 - 偏好学习
        loss = -torch.log(torch.sigmoid(score_better - score_worse))
        
        # 4. 优化
        loss.backward()
        self.optimizer.step()
    
    def get_reward(self, prompt, response):
        # 推理时计算单个回答的奖励
        return self.model(prompt, response)

    def collect_preferences(self):
        # 收集人类偏好数据的方法:
        # 1. 对每个prompt生成多个回答
        responses = self.generate_responses(prompt)
        
        # 2. 人类标注偏好
        # - 直接比较两个回答
        # - 对多个回答排序
        # - 给出具体分数
        preferences = self.human_annotate(responses)
        
        return preferences
```

#### PPOTraining
- 轨迹收集
 - rollout：使用当前策略采样
 - reward_computation：使用RM计算奖励
 - advantage_estimation：GAE计算优势
- 策略更新
  - policy_loss PPO-Clip目标
  - value_loss 价值函数估计
  - kl_penalty KL散度约束
- 关键技巧
  - value_clipping 限制价值估计更新
  - gradient_clipping 梯度裁剪
  - mini_batch 小批量训练


```python

class PPOTraining:
    def __init__(self):
        self.policy = None          # 策略网络
        self.value = None           # 价值网络
        self.reward_model = None    # 奖励模型
        
    def get_reward(self, state, action):
        # 使用reward model计算奖励
        return self.reward_model(state, action)
    
    def training_step(self):
        # 1. 收集轨迹
        trajectories = self.collect_trajectories()
        
        # 2. 计算奖励
        rewards = [self.get_reward(s, a) for s, a in trajectories]
        
        # 3. 更新策略
        self.update_policy(trajectories, rewards)
```
#### 高级技巧和优化
##### A. 混合训练策略：
- SFT+RM混合
  - 同时训练SFT和RM,提高模型效率和性能
- 渐进式训练
  - 逐步增加任务难度,更好的泛化能力
- 多任务训练
  - 同时优化多个目标,更全面的能力提升

##### B. 优化策略：
- 奖励塑形
 - 设计复合奖励函数, 基础奖励 KL惩罚 多样性奖励
- 样本效率
 - 提高训练效率, [经验回放 重要性采样 优先级采样]
- 稳定性提升
 - 提高训练稳定性,[梯度累积 学习率调度 早停策略]

#### 关键挑战和解决方案
- 奖励偏差: 
    - 问题: 奖励模型可能存在偏见,
    - 解决: [多样化数据, 公平性约束, 人类校验]

- 探索效率: 
    - 问题: 策略搜索空间巨大,
    - 解决: [引导探索, 课程学习, 分层强化学习]

- 计算成本: 
    - 问题: 训练资源消耗大,
    - 解决: [模型蒸馏, 参数高效微调, 分布式训练]

#### 评估和监控
- 训练监控:
  - loss_tracking: 各组件损失函数
  - reward_stats: 奖励分布统计
  - policy_metrics: 策略变化指标

 - 性能评估
   -  human_eval: 人类评估结果
   -  automated_metrics: 自动化指标
   -  behavior_analysis: 行为分析


### reward_approaches
- PPO
  - 特点: 使用置信区间限制策略更新
  - 优势:
  - 与RM关系: 使用RM提供奖励信号"
- DPO
  - 特点:直接优化策略差异
  - 优势:不需要显式的RM,隐式学习奖励
- RLHF+PPO

#### 数据收集方法
- 直接生成: 让模型对同一prompt生成多个回答
- 温度采样: 使用不同温度参数生成多样化输出
- 人工编写: 人工创作多个不同质量的回答
- 真实数据: 收集真实人类对话/写作样本
#### 构建偏好对
#### 标注策略
- 二元比较: 直接选择哪个更好
- 李克特量表: 1-5分打分
- 相对排序: 多个答案排序
- 条件评估: 根据特定标准评分

#### 训练过程



原理与方法
A. 数据收集
对同一个输入x，收集两个不同的输出y1, y2  
让人类标注者判断哪个输出更好  
构建偏好对(preferred, non-preferred)  

B. 训练过程
将人类的偏好判断转化为二元分类问题  
使用成对比较(paired comparison)方法  
目标是让模型对较好的输出赋予更高的奖励值  

C. 关键思路
```python
# 假设有输入x和两个输出y_w(better), y_l(worse)
reward_better = r_theta(x, y_w)  # 预测better样本的奖励
reward_worse = r_theta(x, y_l)   # 预测worse样本的奖励
delta = reward_better - reward_worse  # 计算差值
prob = sigmoid(delta)  # 转换为胜率
```

δ = r_w - r_l = log odds = log(p/(1-p))
r_w: 较好输出的奖励值  
r_l: 较差输出的奖励值  
δ: 奖励差值(logit)  
p/(1-p): 优胜几率(odds)  
σ(δ) 表示 sigmoid 函数  

> 优胜几率(odds)
> 如 p = 0.8，表示80%的概率获胜  
> 则 1-p = 0.2，表示20%的概率失败  
> odds = 0.8/0.2 = 4，表示"赢/输"的比率为4:1  


![image](https://github.com/user-attachments/assets/c6609699-0a92-4156-a0bc-bf8d14b29501)


![image](https://github.com/user-attachments/assets/8aa5b5a9-1e1d-42fc-8612-7ff94b0705fa)
- C(2,K)是组合数，表示从K个样本中取2个的组合数,归一化系数
- rθ是奖励模型.x是输入,yw是较好的输出,yl是较差的输出,计算奖励差值
- 

```python
def compute_loss(reward_model, x, y_better, y_worse):
    # 计算奖励值
    reward_better = reward_model(x, y_better)
    reward_worse = reward_model(x, y_worse)
    
    # 计算奖励差值
    delta = reward_better - reward_worse
    
    # 通过sigmoid转换为概率
    prob = torch.sigmoid(delta)
    
    # 计算负log likelihood
    loss = -torch.log(prob)
    
    return loss.mean()
```

