# 1. 基础概念
## 1.1 State, s
状态，系统在某一时刻的观测值或环境信息，可以是离散的（如棋盘位置）或连续的（如机器人坐标）
## 1.2 Action, a
动作 ，智能体在状态 s下执行的操作，可以是离散（如左/右移动）或连续（如施加力的大小）
state transition
![image](https://github.com/user-attachments/assets/b51576ba-748f-4647-9432-c7f335fe9ce0)
## 1.3 Policy, π
根据观测到的状态，如何做出决策，即如何从动作空间中选取一个动作。
the probability of taking action a given state s
π(a|s) =P(A=a|S=s)
- 确定性策略: 确定策略记作 µ : S -> A，它把状态 s 作为输入，直接输出动作 a = µ(s)，
而不是输出概率值。
![image](https://github.com/user-attachments/assets/5f9b0ed7-44ee-4c45-9883-e399b7aff66d)
- 随机性策略: 以概率分布选择动作
  - 离散动作空间：
  - 连续动作空间： 概率密度函数（Probability Density Function, PDF）
![image](https://github.com/user-attachments/assets/68ee8be4-39ac-4bb8-89f4-baf6a493473e)

## 1.4 Reward R
奖励与回报


## 1.5 Value Function
- 状态值函数V(s): 状态s的期望长期收益
- 动作值函数Q(s,a): 在状态s执行动作a的期望收益：

### 1.5.1 discounted return:  aka cumulative discounted future reward
从 t 时刻起，未来所有奖励的（加权）和
![image](https://github.com/user-attachments/assets/9ba19c77-6f9e-4443-9bc5-ee5bc98a33dd)

### 1.5.2 action-value function for policy 𝝿
对 $$U_t$$ 求期望，消除掉其中的随机性
![image](https://github.com/user-attachments/assets/d0307110-ea5d-4596-bee9-5bc8f080b200)

### 1.5.3 optimal action-value function
选择最好的策略函数 π
![image](https://github.com/user-attachments/assets/177dcd3c-d930-4dd1-9abd-c647c70d722c)
![image](https://github.com/user-attachments/assets/6e3581c3-3086-47ca-8e18-283f983be7a8)

### 1.5.4 state-value function
目标：评估当前状态 $$s_t$$ 是否对自己有利
把动作 $$A_t$$ 作为随机变量，然后关于 $$A_t$$ 求期望，把  $$A_t$$ 消掉
![image](https://github.com/user-attachments/assets/261fa598-d004-4c07-92c3-111bfdf88456)

### 1.6 advantage
 $$A(s,a)=Q(s,a)−V(s)$$ 
选取某个动作后得到的动作价值相对于平均动作价值的增量，当优势函数大于0时，说明当前的动作选择是优于平均的

# 其他概念【todo】
马尔可夫决策过程（Markov decision process， MDP）

# 参考
1. 深度强化学习/Deep Reinforcement Learning 王树森 黎彧君 张志华 著
