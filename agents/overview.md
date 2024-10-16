# 架构
## Planning
- Task Decomposition
  - Chain of thought (CoT; [Wei et al. 2022](https://arxiv.org/abs/2201.11903)) 
    > think step by step
  - Tree of Thoughts (ToT;  [Yao et al. 2023](https://arxiv.org/abs/2305.10601))
    - Breadth-First Search，BFS
    - Depth-First Search，DFS
    > (1) by LLM with simple prompting like "Steps for XYZ.\n1.", "What are the subgoals for achieving XYZ?",   
    > (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel   
    > (3) with human inputs.
  - LLM+P (PDDL（Planning Domain Definition Language，一种规划问题描述语言）,[Liu et al. 2023](https://arxiv.org/abs/2304.11477))
- Reflection
  - ReAct (reasoning and acting, [Yao et al. 2023](https://arxiv.org/abs/2210.03629))
    > Thought: ...   
    > Action: ...   
    > Observation: ...   
    >   ... (Repeated many times)
    ![image](https://github.com/user-attachments/assets/949e31bd-1f8e-48dc-b37f-31f790ad7992)

  - Reflexion ([Shinn & Labash 2023](https://arxiv.org/abs/2303.11366))
    ![image](https://github.com/user-attachments/assets/898448a2-b70e-475c-a4bd-e702a96d89db)
  - Chain of Hindsight (CoH; [Liu et al. 2023](https://arxiv.org/abs/2302.02676))
    - Algorithm Distillation (AD; Laskin et al. 2023): to learn the process of RL instead of training a task-specific policy itself.
      ![image](https://github.com/user-attachments/assets/5c17a523-ccac-4553-a258-143c1a45c1a1)


- Self-critics
- Chain of thoughts
- Subgoal decomposition

![image](https://github.com/user-attachments/assets/bb1a490f-cf36-449e-bb68-0ec21a8f6da8)


## Memory
Sensory Memory
- visual, auditory
- Short-term memory (STM)
- Long-term memory
   - Explicit / declarative memory
   - Implicit / procedural memory
  ![image](https://github.com/user-attachments/assets/d5ecef11-6546-4e07-8815-8cd97201aa50)


## Tool use
## Action
