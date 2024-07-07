## GPU 结构
- SM : 每个GPU包含多个SM, 每个SM独立执行计算任务
- L2 Cache
- HBM（High Bandwidth Memory）- 与memory controller连接, 在GPU外部
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/4d6c59d6-ebbd-4857-9c0c-b0c6311b3576)

> 自制的玩具[🙅不要信它]
<img width="600" length="800" alt="2" src="https://github.com/hinswhale/AI-Learning/assets/22999866/041c5dc9-690f-4394-b8b2-385e54833e94">

--

### A100
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/eddd26a2-2a4d-4fd9-b7d4-12a60482d95c)

### SM
- L1 cache （由SRAM构成）
- Register File （由SRAM构成）
- FP32 core
- FP64 core
- Tensor core
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/b0818c7d-9f7e-4b8a-bd0b-53798d60864b)

## SRAM HBM  DRAM
- SRAM（Static Random Access Memory） - SM里， `L1 cache` &  `Register File `
- HBM（High Bandwidth Memory）- 支持多个SM并行访问
- DRAM（Dynamic Random Access Memory）- 在没有HBM的系统中，DRAM作为GPU的全局内存 & CPU里
- **SRAM > HBM > DRAM**


<img width="400" length="500" alt="1" src="https://github.com/hinswhale/AI-Learning/assets/22999866/1b8cffef-f3a6-4a82-8ed5-6a618823cfc3">
<img width="400" length="500" alt="2" src="https://github.com/hinswhale/AI-Learning/assets/22999866/4deecb4d-54f3-4966-82d3-1b25b82027a0">
<img width="400" length="500" alt="2" src="https://github.com/hinswhale/AI-Learning/assets/22999866/fe65144a-7810-4eca-91ed-39235c902369">

