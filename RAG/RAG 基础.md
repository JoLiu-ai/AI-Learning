
## 1.什么是RAG
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/6ec3dd8e-c74d-4069-a483-407717fb7d8c)
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/337143bd-037e-47dc-8d6e-e47aff8a4c1c)
<div>
  <img src="https://github.com/hinswhale/AI-Learning/assets/22999866/6ec3dd8e-c74d-4069-a483-407717fb7d8c" alt="Image 1" style="width: 45%; margin-right: 5px;" />
  <img src="https://github.com/hinswhale/AI-Learning/assets/22999866/6ec3dd8e-c74d-4069-a483-407717fb7d8c" alt="Image 2" style="width: 45%;" />
</div>

## 2. 为什么用RAG
  * 数据私有问题
  * 长文本遗忘问题
  * 时效性问题：训练大模型的成本问题
  * 来源验证和可解释性
## 3. RAG 基础流程
  先用用户的query在外部数据库里检索到候选答案，再用LLM对答案进行加工。
## 4.RAG关键模块
粗分： 
  * indexing
  * retrieve
  * generation
细分：
  * 数据处理 (data processing)
  * 文档划分（Document Split / Chunking）
  *  向量化（embedding）及创建索引(create index)： 原始的文本内容 + Metadata
  * 文档获取（Retrieve）
  * Prompt 工程（Prompt Engineering）
  * 大模型问答（LLM）
  * 响应合成
## 5. RAG vs. SFT
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/ff8a21b0-7a43-4c8a-a92c-d0dbf222cac7)
思考：当大模型长文本问题解决，RAG是不是没用了？
  * 模型训练成本问题
  * 外部数据：隐私和安全
  * so on



![image](https://github.com/hinswhale/AI-Learning/assets/22999866/2d7a3b11-c409-4bf4-ac31-23242df2e421)

## Reference
1. https://docs.google.com/presentation/d/1C9IaAwHoWcc4RSTqo-pCoN3h0nCgqV2JEYZUJunv_9Q/edit#slide=id.g2b6714d62f7_0_0


