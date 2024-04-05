## 1. 为什么用RAG
* 数据私有问题
* 长文本遗忘问题
* 时效性问题：训练大模型的成本问题
* 来源验证和可解释性
## 2. RAG 基础流程
  先用用户的query在外部数据库里检索到候选答案，再用LLM对答案进行加工。
## 3.RAG关键模块
* 数据处理 (data processing)
* 文档划分（Document Split / Chunking）
*  向量化（embedding）及创建索引(create index)： 原始的文本内容 + Metadata
* retrieve：检索
* 文档获取（Retrieve）
* Prompt 工程（Prompt Engineering）
* 大模型问答（LLM）
* 响应合成
## 4. RAG vs. SFT
思考：当大模型长文本问题解决，RAG是不是没用了？
* 模型训练成本问题
* 外部数据：隐私和安全
* so on

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/ff8a21b0-7a43-4c8a-a92c-d0dbf222cac7)



