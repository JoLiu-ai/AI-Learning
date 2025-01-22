- 自动化评估：
   使用工具和指标自动衡量 RAG 性能。可以通过准备测试数据集，利用相关工具平台如 LangSmith、Langfuse 等进行评估，也可使用 Trulens、RAGAS 等框架来实现自动化评估

![image](https://github.com/user-attachments/assets/c7f498f8-2f03-4c97-8ef1-fc4b3a5cd65e)



### Groundedness（事实准确性）
- 评估 LLM 的回答是否基于源文档
- 方法：
    - 使用 COT 方法，评估 LLM 的回答是否基于源文档
    - 将 LLM 的回答分解成多个陈述
    - 检查每个陈述是否能在源文档中找到支持证据
    - 计算有支持证据的陈述比例作为准确性分数

### QA Relevance（问答相关性）
![image](https://github.com/user-attachments/assets/b3d6f2ab-26f8-4ae2-9f7f-a4af995dceb9)

- 评估回答是否真正回应了问题。
- 思路：
   - 根据给定的答案 a(q) 生成n个潜在问题 qi
   - 我们利用文本Embedding模型获得所有问题的Embeddings。
   - 对于每个 qi ，我们计算 sim(q, qi) 与原始问题 q 的相似性

### Question-Statement Relevance（问题-上下文相关性）
- 评估检索的文档片段是否与问题相关，度量检索质量，，要评估检索上下文支持查询的程度
- 思路
   - 使用 LLM 从上下文（c(q)）中提取了一组关键句子（Sext）
   - 使用以下公式在句子级别计算相关性:

![image](https://github.com/user-attachments/assets/8c35f152-a215-4d72-a126-37f7e4b7dc83)


### Context Recall 上下文召回
- 检索到的上下文与标注答案之间的一致程度
![image](https://github.com/user-attachments/assets/3c5c0f23-4d1b-4f09-8c35-2806cc62d2bd)


### Context Precision 上下文精度
- 衡量在检索到的上下文中，所有与基本事实相关的条目是否都排在靠前的位置
![image](https://github.com/user-attachments/assets/718baf1c-730c-4c95-9ebc-c3a3787c6582)

- 如果相关的召回内容很少，但这些少量的内容都排在很高的位置，那么上下文精度的得分也会很高，这就可能导致评估结果不能真实反映检索系统的整体性能，存在一定的片面性。

# 参考文献
1. 英文原文：https://ai.plainenglish.io/advanced-rag-03-using-ragas-llamaindex-for-rag-evaluation-84756b82dca7
