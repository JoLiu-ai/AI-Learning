# 链类型使用建议：

## 1. Stuff Chain
- 工作方式：
  * 将所有文档合并成一个大文档
  * 一次性发送给 LLM 处理
- 优点：
  * 最简单直接
  * 保持完整上下文
  * 单次 LLM 调用，速度快
- 缺点：
  * 受限于 LLM 的上下文长度限制
  * 不适合处理大量文档
- 适用场景：
  * 文档总量小
  * 需要考虑多个文档间的关系
  * 需要快速响应

## 2. Map Reduce Chain
- 工作方式：
  * Map：每个文档单独发送给 LLM 处理
  * Reduce：合并所有中间答案，再次处理得到最终答案
- 优点：
  * 可以处理大量文档
  * 支持并行处理
  * 适合分布式环境
- 缺点：
  * 可能丢失文档间的关系
  * 需要多次 LLM 调用
  * 最终合并可能损失细节
- 适用场景：
  * 大量文档处理
  * 文档之间相对独立
  * 可以接受并行处理开销

## 3. Refine Chain
- 工作方式：
  * 先处理第一个文档得到初始答案
  * 逐个处理后续文档，不断精炼答案
- 优点：
  * 可以生成高质量、连贯的答案
  * 保持上下文连续性
  * 适合需要综合信息的场景
- 缺点：
  * 处理时间较长
  * 需要多次顺序 LLM 调用
  * 不支持并行处理
- 适用场景：
  * 需要高质量答案
  * 文档之间有关联
  * 对处理时间不敏感

## 4. Map Rerank Chain
- 工作方式：
  * 独立处理每个文档段
  * 对每个答案打分
  * 选择最佳答案
- 优点：
  * 可以找到最相关的答案
  * 支持并行处理
  * 适合精确匹配场景
- 缺点：
  * 不综合多个文档的信息
  * 可能错过重要上下文
  * 需要额外的评分逻辑
- 适用场景：
  * 寻找最佳单一答案
  * 问题明确且答案集中
  * 需要准确性排序

选择建议：
1. 如果文档量小（< 3-4个文档）：使用 Stuff Chain
2. 如果文档量大且需要综合答案：使用 Map Reduce Chain
3. 如果需要高质量连贯答案：使用 Refine Chain
4. 如果需要找到最相关的单个答案：使用 Map Rerank Chain

## 代码示例：
```python
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

# Stuff Chain
chain_stuff = load_qa_chain(llm, chain_type="stuff") # map_reduce / refine / map_rerank

# 使用 RetrievalQA 时
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 或 "map_reduce", "refine", "map_rerank"
    retriever=retriever
)
```

"""
