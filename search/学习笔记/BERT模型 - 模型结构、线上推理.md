学习笔记：来自 王树森
- 现代搜索引擎普遍使⽤ BERT 模型计算 q 和 d 的相关性。
- **交叉 BERT 模型**（单塔）准确性好，但是推理代价⼤，通常⽤于链路下游**（精排、粗排）**。
- **双塔 BERT 模型**不够准确，但是推理代价⼩，常⽤于链路上游（**粗排、召回海选）** 。
- 训练相关性 BERT 模型的 4 个步骤：**预训练、后预训练、微调、蒸馏**。

# **交叉 BERT 模型**

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/b2de1528-62bc-4bc8-b681-f5d4aa6722bd)

- **自注意力层对查询词和文档做了交叉**

## **Embedding**

- token embedding：表征 token 本⾝
- position embedding：位置编码，表征token的序
- segment embedding：⽤于区分查询词、标题、正⽂【不必须】
- 每个token被表征为 3个向量，取加和作为token 的表征。

## 字粒度 vs 字词混合粒度

**字粒度**：**将每个汉字/字符作为⼀个 token**。

- 词表较⼩（⼏千），只包含汉字、字母、常⽤字符。
- 优点：实现简单，**无需做分词**。

**字词混合粒度**：**做分词，将分词结果作为 tokens**。

- 词表较⼤（⼏万、⼗⼏万），包含汉字、字母、常⽤符号、常⽤中⽂词语、常⽤英⽂单词。
- 与字粒度相⽐，字词混合粒度得到的**序列长度更短（**即 token数量更少）。
- 参见 WoBERT （[https://github.com/ZhuiyiTechnology/WoBERT）](https://github.com/ZhuiyiTechnology/WoBERT%EF%BC%89)

序列更短（token 数量更少）有什么好处？

- BERT 的计算量是 token 数量的超线性函数。
- 为了控制推理成本，会限定 token 数量，例如 128 或 256。
- 如果⽂档超出 token 数量上限，会被截断，或者做抽取式摘要。
- 使⽤字词混合粒度，token 数量更少，推理成本降低。（字粒度需要 256 token，字词混合粒度只需要 128 token。）

## 推理降本

## 1. 交叉BERT模型的推理降本

- 对每个 (q,d) ⼆元组计算相关性分数 score，代价很⼤。
- ⽤ Redis 这样的 KV 数据库缓存(q, d, score)   。
- (q,d) 作为 key， 相关性分数 (score) 作为 value。
- 如果命中缓存，则避免计算。
- 如果超出内存上限，按照 least recently used (LRU) 清理缓存。

### 2. 模型量化技术，例如将 float32 转化成 int8。

- 训练后量化 (post-training quantization，PTQ)。
- 训练中量化 (quantization-aware training，QAT)。

### 3. 使⽤⽂本摘要降低 token 数量。

- 如果⽂档长度超出上限，则⽤摘要替换⽂档。
- 在⽂档发布时计算摘要。可以是抽取式，也可以是⽣成式。
- 如果摘要效果好，可以将 token 数量上限降低，⽐如从 128 降低到 96。

# **双塔 BERT 模型**

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/d54a878a-64e2-47e5-879d-1c9b34b70710)

# 总结

### 交叉BERT模型

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/0344297d-857b-4ad9-bb6d-53e94e8e482a)

### 双塔BERT模型
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/577b3557-27bd-44a2-afa8-42e929ab9a26)

