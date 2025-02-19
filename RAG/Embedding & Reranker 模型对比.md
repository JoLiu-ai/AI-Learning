| **模型类别**   | **模型名称**                      | **具体参数/版本**             | **语言支持** | **模型类型**   | **特点说明**                                              |
|----------------|----------------------------------|----------------------------|--------------|----------------|-----------------------------------------------------------|
| **Embedding**  | OpenAI                           | text-embedding-3-small      | English      | 商业API        | 8k上下文，MTEB英文榜62.3分，实时性要求高的问答系统               |
|                | Cohere                           | embed-english-v3.0          | English      | 商业API        | 支持search_document压缩模式                                  |
|                | BAAI                             | bge-large-en-v1.5           | English      | 开源           | 指令优化，MTEB英文榜64.2分，高精度检索、长文本处理               |
|                | Voyage                           | voyage-large-2              | English      | 商业API        | 16k上下文，金融领域优化                                      |
|                | Google                           | text-embedding-004          | 多语言       | 商业API        | 支持1024维向量                                              |
| **Reranker**   | BAAI                             | bge-reranker-large          | 中/英        | 开源           | 基于交叉熵优化，中英文双语支持，MRR@10达39.4，企业级搜索、多语言场景 |
|                | Cohere                           | rerank-english-v3.0         | English      | 商业API        | 延迟120ms，流式返回支持， 高并发在线服务                       |
|                | Sentence-Transformers            | cross-encoder/ms-marco-MiniLM-L-6 | English  | 开源           | 轻量化（60MB）                                               |
