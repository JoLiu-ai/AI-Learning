### todo：增加参考资料

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/78ae6875-9c1d-464b-980d-186d61e40e5c)Basics

- Indexing
- Retrieval
- Generation


Advanced

- Query transformations
- Routing
- Query construction
- Indexing
- Retrieval
- Generation

# 预处理：分块/向量化

## **分块 Chunking**

**块的大小是一个需要深思熟虑的参数**——取决于你所使用的嵌入模型以及它在 tokens 方面的容量

## **向量化 Vectorisation**

查看 MTEB 排行榜以获取最新更新

https://paperswithcode.com/sota/text-retrieval-on-mteb

搜索优化的模型，如 bge-large 或 E5 embeddings 系列

# **搜索索引**

keywords：`搜索索引`，`元数据`，**分层索引**

## **1. 向量存储索引 Vector store index**

**the search index**

1.  **a flat index  平面索引[**最朴素的实现**]** —a brute force distance calculation between the query vector and all the chunks’ vectors
2. **a vector index 向量索引 - more efficient retrieval**  like [faiss](https://faiss.ai/), [nmslib](https://github.com/nmslib/nmslib) or [annoy](https://github.com/spotify/annoy), using some Approximate Nearest Neighbours implementation like clustring, trees or [HNSW](https://www.pinecone.io/learn/series/faiss/hnsw/) algorithm.
    
    托管服务，如 OpenSearch、ElasticSearch 和向量数据库，它们自动处理前文提到的数据摄取流程，例如 **[Pinecone](https://www.pinecone.io/)**、**[Weaviate](https://weaviate.io/)** 和 **[Chroma](https://www.trychroma.com/)**。
    
3.  **store metadata along with vectors** and then use **metadata filters** 
4. list index, tree index, and keyword table index

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/c98d0e1d-eb32-4bca-9d0b-eca5b6613636)


## **2. 分层索引  2 Hierarchical indices**

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/c98d0e1d-eb32-4bca-9d0b-eca5b6613636)


**create two indices /'ɪndɪsiz/**

- **one composed of summaries**  摘要
- **the other one composed of document chunks** 文档块组成，
- first filtering out the relevant docs by summaries and then searching just inside this relevant group 分两步进行搜索，首先通过摘要过滤掉相关文档，然后仅在这个相关组中搜索。 ****

## **3. 假设问题和 HyDE： Hypothetical Questions and HyDE**

 **Hypothetical**/ˌhaɪpəˈθetɪkl/

- **假设问题**
    - to **generate a question for each chunk and embed these questions in vectors**
    - 要求为 LLM 每个块生成一个问题，并将这些问题 embed 到向量中。在实际操作中，这些问题向量构成一个索引，用于对用户的查询进行匹配搜索（这里是用问题向量而非原文档的内容向量来构成索引），检索到相应问题后，再链接回原始文档的相应部分，作为大语言模型提供答案的背景信息。
    
    a **higher semantic similarity between query and hypothetical question** /sɪ'mæntɪk/
    
    这种方法提高了搜索质量，因为与实际块相比，查询和假设问题之间的语义相似性更高（即 Q2Q）。
    
- HyDE -一种反向逻辑方法
    
    这种方法中，你会让大语言模型针对一个查询生成一个假设性的回应，然后结合这个回应的向量和查询的向量，共同用于提升搜索的效果。
    
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/d98b5c87-dcca-4860-8f0f-42839e7c6079)


## **4. 语境增强 Context enrichment**

**通过检索更小的信息块来提高搜索质量**，**同时为大语言模型增加更多周围语境以便其进行推理**。

 **to retrieve smaller chunks for better search quality**, **but add up surrounding context for LLM to reason upon**.

一是通过增加检索到的小块周围的句子来扩大语境；

二是递归地将文档分割成包含小块的大块。

to expand context by sentences around the smaller retrieved chunk 

 to split documents recursively into a number of larger parent chunks, containing smaller child chunks.

### **句子窗口检索法  [](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo.html)Sentence Window Retrieval**

- `文档的每个句子都被单独编码`，这样可以极大提高查询与语境之间的余弦距离搜索的准确性。
- 我们在检索到的`最相关单句之前后各扩展*k*个句子`，然后把这个扩展后的语境送给 LLM 进行推理。

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/d5964ce7-9898-4821-a421-f295426c8c67)


 **自动合并检索法** **（也称为** **父文档检索法) Auto-merging Retriever (aka Parent Document Retriever)**

文档被分割成层次化的块结构，最小的叶子块被送至索引。在检索时，我们会找出 k 个叶子块，如果存在 n 个块都指向同一父块，我们就用这个父块替换它们，并把它送给 LLM 用于生成答案。

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/2cce0c86-6830-4990-bcd5-ec8a079108f3)

 ****

## **融合检索或混合搜索 Fusion retrieval or hybrid search**

传统搜索使用如 **[tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)** 或行业标准的 **[BM25](https://en.wikipedia.org/wiki/Okapi_BM25)** 等稀疏检索算法，

现代搜索则采用语义或向量方法。  a faiss vector index

这两种方法的结合就能产生出色的检索结果。

In LangChain  [Ensemble Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble) class,  RRF for reranking

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/59481a94-b799-4a08-87db-89e274a5356f)


# **重新排名与过滤 Reranking & filtering**

后处理：to refine them through filtering, re-ranking or some transformation.

**filtering out results based on `similarity score`, `keywords`, `metadata` or `reranking` them with other models** like an LLM, [sentence-transformer cross-encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html)【**[句子 - 转换器交叉编码器](https://www.sbert.net/examples/applications/cross-encoder/README.html)**】, Cohere reranking [endpoint](https://txt.cohere.com/rerank/)【Cohere 的重新排名**[接口](https://txt.cohere.com/rerank/)**】or based on metadata like date recency【日期新近度】 — basically, all you could imagine.

---

**agentic behaviour — some complex logic involving LLM reasoning within our RAG pipeline.**

# **4. 查询变换 Query transformations**

**Query transformations are a family of techniques using an LLM as a reasoning engine to modify user input in order to improve retrieval quality.**

**对于复杂的查询，大语言模型能够将其拆分为多个子查询。**

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/0d0c06d0-5b47-4941-8192-05b7e1bcb4b7)


 [Multi Query Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever?ref=blog.langchain.dev) in Langchain and as a [Sub Question Query Engine](https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine.html) in Llamaindex.

1. **[Step-back prompting](https://arxiv.org/pdf/2310.06117.pdf?ref=blog.langchain.dev) uses LLM to generate a more general query**, retrieving for which we obtain a more general or high-level context useful to ground the answer to our original query on.Retrieval for the original query is also performed and both contexts are fed to the LLM on the final answer generation step.Here is a LangChain [implementation](https://github.com/langchain-ai/langchain/blob/master/cookbook/stepback-qa.ipynb?ref=blog.langchain.dev).
2. **Query re-writing uses LLM to reformulate initial query** in order to improve retrieval. Both [LangChain](https://github.com/langchain-ai/langchain/blob/master/cookbook/rewrite.ipynb?ref=blog.langchain.dev) and [LlamaIndex](https://llamahub.ai/l/llama_packs-fusion_retriever-query_rewrite) have implementations, tough a bit different, I find LlamaIndex solution being more powerful here.

---

**accurately back reference our sources**

如何能够**准确地标注我们所引用的各个来源**。

1. **Insert this referencing task into our prompt** and ask LLM to mention ids of the used sources.
2. **Match the parts of generated response to the original text chunks** in our index — llamaindex offers an efficient [fuzzy matching based solution](https://github.com/run-llama/llama-hub/tree/main/llama_hub/llama_packs/fuzzy_citation) for this case. In case you have not heard of fuzzy matching, this is an [incredibly powerful string matching technique](https://towardsdatascience.com/fuzzy-matching-at-scale-84f2bfd0c536).

---

# **5. Chat Engine**

关键key：

the **chat logic, taking into account the dialogue context**

**问题：**

支持对后续问题、代词指代或与先前对话上下文相关的用户指令的处理

support `follow up questions`,  `anaphora`, `arbitrary user commands` relating to the previous dialogue context.

**解决方案：考虑聊天上下文的`查询压缩技术`**

**`query compression technique`, taking chat context into account** along with the user query.

[ContextChatEngine](https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_context.html)

- 检索与用户查询相关的上下文【retrieving context relevant to user’s query】 + 来自 *内存* 缓冲区的聊天历史【*memory* buffer 】 → LLM→生成下一步答案

 **[CondensePlusContextMode](https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_condense_plus_context.html)** 

- 聊天历史和最新消息(新的查询) + 索引→ 检索到的上下文 + 原始用户消息 →LLM →生成下一步答案

 [OpenAI agents based Chat Engine](https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_openai.html) 

- LlamaIndex 还支持基于 **[OpenAI 智能体的聊天引擎](https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_openai.html)**，提供了一种更灵活的聊天模式，而 Langchain 也 **[支持](https://python.langchain.com/docs/modules/agents/agent_types/openai_multi_functions_agent)** OpenAI 的功能性 API。

 [ReAct Agent](https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_react.html)

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/b74e5b75-b72e-49ec-a022-85d9c26ae544)


# **6. Query Routing**

**Query routing is the step of LLM-powered decision making upon what to do next given the user query**

Query routers are also used to select an index, or, broader, data store, where to send user query

**Defining the query router includes setting up the choices it can make.**

# **7. Agents in RAG**

![image](https://github.com/hinswhale/AI-Learning/assets/22999866/fc9addca-18a4-4b5f-a139-d9ad594490ae)

# **8. Response synthesiser**

The main approaches to response synthesis are: 

1. iteratively refine the answer by sending retrieved context to LLM chunk by chunk 

2. summarise the retrieved context to fit into the prompt 

3. generate multiple answers based on different context chunks and then to concatenate or summarise them. 

---

# **Encoder and LLM fine-tuning**

- the Transformer **Encoder, responsible for embeddings quality and thus context retrieval quality**
- an **LLM, responsible for the best usage of the provided context to answer user query【 a good few shot learner】**

`might narrow down the model’s capabilities in general`

## **Encoder fine-tuning**

 [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) (top 4 of the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) at the time of writing) 

a narrow domain dataset

## **Ranker fine-tuning**

**在你不完全信任基础编码器的情况下，使用交叉编码器 (cross-encoder) 对检索到的结果进行重新排列。** 这个过程是这样的：你把查询和每个前 k 个检索到的文本块一起送入交叉编码器，中间用 SEP (分隔符) Token 分隔，并对它进行微调，使其对相关的文本块输出 1，对不相关的输出 0。一个这种微调过程的优秀例子可以在**[这里](https://docs.llamaindex.ai/en/latest/examples/finetuning/cross_encoder_finetuning/cross_encoder_finetuning.html#)**找到，结果显示通过交叉编码器微调，成对比较得分提高了 4%

https://docs.llamaindex.ai/en/latest/examples/finetuning/cross_encoder_finetuning/cross_encoder_finetuning.html#

## **LLM fine-tuning**

# **Evaluation**

**answer relevance, answer groundedness, faithfulness and retrieved context relevance**.

**答案的相关性、答案的基于性、真实性和检索到的内容的相关性**。

[precision](https://docs.ragas.io/en/latest/concepts/metrics/context_precision.html) and [recall](https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html) 评估 RAG 方案的检索性能

 [Building and Evaluating Advanced RAG](https://learn.deeplearning.ai/building-evaluating-advanced-rag/) by Andrew NG, LlamaIndex and the evaluation framework [Truelens](https://github.com/truera/trulens/tree/main), they suggest the **RAG triad** — **retrieved context relevance** to the query, **groundedness** (how much the LLM answer is supported by the provided context) and **answer relevance** to the query.

- 最关键且可控的指标：**检索内容的相关性** the **retrieved context relevance**
    
    上述高级 RAG 管道的前 1-7 部分加上编码器和排名器的微调部分，这些都是为了提高这个指标
    
- 提高答案的相关性和基于性 answer relevance and groundedness.
    
    第 8 部分和大语言模型的微调则专注于提高答案的相关性和基于性。
    

检索器评估管道

https://github.com/run-llama/finetune-embedding/blob/main/evaluate.ipynb

- the **hit rate**
- the **Mean Reciprocal Rank 平均倒数排名 -** a common search engine metric
- faithfulness abd relevance 生成答案的质量指标，如真实性和相关性

LangChain： [LangSmith](https://docs.smith.langchain.com/)  可以实现自定义的评估器，还能监控 RAG 管道内的运行轨迹，进而增强系统的透明度

LlamaIndex  a [rag_evaluator llama pack](https://github.com/run-llama/llama-hub/tree/dac193254456df699b4c73dd98cdbab3d1dc89b0/llama_hub/llama_packs/rag_evaluator) 帮助你利用公共数据集来对你的管道进行评估
