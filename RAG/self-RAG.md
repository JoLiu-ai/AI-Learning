### the issues of RAG

一般的RAG应用会无差别地访问向量库获取上下文，而不管其是否真的需要。这样有可能会引入主题无关的上下文，进而导致低质量的文本生成内容。背后的原因是：推理LLM并没有对上下文进行适配性训练，以使生成结果与上下文语义保持一致。检索上下文有可能引入有冲突的观点。

- top-k documents do not contain all the answers.
- computing similarity between document chunks and prompt does not always yield relevant contexts.

# What is self-rag

### 方法

思路：并非每个问题都需要从向量数据库中检索答案。即使检索到了数据库的片段，也需要先判断是否真的相关和真的有用，才将检索到的片段纳入答案中。
The authors develop a clever way for a fine-tuned LM (Llama2–7B and 13B) to output special tokens [Retrieval], [No Retrieval], [Relevant], [Irrelevant], [No support / Contradictory], [Partially supported], [Utility], etc. appended to LM generations to decide whether or not a context is relevant/irrelevant, the LM generated text from the context is supported or not, and the utility of the generation.

https://github.com/hinswhale/AI-Learning/assets/22999866/a0526c3c-9984-4278-bbb6-7b9656eac536

### reflection token

- Retrieve
- Critique
    - IsREL、IsSUP、IsUSE

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/77e4b42d-6db5-40d4-b9ab-f93f460a1fcf/Untitled.png)

## **训练**:

在训练过程中，需要两个模型：

- 评价模型 **critic model** 【C】，
- 生成模型Generator Model 【M】；

利用评价模型C来生成M模型所需的监督数据，在推理过程中只依赖M模型，不用C模型。

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/f9dddfe4-d465-475f-ad2a-927ecddbf032/Untitled.png)

### **critic model**

**1/ 数据收集**

引导 GPT-4 生成反思 token， 指令（"给定指令，判断从网上查找一些外部文档是否有助于生成更好的回复"）

**2/ 批评学习(critic learning)**

训练一个简单的 LM 来对生成的输出（提示或提示 + RAG 增强输出）进行分类，并在末尾添加相关的特殊标记。

训练的目标为：

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/0d0c290f-22c1-4a8c-843e-75cd7d8564cd/Untitled.png)

### **generator model**

生成器模型使用标准的下一个标记预测目标，学习生成连续标记以及特殊标记，以检索/批判各代。模型只需在适当的时候生成特殊标记，而不会改变底层 LM。

**1/ 数据收集**

- 对于输出 y 中的每个片段 yt，模型会使用 C（评判模型）来评估是否需要进一步的检索；
- 如果需要检索，会添加一个特殊的检索token: Retrieve=Yes，接着，R（检索算法）会检索最相关的 K 个文章或段落，记为 D；
- 对于每一个检索到的文章或段落，C（评判模型）会进一步评估这个段落是否与当前的任务相关，并给出一个 IsREL（是否相关）的预测；
- 如果该段落被认为是相关的，C 会进一步评估这个段落是否支持模型的生成，并给出一个 IsSUP（是否支持）的预测；
- IsUSE 可能代表着模型对检索到的内容的整体效用或有用性的评估；
- 最后，与反思tokens一起增强的输出和原始的输入对被添加到 Dgen，作为一个训练数据集。

**2/ 生成学习(generator learning)**

- 使用反思tokens的经过修改过的语料库Dgen来训练生成器模型；
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/966854ea-0051-4d38-babf-562d8bf5c481/Untitled.png)
    

**要点：**

1. **Critic 评估每段文章的相关性和支持性，并相应地附加标记。最终通过输出反思字符进行增强。**
2. **使用标准的 next token 目标在此增强语料库上训练生成模型，预测目标输出和反思字符。在训练期间，检索到的文本块被屏蔽，并通过反思字符 Critique 和 Retrieve 扩展词汇量。Self-RAG 模型还包含特殊令牌来控制和评估其自身的预测，从而实现更精细的输出生成。**

## **推理**

1. retrieve on demand：预测prompt是否需要上下文来增强文本生成结果。如果需要，则标记一个特殊的**retrieval token**，同时根据需要调用retriever模块。
2. generate segment in parallel：并行处理检索的上下文，评判上下文对prompt的相关性，同时生成相应的结果。
3. Critique outputs and select best segment ：评判每个上下文对相应结果的支持程度，同时选择一个最好的生成结果

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/f04f57a1-2ed9-4b0a-882f-25d5f48a0b13/Untitled.png)

主要步骤概括如下：

1. 判断是否需要额外检索事实性信息（retrieve on demand），仅当有需要时才召回
2. 平行处理每个片段：生产prompt+一个片段的生成结果
3. 使用反思字段，检查输出是否相关，选择最符合需要的片段；
4. 再重复检索
5. 生成结果会引用相关片段，以及输出结果是否符合该片段，便于查证事实。

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/cf1f3a5b-8505-4452-8168-8f40ca94618c/0d8c4808-48fa-4373-9beb-29e9ac6b5abc/Untitled.png)

# 资料

1. [https://medium.com/@bohachu/self-rag光是用向量資料庫語意檢索不夠精準啦-來點自我批評吧-616e930c9f33](https://medium.com/@bohachu/self-rag%E5%85%89%E6%98%AF%E7%94%A8%E5%90%91%E9%87%8F%E8%B3%87%E6%96%99%E5%BA%AB%E8%AA%9E%E6%84%8F%E6%AA%A2%E7%B4%A2%E4%B8%8D%E5%A4%A0%E7%B2%BE%E6%BA%96%E5%95%A6-%E4%BE%86%E9%BB%9E%E8%87%AA%E6%88%91%E6%89%B9%E8%A9%95%E5%90%A7-616e930c9f33)
2. [从 RAG 到 Self-RAG —— LLM 的知识增强](https://zhuanlan.zhihu.com/p/661465330)
