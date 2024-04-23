

### the issues of RAG
一般的RAG应用会无差别地访问向量库获取上下文，而不管其是否真的需要。这样有可能会引入主题无关的上下文，进而导致低质量的文本生成内容。背后的原因是：推理LLM并没有对上下文进行适配性训练，以使生成结果与上下文语义保持一致。检索上下文有可能引入有冲突的观点。

* 1. top-k documents do not contain all the answers.
* 2. computing similarity between document chunks and prompt does not always yield relevant contexts. 

# What is self-rag
思路：并非每个问题都需要从向量数据库中检索答案。即使检索到了数据库的片段，也需要先判断是否真的相关和真的有用，才将检索到的片段纳入答案中。 
 The authors develop a clever way for a fine-tuned LM (Llama2–7B and 13B) to output special tokens [Retrieval], [No Retrieval], [Relevant], [Irrelevant], [No support / Contradictory], [Partially supported], [Utility], etc. appended to LM generations to decide whether or not a context is relevant/irrelevant, the LM generated text from the context is supported or not, and the utility of the generation.
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/a0526c3c-9984-4278-bbb6-7b9656eac536)

###  workflow: 

1. retrieve on demand：首先，Self-RAG解码检索令牌（retrieval token）以评估是否需要检索，并控制检索组件。如果需要检索，LM将调用外部检索模块查找相关文档。
2. generate segment in parallel：如果不需要检索，模型会预测下一个输出段。如果需要检索，模型首先生成批评令牌（critique token）来评估检索到的文档是否相关，然后根据检索到的段落生成后续内容。
3. Critique outputs and select best segment ：如果需要检索，模型进一步评估段落是否支持生成。最后，一个新的批评令牌（critique token）评估响应的整体效用。
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/84fceee0-e2c8-476d-a4da-8ab1f9b5d91f)

reflection token
 * Retrieve
 * Critique
   * IsREL、IsSUP、IsUSE


预测prompt是否需要上下文来增强文本生成结果。如果需要，则标记一个特殊的retrieval token，同时根据需要调用retriever模块。
并行处理检索的上下文，评判上下文对prompt的相关性，同时生成相应的结果。
评判每个上下文对相应结果的支持程度，同时选择一个最好的生成结果。


Self-RAG 的训练分两步进行。
第一步，训练一个简单的 LM 来对生成的输出（提示或提示 + RAG 增强输出）进行分类，并在末尾添加相关的特殊标记。这个 "批评者模型 "是通过 GPT-4 注释训练出来的。具体来说，GPT-4 是使用特定类型的指令（"给定指令，判断从网上查找一些外部文档是否有助于生成更好的回复"）进行提示的。

In step 2, the generator model model, using a standard next token prediction objective, learns to generate continuations, as well as special tokens to retrieve/critique generations. Unlike other fine-tuning or RLHF methods where downstream training can impact model outputs and make future generations biased, through this simple approach, the model is trained only to generate special tokens as appropriate, and otherwise not change the underlying LM! Which is brilliant!
在步骤 2 中，生成器模型使用标准的下一个标记预测目标，学习生成连续标记以及特殊标记，以检索/批判各代。与其他微调或 RLHF 方法不同的是，其他方法的下游训练可能会影响模型输出并使后代产生偏差，而通过这种简单的方法，模型只需在适当的时候生成特殊标记，而不会改变底层 LM！这真是太棒了！

# 资料
1. https://medium.com/@bohachu/self-rag%E5%85%89%E6%98%AF%E7%94%A8%E5%90%91%E9%87%8F%E8%B3%87%E6%96%99%E5%BA%AB%E8%AA%9E%E6%84%8F%E6%AA%A2%E7%B4%A2%E4%B8%8D%E5%A4%A0%E7%B2%BE%E6%BA%96%E5%95%A6-%E4%BE%86%E9%BB%9E%E8%87%AA%E6%88%91%E6%89%B9%E8%A9%95%E5%90%A7-616e930c9f33
2. 
