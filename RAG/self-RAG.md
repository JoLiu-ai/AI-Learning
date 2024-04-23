
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/a0526c3c-9984-4278-bbb6-7b9656eac536)

一般的RAG应用会无差别地访问向量库获取上下文，而不管其是否真的需要。这样有可能会引入主题无关的上下文，进而导致低质量的文本生成内容。背后的原因是：推理LLM并没有对上下文进行适配性训练，以使生成结果与上下文语义保持一致。检索上下文有可能引入有冲突的观点。

1/ top-k documents do not contain all the answers 
2/ The other issue is that computing similarity between document chunks and prompt does not always yield relevant contexts. 

# what is self-rag
思路：并非每个问题都需要从向量数据库中检索答案。即使检索到了数据库的片段，也需要先判断是否真的相关和真的有用，才将检索到的片段纳入答案中。 
 The authors develop a clever way for a fine-tuned LM (Llama2–7B and 13B) to output special tokens [Retrieval], [No Retrieval], [Relevant], [Irrelevant], [No support / Contradictory], [Partially supported], [Utility], etc. appended to LM generations to decide whether or not a context is relevant/irrelevant, the LM generated text from the context is supported or not, and the utility of the generation.

** 2步
prompt or prompt + RAG augmented output

Self-RAG 的训练分两步进行。
第一步，训练一个简单的 LM 来对生成的输出（提示或提示 + RAG 增强输出）进行分类，并在末尾添加相关的特殊标记。这个 "批评者模型 "是通过 GPT-4 注释训练出来的。具体来说，GPT-4 是使用特定类型的指令（"给定指令，判断从网上查找一些外部文档是否有助于生成更好的回复"）进行提示的。

In step 2, the generator model model, using a standard next token prediction objective, learns to generate continuations, as well as special tokens to retrieve/critique generations. Unlike other fine-tuning or RLHF methods where downstream training can impact model outputs and make future generations biased, through this simple approach, the model is trained only to generate special tokens as appropriate, and otherwise not change the underlying LM! Which is brilliant!
在步骤 2 中，生成器模型使用标准的下一个标记预测目标，学习生成连续标记以及特殊标记，以检索/批判各代。与其他微调或 RLHF 方法不同的是，其他方法的下游训练可能会影响模型输出并使后代产生偏差，而通过这种简单的方法，模型只需在适当的时候生成特殊标记，而不会改变底层 LM！这真是太棒了！

