# why fusion
## rag advantages:

* Vector Search Fusion: RAG introduces a novel paradigm by integrating vector search capabilities with generative models. This fusion enables the generation of richer, more context-aware outputs from large language models (LLMs).
* Reduced Hallucination: RAG significantly diminishes the LLM’s propensity for hallucination, making the generated text more grounded in data.
* Personal and Professional Utility: From personal applications like sifting through notes to more professional integrations, RAG showcases versatility in enhancing productivity and content quality while being based on a trustworthy data source.

## rag limitations
* `Constraints with Current Search Technologies:` RAG is limited by the same things limiting our retrieval-based lexical and vector search technologies.
* `Human Search Inefficiencies`: Humans are not great at writing what they want into search systems, such as typos, vague queries, or limited vocabulary, which often lead to missing the vast reservoir of information that lies beyond the obvious top search results. While RAG assists, it hasn’t entirely solved this problem.
* `Over-Simplification of Search`: Our prevalent search paradigm linearly maps queries to answers, lacking the depth to understand the multi-dimensional nature of human queries. This linear model often fails to capture the nuances and contexts of more complex user inquiries, resulting in less relevant results.

#  RAG fusion
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/1e5a75ad-3ec7-4403-ad21-fbb6e2c3a82b)
重点： Multi-Query Generation + RRF
## Workflow:
1. Multi-Query Generation: 通过LLM将Query转换为相似但不同的查询
   重点：`prompt engineering ` 
   ![image](https://github.com/hinswhale/AI-Learning/assets/22999866/d50f586f-39ac-455b-8170-080cd14f4434)

```python
# Function to generate queries using OpenAI's ChatGPT
def generate_queries_chatgpt(original_query):

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
            {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
            {"role": "user", "content": "OUTPUT (4 queries):"}
        ]
    )

    generated_queries = response.choices[0]["message"]["content"].strip().split("\n")
    return generated_queries
```
2. Vector Search via  original + generated  Query
3.  Use `RRF( reciprocal rank fusion )` to aggregate and refine  the above results
   ![image](https://github.com/hinswhale/AI-Learning/assets/22999866/630de3d7-da28-4a03-ad44-045c5f8389f0)
![image](https://github.com/hinswhale/AI-Learning/assets/22999866/52f559b2-08a3-4fd8-9f0b-490da7dbfa71)
其中，rank是按照距离排序的文档在各自集合中的排名，k是常数平滑因子，一般取k=60。RRF将不同检索器的结果综合评估得到每个chunk的统一得分。
   ```python
# Reciprocal Rank Fusion algorithm
def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}
    print("Initial individual search result ranks:")
    for query, doc_scores in search_results_dict.items():
        print(f"For query '{query}': {doc_scores}")
        
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
            print(f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    print("Final reranked results:", reranked_results)
    return reranked_results
```
4.  put the contents aggregated above to a large language model, generate output, considering all the queries and the reranked list of results.
```python
# Dummy function to simulate generative output
def generate_output(reranked_results, queries):
    return f"Final output based on {queries} and reranked documents: {list(reranked_results.keys())}"
```

# 参考资料
1. https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1
2. https://medium.com/@kiran.phd.0102/rag-fusion-revolution-a-paradigm-shift-in-generative-ai-2349b9f81c66
