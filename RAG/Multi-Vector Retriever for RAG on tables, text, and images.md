| Idea                   | Example                                                           | Sources                                  |
|------------------------|-------------------------------------------------------------------|------------------------------------------|
| Base case RAG          | Top K retrieval on embedded document chunks, return doc chunks for LLM context window  | LangChain vectorstores, embedding models |
| Summary embedding      | Top K retrieval on embedded document summaries, but return full doc for LLM context window | LangChain Multi Vector Retriever         |
| Windowing              | Top K retrieval on embedded chunks or sentences, but return expanded window or full doc | LangChain Parent Document Retriever      |
| Metadata filtering     | Top K retrieval with chunks filtered by metadata                   | Self-query retriever                     |
| Fine-tune RAG embeddings | Fine-tune embedding model on your data                            | LangChain fine-tuning guide             |
| 2-stage RAG            | First stage keyword search followed by second stage semantic Top K retrieval | Cohere re-rank                          |


# 资料：
1. https://blog.langchain.dev/semi-structured-multi-modal-rag/
