# self querying retrieval
# Parent Document Retriever
```python
from langchain.retrievers import ParentDocumentRetriever
# This text splitter is used to create the parent documents - The big chunks
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# This text splitter is used to create the child documents - The small chunks
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="split_parents", embedding_function=bge_embeddings) #OpenAIEmbeddings()

# The storage layer for the parent documents
store = InMemoryStore()
big_chunks_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
big_chunks_retriever.add_documents(docs)
```

# tools
## vector stores
* chroma 
## embeddings
* BGE embeddings
* OpenAI

```python
from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)
vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=bge_embeddings  #OpenAIEmbeddings()
)
```

```python
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)
```
## models

# 资料
1. [Advanced RAG 01 - Self Querying Retrieval](https://www.youtube.com/watch?v=f4LeWlt3T8Y&list=PL8motc6AQftn-X1HkaGG9KjmKtWImCKJS&index=8)
