"""
Based on the article and related code。
Reference: Advance Retrieval Techniques in RAG
https://ai.gopubby.com/advance-retrieval-techniques-in-rag-5fdda9cc304b

Modification：Some packages have been updated to their latest versions
There seems to be an issue with evals; updates are ongoing.
"""
import os
import numpy as np


from typing import List
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    load_index_from_storage,
    StorageContext
)

from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

from trulens.apps.langchain import TruChain
from trulens.core import TruSession
from trulens.dashboard import run_dashboard
from trulens.feedback.v2.feedback import Groundedness
from trulens.providers.openai import OpenAI
from trulens.apps.llamaindex import TruLlama


OPENAI_API_KEY='sk-****'
OPENAI_BASE_URL="***8"

Settings.llm = OpenAI(model="gpt-3.5-turbo", api_base=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_base=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900

# check if data indexes already exists
if not os.path.exists("./storage"):
    # load data
    documents = SimpleDirectoryReader(
        input_dir="../dataFiles").load_data(show_progress=True)

    # create nodes parser
    node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)

    # split into nodes
    base_nodes = node_parser.get_nodes_from_documents(documents=documents)

    # creating index
    # a vector store index only needs an embed model
    index = VectorStoreIndex(
        base_nodes,
    )

    # store index
    index.storage_context.persist()
else:
    # load existing index
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context=storage_context)


# RAG pipeline evals
session = TruSession()

provider = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(TruLlama.select_source_nodes().node.text)
    .on_output()
).aggregate(np.mean)


# Question/answer relevance between overall question and answer.
# 定义答案相关性反馈
f_qa_relevance = Feedback(provider.relevance, name = "Answer Relevancee").on_input_output()

# Question/statement relevance between question and each context chunk.
# 定义上下文相关性反馈
f_qs_relevance = Feedback(provider.qs_relevance, name = "Context Relevance"
                          ).on_input().on(
    TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

tru_query_engine_recorder = TruLlama(query_engine,
    app_id='LlamaIndex',
    feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])


# eval using context window
with tru_query_engine_recorder as recording:
    query_engine.query("What did the president say about covid-19")


# run dashboard
run_dashboard(session=session)



