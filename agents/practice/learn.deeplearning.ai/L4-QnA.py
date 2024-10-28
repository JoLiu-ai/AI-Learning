from langchain.embeddings import OpenAIEmbeddings  # Import the appropriate embedding model
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from IPython.display import display, Markdown
from langchain.vectorstores import FAISS

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
file = 'data.csv'
loader = CSVLoader(file_path=file)

# 方法1：手动构建检索和查询流程
def manual_retrieval_method():
    """
    1. 手动控制每个步骤
    2. 直接构建向量存储
    3. 手动进行相似度搜索
    4. 手动合并文档
    5. 手动构造提示词
    6. 直接调用LLM
    """
    # 1. 初始化组件
    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
    embedding_model = OpenAIEmbeddings()

    
    # 2. 加载文档
    docs = loader.load()
    
    # 3. 创建向量存储
    db = DocArrayInMemorySearch.from_documents(
        docs,
        embedding_model
    )
    # db = FAISS.from_documents(documents, embeddings)
    
    # 4. 相似度搜索
    retrieved_docs = db.similarity_search(query)
    
    # 5. 合并文档内容
    combined_docs = "".join([doc.page_content for doc in retrieved_docs])
    
    # 6. 构造提示词并调用LLM
    full_query = f"{combined_docs} Question: {query}"
    response = llm.call_as_llm(full_query)
    
    return response

# 方法2：使用VectorstoreIndexCreator高级API
def vectorstore_index_method():
    """
    1. VectorstoreIndexCreator自动处理:
       - 文档加载
       - 文档分割
       - 向量存储创建
       - 检索器配置
    2. 内置了合理的提示词模板
    3. 自动处理文档检索和LLM调用
    """
    # 1. 初始化组件
    llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
    embedding_model = OpenAIEmbeddings()
    
    # 2. 使用VectorstoreIndexCreator一步完成索引创建
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch,
        embedding=embedding_model
    ).from_loaders([loader])
    """
    index_creator = VectorstoreIndexCreator(
    vectorstore_cls=FAISS,
    embedding=embeddings,
    text_splitter=CharacterTextSplitter(chunk_size=300, chunk_overlap=0),
)
"""
    
    # 3. 直接查询
    response = index.query(query, llm=llm)
    
    return response

