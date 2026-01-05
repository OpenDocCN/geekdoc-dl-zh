# 答案 2.6

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-2/Answers%202.6/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-2/Answers%202.6/)

## 理论

1.  RAG-QA 的三个阶段：接受查询、检索相关文档和生成答案。

1.  上下文窗口约束：由于 LLM 上下文有限，您不能传递每个片段。MapReduce 和 Refine 帮助聚合或迭代地跨多个文档细化信息。

1.  向量数据库：存储文档嵌入并提供基于语义相似度的最快相关文档检索。

1.  RetrievalQA 链：结合检索和答案生成，提高结果的相关性和准确性。

1.  MapReduce 和 Refine：MapReduce 从许多文档中快速生成摘要；Refine 顺序改进答案，这在精度至关重要的场合很有用。根据任务选择。

1.  分布式系统：在分布式设置中，需考虑网络延迟和序列化。

1.  实验：尝试 MapReduce 和 Refine；其有效性很大程度上取决于数据类型和问题风格。

1.  RetrievalQA 限制：没有内置的对话记忆，这使得在后续操作中维持上下文变得困难。

1.  对话记忆：在较长的对话中，需要整合之前的回合并提供上下文答案。

1.  进一步研究：新的 LLM 方法、它们对 RAG 系统的影响以及 RAG 链中的记忆策略。

## 实践任务

1.

```py
from  langchain.vectorstores  import Chroma
from  langchain_openai  import OpenAIEmbeddings

def  initialize_vector_database(directory_path):
    # Initialize an embeddings generator (OpenAI) to create vector representations for text
    embeddings_generator = OpenAIEmbeddings()

    # Initialize a Chroma vector database pointing to a persistence directory
    # and the embedding function to use
    vector_database = Chroma(persist_directory=directory_path, embedding_function=embeddings_generator)

    # Display current document count to verify initialization
    # Assumes Chroma exposes `_collection.count()`
    document_count = vector_database._collection.count()
    print(f"Documents in VectorDB: {document_count}")

# Example usage of initialize_vector_database:
documents_storage_directory = 'path/to/your/directory'
initialize_vector_database(documents_storage_directory) 
```

2.

```py
from  langchain.vectorstores  import Chroma
from  langchain_openai  import OpenAIEmbeddings, ChatOpenAI
from  langchain.chains  import RetrievalQA
from  langchain.prompts  import PromptTemplate

def  setup_retrieval_qa_chain(model_name, documents_storage_directory):
    # Initialize embeddings and Chroma vector store
    embeddings_generator = OpenAIEmbeddings()
    vector_database = Chroma(persist_directory=documents_storage_directory, embedding_function=embeddings_generator)

    # Initialize the language model (LLM) used in the RetrievalQA chain
    language_model = ChatOpenAI(model=model_name, temperature=0)

    # Define a custom prompt template to format LLM inputs
    custom_prompt_template = """To better assist with the inquiry, consider the details provided below as your reference...
{context}
Inquiry: {question}
Insightful Response:"""

    # Create the RetrievalQA chain, passing the LLM, a retriever from the vector DB,
    # requesting source documents, and using the custom prompt
    question_answering_chain = RetrievalQA.from_chain_type(
        language_model,
        retriever=vector_database.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(custom_prompt_template)}
    )

    return question_answering_chain

# Example usage of setup_retrieval_qa_chain:
model_name = "gpt-4o-mini"
documents_storage_directory = 'path/to/your/documents'
qa_chain = setup_retrieval_qa_chain(model_name, documents_storage_directory) 
```

3.

```py
# Assume setup_retrieval_qa_chain has been defined in the same script or imported.

# Configure to demonstrate both techniques (MapReduce and Refine)
model_name = "gpt-3.5-turbo"
documents_storage_directory = 'path/to/your/documents'
qa_chain = setup_retrieval_qa_chain(model_name, documents_storage_directory)

# Create QA chains: one for MapReduce, one for Refine
question_answering_chain_map_reduce = RetrievalQA.from_chain_type(
    qa_chain.llm,
    retriever=qa_chain.retriever,
    chain_type="map_reduce"  # Use MapReduce chain type
)

question_answering_chain_refine = RetrievalQA.from_chain_type(
    qa_chain.llm,
    retriever=qa_chain.retriever,
    chain_type="refine"  # Use Refine chain type
)

# Example query to test both techniques
query = "What is the importance of probability in machine learning?"

# Run MapReduce and print the answer
response_map_reduce = question_answering_chain_map_reduce({"query": query})
print("MapReduce answer:", response_map_reduce["result"])

# Run Refine and print the answer
response_refine = question_answering_chain_refine({"query": query})
print("Refine answer:", response_refine["result"]) 
```

4.

```py
def  handle_conversational_context(initial_query, follow_up_query, qa_chain):
  """
 Simulate handling a follow‑up question in a longer conversation.

 Args:
 - initial_query (str): First user query.
 - follow_up_query (str): Follow‑up query referring to prior context.
 - qa_chain (RetrievalQA): Initialized QA chain that can answer queries.

 Returns:
 - None: Prints both answers directly to the console.
 """
    # Generate the answer to the initial query
    initial_response = qa_chain({"query": initial_query})
    print("Answer to initial query:", initial_response["result"])

    # Generate the answer to the follow‑up query (note: no dialogue memory)
    follow_up_response = qa_chain({"query": follow_up_query})
    print("Answer to follow‑up query:", follow_up_response["result"])

# Example usage
a_initial = "Does the curriculum cover probability theory?"
a_follow_up = "Why are those prerequisites important?"
handle_conversational_context(a_initial, a_follow_up, qa_chain) 
```
