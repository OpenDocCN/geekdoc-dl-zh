# 2.6 RAG 系统 — QA 技术

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-2/2.6%20RAG%20%E2%80%94%20Techniques%20for%20QA/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-2/2.6%20RAG%20%E2%80%94%20Techniques%20for%20QA/)

Retrieval-Augmented Generation (RAG)结合了检索和生成，改变了我们处理大型语料库以构建准确 QA 系统和聊天机器人的方式。一个关键阶段是将检索到的文档与原始查询一起输入模型以生成答案。在检索到相关材料后，它们必须被综合成一个连贯的答案，将内容与查询的上下文相结合，并利用模型的能力。整体流程很简单：系统接受一个问题；从向量存储中检索相关片段；然后将检索到的内容与问题一起输入 LLM 以形成答案。默认情况下，您可以发送所有检索到的部分到上下文中，但上下文窗口限制通常导致像 MapReduce、Refine 或 Map-Rerank 这样的策略——它们在多份文档中聚合或迭代改进答案。

在使用 LLM 进行 QA 之前，确保环境已设置：导入、API 密钥、模型版本等。

```py
import  os
from  openai  import OpenAI
from  dotenv  import load_dotenv
import  datetime

# Load environment variables and configure the OpenAI API key
load_dotenv()
client = OpenAI()

# Configure LLM versioning
current_date = datetime.datetime.now().date()
llm_name = "gpt-3.5-turbo"
print(f"Using LLM version: {llm_name}") 
```

接下来，从存储嵌入的向量数据库（VectorDB）中检索与查询相关的文档。

```py
# Import the vector store and embedding generator
from  langchain.vectorstores  import Chroma
from  langchain_openai  import OpenAIEmbeddings

# Directory where the vector database persists its data
documents_storage_directory = 'docs/chroma/'

# Initialize the embedding generator using OpenAI embeddings
embeddings_generator = OpenAIEmbeddings()

# Initialize the vector database with the persistence directory and embedding function
vector_database = Chroma(persist_directory=documents_storage_directory, embedding_function=embeddings_generator)

# Show the current number of documents in the vector database
print(f"Documents in VectorDB: {vector_database._collection.count()}") 
```

`RetrievalQA`结合了检索和生成：LLM 根据检索到的文档回答。首先，初始化语言模型，

```py
from  langchain_openai  import ChatOpenAI

# Initialize the chat model with the selected LLM
language_model = ChatOpenAI(model=llm_name, temperature=0) 
```

然后使用自定义提示配置 RetrievalQA 链，

```py
# Import required LangChain modules
from  langchain.chains  import RetrievalQA
from  langchain.prompts  import PromptTemplate

# Create a custom prompt template to guide the LLM to use the provided context effectively
custom_prompt_template = """To better assist with the inquiry, consider the details provided below as your reference...
{context}
Inquiry: {question}
Insightful Response:"""

# Initialize the RetrievalQA chain with the custom prompt
a_question_answering_chain = RetrievalQA.from_chain_type(
    language_model,
    retriever=vector_database.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PromptTemplate.from_template(custom_prompt_template)}
) 
```

然后在简单查询上检查答案。

```py
# Provide a sample query
query = "Is probability a class topic?"
response = a_question_answering_chain({"query": query})
print("Answer:", response["result"]) 
```

接下来是高级 QA 链类型。MapReduce 和 Refine 在处理多份文档时有助于绕过上下文窗口限制：MapReduce 并行聚合，而 Refine 按顺序改进答案。

```py
# Configure a QA chain to use MapReduce, aggregating answers from multiple documents
question_answering_chain_map_reduce = RetrievalQA.from_chain_type(
    language_model,
    retriever=vector_database.as_retriever(),
    chain_type="map_reduce"
)

# Run MapReduce with the user query
response_map_reduce = question_answering_chain_map_reduce({"query": query})

# Show the aggregated answer
print("MapReduce answer:", response_map_reduce["result"])

# Configure a QA chain to use Refine, which iteratively improves the answer
question_answering_chain_refine = RetrievalQA.from_chain_type(
    language_model,
    retriever=vector_database.as_retriever(),
    chain_type="refine"
)

# Run Refine with the same user query
response_refine = question_answering_chain_refine({"query": query})

# Show the refined answer
print("Refine answer:", response_refine["result"]) 
```

在实践中，考虑：根据任务选择 MapReduce 和 Refine（前者用于从多个来源快速聚合；后者用于更高精度和迭代改进）；在分布式系统中，性能取决于网络延迟和序列化；有效性随数据而变化，因此进行实验。

RetrievalQA 的一个显著局限性是缺乏对话历史，这会降低处理后续问题的能力。局限性的演示：

```py
# Import a QA chain from a hypothetical library
from  some_library  import question_answering_chain as qa_chain

# Define an initial question related to course content
initial_question_about_course_content = "Does the curriculum cover probability theory?"
# Generate an answer to the initial question
response_to_initial_question = qa_chain({"query": initial_question_about_course_content})

# Define a follow‑up question without explicitly preserving conversation context
follow_up_question_about_prerequisites = "Why are those prerequisites important?"
# Generate an answer to the follow‑up question
response_to_follow_up_question = qa_chain({"query": follow_up_question_about_prerequisites})

# Display both answers — initial and follow‑up
print("Answer to the initial question:", response_to_initial_question["result"])
print("Answer to the follow‑up question:", response_to_follow_up_question["result"]) 
```

这强调了将对话记忆集成到 RAG 系统中的重要性。

## 结论

RAG 的高级 QA 技术提供更动态和准确的答案。通过仔细的`RetrievalQA`实现和处理其局限性，可以构建能够与用户进行实质性对话的系统。

## 进一步阅读

+   探索 LLMs 的最新进展及其对 RAG 的影响。

+   研究将对话记忆集成到 RAG 框架中的策略。

本章为理解和实践 RAG 的高级 QA 技术以及进一步创新 AI 交互提供了基础。

## 理论问题

1.  在 RAG 中命名 QA 的三个阶段。

1.  上下文窗口限制是什么，MapReduce/Refine 如何帮助绕过它们？

1.  为什么向量数据库（VectorDB）对于 RAG 中的检索很重要？

1.  `RetrievalQA`是如何结合检索和生成的？

1.  比较 MapReduce 和 Refine 方法。

1.  在分布式系统中，哪些实际因素很重要（网络延迟、序列化）？

1.  为什么实验两种方法都很重要？

1.  缺失的对话历史如何影响后续问题的处理？

1.  为什么要将对话记忆集成到 RAG 中？

1.  为了深化 RAG 专业知识，接下来应该研究什么？

## 实践任务

1.  初始化一个向量数据库（Chroma + OpenAIEmbeddings），并打印它包含的文档数量。

1.  使用自定义提示配置`RetrievalQA`，指定模型和数据存储目录。

1.  在单个查询上展示`MapReduce`和`Refine`，并打印出结果答案。

1.  模拟一个不保留对话上下文的后续问题，以展示`RetrievalQA`的限制。
