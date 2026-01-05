# 2.5 语义搜索——高级策略

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-2/2.5%20Semantic%20Search%20%E2%80%94%20Advanced%20Strategies/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-2/2.5%20Semantic%20Search%20%E2%80%94%20Advanced%20Strategies/)

从大型语料库中精确提供相关信息是智能系统（如聊天机器人和问答）的关键。基本的语义搜索是一个良好的起点，但存在一些边缘情况，其质量和结果多样性不足。本章探讨了高级检索技术以提高精确性和多样性。

仅基于语义邻近度的搜索并不总是能产生最有信息和多样化的结果集。高级方法添加机制以平衡多样性和相关性——这对于需要细微差别的复杂查询尤为重要。

进入最大边际相关性（MMR）。MMR 平衡相关性和多样性：它选择与查询接近但彼此不相似的文档。这减少了冗余并有助于覆盖答案的不同方面。

该过程看起来是这样的：首先，通过语义相似性选择“候选人”；然后选择一个最终集，该集同时考虑了与查询的相关性和与已选文档的不相似性。结果是更广泛、更有用的结果集。

接下来是自查询检索。这种方法适用于包含语义内容和元数据（例如，“1980 年发布的科幻电影”）的查询。它将请求分为语义组件（用于嵌入搜索）和元数据过滤器（例如，“发布年份 = 1980”）。

最后，上下文压缩仅从检索到的文档中提取最相关的片段。当你不需要整个文档时，这很有用。它需要额外的处理步骤（找到最相关的部分），但显著提高了准确性和特异性。

转向加强语义搜索的高级检索技术实践方面：在 RAG（检索增强生成）系统（如聊天机器人和问答）中，检索相关文档是一个关键阶段。以下技术有助于处理基本搜索中的边缘情况，并增加结果集的多样性和特异性。

在 RAG（检索增强生成）系统（如聊天机器人和问答）中，检索相关文档是一个关键阶段。以下技术有助于处理基本搜索中的边缘情况，并增加结果集的多样性和特异性。

在开始之前，导入必要的库并配置对外部服务的访问（例如，OpenAI 用于嵌入）。

```py
# Import required libraries
import  os
from  openai  import OpenAI
import  sys

# Add the project root to sys.path for relative imports
sys.path.append('../..')

# Load environment variables from .env for safe API key management
from  dotenv  import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Initialize OpenAI using environment variables
client = OpenAI()

# Ensure required packages are installed, including `lark` for parsing if needed
# !pip install lark 
```

现在配置一个向量存储库，以高效执行基于意义的搜索（使用映射到高维向量的嵌入）。

```py
# Import the Chroma vector store and OpenAI embeddings from LangChain
from  langchain.vectorstores  import Chroma
from  langchain_openai  import OpenAIEmbeddings

# Directory for the vector database to persist its data
persist_directory = 'vector_db/chroma/'

# Initialize the embedding function using an OpenAI model
embedding_function = OpenAIEmbeddings()

# Create a Chroma vector database with persistence and the embedding function
vector_database = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function
)

# Print the current record count to verify readiness
# Note: Using internal API for demo purposes. For production, use len() or other supported methods.
print(len(vector_database.get()['ids']))  # Alternative to deprecated _collection.count() 
```

添加一个小型演示集以展示相似性搜索和 MMR。

```py
# A small set of texts to populate the database
texts = [
    "The Death Cap mushroom has a notable large fruiting body, often found above ground.",
    "Among mushrooms, the Death Cap stands out for its large fruiting body, sometimes appearing in all-white.",
    "The Death Cap, known for its toxicity, is one of the most dangerous mushrooms.",
]

# Create a tiny demonstration vector database from the texts
demo_vector_database = Chroma.from_texts(texts, embedding_function=embedding_function)

# A sample query for the demo vector database
query_text = "Discuss mushrooms characterized by their significant white fruiting bodies"

# Similarity search: top‑2 most relevant
similar_texts = demo_vector_database.similarity_search(query_text, k=2)
print("Similarity search results:", similar_texts)

# MMR search: diverse yet relevant (fetch extra candidates)
diverse_texts = demo_vector_database.max_marginal_relevance_search(query_text, k=2, fetch_k=3)
print("Diverse search (MMR) results:", diverse_texts) 
```

一个常见问题是结果过于相似。MMR 平衡相关性和多样性，减少重复并扩大覆盖范围。一个实际的 MMR 示例：

```py
# An information‑seeking query
query_for_information = "what insights are available on data analysis tools?"

# Standard similarity search: top‑3 relevant documents
top_similar_documents = vector_database.similarity_search(query_for_information, k=3)

# Show the beginning of the first two documents for comparison
print(top_similar_documents[0].page_content[:100])
print(top_similar_documents[1].page_content[:100])

# Note potential overlap. Introduce diversity with MMR.
diverse_documents = vector_database.max_marginal_relevance_search(query_for_information, k=3)

# Show the beginning of the first two diverse documents to observe differences
print(diverse_documents[0].page_content[:100])
print(diverse_documents[1].page_content[:100]) 
```

此示例显示了标准相似度搜索和 MMR 之间的差异：后者产生相关但重复性较低的结果。

### 通过元数据提高准确性

元数据有助于通过属性（来源、日期等）精炼查询和过滤结果。

#### 元数据过滤搜索

```py
# A query scoped to a specific context
specific_query = "what discussions were there about regression analysis in the third lecture?"

# Similarity search with a metadata filter to target a specific lecture
targeted_documents = vector_database.similarity_search(
    specific_query,
    k=3,
    filter={"source": "documents/cs229_lectures/MachineLearning-Lecture03.pdf"}
)

# Inspect metadata to highlight the specificity of the search
for document in targeted_documents:
    print(document.metadata) 
```

### 结合元数据和自查询检索器

自查询检索器从单个用户短语中提取语义查询和元数据过滤器——无需手动指定过滤器。

#### 初始化和元数据描述

在运行元数据感知搜索之前，请定义要使用的元数据属性：

```py
# Import required modules from LangChain
from  langchain_openai  import OpenAI
from  langchain.retrievers.self_query.base  import SelfQueryRetriever
from  langchain.chains.query_constructor.base  import AttributeInfo

# Define the metadata attributes with detailed descriptions
metadata_attributes = [
    AttributeInfo(
        name="source",
        description="Specifies the source document path.",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="Page number within the lecture document.",
        type="integer",
    ),
]

# Note: switching to the OpenAI model gpt‑4o‑mini, as the previous default is deprecated
document_content_description = "Detailed lecture notes"
language_model = OpenAI(model='gpt-4o-mini', temperature=0) 
```

#### 配置自查询检索器

```py
# Initialize the Self‑Query Retriever with the LLM, vector DB, and metadata attributes
self_query_retriever = SelfQueryRetriever.from_llm(
    language_model,
    vector_database,
    document_content_description,
    metadata_attributes,
    verbose=True
) 
```

#### 使用自动推断的元数据运行查询

```py
# A query that encodes context directly in the question
specific_query = "what insights are provided on regression analysis in the third lecture?"

# Note: the first run may emit a deprecation warning for `predict_and_parse`; you can ignore it.
# Retrieve documents relevant to the specific query using inferred metadata
relevant_documents = self_query_retriever.get_relevant_documents(specific_query)

# Display metadata to demonstrate specificity
for document in relevant_documents:
    print(document.metadata) 
```

## 实现上下文压缩

上下文压缩通过提取与给定查询最相关的文档段落来工作。这种方法不仅减少了 LLMs 的计算负担，而且通过关注最相关的信息来提高答案质量。

### 设置环境

在深入研究上下文压缩的细节之前，请确保您的环境已正确配置了必要的库：

```py
# Import classes for contextual compression and document retrieval
from  langchain.retrievers  import ContextualCompressionRetriever
from  langchain.retrievers.document_compressors  import LLMChainExtractor
from  langchain_openai  import OpenAI 
```

### 初始化压缩工具

接下来，使用一个预训练的语言模型初始化压缩机制，该模型将识别并提取文档的相关部分：

```py
# Initialize the language model with deterministic settings
language_model = OpenAI(temperature=0, model="gpt-4o-mini")

# Create a compressor that uses the LLM to extract relevant segments
document_compressor = LLMChainExtractor.from_llm(language_model) 
```

### 创建上下文压缩检索器

压缩器准备就绪后，配置一个将上下文压缩集成到检索过程中的检索器：

```py
# Combine the document compressor with the vector DB retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=document_compressor,
    base_retriever=vector_database.as_retriever()
) 
```

运行查询并查看压缩感知检索器如何返回一组更聚焦的文档：

```py
# Define a query to look for relevant document segments
query_text = "what insights are offered on data analysis tools?"

# Retrieve documents relevant to the query, automatically compressed for relevance
compressed_documents = compression_retriever.get_relevant_documents(query_text)

# Helper to pretty‑print compressed document contents
def  pretty_print_documents(documents):
    print(f"\n{'-'  *  100}\n".join([f"Document {index  +  1}:\n\n" + doc.page_content for index, doc in enumerate(documents)]))

# Display the compressed documents
pretty_print_documents(compressed_documents) 
```

上下文压缩旨在通过关注与查询最相关的段落来提取文档的精华。结合 MMR，它平衡了相关性和多样性，为话题提供了更广阔的视角。配置具有压缩和 MMR 的检索器：

```py
# Initialize a retriever that uses both contextual compression and MMR
compression_based_retriever = ContextualCompressionRetriever(
    base_compressor=document_compressor,
    base_retriever=vector_database.as_retriever(search_type="mmr")
)

# A query to test the combined approach
query_for_insights = "what insights are available on statistical analysis methods?"

# Retrieve compressed documents using the compression‑aware retriever
compressed_documents = compression_based_retriever.get_relevant_documents(query_for_insights)

# Pretty‑print the compressed documents
pretty_print_documents(compressed_documents) 
```

此方法优化了检索，确保结果不仅相关，而且多样，防止冗余并提高用户对主题的理解。

除了语义搜索之外，还有其他检索方法。TF-IDF（词频-逆文档频率）是衡量一个词在集合中重要性的统计度量：它考虑了文档中的词频和在语料库中的稀有度；高值表示良好的描述符，适用于精确匹配搜索。SVM（支持向量机）可用于文档分类，并通过过滤或按预定义类别对文档进行排序间接提高检索。

## 有用链接

+   LangChain 自查询检索器：https://python.langchain.com/docs/modules/data_connection/retrievers/self_query

+   LangChain 最大边际相关度检索器：https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/mmr

+   TF-IDF 解释：https://www.youtube.com/watch?v=BtWcKEmM0g4

+   SVM 解释：https://www.youtube.com/watch?v=efR1C6CvhmE

## 理论问题

1.  最大边际相关度（MMR）相对于标准相似度搜索在文档检索中提供了哪些优势？

1.  元数据如何提高语义搜索中的精确性和相关性？

1.  描述 Self-Query Retriever 的工作原理及其关键优势。

1.  在信息检索中何时使用 TF-IDF 和 SVM 是合理的，它们与基于嵌入的方法有何不同？

## 实际任务

1.  使用不同参数实现 MMR：对`k`和`fetch_k`进行实验，并分析它们如何影响多样性和相关性。

1.  扩展元数据：添加新类型（例如，作者、出版日期、关键词）并用于过滤搜索。

1.  集成 Self-Query Retriever：扩展元数据属性描述以包括新字段，并验证它是否可以自动形成复杂、受限的查询。

1.  比较方法：在你的集合上实现基于 TF-IDF 或 SVM 的简单搜索，并与语义搜索进行比较，注意不同场景中的优势和劣势。

## 最佳实践

除了有效地使用各种检索技术外，遵循最佳实践以确保最大性能和可靠性。

### 选择正确的策略

在 MMR、Self-Query Retriever 或普通相似度搜索之间进行选择取决于应用需求。当你需要多样化的结果时，MMR 是最优的。如果用户查询包含显式元数据，Self-Query Retriever 简化了过程。标准相似度搜索适合简单的查询。

### 性能优化

向量数据库的性能，尤其是在大规模上，至关重要。定期索引、缓存常用查询和硬件优化可以显著加快检索速度。分布式向量数据库也有助于扩展。

### 元数据管理

结构良好且准确的元数据可以显著提高搜索质量。建立一个深思熟虑的元数据架构，并在整个集合中一致地应用它。使用 LLM 自动生成元数据可能会有所帮助，但需要仔细验证。

### 监控和迭代

检索系统需要持续监控性能和结果质量。收集用户反馈、分析相关性指标，并通过 A/B 测试检索策略来迭代改进系统。

## 结论

本章概述了旨在改进语义搜索系统的先进检索技术。通过解决多样性、特异性和相关性的限制，这些方法为更智能和有效的检索提供了一条途径。通过实际应用 MMR、自查询检索、上下文压缩和替代文档检索方法，开发者可以构建不仅理解查询的语义内容，还能提供丰富、多样和有针对性的答案的系统。

遵循最佳实践确保检索系统既高效又有效。随着 NLP 的不断发展，跟上检索技术的前沿将是在语义搜索能力上保持优势的关键。

总之，将高级检索技术集成到语义搜索系统中是一个重要的进步。通过谨慎选择和优化，开发者可以构建出能够显著提升用户体验的解决方案，通过响应复杂查询提供准确、多样化和上下文相关的信息。

## 其他理论问题

1.  解释最大边际相关性（MMR）的原则及其在提高检索质量中的作用。

1.  自查询检索如何处理结合语义内容和元数据的查询？

1.  解释文档检索中的上下文压缩及其重要性。

1.  列出使用 OpenAI API 和 LangChain 进行高级检索的环境设置步骤。

1.  初始化向量数据库如何实现高效的语义相似性搜索？

1.  描述如何填充和使用向量数据库以进行相似性和多样化（MMR）搜索。

1.  在高级文档检索中，MMR 如何确保多样性？

1.  如何利用元数据来增加文档检索系统的特定性？

1.  讨论自查询检索在语义搜索中的优势和挑战。

1.  上下文压缩在减少计算负载和提高答案质量方面扮演什么角色？

1.  在实现语义搜索系统的高级检索时，哪些最佳实践最为重要？

1.  比较基于向量的检索与 TF-IDF 和 SVM 在文档检索中的有效性。

1.  集成高级检索技术如何提高语义搜索系统的性能和用户体验？

1.  未来的 NLP 进步可能会对语义搜索的高级检索产生什么影响？

## 其他实用任务

1.  实现一个名为 `VectorDatabase` 的 Python 类，包含以下方法：

1.  `__init__(self, persist_directory: str)`: 初始化向量数据库及其持久化目录。

1.  `add_text(self, text: str)`: 使用 OpenAI 嵌入将文本嵌入到高维向量中并存储它。假设存在一个函数 `openai_embedding(text: str) -> List[float]` 返回嵌入向量。

1.  `similarity_search(self, query: str, k: int) -> List[str]`: 执行相似性搜索并返回最相似的 `k` 个文本。使用简化的相似性函数。

1.  编写一个函数 `compress_document`，它接受一个字符串列表（文档）和一个查询字符串，并返回一个字符串列表，其中每个元素都是与查询相关的文档压缩段。假设存在一个外部实用工具 `compress_segment(segment: str, query: str) -> str`，该工具用于压缩单个段以匹配查询。

1.  实现 `max_marginal_relevance` 函数，它接受文档 ID 列表、查询和两个参数 `lambda` 和 `k`，并返回根据最大边际相关性标准选择的 `k` 个 ID 列表。假设存在相似性函数 `similarity(doc_id: str, query: str) -> float` 和多样性函数 `diversity(doc_id1: str, doc_id2: str) -> float`。

1.  编写`initialize_vector_db`函数，演示如何使用预定义的文本列表填充向量数据库，然后运行相似性和多样化搜索，打印出两组结果。使用任务 1 中的`VectorDatabase`类作为后端存储。
