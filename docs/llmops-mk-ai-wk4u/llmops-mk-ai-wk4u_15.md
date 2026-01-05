# 2.4 嵌入的力量

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-2/2.4%20The%20Power%20of%20Embeddings/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-2/2.4%20The%20Power%20of%20Embeddings/)
> 
> **要求：** `pip install langchain langchain-community langchain-openai chromadb`
> 
> **注意：** LangChain 的导入已更新。对于最新版本，请使用 `from langchain_community.vectorstores import Chroma` 和 `from langchain_community.document_loaders import PyPDFLoader` 而不是旧的 `langchain.*` 路径。

嵌入是文本的数值表示：单词、句子和文档被映射到高维空间中的向量，语义相似的文本在几何上靠近。这些表示是从大型语料库中学习到的：模型将一个词与其上下文相关联并捕获语义关系，因此同义词和在相似上下文中出现的术语彼此靠近。因此，语义搜索超越了精确的“关键词”匹配：为每个文档（或块）和用户查询计算嵌入，通过余弦或其他指标比较向量邻近度，并按语义相似度对材料进行排序——即使没有精确匹配。这改变了我们分析、存储和搜索的方式：交互变得更加有意义，推荐更加精确。

在嵌入之上是向量存储——针对向量存储和快速最近邻搜索优化的数据库。它们使用专门的索引和算法来回答大型数据集上的相似性查询，适用于研究和生产。根据数据大小（从小型集的内存选项到大规模的分布式系统）、持久性（您是否需要耐用的磁盘存储或用于原型的临时存储）以及用例（实验室与生产）进行选择。对于快速原型设计，Chroma 是一个常见的选择——一个轻量级的内存存储；对于更大和长期运行的系统，使用分布式/云向量数据库。在典型的语义搜索管道中，文档首先被分割成有意义的块，然后计算嵌入并索引；在查询时，计算其嵌入，检索最近的块，并将提取的部分加上查询输入到 LLM 以生成一个连贯的答案。

在深入研究嵌入和向量数据库之前，准备环境：导入、API 密钥和基本配置。

```py
import  os
from  openai  import OpenAI
import  sys
from  dotenv  import load_dotenv, find_dotenv

sys.path.append('../..')

load_dotenv(find_dotenv())

client = OpenAI() 
```

接下来，加载文档并将它们分割成具有语义意义的片段——这使数据更容易管理，并为嵌入创建做好准备。我们将使用一系列 PDF 文件（包含一些“噪声”如重复内容）进行演示：

```py
from  langchain.document_loaders  import PyPDFLoader

pdf_document_loaders = [
    PyPDFLoader("docs/doc1.pdf"),
    PyPDFLoader("docs/doc2.pdf"),
    PyPDFLoader("docs/doc3.pdf"),
]

loaded_documents_content = []

for document_loader in pdf_document_loaders:
    loaded_documents_content.extend(document_loader.load()) 
```

加载后，将文档分割成块以提高可管理性和下游效率：

```py
from  langchain.text_splitter  import RecursiveCharacterTextSplitter

document_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)
document_splits = document_splitter.split_documents(documents) 
```

现在为每个块计算嵌入：将文本转换为反映语义意义的向量。

```py
from  langchain_openai  import OpenAIEmbeddings
import  numpy  as  np

embedding_generator = OpenAIEmbeddings()

sentence_examples = ["I like dogs", "I like canines", "The weather is ugly outside"]
embeddings = [embedding_generator.embed_query(sentence) for sentence in sentence_examples]

similarity_dog_canine = np.dot(embeddings[0], embeddings[1])
similarity_dog_weather = np.dot(embeddings[0], embeddings[2]) 
```

在向量存储中索引向量以实现快速相似性搜索。对于演示，Chroma——一个内存选项——表现良好：

```py
from  langchain.vectorstores  import Chroma

persist_directory = 'chroma_db/'

# Clear previous database if exists (use Python for cross-platform compatibility)
import  shutil
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

vector_database = Chroma.from_documents(
    documents=document_splits,
    embedding=embedding_generator,
    persist_directory=persist_directory
) 
```

现在进行相似性搜索——这是嵌入+向量数据库大放异彩的地方：快速选择查询中最相关的片段。

```py
query = "Is there an email I can ask for help?"
retrieved_documents = vector_database.similarity_search(query, k=3)
print(retrieved_documents[0].page_content) 
```

最后，考虑边缘情况和搜索质量改进。即使是一个有用的基线也会遇到问题：重复和不相关的文档是常见的导致结果下降的问题。

```py
# Query example illustrating a failure mode
query_matlab = "What did they say about MATLAB?"

# Detect duplicate fragments in search results
retrieved_documents_matlab = vector_database.similarity_search(query_matlab, k=5) 
```

从那里，你可以应用策略来减轻此类失败，检索到既相关又足够多样化的片段。总的来说，嵌入和向量数据库是大型语料库语义搜索的强大组合：坚实的文本准备、深思熟虑的索引和快速的最近邻查询使系统能够理解复杂的提示；分析失败并添加技术进一步提高了鲁棒性和准确性。对于更深入的研究，请参阅 OpenAI API 文档中关于嵌入生成和比较技术和使用场景的向量数据库调查。

## 理论问题

1.  将文本转换为嵌入的主要目标是什么？

1.  嵌入如何帮助衡量单词和句子的语义相似度？

1.  描述单词嵌入的创建过程和上下文的作用。

1.  嵌入如何改进基于关键词的语义搜索？

1.  文档和查询嵌入在语义搜索中扮演什么角色？

1.  什么是向量存储，为什么它对高效搜索很重要？

1.  选择向量数据库时，哪些标准很重要？

1.  为什么 Chroma 对原型设计很方便，以及它的局限性是什么？

1.  描述一个使用嵌入和向量数据库的语义搜索管道。

1.  文档分割如何提高搜索粒度和相关性？

1.  为什么嵌入块，以及这如何帮助检索？

1.  为什么需要对向量存储进行索引以进行相似性搜索？

1.  查询是如何处理的，以及使用了哪些相似性度量？

1.  答案生成如何改善语义搜索应用的用户体验？

1.  需要哪些环境设置步骤？

1.  请举一个加载和分割文本对搜索质量至关重要的例子。

1.  嵌入如何“转换”文本，以及如何展示向量相似性？

1.  配置 Chroma 时应该考虑什么？

1.  相似性搜索如何找到相关片段？

1.  语义搜索中典型的失败有哪些，如何解决它们？

## 实践任务

1.  实现 `generate_embeddings`，它返回字符串的“嵌入”列表（例如，通过字符串长度模拟）。

1.  实现 `cosine_similarity` 以计算两个向量之间的余弦相似度。

1.  使用 `add_vector` 和 `find_most_similar`（基于余弦）创建 `SimpleVectorStore`。

1.  从文件中加载文本，将其分割成给定大小的块（例如，500 个字符），并打印它们。

1.  实现 `query_processing`：生成查询嵌入（占位符），在 `SimpleVectorStore` 中找到最近的块，并打印它。

1.  实现 `remove_duplicates`：返回一个不包含重复块（精确匹配或通过相似度阈值）的列表。

1.  初始化 `SimpleVectorStore`，添加占位符嵌入，运行语义搜索，并打印前 3 个结果。

1.  实现 `embed_and_store_documents`：为块生成占位符嵌入，将它们存储在 `SimpleVectorStore` 中，并返回。

1.  实现 `vector_store_persistence`：演示保存/加载 `SimpleVectorStore`（序列化/反序列化）。

1.  实现 `evaluate_search_accuracy`：对于查询和预期块，运行搜索并计算匹配率。
