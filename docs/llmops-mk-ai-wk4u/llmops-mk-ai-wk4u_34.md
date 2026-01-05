# 答案 2.4

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-2/Answers%202.4/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-2/Answers%202.4/)

## 理论

1.  嵌入的目的：将文本转换为保留语义意义的数值向量，使计算机能够“理解”文本。

1.  语义相似性：在高维空间中，具有相关意义的单词/句子具有相似的向量。

1.  嵌入是如何学习的：在文本语料库上训练；词向量取决于使用上下文（分布语义）。

1.  语义搜索中的嵌入：允许通过意义检索相关文档，即使没有精确的关键词匹配。

1.  匹配文档和查询：文档嵌入代表整体意义；查询嵌入捕捉用户意图；比较它们揭示相关匹配。

1.  向量存储：一个针对快速最近邻搜索优化的嵌入数据库。

1.  选择存储：取决于数据大小、持久化需求和目的（研究、原型、生产）。

1.  染色体用于原型设计：在小/内存场景中方便（快速），但在持久化和扩展方面有限。

1.  典型管道：分割文本 → 生成嵌入 → 在向量存储中索引 → 处理查询 → 生成答案。

1.  分割：提高粒度；匹配发生在有意义的片段（块）级别，而不是整个文档。

1.  嵌入生成：将文本转换为适合计算比较的向量。

1.  在存储中索引：使快速检索语义相似的片段成为可能。

1.  查询处理：创建查询嵌入，并使用度量（余弦相似度、欧几里得距离等）搜索相似片段。

1.  答案生成：使用检索到的片段与原始查询一起生成一个连贯的答案。

1.  环境设置：安装库、配置 API 密钥，并为嵌入和向量存储设置。

1.  加载和分割文档：对有效的文本管理和高质量检索至关重要。

1.  说明相似性：可以通过点积或余弦相似度来展示。

1.  染色体具体细节：注意持久化目录，清除过时数据，以及正确的集合初始化。

1.  相似性搜索：找到与查询最相关的片段。

1.  常见故障和补救措施：通过过滤和仔细的管道调整可以解决重复和不相关的结果。

## 练习

1.

```py
def  generate_embeddings(sentences):
  """
 Generate a simple placeholder embedding for each sentence based on its length.

 Args:
 - sentences (list of str): List of sentences to embed.

 Returns:
 - list of int: One embedding per sentence (the sentence length).
 """
    return [len(sentence) for sentence in sentences]

def  cosine_similarity(vector_a, vector_b):
  """
 Compute cosine similarity between two vectors.

 Args:
 - vector_a (list of float): First vector.
 - vector_b (list of float): Second vector.

 Returns:
 - float: Cosine similarity between `vector_a` and `vector_b`.
 """
    dot_product = sum(a*b for a, b in zip(vector_a, vector_b))
    magnitude_a = sum(a**2 for a in vector_a) ** 0.5
    magnitude_b = sum(b**2 for b in vector_b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)

# Example usage:
sentences = ["Hello, world!", "This is a longer sentence.", "Short"]
embeddings = generate_embeddings(sentences)
print("Embeddings:", embeddings)

vector_a = [1, 2, 3]
vector_b = [2, 3, 4]
similarity = cosine_similarity(vector_a, vector_b)
print("Cosine similarity:", similarity) 
```

2.

```py
def  cosine_similarity(vector_a, vector_b):
  """Compute cosine similarity between two vectors."""
    dot_product = sum(a*b for a, b in zip(vector_a, vector_b))
    magnitude_a = sum(a**2 for a in vector_a) ** 0.5
    magnitude_b = sum(b**2 for b in vector_b) ** 0.5
    if magnitude_a == 0 or magnitude_b == 0:
        return 0  # Avoid division by zero
    return dot_product / (magnitude_a * magnitude_b) 
```

3.

```py
class  SimpleVectorStore:
    def  __init__(self):
        self.vectors = []  # Initialize empty list to store vectors

    def  add_vector(self, vector):
  """Add a vector to the store."""
        self.vectors.append(vector)

    def  find_most_similar(self, query_vector):
  """Find and return the vector most similar to `query_vector`."""
        if not self.vectors:
            return None  # Return None if the store is empty
        similarities = [cosine_similarity(query_vector, vector) for vector in self.vectors]
        max_index = similarities.index(max(similarities))
        return self.vectors[max_index] 
```

4.

```py
import  sys

def  split_text_into_chunks(text, chunk_size):
  """Split the given text into chunks of the specified size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def  load_and_print_chunks(file_path, chunk_size):
  """Load text from a file, split it into chunks, and print each chunk."""
    try:
        with open(file_path, 'r') as file:
            text = file.read()
            chunks = split_text_into_chunks(text, chunk_size)
            for i, chunk in enumerate(chunks, 1):
                print(f"Chunk {i}:\n{chunk}\n{'-'*50}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <file_path> <chunk_size>")
        sys.exit(1)
    file_path = sys.argv[1]
    chunk_size = int(sys.argv[2])
    load_and_print_chunks(file_path, chunk_size) 
```

5.

```py
# Assume SimpleVectorStore and cosine_similarity are defined earlier.

def  generate_query_embedding(query):
  """
 Generate a simple placeholder embedding for the query based on its length.
 In real scenarios, you would use a model for embeddings.
 """
    return [len(query)]

def  query_processing(store, query):
  """
 Process a query: generate its embedding, find the most similar fragment in the
 vector store, and print it.
 """
    query_embedding = generate_query_embedding(query)
    most_similar = store.find_most_similar(query_embedding)
    if most_similar is not None:
        print("Most similar document fragment:", most_similar)
    else:
        print("No document fragments found.") 
```

6.

```py
def  remove_duplicates(document_chunks):
  """Remove duplicate document fragments by exact content match."""
    unique_chunks = []
    for chunk in document_chunks:
        if chunk not in unique_chunks:
            unique_chunks.append(chunk)
    return unique_chunks 
```

7.

```py
# Initialize SimpleVectorStore for demonstration
store = SimpleVectorStore()

# Placeholder document fragments and their embeddings
document_chunks = ["Document chunk 1", "Document chunk 2", "Document chunk 3"]
# Simulate embeddings based on length
document_embeddings = [[len(chunk)] for chunk in document_chunks]

# Add generated document embeddings to the store
for embedding in document_embeddings:
    store.add_vector(embedding)

# Perform similarity search with a sample query
query = "Document"
query_embedding = generate_query_embedding(query)

# Find the most similar document fragments via cosine similarity
similarities = [(cosine_similarity(query_embedding, doc_embedding), idx) for idx, doc_embedding in enumerate(document_embeddings)]
similarities.sort(reverse=True)  # Sort by similarity descending
top_n_indices = [idx for _, idx in similarities[:3]]  # Indices of top‑3 fragments

# Print IDs or contents of the top‑3 most similar document fragments
print("Top‑3 most similar document fragments:")
for idx in top_n_indices:
    print(f"{idx  +  1}: {document_chunks[idx]}") 
```

8.

```py
def  embed_and_store_documents(document_chunks):
  """
 Generate embeddings for each document fragment and store them in SimpleVectorStore.

 Args:
 - document_chunks (list of str): List of document fragments.

 Returns:
 - SimpleVectorStore: Vector store initialized with document embeddings.
 """
    store = SimpleVectorStore()
    for chunk in document_chunks:
        # Placeholder embedding based on fragment length
        embedding = [len(chunk)]
        store.add_vector(embedding)
    return store 
```

9.

```py
import  json

def  save_vector_store(store, filepath):
  """
 Save the state of a SimpleVectorStore to the specified file.

 Args:
 - store (SimpleVectorStore): Vector store to save.
 - filepath (str): Path to the output file.
 """
    with open(filepath, 'w') as file:
        json.dump(store.vectors, file)

def  load_vector_store(filepath):
  """
 Load a SimpleVectorStore from the specified file.

 Args:
 - filepath (str): Path to the input file.

 Returns:
 - SimpleVectorStore: Loaded vector store.
 """
    store = SimpleVectorStore()
    with open(filepath, 'r') as file:
        store.vectors = json.load(file)
    return store

def  vector_store_persistence():
  """Demonstrate saving and loading the state of SimpleVectorStore."""
    store = SimpleVectorStore()  # Assume it is already populated
    filepath = 'vector_store.json'

    # Example of saving and loading
    save_vector_store(store, filepath)
    loaded_store = load_vector_store(filepath)
    print("Vector store loaded with vectors:", loaded_store.vectors) 
```

10.

```py
def  evaluate_search_accuracy(queries, expected_chunks):
  """
 Evaluate similarity‑search accuracy for a list of queries and expected results.

 Args:
 - queries (list of str): Query strings.
 - expected_chunks (list of str): Expected most similar document fragments for each query.

 Returns:
 - float: Retrieval accuracy (fraction of correctly found fragments).
 """
    correct = 0
    # Embed and store documents plus some extras to ensure uniqueness
    store = embed_and_store_documents(expected_chunks + list(set(expected_chunks) - set(queries)))

    for query, expected in zip(queries, expected_chunks):
        query_embedding = generate_query_embedding(query)
        most_similar = store.find_most_similar(query_embedding)
        # Assume expected_chunks map to embeddings by length in the same way
        if most_similar and most_similar == [len(expected)]:
            correct += 1

    accuracy = correct / len(queries)
    return accuracy

# Assume embed_and_store_documents, generate_query_embedding, and SimpleVectorStore
# are implemented as described above. 
```
