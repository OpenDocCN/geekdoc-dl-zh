# 答案 2.5

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-2/Answers%202.5/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-2/Answers%202.5/)

## 理论

1.  最大边际相关性（MMR）：通过选择接近查询但彼此不同的文档来平衡相关性和多样性。

1.  自查询检索：将查询拆分为语义内容和元数据约束，以实现精确的内容加属性检索。

1.  上下文压缩：仅从文档中提取最相关的部分，减少噪声并提高答案质量。

1.  环境设置：安装库、配置 API 访问（用于嵌入）和初始化向量存储库——这是高级检索的基础。

1.  向量存储：存储嵌入并支持快速相似度搜索。

1.  填充存储库：添加文本并运行相似度搜索；MMR 有助于消除冗余。

1.  使用 MMR（最大边际相关性）提升多样性：减少近重复项的聚类并扩大覆盖范围。

1.  用于特定性的元数据：属性（例如，日期、类型）提高精度和相关性。

1.  自查询检索器：自动从用户输入中提取语义和元数据部分。

1.  上下文压缩的好处：节省计算资源并关注本质。

1.  最佳实践：调整 MMR，明智地利用元数据，仔细配置压缩，并彻底准备文档。

1.  结合方法：基于嵌入的检索在意义上表现卓越，而 TF-IDF 或 SVM 可以帮助处理基于关键词或分类的场景。

1.  高级技术的优势：提高了精度、多样性、上下文和整体用户体验。

1.  NLP 展望：持续的进步将带来更智能地处理复杂查询的能力。

## 实际任务

1.

```py
from  typing  import List
import  numpy  as  np

def  openai_embedding(text: str) -> List[float]:
    # Placeholder: return a random vector instead of calling OpenAI.
    return np.random.rand(768).tolist()

def  cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1 = np.array(vec1); v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

class  VectorDatabase:
    def  __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.database = []  # (text, embedding)

    def  add_text(self, text: str):
        self.database.append((text, openai_embedding(text)))

    def  similarity_search(self, query: str, k: int) -> List[str]:
        q = openai_embedding(query)
        scored = [(t, cosine_similarity(q, e)) for t, e in self.database]
        return [t for t, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]

if __name__ == "__main__":
    db = VectorDatabase("path/to/persist")
    db.add_text("The quick brown fox jumps over the lazy dog.")
    db.add_text("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    db.add_text("Python is a popular programming language for data science.")
    print("Similarity results:", db.similarity_search("Programming in Python", 2)) 
```

2.

```py
def  compress_segment(segment: str, query: str) -> str:
    # Placeholder: return half the segment.
    return segment[:len(segment)//2]

def  compress_document(document: List[str], query: str) -> List[str]:
    return [compress_segment(s, query) for s in document]

doc = [
    "The first chapter introduces the concepts of machine learning.",
    "Machine learning techniques are varied and serve different purposes.",
    "In data analysis, regression models can predict continuous outcomes.",
]
print("Compressed:", compress_document(doc, "machine learning")) 
```

3.

```py
def  similarity(doc_id: str, query: str) -> float: return 0.5
def  diversity(doc_id1: str, doc_id2: str) -> float: return 0.5

def  max_marginal_relevance(doc_ids: List[str], query: str, lambda_param: float, k: int) -> List[str]:
    selected, remaining = [], doc_ids.copy()
    while len(selected) < k and remaining:
        scores = {
            d: lambda_param * similarity(d, query) - (1 - lambda_param) * max([diversity(d, s) for s in selected] or [0])
            for d in remaining
        }
        nxt = max(scores, key=scores.get)
        selected.append(nxt)
        remaining.remove(nxt)
    return selected

print(max_marginal_relevance(["d1","d2","d3"], "query", 0.7, 2)) 
```

4.

```py
def  initialize_vector_db():
    # Initialize the vector DB using the VectorDatabase class defined above
    vector_db = VectorDatabase("path/to/persist/directory")

    # Sample texts to add
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Python is a popular programming language for data science."
    ]

    for text in texts:
        vector_db.add_text(text)

    # Similarity search
    query = "data science"
    similar_texts = vector_db.similarity_search(query, 2)
    print("Similarity search results:", similar_texts)

    # Placeholder for “diverse search” demonstration — call MMR or similar here in a real setup
    print("Diverse search (simulated):", similar_texts)

# Run the demonstration
initialize_vector_db() 
```
