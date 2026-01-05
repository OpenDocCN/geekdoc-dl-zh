# 2.3 深入探讨文本分割

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-2/2.3%20Deep%20Dive%20into%20Text%20Splitting/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-2/2.3%20Deep%20Dive%20into%20Text%20Splitting/)

分割（分段）发生在将数据加载到“文档”格式之后，但在索引或存储之前。目标是产生具有语义意义的块，这些块对搜索和数据分析很有用，同时不会在边界处破坏意义。最重要的两个参数是块大小和重叠。大小以字符或标记衡量（较大的块携带更多上下文；较小的块更容易处理）。重叠是相邻块之间的“交接”，有助于保持连贯。LangChain 提供了几种策略：基于字符和标记的分割，遵循分隔符层次结构的递归方法，以及针对代码和 Markdown 的专用分割器，这些分割器尊重语法和标题。还有两种操作模式——创建文档（接受原始文本列表并返回分割文档）和分割文档（分割先前加载的文档）——因此根据您是否处理字符串或文档对象进行选择。在实践中，CharacterTextSplitter（当语义不太重要时的简单基于字符的分割）和 TokenTextSplitter（基于标记的分割以适应 LLM 限制）是最常见的。当结构很重要时，遵循层次结构的递归分割器非常有帮助。在专用选项中包括 LanguageTextSplitter 用于代码和 MarkdownHeaderTextSplitter 用于通过标题进行分割，同时保留此结构在元数据中。

在应用分割器之前，快速设置环境是有用的：导入、API 密钥和依赖项。

```py
import  os
from  openai  import OpenAI
import  sys
from  dotenv  import load_dotenv, find_dotenv

# Add the path to access project modules
sys.path.append('../..')

# Load environment variables from the .env file
load_dotenv(find_dotenv())

# Initialize the OpenAI client using environment variables
client = OpenAI() 
```

分割策略强烈影响搜索和数据分析质量，因此调整参数以保留相关性和连贯性。基本选择是 CharacterTextSplitter 和 RecursiveCharacterTextSplitter；根据您数据的结构和性质进行选择。下面是紧凑的示例：首先，一个简单的分割器，具有可选的重叠以帮助保持上下文，

```py
from  langchain.text_splitter  import CharacterTextSplitter

# Define chunk size and overlap for splitting
chunk_size = 26
chunk_overlap = 4

# Initialize a CharacterTextSplitter
character_text_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
) 
```

然后是一个递归分割器，对于“通用”文本，通过遵循分隔符的层次结构更仔细地保留语义——从段落到句子到单词。

```py
from  langchain.text_splitter  import RecursiveCharacterTextSplitter

# Initialize a RecursiveCharacterTextSplitter
recursive_character_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
) 
```

接下来是一些实际示例。从简单的字符串开始，

```py
# A simple alphabet string example
alphabet_text = 'abcdefghijklmnopqrstuvwxyz'

# Try splitting the alphabet string with both splitters
recursive_character_text_splitter.split_text(alphabet_text)
character_text_splitter.split_text(alphabet_text, separator=' ') 
```

然后查看最小分割器实现及其在基本输入上的行为。

```py
# A class that splits text into chunks based on character count.
class  CharacterTextSplitter:
    def  __init__(self, chunk_size, chunk_overlap=0):
  """
 Initialize the splitter with the given chunk size and overlap.

 Args:
 - chunk_size: Number of characters each chunk should contain.
 - chunk_overlap: Number of characters to overlap between neighboring chunks.
 """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def  split_text(self, text):
  """
 Split the given text into chunks according to the configured size and overlap.

 Args:
 - text: The string to split.

 Returns:
 A list of text chunks.
 """
        chunks = []
        start_index = 0

        # Continue splitting until the end of the text is reached.
        while start_index < len(text):
            end_index = start_index + self.chunk_size
            chunks.append(text[start_index:end_index])
            # Advance start index for the next chunk accounting for overlap.
            start_index = end_index - self.chunk_overlap
        return chunks

# Extend CharacterTextSplitter with recursive splitting capabilities.
class  RecursiveCharacterTextSplitter(CharacterTextSplitter):
    def  split_text(self, text, max_depth=10, current_depth=0):
  """
 Recursively split text into smaller chunks until each chunk is below the
 size threshold or the maximum recursion depth is reached.

 Args:
 - text: The string to split.
 - max_depth: Maximum recursion depth to prevent infinite recursion.
 - current_depth: Current recursion depth.

 Returns:
 A list of text chunks.
 """
        # Base case: if max depth reached or text already below threshold, return as-is.
        if current_depth == max_depth or len(text) <= self.chunk_size:
            return [text]
        else:
            # Split into two halves and recurse on each.
            mid_point = len(text) // 2
            first_half = text[:mid_point]
            second_half = text[mid_point:]
            return self.split_text(first_half, max_depth, current_depth + 1) + \
                   self.split_text(second_half, max_depth, current_depth + 1)

# Example usage of the above classes:

# Define chunk size and overlap for splitting.
chunk_size = 26
chunk_overlap = 4

# Initialize the CharacterTextSplitter with the specified size and overlap.
character_text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Initialize the RecursiveCharacterTextSplitter with the specified size.
recursive_character_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)

# Example text to split.
alphabet_text = 'abcdefghijklmnopqrstuvwxyz'

# Use both splitters and store results.
recursive_chunks = recursive_character_text_splitter.split_text(alphabet_text)
simple_chunks = character_text_splitter.split_text(alphabet_text)

# Print results from the recursive splitter.
print("Recursive splitter chunks:")
for chunk in recursive_chunks:
    print(chunk)

# Print results from the simple splitter.
print("\nSimple splitter chunks:")
for chunk in simple_chunks:
    print(chunk) 
```

上面的例子说明了在基本字符串上分割的行为——有和无显式分隔符。现在考虑两种高级技术。首先，处理更复杂的文本，其中显式设置分隔符的层次结构和块大小是有帮助的：

```py
# A sample complex text
complex_text = """When writing documents, writers will use document structure to group content...
Sentences have a period at the end, but also, have a space."""

# Apply recursive splitting with configured chunk size and separators
recursive_character_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""]
)
recursive_character_text_splitter.split_text(complex_text) 
```

这会产生尊重文档内部结构的连贯块。其次，基于标记的分割，其中 LLM 上下文窗口由标记定义，并且必须严格遵守限制：

```py
from  langchain.text_splitter  import TokenTextSplitter

# Initialize a TokenTextSplitter
token_text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

# Split document pages by tokens
document_chunks_by_tokens = token_text_splitter.split_documents(pages) 
```

最后，通过 Markdown 标题进行分割，其中文档的逻辑组织指导了分割，并且检测到的标题在块元数据中得到了保留。

```py
from  langchain.text_splitter  import MarkdownHeaderTextSplitter

# Define the headings to split on in a Markdown document
markdown_headers = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

# Initialize a MarkdownHeaderTextSplitter
markdown_header_text_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=markdown_headers
)

# Split a real Markdown document while preserving heading metadata
markdown_document_splits = markdown_header_text_splitter.split_text(markdown_document_content) 
```

一些快速的建议：保留语义并考虑源文档的结构；管理重叠——只需足够维持连贯性，避免不必要的冗余；使用并丰富元数据以改善检索和回答时的上下文。

## 理论问题

1.  文档分割的目标是什么？

1.  块大小如何影响处理？

1.  为什么需要重叠，以及它如何帮助分析？

1.  `CharacterTextSplitter`和`TokenTextSplitter`有何不同，它们在哪里使用？

1.  递归分割器是什么，它与基本分割器有何不同？

1.  对于代码和 Markdown，存在哪些专门的分割器，它们的优点是什么？

1.  在分割之前需要设置哪些环境？

1.  列出`RecursiveCharacterTextSplitter`的优点和缺点，以及需要调整的重要参数。

1.  当比较简单和递归方法时，“字母表”示例展示了什么？

1.  在选择 LLMs 的字符和标记时，你应该注意什么？

1.  通过 Markdown 标题分割如何保留逻辑结构，为什么那很重要？

1.  哪些最佳实践有助于保留语义和管理重叠？

## 实践任务

1.  编写一个函数`split_by_char(text, chunk_size)`，该函数返回一个固定大小的块列表。

1.  将`chunk_overlap`参数添加到`split_by_char`中并实现重叠。

1.  实现一个类`TokenTextSplitter(chunk_size, chunk_overlap)`，它有一个`split_text`方法，通过标记（由空格分隔的标记）分割文本。

1.  编写一个函数`recursive_split(text, max_chunk_size, separators)`，该函数通过给定的分隔符列表递归地分割文本。

1.  实现一个类`MarkdownHeaderTextSplitter(headers_to_split_on)`，它有一个`split_text`方法，通过指定的标题分割 Markdown，并返回带有相应元数据的块。
