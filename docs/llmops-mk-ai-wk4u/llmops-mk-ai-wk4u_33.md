# 答案 2.3

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-2/Answers%202.3/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-2/Answers%202.3/)

## 理论

1.  分割的目标是生成有意义的块，以进行有效的搜索和分析。

1.  块大小控制粒度：较大→更多上下文；较小→更容易处理但风险丢失连贯性。

1.  重叠在边界处保留上下文并防止丢失重要信息。

1.  `CharacterTextSplitter`按字符分割；`TokenTextSplitter`按标记分割（对于 LLM 限制很有用）。

1.  递归分割器使用分隔符的层次结构（段落/句子/单词）来保留语义。

1.  专用分割器：`LanguageTextSplitter`用于代码（语法感知）和`MarkdownHeaderTextSplitter`（标题级别，添加元数据）。

1.  环境：库、键、依赖项、导入——用于稳健处理。

1.  `RecursiveCharacterTextSplitter`保留语义并适应结构；调整大小/重叠/深度。

1.  “字母”演示突出了差异：切片与语义感知分割。

1.  字符与标记取决于模型限制、语义需求和文本性质。

1.  通过 Markdown 标题分割保留逻辑结构。

1.  最佳实践：保持意义，调整重叠（避免冗余），丰富块元数据。

## 实际任务

1.

```py
def  split_by_char(text, chunk_size):
  """
 Split text into chunks of a fixed size.

 Args:
 - text (str): The text to split.
 - chunk_size (int): The size of each chunk.

 Returns:
 - list[str]: List of string chunks.
 """
    chunks = []
    for start_index in range(0, len(text), chunk_size):
        chunks.append(text[start_index:start_index + chunk_size])
    return chunks

# Example usage
text = "This is a sample text for demonstration purposes."
chunk_size = 10

chunks = split_by_char(text, chunk_size)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}") 
```

2.

```py
def  split_by_char(text, chunk_size):
  """
 Split text into chunks of a fixed size.

 Args:
 - text (str): The text to split.
 - chunk_size (int): The size of each chunk.

 Returns:
 - list: List of text chunks.
 """
    chunks = []  # Initialize an empty list to store chunks
    for start_index in range(0, len(text), chunk_size):
        # Append a chunk (substring) starting at start_index
        chunks.append(text[start_index:start_index + chunk_size])
    return chunks

# Example usage
text = "This is a sample text for demonstration purposes."
chunk_size = 10

chunks = split_by_char(text, chunk_size)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}") 
```

3.

```py
class  TokenTextSplitter:
    def  __init__(self, chunk_size, chunk_overlap=0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def  split_text(self, text):
        tokens = text.split()  # Split text into tokens by spaces
        chunks = []
        start_index = 0

        while start_index < len(tokens):
            # Ensure end_index does not exceed total tokens length
            end_index = min(start_index + self.chunk_size, len(tokens))
            chunk = ' '.join(tokens[start_index:end_index])
            chunks.append(chunk)
            # Update start_index accounting for overlap
            start_index += self.chunk_size - self.chunk_overlap
            if self.chunk_overlap >= self.chunk_size:
                print("Warning: `chunk_overlap` should be less than `chunk_size` to avoid overlap issues.")
                break

        return chunks 
```

# 示例用法：

```py
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
markdown_text = """
# Header 1
This is some text under header 1.
## Header 2
This is some text under header 2.
### Header 3
This is some text under header 3.
"""

chunks = splitter.split_text(markdown_text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:
{chunk}
---") 
```

此实现执行以下操作：

+   在初始化期间，它按长度降序排列标题标记，以便较长的（更具体的）Markdown 标题首先匹配。这很重要，因为 Markdown 标题级别通过`#`字符的数量来区分，我们希望匹配最具体的标题。

+   它编译一个正则表达式，匹配行首指定的任何标题标记。

+   `split_text`方法遍历输入 Markdown 的每一行，检查标题匹配。当它找到一个标题时，它适当地开始或结束一个块。每个块包括其起始标题和所有后续行，直到下一个相同或更高优先级的标题。

4.

```py
def  recursive_split(text, max_chunk_size, separators):
    if not separators:  # Base case: no separators left
        return [text]

    if len(text) <= max_chunk_size:  # Already within size
        return [text]

    # Try to split with the first separator
    separator = separators[0]
    parts = text.split(separator)

    if len(parts) == 1:  # Separator not found, try next
        return recursive_split(text, max_chunk_size, separators[1:])

    chunks = []
    current_chunk = ""
    for part in parts:
        # If adding the part would exceed the limit and we already have content, store current and start new
        if len(current_chunk + part) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = part + separator
        else:
            current_chunk += part + separator

    # Recurse on the remaining text to ensure size constraints
    if current_chunk.strip():
        chunks.extend(recursive_split(current_chunk.strip(), max_chunk_size, separators))

    # Flatten nested lists from recursion
    flat_chunks = []
    for chunk in chunks:
        if isinstance(chunk, list):
            flat_chunks.extend(chunk)
        else:
            flat_chunks.append(chunk)

    return flat_chunks 
```

5. 要实现`MarkdownHeaderTextSplitter`，我们遵循以下步骤：

1.  初始化：存储标题模式及其名称/级别，以便在分割时使用。

1.  文本分割：解析输入 Markdown，通过给定的模式识别标题，并将其分割成块。每个块从标题开始，包括直到下一个相同或更高优先级标题的所有后续行。

```py
import  re

class  MarkdownHeaderTextSplitter:
    def  __init__(self, headers_to_split_on):
        # Sort headers by marker length (longer first) for correct matching
        self.headers_to_split_on = sorted(headers_to_split_on, key=lambda x: len(x[0]), reverse=True)
        self.header_regex = self._generate_header_regex()

    def  _generate_header_regex(self):
        # Build a regex matching any of the specified header markers
        header_patterns = [re.escape(header[0]) for header in self.headers_to_split_on]
        combined_pattern = '|'.join(header_patterns)
        return re.compile(r'(' + combined_pattern + r')\s*(.*)')

    def  split_text(self, markdown_text):
        chunks = []
        current_chunk = []
        lines = markdown_text.split('\n')

        for line in lines:
            # Check if the line starts with one of the header markers
            match = self.header_regex.match(line)
            if match:
                # If we already collected lines, store the previous chunk
                if current_chunk:
                    chunks.append('\n'.join(current_chunk).strip())
                    current_chunk = []
                current_chunk.append(line)
            else:
                current_chunk.append(line)

        # Append the last collected chunk if present
        if current_chunk:
            chunks.append('\n'.join(current_chunk).strip())

        return chunks 
```

# 示例用法：

```py
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
markdown_text = """
# Header 1
This is some text under header 1.
## Header 2
This is some text under header 2.
### Header 3
This is some text under header 3.
"""

chunks = splitter.split_text(markdown_text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:
{chunk}
---") 
```

此实现执行以下操作：

+   在初始化期间，它按长度降序排列标题标记，以便较长的（更具体的）Markdown 标题首先匹配。这很重要，因为 Markdown 标题级别通过`#`字符的数量来区分，我们希望匹配最具体的标题。

+   它编译一个正则表达式，匹配行首指定的任何标题标记。

+   `split_text` 方法遍历输入 Markdown 的每一行，检查是否存在标题匹配。当找到标题时，它适当地开始或结束一个块。每个块包括其起始标题以及所有后续行，直到下一个相同或更高优先级的标题。
