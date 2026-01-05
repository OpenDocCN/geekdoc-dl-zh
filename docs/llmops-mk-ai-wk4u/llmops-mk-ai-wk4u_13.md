# 2.2 LangChain 文档加载器

> [原文链接](https://boramorka.github.io/LLM-Book/en/CHAPTER-2/2.2%20LangChain%20Document%20Loaders/)

对于由 LLM 驱动的数据应用和对话式界面，高效地加载数据、规范化数据并在不同来源中使用数据至关重要。在 LangChain 生态系统中，“加载器”是提取信息自网站、数据库和媒体文件并将其转换为具有内容和元数据的标准文档对象的组件。支持数十种格式（PDF、HTML、JSON 等）和来源——从公共的（YouTube、Twitter、Hacker News）到企业工具（Figma、Notion）。还有用于表格和服务数据的加载器（Airbyte、Stripe、Airtable 等），不仅支持非结构化数据的语义搜索和 QA，还支持严格结构化数据集。这种模块化允许您构建目标管道：有时只需加载和清理文本就足够了；其他时候您将自动创建嵌入，提取实体，聚合和总结。

我们从基本环境准备开始：安装依赖项，配置 API 密钥，并从`.env`文件中读取它们以安全访问外部数据。

```py
# Install required packages (they may already be present in your environment)
# !pip install langchain dotenv

import  os
from  dotenv  import load_dotenv, find_dotenv

# Load environment variables from .env
load_dotenv(find_dotenv())

# Fetch the OpenAI API key from the environment
openai_api_key = os.environ['OPENAI_API_KEY'] 
```

常见场景是处理 PDF 文件。以下示例展示了如何加载文档（例如，讲座记录），清理和分词文本，统计词频，并将清理后的版本保存以供后续分析；我们明确处理并记录空页，并且可以提供元数据进行抽查。

```py
from  langchain.document_loaders  import PyPDFLoader
import  re
from  collections  import Counter

# Initialize PDF loader with the path to your PDF document
# Replace with your own file path or use a URL-based loader for remote PDFs
pdf_loader = PyPDFLoader("path/to/your/document.pdf")  # Example: "lecture.pdf"

# Load the document pages
document_pages = pdf_loader.load()

# Clean and tokenize
def  clean_and_tokenize(text):
    # Remove non‑alphabetic characters and split into words
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return words

word_frequencies = Counter()

for page in document_pages:
    if page.page_content.strip():
        words = clean_and_tokenize(page.page_content)
        word_frequencies.update(words)
    else:
        print(f"Empty page found at index {document_pages.index(page)}")

print("Most frequent words:")
for word, freq in word_frequencies.most_common(10):
    print(f"{word}: {freq}")

# Inspect metadata of the first page
first_page_metadata = document_pages[0].metadata
print("\nFirst page metadata:")
print(first_page_metadata)

# Optionally save cleaned text to a file
with open("cleaned_lecture_series_lecture01.txt", "w") as text_file:
    for page in document_pages:
        if page.page_content.strip():
            cleaned_text = ' '.join(clean_and_tokenize(page.page_content))
            text_file.write(cleaned_text + "\n") 
```

视频同样重要。我们可以通过 LangChain 使用 Whisper 从 YouTube 获取音频，并立即开始分析：分割成句子，使用`TextBlob`评估情感（极性和主观性），可选地添加实体提取、关键词检测和摘要。

```py
from  langchain.document_loaders.generic  import GenericLoader
from  langchain.document_loaders.parsers  import OpenAIWhisperParser
from  langchain.document_loaders.blob_loaders.youtube_audio  import YoutubeAudioLoader
from  nltk.tokenize  import sent_tokenize
from  textblob  import TextBlob
import  os

import  nltk
nltk.download('punkt')

video_url = "https://www.youtube.com/watch?v=example_video_id"
audio_save_directory = "docs/youtube/"
os.makedirs(audio_save_directory, exist_ok=True)

youtube_loader = GenericLoader(
    YoutubeAudioLoader([video_url], audio_save_directory),
    OpenAIWhisperParser()
)

youtube_documents = youtube_loader.load()

transcribed_text = youtube_documents[0].page_content[:500]
print(transcribed_text)

sentences = sent_tokenize(transcribed_text)

print("\nFirst 5 sentences:")
for sentence in sentences[:5]:
    print(sentence)

sentiment = TextBlob(transcribed_text).sentiment
print("\nSentiment:")
print(f"Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")

# Polarity: float in [-1.0, 1.0] (negative to positive)
# Subjectivity: float in [0.0, 1.0] (objective to subjective) 
```

对于网络内容，我们通过 URL 加载页面，清理 HTML，提取链接和标题，然后进行简单的总结：句子分词，停用词过滤，频率分析，以及简要摘要。

```py
from  langchain.document_loaders  import WebBaseLoader
from  bs4  import BeautifulSoup
from  nltk.tokenize  import sent_tokenize
from  nltk.corpus  import stopwords
from  nltk.probability  import FreqDist
from  nltk  import download
download('punkt')
download('stopwords')

web_loader = WebBaseLoader("https://example.com/path/to/document")
web_documents = web_loader.load()

soup = BeautifulSoup(web_documents[0].page_content, 'html.parser')

for script_or_style in soup(["script", "style"]):
    script_or_style.decompose()

clean_text = ' '.join(soup.stripped_strings)
print(clean_text[:500])

links = [(a.text, a['href']) for a in soup.find_all('a', href=True)]
print("\nExtracted links:")
for text, href in links[:5]:
    print(f"{text}: {href}")

headings = [h1.text for h1 in soup.find_all('h1')]
print("\nHeadings found:")
for heading in headings:
    print(heading)

sentences = sent_tokenize(clean_text)
stop_words = set(stopwords.words("english"))
filtered_sentences = [' '.join([w for w in s.split() if w.lower() not in stop_words]) for s in sentences]

word_freq = FreqDist(w.lower() for s in filtered_sentences for w in s.split())

print("\nMost frequent words:")
for word, frequency in word_freq.most_common(5):
    print(f"{word}: {frequency}")

print("\nContent summary:")
for sentence in sentences[:5]:
    print(sentence) 
```

结构化 Notion 导出也易于处理：加载 Markdown 文件，转换为 HTML 以便方便解析，提取标题和链接，将元数据和解析内容放入 DataFrame 中，过滤（例如，通过标题中的关键词），如果存在，计算类别细分。

```py
from  langchain.document_loaders  import NotionDirectoryLoader
import  markdown
from  bs4  import BeautifulSoup
import  pandas  as  pd

notion_directory = "docs/Notion_DB"
notion_loader = NotionDirectoryLoader(notion_directory)
notion_documents = notion_loader.load()

print(notion_documents[0].page_content[:200])
print(notion_documents[0].metadata)

html_content = [markdown.markdown(doc.page_content) for doc in notion_documents]

parsed_data = []
for content in html_content:
    soup = BeautifulSoup(content, 'html.parser')
    headings = [heading.text for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
    links = [(a.text, a['href']) for a in soup.find_all('a', href=True)]
    parsed_data.append({'headings': headings, 'links': links})

df = pd.DataFrame({
    'metadata': [doc.metadata for doc in notion_documents],
    'parsed_content': parsed_data
})

keyword = 'Project'
filtered_docs = df[df['metadata'].apply(lambda x: keyword.lower() in x.get('title', '').lower())]

print("\nDocuments with the keyword in the title:")
print(filtered_docs)

# Example category summary (if categories exist in metadata)
if 'category' in df['metadata'].iloc[0]:
    category_counts = df['metadata'].apply(lambda x: x['category']).value_counts()
    print("\nDocuments by category:")
    print(category_counts) 
```

在使用加载器时，请注意外部 API 成本（例如，Whisper）并优化调用；在加载后立即规范化数据（清理、分块等）；如果缺少来源——向 LangChain 开源贡献您自己的加载器。方便查阅文档以获取指导：LangChain（https://github.com/LangChain/langchain）和 OpenAI Whisper（https://github.com/openai/whisper）。这种做法为更高级的数据处理和集成到 LLM 应用奠定了基础。

## 理论问题

1.  LangChain 中的文档加载器是什么？它们扮演什么角色？

1.  非结构化数据的加载器与结构化数据的加载器有何不同？

1.  如何为加载器（包、API 密钥、`.env` 文件）准备环境？

1.  `PyPDFLoader` 如何工作，PDF 预处理提供了什么？

1.  处理 PDF 时，为什么需要清理和分词文本？

1.  如何通过 LangChain 使用 Whisper 转录 YouTube 视频？

1.  如何将句子分词和情感分析应用于转录文本？

1.  如何使用 `WebBaseLoader` 加载和处理网页内容？

1.  如何通过 URL 提取和总结页面内容？

1.  `NotionDirectoryLoader` 如何帮助分析 Notion 导出？

1.  使用加载器时，哪些实践很重要（成本意识、预处理）？

1.  为什么以及如何向 LangChain 添加新的加载器？

## 实践任务

1.  修改 PDF 分析以忽略停用词（`nltk.stopwords`）；打印前五个最频繁的非停用词。

1.  编写一个函数，用于转录 YouTube URL（Whisper）并返回前 100 个单词；包括错误处理。

1.  创建一个脚本：通过 URL 加载页面，去除 HTML 标签，并打印干净的文本（使用 BeautifulSoup）。

1.  对于 Notion 导出目录：将 Markdown 转换为 HTML，提取并打印所有链接（文本 + href）。

1.  使用 `TextBlob` 情感扩展 YouTube 转录：打印极性和粗略标签（正面/中性/负面）。

1.  从 Notion 文档构建 DataFrame，添加“单词计数”列，并打印三个最长文档的标题。

1.  对于给定的 URL - 加载页面，提取主要文本，并打印一个简单的摘要（第一句和最后一句）。
