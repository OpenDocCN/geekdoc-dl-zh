# 答案 2.2

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-2/Answers%202.2/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-2/Answers%202.2/)

## 理论

1.  文档加载器是 LangChain 组件，它们从各种来源和格式中提取数据，将它们转换为统一的文档对象（内容 + 元数据）。它们是使 PDF、HTML、JSON 等文件进入 LLM 应用的基础。

1.  无结构加载器针对 YouTube、Twitter、Figma、Notion 等资源，而结构化加载器针对表格/服务源（Airbyte、Stripe、Airtable）等，这使它们能够实现语义搜索和表格问答。

1.  环境准备包括安装包、配置 API 密钥（例如，OpenAI）和从`.env`文件中加载环境变量。

1.  `PyPDFLoader`通过路径加载 PDF 文件，让您提取、清洗和分词文本以进行下游分析（词频、处理空白页等）。

1.  清洗和分词——去除噪声、转换为小写和分割成单词——是基本规范化步骤，用于准确计数和进一步处理。

1.  对于 YouTube：使用`GenericLoader` + `YoutubeAudioLoader` + `OpenAIWhisperParser`下载音频并通过 Whisper 进行转录。

1.  句子分词允许更细致的分析。情感分析（例如，使用`TextBlob`）提供极性/主观性信息。

1.  对于网页内容，使用`WebBaseLoader(url)`加上`BeautifulSoup`来去除标记并提取链接/标题和其他结构。

1.  清洗后，您可以根据停用词过滤和词频提取关键信息并构建简洁的摘要。

1.  `NotionDirectoryLoader`读取导出的 Notion 数据（Markdown），将其转换为 HTML 以进行解析，提取结构（标题、链接），并将其存储在 DataFrame 中以进行过滤和总结。

1.  实用技巧：优化 API 调用以控制成本，在加载后立即进行预处理，并考虑为 LangChain 贡献新的加载器。

1.  贡献新的加载器可以扩展生态系统，拓宽支持的来源，并建立贡献者的专业知识。

## 实践

1.

```py
from  langchain.document_loaders  import PyPDFLoader
import  re
from  collections  import Counter
import  nltk

nltk.download('stopwords')
from  nltk.corpus  import stopwords

stop_words = set(stopwords.words('english'))

pdf_loader = PyPDFLoader("path/to/your/document.pdf")
document_pages = pdf_loader.load()

def  clean_tokenize_and_remove_stopwords(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return [w for w in words if w not in stop_words]

word_frequencies = Counter()
for page in document_pages:
    if page.page_content.strip():
        words = clean_tokenize_and_remove_stopwords(page.page_content)
        word_frequencies.update(words)

print("Top‑5 most frequent non‑stop words:")
for word, freq in word_frequencies.most_common(5):
    print(f"{word}: {freq}") 
```

2.

```py
from  langchain.document_loaders.generic  import GenericLoader
from  langchain.document_loaders.parsers  import OpenAIWhisperParser
from  langchain.document_loaders.blob_loaders.youtube_audio  import YoutubeAudioLoader
import  nltk

nltk.download('punkt')
from  nltk.tokenize  import word_tokenize

def  transcribe_youtube_video(video_url):
    audio_save_directory = "temp_audio/"
    try:
        youtube_loader = GenericLoader(
            YoutubeAudioLoader([video_url], audio_save_directory),
            OpenAIWhisperParser()
        )
        youtube_documents = youtube_loader.load()
        transcribed_text = youtube_documents[0].page_content
        first_100_words = ' '.join(word_tokenize(transcribed_text)[:100])
        return first_100_words
    except Exception as e:
        print(f"Error: {e}")
        return None

video_url = "https://www.youtube.com/watch?v=example_video_id"
print(transcribe_youtube_video(video_url)) 
```

3.

```py
import  requests
from  bs4  import BeautifulSoup

def  load_and_clean_web_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        clean_text = soup.get_text(separator=' ', strip=True)
        print(clean_text)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Parsing error: {e}")

url = "https://example.com"
load_and_clean_web_content(url) 
```

4.

```py
import  markdown
from  bs4  import BeautifulSoup
import  os

def  convert_md_to_html_and_extract_links(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as md_file:
                md_content = md_file.read()
            html_content = markdown.markdown(md_content)
            soup = BeautifulSoup(html_content, 'html.parser')
            links = soup.find_all('a', href=True)
            print(f"Links in {filename}:")
            for link in links:
                print(f"Text: {link.text}, Href: {link['href']}")
            print("------" * 10)

directory_path = "path/to/your/notion/data"
convert_md_to_html_and_extract_links(directory_path) 
```

5.

```py
from  textblob  import TextBlob
from  langchain.document_loaders.generic  import GenericLoader
from  langchain.document_loaders.parsers  import OpenAIWhisperParser
from  langchain.document_loaders.blob_loaders.youtube_audio  import YoutubeAudioLoader

def  transcribe_and_analyze_sentiment(video_url):
    # Directory to temporarily store downloaded audio
    audio_save_directory = "temp_audio/"

    try:
        # Initialize the loader with YouTube Audio Loader and Whisper Parser
        youtube_loader = GenericLoader(
            YoutubeAudioLoader([video_url], audio_save_directory),
            OpenAIWhisperParser()
        )

        # Load the document (transcription)
        youtube_documents = youtube_loader.load()

        # Access the transcribed content
        transcribed_text = youtube_documents[0].page_content

        # Analyze sentiment with TextBlob
        blob = TextBlob(transcribed_text)
        polarity = blob.sentiment.polarity

        # Map polarity to a simple label
        if polarity > 0.1:
            sentiment_label = "positive"
        elif polarity < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        print(f"Polarity: {polarity:.3f}")
        print(f"Sentiment: {sentiment_label}")

        return transcribed_text, polarity, sentiment_label

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

# Example usage
video_url = "https://www.youtube.com/watch?v=example_video_id"
text, polarity, sentiment = transcribe_and_analyze_sentiment(video_url) 
```

6.

```py
import  pandas  as  pd
import  os
import  markdown
from  bs4  import BeautifulSoup

def  create_notion_dataframe_with_word_count(directory_path):
    documents_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as md_file:
                md_content = md_file.read()
            html_content = markdown.markdown(md_content)
            soup = BeautifulSoup(html_content, 'html.parser')
            clean_text = soup.get_text()
            word_count = len(clean_text.split())
            title = filename.replace('.md', '')
            for line in md_content.split('\n'):
                if line.strip().startswith('#'):
                    title = line.strip('#').strip()
                    break
            documents_data.append({
                'filename': filename,
                'title': title,
                'word_count': word_count,
                'content': clean_text[:100] + '...'
            })
    df = pd.DataFrame(documents_data)
    df_sorted = df.sort_values('word_count', ascending=False)
    print("Top 3 longest docs:")
    for _, row in df_sorted.head(3).iterrows():
        print(f"{row['title']} ({row['word_count']} words)")
    return df_sorted

directory_path = "path/to/your/notion/data"
df = create_notion_dataframe_with_word_count(directory_path)
print(df.head()) 
```

7.

```py
import  requests
from  bs4  import BeautifulSoup
import  re

def  load_and_summarize_web_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        clean_text = soup.get_text(separator=' ', strip=True)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if s.strip()]
        if len(sentences) >= 2:
            summary = f"First: {sentences[0]}.\nLast: {sentences[-1]}."
        elif len(sentences) == 1:
            summary = f"Single sentence: {sentences[0]}."
        else:
            summary = "No sentences extracted."
        print("Simple summary:")
        print(summary)
        return summary
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except Exception as e:
        print(f"Parsing error: {e}")
        return None

url = "https://example.com"
summary = load_and_summarize_web_content(url) 
```
