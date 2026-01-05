# 答案 1.1

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-1/Answers%201.1/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-1/Answers%201.1/)

## 理论

1.  集成 OpenAI API 的关键好处：生成自然文本、自动化支持、提高内容创作以及通过高级 AI 扩展应用程序功能——提升用户参与度和运营效率。

1.  获取和保障 API 密钥：在 OpenAI 平台上注册、选择一个计划，并在仪表板上获取您的密钥。将密钥存储在环境变量或密钥管理器中；永远不要将其提交到存储库——这可以防止未经授权的访问和潜在损失。

1.  `temperature`：控制生成文本的创造性和多样性。低值使响应更可预测；高值增加多样性。根据任务选择。

1.  密钥应存储在代码之外（环境变量或密钥管理器）以避免通过源代码和版本控制系统（VCS）泄露。

1.  模型选择影响质量、速度和成本。平衡模型能力和资源以适应您的应用程序需求。

1.  响应元数据（例如，`usage`中的令牌计数）有助于优化提示、管理成本并更有效地使用 API。

1.  交互式界面包括对话历史、输入小部件、发送按钮和用于显示响应的面板。它实时更新，随着答案的到来而更新。

1.  最佳实践：后处理（风格和语法）、个性化到用户上下文、收集反馈以及监控性能和支出。

1.  易犯错误：未经检查就过度信任模型输出。使用验证、自动和手动审查的混合、监控和微调。

1.  道德和隐私：遵守数据法规、关于 AI 角色的透明度、实施审查/纠正流程，并考虑社会影响。

## 实践

下面是 OpenAI API 的 Python 脚本进展——从基本请求到错误处理和 CLI。

### 任务 1：基本 API 请求

```py
from  openai  import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the future of AI?"}],
    max_tokens=100,
)

print(response.choices[0].message.content) 
```

### 任务 2：安全密钥处理

```py
import  os
from  openai  import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the future of AI?"}],
    max_tokens=100,
)

print(response.choices[0].message.content) 
```

### 任务 3：解释响应

```py
import  os
from  openai  import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the future of AI?"}],
    max_tokens=100,
)

print("Response:", response.choices[0].message.content.strip())
print("Model used:", response.model)
print("Finish reason:", response.choices[0].finish_reason) 
```

### 任务 4：错误处理

```py
import  os
from  openai  import OpenAI
from  openai  import APIConnectionError, RateLimitError, APIStatusError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the future of AI?"}],
        max_tokens=100,
    )
    print("Response:", response.choices[0].message.content.strip())
    print("Model used:", response.model)
    print("Finish reason:", response.choices[0].finish_reason)
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except APIConnectionError as e:
    print(f"Connection error: {e}")
except APIStatusError as e:
    print(f"API returned an error: {e}")
except Exception as e:
    print(f"Other error occurred: {e}") 
```

### 任务 5：无后处理的 CLI 聊天

```py
from  openai  import OpenAI
import  os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def  chat_with_openai():
    print("Starting chat with OpenAI. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_input}],
                max_tokens=100,
            )
            print("OpenAI:", response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_with_openai() 
```

### 任务 6：后处理

```py
from  openai  import OpenAI
import  os
from  textblob  import TextBlob

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def  post_process_response(response_text):
    blob = TextBlob(response_text)
    corrected_text = str(blob.correct())
    formatted_text = " ".join(corrected_text.split())
    return formatted_text

def  chat_with_openai():
    print("Starting chat with OpenAI. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_input}],
                max_tokens=100,
            )
            processed = post_process_response(response.choices[0].message.content)
            print("OpenAI:", processed)
        except Exception as e:
            print(f"Other error occurred: {e}")

if __name__ == "__main__":
    chat_with_openai() 
```

### 任务 7-8（想法）

+   为用户提供的主题生成后处理大纲，并输出一个项目符号列表。

+   将每个调用对文件的响应时间和令牌使用情况记录下来，以供后续分析和优化。
