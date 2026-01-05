# 1.1 简介

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-1/1.1%20Introduction/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-1/1.1%20Introduction/)

本章重点介绍将 OpenAI API 集成到您的服务中的实际操作，着重于使用 GPT 模型生成文本响应。我们将简要介绍从安装和安全的设置到您的第一个请求、解释响应以及将结果嵌入应用程序的路径。这些材料面向需要快速且可靠地将模型连接到产品的机器学习工程师、数据科学家、软件开发人员和相关专业人士。

OpenAI 通过 API 提供对一系列语言模型（包括生成预训练转换器 GPT）的访问。这些模型理解并生成类似人类的文本，使它们成为从自动化客户支持到内容生成等任务的有力工具。首先，安装当前客户端版本：

```py
pip install --upgrade openai 
```

接下来，你需要一个 API 密钥，在 OpenAI（https://openai.com/）注册并选择合适的定价计划后获得。该密钥是唯一的，用于签名请求，必须严格保密：存储在环境变量中，在本地开发中存储在 `.env` 文件中；在生产中，使用密钥管理器。有了这个最小设置，你可以发送一个简单的文本生成请求并将答案打印到控制台：

```py
from  openai  import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is artificial intelligence?"}],
    max_tokens=100,
)
print(response.choices[0].message.content) 
```

为了获得可预测的结果，重要的是要记住请求是如何形成的：你选择一个模型，制作一个提示（一个问题或指令），并设置生成参数。例如，`temperature` 控制创造性和随机性：它越高，答案就越多样化。客户端库从环境变量中读取 API 密钥；在正确配置的情况下，你只需组装消息列表并指定模型——SDK 处理其余部分。

API 响应包含生成的文本和有用的元数据。结构上包括一个 `choices` 字段（一个或多个答案变体）和 `usage`（令牌统计）以帮助估计成本并优化请求：

```py
{
  "id":  "chatcmpl-XYZ123",
  "object":  "chat.completion",
  "created":  1613679373,
  "model":  "gpt-4o-mini",
  "choices":  [
  {
  "index":  0,
  "message":  {
  "role":  "assistant",
  "content":  "The generated text response to your prompt."
  },
  "logprobs":  null,
  "finish_reason":  "stop"
  }
  ],
  "usage":  {
  "prompt_tokens":  15,
  "completion_tokens":  25,
  "total_tokens":  40
  }
} 
```

> **注意：** 响应包含 `choices[].message.content` 中的生成文本和 `usage` 统计数据，以帮助估计成本并优化请求。

在集成时，内置错误处理：网络不可靠，限制有限，请求参数可能无效。一个简单的 `try/except` 框架可以帮助你正确响应连接问题、配额超额和 API 状态错误，而不会使你的应用程序崩溃：

```py
import  os
from  openai  import OpenAI
from  openai  import APIConnectionError, RateLimitError, APIStatusError

# The client reads OPENAI_API_KEY from the environment by default
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "This is a test prompt"}],
        max_tokens=50,
    )
    print(response.choices[0].message.content)
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except APIConnectionError as e:
    print(f"Connection error: {e}")
except APIStatusError as e:
    print(f"API returned an error: {e}")
except Exception as e:
    print(f"Other error: {e}") 
```

除了错误处理之外，使用 `usage` 元数据和其它响应字段来监控成本、时间和有效性，这样你可以调整提示、限制长度、选择成本效益高的模型，并保持支出在控制之下。

在应用场景中，生成通常嵌入在对话界面中。以下是一个使用 Panel 构建的交互式客户端的简要示例：用户输入查询，系统处理它，并显示答案。代码说明了更新历史记录和布局 UI 元素，这些元素很容易根据您的需求进行修改：

```py
import  panel  as  pn  # For building the GUI

# Conversation history and UI elements
conversation_history = []
input_widget = pn.widgets.TextInput(placeholder='Enter your query...')
submit_button = pn.widgets.Button(name="Send")
panels = []

def  update_conversation(event):
  """
 Handles user input, calls the request processing function, and updates the conversation output.
 """
    user_query = input_widget.value
    if user_query:  # Ensure the string is not empty
        response, conversation_history = process_user_query(user_query, conversation_history)
        panels.append(pn.Row('User:', pn.pane.Markdown(user_query)))
        panels.append(pn.Row('Assistant:', pn.pane.Markdown(response, background='#F6F6F6')))
        input_widget.value = ''  # Clear the input field

# Bind the handler to the button
submit_button.on_click(update_conversation)

# Interface layout
conversation_interface = pn.Column(
    input_widget,
    submit_button,
    pn.panel(update_conversation, loading_indicator=True),
)

# Display the interface
conversation_interface.servable() 
```

小贴士：使用“助手正在输入...”指示器和其他反馈信号来改善用户体验，使对话感觉生动。从那里，关键在于您如何使用模型的答案。在聊天机器人中，您可以直接显示回复，注意格式和相关性；对于生成文章和报告，后处理有助于——格式化、模板化和将多个答案组合成连贯的文本；对于 Web 应用中的动态内容，验证相关性和一致性，并计划定期更新。

好的做法是添加后处理（语法和风格检查、与品牌声音保持一致），个性化（尊重上下文、偏好和用户历史），收集反馈以改进提示和参数，以及监控/分析：响应时间、参与度、token 使用和其他有助于您负责任地优化系统的指标。为了性能，考虑缓存频繁查询、批处理，并为任务和预算选择适当大小的模型。不要盲目相信模型输出：验证准确性和适宜性，并添加验证和过滤器。

要深入了解，请学习官方 OpenAI 文档，关注更新，并参与专业社区。这些材料为快速集成奠定了基础，并为智能文本交互的高级场景打开了大门。

## 理论问题

1.  集成 OpenAI API 对机器学习工程师、数据科学家和开发者有哪些主要好处？

1.  描述如何获取 OpenAI API 密钥，并解释为什么保护它很重要。

1.  `temperature`的作用是什么，它如何影响生成结果？

1.  为什么 API 密钥应该存储在环境变量或秘密管理器中，而不是直接在代码中？

1.  为什么模型选择对质量、速度和成本至关重要？

1.  响应元数据如何帮助优化请求和管理 token 支出？

1.  列出创建简单对话界面及其关键组件的步骤。

1.  哪些集成最佳实践适用于聊天机器人、内容生成和动态内容？

1.  列举在使用 API 时常见的陷阱以及预防方法。

1.  如何确保道德标准和保护用户隐私？

## 实践任务

1.  编写一个 Python 脚本，使用 OpenAI API 回答“人工智能的未来是什么？”的问题。将答案限制在 100 个 token 以内。

1.  将任务 1 中的脚本修改为从环境变量中读取 API 密钥，而不是将其硬编码。

1.  将任务 2 中的脚本扩展，打印出答案文本、模型名称、token 计数以及生成停止的原因。

1.  使用`try/except`为任务 3 中的脚本添加错误处理（例如处理速率限制、无效请求等）。

1.  创建一个简单的命令行界面（CLI），它可以实时发送提示并流式传输答案，并具有错误处理功能。

1.  为任务 5 中的 CLI 添加答案后处理：去除多余的空白字符，进行基本的语法纠正（例如使用`textblob`）或进行自己的格式化。

1.  开发一个脚本，对于用户提供的主题，生成一个发布计划并以项目符号列表的形式输出。

1.  在任何脚本中，添加响应时间和令牌使用的日志记录，并将这些指标存储起来以供后续分析和优化。
