# 工具使用与函数调用

使用工具的语言模型是扩展其能力的一种自然方式，尤其是在外部工具包含所需信息或需要与复杂网络系统交互的智能体中，这种方式尤其适用于高精度任务。这些策略可以被视为工具使用的一般类别。一个 AI 模型通过输出特殊标记来触发特定端点，从而使用任何外部工具。这些工具可以是高度具体的，例如返回特定地点天气的功能，也可以是代码解释器或搜索引擎，它们是复杂行为的基本构建块。本章概述了现代语言模型中工具使用的起源、其基本原理和格式，以及当前在主要模型中有效利用工具的权衡。

“工具使用”这一术语的确切起源尚不明确，但这一想法的起源远早于 ChatGPT 时代，那时强化学习与人类反馈（RLHF）得到了广泛的应用。大约在 2015 年，早期例子试图构建现代语言模型之前的系统，如神经程序员解释器（NPI）[[290]](ch021.xhtml#ref-reed2015neural)，“一种循环和组合的神经网络，它学会表示和执行程序。”随着语言模型的普及，许多子领域都在使用外部能力的集成来提升性能。为了获取除了权重之外的信息，许多人使用了检索增强生成[[291]](ch021.xhtml#ref-lewis2020retrieval)或网络浏览[[4]](ch021.xhtml#ref-nakano2021webgpt)。不久之后，其他人开始探索与程序[[292]](ch021.xhtml#ref-gao2023pal)或工具[[293]](ch021.xhtml#ref-parisi2022talm)集成的语言模型。

随着该领域的成熟，这些模型在底层语言模型的大幅改进之外，还获得了更复杂的能力。例如，ToolFormer 可以使用“计算器、问答系统、两个不同的搜索引擎、翻译系统和日历”[[294]](ch021.xhtml#ref-schick2023toolformerlanguagemodelsteach)。不久之后，Gorilla 被训练使用 1645 个 API（来自 PyTorch Hub、TensorFlow Hub v2 和 HuggingFace），其评估 APIBench 成为流行伯克利函数调用排行榜[[295]](ch021.xhtml#ref-patil2023gorilla)的基础。自这些早期模型以来，被调用的动作种类已经大幅增加。

工具使用模型现在与常规语言模型交互深度融合。模型上下文协议（MCP）作为一种常见的格式出现，用于将语言模型连接到外部数据源（或工具）[[296]](ch021.xhtml#ref-anthropic_mcp_2024)。随着模型的增强和格式的改进，工具使用语言模型被用于许多场景，包括在流行的应用程序如 Microsoft Office 或 Google Workspace 中的生产力副驾驶，科学领域 [[297]](ch021.xhtml#ref-bran2023chemcrow)，医学领域 [[298]](ch021.xhtml#ref-li2024mmedagent)，编码代理 [[299]](ch021.xhtml#ref-zhang2024codeagent) 如 Claude Code 或 Cursor，与数据库的集成，以及许多其他自主工作流程。

## 生成过程中的工具调用交织

函数调用代理呈现数据的方式与其他后训练阶段非常相似。新增的内容是系统提示中的内容，指导模型它有哪些可用的工具。以下是一个带有系统提示和可用工具的 JSON 格式示例格式化数据点：

```py
<system>
You are a function-calling AI model. You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.
</system>

<functions>
[
  {
    "name": "get_id",
    "description": "Fetches the ID of a movie based on the given search query from the RapidAPI similar movies service.",
    "parameters": {
      "q": {
        "description": "The search string for the movie title.",
        "type": "str",
        "default": "titanic"
      }
    }
  },
  {
    "name": "search_torrents",
    "description": "Search for torrents based on given keywords using the RapidAPI service.",
    "parameters": {
      "keywords": {
        "description": "Keywords to search for torrents.",
        "type": "str",
        "default": "Meg 2 The Trench"
      },
      "quantity": {
        "description": "Number of torrent results to return. Maximum value is 40.",
        "type": "int",
        "default": 40
      },
      "page": {
        "description": "Page number for paginated results. Defaults to 1.",
        "type": "int",
        "default": 1
      }
    }
  },
  {
    "name": "basic_info",
    "description": "Fetches detailed information about a cast member such as name, profession, birth and death year, bio, poster, and best titles.",
    "parameters": {
      "peopleid": {
        "description": "The ID of the cast member whose details are to be fetched.",
        "type": "str",
        "default": "nm0000375"
      }
    }
  }
]
</functions>

<user>
...
</user>
```

当语言模型生成时，如果遵循上述示例，它将生成`search_torrents("Star Wars")`这样的标记来搜索《星球大战》。这通常编码在特殊的格式化标记中，然后下一个插入序列的标记将包含工具输出。通过这种方式，模型可以学会完成比许多简单独立模型更具挑战性的任务。

工具使用的一种流行形式是代码执行，允许模型对复杂的逻辑或数学问题获得精确答案。例如，在语言模型执行期间，代码执行可以发生在推理模型的思考标记中。与函数调用一样，首先是对要执行的代码（由模型生成）的标签，然后是输出标签。

```py
<|user|>
What is the 50th number in a fibonacci sequence?</s>
<|assistant|>
<think>
Okay, I will compute the 50-th Fibonacci number with a simple loop, then return the result.

<code>
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

fib(50)
</code>

<output>
12586269025
</output>
</think>
<answer>
The 50-th Fibonacci number is 12 586 269 025.
</answer>
```

## 多步工具推理

OpenAI 的 o3 模型在多步工具使用如何与语言模型集成方面迈出了实质性的飞跃。这种行为与社区中较早的研究趋势相关。例如，ReAct [[300]](ch021.xhtml#ref-yao2023react) 展示了如何将动作和推理交织到一个模型生成中：

> 在本文中，我们探讨了以交织方式使用 LLM 生成推理痕迹和特定任务动作，从而在两者之间实现更大的协同作用：推理痕迹帮助模型诱导、跟踪和更新行动计划，以及处理异常，而动作则允许它与外部来源（如知识库或环境）交互并收集更多信息。

随着工具使用能力的巩固和推理模型的起飞，多轮工具使用已成为一个令人兴奋的研究领域 [[286]](ch021.xhtml#ref-wang2025ragenunderstandingselfevolutionllm)。

## 模型上下文协议（MCP）

模型上下文协议（MCP）是一种连接语言模型到外部数据源和信息系统的标准 [[296]](ch021.xhtml#ref-anthropic_mcp_2024)。MCP 并非关注每个外部系统特定的工具调用格式，而是通过标准化协议使模型能够访问丰富的上下文信息。

MCP 是本章工具使用内容上的简单补充——它是如何以可预测的 JSON 模式将上下文（数据+操作）传递给语言模型的。与模型交互的 MCP 服务器具有核心原语：资源（只读数据块）、提示（模板化消息/工作流程）和工具（模型可以调用的函数）。因此，MCP 架构可以总结如下：

+   MCP 服务器封装特定的数据源或能力。

+   MCP 客户端（例如，Claude 桌面版，IDE 插件）聚合一个或多个服务器。

+   主机，例如 Claude 或 ChatGPT 应用，提供用户/LLM 界面；切换模型供应商或后端工具仅意味着交换中间的客户端。

MCP 使工具使用模型的开发者能够使用相同的架构将他们的服务器或客户端连接到不同的模型，同时模型也有一个可预测的格式，可以用来集成外部组件。这些共同为工具使用模型在实际应用领域提供了一个更加可预测的开发环境。

## 实现

在实现工具使用模型时，有多个格式化和遮蔽决策：

+   **Python 与 JSON 格式**：在本章中，我们包括了将工具使用格式化为 JSON 数据结构和 Python 代码的示例。模型倾向于选择一种结构，而行业中的不同提供商使用不同的格式。

+   **遮蔽工具输出**：在训练工具使用模型时，一个重要的细节是工具输出中的标记被从模型的训练损失中遮蔽。这确保了模型不会学习预测它在使用中直接生成的系统之外的输出（类似于其他训练后阶段的提示遮蔽）。

+   **多轮格式化工具调用**：在实现工具调用模型时，通常会将数据加载格式添加更多结构。训练后数据集的标准做法是用户和助手（以及通常的系统消息）之间的消息列表交替。工具使用的整体结构相同，但模型的轮次被分割成由每个工具调用分隔的内容子部分。以下是一个示例。

```py
[](#cb3-1)messages = 
[{
[](#cb3-3)"content": "You are a function calling AI model. You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.",
[](#cb3-4)"function_calls": null,
[](#cb3-5)"functions": "[{\"name\": \"live_giveaways_by_type\", \"description\": \"Retrieve live giveaways from the GamerPower API based on the specified type.\", \"parameters\": {\"type\": {\"description\": \"The type of giveaways to retrieve (e.g., game, loot, beta).\", \"type\": \"str\", \"default\": \"game\"}}}]",
[](#cb3-6)"role": "system"
[](#cb3-7)},
[](#cb3-8){
[](#cb3-9)"content": "Where can I find live giveaways for beta access and games?",
[](#cb3-10)"function_calls": null,
[](#cb3-11)"functions": null,
[](#cb3-12)"role": "user"
[](#cb3-13)},
[](#cb3-14){
[](#cb3-15)"content": null,
[](#cb3-16)"function_calls": "live_giveaways_by_type(type='beta')\nlive_giveaways_by_type(type='game')",
[](#cb3-17)"functions": null,
[](#cb3-18)"role": "assistant"
[](#cb3-19)}
[](#cb3-20)]
```

+   **标记化和消息格式细节**：在 OpenAI 消息格式中，工具调用通常通过聊天模板（控制发送给模型的格式代码）进行标记化，将结构化的 JSON 表示转换为原始标记流。这个过程在不同模型架构中有所不同——一些使用特殊标记来界定工具调用，而另一些则在标记流本身中保持结构化格式。[聊天模板游乐场](https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Qwen/Qwen3-8B)提供了一个交互式环境，可以探索不同模型如何将消息格式转换为标记流。

+   **推理标记连续性**：随着推理模型的出现，它们在答案之前有独立的“推理”标记流，对于如何在循环中使用工具处理这些标记存在不同的实现。一些模型在单个回合的工具调用步骤之间保留推理标记，以保持多个工具调用的上下文。然而，这些标记通常在回合之间被清除，以最小化服务成本。

+   **不同提供商的 API 格式**（截至 2025 年 7 月）：不同的提供商使用概念上相似但技术上不同的格式。OpenAI 使用具有唯一 ID 的 `tool_calls` 数组，Anthropic 采用详细的 `input_schema` 规范，并使用 `<thinking>` 标签，而 Gemini 提供函数调用模式（AUTO/ANY/NONE）。当通过 API 使用这些模型时，可用的工具以 JSON 格式定义，然后模型响应中的工具输出存储在标准“生成的标记”之外的一个单独字段中。例如，开源的 vLLM 推理代码库实现了广泛的解析逻辑，支持多种工具调用模式和特定于模型的解析器，为底层实现考虑提供了见解 [[301]](ch021.xhtml#ref-kwon2023efficient)。
