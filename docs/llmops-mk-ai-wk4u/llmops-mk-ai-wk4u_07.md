# 1.4 高级机器推理：策略

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-1/1.4%20Advanced%20Machine%20Reasoning/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-1/1.4%20Advanced%20Machine%20Reasoning/)

高级机器推理汇集了一系列实践，帮助语言模型更可靠、更透明地解决复杂任务。思维链（CoT）鼓励逐步解决方案，将问题分解为逻辑阶段。这种方法提高了准确性，并使推理可审计：用户可以看到模型如何得出答案，这对于多约束任务、比较分析和计算特别有帮助。在教育中，CoT 模仿导师引导你通过每个步骤，而不是给出最终答案。在客户支持中，它有助于拆解复杂请求：澄清细节、检查假设、纠正误解并提供正确结论。

与 CoT（思维链）并行，团队通常会使用内心独白，其中中间推理被隐藏，只展示结果（或最小逻辑片段）。当暴露内部步骤可能损害学习（避免“剧透”）、涉及敏感信息或额外细节会降低用户体验时，这种方法是合适的。

要使示例可重复，首先准备环境和 API 客户端。

```py
# Import libraries and load keys
import  os
from  openai  import OpenAI
from  dotenv  import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
client = OpenAI() 
```

```py
def  get_response_for_queries(query_prompts,
                             model_name="gpt-4o-mini",
                             response_temperature=0,
                             max_response_tokens=500):
  """
 Returns the model response based on a list of messages (system/user...).
 """
    model_response = client.chat.completions.create(
        model=model_name,
        messages=query_prompts,
        temperature=response_temperature,
        max_tokens=max_response_tokens,
    )
    return model_response.choices[0].message.content 
```

接下来，我们将为请求设置包装器，并转向 CoT 提示，其中推理被结构化为特殊分隔符下的步骤。系统消息描述分析阶段，用户输入被包装在分隔符中，简化了解析和后续处理。

```py
step_delimiter = "####"

system_prompt = f"""
Follow the steps, separating them with the '{step_delimiter}' marker.

Step 1:{step_delimiter} Check whether the question is about a specific product (not a category).

Step 2:{step_delimiter} If yes, match it to the product list (brand, specs, price).

[Insert your product list here]

Step 3:{step_delimiter} Identify the user’s assumptions (comparisons/specifications).

Step 4:{step_delimiter} Verify those assumptions against the product data.

Step 5:{step_delimiter} Correct inaccuracies using only the list and respond politely.
"""

example_query_1 = "How does the BlueWave Chromebook compare to the TechPro Desktop in terms of cost?"
example_query_2 = "Are televisions available for sale?"

query_prompts_1 = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': f"{step_delimiter}{example_query_1}{step_delimiter}"},
]

query_prompts_2 = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': f"{step_delimiter}{example_query_2}{step_delimiter}"},
] 
```

```py
response_to_query_1 = get_response_for_queries(query_prompts_1)
print(response_to_query_1)

response_to_query_2 = get_response_for_queries(query_prompts_2)
print(response_to_query_2) 
```

要比较方法，首先打印出包含中间 CoT 步骤的完整答案，然后应用只向用户展示最终部分的内心独白变体。如果模型返回由`step_delimiter`分隔的步骤文本，你可以只保留最终部分——在不需要“内部工作”的界面中保持简洁。

```py
try:
    final_response = response_to_query_2.split(step_delimiter)[-1].strip()
except Exception:
    final_response = "Sorry, there was a problem. Please try another question."

print(final_response) 
```

结果是同一解决方案的两种模式：一种是详细的，显示了可见的步骤链，另一种是简洁的，只显示结果。清晰的提示设计在这两种情况下都有帮助；根据观察到的行为不断改进你的提示。当 UI 清晰度很重要且不希望有额外细节时，优先选择内心独白进行显示，同时仍然利用内部逐步分析进行质量控制。

## 理论问题

1.  思维链（CoT）是什么，为什么它对多步骤任务有用？

1.  思维链的透明度如何增加用户对模型答案的信任？

1.  思维链在教育场景中如何发挥作用？

1.  推理链如何提高支持聊天机器人的答案质量？

1.  内心独白是什么，它与 CoT 在用户看到的内容方面有何不同？

1.  在处理敏感信息时，为什么内心独白很重要？

1.  内心独白如何在不泄露“剧透”的情况下帮助学习场景？

1.  准备 OpenAI API 示例环境需要哪些步骤？

1.  `get_response_for_queries` 函数是如何结构的？

1.  CoT 提示如何简化处理复杂查询？

1.  系统或用户提示结构如何帮助回答产品问题？

1.  为什么在使用内心独白时，只提取答案的最后部分是有用的？

## 实践任务

1.  实现 `chain_of_thought_prompting(query)`，它生成一个具有步骤结构的系统提示，并将用户查询包裹在分隔符中。

1.  编写 `get_final_response(output, delimiter)` 以提取答案的最后部分并处理可能出现的错误。

1.  创建一个脚本，发送两个查询——一个带有 CoT，一个带有内心独白——并打印出两个响应。

1.  实现 `validate_response_structure(resp, delimiter)` 以检查答案是否包含所需数量的步骤。

1.  构建一个 `QueryProcessor` 类，它封装了 CoT 和内心独白的逻辑（键加载、提示组装、请求发送、后处理和错误处理）。
