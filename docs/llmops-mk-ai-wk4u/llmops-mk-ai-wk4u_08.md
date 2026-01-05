# 1.5 提示链的力量

> [`boramorka.github.io/LLM-Book/en/CHAPTER-1/1.5%20The%20Power%20of%20Prompt%20Chaining/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-1/1.5%20The%20Power%20of%20Prompt%20Chaining/)

提示链通过一系列简单、相互关联的步骤解决复杂任务。而不是一个“单体”请求，你构建一个小的提示链：每个步骤解决一个特定的子任务并为下一步准备上下文。这减少了错误，使模型行为更容易控制，并增加了可观察性：更容易看到错误发生的地方和原因，并精确地干预。这就像一步一步地烹饪复杂的菜肴或使用软件中的模块化架构——总是更容易调试和维护一系列小而清晰的操作，而不是一个像意大利面一样的单一步骤。

实际好处是显而易见的：你可以通过在每个步骤中检查点状态并调整下一步以适应前一步的结果来编排工作流程；节省上下文和预算，因为长提示成本更高，而每个链步骤只使用所需的最小量；通过隔离问题来减少错误；并且只加载相关信息，尊重 LLM 上下文限制。从方法论上讲，这意味着分解任务，明确管理步骤之间的状态，为每个提示设计狭窄的焦点，添加加载数据和预处理数据的工具，并动态注入当前需要的上下文片段。最佳实践很简单：当单个提示足够时不要过于复杂；保持清晰；保持和更新外部上下文；考虑效率（质量、成本、延迟）；并端到端测试链。

下面是一个按顺序组装端到端场景的示例：实体提取、查询简单的“数据库”、解析 JSON 和编写用户界面答案——然后将所有这些整合成一个单一的支持流程。

```py
import  os
from  openai  import OpenAI
from  dotenv  import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
client = OpenAI() 
```

现在从用户请求中提取实体。第一步在系统指令中设置任务和输出格式。用户输入由分隔符界定，这使得控制数据边界和将结果传递到链条中变得更容易。

```py
def  retrieve_model_response(message_sequence, model="gpt-4o-mini", temperature=0, max_tokens=500):
    response = client.chat.completions.create(
        model=model,
        messages=message_sequence,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

system_instruction = """
You will receive support requests. The request is delimited by '####'.
Return a Python list of objects, each representing a product or category mentioned in the request.
"""

user_query = "#### Tell me about the SmartX ProPhone and the FotoSnap DSLR Camera, and also about your televisions ####"

message_sequence = [
    {'role': 'system', 'content': system_instruction},
    {'role': 'user', 'content': user_query},
]

extracted_info = retrieve_model_response(message_sequence)
print(extracted_info) 
```

接下来，连接一个数据源并找到特定的产品或类别。即使是一个内存中的“数据库”也能演示这个想法：提取实体 → 移动到结构化数据 → 准备答案的事实。

```py
product_database = {
    "TechPro Ultrabook": {
        "name": "TechPro Ultrabook",
        "category": "Computers and Laptops",
    },
    # ...
}

def  get_product_details_by_name(product_name):
    return product_database.get(product_name, None)

def  get_products_in_category(category_name):
    return [p for p in product_database.values() if p["category"] == category_name]

print(get_product_details_by_name("TechPro Ultrabook"))
print(get_products_in_category("Computers and Laptops")) 
```

在下一步中，将模型在实体提取过程中可能返回的 JSON 字符串转换为 Python 对象，以便进行后续的链步骤。

```py
import  json

def  json_string_to_python_list(json_string):
    if json_string is None:
        return None
    try:
        json_string = json_string.replace("'", '"')
        return json.loads(json_string)
    except json.JSONDecodeError:
        print("Error: Invalid JSON string")
        return None

json_input = "[{\"category\": \"Smartphones and Accessories\", \"products\": [\"SmartX ProPhone\"]}]"
python_list = json_string_to_python_list(json_input)
print(python_list) 
```

最后，从生成的结构中编写一个简洁的用户界面答案。你可以将这个格式化层替换为模板、本地化或针对你的 UX 调整的生成。

```py
def  generate_response_from_data(product_data_list):
    response_string = ""
    if product_data_list is None:
        return response_string
    for data in product_data_list:
        response_string += json.dumps(data, indent=4) + "\n"
    return response_string

response_instruction = """
You are a support assistant. Answer briefly and ask clarifying questions when needed.
"""

final_response = generate_response_from_data(python_list)
print(final_response) 
```

我们将以一个端到端的支持场景结束：首先检测对摄影的兴趣，然后提供故障排除，明确保修范围，并以配件推荐结束——一个链条中的四个步骤，每个步骤都建立在之前的结果之上。

```py
system_instruction = """
You will receive support requests delimited by '####'.
Return a list of objects: the mentioned products/categories.
"""

user_query_1 = "#### I'm interested in upgrading my photography gear. Can you tell me about the latest DSLR cameras and compatible accessories? ####"

message_sequence_1 = [
    {'role': 'system', 'content': system_instruction},
    {'role': 'user', 'content': user_query_1},
]
response_1 = retrieve_model_response(message_sequence_1)
print("Product query response:", response_1)

troubleshooting_query = "#### I just bought the FotoSnap DSLR Camera you recommended, but I'm having trouble connecting it to my smartphone. What should I do? ####"
system_instruction_troubleshooting = "Provide step‑by‑step troubleshooting advice for the customer’s issue."
message_sequence_2 = [
    {'role': 'system', 'content': system_instruction_troubleshooting},
    {'role': 'user', 'content': troubleshooting_query},
]
response_2 = retrieve_model_response(message_sequence_2)
print("Troubleshooting response:", response_2)

follow_up_query = "#### Also, could you clarify what the warranty covers for the FotoSnap DSLR Camera? ####"
system_instruction_follow_up = "Provide detailed information about the product’s warranty coverage."
message_sequence_3 = [
    {'role': 'system', 'content': system_instruction_follow_up},
    {'role': 'user', 'content': follow_up_query},
]
response_3 = retrieve_model_response(message_sequence_3)
print("Warranty information response:", response_3)

additional_assistance_query = "#### Given your interest in photography, would you like recommendations for lenses and tripods compatible with the FotoSnap DSLR Camera? ####"
system_instruction_additional_assistance = "Suggest accessories that complement the user’s existing products."
message_sequence_4 = [
    {'role': 'system', 'content': system_instruction_additional_assistance},
    {'role': 'user', 'content': additional_assistance_query},
]
response_4 = retrieve_model_response(message_sequence_4)
print("Additional assistance response:", response_4) 
```

最后，提示链式操作为你提供了一个强大且易于理解的流程：它节省了上下文和预算，更精确地定位错误，并保留了根据用户任务定制答案的灵活性。

## 理论问题

1.  提示链式操作是什么？它与使用一个长提示有何不同？

1.  提供两个类比并解释它们如何映射到链式操作。

1.  链式操作如何帮助管理工作流程？

1.  使用链式操作时，节省来自哪里？

1.  链式操作如何减少复杂任务中的错误？

1.  考虑到 LLM 上下文限制，动态数据加载为什么有用？

1.  描述链式操作的逐步方法论以及每一步的作用。

1.  列出确保链式操作效率的最佳实践。

1.  示例中使用了哪些库以及用途是什么？

1.  系统消息如何引导模型的答案？

1.  产品数据库的作用是什么？如何查询它？

1.  为什么需要将 JSON 字符串转换为 Python 对象，以及如何进行转换？

1.  如何通过格式化处理后的数据来提高服务质量？

1.  终端到终端场景如何通过链式操作来适应用户需求？

## 实践任务

1.  实现`retrieve_model_response`函数，包含`model`、`temperature`和`max_tokens`参数。

1.  使用系统指令展示实体提取的示例。

1.  创建一个迷你产品数据库和按名称或类别查询的功能。

1.  实现带有错误处理的 JSON 到 Python 列表的转换。

1.  编写`generate_response_from_data`函数，该函数将数据列表格式化为用户友好的答案。

1.  基于上述功能，编写一个端到端支持场景（查询 → 故障排除 → 保修 → 推荐）。
