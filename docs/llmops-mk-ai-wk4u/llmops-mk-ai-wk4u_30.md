# 答案 1.5

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-1/Answers%201.5/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-1/Answers%201.5/)

## 理论

1.  提示连接将复杂任务分解为一系列相互关联的步骤（提示），每个步骤解决一个子任务。与“单体”方法不同，它简化并提高了控制。

1.  类比：复杂菜肴的逐步烹饪；模块化开发，每个模块都对最终结果做出贡献。

1.  连接中的工作流程管理意味着在每个步骤后检查状态，并根据迄今为止的结果调整下一步。

1.  资源节省：每个步骤只处理所需的内容，与一个长提示相比减少计算。

1.  错误减少：专注于单个子任务简化了调试并使针对性的改进成为可能。

1.  由于上下文限制，动态信息加载很重要；连接根据需要注入相关数据。

1.  核心步骤：任务分解、状态管理、提示设计、数据加载/预处理、动态上下文注入。

1.  最佳实践：避免不必要的复杂性，编写清晰的提示，管理外部环境，追求效率，并持续测试。

1.  这些示例使用 `dotenv` 和 `openai` 进行配置和 API 调用。

1.  系统消息定义结构和格式，提高精确性和一致性。

1.  产品数据库存储详细信息；通过名称或类别查找函数支持有效的支持答案。

1.  将 JSON 字符串转换为 Python 对象简化了链中的下游处理。

1.  从数据中格式化用户答案保持交互信息丰富且相关。

1.  连接操作让系统从初始请求过渡到故障排除、保修和建议，覆盖复杂支持场景。

## 实践

1.  `retrieve_model_response` 函数：

    ```py
    from  openai  import OpenAI

    client = OpenAI()

    def  retrieve_model_response(message_sequence, model="gpt-4o-mini", temperature=0, max_tokens=500):
        response = client.chat.completions.create(
            model=model,
            messages=message_sequence,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content 
    ```

1.  从请求中提取产品/类别：

    ```py
    system_instruction = """
    You will receive support requests. The request will be delimited by '####'.
    Output a Python list of objects, each representing a product or category mentioned in the request.
    """

    user_query = "#### Tell me about SmartX ProPhone and FotoSnap DSLR Camera, and also your televisions ####"

    message_sequence = [
        {'role': 'system', 'content': system_instruction},
        {'role': 'user', 'content': user_query},
    ]

    extracted_info = retrieve_model_response(message_sequence)
    print(extracted_info) 
    ```

1.  产品数据库助手：

    ```py
    product_database = {
        "SmartX ProPhone": {
            "name": "SmartX ProPhone",
            "category": "Smartphones and Accessories",
        },
        "FotoSnap DSLR Camera": {
            "name": "FotoSnap DSLR Camera",
            "category": "Cameras & Photography",
        },
        "UltraView HD TV": {
            "name": "UltraView HD TV",
            "category": "Televisions",
        },
    }

    def  get_product_details_by_name(product_name):
        return product_database.get(product_name, "Product not found.")

    def  get_products_in_category(category_name):
        return [p for p in product_database.values() if p["category"] == category_name]

    print(get_product_details_by_name("SmartX ProPhone"))
    print(get_products_in_category("Smartphones and Accessories")) 
    ```

1.  JSON 字符串到列表：

    ```py
    import  json

    def  json_string_to_python_list(json_string):
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None

    json_input = '[{"category": "Smartphones and Accessories", "products": ["SmartX ProPhone"]}]'
    python_list = json_string_to_python_list(json_input)
    print(python_list) 
    ```

1.  生成面向用户的答案：

    ```py
    def  generate_response_from_data(product_data_list):
        if not product_data_list:
            return "We couldn't find products matching your request."

        response_string = ""
        for product_data in product_data_list:
            response_string += f"Product: {product_data['name']}\n"
            response_string += f"Category: {product_data['category']}\n\n"
        return response_string

    python_list = [{'category': 'Smartphones and Accessories', 'products': ['SmartX ProPhone']}]
    final_response = generate_response_from_data(python_list)
    print(final_response) 
    ```

1.  端到端支持场景：描述助手如何使用上述功能处理初始产品查询、故障排除、保修问题和配件推荐。

    ```py
    # 1) Initial product inquiry: extract entities and list details
    system_instruction_catalog = """
    You will receive support requests delimited by '####'.
    Return a Python list of objects: mentioned products/categories.
    """

    user_query_1 = "#### I'm interested in upgrading my smartphone. What can you tell me about the latest models? ####"

    message_sequence_1 = [
        {'role': 'system', 'content': system_instruction_catalog},
        {'role': 'user', 'content': user_query_1},
    ]
    extracted = retrieve_model_response(message_sequence_1)
    print("Extracted entities:", extracted)

    # Suppose we parsed 'extracted' to a Python list called parsed_entities (omitted for brevity)
    # You could then look up details via your product DB helpers:
    # for e in parsed_entities: ... get_product_details_by_name(...), get_products_in_category(...)

    # 2) Troubleshooting: step‑by‑step guidance for a specific product issue
    troubleshooting_query = "#### I just bought the FotoSnap DSLR Camera you recommended, but I can't pair it with my smartphone. What should I do? ####"
    system_instruction_troubleshooting = "Provide step‑by‑step troubleshooting advice for the customer’s issue."
    message_sequence_2 = [
        {'role': 'system', 'content': system_instruction_troubleshooting},
        {'role': 'user', 'content': troubleshooting_query},
    ]
    troubleshooting_response = retrieve_model_response(message_sequence_2)
    print("Troubleshooting response:\n", troubleshooting_response)

    # 3) Warranty: clarify coverage details
    follow_up_query = "#### Also, could you clarify what the warranty covers for the FotoSnap DSLR Camera? ####"
    system_instruction_warranty = "Provide detailed information about the product’s warranty coverage."
    message_sequence_3 = [
        {'role': 'system', 'content': system_instruction_warranty},
        {'role': 'user', 'content': follow_up_query},
    ]
    warranty_response = retrieve_model_response(message_sequence_3)
    print("Warranty response:\n", warranty_response)

    # 4) Recommendations: suggest compatible accessories based on user interest
    additional_assistance_query = "#### Given your interest in photography, would you like recommendations for lenses and tripods compatible with the FotoSnap DSLR Camera? ####"
    system_instruction_recommendations = "Suggest accessories that complement the user’s existing products."
    message_sequence_4 = [
        {'role': 'system', 'content': system_instruction_recommendations},
        {'role': 'user', 'content': additional_assistance_query},
    ]
    recommendations_response = retrieve_model_response(message_sequence_4)
    print("Accessory recommendations:\n", recommendations_response) 
    ```

    此序列演示了一个完整、连接的工作流程，其中助手：- 从提到的实体中提取并咨询产品数据库。- 提供针对问题的逐步故障排除。- 清晰简洁地解释保修范围。- 提供符合用户兴趣的个性化配件推荐。
