# 答案 3.3

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-3/Answers%203.3/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-3/Answers%203.3/)

## 理论

1.  设置问答生成器环境包括导入所需的库、抑制非必要警告和加载 API 密钥（CircleCI、GitHub、OpenAI）。

1.  数据集结构应包括问题模板和按主题、类别和事实组织的“问答库”。例如：“历史”、“技术”、“地理”以及相应的事实。

1.  提示工程指导 AI 生成与所选类别相关的内 容。提示模板可以规定从库中选择主题并形成问答问题。

1.  LangChain 的作用是结构化提示、选择语言模型（LLM）并为处理输出配置解析器。

1.  问答生成管道是由结构化提示、模型和解析器组成的，使用 LangChain 表达式语言实现。

1.  如 `evaluate_quiz_content` 这样的函数通过检查预期关键词来评估生成问答内容的相关性和正确性。

1.  通过 `evaluate_request_refusal` 测试适当的拒绝处理，确保系统返回预期范围的拒绝。

1.  “科学”测试检查生成的问题是否包含科学主题的指标（例如，“物理”、“化学”、“生物学”、“天文学”）。

1.  Python 项目的 CircleCI 配置的基本组件包括：版本、 orbs、作业（构建/测试）、Docker 镜像、步骤（检出/测试）和工作流程。

1.  为项目定制 CircleCI 配置涉及设置 Python 版本、测试命令，并添加额外步骤以准确反映实际的构建、测试和部署过程。

## 实践

任务的解决方案：

### 任务 1：创建问答数据集

我们定义一个代表问答项集合的 Python 字典，按主题组织，每个主题都有其类别和事实。

```py
quiz_bank = {
    "Historical Conflict": {
        "categories": ["History", "Politics"],
        "facts": [
            "Began in 1914 and ended in 1918",
            "Involved two major alliances: the Allies and the Central Powers",
            "Known for the extensive use of trench warfare on the Western Front"
        ]
    },
    "Revolutionary Communication Technology": {
        "categories": ["Technology", "History"],
        "facts": [
            "Invented by Alexander Graham Bell in 1876",
            "Revolutionized long-distance communication",
            "First words transmitted were 'Mr. Watson, come here, I want to see you'"
        ]
    },
    "Iconic American Landmark": {
        "categories": ["Geography", "History"],
        "facts": [
            "Gifted to the United States by France in 1886",
            "Symbolizes freedom and democracy",
            "Located on Liberty Island in New York Harbor"
        ]
    }
} 
```

### 任务 2：使用提示生成问答

此函数根据给定的类别生成问答问题，通过引用 `quiz_bank` 中的相关主题和事实。它展示了 Python 中的字符串操作和格式化，以构建有意义的问答问题。

```py
def  generate_quiz_questions(category):
    # A list to store generated questions
    generated_questions = []

    # Iterate over each subject in the quiz bank
    for subject, details in quiz_bank.items():
        # Check whether the category appears in the subject’s categories
        if category in details["categories"]:
            # For each fact, create a question and add it to the list
            for fact in details["facts"]:
                question = f"What is described by the following fact: {fact}? Answer: {subject}."
                generated_questions.append(question)

    return generated_questions

# Example usage
history_questions = generate_quiz_questions("History")
for question in history_questions:
    print(question) 
```

### 任务 3：实现 LangChain 风格的提示结构

为了模拟使用 LangChain 结构化问答提示，我们可以定义一个 Python 函数，将问答列表格式化为结构化提示。这个结构化提示模仿了详细的指令和格式，这将指导 LLM 生成或处理问答内容。

```py
def  structure_quiz_prompt(quiz_questions):
    # Define a delimiter for separating questions
    section_delimiter = "####"

    # Start with an introductory instruction
    structured_prompt = "Instructions for generating a personalized quiz:\nEach question is separated by four hash symbols (####)\n\n"

    # Add each question, separated by the delimiter
    for question in quiz_questions:
        structured_prompt += f"{section_delimiter}\n{question}\n"

    return structured_prompt

# Example usage
quiz_questions = [
    "In which year was the Declaration of Independence signed?",
    "Who invented the telephone?"
]
print(structure_quiz_prompt(quiz_questions)) 
```

此函数接受一个问答列表，并返回一个字符串，将它们结构化以模拟问答生成 LLM 的输入，使用指定的分隔符来分隔问题。

### 任务 4：问答生成管道

```py
def  generate_quiz_questions(category):
  """
 Simulates generating quiz questions based on a category.
 """
    # Placeholder for simple generation logic by category
    questions = {
        "Science": ["What is the chemical symbol for water?", "Which planet is known as the Red Planet?"],
        "History": ["Who was the first President of the United States?", "In what year did the Titanic sink?"]
    }
    return questions.get(category, [])

def  structure_quiz_prompt(quiz_questions):
  """
 Structures a chat prompt with the provided quiz questions.
 """
    section_delimiter = "####"
    prompt = "Generated quiz questions:\n\n"
    for question in quiz_questions:
        prompt += f"{section_delimiter} Question: {question}\n"
    return prompt

def  select_language_model():
  """
 Simulates selecting a language model.
 """
    # For this example, assume the model is a constant string
    return "gpt-3.5-turbo"

def  execute_language_model(prompt):
  """
 Simulates executing the selected language model with the given prompt.
 """
    # Normally this would send the prompt to the model and receive output.
    # Here we simulate it by echoing the prompt with a confirmation.
    return f"The model received the following prompt: {prompt}\nModel: 'Questions created successfully.'"

def  generate_quiz_pipeline(category):
  """
 Simulates creating and executing a quiz generation pipeline using placeholders.
 """
    # Step 1: Generate questions based on the chosen category
    quiz_questions = generate_quiz_questions(category)

    # Step 2: Structure the prompt with the generated questions
    prompt = structure_quiz_prompt(quiz_questions)

    # Step 3: Select the language model to simulate
    model_name = select_language_model()

    # Step 4: Execute the language model with the structured prompt
    model_output = execute_language_model(prompt)

    # Final: Return a message simulating pipeline execution
    return f"Pipeline executed using model: {model_name}. Output: {model_output}"

# Example usage
print(generate_quiz_pipeline("Science")) 
```

这组函数模拟了一个测验生成管道：基于类别生成问题，将它们结构化为提示，选择模型，并执行它以生成模拟输出。

### 任务 5：可重用测验生成函数

```py
def  create_structured_prompt(system_prompt_message, user_question_template="{question}"):
  """
 Creates a structured prompt using a system message and a user question template.
 """
    prompt = (
        f"System instructions: {system_prompt_message}\n"
        f"User template: {user_question_template}\n"
    )
    return prompt

def  select_language_model():
  """
 Simulates selecting a language model and temperature.
 """
    return "gpt-3.5-turbo", 0

def  simulate_model_response(structured_prompt):
  """
 Simulates generating a response from the selected language model based on the structured prompt.
 """
    # Here the actual API call to the language model would occur
    # For simulation purposes we return a mock response
    return "A mock quiz has been generated based on the structured prompt."

def  setup_output_parser(model_output):
  """
 Simulates configuring an output parser for formatting the model’s response.
 """
    # Simple formatting for demonstration
    formatted_output = f"Formatted quiz: {model_output}"
    return formatted_output

def  generate_quiz_assistant_pipeline(system_prompt_message, user_question_template="{question}"):
    print("Creating a structured prompt with the system message and user question template...")
    structured_prompt = create_structured_prompt(system_prompt_message, user_question_template)

    print("Selecting language model: GPT-3.5-turbo with temperature 0")
    model_name, temperature = select_language_model()

    print("Simulating language model response...")
    model_output = simulate_model_response(structured_prompt)

    print("Configuring output parser to format responses")
    formatted_output = setup_output_parser(model_output)

    print("Assembling components into a quiz generation pipeline...")
    return formatted_output

# Example usage with a detailed system prompt
system_prompt_message = "Please generate a quiz based on the following categories: Science, History."
print(generate_quiz_assistant_pipeline(system_prompt_message)) 
```

这些函数提供了构建基于 AI 的测验生成提示的结构化过程的基本模拟，组装管道以执行该生成，并创建一个具有可定制参数的可重用函数来生成测验。

### 任务 6：评估生成的测验内容

此函数接受生成的测验内容和预期关键词列表，以确保输出与预期主题或科目相符。如果预期关键词中没有任何一个存在，它将引发断言错误，表明预期内容和生成内容之间存在不匹配。

```py
def  evaluate_quiz_content(generated_content, expected_keywords):
    # Check whether any expected keyword appears in the generated content
    if not any(keyword.lower() in generated_content.lower() for keyword in expected_keywords):
        raise AssertionError("The generated content does not contain any of the expected keywords.")
    else:
        print("The generated content successfully contains the expected keywords.")

# Example usage
generated_content = "The law of universal gravitation was formulated by Isaac Newton in the 17th century."
expected_keywords = ["gravity", "Newton", "physics"]
evaluate_quiz_content(generated_content, expected_keywords) 
```

### 任务 7：处理无效测验请求

此函数模拟评估系统对无效测验请求的响应。它验证生成的拒绝是否与预期的拒绝响应相匹配，确认系统无法满足的请求的正确处理。

```py
def  evaluate_request_refusal(invalid_request, expected_response):
    # Simulate generating a response to an invalid request
    generated_response = f"Unable to generate a quiz for: {invalid_request}"  # Placeholder for an actual refusal response

    # Check whether the generated response matches the expected refusal response
    assert generated_response == expected_response, "The refusal response does not match the expected response."
    print("The refusal response correctly matches the expected response.")

# Example usage
invalid_request = "Generate a quiz about unicorns."
expected_response = "Unable to generate a quiz for: Generate a quiz about unicorns."
evaluate_request_refusal(invalid_request, expected_response) 
```

### 任务 8：科学测验评估测试

此函数演示了在特定测试场景中使用 `evaluate_quiz_content`——检查生成的科学测验是否包含与预期科学主题相关的问题。它模拟生成测验内容，然后对其科学导向的关键词进行评估。

```py
def  test_science_quiz():
    # Simulate generating quiz content
    generated_content = "The study of the natural world through observation and experiment is known as science. Key subjects include biology, chemistry, physics, and Earth sciences."

    # Define expected keywords or subjects for a science quiz
    expected_science_subjects = ["biology", "chemistry", "physics", "Earth sciences"]

    # Use evaluate_quiz_content to check for expected keywords
    try:
        evaluate_quiz_content(generated_content, expected_science_subjects)
        print("Science quiz content evaluation passed successfully.")
    except AssertionError as e:
        print(f"Science quiz content evaluation failed: {e}")

# Example usage
test_science_quiz() 
```

综合来看，这些函数提供了评估生成测验内容的相关性和准确性的机制，适当地处理无效请求，并运行有针对性的测试以确保测验内容符合特定的教育或主题标准。
