# 3.3 AI 问答生成机制

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-3/3.3%20AI%20Quiz%20Generation%20Mechanism/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-3/3.3%20AI%20Quiz%20Generation%20Mechanism/)

本章从头到尾组装了一个工作的人工智能问答生成器：我们设置环境并访问外部服务，准备主题/类别/事实的紧凑数据集，设计一个提示，以确保问题严格匹配所选类别，并将其所有内容连接到 LangChain 管道中。我们首先设置环境并获取密钥；为了保持输出整洁，您可以抑制非必要的警告。

```py
# Use the warnings library to control warning messages
import  warnings

# Ignore all warnings to ensure clean runtime output
warnings.filterwarnings('ignore')

# Load API keys for third‑party services used in the project
from  utils  import get_circle_ci_api_key, get_github_api_key, get_openai_api_key

# Obtain individual API keys for CircleCI, GitHub, and OpenAI
circle_ci_api_key = get_circle_ci_api_key()
github_api_key = get_github_api_key()
openai_api_key = get_openai_api_key() 
```

接下来，我们构建应用程序的核心——一个紧凑的数据集，从中将组成问题：我们固定主题、类别和事实，这些将用于构建问答。

```py
# Define a template for structuring quiz questions
quiz_question_template = "{question}"

# Initialize a quiz bank with subjects, categories, and facts
quiz_bank = """
Here are three new quiz questions following the given format:

1\. Subject: A Historical Conflict 
 Categories: History, Politics 
 Facts: 
 - Began in 1914 and ended in 1918 
 - Involved two major alliances: the Allies and the Central Powers 
 - Known for extensive trench warfare on the Western Front 

2\. Subject: A Revolutionary Communication Technology 
 Categories: Technology, History 
 Facts: 
 - Invented by Alexander Graham Bell in 1876 
 - Revolutionized long‑distance communication 
 - The first words transmitted were "Mr. Watson, come here, I want to see you" 

3\. Subject: An Iconic American Landmark 
 Categories: Geography, History 
 Facts: 
 - Gifted to the United States by France in 1886 
 - Symbolizes freedom and democracy 
 - Located on Liberty Island in New York Harbor 
""" 
```

为了确保问题与用户选择的类别相关，我们设计了一个详细的提示模板：从类别选择通过问答库到以规定的格式制定问题。

```py
# Define a delimiter to separate different parts of the quiz prompt
section_delimiter = "####"

# Create a detailed prompt template guiding the AI to generate user‑customized quizzes
quiz_generation_prompt_template = f"""
Instructions for generating a customized quiz:
Each question is separated by four hashes, i.e. {section_delimiter}

The user chooses a category for the quiz. Ensure the questions are relevant to the chosen category.

Step 1:{section_delimiter} Identify the user‑selected category from the list below:
* History
* Technology
* Geography
* Politics

Step 2:{section_delimiter} Choose up to two subjects that match the selected category from the quiz bank:

{quiz_bank}

Step 3:{section_delimiter} Create a quiz based on the selected subjects by formulating three questions per subject.

Quiz format:
Question 1:{section_delimiter} <Insert Question 1>
Question 2:{section_delimiter} <Insert Question 2>
Question 3:{section_delimiter} <Insert Question 3>
""" 
```

使用此模板，我们转向 LangChain：形成一个 ChatPrompt，选择一个模型和一个解析器，将响应标准化为可读形式。

```py
# Import required components from LangChain for prompt structuring and LLM interaction
from  langchain.prompts  import ChatPromptTemplate
from  langchain_openai  import ChatOpenAI
from  langchain.schema.output_parser  import StrOutputParser

# Convert the detailed quiz generation prompt into a structured format for the LLM
structured_chat_prompt = ChatPromptTemplate.from_messages([("user", quiz_generation_prompt_template)])

# Select the language model for quiz question generation
language_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Configure an output parser to convert the LLM response into a readable format
response_parser = StrOutputParser() 
```

现在我们使用 LangChain 表达式语言将所有内容连接起来，形成一个可重复生成的单一管道。

```py
# Compose the structured prompt, language model, and output parser into a quiz generation pipeline
quiz_generation_pipeline = structured_chat_prompt | language_model | response_parser

# Execute the pipeline to generate a quiz (example invocation not shown) 
```

接下来，将问答生成的设置和执行封装成一个可重用的函数。这增加了模块化并简化了维护。`generate_quiz_assistant_pipeline`将提示创建、模型选择和解析捆绑到一个工作流程中。

快速概述：`generate_quiz_assistant_pipeline`灵活，允许插入不同的模板和配置（模型/解析器）。函数定义：

```py
from  langchain.prompts  import ChatPromptTemplate
from  langchain_openai  import ChatOpenAI
from  langchain.schema.output_parser  import StrOutputParser

def  generate_quiz_assistant_pipeline(
    system_prompt_message,
    user_question_template="{question}",
    selected_language_model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    response_format_parser=StrOutputParser()):
  """
 Assembles the components required to generate quizzes through an AI‑based process.

 Parameters:
 - system_prompt_message: A message containing instructions or context for quiz generation.
 - user_question_template: A template for structuring user questions; defaults to a simple placeholder.
 - selected_language_model: The AI model used to generate content; a default model is provided.
 - response_format_parser: A mechanism for parsing the LLM response into the desired format.

 Returns:
 A LangChain pipeline that, when invoked, generates a quiz based on the provided system message and user template.
 """

    # Create a structured chat prompt from the system and user messages
    structured_chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_message),
        ("user", user_question_template),
    ])

    # Compose the chat prompt, language model, and output parser into a single pipeline
    quiz_generation_pipeline = structured_chat_prompt | selected_language_model | response_format_parser

    return quiz_generation_pipeline 
```

实际应用。该函数隐藏了组件组合的复杂性：只需用所需的参数调用`generate_quiz_assistant_pipeline`以生成主题/类别问答，并轻松集成到更大的系统中。一些实用提示：

+   **配置**：使用参数灵活调整过程。

+   **模型选择**：在质量/创意权衡中尝试不同的模型。

+   ## **提示设计**：仔细规划`user_question_template`和`system_prompt_message`。

    **错误处理**：考虑 API 限制和意外响应。

将此函数包含到您的项目中可以简化创建 AI 驱动的问答，使创新的教育工具和互动内容成为可能。

为了添加质量检查，引入`evaluate_quiz_content`：它验证生成的问答包含预期的主题关键词——这对于学习场景中的相关性和正确性至关重要。

现在关于内容评估。该函数与生成管道集成：它接受系统消息（指令/上下文）、一个特定的请求（例如，问答的主题）以及应在结果中出现的预期单词/短语列表。函数定义：

```py
def  evaluate_quiz_content(
    system_prompt_message,
    quiz_request_question,
    expected_keywords,
    user_question_template="{question}",
    selected_language_model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    response_format_parser=StrOutputParser()):
  """
 Evaluates the generated quiz content to ensure it includes expected keywords or phrases.

 Parameters:
 - system_prompt_message: Instructions or context for quiz generation.
 - quiz_request_question: The specific question or request that triggers quiz generation.
 - expected_keywords: A list of words or phrases that must be present in the quiz content.
 - user_question_template: A template for structuring user questions; defaults to a simple placeholder.
 - selected_language_model: The AI model used to generate content; a default model is provided.
 - response_format_parser: A mechanism for parsing the LLM response into the desired format.

 Raises:
 - AssertionError: If none of the expected keywords are found in the generated quiz content.
 """

    # Use the helper to generate quiz content based on the provided request
    generated_content = generate_quiz_assistant_pipeline(
        system_prompt_message,
        user_question_template,
        selected_language_model,
        response_format_parser).invoke({"question": quiz_request_question})

    print(generated_content)

    # Verify that the generated content includes at least one of the expected keywords
    assert any(keyword.lower() in generated_content.lower() for keyword in expected_keywords), \
        f"Expected the generated quiz to contain one of '{expected_keywords}', but none were found." 
```

考虑一个例子：生成并评估一个科学问答。

```py
# Define the system message (or prompt template), the specific request, and the expected keywords
system_prompt_message = quiz_generation_prompt_template  # Assumes this variable was defined earlier in your code
quiz_request_question = "Generate a quiz about science."
expected_keywords = ["renaissance innovator", "astronomical observation tools", "natural sciences"]

# Call the evaluation function with the test parameters
evaluate_quiz_content(
    system_prompt_message,
    quiz_request_question,
    expected_keywords
) 
```

此示例展示了`evaluate_quiz_content`如何确认科学测验包含相关的主题（图表、仪器、概念）。良好实践：

+   **关键词选择** — 使其足够具体，但留有变通的空间。

+   **广泛检查** — 使用多个关键词集针对不同主题。

+   **迭代方法** — 根据评估结果精炼模板/参数/数据集。

结构化测试有助于保持质量并揭示提高相关性和参与度的机会。

为了处理超出范围请求，引入`evaluate_request_refusal`，它测试在不适当的场景中的适当拒绝。这对信任和用户体验（UX）很重要：该函数模拟系统应该拒绝的情况（基于相关性/限制）并验证是否返回了预期的拒绝消息。函数定义：

```py
def  evaluate_request_refusal(
    system_prompt_message,
    invalid_quiz_request_question,
    expected_refusal_response,
    user_question_template="{question}",
    selected_language_model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    response_format_parser=StrOutputParser()):
  """
 Evaluates the system’s response to ensure it correctly refuses invalid or out‑of‑scope requests.

 Parameters:
 - system_prompt_message: Instructions or context for quiz generation.
 - invalid_quiz_request_question: A request that the system should decline.
 - expected_refusal_response: The expected text indicating the system’s refusal to fulfill the request.
 - user_question_template: A template for structuring user questions; defaults to a simple placeholder.
 - selected_language_model: The AI model used to generate content; a default model is provided.
 - response_format_parser: A mechanism for parsing the LLM response into the desired format.

 Raises:
 - AssertionError: If the system’s response does not contain the expected refusal message.
 """

    # Align parameter order with what `generate_quiz_assistant_pipeline` expects
    generated_response = generate_quiz_assistant_pipeline(
        system_prompt_message,
        user_question_template,
        selected_language_model,
        response_format_parser).invoke({"question": invalid_quiz_request_question})

    print(generated_response)

    # Check that the system’s response contains the expected refusal phrase
    assert expected_refusal_response.lower() in generated_response.lower(), \
        f"Expected a refusal message '{expected_refusal_response}', but got: {generated_response}" 
```

为了说明`evaluate_request_refusal`，考虑一个场景，其中测验生成器应拒绝创建测验，因为请求超出了其范围或不受当前配置支持。

```py
# Define the system message (or prompt template), an out‑of‑scope request, and the expected refusal message
system_prompt_message = quiz_generation_prompt_template  # Assumes this variable was defined earlier in your code
invalid_quiz_request_question = "Generate a quiz about Rome."
expected_refusal_response = "I'm sorry, but I can't generate a quiz about Rome at this time."

# Run the refusal evaluation with the specified parameters
evaluate_request_refusal(
    system_prompt_message,
    invalid_quiz_request_question,
    expected_refusal_response
) 
```

此示例演示了如何测试测验生成器对应该拒绝的请求的反应：通过检查预期的拒绝消息，我们确保系统在面临无法满足的请求时表现正确。提示和建议：

+   **清晰的拒绝消息**：使其信息丰富，以便用户了解为什么请求无法完成。

+   **全面测试**：使用各种场景，包括不支持的主题或格式，以彻底评估拒绝逻辑。

+   **精炼和反馈**：迭代拒绝逻辑和消息以改善用户理解和满意度。

+   **考虑用户体验**：在可能的情况下，提供替代方案或建议以保持积极的互动。

实施和测试拒绝场景确保测验生成器可以可靠地处理各种请求，即使在无法提供请求内容的情况下也能保持稳健性和用户信任。

为了将提供的模板适应于一个以科学主题测验为重点的实际测试场景，我们添加了一个`test_science_quiz`函数。它评估 AI 生成的测验问题是否真正集中在预期的科学主题或科目上。通过集成`evaluate_quiz_content`，我们可以确保测验包含科学类别特有的特定关键词或主题。

最后，我们为科学测试案例定制`evaluate_quiz_content`函数：该函数检查生成的内容是否符合预期的科学主题。测试科学测验的函数定义：

```py
def  test_science_quiz():
  """
 Tests the quiz generator’s ability to create science‑related questions by checking for expected subjects.
 """
    # Define the request to generate a quiz question
    question_request = "Generate a quiz question."

    # The list of expected keywords or subjects indicating scientific alignment
    expected_science_subjects = ["physics", "chemistry", "biology", "astronomy"]

    # The system message or prompt template configured for quiz generation
    system_prompt_message = quiz_generation_prompt_template  # This should be defined earlier in your code

    # Invoke the evaluation with science‑specific parameters
    evaluate_quiz_content(
        system_prompt_message=system_prompt_message,
        quiz_request_question=question_request,
        expected_keywords=expected_science_subjects
    ) 
```

此函数封装了验证逻辑：对于科学请求，内容必须包含预期的科学主题/关键词。调用`test_science_quiz`模拟请求并检查科学主题 — 这是正确生成的关键指标。根据你的领域和覆盖范围细化关键词列表，扩展其他类别（历史/地理/艺术）的测试，并分析失败：比较预期与结果以改进提示逻辑/数据集。结构化测试有助于保持质量并发现改进相关性和参与度的机会。

最后 — 快速了解 CI/CD：存储库根目录中的`.circleci/config.yml`文件描述了一个基于 YAML 的管道（构建/测试/部署）。以下是 Python 项目自动测试的草图：

```py
version:  2.1

orbs:
  python:  circleci/python@1.2.0  # Use the Python orb to simplify your config

jobs:
  build-and-test:
  docker:
  -  image:  cimg/python:3.8  # Specify the Python version
  steps:
  -  checkout  # Check out the source code
  -  restore_cache:  # Restore cache to save time on dependencies installation
  keys:
  -  v1-dependencies-{{ checksum "requirements.txt" }}
  -  v1-dependencies-
  -  run:
  name:  Install Dependencies
  command:  pip install -r requirements.txt
  -  save_cache:  # Cache dependencies to speed up future builds
  paths:
  -  ./venv
  key:  v1-dependencies-{{ checksum "requirements.txt" }}
  -  run:
  name:  Run Tests
  command:  pytest  # Or any other command to run your tests

workflows:
  version:  2
  build_and_test:
  jobs:
  -  build-and-test 
```

关键元素：`version` — 配置版本（通常是 2.1）；`orbs` — 可重用块，这里`python` orb 帮助设置环境；`jobs` — 一组任务，这里是一个单一的`build-and-test`；`docker` — 运行的镜像（例如，`cimg/python:3.8`）；`steps` — 序列（检出、缓存、依赖、测试）；`workflows` — 将作业与流程绑定并按规则触发它们。

要定制：在`docker`下选择你的 Python 版本，将`pytest`替换为你的测试命令，并添加额外的步骤（数据库、环境变量等）作为额外的`- run:`块。提交`.circleci/config.yml`后，CircleCI 将检测配置并在每次提交时根据你的规则运行管道。

## 理论问题

1.  设置基于 AI 的测验生成器环境需要哪些组件？

1.  你如何构建用于生成测验问题的数据集？包括类别和事实的示例。

1.  提示工程如何影响定制化测验生成？提供示例提示模板。

1.  解释 LangChain 在为 LLM 处理构建提示中的作用。

1.  使用 LangChain 表达式语言时，测验生成管道由哪些构成？

1.  如何确保评估函数能够保证生成测验内容的相关性和准确性？

1.  描述一种测试系统在特定条件下拒绝生成测验的能力的方法。

1.  如何测试 LLM 生成的测验问题是否符合预期的科学主题或科目？

1.  描述 Python 项目的 CircleCI 配置文件的关键组件，包括自动测试执行。

1.  讨论定制 CircleCI 配置以匹配项目特定需求的重要性。

## 实践作业

1.  创建测验数据集：定义一个名为`quiz_bank`的 Python 字典，表示一组测验条目，每个条目包含与示例类似的科目、类别和事实。确保你的字典支持对科目、类别和事实的轻松访问。

1.  使用提示生成试题：实现一个名为 `generate_quiz_questions(category)` 的函数，该函数接受一个类别（例如，“历史”，“技术”）作为输入，并返回基于 `quiz_bank` 中的主题和事实生成的试题列表。使用字符串操作或模板构建问题。

1.  实现 LangChain 风格的提示结构：通过编写一个名为 `structure_quiz_prompt(quiz_questions)` 的函数来模拟使用 LangChain 的功能，该函数接受一个试题问题列表，并返回一个类似描述的结构化聊天提示，而不实际集成 LangChain。

1.  试题生成管道：创建一个名为 `generate_quiz_pipeline()` 的 Python 函数，该函数模拟使用 LangChain 组件创建和运行试题生成管道，并使用占位符。该函数应打印一条模拟管道执行的消息。

1.  可重用试题生成函数：实现一个名为 `generate_quiz_assistant_pipeline(system_prompt_message, user_question_template="{question}")` 的 Python 函数，该函数模拟组装试题生成所需的组件。使用字符串格式化从输入构建详细的提示。

1.  评估生成的试题内容：编写一个名为 `evaluate_quiz_content(generated_content, expected_keywords)` 的函数，该函数接受生成的试题内容和预期关键词列表，并检查内容是否包含任何关键词。如果没有找到任何关键词，则引发一个带有自定义消息的断言错误。

1.  处理无效的试题请求：开发一个名为 `evaluate_request_refusal(invalid_request, expected_response)` 的函数，该函数模拟评估系统对无效试题请求的响应。该函数应验证拒绝文本是否与预期的拒绝响应匹配。

1.  科学试题评估测试：开发一个名为 `test_science_quiz()` 的 Python 函数，该函数使用 `evaluate_quiz_content` 函数来测试生成的科学试题是否包含与预期科学主题相关的题目，例如“物理”或“化学”。
