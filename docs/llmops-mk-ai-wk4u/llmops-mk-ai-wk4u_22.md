# 3.2 使用 Kubeflow Pipelines 的工作流

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-3/3.2%20Workflow%20with%20Kubeflow%20Pipelines/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-3/3.2%20Workflow%20with%20Kubeflow%20Pipelines/)
> 
> **要求:** `pip install kfp>=2.0.0 google-cloud-aiplatform`

让我们看看如何使用 Kubeflow Pipelines 编排和自动化 ML 工作流——这是一个开源框架，使数据科学家、ML 工程师和开发者更容易构建、部署和运行复杂的步骤链。自动化节省时间并确保可重复性和稳定性——这是可靠 ML 系统的基础。我们首先设置 SDK 和管道的“构建块”。 

首先，从 Kubeflow Pipelines SDK 导入所需的模块。这些模块是定义管道的构建块。

```py
# Import the DSL (domain‑specific language) and the compiler from the Kubeflow Pipelines SDK
from  kfp  import dsl
from  kfp  import compiler 
```

在这里，`dsl`提供了用于描述组件和结构的装饰器和类，而`compiler`将管道编译为 Kubeflow 引擎的可执行格式。

库快速演变，因此关于即将到来的更改或弃用的警告很常见。为了在学习或演示期间保持输出整洁，你可以选择性地隐藏它们（但定期查看发布说明是明智的）：

```py
# Suppress FutureWarning originating from the Kubeflow Pipelines SDK
import  warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='kfp.*') 
```

这使用了标准的`warnings`模块来过滤`kfp.*`中的`FutureWarning`，帮助你专注于重要信息。

请记住：关注 Kubeflow Pipelines 的发布并选择性地抑制警告——完全静音它们可能会隐藏真正的问题。

对于详细信息，请随时查阅 Kubeflow Pipelines 文档和 MLOps 指南（例如，关于持续交付和自动化管道的 Google Cloud 材料）。掌握它们可以显著提高 ML 工作流的效率和可靠性。

Kubeflow 将 ML 工作流结构化为可重用的组件和管道：组件是隔离的步骤（预处理、训练、部署等），而管道是输出成为后续步骤输入的组成，形成一个端到端的过程。

作为参考点，从一个简单的“问候”组件开始，该组件接受一个名字并返回一个字符串。这是使用 Kubeflow Pipelines SDK 定义组件的基本演示：

```py
# Import the DSL module to define components and pipelines
from  kfp  import dsl

# Define a simple component using the @dsl.component decorator
@dsl.component
def  greet_person(name: str) -> str:
    # Form a greeting by combining "Hello" with the input name
    greeting_message = f'Hello, {name}!'

    # Return the constructed greeting message
    return greeting_message 
```

`@dsl.component`装饰器将函数标记为管道组件；`greet_person`接受`name`并形成一个可以传递到真实管道中的问候语。

保持输入/输出接口清晰，并设计组件以便它们可以在管道中重用。

当与组件一起工作时，了解输出和`PipelineTask`：带有`@dsl.component`标记的函数在管道内部调用时，不会返回“就绪”数据。它返回一个表示步骤执行的`PipelineTask`对象，并作为传递数据的链接。

```py
# Assign the result of calling the component function to a variable
hello_task = greet_person(name="Erwin")
print(hello_task) 
```

该组件返回一个`PipelineTask`，而不是一个字符串。

#### 通过`.output`访问数据

要在管道中使用组件的输出，请参考`PipelineTask`对象的`.output`属性。它允许您将一步的结果传递到下一步，从而组织管道的数据流。

```py
# Access the component’s output via the .output attribute
print(hello_task.output) 
```

`.output`属性具有内置数据类型（String/Integer/Float/Boolean/List/Dict），在管道组件之间兼容。

#### 仅使用命名参数

重要：所有组件参数都是通过名称（关键字参数）传递的。这增加了清晰度并防止了错误，尤其是在组件有多个输入时。

```py
# This will raise an error because it uses a positional argument
# hello_task = greet_person("Erwin")

# Correct: call with a named argument
hello_task = greet_person(name="Erwin") 
```

小贴士 - 参数名称：始终使用命名参数调用组件。 - 组件输出：通过`PipelineTask.output`计划步骤之间的数据传递。

### 连接组件：传递输出

在组件的基础上，让我们创建一个管道，其中一个组件的输出作为另一个组件的输入——这是 Kubeflow Pipelines 的核心功能。

#### 依赖组件

定义第二个组件，它接受第一个组件的问候语并附加一个后续问题。这展示了管道的一个步骤如何依赖于前一个步骤的结果。

```py
# Import DSL to define components
from  kfp  import dsl

# Define a component that depends on another component’s output
@dsl.component
def  ask_about_wellbeing(greeting_message: str) -> str:
    # Form a new message that includes the greeting and a follow‑up question
    follow_up_message = f"{greeting_message}. How are you?"

    # Return the new message
    return follow_up_message 
```

#### 在组件之间传递输出

现在将第一个组件（`greet_person`）的输出传递给第二个组件（`ask_about_wellbeing`）。这是连接组件和组织管道数据流的关键步骤。

```py
# Create a task for the first component and keep its output
greeting_task = greet_person(name="Erwin")

# Feed the first component’s output into the second component
wellbeing_task = ask_about_wellbeing(greeting_message=greeting_task.output)
print(wellbeing_task)
print(wellbeing_task.output) 
```

在这里，`greeting_task.output`作为`greeting_message`传递给第二个组件，展示了数据如何在管道步骤之间流动。

#### 常见错误：传递`PipelineTask`而不是`.output`

在连接组件时，务必传递`PipelineTask.output`属性——而不是`PipelineTask`对象本身。传递任务对象将失败，因为组件期望内置数据类型，而不是任务对象。

```py
# Incorrect: passing a PipelineTask instead of its output — this will error
# wellbeing_task = ask_about_wellbeing(greeting_message=greeting_task)

# Correct: pass the task’s .output attribute
wellbeing_task = ask_about_wellbeing(greeting_message=greeting_task.output) 
```

#### 实用技巧

+   总是传递`.output`作为依赖项：在连接组件时，确保传递前一个任务的`.output`。

+   单独测试组件：在集成之前验证每个组件，以尽早发现问题。

在 Kubeflow Pipelines 中掌握组件连接技术，让您构建模块化、可读性和灵活的机器学习工作流程。它还提高了协作，并鼓励跨项目重用，加速开发。

### 在 Kubeflow 中构建和理解管道

Kubeflow Pipelines 编排复杂的流程。一个管道将多个组件连接起来，让数据从一个组件流向另一个组件，形成一个端到端的过程。以下是使用上述组件定义简单管道的方法。

#### 定义管道

我们将创建一个管道，将`greet_person`和`ask_about_wellbeing`连接起来。它接受一个名字，用它来问候这个人，然后进行后续询问。这展示了如何定义管道和正确处理组件输出。

```py
# Import DSL to define pipelines
from  kfp  import dsl

# Define a pipeline that orchestrates the greeting and follow‑up components
@dsl.pipeline
def  hello_and_wellbeing_pipeline(recipient_name: str) -> str:
    # Task for the greet_person component
    greeting_task = greet_person(name=recipient_name)

    # Task for ask_about_wellbeing, using greeting_task’s output
    wellbeing_task = ask_about_wellbeing(greeting_message=greeting_task.output)

    # Return the final message produced by wellbeing_task
    return wellbeing_task.output 
```

`recipient_name`参数传递给`greet_person`。它的输出（`greeting_task.output`）成为`ask_about_wellbeing`的输入。管道返回`wellbeing_task.output`，说明了通过管道的数据流。

#### 执行和处理输出

当你在代码中“运行”流水线定义时，你可能期望直接得到最终的字符串（例如，"Hello, Erwin. How are you?"）。但由于 Kubeflow 流水线的工作方式，流水线函数本身返回的是一个`PipelineTask`，而不是原始输出数据。

```py
# Run the pipeline with a recipient name
pipeline_output = hello_and_wellbeing_pipeline(recipient_name="Erwin")
print(pipeline_output) 
```

这强调了关键点：流水线函数描述了一个工作流程；实际的执行发生在 Kubeflow 流水线环境中，数据在组件之间传递，输出根据流水线图进行处理。

#### 错误处理：错误的返回类型

如果你尝试返回`PipelineTask`本身而不是其`.output`，流水线将失败。流水线的返回必须是最终组件产生的数据类型，与预期输出匹配。

```py
# Incorrect pipeline that returns a PipelineTask object
@dsl.pipeline
def  hello_and_wellbeing_pipeline_with_error(recipient_name: str) -> str:
    greeting_task = greet_person(name=recipient_name)
    wellbeing_task = ask_about_wellbeing(greeting_message=greeting_task.output)

    # Incorrect: returning the PipelineTask itself
    return wellbeing_task
    # This will error 
```

#### 实用技巧

+   返回类型：确保流水线的返回类型与最终组件产生的数据类型匹配。这对于正确的执行和输出处理至关重要。

+   流水线执行：在脚本或笔记本中调用流水线定义准备工作流程。实际的执行发生在 Kubeflow 流水线中，其中基础设施运行流水线。

这个例子展示了如何在 Kubeflow 中定义一个简单而有效的流水线。它强调了理解组件输出、数据流和 Kubeflow 编排功能的重要性。这些概念是构建可扩展、可靠的 ML 工作流程的基础。

### 实现和运行 Kubeflow 流水线

实现 Kubeflow 流水线涉及关键步骤：定义组件，将它们编排成流水线，将流水线编译成可执行格式，并在合适的环境中运行。我们使用`hello_and_wellbeing_pipeline`来展示这些步骤。

#### 编译流水线

Kubeflow 流水线使用 YAML 作为可执行规范。编译将 Python 定义转换为描述流水线 DAG、组件和数据流的静态配置。

```py
# Import the compiler from the Kubeflow Pipelines SDK
from  kfp  import compiler

# Compile the pipeline to a YAML file
compiler.Compiler().compile(hello_and_wellbeing_pipeline, 'pipeline.yaml') 
```

这生成了`pipeline.yaml`，这是流水线的编译表示。这个 YAML 是你部署到运行时使用的。

#### 检查编译后的流水线

查看 YAML 有助于理解结构是如何被捕获的。虽然不是必需的，但对于学习和调试很有用。

```py
# Inspect the compiled pipeline YAML (cross-platform Python approach)
from  pathlib  import Path
print(Path('pipeline.yaml').read_text()) 
```

或者，从命令行：

```py
cat  pipeline.yaml  # Linux/macOS
type  pipeline.yaml  # Windows 
```

#### 运行流水线

使用 Vertex AI 流水线（Google Cloud 上的托管、无服务器环境）运行编译后的流水线，无需管理基础设施。

首先，定义流水线参数——参数化运行的输入：

```py
# Define pipeline arguments
pipeline_arguments = {
    "recipient_name": "World!",
} 
```

然后使用`google.cloud.aiplatform.PipelineJob`来配置和提交运行：

```py
from  google.cloud.aiplatform  import PipelineJob

job = PipelineJob(
    template_path="pipeline.yaml",
    display_name="hello_and_wellbeing_ai_pipeline",
    parameter_values=pipeline_arguments,
    location="us-central1",
    pipeline_root="./",
)

job.submit()
print(job.state) 
```

注意：由于类/笔记本的限制，我们在这里不执行它。请在自己的 Google Cloud 项目中运行它。

### 摘要

我们介绍了实现 Kubeflow 流水线的步骤：定义组件和流水线，将其编译成可部署的格式，并在托管环境中运行。通过这些步骤，你可以有效地自动化和扩展 ML 工作流程。

### 使用 Kubeflow 自动化和编排微调流水线

作为实际示例，使用 Kubeflow Pipelines 自动化和编排 Google 的 PaLM 2 的参数高效微调（PEFT）管道。重复使用现有管道显著减少了开发时间并保留了最佳实践。

#### 重复使用现有管道以提高效率

重复使用提供的管道可以加速实验和部署，特别是对于大型模型。在此，我们专注于 Google 的 PEFT 管道，它允许我们在我们的数据集上从零开始微调基模型。

#### 数据准备和模型版本控制

使用两个 JSONL 文件进行训练和评估。删除时间戳确保协作者之间的一致性。

```py
TRAINING_DATA_URI = "./tune_data_stack_overflow_python_qa.jsonl"
EVALUATION_DATA_URI = "./tune_eval_data_stack_overflow_python_qa.jsonl"

import  datetime
date = datetime.datetime.now().strftime("%H:%d:%m:%Y")
MODEL_NAME = f"deep-learning-ai-model-{date}" 
```

设置核心超参数：

```py
TRAINING_STEPS = 200
EVALUATION_INTERVAL = 20 
```

认证并设置项目上下文（示例辅助）：

```py
from  utils  import authenticate
credentials, PROJECT_ID = authenticate()
REGION = "us-central1" 
```

定义管道参数：

```py
pipeline_arguments = {
    "model_display_name": MODEL_NAME,
    "location": REGION,
    "large_model_reference": "text-bison@001",
    "project": PROJECT_ID,
    "train_steps": TRAINING_STEPS,
    "dataset_uri": TRAINING_DATA_URI,
    "evaluation_interval": EVALUATION_INTERVAL,
    "evaluation_data_uri": EVALUATION_DATA_URI,
} 
```

通过`PipelineJob`提交作业（启用缓存以重复使用未更改的步骤输出）：

```py
from  google.cloud.aiplatform  import PipelineJob

pipeline_root = "./"

job = PipelineJob(
    template_path=template_path,
    display_name=f"deep_learning_ai_pipeline-{date}",
    parameter_values=pipeline_arguments,
    location=REGION,
    pipeline_root=pipeline_root,
    enable_caching=True,
)

job.submit()
print(job.state) 
```

#### 结论

此示例说明了使用 Kubeflow Pipelines 自动化和编排针对基模型的微调管道。通过重复使用现有管道、指定关键参数和在受管理环境中执行，可以有效地对特定数据集上的大型模型（如 PaLM 2）进行微调。这种方法加速了开发并嵌入 MLOps 最佳实践，如版本控制、可重复性和高效资源使用。

## 理论问题

1.  Kubeflow Pipelines 在自动化 ML 工作流程和确保可重复性中的作用。

1.  SDK 中`dsl`和`compiler`模块的功能。

1.  如何在保持日志可读的同时管理`FutureWarning`，不遗漏重要更改。

1.  为什么清晰的接口和重复使用可以提高模块化和效率。

1.  `@dsl.component`装饰器的目的。

1.  调用一个组件时`PipelineTask`对象代表什么以及为什么它有用。

1.  如何将一个组件的输出作为另一个组件的输入。

1.  为什么组件只接受命名参数。

1.  如何连接组件以及`.output`属性的作用。

1.  如何定义管道以及返回正确值时应注意的事项。

1.  编译、检查和运行管道的步骤，以及 YAML 的作用。

1.  如何通过重复使用管道（例如，PEFT for PaLM 2）加快工作并保持最佳实践。

1.  为什么在 MLOps 中版本化数据和模型；给出一个版本标识符的例子。

1.  如何指定模型微调的管道参数。

1.  在 Kubeflow 中自动化和编排复杂工作流程（针对大型模型）的优缺点。

## 实际任务

1.  从 Kubeflow SDK 导入`dsl`和`compiler`并抑制`kfp.*`中的`FutureWarning`。

1.  使用`@dsl.component`定义一个组件`add_numbers(a: int, b: int) -> int`。

1.  从任何模块中抑制`DeprecationWarning`（通过`warnings`）。

1.  创建两个组件：一个返回一个数字，另一个将其加倍；在管道中连接它们。

1.  使用`compiler`将简单的管道编译为 YAML。

1.  展示如何调用组件返回一个`PipelineTask`以及如何访问`.output`。

1.  展示从管道函数返回`PipelineTask`时的错误，然后用注释修复它。

1.  编写一个 JSON 到 JSON 的预处理脚本（过滤/映射），模拟预处理组件。

1.  添加一个用于版本控制的函数：将当前日期和时间附加到基础模型名称上。

1.  提供参数并将编译后的 YAML 提交到运行时（伪 API）。
