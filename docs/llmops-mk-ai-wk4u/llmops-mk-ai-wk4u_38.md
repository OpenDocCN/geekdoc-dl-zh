# 答案 3.2

> 原文：[`boramorka.github.io/LLM-Book/en/CHAPTER-3/Answers%203.2/`](https://boramorka.github.io/LLM-Book/en/CHAPTER-3/Answers%203.2/)

## 理论

1.  Kubeflow Pipelines 自动化 ML 工作流程，通过高效管理复杂管道提供可重复性和节省时间。

1.  `dsl`模块提供用于定义组件和管道结构的装饰器和类，而`compiler`负责将管道编译成 Kubeflow 引擎可执行的形式。

1.  可以选择性地抑制`FutureWarning`消息以提高日志可读性；同时，跟踪文档更改并相应地更新代码也很重要。

1.  明确定义的接口和组件可重用性简化了集成，增加了模块化和整体系统效率。

1.  `@dsl.component`装饰器将一个函数标记为管道组件，它是工作流程中的一个独立、可重用的步骤。

1.  调用一个组件返回一个`PipelineTask`对象，它表示管道步骤的运行时实例，并用于在组件之间传递数据。

1.  通过`PipelineTask`对象的`.output`属性传递组件的输出。

1.  使用命名参数可以提高代码清晰度并帮助防止错误，尤其是在处理许多输入参数时。

1.  在管道中链式连接组件时，你必须将一个组件的`.output`作为输入传递给另一个组件，以确保正确的数据流。

1.  使用`@dsl.pipeline`装饰器声明管道，它负责编排组件。重要方面包括执行环境和正确处理输出。

1.  管道编译是将其 Python 定义转换为 YAML 文件的过程，然后可以将其上传并在目标 Kubeflow 环境中运行。

1.  重复使用现成的管道（例如，PEFT for PaLM 2）可以显著加快开发速度并帮助保持最佳实践。

1.  模型版本控制对于 MLOps 至关重要，它确保了可重复性和可审计性。例如，你可以在模型名称中添加日期和时间。

1.  管道参数设置输入数据和配置，这对于正确执行至关重要。

1.  Kubeflow 中的自动化和编排可以提高效率并实现可扩展性，但需要仔细规划和深入理解组件和数据流。

## 实践

任务解决方案：

### 1. 设置 Kubeflow Pipelines SDK

```py
# Import the required modules from the Kubeflow Pipelines SDK
from  kfp  import dsl, compiler

# Suppress FutureWarning messages from the Kubeflow Pipelines SDK
import  warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='kfp.*') 
```

此脚本导入`dsl`和`compiler`，并抑制来自`kfp.*`模块的`FutureWarning`消息。

### 2. 定义简单的管道组件

```py
from  kfp  import dsl

# Define a simple component that adds two numbers
@dsl.component
def  add_numbers(num1: int, num2: int) -> int:
    return num1 + num2 
```

标记有`@dsl.component`装饰器的组件函数`add_numbers`接受两个整数并返回它们的和。

### 3. 抑制特定警告

```py
import  warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
```

此脚本抑制了所有模块的`DeprecationWarning`。

### 4. 链接管道中的组件

```py
from  kfp  import dsl

# Component that generates a fixed number
@dsl.component
def  generate_number() -> int:
    return 42

# Component that doubles the input number
@dsl.component
def  double_number(input_number: int) -> int:
    return input_number * 2

# Define a pipeline that connects two components
@dsl.pipeline(
    name="Number doubling pipeline",
    description="A pipeline that generates a number and doubles it."
)
def  number_doubling_pipeline():
    # Step 1: Generate a number
    generated_number_task = generate_number()

    # Step 2: Double the generated number
    double_number_task = double_number(input_number=generated_number_task.output) 
```

管道由两个组件组成：`generate_number`，它生成一个固定数字，和 `double_number`，它将输入加倍。通过将第一个组件的 `.output` 作为第二个组件的输入来建立连接。

### 5. 编译和准备管道以执行

```py
from  kfp  import compiler

# Assume the pipeline definition is named number_doubling_pipeline
pipeline_func = number_doubling_pipeline

# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=pipeline_func,
    package_path='number_doubling_pipeline.yaml'
) 
```

管道被编译成 `number_doubling_pipeline.yaml` 文件，该文件可以上传并在 Kubeflow 环境中运行。

### 6. 与 `PipelineTask` 对象一起工作

```py
# This is a hypothetical function that cannot be executed as‑is. It is intended to illustrate the concept.
def  handle_pipeline_task():
    # Hypothetical call to a component function named my_component
    # In a real scenario, this should occur inside a pipeline function
    task = my_component(param1="value")

    # Access the component’s output
    # This line is illustrative and typically used to pass outputs between components in a pipeline
    output = task.output

    print("Accessing the component output:", output)

# Note: In real usage, my_component would be defined as a Kubeflow Pipeline component,
# and task manipulations should occur within the context of a pipeline function. 
```

示例显示，调用组件返回一个 `PipelineTask` 对象，其结果通过 `task.output` 访问。在实际操作中，此类对象在管道函数内部进行操作。

### 7. 处理管道定义中的错误

```py
from  kfp  import dsl

# Incorrect pipeline definition
@dsl.pipeline(
    name='Incorrect Pipeline',
    description='An example that attempts to return a PipelineTask object directly.'
)
def  incorrect_pipeline_example():
    @dsl.component
    def  generate_number() -> int:
        return 42

    generated_number_task = generate_number()
    # Incorrect attempt to return a PipelineTask object directly
    return generated_number_task  # This will cause an error

# Correct pipeline definition
@dsl.pipeline(
    name='Correct Pipeline',
    description='A corrected example that does not attempt to return a PipelineTask object.'
)
def  correct_pipeline_example():
    @dsl.component
    def  generate_number() -> int:
        return 42

    generated_number_task = generate_number()
    # Correct approach: do not attempt to return a PipelineTask directly from a pipeline function.
    # A pipeline function should not return anything.

# Explanation: a pipeline function orchestrates steps and data flow, but does not return data directly.
# Attempting to return a PipelineTask from a pipeline function is incorrect, because the pipeline definition
# should describe component structure and dependencies, not process data directly.
# The corrected version removes the return statement, which matches the expected behavior of pipeline functions. 
```

### 8. 自动化模型训练的数据准备

```py
import  json

# Simulated data preparation for model training
def  preprocess_data(input_file_path, output_file_path):
    # Read data from a JSON file
    with open(input_file_path, 'r') as infile:
        data = json.load(infile)

    # Perform a simple transformation: filter data
    # For illustration, assume we only need items meeting a certain condition
    # Example: filter items where the value of "useful" is True
    filtered_data = [item for item in data if item.get("useful", False)]

    # Save the transformed data to another JSON file
    with open(output_file_path, 'w') as outfile:
        json.dump(filtered_data, outfile, indent=4)

# Example usage
preprocess_data('input_data.json', 'processed_data.json')

# Note: This script assumes the file 'input_data.json' exists in the current directory
# and will save processed data to 'processed_data.json'.
# In a real scenario, paths and transformation logic should be adjusted to your requirements. 
```

此脚本演示了一个简单的数据准备过程：从 JSON 文件中读取数据，对其进行转换（通过条件过滤），并将处理后的数据保存到另一个 JSON 文件中。此类任务可以封装在 Kubeflow Pipeline 组件中，以自动化 ML 训练工作流程中的数据准备步骤。

### 9. 在管道中实现模型版本控制

```py
from  datetime  import datetime

def  generate_model_name(base_model_name: str) -> str:
    # Generate a timestamp in the format "YYYYMMDD-HHMMSS"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Append the timestamp to the base model name to create a unique name
    model_name = f"{base_model_name}-{timestamp}"
    return model_name

# Example usage
base_model_name = "my_model"
model_name = generate_model_name(base_model_name)
print("Generated model name:", model_name)

# This function generates a unique model name by appending the current date and time to the base model name.
# This practice helps with model versioning, making it easier to track and manage different model versions in ML operations. 
```

### 10. 参数化和执行 Kubeflow 管道

为了完成此任务，假设我们在一个可以访问 Kubeflow Pipeline 执行 API 的环境中工作。由于执行细节因平台和 API 版本而异，以下脚本是一个基于常见模式的假设示例。

```py
# Assume the necessary imports and configuration for interacting with the execution environment are present

def  submit_pipeline_execution(compiled_pipeline_path: str, pipeline_arguments: dict):
    # Placeholder for the API/SDK method to submit a pipeline for execution
    # In a real scenario, this would use the Kubeflow Pipelines SDK or a cloud provider SDK
    # For example, using the Kubeflow Pipelines SDK or a cloud service like Google Cloud AI Platform Pipelines

    # Assume a function `submit_pipeline_job` exists and can be used to submit
    # This function would be part of the SDK or the environment’s API
    submit_pipeline_job(compiled_pipeline_path, pipeline_arguments)

# Example pipeline arguments
pipeline_arguments = {
    "recipient_name": "Alice"
}

# Path to the compiled Kubeflow pipeline YAML file
compiled_pipeline_path = "path_to_compiled_pipeline.yaml"

# Submit the pipeline for execution
submit_pipeline_execution(compiled_pipeline_path, pipeline_arguments)

# Note: This example assumes a `submit_pipeline_job` function exists, which will be specific
# to the environment’s API or SDK. In a real implementation, replace this placeholder
# with actual code that interacts with the Kubeflow Pipelines API or a managed service API, such as Google Cloud AI Platform. 
```

此脚本描述了如何参数化和提交编译后的 Kubeflow 管道以进行执行，假设有适当的 API 或 SDK 方法可用（在这个假设示例中为 `submit_pipeline_job`）。实际的提交方法取决于您的执行环境或云提供商。
