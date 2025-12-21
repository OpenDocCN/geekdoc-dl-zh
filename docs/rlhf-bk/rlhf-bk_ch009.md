# 指令微调

早期的大规模预训练语言模型是用下一个标记预测目标进行训练的，并且默认情况下没有提供遵循指令的显式接口。在 GPT-3[[167]](ch021.xhtml#ref-brown2020language)发布时，提示和情境学习成为了一种广泛使用的方法，通过展示情境中的示例并要求模型完成类似任务，来适应单个模型到许多任务（尽管特定任务的微调仍然很常见）。一个实际的下一步是指令微调，它教会模型以指令-响应格式进行响应，而不仅仅是继续文本。

指令微调的兴起得益于两条工作线的交汇。首先，自然语言处理（NLP）从定制的微调任务设置转向统一的“文本到文本”或指令框架，这使得标准化多样化的数据集并在许多任务上训练单个模型变得简单。统一任务框架的突出例子包括*使用统一的文本到文本转换器探索迁移学习的极限*（T5 模型）[[168]](ch021.xhtml#ref-raffel2020exploring)，*微调语言模型是零样本学习者*（FLAN 数据集）[[169]](ch021.xhtml#ref-wei2021finetuned)，*多任务提示训练实现零样本任务泛化*（T0 模型）[[170]](ch021.xhtml#ref-sanh2021multitask)，以及*通过自然语言众包指令实现跨任务泛化*（Natural Instructions 数据集）[[171]](ch021.xhtml#ref-mishra2021cross)。其次，预训练语言模型的扩展和提示/情境学习的兴起表明，单个模型可以跨任务泛化，但当模型明确地训练在指令-响应示例上时，泛化变得更加可靠。这些趋势共同导致了在大量指令集合上微调预训练语言模型的时代——现在通常称为指令微调（IFT）或监督微调（SFT），其中训练通用模型变得对更广泛的受众可访问。

自从被发现以来，指令微调（也俗称*指令调整*）已经成熟，并且成为许多语言建模管道的标准实践。在其核心，指令微调（IFT）是适应语言模型到期望的任务分布的最简单方法。它作为 RLHF 的基础，为模型准备了一种称为问答的指令格式，并且是那些试图将现代技术应用于新领域的人使用的第一个工具。如果没有基本的指令遵循能力，本书中讨论的大多数管道——从偏好数据收集到在线 RLHF 优化——都无法执行。

## 聊天模板和指令结构

训练后过程的开始是定义一个格式化用户查询的模式，以便它们可以被通过标记器处理信息的语言模型轻松读取。当使用预训练的语言模型时，提示非常简单，模型只知道几个标记：一个序列开始标记（例如，`<bos_token>`），一个序列结束标记（例如，`<eos_token>`），以及一个填充标记（用于管理空组件的训练批次）。这意味着，为了提示基础模型，用户输入一个标记序列，模型可以从中继续，例如：

```py
<bos_token> The capital of the United States is
```

然后，模型将生成标记，直到其上下文窗口耗尽，或者生成序列结束标记。

所有训练后的阶段，从指令调整到 RLHF 和其他方法，都依赖于这种格式来训练模型。处理与用户交互结构的工具被称为**聊天模板**。

下面是一个我们将要分解的例子：

```py
{% if messages[0]['role'] == 'system' %}
    {# If the conversation begins with a system message, treat it as a special first turn.
       We set an offset so the user/assistant alternation check lines up correctly. #}
    {% set offset = 1 %}
{% else %}
    {# No system message: user should be the first non-empty turn. #}
    {% set offset = 0 %}
{% endif %}

{# Emit the beginning-of-sequence token (model-specific). #}
{{ bos_token }}

{# Serialize each message into the model's chat-markup tokens. #}
{% for message in messages %}
    {# Enforce role alternation: (system), user, assistant, user, assistant, ...
       The boolean expression compares "is this a user message?" against whether the
       current index (plus offset) is expected to be user or assistant. #}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {# Wrap each message with special tokens:
       - <|im_start|><role>\n
       - message content (trimmed)
       - <|im_end|>\n
       This produces a single flat token sequence the LM can train on. #}
    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
{% endfor %}

{# Optionally append an "assistant" start tag with no content.
   This cues generation to continue from the assistant role. #}
{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
```

这是将包含消息和角色的字典列表转换为语言模型可以预测的标记的原始代码。

所有传递给模型的都是分配了一个角色。传统的三个角色是`system`、`user`和`assistant`。

`system`标签仅用于对话的第一个消息；它包含不会从用户那里接收或暴露给用户的文本指令。这些**系统提示**用于向模型提供额外的上下文，例如日期和时间，或修正行为。作为一个有趣的例子，模型可以被告知诸如“你是一个友好的聊天机器人，总是以海盗的风格回应。”之类的事情。

接下来，其他两个角色很简单：**user**包含使用 AI 的人的消息，而**assistant**包含模型的响应（即作为 AI 助手参与）。

为了将所有这些信息转换为标记，我们使用上面列出的代码。该模型有一系列*特殊标记*，用于将各种消息彼此分开。如果我们用示例查询“一个人一次能吃多少直升机？”运行上述代码，传递给模型的标记序列将如下所示：

```py
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
```

注意序列中的最后一个标记是`<|im_start|>assistant`。这就是模型知道继续生成标记，直到最终生成其序列结束标记的方式，在这种情况下是`<|im_end|>`。

通过将所有问答对数据（以及下游偏好调整数据）打包成这种格式，现代语言模型会以完美的一致性遵循它。这是指令调整模型用于与用户和存储在 GPU 或其他计算设备上的模型交换信息的语言。

这种行为可以天真地扩展到多个回合，如下所示：

```py
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
Oh just 6.<|im_end|>
<|im_start|>user
Are you sure about that?<|im_end|>
<|im_start|>assistant
```

在开源生态系统中，将聊天模板应用于消息列表的标准方法是保存为 `apply_chat_template` 的 Jinja 代码片段，保存在分词器中。

上述聊天模板是 OpenAI 的聊天标记语言（ChatML）的衍生品，ChatML 是早期尝试标准化消息格式的尝试。现在，OpenAI 和其他模型提供商使用一种分层系统，用户可以配置系统消息，但可能或可能不会向用户揭示更高级别的指令 [[172]](ch021.xhtml#ref-wallace2024instruction)。

存在许多其他聊天模板。一些其他例子包括 Zephyr 的 [[21]](ch021.xhtml#ref-tunstall2023zephyr)：

```py
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
```

或者 Tülu 的：

```py
<|user|>
How are you doing?
<|assistant|>
I'm just a computer program, so I don't have feelings, but I'm functioning as expected. How can I assist you today?<|endoftext|>
```

此外，许多聊天模板包括用于工具使用等任务的格式化和其他标记。

## 指令调整的最佳实践

指令调整作为后训练和创建有帮助的语言模型的基础已被确立。有许多方法可以实现成功的指令调整。例如，通过量化一些模型参数进行高效的微调，使得训练变得非常容易 [[173]](ch021.xhtml#ref-dettmers2023qlora)。此外，在狭窄领域，例如聊天对齐，即没有像数学或代码这样的更难技能的情况下，小型、专注的数据集可以实现强大的性能 [[13]](ch021.xhtml#ref-zhou2023lima)。

在 ChatGPT 发布后不久，像 No Robots 这样的只有 10K 个样本的人类数据集是最先进的 [[174]](ch021.xhtml#ref-no_robots)。几年后，大规模合成数据集在大多数任务上表现最佳 [[6]](ch021.xhtml#ref-lambert2024t)。

一些原则仍然存在：

+   高质量数据是性能的关键。完成项是模型实际学习的内容（在许多情况下，提示没有被预测，因此模型没有学习预测提示）。

+   ~1M 个提示可以用来创建一个能够进行优秀 RLHF 和后训练的模型。进一步的扩展仍然有帮助，但回报迅速减少。

+   最好的提示是与感兴趣的下游任务相似分布的提示。

+   如果在指令调整之后进行多个训练阶段，模型可以从过程中的某些噪声中恢复。优化整体优化比每个单独的阶段更重要。

## 实现

虽然损失函数与预训练相同，但有一些关键的实施细节与用于预训练的设置不同。许多实践，例如决定用于将模型分片到多个 GPU 的并行类型，与预训练相同，只是使用的机器总数通常较低（对于下面列出的第一个技术变化）：

+   **更小的批次大小**：与预训练相比，指令微调（以及其他后训练技术，如偏好微调）使用显著更小的批次大小。例如，OLMo 2 在 7B 预训练中使用 1024 个序列的批次大小，在 13B 预训练中使用 2048 个序列，而在后训练阶段两者都只使用 256 个序列的批次大小 [[59]](ch021.xhtml#ref-olmo20242)。较小的批次大小意味着这些训练作业不能像预训练那样跨那么多设备进行分片 - 实际上，分布式训练设置中每个设备的批次大小有最小限制，因此如果您尝试保留较小的全局批次大小进行 SFT，可以使用累积更少的 GPU。实际上，由于 SFT 的训练令牌计数远小于预训练，并且后训练需要为多个种子进行训练以获得最佳最终性能，因此强制批次大小以减少每个训练作业的并发 GPU 分配并不是一个限制因素。

+   **提示掩码**：在预训练时，批次中的每个令牌都会进行自回归预测，然后对它们应用损失。对于指令微调，提示令牌被掩码，因此模型不会学习准确预测用户查询 - 只预测响应。这同样适用于其他后训练算法。

+   **多轮掩码**：对于多轮对话，有两种常见的掩码选择。（1）*仅最后轮次*：只有最后助手轮次的令牌包含在损失中，而所有早期上下文（包括早期助手轮次）都被掩码。长对话仍然可以“展开”成多个训练样本：对于 <semantics><mi>N</mi><annotation encoding="application/x-tex">N</annotation></semantics> 轮次的对话，每个示例预测一个助手响应，同时掩码所有先前上下文（并排除任何后续轮次）。（2）*仅掩码用户轮次*：所有用户轮次都被掩码，但*每个*助手轮次都包含在损失中。如果您想要更多（更短）的训练示例，您仍然可以在此设置中展开，但关键区别在于直接训练中间助手回复。

+   **与预训练相同的损失函数**：指令微调使用与预训练语言模型中使用的相同自回归损失函数，但数据集和掩码（仅在完整序列上训练，而预训练文档可以跨批次分割）等方面有显著不同。
