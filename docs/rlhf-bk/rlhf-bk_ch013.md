# 宪章 AI 与 AI 反馈

来自 AI 反馈的强化学习（RLAIF）是一套更广泛的技术，用于使用 AI 来增强或生成反馈数据，包括成对偏好 [[227]](ch021.xhtml#ref-lee2023rlaif) [[228]](ch021.xhtml#ref-sharma2024critical) [[229]](ch021.xhtml#ref-castricato2024suppressing)。使用 RLAIF 的动机有很多，既可以完全取代人类反馈，也可以增强它。AI 模型比人类便宜得多，单条人类偏好数据成本约为 1 美元或更高（甚至每个提示词高达 10 美元以上），而使用前沿 AI 模型（如 GPT-4o）的 AI 反馈成本不到 0.01 美元。这种成本差异使得 RLHF 方法的实验市场对之前因价格而被排除在外的人群开放。除了价格之外，AI 反馈在性能上引入了与人类反馈不同的*权衡*，这些权衡仍在研究中。AI 反馈的峰值性能至少与基于技能的评价中的人类数据在同一水平，但尚未研究人类数据是否允许在现实世界的产品设置或新的训练方法（如角色训练）中对模型进行更精细的控制。

术语 RLAIF 是在 Anthropic 的工作*宪章 AI：从 AI 反馈中获得无害性* [[19]](ch021.xhtml#ref-bai2022constitutional)中引入的，这导致 AI 社区对方法之间关系产生初步的困惑。自从宪章 AI（CAI）论文发布和 RLAIF 的正式化以来，RLAIF 已成为后训练和 RLHF 文献中的默认方法——例子多得无法轻易列举。应将这种关系理解为 CAI 是启动更广泛 RLAIF 领域的例子。

人类数据与 AI 反馈数据之间差异的一个经验法则是：

1.  人类数据是高噪声和低偏差的，

1.  合成偏好数据是低噪声和高偏差的，

许多学术成果显示，如何在 RLHF 工作流程中用 AI 偏好数据替代，并实现强大的评估分数 [[230]](ch021.xhtml#ref-miranda2024hybrid)，但显示了 RLHF 文献与工业最佳实践之间的分离。

## 宪章 AI

宪章 AI（CAI）的方法，Anthropic 在他们的 Claude 模型中广泛使用，是合成数据用于 RLHF 训练的最早、最大规模的应用。宪章 AI 有两个合成数据的使用：

1.  对指令调整数据的批评，以遵循一系列原则，如“答案是否鼓励暴力”或“答案是否真实”。当模型生成问题的答案时，它会将答案与宪章中的原则列表进行核对，随着时间的推移改进答案。然后，他们在生成的数据集上微调模型。

1.  通过使用语言模型来回答从宪法中随机抽取的原则（类似于这篇关于原则引导奖励模型的论文）的上下文中哪个补全更好，生成成对偏好数据。然后，RLHF 使用合成数据按正常流程进行，因此得名 RLAIF。

大部分情况下，CAI 因上述第二部分，即偏好数据而闻名，但用于指令数据的方法在训练后的一般数据过滤和合成数据生成方法中得到应用。

CAI 可以形式化为以下内容。

通过使用一组由人类编写的原则，他们称之为“宪法”，Bai 等人（2022 年）使用一个单独的 LLM 生成用于微调的人工偏好和指令数据[[19]](ch021.xhtml#ref-bai2022constitutional)。宪法<semantics><mi>𝒞</mi><annotation encoding="application/x-tex">\mathcal{C}</annotation></semantics>是一组书面原则，指示在批判阶段需要关注的特定方面。指令数据是通过反复采样一个原则<semantics><mrow><msub><mi>c</mi><mi>i</mi></msub><mo>∈</mo><mi>𝒞</mi></mrow><annotation encoding="application/x-tex">c_i \in \mathcal{C}</annotation></semantics>并要求模型修改其最新的输出<semantics><msup><mi>y</mi><mi>i</mi></msup><annotation encoding="application/x-tex">y^i</annotation></semantics>以与提示<semantics><mi>x</mi><annotation encoding="application/x-tex">x</annotation></semantics>对齐<semantics><msub><mi>c</mi><mi>i</mi></msub><annotation encoding="application/x-tex">c_i</annotation></semantics>来精心制作的。这产生了一系列指令变体<semantics><mrow><mo stretchy="false" form="prefix">{</mo><msup><mi>y</mi><mn>0</mn></msup><mo>,</mo><msup><mi>y</mi><mn>1</mn></msup><mo>,</mo><mi>⋯</mi><mo>,</mo><msup><mi>y</mi><mi>n</mi></msup><mo stretchy="false" form="postfix">}</mo></mrow><annotation encoding="application/x-tex">\{y⁰, y¹, \cdots, y^n\}</annotation></semantics>，这些变体用于批判。最终数据点是提示<semantics><mi>x</mi><annotation encoding="application/x-tex">x</annotation></semantics>与最终的补全<semantics><msup><mi>y</mi><mi>n</mi></msup><annotation encoding="application/x-tex">y^n</annotation></semantics>，对于某个<semantics><mi>n</mi><annotation encoding="application/x-tex">n</annotation></semantics>。

通过使用来自 <semantics><mi>𝒞</mi><annotation encoding="application/x-tex">\mathcal{C}</annotation></semantics> 的原则子集作为反馈模型的上下文，以类似但更简单的方式构建偏好数据。反馈模型被呈现一个提示 <semantics><mi>x</mi><annotation encoding="application/x-tex">x</annotation></semantics>，一组原则 <semantics><mrow><mo stretchy="false" form="prefix">{</mo><msub><mi>c</mi><mn>0</mn></msub><mo>,</mo><mi>⋯</mi><mo>,</mo><msub><mi>c</mi><mi>n</mi></msub><mo stretchy="false" form="postfix">}</mo></mrow><annotation encoding="application/x-tex">\{c_0, \cdots, c_n\}</annotation></semantics>，以及两个标记为答案（A）和（B）的完成 <semantics><msub><mi>y</mi><mn>0</mn></msub><annotation encoding="application/x-tex">y_0</annotation></semantics> 和 <semantics><msub><mi>y</mi><mn>1</mn></msub><annotation encoding="application/x-tex">y_1</annotation></semantics>，这些答案来自之前的 RLHF 数据集。记录反馈模型输出（A）或（B）的概率作为奖励模型的训练样本

## 用于判断的特定 LLM

随着 RLAIF 方法越来越普遍，许多人都在思考是否应该使用与生成评论或评分相同的模型来生成响应。特别是，所使用的 LLM 作为裁判的校准问题引起了质疑。一些研究表明，LLM 是不一致的评估者 [[231]](ch021.xhtml#ref-wang2023large) 并且更倾向于自己的响应而不是其他模型的响应（称为自我偏好偏差） [[232]](ch021.xhtml#ref-panickssery2024llm)。

因此，许多人都在思考是否应该使用与生成评论或评分相同的模型来生成响应。是否应该训练一个专门用于此目的的独立模型？已经发布了多个模型，旨在作为数据标注工具替代前沿模型，例如评论模型 Shepherd [[233]](ch021.xhtml#ref-wang2023shepherd) 和 CriticLLM [[234]](ch021.xhtml#ref-ke2023critiquellm) 或类似于 Auto-J [[235]](ch021.xhtml#ref-li2023generative)、Prometheus [[138]](ch021.xhtml#ref-kim2023prometheus)、Prometheus 2 [[236]](ch021.xhtml#ref-kim2024prometheus) 或 Prometheus-Vision [[237]](ch021.xhtml#ref-lee2024prometheus) 的评估响应性能的模型，但它们在文档化的训练方案中并未得到广泛应用。有些人发现通过重复采样 [[238]](ch021.xhtml#ref-brown2024large) [[239]](ch021.xhtml#ref-zhao2025sample) [[240]](ch021.xhtml#ref-kalra2025verdict)、自我完善 [[241]](ch021.xhtml#ref-madaan2023self) 或锦标赛排名 [[242]](ch021.xhtml#ref-pace2024west) 可以提供对真实判断或更高质量的偏好对的更好估计。其他校准技术共同进化了模型的生成和判断能力 [[243]](ch021.xhtml#ref-wu2024meta)。

## 进一步阅读

与宪法 AI 相关的研究方向和扩展有很多，但其中很少被记录为 RLHF 和后训练食谱中的明确改进。目前，它们被包括为进一步阅读的内容。

+   OpenAI 发布了模型规范 [[124]](ch021.xhtml#ref-openai2024modelspec)，这是一份说明其模型预期行为的文档，并声明他们正在探索直接引用文档的方法来实现模型的对齐（这可以被视为与 CAI 的紧密伙伴）。OpenAI 继续使用名为 Deliberative Alignment [[244]](ch021.xhtml#ref-guan2024deliberative) 的方法来训练他们的推理模型，如 o1，以在引用这些安全或行为策略的同时对齐模型。

+   Anthropic 继续在模型训练中使用 CAI，更新了 Claude 使用的宪法 [[245]](ch021.xhtml#ref-Anthropic2023ClaudesConstitution) 并实验了种群集体如何就模型原则达成共识以及这如何改变模型行为 [[246]](ch021.xhtml#ref-ganguli2023)。

+   开源社区已经探索了将 CAI 应用于开放数据集的复制 [[247]](ch021.xhtml#ref-Huang2024cai) 以及探索在 LM 之间创建对话数据的方法 [[248]](ch021.xhtml#ref-lambert2024self)。

+   其他工作使用了基于原则的偏好或反馈以及不同的优化方法。[[249]](ch021.xhtml#ref-sun2023principledriven) 使用原则作为奖励模型的上下文，该模型用于训练 Dromedary 模型 [[250]](ch021.xhtml#ref-sun2024salmon)。[[37]](ch021.xhtml#ref-glaese2022improving) 使用原则来提高 RLHF 过程中人类判断的准确性。[[251]](ch021.xhtml#ref-liu2025inference) 训练一个奖励模型在推理时生成自己的原则，并使用这些原则来提供最终评分。[[252]](ch021.xhtml#ref-franken2024self) 将遵循原则表述为一个预训练模型可以无标签学习的互信息最大化问题。
