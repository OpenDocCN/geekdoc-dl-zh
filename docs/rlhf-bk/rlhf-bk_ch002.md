# 关键相关作品

RLHF 及其相关方法非常新颖。我们强调历史，以展示这些程序是如何最近才形式化的，以及学术文献中有多少这样的文档。因此，我们想强调 RLHF 发展非常迅速，所以本章为将表达对某些方法的不确定性以及预期某些细节会在几个核心实践中发生变化的书奠定了基础。否则，这里列出的论文和方法展示了为什么 RLHF 管道中的许多部分是这样的，因为一些开创性的论文是为与现代语言模型完全不同的应用而写的。

在本章中，我们详细介绍了将 RLHF 领域带到今天的关键论文和项目。这并不是对 RLHF 及其相关领域的全面回顾，而是一个起点和重述，说明了我们是如何到达今天的。它有意关注了导致 ChatGPT 的近期工作。在 RL 文献中，关于从偏好中学习有大量的进一步工作 [[26]](ch021.xhtml#ref-wirth2017survey)。要获得更详尽的列表，你应该使用一篇合适的调查论文 [[27]](ch021.xhtml#ref-kaufmann2023survey), [[28]](ch021.xhtml#ref-casper2023open)。

## 2018 年之前的起源：基于偏好的 RL

随着深度强化学习的兴起，该领域最近变得流行，并扩展为对许多大型科技公司中 LLMs 应用更广泛的研究。然而，今天使用的许多技术都与早期关于 RL 的文献中的核心技术密切相关。

第一篇采用类似现代 RLHF 方法的论文是 *TAMER*。*TAMER：通过评估强化手动训练一个智能体* 提出了一种方法，其中人类通过迭代评分智能体的动作来学习一个奖励模型，该模型用于学习动作策略 [[29]](ch021.xhtml#ref-knox2008tamer)。其他同时或随后提出的工作提出了一个演员-评论家算法 COACH，其中人类反馈（正面和负面）用于调整优势函数 [[30]](ch021.xhtml#ref-macglashan2017interactive)。

主要参考文献，Christiano 等人 2017 年的论文，是 RLHF 应用于 Atari 游戏内代理轨迹之间的偏好[[1]](ch021.xhtml#ref-christiano2017deep)的应用。这项引入 RLHF 的工作紧随 DeepMind 在强化学习上的开创性工作——深度 Q 网络（DQN），该工作表明 RL 代理可以从头开始学习解决流行的视频游戏。这项工作表明，在某些领域，人类在轨迹之间进行选择可能比直接与环境交互更有效。这使用了一些巧妙条件，但仍然令人印象深刻。这种方法通过更直接的奖励建模[[31]](ch021.xhtml#ref-ibarz2018reward)和早期 RLHF 工作中深度学习的采用得到了扩展，仅一年后，通过将神经网络模型扩展到 TAMER，就达到了顶峰[[32]](ch021.xhtml#ref-warnell2018deep)。

这个时代开始转变，因为作为一般概念的奖励模型被提出作为一种研究对齐的方法，而不仅仅是解决 RL 问题的工具[[33]](ch021.xhtml#ref-leike2018scalable)。

## 2019 至 2022 年：语言模型上的从人类偏好进行强化学习

从人类反馈中进行强化学习，也常在其早期被称为从人类偏好中进行强化学习，很快被转向扩展大型语言模型的 AI 实验室所采纳。这部分工作的大部分始于 2018 年的 GPT-2 和 2020 年的 GPT-3 之间。2019 年的早期工作《从人类偏好微调语言模型》与 RLHF 的现代工作以及本书中将要涉及的内容有许多显著相似之处[[34]](ch021.xhtml#ref-ziegler2019fine)。许多经典术语，如学习奖励模型、KL 距离、反馈图等，都在这篇论文中得到了形式化——只是最终模型的评估任务和能力与今天人们所做的工作有所不同。从这里，RLHF 被应用于各种任务。重要例子包括通用摘要[[2]](ch021.xhtml#ref-stiennon2020learning)、书籍的递归摘要[[35]](ch021.xhtml#ref-wu2021recursively)、指令遵循（InstructGPT）[[3]](ch021.xhtml#ref-ouyang2022training)、浏览器辅助问答（WebGPT）[[4]](ch021.xhtml#ref-nakano2021webgpt)、支持带有引用的答案（GopherCite）[[36]](ch021.xhtml#ref-menick2022teaching)以及通用对话（Sparrow）[[37]](ch021.xhtml#ref-glaese2022improving)。

除了应用之外，许多开创性的论文定义了 RLHF 未来的关键领域，包括以下内容：

1.  奖励模型过度优化[[38]](ch021.xhtml#ref-gao2023scaling)：RL 优化器对偏好数据训练的模型过度拟合的能力，

1.  语言模型作为对齐的一般研究领域[[18]](ch021.xhtml#ref-askell2021general)，以及

1.  红队测试[[39]](ch021.xhtml#ref-ganguli2022red)——评估语言模型安全性的过程。

工作继续在改进 RLHF 以应用于聊天模型方面进行。Anthropic 继续在 Claude 的早期版本[[5]](ch021.xhtml#ref-bai2022training)和早期的 RLHF 开源工具[[40]](ch021.xhtml#ref-ramamurthy2022reinforcement),[[41]](ch021.xhtml#ref-havrilla-etal-2023-trlx),[[42]](ch021.xhtml#ref-vonwerra2022trl)中广泛使用。

## 2023 年至今：ChatGPT 时代

ChatGPT 的发布非常明确地阐述了 RLHF 在其训练中的角色[[43]](ch021.xhtml#ref-openai2022chatgpt)：

> 我们使用人类反馈强化学习（RLHF）训练了这个模型，使用与 InstructGPT 相同的方法，但在数据收集设置上略有不同。

从那时起，RLHF 在领先的语言模型中被广泛使用。它被广泛认知为在 Anthropic 的 Constitutional AI 用于 Claude [[19]](ch021.xhtml#ref-bai2022constitutional)，Meta 的 Llama 2 [[44]](ch021.xhtml#ref-touvron2023llama)和 Llama 3 [[24]](ch021.xhtml#ref-dubey2024llama)，Nvidia 的 Nemotron [[25]](ch021.xhtml#ref-adler2024nemotron)，Ai2 的 Tülu 3 [[6]](ch021.xhtml#ref-lambert2024t)，以及更多模型中使用。

现在，RLHF 正在发展成为更广泛的偏好微调（PreFT）领域，包括新的应用，如中间推理步骤的进程奖励[[45]](ch021.xhtml#ref-lightman2023let)，在第七章中介绍；受直接偏好优化（DPO）启发的直接对齐算法[[20]](ch021.xhtml#ref-rafailov2024direct)，在第十二章中介绍；从代码或数学的执行反馈中学习[[46]](ch021.xhtml#ref-kumar2024training),[[47]](ch021.xhtml#ref-singh2023beyond)以及受 OpenAI 的 o1 [[48]](ch021.xhtml#ref-openai2024o1)启发的其他在线推理方法，在第十四章中介绍。
