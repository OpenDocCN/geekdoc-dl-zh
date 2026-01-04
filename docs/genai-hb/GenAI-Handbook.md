

# 生成式人工智能手册

> 来源：[`genai-handbook.github.io/`](https://genai-handbook.github.io/)

威廉·布朗

[@willccbb](https://x.com/willccbb) | [willcb.com](https://willcb.com)

[v0.1](https://github.com/genai-handbook/genai-handbook.github.io)（2024 年 6 月 5 日）

# 简介

本文档旨在作为学习现代人工智能系统关键概念的指南。鉴于人工智能近期发展的速度，确实没有一本好的教科书式资料可以用来了解最新的 LLMs 或其他生成模型创新，然而，关于这些主题的优质解释资源（博客文章、视频等）散布在互联网的各个角落。我的目标是把这些“最佳”资源组织成教科书式的展示，这可以作为实现个人人工智能相关学习目标的路线图。我希望这将成为一份“活文档”，随着新创新和范例不可避免地出现而更新，并且理想情况下也能从社区输入和贡献中受益。本指南面向那些有一定技术背景、出于好奇心或潜在职业兴趣而想要深入研究人工智能的人。我将假设您有一些编程和高中水平数学的经验，但否则将提供填补任何其他先决条件的指南。如果您认为应该添加任何内容，请告诉我！

## 人工智能领域

截至 2024 年 6 月，自[ChatGPT](http://chat.openai.com)由[OpenAI](https://openai.com/)发布以来，已经有大约 18 个月的时间，世界开始更多地讨论人工智能。自那时以来发生了很多事情：像[Meta](https://llama.meta.com/)和[Google](https://gemini.google.com/)这样的科技巨头发布了他们自己的大型语言模型，像[Mistral](https://mistral.ai/)和[Anthropic](https://www.anthropic.com/)这样的新组织也证明他们是严肃的竞争者，无数初创公司开始基于他们的 API 构建，每个人都[争相](https://finance.yahoo.com/news/customer-demand-nvidia-chips-far-013826675.html)寻找强大的 Nvidia GPU，论文以惊人的速度出现在[ArXiv](https://arxiv.org/list/cs.AI/recent)上，由 LLM 驱动的[物理机器人](https://www.figure.ai/)和[人工程序员](https://www.cognition-labs.com/introducing-devin)的演示在社交媒体上引起了人们的关注，而且似乎[聊天机器人](https://www.businessinsider.com/chat-gpt-effect-will-likely-mean-more-ai-chatbots-apps-2023-2)正在以不同的成功程度进入在线生活的各个方面。在 LLM 竞赛的同时，通过扩散模型进行图像生成的技术也得到了快速发展；[DALL-E](https://openai.com/dall-e-3)和[Midjourney](https://www.midjourney.com/showcase)展示的结果越来越令人印象深刻，常常让社交媒体上的人类感到困惑，随着[Sora](https://openai.com/sora)、[Runway](https://runwayml.com/)和[Pika](https://pika.art/home)的进步，高质量视频生成似乎也即将到来。关于“AGI”何时到来，AGI 究竟是什么，开放模型与封闭模型的优点，价值对齐，超级智能，存在风险，虚假新闻以及经济未来的争论正在持续。许多人担心自动化会导致失业，或者对自动化可能带来的进步感到兴奋。而世界仍在前进：芯片速度更快，数据中心更大，模型更智能，上下文更长，能力通过工具和视觉得到增强，而且并不完全清楚这一切将走向何方。如果你在 2024 年关注“AI 新闻”，常常会感觉几乎每天都有某种重大新突破发生。这需要跟上，尤其是如果你是刚开始关注的话。

由于进步发生得如此之快，那些寻求“加入行动”的人自然倾向于选择最新的最佳工具（根据写作时的信息，可能是 [GPT-4o](https://openai.com/index/hello-gpt-4o/)，[Gemini 1.5 Pro](https://deepmind.google/technologies/gemini/pro/)，或 [Claude 3 Opus](https://www.anthropic.com/news/claude-3-family)，具体取决于你询问的对象）并尝试在这些工具之上构建网站或应用程序。当然，这里有很大的发展空间，但这些工具会迅速变化，对底层基础有扎实的理解将使你更容易充分利用你的工具，快速掌握新工具，并评估成本、性能、速度、模块化和灵活性等方面的权衡。此外，创新不仅仅发生在应用层，像 [Hugging Face](https://huggingface.co/)，[Scale AI](https://scale.com/) 和 [Together AI](https://www.together.ai/) 这样的公司通过专注于开放权重模型的推理、训练和工具（以及其他方面）已经获得了立足点。无论你是想参与开源开发，从事基础研究，还是在成本或隐私问题排除外部 API 使用的环境中利用 LLM，了解这些事物的工作原理以供调试或修改是很有帮助的。从更广泛的职业角度来看，许多当前的“AI/ML 工程师”职位将重视实际知识，除了高级框架之外，就像“数据科学家”职位通常寻求对理论和基础知识的深刻理解，而不仅仅是 ML 框架 *du jour* 的熟练度。深入研究是一条更难的路径，但我认为这是值得的。但是，考虑到过去几年创新发生的速度，你应该从哪里开始？哪些主题是必不可少的，你应该按什么顺序学习它们，哪些可以略读或跳过？

## 内容景观

教科书对于提供“关键思想”较为稳定的领域的宏观路线图非常有用，但据我所知，目前并没有一本公开可用的、具有教科书式全面性和组织的 ChatGPT 后“人工智能指南”。现在写一本涵盖当前人工智能状态的常规教科书似乎也没有太多意义；许多关键思想（例如 QLoRA、DPO、vLLM）不过一年左右，而到印刷时，该领域可能已经发生了巨大的变化。《深度学习》（Goodfellow 等人著）这本书已经近十年历史，对通过 RNNs 的语言建模只是简略提及。较新的《深入浅出深度学习》（http://d2l.ai）一书涵盖了 Transformer 架构和 BERT 模型的微调，但像 RLHF 和 RAG（按照一些更前沿主题的标准来说已经“过时”）这样的主题并未涉及。《“动手实践大型语言模型”》（https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/）这本书可能不错，但它尚未正式出版（目前在线上付费墙后提供），而且预计出版时也不会免费。如果你是斯坦福大学的学生，那么 [CS224n](https://web.stanford.edu/class/cs224n/index.html#coursework) 课程看起来很棒，但如果没有登录，你只能访问幻灯片和主要由密集学术论文组成的阅读清单。微软的《“初学者生成式人工智能指南”》（https://microsoft.github.io/generative-ai-for-beginners/#/）对于使用流行框架进行实践相当可靠，但它更侧重于应用而不是理解基础原理。

我所知道的与心中所想最接近的资源是 Maxime Labonne 在 Github 上的 [LLM 课程](https://github.com/mlabonne/llm-course)。该课程包含许多交互式代码笔记本，以及学习底层概念的资源链接，其中一些与我在这里将要包括的内容有重叠。我建议将其作为阅读此手册时的主要辅助指南，尤其是如果你对应用感兴趣；这份文档不包括笔记本，但我所涵盖的主题范围更广，包括一些不太“标准”的研究线索，以及多模态模型。

尽管如此，还有大量其他高质量且易于获取的内容涵盖了 AI 的最新进展——只是没有很好地组织起来。快速了解新创新的最佳资源通常是单独的博客文章或 YouTube 视频（以及 Twitter/X 帖子、Discord 服务器和 Reddit、LessWrong 上的讨论）。我的目标是提供一份路线图，以导航所有这些内容，以教科书式的展示方式组织，而不在个别解释器上重新发明轮子。在整个过程中，我将尽可能包括多种内容形式（例如视频、博客和论文），以及我对目标相关知识优先级的看法和关于“心智模型”的笔记，这些模型在我首次接触这些主题时发现很有用。

我创建这份文档**不是**作为一个“生成式 AI 专家”，而是一个最近在短时间内快速学习了许多这些主题的人。虽然我从 2016 年左右就开始在 AI 领域工作（如果我们把为视觉模型进行评估的实习项目算作“开始”），但我直到 18 个月前才开始密切关注 LLM 的发展，随着 ChatGPT 的发布。我大约在 12 个月前开始使用开放权重 LLM。因此，我花了很多时间在过去一年里筛选博客文章、论文和视频，寻找精华；这份文档希望是那条路径的一个更直接的版本。它也作为我许多与朋友交谈的总结，我们试图找到并分享有用的直觉，以便加快彼此的学习。整理这份资料也极大地促进了我对自身理解的补充；直到几周前，我还不了解 FlashAttention 是如何工作的，而且我仍然认为自己对状态空间模型的理解还不够深入。但我知道的比开始时多了很多。

## 资源

我们将从中汲取的一些资料包括：

+   博客：

    +   [Hugging Face](https://huggingface.co/blog)博客文章

    +   [Chip Huyen](https://huyenchip.com/blog/)的博客

    +   [Lilian Weng](https://lilianweng.github.io/)的博客

    +   [Tim Dettmers](https://timdettmers.com/)的博客

    +   [走向数据科学](https://towardsdatascience.com/)

    +   [Andrej Karpathy](https://karpathy.github.io/)的博客

    +   Sebastian Raschka 的[“AI 前沿”](https://magazine.sebastianraschka.com/)博客

+   YouTube：

    +   Andrej Karpathy 的[“从零到英雄”](https://karpathy.ai/zero-to-hero.html)视频

    +   [3Blue1Brown](https://www.youtube.com/c/3blue1brown)视频

    +   互信息

    +   StatQuest

+   教科书

    +   [d2l.ai](http://d2l.ai)互动教科书

    +   [深度学习](https://www.deeplearningbook.org/)教科书

+   网络课程：

    +   Maxime Labonne 的[LLM 课程](https://github.com/mlabonne/llm-course)

    +   微软的[“生成式 AI 入门”](https://microsoft.github.io/generative-ai-for-beginners/#/)

    +   Fast.AI 的[“为程序员设计的实用深度学习”](https://course.fast.ai/)

+   各式各样的大学讲义

+   原始研究论文（较少使用）

我会经常引用原始论文中的关键思想，但我们的重点将放在更简洁、更概念化的解释性内容上，面向学生或从业者，而不是经验丰富的 AI 研究人员（尽管希望随着你通过这些资源的学习，进行 AI 研究的可能性会变得不那么令人畏惧）。在可能的情况下，将提供多个资源和媒体格式的指针，并对其相对优点进行一些讨论。

## 前期准备

### 数学

如果你想要理解现代深度学习，微积分和线性代数几乎是不可避免的，因为深度学习主要是由矩阵乘法和梯度回传驱动的。许多技术人员在多元微积分或初等线性代数结束他们的正式数学教育，并且似乎很常见的是，人们会因不得不记忆一系列不直观的恒等式或手动求逆矩阵而留下苦涩的印象，这可能会让人对进一步深化数学教育的前景感到气馁。幸运的是，我们不需要自己进行这些计算——编程库会为我们处理——而且更重要的是，我们需要对以下概念有实际了解：

+   梯度和它们与局部极小值/极大值的关系

+   微分链式法则

+   矩阵作为向量的线性变换

+   基础概念，如基/秩/张量积/独立性等。

优秀的可视化确实可以帮助这些概念深入人心，我认为没有比这两个来自[3Blue1Brown](https://www.youtube.com/@3blue1brown/playlists)的 YouTube 系列更好的资源了：

+   [微积分的本质](https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr&pp=iAQB)

+   [线性代数的本质](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&pp=iAQB)

如果你的数学基础薄弱，我当然会鼓励你在深入研究之前（重新）观看这些视频。为了测试你的理解，或者作为我们即将前往的方向的预览，该频道上较短的[神经网络](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)视频系列也非常出色，该系列最新几集对语言建模中的 Transformer 网络进行了很好的概述。

水 loo 大学提供的这些[讲义](https://links.uwaterloo.ca/math227docs/set4.pdf)涵盖了与优化相关的多元微积分的一些有用内容，Sheldon Axler 的《“线性代数正确完成”》（[“Linear Algebra Done Right”](https://linear.axler.net/LADR4e.pdf)）是线性代数的一个很好的参考文本。[“凸优化”](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)由 Boyd 和 Vandenberghe 所著，展示了这些主题如何为机器学习中遇到的优化问题奠定基础，但请注意，它相当技术性，如果你主要对应用感兴趣，可能不是必需的。

线性规划当然值得理解，它基本上是你将遇到的最简单的高维优化问题（但仍然非常实用）；这个[视频](https://www.youtube.com/watch?v=E72DWgKP_1Y)应该会给你大部分的核心思想，如果你想要更深入地了解数学，Ryan O’Donnell 的[视频](https://www.youtube.com/watch?v=DYAIIUuAaGA&list=PLm3J0oaFux3ZYpFLwwrlv_EHH9wtH6pnX&index=66)（系列中的 17a-19c，取决于你想要深入到什么程度）是极好的。Tim Roughgarden 的这些讲座（[#10](https://www.youtube.com/watch?v=v-chgwlwqTk), [#11](https://www.youtube.com/watch?v=IWzErYm8Lyk)）也展示了线性规划与我们将要探讨的“在线学习”方法之间的某些迷人联系，这将为生成对抗网络（以及其他许多事物）的概念基础形成。

### 编程

如今，大多数机器学习代码都是用 Python 编写的，这里的一些参考资料将包括用于说明讨论主题的 Python 示例。如果你对 Python 不熟悉，或者对编程一无所知，我听说 Replit 的[100 天 Python 课程](https://replit.com/learn/100-days-of-python)对于入门来说是个不错的选择。一些系统级主题也会涉及到 C++或 CUDA 的实现——我承认我对这两者都不是很精通，并将更多地关注可以通过 Python 库访问的高级抽象，但无论如何，我仍会在相关部分包含这些语言的潜在有用参考资料。

## 组织

本文档分为几个部分和章节，如下所示，并在侧边栏中列出。我们鼓励你跳转到对你个人学习目标最有用的部分。总的来说，我建议首先快速浏览许多链接资源，而不是逐字阅读（或观看）。这应该至少能让你对任何特定学习目标的依赖性方面的知识差距有一个感觉，这将有助于指导更专注的第二遍阅读。

# 第一部分：顺序预测的基础

**目标：** 回顾机器学习基础知识 + 调查“顺序预测”范畴下的（非深度学习）方法。

在本节中，我们将快速概述统计预测和强化学习的经典主题，我们将在后续章节中直接引用这些主题，同时突出一些我认为对于理解大型语言模型（LLMs）非常有用的概念模型，而这些模型通常被深度学习速成课程所忽略——特别是时间序列分析、后悔最小化和马尔可夫模型。

## 统计预测与监督学习

在深入研究深度学习和大型语言模型之前，对概率理论和机器学习的一些基础概念有一个扎实的掌握将非常有用。特别是，了解以下内容很有帮助：

+   随机变量、期望和方差

+   监督学习与无监督学习

+   回归与分类

+   线性模型和正则化

+   实验风险最小化

+   假设类和偏差-方差权衡

对于一般概率理论，对中心极限定理如何工作的扎实理解可能是你在处理我们稍后将要讨论的一些主题之前需要了解多少随机变量的一个合理试金石。这个由 3Blue1Brown 制作的精美动画[视频](https://www.youtube.com/watch?v=zeJD6dqJ5lo)是一个很好的起点，如果你愿意，该频道上还有几个其他很好的概率视频可以查看。UBC 的这套[课程笔记](https://blogs.ubc.ca/math105/discrete-random-variables/)涵盖了随机变量的基础知识。如果你喜欢黑板讲座，我非常喜欢 YouTube 上 Ryan O’Donnell 的 CMU 课程，特别是关于随机变量和中心极限定理的[视频](https://www.youtube.com/watch?v=r9S2fMQiP2E&list=PLm3J0oaFux3ZYpFLwwrlv_EHH9wtH6pnX&index=13)（来自优秀的“CS 理论工具包”课程）是一个很好的概述。

为了理解线性模型和其他关键机器学习原理，Hastie 的《统计学习基础》（[Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf)）的前两章（“引言”和“监督学习概述”）应该足以开始学习。

一旦你熟悉了基础知识，Twitter/X 用户[@ryxcommar](https://twitter.com/ryxcommar)的这篇[博客文章](https://ryxcommar.com/2019/09/06/some-things-you-maybe-didnt-know-about-linear-regression/)就很好地讨论了一些与线性回归相关的常见陷阱和误解。[StatQuest](https://www.youtube.com/@statquest/playlists)在 YouTube 上有一些可能也很有帮助的视频。

机器学习的介绍往往强调线性模型，这有很好的理由。现实世界中的许多现象都可以用线性方程很好地建模——过去 7 天的平均气温很可能是对明天气温的一个合理猜测，除非有关于天气模式预测的其他信息。与非线性对应物相比，线性系统和模型更容易研究、解释和优化。对于具有特征之间潜在非线性依赖关系的更复杂和高维问题，通常很有用去问：

+   对于这个问题，线性模型是什么？

+   为什么线性模型会失败？

+   在给定问题的语义结构下，最好的添加非线性方法是什么？

尤其是这种框架将有助于激发我们稍后将要探讨的一些模型架构（例如 LSTMs 和 Transformers）。

## 时间序列分析

为了理解更复杂的生成式 AI 方法的机制，你需要了解多少关于时间序列分析的知识？

**简短回答：**对于 LLM 来说，只需要一点点，而对于扩散模型来说，则需要更多。

对于基于现代 Transformer 的 LLM，了解以下内容将很有用：

+   序列预测问题的基本设置

+   自回归模型的概念

在脑海中“可视化”一个多亿参数模型的全套机制实际上并没有一个连贯的方法，但更简单的自回归模型（如 ARIMA）可以作为很好的心理模型来外推。

当我们到达神经状态空间模型时，对线性时不变系统和控制理论（它们与经典时间序列分析有许多联系）的工作知识将有助于直觉，但扩散确实是深入到随机微分方程以获得全面了解的地方。但我们可以暂时搁置这个问题。

来自 Towards Data Science 的这篇博客文章([使用随机模型进行预测](https://towardsdatascience.com/forecasting-with-stochastic-models-abf2e85c9679))简明扼要，介绍了基本概念，以及一些标准的自回归模型和代码示例。

如果你想要对数学有更深入的了解，那么来自 UAlberta“时间序列分析”课程的这套[笔记](https://sites.ualberta.ca/~kashlak/data/stat479.pdf)很不错。

## 在线学习和遗憾最小化

对于是否需要牢固掌握遗憾最小化的重要性，存在争议，但我认为基本的熟悉度是有用的。这里的设置与监督学习类似，但：

+   点以任意顺序逐个到达

+   我们希望在整个序列中保持低平均误差

如果你眯起眼睛并倾斜你的头部，大多数为这些问题设计的算法基本上看起来像梯度下降，通常需要精心选择正则化器和学习率，以便数学计算能够顺利进行。但这里有很多令人满意的数学。我对它有很深的感情，因为它与我博士期间研究的大多数研究相关。我认为它从概念上来说是迷人的。就像之前关于时间序列分析的部分一样，在线学习在技术上属于“序列预测”，但你实际上并不需要它来理解 LLM。

我们将考虑的最直接的联系是在第八节中查看生成对抗网络。在游戏中的后悔最小化和均衡之间存在许多深度联系，而生成对抗网络基本上是通过让两个神经网络相互玩游戏来工作的。实用的基于梯度的优化算法，如 Adam，也源于这个领域，这要归功于 AdaGrad 算法的引入，该算法最初是在在线和对抗性设置中进行分析的。在其他见解方面，我发现以下观点很有用：如果你使用合理的学习率计划进行基于梯度的优化，那么处理数据点的顺序实际上并不重要。梯度下降可以处理它。

我鼓励你至少浏览一下 Elad Hazan 的《“在线凸优化导论”》（[“Introduction to Online Convex Optimization”](https://arxiv.org/pdf/1909.05207.pdf)）的第一章，以了解后悔最小化的目标。我在这本书上花了很多时间，我认为它非常出色。

## 强化学习

当我们在第四节中查看微调方法时，强化学习（RL）将最直接地出现，并且也可能是一个思考“代理”应用和一些针对状态空间模型出现的“控制理论”概念的有用心理模型。像这份文档中讨论的许多主题一样，如果你愿意，你可以深入探索许多不同的 RL 相关线索；就语言建模和对齐而言，最重要的是要熟悉马尔可夫决策过程的基本问题设置、策略和轨迹的概念，以及 RL 的标准迭代+基于梯度的优化方法的高级理解。

Lilian Weng 的这篇[博客文章](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)是一个很好的起点，尽管它相对简洁，但其中充满了重要的 RL 思想。它还涉及到了与 AlphaGo 和游戏玩法之间的联系，这可能会引起你的兴趣。

Sutton 和 Barto 合著的教科书《“强化学习：导论”》（[“Reinforcement Learning: An Introduction”](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)）通常被认为是该领域的经典参考文本，至少对于“非深度”方法来说是这样。这是我最初学习强化学习时的主要指南，它对 Lilian 博客文章中提到的许多主题进行了更深入的探讨。

如果你想要跳到一些更具神经风格的内容，Andrej Karpathy 有一个关于深度强化学习的优秀[博客文章](https://karpathy.github.io/2016/05/31/rl/)；Yuxi Li 的这篇[手稿](https://arxiv.org/pdf/1810.06339)和 Aske Plaat 的这篇[教科书](https://arxiv.org/pdf/2201.02135)可能对进一步深入研究有所帮助。

如果你喜欢 3Blue1Brown 风格的动画视频，系列[“通过书籍学习强化学习”（“Reinforcement Learning By the Book”）](https://www.youtube.com/playlist?list=PLzvYlJMoZ02Dxtwe-MmH4nOB5jYlMGBjr) 是一个很好的替代选择，它传达了 Sutton 和 Barto 的大量内容，以及一些深度强化学习，并使用引人入胜的视觉化呈现。

## 马尔可夫模型

在马尔可夫决策过程中运行固定策略会产生一个马尔可夫链；类似这种设置的流程相当普遍，许多机器学习的分支都涉及在马尔可夫假设下（即给定当前状态，缺乏路径依赖性）对系统进行建模。Aja Hammerly 的这篇[博客文章](https://thagomizer.com/blog/2017/11/07/markov-models.html)很好地说明了通过马尔可夫过程思考语言模型，而“数据科学论文集”中的这篇[文章](https://ericmjl.github.io/essays-on-data-science/machine-learning/markov-models/)则提供了构建自回归隐马尔可夫模型的例子和代码，这些模型将开始与我们稍后将要查看的一些神经网络架构有模糊相似之处。

Simeon Carstens 的这篇[博客文章](https://www.tweag.io/blog/2019-10-25-mcmc-intro1/)对马尔可夫链蒙特卡洛方法进行了很好的介绍，这是一种强大的、广泛使用的从隐式表示的分布中进行采样的技术，对于从随机梯度下降到扩散等概率主题的思考非常有帮助。

马尔可夫模型也是许多贝叶斯方法的核心。参见 Zoubin Ghahramani 的这篇[教程](https://mlg.eng.cam.ac.uk/zoubin/papers/ijprai.pdf)，它提供了一个很好的概述，教科书[“模式识别与机器学习”（“Pattern Recognition and Machine Learning”）](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)从贝叶斯的角度探讨了多个机器学习主题（以及一个更深入的 HMM 介绍），以及 Goodfellow 等人“深度学习”教科书的这一[章节](https://www.deeplearningbook.org/contents/graphical_models.html)，其中探讨了与深度学习的联系。

# 第二部分：神经网络序列预测

**目标：** 调查深度学习方法及其在序列和语言建模中的应用，直至基本的 Transformer。

在这里，与手册中的其他任何部分相比，我将主要借鉴现有的教学资源序列。许多人在各种格式下已经很好地覆盖了这一材料，没有必要重新发明轮子。

从神经网络的基本知识到 Transformers（2024 年大多数前沿 LLMs 的主导架构）有几种不同的路径。一旦我们覆盖了基础知识，我主要会关注“深度序列学习”方法，如 RNNs。许多深度学习书籍和课程会更加强调卷积神经网络（CNNs），这对于图像相关应用非常重要，并且在历史上是“扩展”特别成功的第一个领域之一，但技术上它们与 Transformers 相当脱节。它们将在我们讨论状态空间模型时出现，并且对于视觉应用肯定很重要，但你现在可以放心地跳过它们。然而，如果你急于想要快速进入新内容，一旦你对前馈神经网络感到舒适，你可以直接深入研究只包含解码器的 Transformers——这是 Andrej Karpathy 的出色视频[“让我们构建 GPT”（https://www.youtube.com/watch?v=kCc8FmEb1nY）]所采取的方法，将其视为神经网络 n-gram 模型预测下一个标记的扩展。这可能是你以不到 2 小时的速度运行 Transformers 的最佳选择。但如果你有更多时间，了解 RNNs、LSTMs 和编码器-解码器 Transformers 的历史无疑是值得的。

这一节主要包含指向以下来源的内容的指示（以及一些博客文章）：

+   [“深入浅出深度学习”（d2l.ai）](http://d2l.ai)互动式教科书（图形优美，内联代码，一些理论）

+   3Blue1Brown 的[“神经网络”](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)视频系列（大量动画）

+   Andrej Karpathy 的[“从零到英雄”（https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ）]视频系列（现场编码+很好的直觉）

+   [“StatQuest with Josh Starmer”](https://www.youtube.com/@statquest)视频

+   Goodfellow 等人所著的[“深度学习”](https://www.deeplearningbook.org/)教科书（理论导向，无 Transformers）

如果你的重点是应用，你可能会发现互动式书籍[“使用 PyTorch 和 Scikit-Learn 进行机器学习”（https://github.com/rasbt/machine-learning-book/tree/main）]很有用，但我在个人层面上并不那么熟悉它。

对于这些主题，你也许可以通过向你偏好的 LLM 聊天界面提出概念性问题来应对。这很可能在后面的章节中不成立——其中一些主题是在许多当前 LLMs 的知识截止日期之后引入的，而且关于它们的网络文本也相对较少，因此你可能会遇到更多的“幻觉”。

## 使用神经网络进行统计预测

我实际上并不确定我第一次是从哪里了解到神经网络的——它们在技术讨论和一般在线媒体中如此普遍，以至于即使你没有正式学习，我也假设你已经通过渗透学到了很多。尽管如此，那里有很多有价值的解释者，我会突出一些我最喜欢的。

+   3Blue1Brown 的[“神经网络”](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)系列的前 4 个视频将带你从基本定义到反向传播的机制。

+   这篇来自 Andrej Karpathy（当时他还是一名博士生）的[博客文章](https://karpathy.github.io/neuralnets/)是一个扎实的速成课程，配有他关于从头开始构建反向传播的[视频](https://www.youtube.com/watch?v=VMj-3S1tku0)。

+   这篇来自 Chris Olah 的[博客文章](https://colah.github.io/posts/2015-08-Backprop/)对神经网络反向传播背后的数学进行了简洁的概述。

+   [d2l.ai](http://d2l.ai)书籍的第 3-5 章作为“经典教科书”形式的深度网络回归+分类的展示非常出色，其中包含了代码示例和可视化。

## 循环神经网络

RNNs 是我们开始给模型添加“状态”的地方（随着我们处理越来越长的序列），并且与隐藏马尔可夫模型有一些高级相似性。这篇来自[Andrej Karpathy](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)的博客文章是一个很好的起点。[d2l.ai](http://d2l.ai)书籍的第九章对主要思想和代码都很有帮助；如果你想了解更多理论，可以查看“深度学习”的第十章[Chapter 10](https://www.deeplearningbook.org/contents/rnn.html)。

对于视频，[这里](https://www.youtube.com/watch?v=AsNTP8Kwu80)有一个来自 StatQuest 的不错视频。

## LSTM 和 GRU

长短期记忆（LSTM）网络和门控循环单元（GRU）网络在 RNN 的基础上增加了更专业的状态表示机制（具有“记忆”、“遗忘”和“重置”等语义灵感），这对于在更具挑战性的数据域（如语言）中提高性能非常有用。

[d2l.ai](http://d2l.ai)的第十章对这两者都进行了很好的介绍（直到 10.3）。Chris Olah 的[“理解 LSTM 网络”](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)博客文章也非常出色。来自“The AI Hacker”的这个[视频](https://www.youtube.com/watch?v=8HyCNIVRbSU)对两者都进行了坚实的高级概述；StatQuest 也有关于 LSTMs 的[视频](https://www.youtube.com/watch?v=YCzL96nL7j0)，但没有 GRUs。GRUs 基本上是 LSTMs 的一个简化替代品，具有相同的基本目标，是否具体涵盖它们取决于你。

LSTM 或 GRU 并不是 Transformer 的真正先决条件，因为 Transformer 是“无状态的”，但它们对于理解神经网络序列的一般挑战和 Transformer 设计选择是有用的，它们还将有助于激发第七节中解决“二次扩展问题”的一些方法。

## 嵌入和主题建模

在消化 Transformer 之前，首先建立一些概念是值得的，这些概念将有助于推理大型语言模型内部底层的运作情况。虽然深度学习在 NLP 领域带来了巨大的进步浪潮，但与处理词频和 n-gram 重叠的“老式”方法相比，推理起来确实要困难一些；然而，尽管这些方法并不总是适用于更复杂的任务，但它们对于神经网络可能学习的“特征”类型是有用的心智模型。例如，了解潜在狄利克雷分配（Latent Dirichlet Allocation）用于主题建模([博客文章](https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2))和[tf-idf](https://jaketae.github.io/study/tf-idf/)对于理解数值相似度或相关性评分对语言的意义是非常有价值的。

将单词（或标记）视为高维的“意义”向量是非常有用的，Word2Vec 嵌入方法很好地说明了这一点——你可能之前已经看到了经典的“国王 - 男人 + 女人 = 女王”示例。Jay Alammar 的[《图解 Word2Vec》](https://jalammar.github.io/illustrated-word2vec/)对于建立这种直觉非常有帮助，斯坦福大学 CS224n 的这些[课程笔记](https://web.stanford.edu/class/cs224n/readings/cs224n_winter2023_lecture1_notes_draft.pdf)也非常优秀。这里还有一个 ritvikmath 的关于 Word2Vec 的[视频](https://www.youtube.com/watch?v=f7o8aDNxf7k)，以及一个来自 Computerphile 的有趣[视频](https://www.youtube.com/watch?v=gQddtTdmG_8)关于神经词嵌入。

除了作为一个有用的直觉和更大语言模型的一个元素之外，独立的神经网络嵌入模型今天也被广泛使用。通常这些是仅包含编码器的 Transformer，通过“对比损失”训练来构建文本输入的高质量向量表示，这对于检索任务（如 RAG）非常有用。参见 Cohere 的这篇[文章+视频](https://docs.cohere.com/docs/text-embeddings)以获得简要概述，以及 Lilian Weng 的这篇[博客文章](https://lilianweng.github.io/posts/2021-05-31-contrastive/)以深入了解。

## 编码器和解码器

到目前为止，我们对网络的输入并没有太多的偏见——无论是数字、字符还是单词——只要它能以某种方式转换为向量表示。循环模型可以配置为输入和输出单个对象（例如，一个向量）或整个序列。这个观察结果使得序列到序列的编码器-解码器架构得以兴起，该架构因机器翻译而闻名，并且是著名论文“[注意力即一切](https://arxiv.org/abs/1706.03762)”中 Transformer 的原设计。在这里，目标是取一个输入序列（例如，一个英文句子），“编码”成捕获其“意义”的向量对象，然后“解码”该对象成另一个序列（例如，一个法语文句）。[d2l.ai](http://d2l.ai)的[第十章](https://d2l.ai/chapter_recurrent-modern/index.html)（10.6-10.8）也涵盖了这一设置，为[第十一章](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)（直到 11.7）中 Transformer 的编码器-解码器公式的介绍奠定了基础。出于历史原因，你当然应该至少浏览一下原始论文，尽管通过[“The Annotated Transformer”](https://nlp.seas.harvard.edu/annotated-transformer/)或如果你想要更多可视化，可能通过[“The Illustrated Transformer”](https://jalammar.github.io/illustrated-transformer/)的内容展示，你可能会得到更多收获。斯坦福 CS224n 的这些[笔记](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf)也非常好。

StatQuest 上有关于[编码器-解码器](https://www.youtube.com/watch?v=L8HKweZIOmg)架构和[注意力](https://www.youtube.com/watch?v=PSs6nxngL6k)的视频，[The AI Hacker](https://www.youtube.com/watch?v=4Bdc55j80l8)对原始 Transformer 的全面讲解。

然而，请注意，这些编码器-解码器 Transformer 与大多数现代 LLMs 不同，后者通常是“仅解码器”的——如果你时间紧迫，直接跳到这些模型并跳过历史课程可能没问题。

## 仅解码器 Transformer

Transformer 内部有很多移动部件——多头注意力、跳跃连接、位置编码等——第一次看到时可能很难全部理解。对为什么做出某些选择建立直觉有很大帮助，因此我建议几乎每个人都要看一些关于它们的视频（即使你通常是一个教科书学习者），主要是因为有一些视频真的很出色：

+   | 3Blue1Brown 的[“But what is a GPT?”](https://www.youtube.com/watch?v=wjZofJX0v4M)和“Attention in transformers, explained visually” – 美丽的动画+讨论，据说还有第三个视频在路上 |
+   | --- | --- |

+   Andrej Karpathy 的[“让我们构建 GPT”](https://www.youtube.com/watch?v=kCc8FmEb1nY)视频——现场编码和出色的解释，对我的一些理解起到了很大的帮助。

这里有一篇由 Cameron Wolfe 撰写的[博客文章](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)，以类似《图解/注释 Transformer》文章的风格介绍了仅解码器架构。在 d2l.ai 中也有一个很好的部分（[11.9](https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html)）涵盖了仅编码器、编码器-解码器和仅解码器 Transformer 之间的关系。

# 第三部分：现代语言模型的基础

**目标：** 调查与训练 LLM 相关的核心主题，重点在于概念原语。

在本节中，我们将探讨一系列概念，这些概念将引导我们从仅解码器的 Transformer 架构过渡到理解当今许多前沿大型语言模型（LLM）背后的实现选择和权衡。如果你首先想对章节中的主题以及一些后续章节有一个鸟瞰图，Sebastian Raschka 的文章《[理解大型语言模型](https://magazine.sebastianraschka.com/p/understanding-large-language-models)》是一个很好的总结，概述了 LLM 环境的概况（至少到 2023 年中为止）。

## 分词

字符级别的分词（如 Karpathy 的几个视频中所示）对于大规模 Transformer 来说通常效率较低，与词级别的分词相比，而天真地选择一个固定的“字典”（例如 Merriam-Webster）的全词可能会在推理时遇到未见过的词或拼写错误。相反，典型的做法是使用子词级别的分词来“覆盖”可能的输入空间，同时保持来自更大分词池的效率提升，使用字节对编码（BPE）等算法来选择合适的标记集。如果你在入门算法课程中见过霍夫曼编码，我认为它在这里是一个有用的类比，尽管输入输出格式明显不同，因为我们事先不知道“标记”的集合。我建议观看 Andrej Karpathy 的[视频](https://www.youtube.com/watch?v=zduSFxRajkE)关于分词，并查看 Masato Hagiwara 的这个分词[指南](https://blog.octanove.org/guide-to-subword-tokenization/)。

## 位置编码

正如我们在上一节中看到的，Transformer 并没有在上下文窗口内具有相同的邻接或位置概念（与 RNN 相比），位置必须用某种类型的向量编码来表示。虽然这可以通过类似 one-hot 编码的简单方法来完成，但这对于上下文扩展来说是不切实际的，并且对于可学习性来说是不理想的，因为它丢弃了序数概念。最初，这是通过正弦位置编码来完成的，如果你熟悉傅里叶特征，可能会觉得有些熟悉；如今，这种类型方法最流行的实现可能是旋转位置编码（RoPE），它在训练期间通常更稳定且学习速度更快。

资源：

+   Harrison Pim 的关于位置编码直觉的[博客文章](https://harrisonpim.com/blog/understanding-positional-embeddings-in-transformer-models)

+   Mehreen Saeed 的关于原始 Transformer 位置编码的[博客文章](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)

+   来自 Eleuther AI 的关于 RoPE 的博客文章：[博客文章](https://blog.eleuther.ai/rotary-embeddings/)，原始 Transformer：https://kazemnejad.com/blog/transformer_architecture_positional_encoding/ https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/

+   DeepLearning Hero 的[动画视频](https://www.youtube.com/watch?v=GQPOtyITy54)

## 预训练配方

一旦你决定在特定数据集（例如 Common Crawl、FineWeb）上预训练一个特定大小的一般性大型语言模型（LLM），在你准备开始之前，仍有许多选择需要做出：

+   注意力机制（多头、多查询、分组查询）

+   激活函数（ReLU, GeLU, SwiGLU）

+   优化器、学习率和调度器（AdamW、预热、余弦衰减）

+   Dropout？

+   超参数选择和搜索策略

+   批处理、并行化策略、梯度累积

+   训练多长时间，多久重复一次数据

+   …以及许多其他变体轴

就我所能了解的，并没有一个适用于所有情况的规则手册来指导如何进行这项工作，但我将分享一些值得考虑的有价值资源，具体取决于你的兴趣：

+   尽管这篇博客文章早于 LLM 时代，但[“训练神经网络的配方”](https://karpathy.github.io/2019/04/25/recipe/)是一篇很好的起点，因为这些问题在深度学习中的许多方面都是相关的。

+   Alpin Dale 的[“新手大型语言模型训练指南”](https://rentry.org/llm-training)，讨论了实践中的超参数选择，以及我们将在未来章节中看到的微调技术。

+   Replit 的[“如何训练你自己的大型语言模型”](https://blog.replit.com/llm-training)有一些关于数据管道和评估的精彩讨论。

+   为了理解注意力机制权衡，请参阅 Shobhit Agarwal 的这篇[帖子“导航注意力景观：MHA，MQA 和 GQA 解码”](https://iamshobhitagarwal.medium.com/navigating-the-attention-landscape-mha-mqa-and-gqa-decoded-288217d0a7d1)。

+   关于“流行默认值”的讨论，请参阅 Deci AI 的这篇[帖子“现代 Transformer 的演变：从‘Attention Is All You Need’到 GQA，SwiGLU 和 RoPE”](https://deci.ai/blog/evolution-of-modern-transformer-swiglu-rope-gqa-attention-is-all-you-need/)。

+   关于学习率调度的详细信息，请参阅 d2l.ai 书籍中的第 12.11 章[链接](https://d2l.ai/chapter_optimization/lr-scheduler.html)。

+   关于围绕“最佳实践”报告的一些争议的讨论，请参阅 Eleuther AI 的这篇[帖子](https://blog.eleuther.ai/nyt-yi-34b-response/)。

## 分布式训练和 FSDP

与训练无法适应单个 GPU（甚至多 GPU 机器）的模型相关联的许多其他挑战，通常需要使用分布式训练协议，如完全分片数据并行（FSDP），在训练期间模型可以在机器之间协同定位。也许也值得了解其前身分布式数据并行（DDP），这在下面链接的第一篇帖子中有介绍。

资源：

+   Meta 的官方 FSDP [博客文章](https://engineering.fb.com/2021/07/15/open-source/fsdp/)（该方法的开创者） https://sumanthrh.com/post/distributed-and-efficient-finetuning/

+   Bar Rozenman 关于 FSDP 的[博客文章](https://blog.clika.io/fsdp-1/)，其中包含许多优秀的可视化

+   Yi Tai 关于在初创环境中预训练模型挑战的[报告](https://www.yitay.net/blog/training-great-llms-entirely-from-ground-zero-in-the-wilderness)

+   Answer.AI 关于将 FSDP 与参数高效微调技术结合使用以在消费级 GPU 上使用的[技术博客](https://www.answer.ai/posts/2024-03-14-fsdp-qlora-deep-dive.html)。

## 缩放定律

了解缩放定律作为元主题是有用的，这在 LLM 的讨论中经常出现（最显著的是在“Chinchilla”[论文](https://arxiv.org/abs/2203.15556)的参考中），比任何特定的经验发现或技术都多。简而言之，通过扩展用于训练语言模型的模型、数据和计算，将产生对模型损失的相当可靠的预测。这然后使得无需运行昂贵的网格搜索即可校准最佳超参数设置。

资源：

+   [Chinchilla 大型语言模型缩放定律](https://medium.com/@raniahossam/chinchilla-scaling-laws-for-large-language-models-llms-40c434e4e1c1)（Rania Hossam 的博客概述）

+   [大型语言模型的新缩放定律](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/new-scaling-laws-for-large-language-models)（LessWrong 上的讨论）

+   [Chinchilla 的深远影响](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications)（LessWrong 上的帖子）

+   [Chinchilla 缩放：复制尝试](https://epochai.org/blog/chinchilla-scaling-a-replication-attempt)（Chinchilla 发现的潜在问题）

+   [缩放定律和涌现特性](https://cthiriet.com/blog/scaling-laws)（Clément Thiriet 的博客帖子）

+   [“语言模型的缩放”](https://www.youtube.com/watch?v=UFem7xa3Q2Q)（斯坦福 CS224n 的视频讲座）

## 混合专家

虽然今天使用的许多知名 LLM（如 Llama3）都是“密集”模型（即没有强制稀疏化），但混合专家（MoE）架构在权衡“知识”和效率方面越来越受欢迎，最著名的是 Mistral AI 的“Mixtral”模型（8x7B 和 8x22B）在开放权重世界中的应用，以及[传闻](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/)将用于 GPT-4。在 MoE 模型中，只有一小部分参数在每个推理步骤中是“活跃”的，有训练好的路由模块用于在每个层选择并行的“专家”。这使得模型在大小（以及可能“知识”或“智能”）上可以增长，同时与同等规模的密集模型相比，在训练或推理上保持高效。

请参阅 Hugging Face 的这篇[博客文章](https://huggingface.co/blog/moe)以获取技术概述，以及 Trelis Research 的这篇[视频](https://www.youtube.com/watch?v=0U_65fLoTq0)以获取可视化解释。

# 第四部分：LLM 的微调方法

**目标：** 调查用于在预训练后改进和“对齐”LLM 输出质量的技巧。

在预训练中，目标基本上是“在随机互联网文本上预测下一个标记”。虽然结果“基础”模型在某些情况下仍然有用，但它们的输出通常是混乱的或“未对齐”的，并且可能不尊重双向对话的格式。在这里，我们将探讨一系列从这些基础模型到更熟悉的友好聊天机器人和助手的技巧。一个很好的配套资源，特别是对于这一部分，是 Maxime Labonne 在 GitHub 上的交互式[LLM 课程](https://github.com/mlabonne/llm-course?tab=readme-ov-file)。

## 指令微调

指令微调（或称为“指令调整”、“监督微调”或“聊天调整”——这里的界限有些模糊）是用于引导大型语言模型（LLMs）遵循特定风格或格式的首要技术（至少最初是这样）。在这里，数据以一系列（输入，输出）对的形式呈现，其中输入是用户需要回答的问题，而模型的目标是预测输出——通常这也涉及到添加特殊的“开始”/“停止”/“角色”标记和其他掩码技术，使模型能够“理解”用户输入与其自身输出的区别。这种技术也广泛应用于具有特定问题结构的特定任务微调数据集（例如翻译、数学、通用问答）。

欲了解简短概述，请参阅 Sebastian Ruder 的这篇[博客文章](https://newsletter.ruder.io/p/instruction-tuning-vol-1)或 Shayne Longpre 的这篇[视频](https://www.youtube.com/watch?v=YoVek79LFe0)。

## 低秩适配器（LoRA）

虽然预训练（和“完全微调”）需要更新模型的所有参数的梯度，但在消费级 GPU 或家庭设置中通常不切实际；幸运的是，通过使用参数高效的微调（PEFT）技术，如低秩适配器（LoRA），通常可以显著降低计算需求。这可以在相对较小的数据集上实现有竞争力的性能，尤其是在特定应用场景中。LoRA 背后的主要思想是通过“冻结”基矩阵并在具有更小内部维度的因子表示中训练来在低秩空间中训练每个权重矩阵，然后将该表示添加到基矩阵中。

资源：

+   LoRA 论文讲解 [(视频，第一部分)](https://youtu.be/dA-NhCtrrVE?si=TpJkPfYxngQQ0iGj)

+   LoRA 代码演示 [(视频，第二部分)](https://youtu.be/iYr1xZn26R8?si=aG0F8ws9XslpZ4ur)

+   [“使用低秩适配器进行参数高效的 LLM 微调”](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html) by Sebastian Raschka

+   [“使用 LoRA 微调 LLMs 的实用技巧”](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) by Sebastian Raschka

此外，一种称为 DoRA 的“分解”LoRA 变体在最近几个月中越来越受欢迎，通常能带来性能提升；有关更多详细信息，请参阅 Sebastian Raschka 的这篇[文章](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch)。

## 奖励模型和强化学习与人类反馈（RLHF）

将语言模型“对齐”的最突出技术之一是从人类反馈中进行强化学习（RLHF）；在这里，我们通常假设一个 LLM 已经被指令调整以尊重聊天风格，并且我们还有一个“奖励模型”，它已经在人类偏好上进行了训练。对于给定的输入的不同输出对，其中人类已经选择了一个首选输出，奖励模型的学习目标是预测首选输出，这涉及到隐式地学习偏好“分数”。这允许启动一个关于人类偏好的通用表示（至少就输出对的数据集而言），这可以用作使用 RL 策略梯度技术（如 PPO）对 LLM 进行持续训练的“奖励模拟器”。

对于概述，请参阅 Hugging Face 的帖子“从人类反馈中展示强化学习（RLHF）”和 Chip Huyen 的“从人类反馈中进行强化学习”，以及 Nathan Lambert 的[RLHF 演讲](https://www.youtube.com/watch?v=2MBJOuVq380)。此外，Sebastian Raschka 的这篇[文章](https://sebastianraschka.com/blog/2024/research-papers-in-march-2024.html)深入探讨了 RewardBench，以及如何通过利用直接偏好优化（Direct Preference Optimization）的想法来评估奖励模型本身，这是另一种将 LLMs 与人类偏好数据对齐的突出方法。

## 直接偏好优化方法

对齐算法的空间似乎正在遵循与我们在十年前看到的随机优化算法相似的轨迹。在这个类比中，RLHF 就像 SGD——它有效，它是原始的，它也已经成为一个通用的“万能”术语，用于指代随后出现的算法类别。也许 DPO 是 AdaGrad，自从其发布以来，沿着相同方向的算法发展迅速（KTO、IPO、ORPO 等），它们的相对优势仍在积极讨论中。也许一年后，每个人都会选择一个标准方法，这将成为对齐的“Adam”。

关于 DPO 背后的理论的概述，请参阅 Matthew Gunton 的这篇[博客文章](https://towardsdatascience.com/understanding-the-implications-of-direct-preference-optimization-a4bbd2d85841)；Hugging Face 的这篇[博客文章](https://huggingface.co/blog/dpo-trl)包含一些代码，并演示了如何在实践中使用 DPO。Hugging Face 的另一篇[博客文章](https://huggingface.co/blog/pref-tuning)也讨论了最近几个月出现的几种 DPO 风格的方法的权衡。

## 上下文缩放

除了任务指定或对齐之外，微调的另一个常见目标是增加模型的有效上下文长度，无论是通过额外训练、调整位置编码参数，还是两者兼而有之。即使向模型上下文中添加更多标记可以进行“类型检查”，但如果模型在预训练期间可能没有看到如此长的序列，那么在额外的更长的示例上进行训练通常是必要的。

资源：

+   [“扩展旋转嵌入以用于长上下文语言模型”](https://gradient.ai/blog/scaling-rotational-embeddings-for-long-context-language-models) by Gradient AI

+   [“扩展 RoPE”](https://blog.eleuther.ai/yarn/) by Eleuther AI，介绍通过注意力温度缩放增加上下文的方法 YaRN

+   [“关于长上下文微调的一切”](https://huggingface.co/blog/wenbopan/long-context-fine-tuning) by Wenbo Pan

## 离心蒸馏和合并

在这里，我们将探讨两种在 LLM 之间巩固知识的不同方法——蒸馏和合并。蒸馏最初是为 BERT 模型推广的，其目标是“蒸馏”较大模型的知识和性能，将其转化为较小的模型（至少对于某些任务），在较小模型训练期间充当“教师”，从而绕过需要大量人工标注数据的需求。

一些关于蒸馏的资源：

+   [“更小、更快、更便宜、更轻：介绍 DistilBERT，BERT 的蒸馏版本”](https://medium.com/huggingface/distilbert-8cf3380435b5) from Hugging Face

+   [“LLM 蒸馏揭秘：完整指南”](https://snorkel.ai/llm-distillation-demystified-a-complete-guide/) from Snorkel AI

+   [“逐步蒸馏”博客](https://blog.research.google/2023/09/distilling-step-by-step-outperforming.html) from Google Research

相反，合并是一种更接近“狂野西部”的技术，主要被开源工程师使用，他们希望结合多个微调工作的优势。对我来说，它竟然能工作，也许为“线性表示假设”（将在下一节讨论可解释性时出现）提供了一些可信度。基本想法是取两个相同基础模型的不同的微调版本，并简单地平均它们的权重。无需训练。技术上，这通常是“球面插值”（或“slerp”），但这基本上只是带有归一化步骤的复杂平均。更多详情，请参阅 Maxime Labonne 的文章 [使用 mergekit 合并大型语言模型](https://huggingface.co/blog/mlabonne/merge-models)。

# 第五节：LLM 评估和应用

**目标：** 调查 LLM 在实践中如何被使用和评估，而不仅仅是“聊天机器人”。

在这里，我们将探讨一些与改进或修改语言模型性能（无需额外训练）相关的话题，以及衡量和理解其性能的技术。

在深入各个章节之前，我推荐这两篇高级概述，它们涉及了我们在这里将要探讨的许多主题：

+   Chip Huyen 的[“为生产构建 LLM 应用”](https://huyenchip.com/2023/04/11/llm-engineering.html)

+   O’Reilly 的[“从一年使用 LLM 构建中学到的经验”第一部分](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/)和[第二部分](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii/)

这些网络课程也有很多相关的互动材料：

+   来自 Maxime Labonne 的[“大型语言模型课程”](https://github.com/mlabonne/llm-course)

+   来自微软的[“面向初学者的生成式 AI”](https://microsoft.github.io/generative-ai-for-beginners/)

## 基准测试

除了在 LLM 训练期间使用的标准数值性能指标（如交叉熵损失和[困惑度](https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72)）之外，前沿 LLM 的真实性能更常见的是根据一系列基准或“评估”来判断。这些常见的类型包括：

+   人工评估输出（例如 [LMSYS Chatbot Arena](https://chat.lmsys.org/)）

+   AI 评估输出（如[RLAIF](https://argilla.io/blog/mantisnlp-rlhf-part-4/)中使用）

+   挑战性问题集（例如 HuggingFace 的[LLM 排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)中的）

查看斯坦福大学 CS224n 的[幻灯片](https://web.stanford.edu/class/cs224n/slides/cs224n-spr2024-lecture11-evaluation-yann.pdf)，以了解概述。Jason Wei 的这篇[博客文章](https://www.jasonwei.net/blog/evals)和 Peter Hayes 的这篇[文章](https://humanloop.com/blog/evaluating-llm-apps?utm_source=newsletter&utm_medium=sequence&utm_campaign=)很好地讨论了设计良好评估的挑战和权衡，并突出了今天使用的一些最突出的评估方法。开源框架[inspect-ai](https://ukgovernmentbeis.github.io/inspect_ai/)的文档也包含了一些关于设计基准和可靠评估管道的有用讨论。

## 样本和结构化输出

当典型的 LLM 推理一次处理一个标记时，有一些参数（如温度、top_p、top_k）控制标记分布，可以修改以控制响应的多样性，以及允许一定程度的“前瞻”的非贪婪解码策略。Maxime Labonne 的这篇[博客文章](https://towardsdatascience.com/decoding-strategies-in-large-language-models-9733a8f70539)很好地讨论了其中的一些。

有时我们还想让我们的输出遵循特定的结构，尤其是如果我们将 LLMs 作为更大系统的一部分而不是仅仅作为聊天界面来使用时。少样本提示效果不错，但并不总是如此，尤其是当输出模式变得更加复杂时。对于 JSON、Pydantic 和 Outlines 等模式类型，是限制 LLMs 输出结构的流行工具。一些有用的资源：

+   [Pydantic 概念](https://docs.pydantic.dev/latest/concepts/models/)

+   [JSON 大纲](https://outlines-dev.github.io/outlines/reference/json/)

+   [大纲评论](https://michaelwornow.net/2023/12/29/outlines-demo) by Michael Wornow

## 提示技术

有许多提示技术，还有更多提示工程指南，它们展示了如何从 LLMs 中诱导出更令人满意的输出。其中一些经典方法：

+   少样本示例

+   思维链

+   检索增强生成（RAG）

+   ReAct

Lilian Weng 的这篇[博客文章](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)讨论了几种最占主导地位的方法，[指南](https://www.promptingguide.ai/techniques)对今天广泛使用的突出技术有很好的覆盖和示例。在 Twitter/X 或 LinkedIn 上进行关键词搜索将为您提供更多内容。我们还将深入探讨 RAG 和代理方法，在后面的章节中。

## 向量数据库和重排序

RAG 系统需要能够快速从大型语料库中检索相关文档的能力。相关性通常由查询和文档的语义嵌入向量的相似度度量来确定，例如余弦相似度或欧几里得距离。如果我们只有少量文档，这可以通过查询与每个文档之间的计算来完成，但当文档数量增加时，这很快就会变得难以处理。这是向量数据库解决的问题，它通过在向量上维护高维索引来允许检索*近似*的 top-K 匹配（比检查所有对要快得多），这些向量有效地编码了它们的几何结构。请参阅 Pinecone 的这些[文档](https://www.pinecone.io/learn/series/faiss/)，它们很好地介绍了几种不同的向量存储方法，如局部敏感哈希和分层可导航小世界，这些方法可以使用流行的 FAISS 开源库实现。Alexander Chatzizacharias 的这篇[演讲](https://www.youtube.com/watch?v=W-i8bcxkXok)也提供了一个很好的概述。

向量检索的另一个相关应用是“重排序”问题，其中模型可以优化除查询相似度之外的指标，例如检索结果中的多样性。请参阅 Pinecone 的这些[文档](https://www.pinecone.io/learn/series/rag/rerankers/)以获取概述。我们将在下一章中了解更多关于 LLMs 实际如何使用检索结果的信息。

## 检索增强生成

过去一年中最受热议的 LLMs 应用之一，检索增强生成（RAG），就是如何“与 PDF 聊天”（如果内容大于模型上下文），以及像 Perplexity 和 Arc Search 这样的应用如何使用网络资源“定位”它们的输出。这种检索通常是通过将每个文档嵌入到向量数据库中进行存储，然后使用用户输入的相关部分进行查询来实现的。

一些概述：

+   [“RAG 解构”](https://blog.langchain.dev/deconstructing-rag/) from Langchain

+   [“使用开源和自定义 AI 模型构建 RAG”](https://www.bentoml.com/blog/building-rag-with-open-source-and-custom-ai-models) from Chaoyu Yang

DeepLearning.AI 的[高级 RAG](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/1/introduction)视频课程也可能有助于探索标准设置的变体。

## 工具使用和“代理”

你可能以某种形式遇到的其他大型应用热门词汇是“工具使用”和“代理”，或“代理编程”。这通常从我们在提示部分看到的 ReAct 框架开始，然后扩展到引发越来越复杂的行为，如软件工程（参见备受瞩目的“Devin”系统，来自 Cognition，以及几个相关的开源项目，如 Devon/OpenDevin/SWE-Agent）。有许多编程框架可以用于在 LLMs 之上构建代理系统，其中 Langchain 和 LlamaIndex 是最受欢迎的两个。似乎还有价值让 LLMs 重写它们自己的提示并评估它们自己的部分输出；这一观察是 DSPy 框架（用于“编译”程序的提示，与一组参考指令或期望输出相对）的核心，该框架最近受到了很多关注。

资源：

+   [“LLM 驱动的自主代理” (文章)](https://lilianweng.github.io/posts/2023-06-23-agent/) from Lilian Weng

+   [“LLM 抽象指南” (文章)](https://www.twosigma.com/articles/a-guide-to-large-language-model-abstractions/) from Two Sigma

+   [“DSPy Explained! (视频)”](https://www.youtube.com/watch?v=41EfOY0Ldkc) by Connor Shorten

也相关的是一些更具体（但可能更实用）的应用，这些应用与数据库相关——参见 Neo4J 的这两篇[博客](https://neo4j.com/developer-blog/knowledge-graphs-llms-multi-hop-question-answering/) [文章](https://neo4j.com/blog/unifying-llm-knowledge-graph/)，讨论了将 LLMs 应用于分析或构建知识图谱，或者查看 Numbers Station 的这篇[博客文章](https://numbersstation.ai/data-wrangling-with-fms-2/)，关于将 LLMs 应用于实体匹配等数据整理任务。

## 合成数据的 LLMs

越来越多的应用正在利用 LLM 生成的数据进行训练或评估，包括蒸馏、数据集增强、AI 辅助评估和标记、自我批评等。这篇[文章](https://www.promptingguide.ai/applications/synthetic_rag)展示了如何在 RAG 环境中构建这样的合成数据集，以及来自 Argilla 的这篇[文章](https://argilla.io/blog/mantisnlp-rlhf-part-4/)概述了 RLAIF，鉴于收集成对人类偏好数据所面临的挑战，RLAIF 通常是 RLHF 的一个流行替代方案。AI 辅助反馈也是 Anthropic 开创的“宪法 AI”对齐方法的一个核心组成部分（请参阅他们的[博客](https://www.anthropic.com/news/claudes-constitution)以获得概述）。

## 表示工程

表示工程是一种通过“控制向量”对语言模型输出进行细粒度引导的新兴且具有前景的技术。它在某种程度上类似于 LoRA 适配器，它通过向网络权重添加低秩偏差来引发特定的响应风格（例如“幽默”、“冗长”、“创新”、“诚实”），但计算效率更高，并且可以在不进行任何训练的情况下实现。相反，该方法只是观察沿着感兴趣轴（例如诚实）变化的输入对激活的差异，这些差异可以人工生成，然后进行降维。

请参阅来自 AI 安全中心（该方法的开创者）的这篇简短[博客文章](https://www.safe.ai/blog/representation-engineering-a-new-way-of-understanding-models)，以获得简要概述，以及来自 Theia Vogel 的这篇[文章](https://vgel.me/posts/representation-engineering/)，其中包含带有代码示例的技术深入探讨。Theia 还在这个[podcast episode](https://www.youtube.com/watch?v=PkA4DskA-6M)中介绍了该方法。

## 机制可解释性

机制可解释性（MI）是通过识别编码在模型权重中的“特征”或“电路”的稀疏表示来理解 LLM 内部工作原理的主导范式。除了使潜在修改或解释 LLM 输出成为可能之外，MI 通常被视为向可能“对齐”日益强大的系统的重要一步。这里的大部分参考文献将来自[Neel Nanda](https://www.neelnanda.io)，他是该领域的领先研究人员，在多种格式下创建了大量关于 MI 的有用教育资源：

+   [“全面机制可解释性解释与术语表”](https://www.neelnanda.io/mechanistic-interpretability/glossary)

+   [“我最喜欢的机制可解释性论文的极有见地的注释列表”](https://www.neelnanda.io/mechanistic-interpretability/favourite-papers)

+   [“机制可解释性快速入门指南”](https://www.lesswrong.com/posts/jLAvJt8wuSFySN975/mechanistic-interpretability-quickstart-guide) (Neel Nanda on LessWrong)

+   [“机制可解释性有多有用？”](https://www.lesswrong.com/posts/tEPHGZAb63dfq2v8n/how-useful-is-mechanistic-interpretability) (Neel 等人，在 LessWrong 上的讨论)

+   [“200 个可解释性问题”](https://docs.google.com/spreadsheets/d/1oOdrQ80jDK-aGn-EVdDt3dg65GhmzrvBWzJ6MUZB8n4/edit#gid=0) (Neel 的开放问题标注电子表格)

此外，Anthropic 的文章[“叠加的玩具模型”](https://transformer-circuits.pub/2022/toy_model/index.html)和[“扩展单义性：从 Claude 3 Sonnet 中提取可解释特征”](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)篇幅较长，但包含了许多这些概念的优秀可视化演示。

## 线性表示假设

从几条可解释性研究线路中涌现的一个主题是观察，Transformer 中的特征内部表示在高维空间中通常是“线性”的（类似于 Word2Vec）。一方面，这可能会一开始看起来令人惊讶，但这也是像基于相似度的检索、合并以及注意力中使用的键值相似度分数等技术的一个基本隐含假设。参见 Beren Millidge 的这篇[博客文章](https://www.beren.io/2023-04-04-DL-models-are-secretly-linear/)，Kiho Park 的这篇[演讲](https://www.youtube.com/watch?v=ko1xVcyDt8w)，以及至少浏览一下这篇论文[“语言模型表示空间和时间”](https://arxiv.org/pdf/2310.02207)的图表。

# 第 VI 节：高效推理的性能优化

**目标：** 调查架构选择和底层技术，以改善资源利用（时间、计算、内存）。

在这里，我们将探讨一些提高从预训练 Transformer 语言模型中进行推理的速度和效率的技术，其中大多数在实践中都相当广泛地使用。首先阅读这篇 Nvidia 的[博客文章](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)，以了解我们将要探讨的几个主题（以及许多其他主题）的快速入门课程是值得的。

## 参数量化

随着领先大型语言模型参数数量的快速增加，以及获取 GPU 以运行模型在成本和可用性方面的困难，对量化 LLM 权重以使用更少的位来使用越来越感兴趣，这通常可以在所需内存减少 50-75%（或更多）的情况下产生可比较的输出质量。通常，这不应该天真地去做；Tim Dettmers 是现代几种量化方法（LLM.int8(), QLoRA, bitsandbytes）的先驱之一，他有一篇很好的[博客文章](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)，介绍了量化原理，以及与大型模型训练中的涌现特征相关的混合精度量化的必要性。其他流行的方法和格式包括 GGUF（用于 llama.cpp）、AWQ、HQQ 和 GPTQ；请参阅 TensorOps 的这篇[文章](https://www.tensorops.ai/post/what-are-quantized-llms)以获取概述，以及 Maarten Grootendorst 的这篇[文章](https://www.maartengrootendorst.com/blog/quantization/)，讨论了它们的权衡。

除了能够在较小的机器上进行推理之外，量化在参数高效的训练中也非常流行；在 QLoRA 中，大多数权重被量化到 4 位精度并冻结，而活跃的 LoRA 适配器以 16 位精度进行训练。请参阅 Tim Dettmers 的这篇[演讲](https://www.youtube.com/watch?v=fQirE9N5q_Y)，或 Hugging Face 的这篇[博客](https://huggingface.co/blog/4bit-transformers-bitsandbytes)以获取概述。Answer.AI 的这篇[博客文章](https://www.answer.ai/posts/2024-03-14-fsdp-qlora-deep-dive.html)还展示了如何将 QLoRA 与 FSDP 结合，以在消费级 GPU 上高效微调 70B+参数模型。

## 推测性解码

推测性解码背后的基本思想是通过主要从远小于原始模型中采样标记，并在输出分布发生偏差时偶尔应用来自较大模型的校正（例如，每*N*个标记），来加速从较大模型的推理。这些批处理一致性检查通常比直接采样*N*个标记要快得多，因此如果较小模型的标记序列只是偶尔发生偏差，整体速度可以大幅提升。

请参阅 Jay Mody 的这篇[博客文章](https://jaykmody.com/blog/speculative-sampling/)，了解原始论文的概述，以及这篇 PyTorch[文章](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)中的一些评估结果。Trelis Research 还提供了一个很好的[视频概述](https://www.youtube.com/watch?v=hm7VEgxhOvk)。

## FlashAttention

计算注意力矩阵通常是在 Transformers 的推理和训练中遇到的主要瓶颈，FlashAttention 已经成为加速这一过程最广泛使用的技巧之一。与我们在第七节中将要看到的一些技术不同，这些技术通过更简洁的表示来*近似*注意力（因此会产生一些表示误差），FlashAttention 是一种*精确*的表示，其加速来自于对硬件感知的实现。它应用了一些技巧——即分块和重新计算——来分解注意力矩阵的表达式，从而显著减少内存 I/O 并提高墙钟性能（即使略微增加了所需的 FLOPS）。

资源：

+   [Tri Dao 的演讲](https://www.youtube.com/watch?v=gMOAud7hZg4) (FlashAttention 的作者)

+   [ELI5: FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) by Aleksa Gordić

## 键值缓存和分页注意力

如上所述的[NVIDIA 博客](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)中提到的，键值缓存在 Transformer 实现矩阵中相当标准，以避免重复计算注意力。这允许在速度和资源利用之间进行权衡，因为这些矩阵保存在 GPU VRAM 中。虽然对于单个“线程”的推理来说管理这一点相当直接，但当考虑到并行推理或单个托管模型实例的多个用户时，会出现许多复杂性。你如何避免重新计算系统提示和少样本示例的值？对于可能或可能不想继续聊天会话的用户，何时应该驱逐缓存元素？PagedAttention 及其流行的实现 [vLLM](https://docs.vllm.ai/en/stable/) 通过利用操作系统中经典的分页思想来解决这一问题，并已成为自托管多用户推理服务器的标准。

资源：

+   [KV 缓存：Transformers 中的内存使用](https://www.youtube.com/watch?v=80bIUggRJf4) (视频，高效自然语言处理)

+   [使用 vLLM 和分页注意力快速服务 LLM](https://www.youtube.com/watch?v=5ZlavKF_98U) (视频，Anyscale)

+   vLLM [博客文章](https://blog.vllm.ai/2023/06/20/vllm.html)

## CPU 负载

在 CPU（与 GPU 相比）上部分或全部运行 LLM 的主要方法是 llama.cpp。有关高级概述，请参阅[这里](https://www.datacamp.com/tutorial/llama-cpp-tutorial)；llama.cpp 作为 LMStudio 和 Ollama 等许多流行的自托管 LLM 工具/框架的后端。以下是一篇[博客文章](https://justine.lol/matmul/)，其中包含一些关于 CPU 性能改进的技术细节。

# 第七节：亚二次上下文缩放

**目标：调查避免 Transformers 中自注意力面临的“二次缩放问题”的方法。

在扩展 Transformer 的大小和上下文长度时，一个主要的瓶颈是注意力的二次性质，其中考虑了所有标记对的交互。在这里，我们将探讨多种绕过这一问题的方法，从目前广泛使用的方法到更具探索性（但很有希望）的研究方向。

## 滑动窗口注意力

在“Longformer”[论文](https://arxiv.org/abs/2004.05150)中引入的滑动窗口注意力，作为标准注意力的亚二次替代品，允许只关注最近标记/状态的滑动窗口（令人震惊，对吧？），而不是整个上下文窗口，前提是这些状态的向量已经关注了早期的那些，因此具有足够的表示能力来编码相关的早期上下文片段。由于其简单性，它已成为亚二次扩展中更广泛采用的方法之一，并被用于 Mistral 流行的 Mixtral-8x7B 模型（以及其他模型）。

资源：

+   “什么是滑动窗口注意力？”（https://klu.ai/glossary/sliding-window-attention）

+   “滑动窗口注意力”（https://medium.com/@manojkumal/sliding-window-attention-565f963a1ffd）

+   “Longformer：长文档 Transformer”（https://www.youtube.com/watch?v=_8KNb5iqblE）

## 环形注意力

另一种对标准注意力机制的修改，环形注意力通过具有“消息传递”结构的增量计算实现亚二次全上下文交互，其中“块”的上下文通过一系列步骤相互通信，而不是一次性全部通信。在每个块内，该技术本质上就是经典注意力。尽管在开放权重世界中它更多地是一个研究方向而不是标准技术，但据传言谷歌的 Gemini 可能正在使用环形注意力来启用其百万以上的标记上下文。

资源：

+   “打破界限：理解上下文窗口限制和环形注意力的概念”（https://medium.com/@tanuj22july/breaking-the-boundaries-understanding-context-window-limitations-and-the-idea-of-ring-attention-170e522d44b2）

+   “理解环形注意力：构建具有近乎无限上下文的 Transformer”（https://www.e2enetworks.com/blog/understanding-ring-attention-building-transformers-with-near-infinite-context）

+   “环形注意力解释”（https://www.youtube.com/watch?v=jTJcP8iyoOM）

## 线性注意力（RWKV）

Receptance-Weighted Key Value（RWKV）架构回归到 RNN 模型（例如 LSTMs）的一般结构，并进行修改以实现更大的扩展和一种*线性*的注意力机制，该机制支持其表示的循环“展开”（允许每个输出标记的恒定计算，随着上下文长度的增加）。

资源：

+   （Huggingface 博客）

+   [“RWKV 语言模型：具有 Transformer 优势的 RNN” - 第一部分](https://johanwind.github.io/2023/03/23/rwkv_overview.html)（博客文章，Johan Wind）

+   [“RWKV 语言模型的工作原理” - 第二部分](https://johanwind.github.io/2023/03/23/rwkv_details.html)

+   [“RWKV：为 Transformer 时代重新发明 RNN（论文解释）”](https://www.youtube.com/watch?v=x8pW19wKfXQ)（视频，Yannic Kilcher）

## 结构化状态空间模型

结构化状态空间模型（SSMs）在当前研究焦点方面已成为 Transformer 最受欢迎的替代方案之一，拥有几个显著的变体（S4、Hyena、Mamba/S6、Jamba、Mamba-2），但它们因复杂性而有些臭名昭著。该架构从经典控制理论和线性时不变系统中汲取灵感，进行了一系列优化，以将连续时间转换为离散时间，并避免大矩阵的密集表示。它们支持循环和卷积表示，这既提高了训练效率，也提高了推理效率，许多变体需要仔细条件化的“隐藏状态矩阵”表示，以支持“记忆”上下文而不需要所有成对注意力。SSMs 似乎在规模上也变得更加实用，最近在高质量文本到语音转换（通过[Cartesia AI](https://www.cartesia.ai/)，由 SSMs 的发明者创立）方面取得了突破性的速度提升。

目前最好的解释可能是“《标注的 S4》”，它专注于 SSMs 起源的 S4 论文。帖子“《Mamba 和状态空间模型的视觉指南》”非常适合直观和视觉理解，数学内容较少，Yannic Kilcher 还有一个关于 SSMs 的[视频](https://www.youtube.com/watch?v=9dSkvxS2EB0)。

最近，Mamba 的作者发布了他们的后续论文“Mamba 2”，以及他们的一系列配套博客文章讨论了 SSM 表示和线性注意力之间的一些新发现的联系，这可能很有趣：

+   [状态空间对偶性（Mamba-2）第一部分 - 模型](https://tridao.me/blog/2024/mamba2-part1-model/)

+   [状态空间对偶性（Mamba-2）第二部分 - 理论](https://tridao.me/blog/2024/mamba2-part2-theory/)

+   [状态空间对偶性（Mamba-2）第三部分 - 算法](https://tridao.me/blog/2024/mamba2-part3-algorithm/)

+   [状态空间对偶性（Mamba-2）第四部分 - 系统](https://tridao.me/blog/2024/mamba2-part4-systems/)

## 超注意力

与 RWKV 和 SSMs 有些相似，HyperAttention 是另一种实现类似注意力机制近线性缩放的提议，它依赖于局部敏感哈希（想想向量数据库）而不是循环表示。我没有看到它像其他方法那样被广泛讨论，但无论如何，它可能值得关注。

概览请参阅 Yousra Aoudi 的这篇[博客文章](https://medium.com/@yousra.aoudi/linear-time-magic-how-hyperattention-optimizes-large-language-models-b691c0e2c2b0)和 Tony Shin 的简短[视频](https://www.youtube.com/watch?v=uvix7XwAjOg)。

# 第八节：序列之外的生成建模

**目标：** 调查构建非顺序内容生成（如图像）的主题，从 GANs 到扩散模型。

到目前为止，我们所看的一切都集中在文本和序列预测上，使用语言模型，但许多其他“生成式 AI”技术需要学习具有较少顺序结构的分布（例如图像）。在这里，我们将检查用于生成建模的几种非 Transformer 架构，从简单的混合模型开始，以扩散模型结束。

## 分布建模

回想我们第一次看到语言模型作为简单的二元分布，在分布建模中最基本的事情就是只是计算你的数据集中的共现概率，并重复它们作为真实情况。这个想法可以扩展到条件采样或分类，称为“朴素贝叶斯”([博客文章](https://mitesh1612.github.io/blog/2020/08/30/naive-bayes) [视频](https://www.youtube.com/watch?v=O2L2Uv9pdDA))，通常是在机器学习入门课程中涵盖的最简单算法之一。

学生通常接下来学习的是高斯混合模型及其期望最大化算法；高斯混合模型 + 期望最大化算法。这篇[博客文章](https://mpatacchiola.github.io/blog/2020/07/31/gaussian-mixture-models.html)和这篇[视频](https://www.youtube.com/watch?v=DODphRRL79c)提供了相当好的概述；这里的核心思想是假设数据分布可以被近似为多元高斯分布的混合。如果可以假设单个组是近似高斯分布的，GMMs 也可以用于聚类。

虽然这些方法在表示复杂结构如图像或语言方面不是很有效，但相关想法将作为我们接下来将要看到的一些更先进方法的部分出现。

## 变分自编码器

自编码器和变分自编码器被广泛用于学习数据分布的压缩表示，并且对于“去噪”输入也很有用，这在讨论扩散时将发挥作用。一些不错的资源：

+   “深度学习”书籍中的[“自编码器”](https://www.deeplearningbook.org/contents/autoencoders.html)章节

+   [博客文章](https://lilianweng.github.io/posts/2018-08-12-vae/) 来自李莲英

+   Arxiv Insights 的[视频](https://www.youtube.com/watch?v=9zKuYvjFFS8)

+   Prakash Pandey 关于 VAEs 和 GANs 的[博客文章](https://towardsdatascience.com/deep-generative-models-25ab2821afd3)

## 生成对抗网络

生成对抗网络（GANs）背后的基本思想是模拟两个神经网络之间的“游戏”——生成器希望创建的样本能够被判别器识别为与真实数据不可区分，而判别器则希望识别生成的样本，并且两个网络会持续训练，直到达到平衡（或所需的样本质量）。根据冯·诺伊曼的零和博弈最小-最大定理，如果你假设梯度下降可以找到全局最小值，并且允许两个网络任意增长，你基本上会得到一个“定理”，它承诺 GANs 在学习分布方面是成功的。诚然，在实际情况中这两者都不是字面意义上的真实，但 GANs 确实往往非常有效（尽管近年来它们在一定程度上已经不再受欢迎，部分原因是同时训练的不稳定性）。

资源：

+   Paperspace 的[“生成对抗网络（GANs）完整指南”](https://blog.paperspace.com/complete-guide-to-gans/)

+   [“生成对抗网络（GANs）：端到端介绍”](https://www.analyticsvidhya.com/blog/2021/10/an-end-to-end-introduction-to-generative-adversarial-networksgans/) by

+   [深度学习，第二十章 - 生成模型](https://www.deeplearningbook.org/contents/generative_models.html)（理论导向）

## 条件 GANs

条件 GANs 是我们从传统的“分布学习”开始，走向更接近交互式生成工具（如 DALL-E 和 Midjourney）的地方，它结合了文本-图像的多模态。一个关键的想法是学习“表示”（在文本嵌入或自编码器的意义上）更抽象，并且可以应用于文本或图像输入。例如，你可以想象通过嵌入文本并将其与图像连接起来，在（图像，标题）对上训练一个传统的 GAN，这样它就可以学习图像和标题的联合分布。请注意，如果输入的一部分（图像或标题）是固定的，这隐式地涉及到学习条件分布，并且这可以扩展以实现自动配图（给定一个图像）或图像生成（给定一个标题）。这个设置有许多变体，各有不同的功能和特性。VQGAN+CLIP 架构值得了解，因为它曾是早期从输入文本生成“AI 艺术”的主要流行来源之一。

资源：

+   Paperspace 的[“实现条件生成对抗网络”](https://blog.paperspace.com/conditional-generative-adversarial-networks/)博客文章

+   Saul Dobilas 的[“条件生成对抗网络——如何控制 GAN 输出”](https://towardsdatascience.com/cgan-conditional-generative-adversarial-network-how-to-gain-control-over-gan-outputs-b30620bd0cc8)

+   [“The Illustrated VQGAN”](https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/) by LJ Miranda

+   [“Using Deep Learning to Generate Artwork with VQGAN-CLIP”](https://www.youtube.com/watch?v=Ih4qOakCZD4) from Paperspace

## 正态化流

正态化流的目的是学习一系列可逆变换，将高斯噪声与输出分布之间建立联系，避免在 GANs 中需要“同时训练”，并且在许多领域的生成建模中已经变得非常流行。在这里，我仅推荐 Lilian Weng 的[“Flow-based Deep Generative Models”](https://lilianweng.github.io/posts/2018-10-13-flow-models/)作为概述——我本人并没有深入研究正态化流，但它们出现的频率足够高，可能值得注意。

## 扩散模型

扩散模型（如 StableDiffusion）背后的一个核心思想是迭代引导应用去噪操作，将随机噪声逐步细化成越来越像图像的东西。扩散起源于随机微分方程和统计物理的世界——与“薛定谔桥”问题和概率分布的最优传输相关——如果你想要全面理解，那么基本上不可避免地需要大量的数学知识。对于相对轻松的介绍，可以参考 Antony Gitau 的[“A friendly Introduction to Denoising Diffusion Probabilistic Models”](https://medium.com/@gitau_am/a-friendly-introduction-to-denoising-diffusion-probabilistic-models-cc76b8abef25)。如果你想要更多数学内容，可以查看 Lilian Weng 的[“What are Diffusion Models?”](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)，以深入了解。如果你对代码和图片（但仍然需要一些数学知识）更感兴趣，可以查看 Hugging Face 的[“The Annotated Diffusion Model”](https://huggingface.co/blog/annotated-diffusion)，以及关于 LoRA 微调扩散模型的这篇 Hugging Face[博客文章](https://huggingface.co/blog/lora)。

# 第 IX 节：多模态模型

**目标：** 调查模型如何同时使用输入和输出的多种模态（文本、音频、图像）。

注意：这是我最不熟悉的一组主题，但为了完整性，我还是想包括在内。在这里，我将减少评论和建议，并在我认为自己有更紧密的故事要讲述时再回来补充。Chip Huyen 的博客文章[“Multimodality and Large Multimodal Models (LMMs)”](https://huyenchip.com/2023/10/10/multimodal.html)提供了一个很好的广泛概述（或者 Kevin Musgrave 的[“How Multimodal LLMs Work”](https://www.determined.ai/blog/multimodal-llms)提供了一个更简洁的概述）。 

## 词汇化超越文本

标记化（tokenization）的概念不仅与文本相关；音频、图像和视频也可以“标记化”以用于 Transformer 风格的架构，并且需要在标记化和其他方法（如卷积）之间考虑一系列权衡。接下来的两个部分将更深入地探讨视觉输入；AssemblyAI 的这篇 [博客文章](https://www.assemblyai.com/blog/recent-developments-in-generative-ai-for-audio/) 讨论了音频标记化和序列模型中的表示，适用于音频生成、文本到语音和语音到文本等应用。

## VQ-VAE

近年来，VQ-VAE 架构在图像生成方面变得非常流行，并且至少是 DALL-E 的早期版本的基础。

资源：

+   [“理解 VQ-VAE（DALL-E 解释第一部分）”（来自 Machine Learning @ Berkeley 博客）](https://mlberkeley.substack.com/p/vq-vae)

+   [“它为什么这么好？（DALL-E 解释第二部分）”（来自 Machine Learning @ Berkeley 博客）](https://mlberkeley.substack.com/p/dalle2)

+   [“理解向量量化变分自编码器（VQ-VAE）”（Shashank Yadav 撰写）](https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a)

## 视觉 Transformer

视觉 Transformer 将 Transformer 架构扩展到图像和视频等领域，并且因为自动驾驶汽车以及多模态 LLMs 的应用而变得流行。d2l.ai 书籍中有一个很好的 [部分](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html)，介绍了它们的工作原理。

[“广义视觉语言模型”（Lilian Weng 撰写）](https://lilianweng.github.io/posts/2022-06-09-vlm/) 讨论了多种不同的方法，用于训练多模态 Transformer 风格的模型。

Encord 博客上的文章 [“视觉语言模型指南”](https://encord.com/blog/vision-language-models-guide/) 概述了多种混合文本和视觉的架构。

如果你对于视觉 Transformer 的实际大规模训练建议感兴趣，苹果公司的 MM1 [论文](https://arxiv.org/abs/2403.09611) 分析了几个架构和数据权衡，并提供了实验证据。

[“人工神经网络中的多模态神经元”（来自 Distill.pub）](https://distill.pub/2021/multimodal-neurons/) 有一些非常有趣的多模态网络概念表示的视觉化。

##### 引用

如果你引用了这里展示的任何特定内容，请直接引用。然而，如果你希望将其作为一项广泛的调查引用，可以使用下面的 BibTeX 引用。

```
@misc{Brown24GAIHB,
  author = {Brown, William},
  title = {Generative AI Handbook: A Roadmap for Learning Resources},
  year = 2024,
  url = {https://genai-handbook.github.io}
} 
```
