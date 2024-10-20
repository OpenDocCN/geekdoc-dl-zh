# 真实世界的 MLOps 示例:超因子中的模型开发

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/mlops-examples-model-development-in-hypefactors>

在“真实世界的 MLOps 示例”系列的第一部分中，[MLOps 工程师 Jules Belveze](https://web.archive.org/web/20221206125202/https://www.linkedin.com/in/jules-belveze) 将带您了解 [Hypefactors](https://web.archive.org/web/20221206125202/https://hypefactors.com/) 的模型开发流程，包括他们构建的模型类型、他们如何设计培训渠道，以及您可能会发现的其他有价值的细节。享受聊天！

## 公司简介

![Hypefactors](img/21c76acf519fb7a33bb0e8383f759279.png)

*Media monitoring dashboard in Hypefactors | [Source](https://web.archive.org/web/20221206125202/https://hypefactors.com/dashboard/)*

[Hypefactors](https://web.archive.org/web/20221206125202/https://hypefactors.com/) 提供一体化媒体智能解决方案，用于管理公关和沟通、跟踪信任度、产品发布以及市场和金融情报。他们运营着大型数据管道，实时传输世界各地的媒体数据。人工智能用于许多以前手动执行的自动化操作。

## 嘉宾介绍

### 你能向我们的读者介绍一下你自己吗？

嘿，斯蒂芬，谢谢你邀请我！我叫朱尔斯。我 26 岁。我在巴黎出生和长大，目前住在哥本哈根。

### 嘿朱尔斯！谢谢你的介绍。告诉我你的背景以及你是如何成为催眠师的。

我拥有法国大学的统计学和概率学士学位以及普通工程学硕士学位。除此之外，我还毕业于丹麦的丹麦技术大学，主修深度学习的数据科学。我对多语言自然语言处理非常着迷(并因此专攻它)。在微软的研究生学习期间，我还研究了高维时间序列的异常检测。

今天，我在一家名为 Hypefactors 的媒体智能技术公司工作，在那里我开发 NLP 模型，帮助我们的用户从媒体领域获得洞察力。对我来说，目前的工作是有机会从原型一直到产品进行建模。我想你可以叫我书呆子，至少我的朋友是这么形容我的，因为我大部分空闲时间不是编码就是听迪斯科黑胶。

## 超因子模型开发

### 你能详细说明你在 Hypefactors 建立的模型类型吗？

尽管我们也有计算机视觉模型在生产中运行，但我们主要为各种用例构建 [NLP(自然语言处理)](/web/20221206125202/https://neptune.ai/blog/category/natural-language-processing)模型。我们需要覆盖多个国家和处理多种语言。多语言方面使得用“经典机器学习”方法开发变得困难。我们在[变形金刚库](https://web.archive.org/web/20221206125202/https://github.com/huggingface/transformers)的基础上打造深度学习模型。

我们在生产中运行各种模型，从跨度提取或序列分类到文本生成。这些模型旨在服务于不同的用例，如主题分类、情感分析或总结。

### 你能从 Hypefactors 中挑选一个用例，从头到尾地向我介绍一下你的机器学习工作流程吗？

我们所有的机器学习项目都倾向于遵循类似的生命周期。我们要么启动一个 ML 项目来改善我们用户的体验，要么为我们客户的体验添加一个有意义的特性，然后将其转化为一个 ML 任务。

让我向您介绍一下我们最新添加的命名实体识别模型的流程。我们从使用开箱即用的模型制作 POC(概念验证)开始，但是由于我们的生产数据和模型微调的数据之间存在一些偏差，我们必须按照我们整齐定义的注释准则在内部标记我们的数据。然后，我们开始设计一个相对简单的模型，并对其进行迭代，直到我们达到与 [SOTA](https://web.archive.org/web/20221206125202/https://paperswithcode.com/task/nested-named-entity-recognition) 相当的性能。然后对模型进行了推理优化，并在现实生活条件下进行了测试。

基于 QA(质量保证)会议的结果，在将模型部署到生产环境之前，我们迭代数据(例如，细化注释指南)以及模型(例如，提高其精度)。一旦部署完成，我们的模型将受到持续监控，并通过主动学习进行定期改进。

![ML workflow at Hypefactors](img/11d33ca200e292b82700c3b916f1b4a1.png)

*ML workflow at Hypefactors | Source: Author*

### 你能描述一下你的模型开发工具吗？

我们使用几种不同的工具进行模型开发。我们最近将我们的代码库移植到了 PyTorch Lightning 和 T2 Hydra 的组合中，以减少样板文件。前者支持四个主要组件之间的结构化代码:

## 

*   1 数据
*   2 型号
*   3 优化
*   4 非必需品

PyTorch Lightning 抽象掉了所有的样板代码和工程逻辑。**自采用以来，我们已经注意到在迭代模型或启动新的概念验证(POC)时速度显著加快**。

此外，Hydra 帮助您“优雅地”编写配置文件。为了帮助我们设计和实现神经网络，我们非常依赖 Transformer 库。在跟踪实验和数据版本化时，我们使用 [Neptune.ai，它与 Lightning](https://web.archive.org/web/20221206125202/https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning) 有着平滑的集成。最后，我们选择 Metaflow 而不是其他工具来设计和运行我们的训练管道。

![Hypefactors model training and evaluation stack](img/5516f9d808a749ba4262f20e4b5ad2b7.png)

*Hypefactors model training and evaluation stack | Source: Author*

### 你的 NLP 用例是如何驱动培训管道设计选择的？

运行端到端的 NLP 训练管道需要大量的计算能力。对我来说，自然语言处理中最艰巨的任务之一就是数据清洗。当处理直接从网络或社交媒体中提取的文本数据时，这变得更加重要。即使像 [B](https://web.archive.org/web/20221206125202/https://arxiv.org/abs/1810.04805) [ERT](https://web.archive.org/web/20221206125202/https://arxiv.org/abs/1810.04805) 或 [GPT](https://web.archive.org/web/20221206125202/https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 这样的大型语言模型相当健壮，数据清理也是至关重要的一步，因为这可以直接影响模型的性能。这意味着相当繁重的预处理，因此需要并行计算。此外，微调预先训练的语言模型需要在针对计算优化的硬件(例如，GPU、TPU 或 IPU)上运行训练。

此外，我们对待 NLP 模型的评估不同于“常规”模型。尽管评估指标很好地代表了模型的性能，但是我们不能仅仅依赖它们。这个问题的一个很好的例子是用于抽象概括的 [ROUGE](https://web.archive.org/web/20221206125202/https://aclanthology.org/W04-1013.pdf) score。尽管 ROUGE 评分可以很好地代表摘要和原文之间的 n 元语法重叠，但仍需要人工检查来评估语义和事实的准确性。这使得不需要任何人工干预的全自动管道变得非常困难。

### 您的培训渠道使用什么工具，它们的主要组成部分是什么？

我们最近开始设计可重用的端到端培训管道，主要是为了节省我们的时间。我们的管道是使用网飞的 Metaflow 设计的，它们都共享相同的构建模块。

在处理之前，我们首先从我们的标注工具中获取新的手工标注的数据。处理后，数据集将与配置文件一起进行版本化。

我们还保存代码和 git 散列，使得完全相同的实验重现成为可能。然后我们开始训练想要的模型。

在训练结束时，最佳重量会保存到内部工具中，并生成一份训练报告，使我们能够将这次跑步与之前的跑步进行比较。最后，我们将检查点导出到 ONNX，并优化推理模型。

*参见:[纵向扩展 PyTorch 推理:用 ONNX 运行时服务数十亿次日常 NLP 推理【微软开源博客】](https://web.archive.org/web/20221206125202/https://cloudblogs.microsoft.com/opensource/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime/)*

我们的管道是这样设计的，任何有一点技术知识的人都可以复制一个实验，或者用新注释的数据或不同的配置训练现有模型的新版本。

### 什么样的工具在外面很容易获得，什么样的工具需要在内部实现？

关于建模方面，我们非常依赖变形金刚库。然而，由于我们用例的特殊性(web 数据和多语言需求)，我们在它的基础上构建模型。使用如此庞大的模型的一个缺点是它们很难扩展。有相当多的工具可以用来缩小基于 transformer 的模型(例如，DeepSpeed、DeepSparse)，但是它们受到基本模型的限制。我们已经实现了一个内部工具，使我们能够训练各种早期存在的架构，执行模型提取、修剪或量化。

[实验跟踪和元数据存储](/web/20221206125202/https://neptune.ai/blog/best-ml-experiment-tracking-tools)空间具有大量易于使用的完整工具，因此我们没有必要重新发明轮子。

ML 工作流编排器也是如此。我们实际上花了相当多的时间挑选一个足够成熟并且学习曲线不太陡的。我们最终选择了 [Metaflow](https://web.archive.org/web/20221206125202/https://metaflow.org/) 而不是 [Kubeflow](https://web.archive.org/web/20221206125202/https://www.kubeflow.org/) 或 [MLFlow](https://web.archive.org/web/20221206125202/https://mlflow.org/) ，因为它易于采用，可用的功能以及不断增长的社区。

总的来说，对于机器学习工作流程的所有不同构建模块，有太多的工具可用，这也可能是压倒性的。

### 您使用什么类型的硬件来训练您的模型，您使用任何类型的并行计算吗？

我们所有的训练流程都在配备一个或多个 GPU 的机器上运行，具体取决于给定任务所需的计算能力。PyTorch Lightning 使得从单 GPU 切换到多 GPU 变得相对容易，并且还提供了各种后端和分布式模式。NLP 任务需要相对繁重的预处理。因此，我们通过 DDP PyTorch 模式使用分布式训练，该模式使用多处理而不是线程来克服 Python 的 GIL 问题。与此同时，我们试图在设计模型时最大限度地利用张量运算，以充分利用 GPU 的能力。

由于我们只对模型进行微调，因此不需要我们执行分片训练。然而，当我们需要快速迭代时，我们偶尔会在 TPU 上训练模型。

说到数据处理，我们使用“[数据集](https://web.archive.org/web/20221206125202/https://huggingface.co/docs/datasets/index)”，这是一个构建在 Apache Arrow 之上的 Python 库，支持更快的 I/O 操作。

### 你希望在不久的将来出现什么工具？

我认为每个机器学习工程师都会同意，目前缺少的是一个统治一切的工具。人们需要至少 5 到 6 种不同的工具用于训练，这使得维护和学习变得很困难。我真的希望我们将很快看到包含多个步骤的新兴工具。

在 NLP 领域，我看到越来越多的人专注于确保标注质量，但是我们仍然受到任务性质的限制。发现错误的标签是一项艰巨的任务，但一个可靠的工具可以真正改变游戏规则。我想大多数数据科学家都会同意，数据检查是一项非常耗时的任务。

此外，工作流程的一个重要方面是模型测试。在 NLP 中，找到保证模型忠实性的相关度量标准是非常棘手的。有几个工具出现了(例如，我们开始使用微软的"[清单](https://web.archive.org/web/20221206125202/https://github.com/marcotcr/checklist)")，但在我看来，在这个领域拥有更广泛的工具会很有趣。

对于每项任务，我们的数据专家都会提出一组行为测试案例，从相对简单到更复杂，分为“测试方面”然后，我们使用*清单*生成不同测试的摘要，并比较实验。这同样适用于模型的可解释性。

* * *

感谢 Jules Belveze 和 Hypefactors 的团队与我们一起创作了这篇文章！