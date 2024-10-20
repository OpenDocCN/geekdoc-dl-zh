# 最佳物流替代方案(2022 年更新)

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/the-best-mlflow-alternatives>

[MLflow](https://web.archive.org/web/20221208050817/https://mlflow.org/) 是一个帮助管理整个机器学习生命周期的开源平台。这包括实验，也包括再现性、部署和存储。这四个元素中的每一个都由一个 MLflow 组件表示:跟踪、项目、模型和注册。

这意味着与 MLflow 合作的数据科学家能够跟踪实验，组织实验，为其他 ML 工程师描述实验，并将其打包到机器学习模型中。在本文中，我们主要关注 MLflow 的实验跟踪功能，并概述其最佳替代方案。

虽然 MLflow 是一个很好的工具，但有些东西可能会更好，尤其是在大型团队中和/或您运行的实验数量非常大的情况下。

你最关心的是什么？MLflow 的主要弱点是什么？

*   **缺少用户管理能力**使得难以处理对不同项目或角色(经理/机器学习工程师)的访问权限。正因为如此，而且没有与他人共享 UI 链接的选项，团队协作在 MLflow 中也很有挑战性。

*   尽管最近有所改进，但在保存实验仪表板视图或按实验参数(模型架构)或属性(数据版本)对运行进行分组时，**并没有提供完全的可定制性**。当你有很多人在同一个项目上工作或者你正在运行数以千计的实验时，这些是非常有用的。
*   说到大量的实验，当你真的想探索你所有的跑步时，UI **会变得相当慢**。
*   除非您想使用 Databricks 平台**,否则您需要自己维护 MLflow 服务器**。这带来了典型的障碍，如访问管理、备份等，更不用说这非常耗时。
*   [开源社区](https://web.archive.org/web/20221208050817/https://stackoverflow.com/questions/tagged/mlflow)充满活力，但是**没有专门的用户支持**在你需要的时候伸出援手。
*   MLflow 非常适合通过 Python 或 R 脚本运行实验，但 Jupyter 笔记本体验并不完美，尤其是如果您想要跟踪机器学习生命周期的一些附加部分，如探索性数据分析或结果探索。
*   一些功能，如记录资源消耗(CPU，GPU，内存)或滚动大量的图像预测或图表还没有出现。

[ML 实验跟踪](/web/20221208050817/https://neptune.ai/experiment-tracking)ML flow 中的功能为那些愿意维护实验数据后端、跟踪 UI 服务器并且不运行大量实验的个人用户或团队带来了巨大的价值。

如果上面提到的一些东西对你和你的团队很重要，你可能想要寻找补充的或者替代的工具。幸运的是，有许多工具可以提供这些缺失的部分或大部分。

在这篇文章中，基于 reddit 上的一些讨论和 T2 的比较，我们给出了 MLflow 的最佳替代方案。

我们认为，以下是 MLflow 的最佳替代方案:

1.  [海王星](#Neptune)
2.  [权重&偏差](#WandB)
3.  [Comet.ml](#Comet)
4.  [Valohai](#Valohai)
5.  [张量板](#Polyaxon)

<https://web.archive.org/web/20221208050817im_/https://neptune.ai/wp-content/uploads/Example-dashboard-metadata-structure.mp4>

*[Source](https://web.archive.org/web/20221208050817/https://app.neptune.ai/o/common/org/example-project-tensorflow-keras/e/TFKERAS-14/dashboard/artifacts-9cc55d46-8e2b-476e-8ce7-f30ff1b01549)* 

Neptune 是一个[元数据存储库](/web/20221208050817/https://neptune.ai/blog/ml-metadata-store)——它充当从数据版本化、实验跟踪到模型注册和监控的 [MLOps](/web/20221208050817/https://neptune.ai/blog/mlops-what-it-is-why-it-matters-and-how-to-implement-it-from-a-data-scientist-perspective) 工作流不同部分之间的连接器。Neptune 使得存储、组织、显示和比较 ML 模型生命周期中生成的所有元数据变得容易。

Neptune 使得存储、组织、显示和比较 ML 模型生命周期中生成的所有元数据变得容易。

您可以记录指标、超参数、交互式可视化、视频、代码、数据版本、[和更多](https://web.archive.org/web/20221208050817/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#tables)，并以定制的结构对其进行组织。一旦登录，一切都在一个直观和干净的用户界面中可见，您可以在那里进行分析和比较。

您还可以[创建包含所有这些元数据的定制仪表板](https://web.archive.org/web/20221208050817/https://docs.neptune.ai/you-should-know/displaying-metadata#creating-dashboards)，并与您的同事、团队经理甚至外部利益相关者共享它们。以下是这种仪表板的一个示例:

有四种不同的比较视图可用——图表、平行坐标、并列表格仪表板和工件比较部分。因此，您可以轻松地评估模型并选择性能最佳的模型。

海王星在生产阶段也非常有用。有了所有记录的元数据，您就知道模型是如何创建的，以及如何再现它。

**海王星——概要:**

如果你想看海王星在行动，检查[这个现场笔记本](https://web.archive.org/web/20221208050817/https://colab.research.google.com/github/neptune-ai/examples/blob/master/how-to-guides/how-it-works/notebooks/Neptune_API_Tour.ipynb)或[这个例子项目](https://web.archive.org/web/20221208050817/https://app.neptune.ai/o/common/org/example-project-tensorflow-keras/experiments?split=tbl&dash=charts&viewId=44675986-88f9-4182-843f-49b9cfa48599)(不需要注册)，只是玩它。

### MLflow vs 海王星

这些工具之间的主要区别是 MLflow 是一个开源解决方案，而 Neptune 是一个托管云服务。它影响了 MLflow 和 Neptune 工作的各个方面。如果你正在寻找一个免费的开源工具，涵盖了广泛的 ML 生命周期步骤，MLflow 可能是你正确的选择。但是您应该记住，尽管 MLflow 可以免费下载，但它确实会产生与维护整个基础设施相关的成本。

如果你更喜欢专注于 ML 过程，而把托管工作留给其他人，Neptune 是一个不错的选择。对于月费，您可以获得出色的用户支持，快速和简单的设置，您不必担心维护，并且该工具伸缩性良好。另外，Neptune 具有用户管理功能，因此它在团队环境中会工作得更好。

查看 Neptune 和 MLflow 的深入对比[。](/web/20221208050817/https://neptune.ai/vs/mlflow)

阅读 Zoined 的[案例研究，了解他们为什么选择 Neptune 而不是 MLflow。](/web/20221208050817/https://neptune.ai/customers/zoined)

WandB 专注于深度学习。用户使用 Python 库跟踪应用程序的实验，并且作为一个团队，可以看到彼此的实验。

与 MLflow 不同，WandB 是一种托管服务，允许您在一个地方备份所有实验，并与团队合作完成一个项目-工作共享功能可供使用。

与 MLflow 类似，在 WandB 中，用户可以记录和分析多种数据类型。

**权重&偏差—汇总**:

*   处理用户管理
*   出色的用户界面让用户可以很好地可视化、比较和组织他们的跑步。
*   团队共享工作:团队共享的多种特性。
*   与其他工具的集成:几个可用的开源集成
*   SaaS/本地实例可用:是/是
*   额外的好处:WandB 记录了模型图，因此您可以在以后检查它。

### MLflow 与重量和偏差

与 Neptune 类似，Weight & Biases 提供了其工具的托管版本。与 MLflow 相反，ml flow 是开源的，需要在自己的服务器上维护。Weights & Biases 提供了实验跟踪、数据集版本控制和模型管理功能，而 MLflow 几乎涵盖了整个 ML 生命周期。最后，WandB 提供了用户管理特性，当你在团队中工作时，这些特性可能对你很重要。

Comet 是一个元机器学习平台，用于跟踪、比较、解释和优化实验和模型。

就像许多其他工具一样——例如 Neptune(Neptune-client specific)或 WandB——Comet 提出了一个开源 Python 库，允许数据科学家将他们的代码与 Comet 集成，并开始在应用程序中跟踪工作。

由于它提供云托管和自托管，用户可以有团队项目，并保存实验历史的备份。

Comet 正在通过预测性早期停止(免费版软件不提供)和神经结构搜索(未来)向更自动化的 ML 方法靠拢。

**彗星—摘要**:

*   处理用户管理
*   团队共享工作:团队共享的多种特性。
*   与其他工具的集成:应该由用户手动开发
*   SaaS/本地实例可用:是/是
*   额外收获:显示平行图来检查参数和指标之间的关系模式

### MLflow vs 彗星

Comet 附带了用户管理特性，并允许在团队内部共享项目——这是 MLfow 中所缺少的。它还提供托管和内部设置，而 MLflow 仅作为开源解决方案提供，需要您在自己的服务器上进行维护。

当涉及到跟踪和可视化实验时，Valohai 采取了一种稍微不同的方法。

该平台为机器学习提出了编排、版本控制和管道管理——简单来说，它们涵盖了 MLflow 在日志记录方面的工作，并额外管理您的计算基础设施。

与 MLflow 一样，用户可以轻松检查和比较多次运行。同时，与众不同的是能够自动启动和关闭用于培训的云机器。

Valohai 允许您使用任何编程语言进行开发——包括 Python 和 R——这对于在固定技术堆栈中工作的团队来说非常方便。

**瓦罗海—摘要:**

*   处理用户管理
*   团队共享工作:多种特性
*   与其他工具的集成:文档中提供的集成示例
*   SaaS/本地实例可用:是/是
*   额外收获:有了训练的基础设施，你可以在瓦罗海管理的环境中进行实验。

### MLflow vs Valohai

根据 [Valohai 自己的对比](https://web.archive.org/web/20221208050817/https://valohai.com/blog/kubeflow-vs-mlflow/)，Valohai 在没有任何设置的情况下提供了类似 MLflow 的实验跟踪。与 MLflow 类似，Valohai 涵盖了 MLOps 领域的很大一部分(包括实验跟踪、模型管理、机器编排和流水线自动化)，但它是一个托管平台，而不是一个开源解决方案。

TensorBoard 是一个用于 TensorFlow 的开源可视化工具包，允许您分析模型训练运行。往往是 TensorFlow 用户的首选。TensorBoard 允许您可视化机器学习实验的各个方面，如度量或模型图，以及查看 tensors 的直方图等。

除了流行的开源版 TensorBoard，还有 TensorBoard.dev，它可以在托管服务器上免费使用。

TensorBoard.dev 允许你上传并与任何人分享你的 ML 实验结果。与 TensorBoard 相比，这是一个重要的升级，协作功能在那里是缺失的。

**tensor board—摘要:**

*   与处理图像相关的成熟功能
*   假设工具(WIT)，这是一个易于使用的界面，用于扩展对黑盒分类和回归 ML 模型的理解
*   提供社区支持的强大用户社区。

### MLflow 与 TensorBoard

这两个工具都是开源的，在处理任何问题方面都受到各自社区的支持。主要区别似乎在于它们各自提供的功能范围。TensorBoard 被描述为 TensorFlow 的可视化工具包，因此它很好地服务于可视化，它允许您跟踪实验并比较它们(有限的功能)。另一方面，MLflow 被证明在 ML 生命周期的更多阶段是有用的。这两个工具都缺乏用户管理和团队共享功能(TensorBoard.dev 提供共享功能，但无法管理数据隐私)。

## 结论

MLflow 是一个很棒的工具，但是有一些它不具备的功能。所以有必要看看外面还有什么。在这个概述中，我们提到了 5 个可能是很好的替代工具，并检查了缺少的框。

如果你寻找 MLflow 替代品的主要原因是缺少协作和用户管理功能，你应该检查 Neptune、Weights & Biases、Comet 或 Valohai。如果您不想自己维护实验跟踪工具，所有这些工具都可以作为托管应用程序使用。

如果你想坚持使用开源工具，TensorBoard 可能是适合你的工具，但你应该记住，就功能而言，它不如 MLflow 先进。

最后，如果你不需要一个几乎覆盖整个 ML 生命周期的工具(像 MLflow 或者 Valohai)，我们推荐你去查 Neptune，Weight & Biases，或者 Comet。

在任何情况下，确保替代解决方案符合您的需求并改进您的工作流程。希望这篇文章能帮你找到。祝你好运！