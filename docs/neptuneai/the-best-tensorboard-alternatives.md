# 最佳的冲浪板替代品

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/the-best-tensorboard-alternatives>

TensorBoard 是 TensorFlow 的开源可视化工具包，可以让您分析模型训练运行。它允许您跟踪和可视化机器学习实验的各个方面，如度量或模型图，查看张量的权重和偏差直方图等。

为此，TensorBoard 符合[对工具的持续需求的趋势，以跟踪和可视化机器学习实验](https://web.archive.org/web/20221208211451/https://www.welcometothejungle.com/en/articles/btc-data-visualization-machine-learning)。虽然它可以让你深入研究实验，但有一些 TensorBoard 没有的功能，这些功能在实验跟踪过程中非常有用。

**使用 TensorBoard 时，您可能面临的主要挑战包括:**

*   TensorBoard 在更多的实验中没有很好地扩展；
*   当你想比较多次运行时，用户体验远非完美；
*   TensorBoard 在本地工作，所以当你在不同的机器上操作时，很难跟踪所有的事情；
*   与他人分享结果是一件痛苦的事情——你只需要找到一个解决方法(比如截图)，因为没有现成的解决方案。

总的来说，当你刚刚开始实验跟踪和可视化，或者你没有运行大量实验时，TensorBoard 似乎是一个很好的工具。用 [TensorFlow](https://web.archive.org/web/20221208211451/https://www.tensorflow.org/) 训练的时候也很方便(不然没那么好设置)。但是，它不如市场上的其他工具先进，并且不能在团队环境中提供最佳体验。

如果你在 TensorBoard 工作时遇到了这些问题中的任何一个，或者只是想看看外面还有什么，那么你来对地方了。

**以下是你应该看看的 TensorBoard 的最佳选择:**

1.  [海王星](#neptune)
2.  [公会 AI](#guildai)
3.  [神圣的](#sacred)
4.  [权重&偏差](#wandb)
5.  [彗星](#comet)

<https://web.archive.org/web/20221208211451im_/https://neptune.ai/wp-content/uploads/Example-dashboard-metadata-structure.mp4>

*[Source](https://web.archive.org/web/20221208211451/https://app.neptune.ai/o/common/org/example-project-tensorflow-keras/e/TFKERAS-14/dashboard/artifacts-9cc55d46-8e2b-476e-8ce7-f30ff1b01549)* 

Neptune 是为进行大量实验的研究和生产团队构建的 [MLOps](/web/20221208211451/https://neptune.ai/blog/mlops-what-it-is-why-it-matters-and-how-to-implement-it-from-a-data-scientist-perspective) 的[元数据存储库](/web/20221208211451/https://neptune.ai/blog/ml-metadata-store)。

它为您提供了一个记录、存储、显示、组织、比较和查询所有模型构建元数据的单一位置。这包括指标和参数，还包括模型检查点、图像、视频、音频文件、数据版本、交互式可视化、[和更多](https://web.archive.org/web/20221208211451/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display)。您还可以[创建包含所有这些元数据的定制仪表板](https://web.archive.org/web/20221208211451/https://docs.neptune.ai/you-should-know/displaying-metadata#creating-dashboards)，并与您的同事、团队经理甚至外部利益相关者共享它们。以下是这种仪表板的一个示例:

如果你在团队中工作，海王星是完美的。它允许您与多个团队成员一起创建项目，管理用户访问，共享工作，并将所有结果备份在一个位置。

评估模型和[比较运行](https://web.archive.org/web/20221208211451/https://docs.neptune.ai/you-should-know/comparing-runs)也很容易，因为有四种不同的比较视图可用——图表、平行坐标、并排表格仪表板和工件比较部分。

**海王星——概要:**

*   通过成千上万次运行进行扩展——无论您有 5 次还是 100 次实验，Neptune 都能提供同样出色的用户体验；
*   UI 很干净，易于导航，非常直观；
*   在[本地版本](https://web.archive.org/web/20221208211451/https://docs.neptune.ai/administration/on-premises-deployment)中可用，但也作为[托管的应用](https://web.archive.org/web/20221208211451/https://docs.neptune.ai/getting-started/installation)；
*   快速简单的设置和出色的客户支持；
*   [团队](https://web.archive.org/web/20221208211451/https://docs.neptune.ai/you-should-know/collaboration-in-neptune)中的协作受到多种特性的有力支持。

如果你想看海王星在行动，检查[这个现场笔记本](https://web.archive.org/web/20221208211451/https://colab.research.google.com/github/neptune-ai/examples/blob/master/how-to-guides/how-it-works/notebooks/Neptune_API_Tour.ipynb)或[这个例子项目](https://web.archive.org/web/20221208211451/https://app.neptune.ai/o/common/org/example-project-tensorflow-keras/experiments?split=tbl&dash=charts&viewId=44675986-88f9-4182-843f-49b9cfa48599)(不需要注册)，只是玩它。

### TensorBoard vs 海王星

TensorBoard 是一个开源工具，可以帮助跟踪和可视化 ML 运行。另一方面，Neptune 是一个托管解决方案，它在实验跟踪领域提供了更多的特性，并且还提供了模型注册、模型监控和数据版本控制功能。Neptune 支持团队协作，比 TensorBoard 更具可扩展性。

检查 TensorBoard 和 Neptune 之间的深度[比较。](/web/20221208211451/https://neptune.ai/vs/tensorboard)

阅读 InstaDeep 的[案例研究，了解他们为什么从 TensorBoard 转行到 Neptune。](/web/20221208211451/https://neptune.ai/customers/instadeep)

这是一个开源的机器学习平台，用于运行和比较模型训练例程。

它主要是一个 CLI 工具，让你以系统的方式运行和比较训练作业，而 Guild AI 则捕获源代码、日志和生成的文件。

与 TensorBoard 不同，它不限于 tensor flow/深度学习工作。相反，Guild AI 是平台和编程语言不可知的，所以你可以在你当前的技术栈中自由使用它。

如果你是一个 CLI 爱好者，这可以成为你的一个工具，大多数使用是通过终端中的命令。

**帮会 AI——总结:**

*   实验跟踪:任何模型训练，任何编程语言
*   在团队中共享工作:不支持
*   与其他工具集成:不支持
*   SaaS/本地实例是否可用:否/是
*   奖励:准备充分的文档

### TensorBoard vs 公会 AI

公会 AI 的范围比 TensorBoard 的要广得多。Guild AI 允许您跟踪实验、调整超参数、自动化管道等，而 TensorBoard 主要用于跟踪和可视化运行。公会人工智能可以运行在任何云或本地环境。另一方面，TensorBoard 是本地托管的。

研究机构 [IDSIA](https://web.archive.org/web/20221208211451/http://www.idsia.ch/) (瑞士人工智能实验室)开发的另一个开源工具。Sacred 是一个 Python 库，帮助配置、组织、记录和复制实验。

Sacred 提供了一种使用配置的编程方式。观察者的概念允许用户跟踪与实验相关的各种类型的数据。

神圣的一个好处是它有自动播种功能——在需要重现实验时非常有用。

与 TensorBoard 不同——与本次比较中的工具相似 Sacred 的优势在于它能够跟踪用任何 Python 库开发的任何模型训练。

**神圣——摘要:**

*   实验跟踪:任何模型训练
*   在团队中共享工作:不支持
*   与其他工具集成:不支持
*   SaaS/本地实例是否可用:否/是

*注意:神圣没有自带合适的用户界面，但有一些你可以连接到它的仪表板工具，如 Omniboard，Sacredboard，或 [Neptune via integration](https://web.archive.org/web/20221208211451/https://docs.neptune.ai/integrations-and-supported-tools/experiment-tracking/sacred) 。*

### TensorBoard vs 神圣

TensorBoard 和 Sacred 都是开源的，适合相当不高级的项目(就可伸缩性而言)。TensorBoard 附带了 UI，而您需要将神圣与仪表板工具配对，因此 TensorBoard 附带了更好的开箱即用可视化功能。

WandB 专注于深度学习。用户使用 Python 库跟踪应用程序的实验，并且作为一个团队，可以看到彼此的实验。

与 TensorBoard 不同，WandB 是一种托管服务，允许您在一个地方备份所有实验，并与团队合作完成一个项目——可以使用工作共享功能。

与 TensorBoard 类似，在 WandB 中，用户可以记录和分析多种数据类型。

**权重&偏差—汇总:**

*   实验跟踪:任何模型训练
*   团队共享工作:团队共享的多种特性。
*   与其他工具的集成:几个可用的开源集成
*   SaaS/本地实例可用:是/是
*   额外的好处:WandB 记录了模型图，因此您可以在以后检查它。

### 张量板与重量和偏差

第一个，TensorBoard 是一个本地运行的开源工具。WandB 提供的托管服务既可以在内部部署，也可以在云中运行。这里，Weight & Biases 提供了比 TensorBoard 更广泛的功能，包括实验跟踪、数据集版本化和模型管理。除此之外，WandB 还有很多支持团队协作的功能，这是 TensorBoard 所没有的。

Comet 是一个元机器学习平台，用于跟踪、比较、解释和优化实验和模型。

就像许多其他工具一样——例如 Neptune 或 WandB——Comet 提出了一个开源的 Python 库，允许数据科学家将他们的代码与 Comet 集成，并开始在应用程序中跟踪工作。

由于它提供云托管和自托管，用户可以有团队项目，并保存实验历史的备份。

Comet 正在通过预测性早期停止(免费版软件不提供)和神经结构搜索(未来)向更自动化的 ML 方法靠拢。

**彗星—摘要:**

*   实验跟踪:任何模型训练
*   团队共享工作:团队共享的多种特性。
*   与其他工具的集成:应该由用户手动开发
*   SaaS/本地实例可用:是/是
*   额外收获:显示平行图来检查参数和指标之间的关系模式。

### 张量板 vs 彗星

Comet 是一种托管服务，可以在内部提供，也可以作为托管应用程序提供。TensorBoard 是一个开源的可视化和跟踪工具，可以在本地使用。虽然 Comet 旨在使数据科学家能够在整个模型生命周期(从研究到生产)中建立更好的模型，但 TensorBoard 专注于实验阶段。

## 总结一下

第一次寻找实验追踪和可视化工具时，TensorBoard 往往似乎是一个不错的选择。它是开源的，提供了所有必要的特性。但是你越是使用它，你的需求就越是增长，你就会注意到一些缺失的部分。这就是为什么最好检查一下还有什么可用的工具，看看其他工具是否能在您的需求列表上勾选更多的框。

如果你是这种情况——你正在寻找一个更先进的工具，类似于使用 TensorBoard 后的下一步——我们建议检查 Neptune 或 Weights & Biases。这些都是出色的托管服务，具有大量功能和团队协作能力。如果你只是想切换到另一个开源解决方案，神圣的可能是正确的选择。

不管你的动机是什么，希望你在这里找到了一些值得检查的 TensorBoard 替代品，我们帮助你做出正确的选择！