# Arize AI 和 Neptune AI 合作伙伴关系:对 ML 模型的持续监控和持续改进

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/arize-partnership>

将最好的机器学习模型交付给生产应该像培训、测试和部署一样简单，对吗？不完全是！**从研究到生产，模型远非完美，而*一旦投入生产，保持*模型的性能更具挑战性。**一旦脱离离线研究环境，模型消耗的数据可能会在分布或完整性方面发生变化，无论是由于在线生产环境中的意外动态还是数据管道中的上游变化。

如果没有正确的工具来检测问题并诊断问题出现的位置和原因，ML 团队就很难知道一旦部署到生产中，何时调整和重新培训他们的模型。作为任何模型性能管理工作流程的一部分，必须建立一个连续的反馈循环。实验、验证、监控和主动改进模型的工具帮助 ML 从业者在生产中一致地部署——并维护——更高质量的模型。

> 这就是为什么 Arize AI 和 Neptune AI 很高兴地宣布建立合作伙伴关系，将 Arize 的 ML 可观测性平台与 Neptune 的 MLOps 元数据存储连接起来。

通过这种合作关系，ML 团队可以更有效地监控其生产模型的执行情况，深入研究有问题的特征和预测群组，并最终做出更明智的再培训决策。

## 阿里斯-海王星公司

世界上有很多 ML 基础设施公司，所以让我们概括一下 Neptune 和 Arize 的专长:

*   Neptune 记录、存储、显示和比较您的模型构建元数据。它有助于实验跟踪、数据和模型版本化，以及模型注册，这样你就能真正知道你正在发布什么样的模型。

*   **Arize 帮助您可视化生产模型性能，并了解漂移和数据质量问题。**

![Arize-drift-monitor](img/80a731e9f542690595a1da165f82fd8b.png)

*Example dashboard in Arize*

ML 生命周期应该被认为是一个跨越数据准备、模型建立和生产阶段的迭代流程。虽然 Arize 平台侧重于生产模型的可观察性，但它监控生产、培训和验证数据集，以构建部署前后可能出现的问题的整体视图。这允许您将任何数据集(包括生产中的前期)设置为比较模型性能的基线或基准。

那么，这有什么关系呢？有了 Neptune 这样的中央 ML 元数据存储，跟踪每个模型版本和模型历史的血统就容易多了。当涉及到试验和优化生产模型时，这对于确保 ML 从业者能够:

1.  为团队遵循的模型版本化和打包建立一个协议
2.  提供通过 CLI 和 SDK 以您在生产中使用的语言查询/访问模型版本的能力
3.  提供一个地方来记录硬件、数据/概念漂移、来自 CI/CD 管道的示例预测、再培训工作和生产模型
4.  为主题专家、制作团队或自动检查的批准建立协议。

**TL；当结合使用 Arize 和 Neptune 时，DR–ML 从业者可以微调模型性能，并在更细粒度的级别上积极改进模型。**

## 从 Arize 和 Neptune 开始

现在，让我们把这些放在一起！

如果你还没有的话，注册一个免费的帐号给[海王星](/web/20221206051205/https://neptune.ai/register)和[阿里斯](https://web.archive.org/web/20221206051205/https://arize.com/request-a-demo/)。

你一做，就查看我们为你准备的[教程](https://web.archive.org/web/20221206051205/https://docs.neptune.ai/integrations-and-supported-tools/model-monitoring/arize)！

你会在那里发现:

*   设置 Neptune 客户端和 Arize 客户端
*   海王星测井培训回电
*   记录培训和验证记录以存档
*   使用 Neptune 存储和版本化模型权重
*   用 Arize 实现生产中的日志记录和版本控制模型

## 结论

有了 Arize 和 Neptune，您将能够高效地完成您最好的机器学习工作:识别和训练您的最佳模型，预发布验证您的模型，并通过简单的集成在模型构建、实验和监控之间创建反馈回路。

### 关于海王星

Neptune 是 MLOps 的一个元数据存储库，为进行大量实验的研究和生产团队而构建。它为您提供了一个中心位置来记录、存储、显示、组织、比较和查询机器学习生命周期中生成的所有元数据。

成千上万的 ML 工程师和研究人员使用 Neptune 进行[实验跟踪](/web/20221206051205/https://neptune.ai/product/experiment-tracking)和[模型注册](/web/20221206051205/https://neptune.ai/product/model-registry)，无论是作为个人还是在大型组织的团队内部。

有了 Neptune，您可以用一个真实的来源取代文件夹结构、电子表格和命名约定，在这个来源中，您所有的模型构建元数据都是有组织的，易于查找、共享和查询。如果你想更多地了解 Neptune 及其功能，请观看[产品之旅](/web/20221206051205/https://neptune.ai/demo)或探索我们的[文档](https://web.archive.org/web/20221206051205/https://docs.neptune.ai/)。

### 关于艾瑞泽

Arize AI 是一个[机器学习可观察性](https://web.archive.org/web/20221206051205/https://arize.com/model-monitoring)平台，帮助 ML 从业者轻松成功地将模型从研究转向生产。Arize 的自动化[模型监控](https://web.archive.org/web/20221206051205/https://arize.com/ml-monitoring/)和分析平台允许 ML 团队在问题出现时快速检测问题，解决问题发生的原因，并提高整体模型性能。通过将离线训练和验证数据集连接到中央推理存储中的在线生产数据，ML 团队可以简化[模型验证](https://web.archive.org/web/20221206051205/https://arize.com/ml-model-failure-modes/)、[漂移检测](https://web.archive.org/web/20221206051205/https://arize.com/take-my-drift-away/)、[数据质量检查](https://web.archive.org/web/20221206051205/https://arize.com/data-quality-monitoring/)和[模型性能管理](https://web.archive.org/web/20221206051205/https://arize.com/monitor-your-model-in-production/)。

Arize AI 在部署的 AI 上充当护栏，为历史上的黑箱系统提供透明度和自省，以确保更有效和更负责任的 AI。要了解更多关于 Arize 或机器学习的可观察性和监控，请访问我们的[博客](https://web.archive.org/web/20221206051205/https://arize.com/blog/)和[资源中心](https://web.archive.org/web/20221206051205/https://arize.com/resource-hub/)！