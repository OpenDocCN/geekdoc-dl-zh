# ML 模型调试的深度指南和你需要知道的工具

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/ml-model-debugging-and-tools>

每个人都对机器学习感到兴奋，但只有少数人知道并理解阻止 ML 被广泛采用的局限性。ML 模型在特定的任务上很棒，但是它们也会犯很多错误。项目成功的关键是理解你的模型是如何失败的，并提前准备好适当的解决方案。

如果您不能在模型表现不佳或行为不当时排除故障，您的组织将无法长期适应和部署 ML——这就是为什么模型调试至关重要。

## 什么是模型调试？

[模型调试](https://web.archive.org/web/20221117203732/https://debug-ml-iclr2019.github.io/)研究 ML 响应函数和决策边界，以检测和纠正 ML 系统中的准确性、公平性、安全性和其他问题。

现在，如果机器学习是软件，难道不能使用与传统软件开发相同的调试工具吗？

调试 ML 模型比调试传统程序更难，原因有几个:

1.  **机器学习不止是代码**

与传统的代码库相比，ML 代码库有更多的活动部分。存在数据集、在训练期间微调的模型权重、在训练期间改变的优化及其梯度等等。你可以说代码在训练期间是“动态的”,之后它停止变化，而数据和模型在训练期间的语义偏差就像一个未被捕获的 bug 一样在代码中根深蒂固。

缺乏适当的工具来检查培训过程是我们许多人使用打印报表和记录器来分析的原因。

2.  **ML 训练期间的监控和干预很困难**

许多 ML 训练代码运行在集群或云中。当您在集群上运行分布式培训作业时，监控进度的主要方法是告诉您的代码生成日志，并将它们保存在一个中心位置以供分析。更好的替代方法是实时监控进度——这在 ML 调试器中是必不可少的。

有一些工具专门用于在培训期间监控你的 ML 模型的进度。以下是这些工具的列表:

*   [Neptune . ai](/web/20221117203732/https://neptune.ai/product)——除了广泛的跟踪和监控功能，它还自动跟踪代码的其他方面，并在表格中比较参数和指标。看看这个[的例子，运行监控实验](https://web.archive.org/web/20221117203732/https://app.neptune.ai/shared/step-by-step-monitoring-experiments-live/e/STEP-22/all?path=monitoring&attribute=cpu)，看看这看起来如何像现场直播。
*   [权重和偏差](https://web.archive.org/web/20221117203732/https://wandb.ai/site)–它有助于实时可视化实验，并且非常容易集成。它自动化超参数优化，并有助于探索可能的模型。
*   [Comet](https://web.archive.org/web/20221117203732/https://www.comet.ml/site/)–与之前的工具类似，Comet 旨在允许数据科学家跟踪和管理他们的实验。

3.  **调试 ML 代码可能需要大量的重写或框架切换**

ML 代码大多基于消除底层复杂性的高级框架，这使得模型难以调试。

## 如何调试 ML 模型，调试用的工具有哪些？

您的调试策略应该考虑三个重要阶段:

*   数据集准备
*   模型结构
*   输出测试

让我们看看这些策略，以及实现它们的工具。

### ML 模型的数据中心调试

确保数据集处于良好状态有三大步骤。

#### 1.验证数据质量

数据质量会显著影响模型性能。不平衡或不常见的特征会引入偏差。因为我们在给模型提供数据，所以当模型预测错误时，很容易挑出低质量的数据，而不是猜测它是否存在。

一旦我们了解了导致数据质量下降的问题，我们就可以立即解决它们。建议确保高质量的实验数据，因为如果没有高质量的数据，我们将无法建立一个理想的模型。以下是导致数据质量差的一些因素和解决方法:

\ u 003 cimg class = \ u 0022 lazy load block-blog-intext-CTA _ _ arrow-image \ u 0022 src = \ u 0022 https://Neptune . ai/WP-content/themes/Neptune/img/image-ratio-holder . SVG \ u 0022 alt = \ u 0022 \ u 0022 width = \ u 002212 \ u 0022 height = \ u 002222 \ u 022

![Imbalanced data - model debugging](img/3182f86d51463a2e0a34ef9426b22ed8.png)

*Fig: Example of imbalanced data*

*   **重采样数据集:**将副本添加到表示不足的类中，或者从表示过度的类中删除数据，可以对数据进行重采样，使其更加平衡。
*   **失衡标准化绩效指标:**使用失衡标准化的指标，如[科恩的 Kappa](https://web.archive.org/web/20221117203732/https://thedatascientist.com/performance-measures-cohens-kappa-statistic/#:~:text=Cohen's%20kappa%20statistic%20is%20a%20very%20good%20measure%20that%20can,class%20and%20imbalanced%20class%20problems.&text=Cohen's%20kappa%20is%20always%20less,way%20to%20interpret%20its%20values.) ，描绘出模型预测准确性的更清晰画面。也可以使用 ROC 曲线。
*   **生成合成数据:**有一些算法可以用来生成合成数据，而不是从像 [SMOTE](https://web.archive.org/web/20221117203732/https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5) (合成少数过采样技术)这样的次要类中复制数据。

\ u 003 cimg class = \ u 0022 lazy load block-blog-intext-CTA _ _ arrow-image \ u 0022 src = \ u 0022 https://Neptune . ai/WP-content/themes/Neptune/img/image-ratio-holder . SVG \ u 0022 alt = \ u 0022 \ u 0022 width = \ u 002212 \ u 0022 height = \ u 002222 \ u 022

数据集可能包含缺少值的行。以下是应对这些问题的一些方法:

*   **如果不影响数据集的大小，删除包含缺失值的行**。
*   **估算或**用平均值或中值替换缺失值。最频繁或任何常数值可用于分类特征，但这可能会在数据中引入偏差。

\ u 003 cimg class = \ u 0022 lazy load block-blog-intext-CTA _ _ arrow-image \ u 0022 src = \ u 0022 https://Neptune . ai/WP-content/themes/Neptune/img/image-ratio-holder . SVG \ u 0022 alt = \ u 0022 \ u 0022 width = \ u 002212 \ u 0022 height = \ u 002222 \ u 022

离群值可以定义为其潜在行为不同于其余数据的数据。由于我们的 ML 模型对训练期间输入的所有数据都很敏感，所以这些异常值，不管它们有多小，都会误导我们的训练过程，导致训练时间更长，模型精度更低。人们可以尝试这些方法来识别和去除异常值:

![Outliers - model debugging](img/7f7639e44cb910f9dec60fb4f636e0e6.png)

*Fig: Three obvious outliers in the dataset presented here*

*   **可视化:**可视化帮助我们了解数据之间的关联程度，并通过设置阈值来移除异常值。这是一个单变量方法，因为我们使用一个变量进行异常值分析。
*   **数学函数:**像 **Z 得分**或 **IQR 得分**这样的函数帮助我们建立数据之间的关系，并从统计上找出异常值。z 得分通过找出数据点与数据集的标准差和均值的关系来描述任何数据点。四分位间距分数是一种统计离差的度量，它具有一个阈值，可用于消除异常值。
*   **聚类算法:**聚类算法可用于根据数据点的密度对其进行聚类，低密度聚类可被标记为异常值，例如，带有噪声的应用的基于密度的空间聚类( [DBSCAN](https://web.archive.org/web/20221117203732/https://en.wikipedia.org/wiki/DBSCAN) )

#### 2.确保高质量的数据分割

我们必须确保数据在统计上被平均分割用于训练和测试。例如，假设我们正在制作一个猫狗分类器。我们有一个包含 10，000 张图片的数据集，其中每个类别有 5，000 张图片。我们以 8:2 的比例分割数据集，8000 张图像用于训练，2000 张图像用于测试:

*   训练集= 5000 张猫图像+ 3000 张狗图像
*   测试集= 2000 张狗图像

该数据集在统计上并不是平均分割的，所以它会使我们的猫模型输出产生偏差，因为训练集有更多的猫的例子。此外，在测试时，我们不会获得任何关于模型对猫图像的行为的信息。

另一件要记住的事情是选择正确的数据分割比例。一个 8:2 的训练:测试比例被称为[](https://web.archive.org/web/20221117203732/https://en.wikipedia.org/wiki/Pareto_principle)****的帕累托原理。**比率可能会根据数据变化。有关比率的快速总结，请查看这个[堆栈溢出线程](https://web.archive.org/web/20221117203732/https://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio)。**

**![Training splot - model debugging](img/1be5b12a4f63a3055037af9363c76632.png)

*Fig: Example way to split the dataset. [[Source](https://web.archive.org/web/20221117203732/https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets)]*

**K 折**是拆分数据的另一种方法。使用这种方法，数据如何分割并不重要，事实证明考虑这种方法是一个很好的优势。这里，原始数据被分成 K 个相同大小的折叠，使用随机抽样，没有重复。然后该模型被训练 K 次，每次使用 K-1 个折叠作为训练集，1 个折叠作为测试集。如果我们设置 K = n，那么就变成了**留一个**的情况！**层状 K 型褶皱**返回开始形成褶皱。折叠是通过保留每个类别的样本百分比来完成的。

#### 3.测试工程数据

特征是模型所基于的主要输入；它们来源于原始数据，这使得它们不同。手动调试时很难大量分析它们。克服这个问题的一个方法是用**单元测试来测试工程特性。单元测试可以被写成测试条件，比如:**

*   "所有的数字特征都是成比例的."
*   "独热编码向量只包含单个 1 和 N-1 个零."
*   "缺失的数据由缺省值(平均值)代替."
*   "异常值通过缩放或剪裁来处理."

**单元数据测试**让您更好地理解数据，并支持明智的决策。这也有助于可伸缩性，因为它确保数据保持准确和可靠。

*关于单元测试数据的基础知识，请阅读* [*单元测试数据:它是什么，你是怎么做的？*](https://web.archive.org/web/20221117203732/https://winderresearch.com/unit-testing-data-what-is-it-and-how-do-you-do-it/)

现在，让我们继续讨论数据验证和/或单元数据测试的工具。它们都是开源的，按字母顺序排列。

1.  [**地狱犬**](https://web.archive.org/web/20221117203732/https://github.com/pyeve/cerberus)

Cerberus 通过验证来守卫数据流的大门。它可以用于基本功能，如类型检查，但它也做自定义验证。它没有任何依赖性，所以它是轻量级的，易于实现。它被设计成非阻塞的，因此具有广泛的可扩展性。

2.  [**避去**](https://web.archive.org/web/20221117203732/https://github.com/awslabs/deequ)

Deequ 是一个建立在 Apache Spark 之上的数据质量断言工具。根据您对质量的定义，Deequ 测量批处理或流数据的相关 KPI，然后生成结果报告。它支持的功能有:对数据质量度量随时间变化的[数据剖析](https://web.archive.org/web/20221117203732/https://github.com/awslabs/deequ/blob/master/src/main/scala/com/amazon/deequ/examples/data_profiling_example.md)、[异常检测](https://web.archive.org/web/20221117203732/https://github.com/awslabs/deequ/blob/master/src/main/scala/com/amazon/deequ/examples/anomaly_detection_example.md)、对大型数据集的[自动约束建议](https://web.archive.org/web/20221117203732/https://github.com/awslabs/deequ/blob/master/src/main/scala/com/amazon/deequ/examples/constraint_suggestion_example.md)，以及使用[度量存储库](https://web.archive.org/web/20221117203732/https://github.com/awslabs/deequ/blob/master/src/main/scala/com/amazon/deequ/examples/algebraic_states_example.md)查询计算出的数据度量。

*在论文中了解更多 Deequ 的内部工作原理* [*自动化大规模数据质量验证*](https://web.archive.org/web/20221117203732/http://www.vldb.org/pvldb/vol11/p1781-schelter.pdf) *。*

![Deequ - model debugging tools](img/1e5424896e53c0d80f4a4be601c49a89.png)

*Fig: Overview of Deequ components. [[Source](https://web.archive.org/web/20221117203732/https://aws.amazon.com/blogs/big-data/test-data-quality-at-scale-with-deequ/)]*

3.  [**远大前程**](https://web.archive.org/web/20221117203732/https://greatexpectations.io/)

Great Expectations 是一个很棒的数据断言库，可以与任何管道工具一起使用。它有四个主要特点。

*   **数据测试**
*   **数据文档**–它提供了清晰易读的文档

![Great expectations - model debugging tools](img/a3532bdf38fdd33f19211820a046e865.png)

*Fig: Example of documentation done by Great Expectations. [[Source](https://web.archive.org/web/20221117203732/https://greatexpectations.io/)]*

*   **自动化数据分析**–它自动生成测试，利用领域专业知识帮助启动数据测试过程
*   **可插拔、可扩展、可伸缩**–每个组件都是可扩展的，可以针对生产就绪数据验证进行扩展。

您可以使用这个库来消除管道债务，并寄予厚望(双关语)。

![Great expectations 2 - model debugging tools](img/628871dbb0c474946643b7895c607131.png)

*Fig: What does Great Expectations do? [[Source](https://web.archive.org/web/20221117203732/https://docs.greatexpectations.io/en/latest/intro.html#what-is-great-expectations)]*

[*点击这里*](https://web.archive.org/web/20221117203732/https://docs.greatexpectations.io/en/latest/reference/supporting_resources.html) *找到远大前程的完整文档。你可以在* [*Slack*](https://web.archive.org/web/20221117203732/https://greatexpectations.io/slack) *上加入他们，问他们任何问题！*

**4。[格里芬](https://web.archive.org/web/20221117203732/https://griffin.apache.org/)**

Griffin 是一个数据质量断言工具，它有一个统一的过程来从不同的角度测量数据质量。它旨在支持大型数据集以及流式和批处理模式。它通过三个步骤处理数据质量问题:

1.  定义数据质量要求，如完整性、概要、准确性等。
2.  基于定义的需求测量数据质量。
3.  报告数据质量，因为指标被驱逐到预定义的目的地。

![Griffin - model debugging tools](img/229aab4549ccc3d1657708ec464d7886.png)

*Fig: Architecture of Griffin. [[Source](https://web.archive.org/web/20221117203732/https://griffin.apache.org/)]*

**5。 [JSON 模式](https://web.archive.org/web/20221117203732/https://json-schema.org/)**

使用模式进行单元测试听起来可能很奇怪，但是许多模式库允许您在规范中强制执行数据要求。

*查看谷歌对* [*数据模式*](https://web.archive.org/web/20221117203732/https://developers.google.com/machine-learning/testing-debugging/common/data-errors) *的简要全面解释。*

验证数据模式就像验证建筑物的蓝图一样。JSON Schema 是一个强大的工具，它还允许您在 API 层测试数据。

*阅读'* [*理解 JSON 模式*](https://web.archive.org/web/20221117203732/https://json-schema.org/understanding-json-schema/) *'。*

### ML 模型的以模型为中心的调试

#### **模型可解释性**

你的机器学习模型是一个黑盒，你想知道**为什么**你的模型做出某些决定？如果你发现它为什么以及如何做决定，你就可以了解问题、数据以及它可能失败的原因。可解释性解释了我们的 ML 模型黑盒的奥秘，并告诉我们**为什么**他们做出某些决定。

解释错误的预测为修复模型提供了方向。

我们需要可解释模型的两个主要原因是:

*   **调试错误**–调试的目的是找出错误的原因并解决问题。如果模型不可解释，错误就不能被调试，问题也不能从根本上解决。
*   **预测边缘案例**–我们已经知道它们是如何构建的。现在，当模型是透明的，我们知道它们是如何做决定的，我们可以根据数据预测所有的结果。这提醒我们注意边缘情况，帮助我们建立一个有效的 ML 模型。

![Model interpretability vs accuracy](img/b7ab89bd19d1e453d2789eda94545aed.png)

*Fig: Different model’s interpretability vs accuracy graph. [[Source](https://web.archive.org/web/20221117203732/https://towardsdatascience.com/guide-to-interpretable-machine-learning-d40e8a64b6cf)]*

机器学习中的可解释性是一个巨大的话题。我建议阅读 Christoph Molnar 的书' [*可解释的机器学习。制作黑盒模型的指南可讲解*](https://web.archive.org/web/20221117203732/https://christophm.github.io/interpretable-ml-book/) *”。*他很好地抓住了可解释 ML 的精髓，并给出了很好的例子。这是一本理解主题的好书。

以下是一些模型解释工具:

1.  [**不在场证明**](https://web.archive.org/web/20221117203732/https://www.seldon.io/tech/products/alibi/)

Alibi 是一个开源库，支持模型检查和解释。它可以让您仔细检查模型在概念漂移和算法偏差方面的性能。它与语言和工具包无关，因此易于集成和使用。

2.  [](https://web.archive.org/web/20221117203732/https://captum.ai/)

 **Captum 是一个基于 Pytorch 的模型可解释性工具。它支持视觉、文本等模型。它有[集成梯度](https://web.archive.org/web/20221117203732/https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)，所以它试图根据对模型输出有贡献的特征来解释模型。它可以通过与 PyTorch 模型交互的算法轻松实现。它可以用在生产中经过训练的模型上。

[![Captum - model debugging tools](img/588d18584218f04abfb0ae81b6afad07.png)](https://web.archive.org/web/20221117203732/https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Captum-model-debugging-tools.png?ssl=1)

*Fig: Sample screenshots of Captum Insights. [[Source](https://web.archive.org/web/20221117203732/https://captum.ai/docs/captum_insights)]*

3.  [**interpret ml**](https://web.archive.org/web/20221117203732/https://interpret.ml/)

一个开源工具包，用最先进的技术来解释模型行为。它支持玻璃盒模型——易于解释的模型，如线性模型、决策树和现有系统的黑盒解释器。它实现了[可解释的助推机器](https://web.archive.org/web/20221117203732/https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf)，一个强大的、可解释的玻璃盒子模型，可以像许多黑盒模型一样精确。InterpretML 支持以下技术:

*查看他们的论文，* [*InterpretML:机器学习的统一框架可解释性*](https://web.archive.org/web/20221117203732/https://arxiv.org/pdf/1909.09223.pdf) *深入了解它是如何工作的。*

4.  [**石灰**](https://web.archive.org/web/20221117203732/https://github.com/marcotcr/lime)

Lime 是**L**ocal**I**interpretable**M**模型不可知论者 **E** 解释的缩写，解释作用于表格或图像的分类器的个体预测。它基于本文[中介绍的工作](https://web.archive.org/web/20221117203732/https://arxiv.org/abs/1602.04938)。它可以解释任何有两个或更多类的黑盒分类器。它采用一个 NumPy 数组或原始测试，并给出每个类的概率。

![Lime - model debugging tools](img/bec77aa0b59f2b624554c5bf15d56451.png)

*Fig: Explaining a prediction with LIME. [[Source](https://web.archive.org/web/20221117203732/https://www.oreilly.com/content/introduction-to-local-interpretable-model-agnostic-explanations-lime/)]*

*查看* [*教程*](https://web.archive.org/web/20221117203732/https://marcotcr.github.io/lime/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.html) *用随机森林分类器讲解石灰的基本用法。*

5.  [**Shap**](https://web.archive.org/web/20221117203732/https://github.com/slundberg/shap)

Shap 做了 Shapley 的基于加法的解释。它基于博弈论的方法来解释任何 ML 模型的输出。SHAP 试图通过计算每个特征对预测的贡献来解释一个实例的预测。SHAP 解释法从联盟博弈论中计算出[沙普利值](https://web.archive.org/web/20221117203732/https://youtu.be/qcLZMYPdpH4)。

6.  [**DALEX**](https://web.archive.org/web/20221117203732/https://github.com/ModelOriented/DALEX)

DALEX 是描述性机器学习解释工具的缩写，它有助于在更深层次上分析和理解模型是如何工作的。

*   DALEX 的关键目标是一个解释者。它在一个预测模型周围创建了一个包装器，然后可以探索这些包装的模型，并与其他局部或全局解释器进行比较。
*   DALEX 使用各种技术，如 [SHAP](https://web.archive.org/web/20221117203732/https://github.com/slundberg/shap) 、[分解](https://web.archive.org/web/20221117203732/https://pbiecek.github.io/breakDown/)、[交互分解](https://web.archive.org/web/20221117203732/https://pbiecek.github.io/breakDown/reference/break_down.html)来分析和理解模型预测。
*   它使用[在其他条件不变的情况下](https://web.archive.org/web/20221117203732/https://github.com/pbiecek/ceterisParibus)来绘制模型预测的变化，以帮助您了解模型对特征值变化的敏感度。
*   它可以被创建并与 sklearn、keras、lightgbm 等流行的框架集成。
*   它可以与 [Neptune](/web/20221117203732/https://neptune.ai/blog/explainable-and-reproducible-machine-learning-with-dalex-and-neptune) 配合使用，为每次训练自动保存和版本化这些讲解器和互动图表。

![DALEX - model debugging tools](img/a37546d4ab16cf5ba42b87b3618323be.png)

*Fig: How DALEX can be used. [[Source](https://web.archive.org/web/20221117203732/https://github.com/ModelOriented/DrWhy/blob/master/README.md)]*

### ML 模型的以预测为中心的调试

最大似然模型预测通常使用诸如均方误差(MSE)、曲线下面积(AUC)等指标来评估。这些度量告诉我们模型的表现如何，但是没有告诉我们为什么模型表现不好。我们知道自己离目标或期望的产出有多近，但不会找出模型失败的地方。因此，可视化调试工具让我们看到了这些指标之外的特性，比如性能比较、数据集上的特性分布等等。

让我们看看一些可视化调试工具:

1.  [**歧**](https://web.archive.org/web/20221117203732/https://eng.uber.com/manifold/)

Manifold 是优步的一个模型无关的可视化调试工具。它通过解释导致模型性能不佳的潜在问题，将模型性能不佳可视化。它揭示了性能较好和较差的数据子集之间的特征分布差异。流形依靠三个关键特征来解释模型输出:

*   **性能比较视图**

跨不同数据子集可视化模型的总体性能。Manifold 使用聚类算法根据性能相似性将预测数据分成 N 个部分。这有助于您挑选出表现不佳的子集进行检查。

![Manifold - model debugging tools](img/a6ec6b49e700b70e41b8e41c834db93a.png)

*Fig: Example of Performance comparison view in Manifold. [[Source](https://web.archive.org/web/20221117203732/https://github.com/uber/manifold)]*

这显示了由用户定义的段组合在一起的数据的特征值。识别任何可能与不准确的预测输出相关的输入要素分布。您可以区分每个特征中的两个分布。这里的分布表示在性能比较视图中选择的两段式组的数据差异。

![Manifold 2 - model debugging tools](img/de9807dec3396bbceb185a6dec01e57f.png)

*Fig: Example of Feature Attribute View in Manifold. [[Source](https://web.archive.org/web/20221117203732/https://github.com/uber/manifold)]*

如果数据集包含地理空间要素，它将与该要素一起显示在地图上。经纬度坐标和 h3 六边形 id 是流形目前支持的特征类型。它通过表示先前选择的两个子集之间的空间分布差异来显示两个段组之间的地理位置差异。

![Manifold 3 - model debugging tools](img/d34f81e51b0e09541f06b5283e225273.png)

*Fig: Manifold architecture. [[Source](https://web.archive.org/web/20221117203732/https://eng.uber.com/manifold/)]*

2.  [](https://web.archive.org/web/20221117203732/https://github.com/microsoft/tensorwatch)

 **这个用于调试工具的高级平台非常易于使用，并且非常具有可扩展性:

*   它在 Jupyter Notebook 中运行，并显示机器学习训练的实时可视化，让您也可以执行关键分析。
*   它也非常灵活；它让您自定义可视化。
*   它有一个独特的能力叫做[懒惰日志模式](https://web.archive.org/web/20221117203732/https://github.com/microsoft/tensorwatch/blob/master/docs/lazy_logging.md)。这将针对实时 ML 训练过程执行任意查询，并返回该查询的结果，在可视化工具的选项中显示它们。

*参见* [*教程*](https://web.archive.org/web/20221117203732/https://github.com/microsoft/tensorwatch#tutorials) *和* [*笔记本*](https://web.archive.org/web/20221117203732/https://github.com/microsoft/tensorwatch/tree/master/notebooks) *深入了解 TensorWatch 的特性。*

3.  [**埃夫马莱**](https://web.archive.org/web/20221117203732/https://efemarai.com/)

Efemarai 是一个独特的 Python 工具，用于可视化、检查和调试。除了基本的用法之外，它还可以用来研究可解释性、健壮性等等。让我们来看看它的一些特性:

*   它可以构建大型多维张量的直观 3D 可视化以及自动生成的计算图形。通过让您直观地检查数据，这有助于您密切关注数据。

![Efemarai - model debugging tools](img/c646c0148c52a501d5e171927215b687.png)

*Fig: Example of multidimensional tensor inspection. [[Source](https://web.archive.org/web/20221117203732/https://efemarai.com/)]*

*   它有一些功能，使你能够毫不费力地检查任何张量或张量元素，没有明确处理(如梯度张量)在几下点击。这样就可以访问代码使用或生成的所有值。
*   所有的断言(由您定义)都是自动监控的，因此您可以快速检测出违反您对数据或代码行为的假设的情况。
*   当张量梯度可用时，它跟踪张量梯度，简化优化过程。

4.  [**亚马逊 SageMaker 调试器**](https://web.archive.org/web/20221117203732/https://aws.amazon.com/sagemaker/debugger/)

SageMaker 是可伸缩和可扩展的。它的调试效果如何？

*   它实时捕捉训练指标，并帮助您分析和优化您的 ML 模型。
*   它不断跟踪训练过程，因此它可以自动终止训练过程。这减少了训练 ML 模型的时间和成本，提高了效率。
*   它可以实时自动分析和监控系统资源利用率，并在发现预定义的瓶颈时发送警报。
*   它支持广泛的 ML 算法和 DL 框架。
*   它不是开源的，所以如果你正在寻找一个工具来进行实验或开始新的尝试，这并不适合你。如果您想在训练时监控您的模型并执行实时分析，那么这个工具可以很好地完成这项工作。

## 结论

机器学习系统比传统软件更难测试，因为我们没有明确地编写。我已经向您展示了一些 ML 模型的调试策略，以及实现它们的工具，然后我们开始检查我们的模型并讨论模型的可解释性。我们还研究了模型调试工具，这些工具跟踪从输入到输出的错误路径。

我希望所有这些能帮助您更好地理解 ML 调试。感谢您的阅读！下面是您可以阅读的内容，以了解更多信息:

### 进一步阅读

**论文**

**博文**

**会议**

**视频**

**书籍********