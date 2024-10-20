# 用于模型训练的十大最佳机器学习工具

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/top-10-best-machine-learning-tools-for-model-training>

与流行的观念相反，[机器学习中的模型训练](https://web.archive.org/web/20221206031300/https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss)不仅仅是一个黑箱活动。为了让机器学习(ML)解决方案始终表现良好，开发人员必须深入研究每个模型，以找到与数据和业务用例的正确匹配。

简单来说，机器学习模型是一个简单的统计方程，它是根据手头的数据随着时间的推移而发展起来的。这个学习过程，也称为训练，从简单到复杂的过程。模型训练工具是一个接口，它使得开发者和机器学习模型的复杂性之间的交互变得容易。

在机器学习中，没有“万金油”——没有一种工具可以解决所有问题，因为现实世界的问题和数据存在巨大差异。但是有一些模型训练工具可以像手套一样适合你——特别是你和你的要求。

为了能够为您的解决方案选择主要的模型培训工具，您需要评估您现有的开发流程、生产基础设施、团队的技能水平、合规性限制以及类似的重要细节，以便能够确定正确的工具。

然而，从长远来看，一个经常被忽视的、导致薄弱的基础和不稳定的解决方案系列的关键特征是模型训练工具跟踪元数据的能力或者与[元数据存储](/web/20221206031300/https://neptune.ai/blog/ml-metadata-store)和监控工具无缝集成的能力。

模型元数据涉及训练参数、实验指标、数据版本、管道配置、重量参考文件等资产。这些数据非常有用，可以减少生产和模型恢复时间。为了选择正确的元数据存储，您的团队可以在构建新解决方案和购买现有解决方案之间进行成本效益分析。

以下是 ML 市场中十大模型培训工具的列表，您可以使用这些工具来评估您的需求是否与该工具提供的功能相匹配。

### 1\. TensorFlow

我记得在实习时遇到过 TensorFlow，在几乎没有探索 scikit-learn 之后，我明显感到害怕。回过头来看，这似乎迫在眉睫，因为 TensorFlow 是一个低级库，需要与模型代码密切合作。开发者可以通过 TensorFlow 实现完全控制，从零开始训练模型。

然而，TensorFlow 也提供了一些预先构建的模型，可用于更简单的解决方案。TensorFlow 最令人满意的特性之一是数据流图，尤其是在开发复杂模型时，它会派上用场。

TensorFlow 支持广泛的解决方案，包括 NLP、计算机视觉、预测 ML 解决方案和强化学习。作为谷歌的一款开源工具，TensorFlow 在全球拥有超过 380，000 名贡献者的社区，因此它在不断发展。

查看如何[跟踪 TensorFlow/Keras 模型培训](https://web.archive.org/web/20221206031300/https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras)。

### 2\. PyTorch

PyTorch 是另一个流行的开源工具，给 TensorFlow 带来了激烈的竞争。PyTorch 有两个重要的特性——在 GPU 上加速处理的张量计算和建立在基于磁带的自动差分系统上的神经网络。

此外，PyTorch 支持许多 ML 库和工具，这些库和工具可以支持各种解决方案。一些例子包括 AllenNLP 和 ELF，这是一个游戏研究平台。PyTorch 除了 Python 还支持 C++和 Java。

PyTorch 和 TensorFlow 之间的一个主要区别是 PyTorch 支持动态数据流图，而 TensorFlow 仅限于静态图。与 TensorFlow 相比，PyTorch 更容易学习和实现，因为 TensorFlow 需要大量的代码工作。

查看如何[跟踪 PyTorch 模型培训](https://web.archive.org/web/20221206031300/https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch)。

### 3\. PyTorch Lightning

PyTorch Lightning 是 PyTorch 之上的一个包装器，主要用于将注意力转移到研究上，而不是工程或多余的任务上。它抽象了模型和公共代码结构的底层复杂性，因此开发人员可以在短时间内关注多个模型。

顾名思义，PyTorch Lightning 的两大优势是速度和规模。它支持 TPU 集成，消除了使用多个 GPU 的障碍。为了规模，PyTorch Lightning 允许实验通过 [grid.ai](https://web.archive.org/web/20221206031300/http://grid.ai/) 在多个虚拟机上并行运行。

PyTorch Lightning 对代码的需求明显减少，因为它有高级包装器。然而，这并不限制灵活性，因为 PyTorch 的主要目标是减少对冗余样板代码的需求。开发人员仍然可以修改并深入到需要定制的领域。

查看如何[跟踪 PyTorch Lightning 模型培训](https://web.archive.org/web/20221206031300/https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning)。

### 4\. Scikit-learn

[Scikit-learn](https://web.archive.org/web/20221206031300/https://github.com/scikit-learn/scikit-learn) 是顶级开源框架之一，非常适合开始机器学习。它有高级包装器，使用户能够使用多种算法，探索广泛的分类、聚类和回归模型。

对于好奇的人来说，scikit-learn 也是一种很好的方式，只需解开代码并遵循依赖关系，就可以更深入地了解模型。Scikit-learn 的文档非常详细，初学者和专家都很容易阅读。

Scikit-learn 非常适合时间和资源有限的 ML 解决方案。它严格以机器学习为中心，并且在过去几年中一直是流行品牌预测解决方案的重要组成部分。

查看如何[跟踪 Scikit-learn 模型培训](https://web.archive.org/web/20221206031300/https://docs.neptune.ai/integrations-and-supported-tools/model-training/sklearn)。

### 5.催化剂

[Catalyst](https://web.archive.org/web/20221206031300/https://catalyst-team.github.io/catalyst/) 是另一个专门为深度学习解决方案打造的 PyTorch 框架。Catalyst 是研究友好型的，负责工程任务，如代码的可重用性和可再现性，促进快速实验。

深度学习一直被认为是复杂的，Catalyst 使开发人员能够用几行代码执行深度学习模型。它支持一些顶级的深度学习模型，如 ranger optimizer、随机加权平均和单周期训练。

Catalyst [保存源代码和环境变量](https://web.archive.org/web/20221206031300/https://analyticsindiamag.com/guide-to-catalyst-a-pytorch-framework-for-accelerated-deep-learning/#:~:text=All%20the%20source%20code%2C%20as%20well%20as%20environment%20variables%2C%20remain%20saved%20thereby%20enabling%20code%20reproducibility)以实现可重复的实验。其他一些值得注意的特性包括模型检查点、回调和提前停止。

检查如何[跟踪 Catalyst 模型培训](https://web.archive.org/web/20221206031300/https://docs.neptune.ai/integrations-and-supported-tools/model-training/catalyst)。

### 6.XGBoost

[XGBoost](https://web.archive.org/web/20221206031300/https://github.com/dmlc/xgboost) 是一种基于树的模型训练算法，使用梯度提升来优化性能。这是一种集成学习技术，即使用几种基于树的算法来实现最佳模型序列。

使用梯度增强，XGBoost 一个接一个地生长树，以便后面的树可以从前面的树的弱点中学习。它通过从前面的树模型中借用信息来逐渐调节弱学习者和强学习者的权重。

为了提高速度，XGBoost 支持跨分布式环境(如 Hadoop 或 MPI)的并行模型加速。XGBoost 非常适合大型训练数据集以及数值和分类特征的组合。

检查如何[跟踪 XGBoost 模型培训](https://web.archive.org/web/20221206031300/https://docs.neptune.ai/integrations-and-supported-tools/model-training/xgboost)。

### 7.LightGBM

[LightGBM](https://web.archive.org/web/20221206031300/https://github.com/microsoft/LightGBM) 和 XGBoost 一样，也是使用基于树的模型的梯度提升算法。但是说到速度，LightGBM 比 XGBoost 占了上风。LightGBM 最适合大型数据集，否则使用其他模型会耗费大量训练时间。

虽然大多数基于树的算法在树的级别或深度上进行分割，但 LightGBM 采用了独特的叶子或宽度分割技术，这已被证明可以提高性能。尽管这可能会使模型过拟合，但是开发人员可以通过调整 max_depth 参数来避免这种情况。

尽管 LightGBM 处理大量数据集，但它需要的内存空间很少，因为它用离散的条块代替了连续的值。它还支持并行学习，这也是一个重要的时间节省。

检查如何[跟踪 LightGBM 模型培训](https://web.archive.org/web/20221206031300/https://docs.neptune.ai/integrations-and-supported-tools/model-training/lightgbm)。

### 8.CatBoost

[CatBoost](https://web.archive.org/web/20221206031300/https://github.com/catboost/catboost) 是一种梯度提升算法，与大多数机器学习模型相比，它通过最少的训练提供了同类最佳的结果。它是一个开源工具，并且因为其易用性而广受欢迎。

CatBoost 减少了预处理工作，因为它可以直接和优化地处理分类数据。它通过生成数字编码和在后台试验各种组合来做到这一点。

尽管 CatBoost 提供了使用一系列多个超参数进行广泛调整的范围，但它不需要太多调整，并且可以在不过度拟合训练数据的情况下产生结果。它非常适合低容量和高容量数据。

### 9.Fast.ai

Fast.ai 的朗朗上口的口号说明了一切——“让神经网络再次变得不酷”。Fast.ai 旨在使深度学习可以跨多种语言、操作系统和小数据集进行。它是基于迁移学习是深度学习的一个关键优势，可以减少大量冗余工程工作的想法而开发的。

它为深度学习模型提供了一个易于使用的高级界面，还允许用户下载一组预先训练好的模型。Fast.ai 有多个包装器，隐藏了底层模型架构的复杂性。这使得开发人员可以专注于数据智能和流程突破。

Fast.ai 也非常受欢迎，因为它分享了他们的免费在线课程“[程序员实用深度学习](https://web.archive.org/web/20221206031300/https://course.fast.ai/)”，该课程不要求任何先决条件，但深入研究了深度学习概念，并说明了如何通过 fast.ai 使其变得简单。

查看如何[跟踪 fast.ai 模型训练](https://web.archive.org/web/20221206031300/https://docs.neptune.ai/integrations-and-supported-tools/model-training/fastai)。

### 10.皮托奇点火

PyTorch Ignite 是一个构建在 PyTorch 之上的包装器，与 PyTorch Lightning 非常相似。两者都提供了模型复杂性的抽象和易于使用的界面，以扩展研究能力和减少冗余代码。

就架构而言，两者之间存在微妙的差异。PyTorch Lightning 有一个标准的可复制界面，Ignite 没有任何标准版本。

虽然它不能支持高度先进的功能，但 Ignite 可以与集成生态系统很好地合作，以支持机器学习解决方案，而 Lightning 支持最先进的解决方案、高级功能和分布式培训。

查看如何[跟踪 PyTorch Ignite 模型培训](https://web.archive.org/web/20221206031300/https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-ignite)。

有几个其他的选项可能不像上面的选择那样受欢迎，但是对于特定的模型训练需求来说是很棒的。

例如:

*   如果有限的 GPU 资源的高速是你的优先事项， [Theano](https://web.archive.org/web/20221206031300/https://github.com/Theano/Theano) 领先。
*   因为。NET 和 C#的能力， [Accord](https://web.archive.org/web/20221206031300/http://accord-framework.net/) 将是理想的。它还有许多音频和图像处理库。
*   ML.NET 是另一个工具。NET 开发人员。
*   NLP 专用和计算机视觉解决方案的其他选项分别包括 [Gensim](https://web.archive.org/web/20221206031300/https://pypi.org/project/gensim/) 和 [Caffe](https://web.archive.org/web/20221206031300/https://caffe.berkeleyvision.org/) 。

总之，在为您的特定解决方案选择合适的解决方案之前，最好先进行彻底的市场调查。它可能不是最流行或最知名的工具，但它绝对是适合你的。

## 最后一个音符

如前所述，没有一种工具可以解决所有商业案例或机器学习问题。即使没有一个工具看起来完全适合你，但是它们的组合可能是理想的选择，因为它们中的大多数都是相互兼容的。

诀窍是首先列出该领域中的一些最佳工具，我们已经为您完成了，然后探索入围的工具，以逐步达到正确的匹配。这里分享的工具很容易安装，并且在它们各自的站点上有大量的文档，便于快速启动！

来源: