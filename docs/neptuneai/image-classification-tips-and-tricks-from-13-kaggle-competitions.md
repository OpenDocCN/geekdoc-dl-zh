# 图像分类:来自 13 个 Kaggle 竞赛的提示和技巧(+大量参考文献)

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/image-classification-tips-and-tricks-from-13-kaggle-competitions>

任何领域的成功都可以提炼为一套小规则和基本原则，当它们结合在一起时会产生巨大的效果。

机器学习和图像分类没有什么不同，工程师可以通过参加像 Kaggle 这样的比赛来展示最佳实践。

在本文中，我将为您提供大量的学习资源，重点关注来自 13 个 Kaggle 竞赛的最佳 Kaggle 内核，其中最突出的竞赛是:

我们将经历调整深度学习解决方案的三个主要领域:

…在此过程中会有许多示例项目(和参考资料)供您查阅。

## 数据

### 图像预处理+ EDA

每个机器学习/深度学习解决方案都是从原始数据开始的。数据处理流程中有两个基本步骤。

第一步是 **[探索性数据分析](https://web.archive.org/web/20221118213143/https://www.kaggle.com/allunia/protein-atlas-exploration-and-baseline)** (EDA)。它帮助我们分析整个数据集并总结其主要特征，如类别分布、大小分布等。可视化方法通常用于显示这种分析的结果。

第二步是 [**图像预处理**](https://web.archive.org/web/20221118213143/https://towardsdatascience.com/image-pre-processing-c1aec0be3edf) ，目的是获取原始图像，通过抑制不必要的失真、调整大小和/或增强重要特征来改善图像数据(也称为图像特征)，使数据更适合模型并提高性能。

您可以深入这些 Kaggle 笔记本，查看一些关于**图像预处理**和 **EDA** 技术的示例:

数据扩充

### **数据扩充**可以通过从现有的训练样本中生成更多的训练数据来扩充我们的数据集。新的样本是通过大量的随机转换生成的，这些随机转换不仅产生看起来可信的图像，还反映了现实生活中的场景——稍后将详细介绍。

这种技术被广泛使用，不仅仅是在数据样本太少而无法训练模型的情况下。在这种情况下，模型开始记忆训练集，但它无法进行归纳(在从未见过的数据上表现很差)。

通常，当一个模型在训练数据上表现很好，但在验证数据上表现很差时，我们把这种情况称为 [***【过拟合】***](https://web.archive.org/web/20221118213143/https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/) 。为了解决这个问题，我们通常试图获得新数据，如果新数据不可用，数据增强就可以解决这个问题。

**注意**:一般的经验法则是总是使用数据扩充技术，因为它有助于将我们的模型暴露给更多的变化并更好地概括。即使我们有一个大数据集，尽管这是以缓慢的训练速度为代价的，因为增强是在运行中完成的(这意味着在训练期间)。

此外，对于每个任务或数据集，我们必须使用反映可能的现实生活场景的增强技术(即，如果我们有一个猫/狗检测器，我们可以使用水平翻转、裁剪、亮度和对比度，因为这些增强技术匹配照片拍摄方式的差异)。

以下是一些 Kaggle 竞赛笔记本，供您在实践中检验流行的数据增强技术:

模型

## 在这里，我们使用一个非常简单的架构创建了一个基本模型，没有任何正则化或丢弃层，看看我们是否可以打破 50%准确性的基线分数。虽然我们不能总是到达那里，但是如果我们在尝试了多种合理的架构之后不能超过基线，也许输入数据并不包含我们的模型做出预测所需的信息。

用杰瑞米·霍华德睿智的话来说:

“你应该能够在 15 分钟内使用 50%或更少的数据集快速测试你是否进入了一个有前途的方向，如果没有，你必须重新思考一切。”

开发一个足够大的模型，使其适应过度([示例项目](https://web.archive.org/web/20221118213143/https://www.kaggle.com/allunia/protein-atlas-exploration-and-baseline#Building-a-baseline-model-)

### 一旦我们的基线模型有足够的能力击败基线得分，我们可以增加基线模型的能力，直到它超过数据集，然后我们移动到应用正则化。我们可以通过以下方式增加模块容量:

添加更多层

*   使用更好的架构
*   更好的培训程序
*   体系结构

### 根据文献，下面的架构改进提高了模型容量，但几乎没有改变计算复杂性。如果你想深入研究相关的例子，它们仍然很有趣:

大多数情况下，模型容量和准确性是正相关的——随着容量的增加，准确性也会增加，反之亦然。

培训程序

### 以下是一些训练程序，您可以使用它们来调整您的模型，并通过示例项目来了解它们是如何工作的:

超参数调谐

### 与参数不同，[超参数](https://web.archive.org/web/20221118213143/https://www.oreilly.com/library/view/evaluating-machine-learning/9781492048756/ch04.html)由您在配置模型时指定(即学习率、时期数、隐藏单元数、批量等)。

除了手动尝试不同的模型配置，您还可以通过使用[超参数调整](/web/20221118213143/https://neptune.ai/blog/hyperparameter-tuning-in-python-a-complete-guide-2020)库，如[**Scikit learn Grid Search**](https://web.archive.org/web/20221118213143/http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)、 **[Keras Tuner](https://web.archive.org/web/20221118213143/https://keras-team.github.io/keras-tuner/) 、**和其他库来自动执行此过程，这些库将尝试您指定范围内的所有超参数组合，并返回最佳性能模型。

需要优化的超参数越多，这个过程就越慢，所以最好选择模型超参数的最小子集进行优化。

并非所有的模型超参数都同样重要。一些超参数对机器学习算法的行为有巨大的影响，进而影响其性能。您应该仔细挑选对模型性能影响最大的那些，并对它们进行调优以获得最佳性能。

正规化

### 这种方法通过惩罚 [***记忆/过拟合***](https://web.archive.org/web/20221118213143/https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/) 和 [***欠拟合***](https://web.archive.org/web/20221118213143/https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/) 来迫使模型学习数据的有意义的和可概括的表示，使得模型在处理它从未见过的数据时更加健壮。

解决上述问题的一个简单方法是获得更多的训练数据，因为基于更多数据训练的模型自然会更好地概括。

这里有一些你可以尝试缓解 [***过拟合***](https://web.archive.org/web/20221118213143/https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/)[***欠拟合***](https://web.archive.org/web/20221118213143/https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/) 的技巧，并附有示例项目链接供你挖掘:

损失函数

## [**损失函数**](https://web.archive.org/web/20221118213143/https://medium.com/@prince.canuma/how-to-develop-your-ai-intuition-ii-9dedb4a41c1c) 也被称为成本函数或目标函数，用于找出模型输出与目标输出之间的差异，并帮助模型最小化它们之间的距离。

以下是一些最受欢迎的损失函数，通过项目示例，您可以找到提高模型容量的技巧:

评估+误差分析

### 在这里，我们进行烧蚀研究，并分析我们的实验结果。我们确定模型的弱点和优点，并确定未来需要改进的地方。在这个阶段，您可以使用下面的技术，看看它们是如何在链接的示例中实现的:

有许多[实验跟踪](/web/20221118213143/https://neptune.ai/experiment-tracking)和管理工具，只需最少的设置即可自动为您保存所有数据，这使得消融研究更容易—[Neptune . ai](https://web.archive.org/web/20221118213143/https://neptune.ai/)在这里做得很好。

结束语

## 有很多方法可以调整你的模型，新的想法总是会出现。深度学习是一个快速发展的领域，没有灵丹妙药。我们必须进行大量实验，足够的试错导致突破。这篇文章已经包含了很多链接，但对于最渴求知识的读者来说，我还在下面添加了一个很长的**参考部分**，供你多阅读和运行一些笔记本。

进一步研究

### 参考

### 论文:

博客:

书籍:

Kaggle 竞赛:

Kaggle 笔记本:

Kaggle notebooks: