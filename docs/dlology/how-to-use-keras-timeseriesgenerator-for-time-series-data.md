# 如何使用 Keras TimeseriesGenerator 处理时序数据

> 原文：<https://www.dlology.com/blog/how-to-use-keras-timeseriesgenerator-for-time-series-data/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零三个月前

([评论](/blog/how-to-use-keras-timeseriesgenerator-for-time-series-data/#disqus_thread))

![series](img/b31aa4804e51b697f69e083daa24337e.png)

您可能遇到过预测模型，其任务是根据历史数据预测未来值。给定时间序列数据，准备输入和输出对是乏味的。我最近遇到了 Keras 内置的实用程序 **TimeseriesGenerator** ，它可以准确地完成我想要的功能。

在下面的演示中，您将了解如何将其应用到您的数据集。

#### 源代码可以在我的 [GitHub 库](https://github.com/Tony607/Keras_TimeseriesGenerator)上找到。

 *想象一下，你是一位拥有敏锐的数据科学意识的基金经理，想要根据公开的股票价格预测今天的道琼斯指数。

![dji](img/00deeb6b4483d6304015ee2d6426fe4e.png)

我们将使用日变化值作为时间序列数据，而不是使用在过去几年中增加了 60%的绝对 DJI 指数值。由于**数据集 _DJI** 代表绝对 DJI 指数，日变化值可通过下式计算

我们可以进一步归一化所有值，并将它们分成训练/测试数据集。

## 单一时间序列预测

你知道 RNN，或者更准确地说，LSTM 网络捕捉时间序列模式，我们可以建立这样一个模型，输入是过去三天的变化值，输出是当天的变化值。第三个是回看长度，可以针对不同的数据集和任务进行调整。简单地说，第 T 天的值是由第 T-3、T-2 和 T-1 天的值预测的。但是我们如何为模型构造训练和测试输入/输出对呢？Keras 的[TimeseriesGenerator](https://keras.io/preprocessing/sequence/#timeseriesgenerator)通过消除我们用来完成这一步骤的样板代码，使我们的生活变得更加轻松。

让我们构建两个时间序列生成器，一个用于训练，一个用于测试。我们使用一个采样率，因为我们不想跳过数据集中的任何样本。

在一个简单的 Keras 模型到位后，我们可以启动培训过程。

训练后的评估和最后的预测也可以这样做。

在此基础上，我们可以从预测的日变化值中重建预测的 DJI 绝对值，方法是首先反转最小-最大归一化过程，并将预测的日变化值与前一天的绝对值相加。

![compare1](img/89308c7901e3b5c35462a9b8e048e386.png)

## 多个时间序列作为输入

您可能已经注意到所有以前的**时间序列生成器**的“数据”和“目标”参数都是相同的，这意味着输入和输出都来自相同的时间序列。如果在现实生活中，今天 DJI 的收盘价可能会受到一些大公司如苹果和亚马逊之前股价的影响呢？我们还想将这些股票日变化值合并到模型输入中。为了实现这一点，您可以首先将所有三个时间序列连接起来，形成一个(T，3)形状的 numpy 数组，然后将预处理后的结果传递给 **TimeseriesGenerator** 的“数据”参数。

最后，确保更改模型输入形状以匹配(None，look_back，3)的输入形状。

## 结论

本快速教程向您展示了如何使用 Keras 的 TimeseriesGenerator 来减轻处理时间序列预测任务时的工作量。它允许您将相同或不同的时间序列作为输入和输出来训练模型。源代码可以在我的 [GitHub 库](https://github.com/Tony607/Keras_TimeseriesGenerator)上找到。

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-use-keras-timeseriesgenerator-for-time-series-data/&text=How%20to%20use%20Keras%20TimeseriesGenerator%20for%20time%20series%20data) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-use-keras-timeseriesgenerator-for-time-series-data/)

*   [←如何在 Google Colab 上运行支持 GPU 和 CUDA 9.2 的 py torch](/blog/how-to-run-pytorch-with-gpu-and-cuda-92-support-on-google-colab/)
*   [如何利用生成对抗网络在 Keras 中进行新颖性检测(上)→](/blog/how-to-do-novelty-detection-in-keras-with-generative-adversarial-network/)*