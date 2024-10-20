# 在 Google Colab 中运行 TensorBoard 的快速指南

> 原文：<https://www.dlology.com/blog/quick-guide-to-run-tensorboard-in-google-colab/>

###### 发帖人:[程维](/blog/author/Chengwei/) 4 年 8 个月前

([评论](/blog/quick-guide-to-run-tensorboard-in-google-colab/#disqus_thread))

![tunnel](img/88bf121f5efed1fcd7c1e38cd2cae8df.png)

*更新:如果您使用最新的 TensorFlow 2.0，请阅读此贴，了解任何 Jupyter 笔记本中的 TensorBoard 原生支持- [如何在 Jupyter 笔记本中运行 tensor board](https://www.dlology.com/blog/how-to-run-tensorboard-in-jupyter-notebook/)*

 *无论你是刚刚开始深度学习，还是经验丰富，想要快速实验，Google Colab 都是一个非常棒的免费工具，可以满足小众需求。如果你还没有看过我以前的教程的话，可以看看它的简短介绍。

您可能还知道 TensorBoard，这是一个在您训练模型时进行可视化的优秀工具。

更有趣的是，如果你正在用 TensorFlow 后端在 Keras API 中构建你的深度学习模型，它会附带基本的 TensorBoard 可视化。

我们将在 Colab 上训练一个 Keras 模型，并在用 TensorBoard 训练时可视化它。

但是有一件事我们需要先解决。您的 Google Colab 虚拟机运行在位于 Google 服务器机房的本地网络上，而您的本地机器可能在世界上的任何其他地方。

如何从本地机器访问 TensorBoard 页面？

我们将使用一个名为 [ngrok](https://ngrok.com/) 的免费服务通过隧道连接到您的本地机器。

这里有一个图表显示它是如何工作的。

![ngrok](img/302ee47efe730bb43acb243990c3b451.png)

你可能有过在 Linux 上建立网络隧道经验，这需要大量的配置和安装。然而，我发现用 ngrok 建立一个可靠的隧道连接非常容易。

您可以在阅读本教程的同时打开本 [Colab 笔记本](https://drive.google.com/file/d/1afN2SALDooZIHbBGmWZMT6cZ8ccVElWk/view?usp=sharing)，享受最佳的学习体验。

## 在 Colab 上设置 ngrok 并运行 TensorBoard

Ngrok 可执行文件可以直接下载到你的 Colab 笔记本上，运行那两行代码。

在此之后，可执行文件 **ngrok** 将被解压到当前目录。

接下来，让我们像这样启动背景中的张量板:

它假设张量板日志路径是”。/log”，这里我们要告诉 Keras 要日志文件。

然后，我们可以运行 ngrok 将 TensorBoard 端口 6006 隧道化到外部世界。该命令也在后台运行。

最后一步，我们获取公共 URL，在那里我们可以访问 colab TensorBoard 网页。

这将输出一个您可以点击的 URL，但是请等待！我们还没有训练我们的模型，所以你还没有从 TensorBoard 得到任何信息。

## Keras 的张量板

在本节中，我们将开始训练一个 Keras 模型，并要求 Keras 将 TensorBoard 日志文件输出到。/log 目录。

Keras 通过[回调](https://keras.io/callbacks/#tensorboard)输出 TensorBoard 日志文件，这允许您可视化您的训练和测试指标的动态图表，以及您的模型中不同层的激活直方图。您可以选择是否可视化单个组件，甚至可以选择希望 Keras 激活和加权直方图的频率。下面的代码将允许您可视化我们的 Keras 模型的所有可用组件。该模型本身是一个简单的两层 3×3 卷积，然后是两个密集层，以分类 MNIST 手写数字数据集。

当这个模型正在训练时，你可以打开之前的 ngrok 链接，它应该会显示 TensorBoard 页面。

如果你是 TensorBoard 的新手，在接下来的部分，我将向你展示一些想法，每一个组件意味着什么，什么时候它们会有用。

## TensorBoard 短途旅行

模型训练时，TensorBoard 会定期自动更新视图。

### 数量

在此选项卡中，您可以看到四个图表，acc、loss、acc_val 和 loss_val，分别代表训练准确性、训练损失、验证准确性和验证损失。在一个理想的训练过程中，我们期望精确度会随着时间的推移而增加，而损失会随着时间的推移而减少。但是，如果您看到在经过一些特定时期的训练后，验证准确性开始下降。您的模型可能会在训练集上过度拟合。我有两个简单的方法来快速制作过度合身的模特 。虽然模型过度拟合还有其他原因，但一个明显的原因是训练数据太小，解决方法是找到或生成更多数据。

![scalars](img/708af0298563559e974fad5632282fab.png)

我们模型的损耗和精确度符合我们的预期。

### 直方图

此选项卡显示模型中不同层的激活权重、偏差、梯度直方图的分布。这个图表有用的一个方法是通过可视化梯度的值，我们可以告诉一个模型是否受到 v 消失/爆炸梯度的影响，给我以前的帖子一个阅读[如何处理 Keras](https://www.dlology.com/blog/how-to-deal-with-vanishingexploding-gradients-in-keras/) 中的消失/爆炸梯度。

![histograms_conv2d_1](img/749f2e2362a227633ae195aa845a03e3.png)

对于其他选项卡,“图形”选项卡显示计算图形，这在构建自定义图层或损失函数时更有用。图像选项卡将模型权重可视化为图像。

## 总结和进一步阅读

您学习了如何在 Google Colab 笔记本上运行 TensorBoard，并通过利用免费的 ngrok 隧道服务在本地机器上访问它。

另外，您也可以在 Colab 中运行下面的代码，使用 [localtunnel](https://localtunnel.github.io/www/) 代替 ngrok。

要进一步了解如何使用 Keras 配置 TensorBoard，请参考[官方文档](https://keras.io/callbacks/#tensorboard)，您可以禁用一些不需要的功能来加速训练，因为为 TensorBoard 生成日志需要大量时间。

此外，如果你正在 TensorFlow 上直接建立模型，但还不太熟悉 TensorBoard，可以在 YouTube 上查看这个[动手 tensor board(tensor flow Dev Summit 2017)](https://www.youtube.com/watch?v=eBbEDRsCmv4)。

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/quick-guide-to-run-tensorboard-in-google-colab/&text=Quick%20guide%20to%20run%20TensorBoard%20in%20Google%20Colab) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/quick-guide-to-run-tensorboard-in-google-colab/)

*   [←如何为 Keras 分类器生成 ROC 图的简单指南](/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/)
*   [如何在 Keras 中使用 return_state 或 return _ sequences→](/blog/how-to-use-return_state-or-return_sequences-in-keras/)*