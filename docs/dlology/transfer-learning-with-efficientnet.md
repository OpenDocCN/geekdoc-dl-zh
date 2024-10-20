# 如何利用 Efficientnet 进行迁移学习

> 原文：<https://www.dlology.com/blog/transfer-learning-with-efficientnet/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年零六个月前

([评论](/blog/transfer-learning-with-efficientnet/#disqus_thread))

![transfer](img/a4b2b2e8d986610c659837d7afc42cae.png)

在本教程中，您将学习如何创建图像分类神经网络来对您的自定义图像进行分类。该网络将基于最新的 EfficientNet，它在 ImageNet 上实现了最先进的精确度，同时比 小8.4 倍，比 快 6.1 倍。

## 为什么选择 EfficientNet？

与达到类似 ImageNet 精度的其他型号相比，EfficientNet 要小得多。例如，您可以在 Keras 应用程序中看到的 ResNet50 型号总共有 23，534，592 个参数，尽管如此，它的性能仍然低于最小的 EfficientNet，后者总共只使用了 5，330，564 个参数。

为什么这么高效？为了回答这个问题，我们将深入研究它的基本模型和构建模块。您可能听说过经典 ResNet 模型的构造块是恒等式和卷积块。

对于 EfficientNet 来说，它的主要积木是移动**倒瓶颈** MBConv，最早是在 [MobileNetV2](https://arxiv.org/abs/1801.04381) 推出的。通过在瓶颈之间直接使用快捷方式，其与扩展层相比连接更少数量的通道，结合**d****ee可分离卷积**，其与传统层相比有效地将计算减少了几乎一个因子 k²。其中 k 代表内核大小，指定 2D 卷积窗口的高度和宽度。

![building_blocks](img/5058005ad78a3091cb225129e15710d1.png)

作者还增加了[压缩和激励](https://arxiv.org/abs/1709.01507) (SE)优化，这有助于进一步性能的提高。 efficient net 的第二个优势是，它通过仔细平衡网络深度、宽度和分辨率来提高扩展效率，从而带来更好的性能。

![size_vs_accuracy](img/ecc35c7a064d71da4d77b46a2a317841.png)

如您所见，从最小的有效网络配置 B0 到最大的 B7，精度稳步提高，同时保持相对较小的尺寸。

## 用 EfficientNet 转移学习

如果你不完全确定我在上一节所讲的内容，没关系。用于图像分类的迁移学习或多或少是模型不可知的。如果你愿意，你可以选择任何其他预先训练的 ImageNet 模型，如 MobileNetV2 或 ResNet50 作为的替代。

预先训练的网络只是先前在大型数据集(如 ImageNet)上训练的保存的网络。学习到的特征可以证明对许多不同的计算机视觉问题是有用的，即使这些新问题可能涉及与原始任务完全不同的类别。例如，一个人可以在 ImageNet 上训练一个网络(其中的类主要是动物和日常物品)，然后重新利用这个训练好的网络做一些像识别图像中的[汽车型号](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)这样的远程工作。对于本教程，我们希望该模型能够在样本数量相对较少的情况下，很好地解决猫和狗的分类问题。

最简单的开始方式是在 Colab 中打开这个笔记本,我会在这篇文章中解释更多细节。

首先克隆我的存储库，它包含 EfficientNet 的 Tensorflow Keras 实现，然后将 cd 放入目录。

EfficientNet 是为 ImageNet 分类构建的，包含 1000 个类别标签。对于我们的数据集，我们只有 2 个。这意味着分类的最后几层对我们没有用。通过将`include_top`参数指定为 False，可以在加载模型时排除它们，这也适用于在 [Keras 应用](https://keras.io/applications/)中可用的其他 ImageNet 模型。

在 EfficientNet 卷积基础模型之上创建我们自己的分类层堆栈。我们采用`GlobalMaxPooling2D`将 4D 的 `(batch_size, rows, cols, channels)`张量转换成形状为`(batch_size, channels)`的 2D 张量。与`Flatten`层相比，`GlobalMaxPooling2D`产生数量少得多的特征，这有效地减少了参数的数量。

为了保持卷积基的权重不变，我们将冻结它，否则，先前从 ImageNet 数据集学习的表示将被破坏。

然后你可以从微软下载并解压`dog_vs_cat`数据。

[笔记本](https://github.com/Tony607/efficientnet_keras_transfer_learning/blob/master/Keras_efficientnet_transfer_learning.ipynb)中有几个数据块，专用于从原始数据集中抽取图像子集，以形成训练/验证/测试集，之后您将看到。

然后你可以用 Keras 的`ImageDataGenerator`对模型进行编译和训练，它在训练过程中增加了各种数据增广选项，以减少过拟合的机会。

另一种使模型表示与当前问题更相关的技术叫做微调。那是基于下面的直觉。

卷积基础中的早期层编码更通用、可重用的特征，而较高层编码更专业的特征。

微调网络的步骤如下:

*   1)在已经训练好的基础网络上添加您的自定义网络。
*   2)冻结基础网络。
*   3)训练你加的部分。
*   4)解冻基础网络中的一些层。
*   5)联合训练这些层和你添加的部分。

我们已经完成了前三个步骤，为了找出要解冻的图层，绘制 Keras 模型很有帮助。

这是卷积基础模型中最后几层的放大视图。

![fine_tuning](img/ea08820154c3c15bae3c611c700d8e3d.png)

设置“`multiply_16`”和可训练的连续层。

然后，您可以为更多的时期再次编译和训练模型。最后，您将拥有一个经过微调的模型，验证准确性提高了 9%。

## 结论和进一步阅读

这篇文章首先简要介绍了 EfficientNet，以及为什么它比传统的 ResNet 模型更有效。Colab Notebook 上的 runnable 示例向您展示了如何构建一个模型来重用 EfficientNet 的卷积库，并对自定义数据集的最后几层进行微调。

完整的源代码可以在[我的 GitHub repo](https://github.com/Tony607/efficientnet_keras_transfer_learning) 上找到。

#### 你可能会发现以下资源很有帮助。

[EfficientNet:反思卷积神经网络的模型缩放](https://arxiv.org/abs/1905.11946)

[MobileNetV2:反向残差和线性瓶颈](https://arxiv.org/abs/1801.04381)

[挤压和激励网络](https://arxiv.org/abs/1709.01507)

[TensorFlow 实现 EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/transfer-learning-with-efficientnet/&text=How%20to%20do%20Transfer%20learning%20with%20Efficientnet) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/transfer-learning-with-efficientnet/)

*   [←如何用 TensorFlow 模型优化将你的 Keras 模型 x5 压缩得更小](/blog/how-to-compress-your-keras-model-x5-smaller-with-tensorflow-model-optimization/)
*   [如何用 mmdetection 训练物体检测模型→](/blog/how-to-train-an-object-detection-model-with-mmdetection/)