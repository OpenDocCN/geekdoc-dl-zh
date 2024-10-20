# Keras 中卷积神经网络用于图像分类的技巧包

> 原文：<https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年前

([评论](/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/#disqus_thread))

![tricks](img/9e8a2a42e2002cea181441e898816cf2.png)

本教程向你展示了如何在 Keras API 中实现一些图像分类任务的技巧，如论文中所述。这些技巧适用于各种 CNN 模型，如 ResNet-50、Inception-V3 和 MobileNet。

## 大批量训练 

对于相同数量的时期，与使用较小批量训练的模型相比，使用较大批量训练的模型会导致验证准确性下降。四种启发式方法有助于最大限度地减少大批量训练的负面影响，提高准确性和训练速度。 

### 线性缩放学习率 

随着批量大小线性增加学习率 

例如

| **批量大小** | **学习率** |
| Two hundred and fifty-six | Zero point one |
| 256 * 2 = 512 | 0.1 * 2 = 0.2 |

在 Keras API 中，您可以像这样缩放学习速率和批量大小。

### 学习率预热

使用太大的学习率可能会导致数值不稳定，尤其是在训练的最开始，这里参数是随机初始化的。在最初的 ***N*** 个时期或 ***m*** 个批次期间，预热策略 i 将学习率从 0 线性增加到初始学习率。 

尽管 Keras 附带了 [LearningRateScheduler](https://keras.io/callbacks/#learningratescheduler) 能够更新每个训练时期的学习率，但为了更好地更新每个批次，这里有一个方法，你可以实现一个定制的 Keras 回调来完成这个任务。

`warm_up_lr.learning_rates`现在包含了每个训练批次的预定学习率的数组，让我们将它可视化。

![warmup](img/10c5fc5a9b24efd5a6a366fc251e3384.png)

### 零 γ 每个 ResNet 块的最后一批归一化层

批量归一化用γ缩放一批输入，用β移位，两个 γ 和 β 是可学习的参数，其元素在 Keras 中默认分别初始化为 1 和 0。

在零γ初始化试探法中，我们为位于残差块末端的所有 BN 层初始化γ = 0。因此，所有残差块仅返回它们的输入，模仿具有较少层数并且在初始阶段更容易训练的网络。

给定一个恒等 ResNet 块，当最后一个 BN 的 γ初始化为零时，这个块只将快捷输入传递给下游层。 ![zero_gamma](img/4fd2dec3d3895a33f3435f967cce85ae.png)


你可以看到这个 ResNet 块是如何在 Keras 中实现的，唯一的变化是行， [BatchNormalization](https://keras.io/layers/normalization/#batchnormalization) 层的`gamma_initializer='zeros'`。

### 无偏衰变 

标准的权重衰减 对所有参数应用 L2 正则化将它们的值推向 0。它包括对层权重应用惩罚。然后将惩罚应用于损失函数。

建议仅对权重应用正则化，以避免过拟合。其他参数，包括 BN 层中的偏压和γ和β，保持不变。 

在 Keras 中，将 L2 正则化应用于核权重是毫不费力的。选项**bias _ regulator**也可用，但不推荐。

## 训练改进

### 余弦学习率衰减 

在前面描述的学习率预热阶段之后，我们通常会从初始学习率开始稳步降低其值。与一些广泛使用的策略(包括指数衰减和阶跃衰减)相比，余弦衰减在开始时缓慢降低学习速率，然后
在中间变得几乎线性降低，并在结束时再次减慢。它潜在地提高了训练进度。
![cosine_decay](img/628b18afc8252b1aa5aab3f4a2a91469.png)

这是一个完整的带有预热阶段的余弦学习率调度程序的例子。在 Keras 中，调度程序在每个更新步骤的粒度上更新学习率。

您选择使用调度器中的**保持 _ 基本 _ 速率 _ 步数**参数，顾名思义，它在进行余弦衰减之前保持特定步数的基本学习速率。由此产生的学习率时间表将有一个平台，如下所示。

![cosine_decay_hold](img/6c8d0174ce44badfa390e34ff08c7599.png)

### 标签平滑

与原始的独热编码输入相比，标签平滑将真实概率的构造改变为，

![label_smoothing](img/d805cb6168aeb594869bcd73cdf4c1ed.png)

其中ε 为小常数，K 为总类数。标签平滑鼓励全连接层的有限输出，以使模型更好地泛化，不容易过度拟合。对于标签噪声，这也是一种高效的和理论上接地的解决方案。你可以在这里阅读更多关于[讨论](https://qr.ae/TUnRbn)的内容。 以下是在训练分类器之前，如何对单热点标签应用标签平滑。

结果

## 结论和进一步阅读

在这篇文章中，还有两个培训优化没有涉及到，即，

*   **知识提炼**利用预先训练的较大模型的输出来训练较小的模型。
*   **Mixup 训练**，某种意义上类似于增强，它通过对两个样本进行加权线性插值，形成新的样本，从而创造更多的数据。看看在另一个帖子中实现这个[。](https://www.dlology.com/blog/how-to-do-mixup-training-from-image-files-in-keras/)

阅读报纸[https://arxiv.org/abs/1812.01187v2](https://arxiv.org/abs/1812.01187v2)了解每一个技巧的详细信息。

源代码可在 [my GitHub](https://github.com/Tony607/Keras_Bag_of_Tricks) 上获得。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/&text=Bag%20of%20Tricks%20for%20Image%20Classification%20with%20Convolutional%20Neural%20Networks%20in%20Keras) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/)

*   [←不平衡数据集的聚焦损失多类分类](/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/)
*   [如何从 Keras 中的图像文件进行混音训练→](/blog/how-to-do-mixup-training-from-image-files-in-keras/)