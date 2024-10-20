# 过拟合模型的两个简单方法

> 原文：<https://www.dlology.com/blog/two-simple-recipes-for-over-fitted-model/>

###### 发帖人:[程维](/blog/author/Chengwei/) 4 年 11 个月前

([评论](/blog/two-simple-recipes-for-over-fitted-model/#disqus_thread))

![football](img/42234940efc9330aa229b0f1d93b550a.png)

过度拟合可能是一个严重的问题，特别是对于小的训练数据集。该模型可能会达到很高的训练精度，但当它带着从未见过的新数据进入现实世界时，它不能很好地概括新的示例。

第一个也是最直观的解决方案肯定是使用更大、更全面的数据集来训练模型，或者对现有数据集应用数据扩充，尤其是对于图像。但是如果我们只有这些数据呢？

在这篇文章中，我们将探索两种简单的技术来处理这个问题，在深度学习模型中使用正则化。

假设你刚刚被法国足球公司聘为人工智能专家。他们想让你推荐法国队守门员应该踢球的位置，这样法国队的球员就可以用头踢球了。

![field_kiank](img/a04ac4956d32806224e3fbc6f00bea01.png)

*图片来源:Coursera -改进深度神经网络*

下面是法国队过去 10 场比赛的 2D 数据。

你的目标是建立一个深度学习模型，找到场上守门员应该踢球的位置。

这个数据集有点嘈杂，但看起来像是一条对角线，将左上方(蓝色)和右下方(红色)分开，效果会很好。

![train data](img/a5e3aad90992646b9c34415e623c41d5.png)

让我们建立一个简单的 Keras 模型，有 3 个隐藏层。

![model](img/09b595a1a6dc1aadef62fc0077f5d3a0.png)

让我们用 1000 个时期的训练和测试数据集来训练和验证我们的模型。

它实现了如下所示的最终训练和验证准确性。看起来模型在训练期间比验证期间表现得更好。

```py
train acc:0.967, val acc: 0.930
```

让我们绘制训练/验证准确性和损失图。

![org_acc_loss](img/3f6470500f11e43f6e557bbd6f85650f.png)

正如我们可以看到的，在大约 600 个纪元后，验证损失停止下降，反而开始增加。模型开始过度适应训练数据集是正常的迹象。

为了清楚地了解最终训练好的模型“思考”的是什么，让我们画出它的决策边界。

![org_boundary](img/c18520b214fb321e39c37c95da3e2d6c.png)

如图所示，边界不清晰，模型过于努力地去拟合那些异常样本。对于深度神经网络来说，这可能变得非常明显，因为它能够学习数据点之间的完整关系，但与此同时，如果我们只用一个小数据集来训练它，它会过度拟合那些有噪声的离群值。

## 配方一——L2 正规化

避免过度拟合的标准方法称为**【L2 正则化】** 。它包括对层权重应用惩罚。然后将惩罚应用于损失函数。

因此，最终正则化的损失函数将包含交叉熵成本以及 L2 正则化成本。

例如，我们可以将层“2”的 L2 正则化成本计算为

```py
np.sum(np.square(W2))
```

其中“W2”是密集层“2”的权重矩阵。我们必须对 W2、W3 这样做，然后将它们相加并乘以正则化因子，正则化因子控制正则化的**强度**。

在 Keras 中，很容易将 L2 正则化应用于核权重。

我们为我们的 Keras 模型选择了因子 0.003，最终达到了的训练和验证精度

```py
train acc:0.943, val acc: 0.940
```

请注意，最终验证精度非常接近训练精度，这是一个好迹象，表明旅游模型不太可能过度拟合训练数据。

![l2_acc_loss](img/fe2beaf0ca9de6b2d3108d6590cbedc0.png)

与之前没有正则化的模型相比，决策边界也相当清晰。

![l2_boundary](img/6bb9b804d13b6a9c28e99fcf61385cae.png)

## 食谱二-辍学

Dropout 是深度学习中普遍使用的正则化技术。它在每次迭代中随机关闭一些神经元。

任何神经元被关闭的概率由**退出率**参数控制。放弃的想法是，在每次迭代中，模型只使用神经元的子集，因此，模型对任何特定神经元变得不太敏感。

有一点要记住。我们只在训练时应用 dropout，因为我们希望使用之前学习的所有神经元的权重进行测试或推理。别担心，当我们调用`fit()` `evaluate()` 或 `predict()` 时，这在 Keras 中是自动处理的。

但是我们如何选择**辍学率**参数呢？

简短回答:如果你不确定，0.5 是一个很好的起点，因为它提供了最大的正则化量。

逐层使用不同的退出率也是可行的。如果前一层具有较大的权重矩阵，我们可以对其应用较大的丢弃。在我们的示例中，我们在 **dense_3** 层之后应用更大的下降，因为它具有最大的权重矩阵 40x40。我们可以在 **dense_2** 层之后应用更小的丢失，比如丢失率 0.4，，因为它具有更小的权重矩阵 20x40。

让我们来看看将辍学应用到我们的模型后的结果。

![dropout_acc_loss](img/cba1c4436c7314c68dd278374881ef9e.png)

决赛

```py
train acc:0.938, val acc: 0.935
```

决定也相当顺利。

![dropout_boundary](img/b72141a9c3bf502c7dde0de3437746c5.png)

## 摘要

我们探索了两种简单的正则化方法来解决深度学习模型在用小数据集训练时遭受过拟合问题。正则化将使权重降低。L2 正则化和丢失是两种有效的正则化技术。在 [my GitHub repo](https://github.com/Tony607/Keras_Regularization) 中查看这篇文章的完整源代码。尽情享受吧！

*   标签:
*   [keras](/blog/tag/keras/) ,
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/two-simple-recipes-for-over-fitted-model/&text=Two%20Simple%20Recipes%20for%20Over%20Fitted%20Model) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/two-simple-recipes-for-over-fitted-model/)

*   [←教老狗新招——训练面部识别模型理解面部情绪](/blog/teach-old-dog-new-tricks-train-facial-identification-model-to-understand-facial-emotion/)
*   [如何在 Keras 中选择优化器的快速说明→](/blog/quick-notes-on-how-to-choose-optimizer-in-keras/)