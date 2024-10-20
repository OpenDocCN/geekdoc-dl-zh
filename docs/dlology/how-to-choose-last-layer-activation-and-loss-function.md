# 如何选择最后一层激活和损失函数

> 原文：<https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/>

###### 发布者:[程维](/blog/author/Chengwei/)五年零一个月前

([评论](/blog/how-to-choose-last-layer-activation-and-loss-function/#disqus_thread))

![choices](img/7f0fbf636928d4f710af2d9f6b100ff7.png)

不多说了，下面是不同任务的末层激活和损失函数对的不同组合。

## 二元分类-狗对猫

Kaggle 上的这个[竞赛是你写一个算法来分类图像是包含一只狗还是一只猫。这是一个二进制分类任务，其中模型的输出是一个范围从 0 到 1 的单个数字，其中较低的值表示图像更像“猫”,而较高的值表示图像更像“狗”。](https://www.kaggle.com/c/dogs-vs-cats)

以下是最后一个完全连接层的代码和用于模型的损失函数

如果你对这个狗对猫任务的完整源代码感兴趣，可以看看 GitHub 上的这个[牛逼教程](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb)。

## 多类单标签分类- MNIST

任务是将手写数字的灰度图像(28 像素乘 28 像素)分类为 10 个类别(0 到 9)。该数据集带有 Keras 包，因此很容易尝试。

最后一层使用“ **softmax** ”激活，这意味着它将返回 10 个概率得分的数组(总和为 1)。每个分数将是当前数字图像属于我们的 10 个数字类之一的概率。

同样，GitHub 上提供了 MNIST 分类的完整源代码。

## 多类别、多标签分类-新闻标签分类

Reuters-21578 是大约 20000 条新闻的集合，分为 672 个标签。它们分为五大类:

*   主题
*   地方
*   人
*   组织
*   交换

例如，一条新闻可以有 3 个标签

*   地点:美国、中国
*   主题:贸易

你可以在我的 GitHub 上看看这个任务的[源代码。](https://github.com/Tony607/Text_multi-class_multi-label_Classification)

我也为这个任务写了另一篇详细的博客，如果你感兴趣的话可以看看。

## 回归任意值——博斯腾房价预测

目标是用给定的数据预测单个连续值，而不是离散的房价标签。

网络以没有任何激活的密集结束，因为应用任何激活函数(如 sigmoid)都会将值约束为 0~1，我们不希望这种情况发生。

**mse** 损失函数，它计算预测值和目标值之差的平方，这是一个广泛用于回归任务的损失函数。

完整的[源代码](https://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/3.7-predicting-house-prices.ipynb)可以在同一个 GitHub repo 中找到。

## 回归到 0 和 1 之间的值

对于像评估喷气发动机的健康状况这样的任务，提供几个传感器记录。我们希望输出是一个从 0 到 1 的连续值，其中 0 表示发动机需要更换，1 表示发动机状况良好，而 0 到 1 之间的值可能表示需要某种程度的维护。与之前的回归问题相比，我们将“sigmoid”激活应用于最后一个密集层，以将值限制在 0 到 1 之间。

如果你有任何问题，请留言。

*   标签:
*   [keras](/blog/tag/keras/) ,
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/&text=How%20to%20choose%20Last-layer%20activation%20and%20loss%20function) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/)

*   [←如何将内容“粘贴”到不支持复制粘贴的 VNC 控制台](/blog/how-to-paste-content-to-a-vnc-console-which-does-not-support-copy-and-paste/)
*   [如何对新闻类别进行多类多标签分类→](/blog/how-to-do-multi-class-multi-label-classification-for-news-categories/)