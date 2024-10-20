# 使用 TensorFlow Hub DELF 模块轻松识别地标图像

> 原文：<https://www.dlology.com/blog/easy-landmark-image-recognition-with-tensorflow-hub-delf-module/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零六个月前

([评论](/blog/easy-landmark-image-recognition-with-tensorflow-hub-delf-module/#disqus_thread))

![eiffel-tower](img/c3056d30561525e628b605079b34eae9.png)

你有没有想过谷歌图片搜索是如何在幕后工作的？我将向您展示如何以最少的配置构建一个利用 TensorFlow Hub 的 DELF(深度本地功能)模块的地标图像识别管道的迷你版本。

阅读时，请随意浏览 [Colab 笔记本](https://drive.google.com/file/d/1d718D4tzhPkRd56z3-oasWkeuRW6IGKI/view?usp=sharing)。

## 图像识别/检索简介

图像检索是在大型数据库中搜索数字图像的任务。它可以分为两类:基于文本的图像检索和基于内容的图像检索。在基于文本的图像检索方法中，我们只需在搜索栏中输入一个查询(或相关词)就可以得到图像结果。在基于内容的图像检索中，我们在搜索域中提供一个样本(但相关的)图像，以获得相似图像的结果。

![google-image-search](img/c7491d407d76a90628c316a9d499a678.png)

这篇文章关注的是基于内容的图像检索，其中图像通过特征提取过程自动标注其视觉内容。视觉内容包括颜色、形状、纹理或任何其他可以从图像本身得到的信息。 提取代表视觉内容的特征，通过高维索引技术进行索引，实现大规模图像检索。

如果你读过我之前的文章“T0:构建一个旅游推荐引擎 T1:T2”，这实际上是一个图像检索模型，依赖于提取的全局图像特征 T3，因此很难处理部分可见性和无关的图像特征。或者，更健壮的图像检索系统是基于局部特征的。它能够处理背景杂波、部分遮挡、多个地标、不同比例的物体等。

以大峡谷中的这两幅马蹄铁图像为例，它们处于不同的光照和尺度下。

![match-images](img/854d5ae98fd0c291c81c1d3c1dd2d3eb.png)

## 什么是 DELF(深度局部特征)模块？

tensor flow Hub上提供的预训练 DELF(深度局部特征)模块可用于图像检索，作为其他关键点检测器和描述符的替代。它用被称为特征描述符的 40 维向量来描述给定图像中每个值得注意的点。

下图显示了两幅图像的 DELF 对应关系。

![horseshoe-delf](img/3827156aa060cad79934bf4971cd7e7f.png)

DELF 使用 Google-Landmarks 数据集进行训练，该数据集包含来自 12 、 894 地标的 1、 060 、 709 图像以及针对地标优化的 036 附加查询图像 

DELF 图像检索系统可以分解为四个主要模块:

1.  密集局部化特征提取，
2.  关键点选择
3.  降维，
4.  标引和检索。

前 3 个块被包装到 TensorFlow Hub DELF 模块中。即便如此，打开黑匣子看看里面还是很有意思的。

dense 局部特征提取块由用分类损失训练的 ResNet50 CNN 特征提取层形成。所获得的特征图被视为局部描述符的密集网格。

特征基于它们的感受域被定位，这可以通过考虑全卷积网络(FCN)的卷积层和汇集层的配置来计算。他们使用感受野中心的像素坐标作为特征位置。下图显示了 256x256 分辨率输入图像的聚合唯一位置。 

![delf-locations](img/d1ad24b8296ddf05be8064b4cccc2f2f.png)

他们采用了两步训练策略，首先将原来的 ResNet50 层微调到 增强局部描述符的区分度，随后训练注意力得分函数来评估模型提取特征的相关性。 

![delf-training](img/70ee6727b1be1e95bf867173f420d15a.png)

在降维块中，特征维数通过 PCA 降低到 40，一种紧凑性和区分性的权衡。 至于索引和检索，我们将在下一节构建。

## 建立图像识别管道

为了进行演示，我们将创建这样一个地标图像识别系统，它将一幅图像作为输入，并判断它是否与 50 个世界著名建筑之一相匹配。

### 索引数据库图像

我们首先从 50 个数据库地标建筑图像中提取特征描述符，并聚集它们的描述符和位置。在我们的例子中，总共有 9953 个聚集的描述符-位置对。

![aggregate](img/e00dd7d0e4509c5590a7c236c9a34d8d.png)

我们的图像检索系统是基于最近邻搜索的，为了促进这一点，我们用所有聚集的描述符建立了一个 KD 树。

索引是离线执行的，并且只构建一次，除非我们将来想要索引更多的数据库映像。请注意，在下面的代码片段中，我们还创建了一个索引边界查找数组，用于在给定聚合描述符索引的情况下反向定位数据库映像索引。

### 运行时查询图像

在运行时，查询图像首先被调整大小并裁剪为 256x256 分辨率，然后 DELF 模块计算其描述符和位置。然后，我们查询 KD 树，为查询图像的每个描述符找到 K 个最近的邻居。接下来，汇总每个数据库映像的所有匹配。最后，我们使用 RANSAC 执行几何验证，并采用内联体的数量作为检索图像的分数。

下图说明了查询管道。

![query](img/61f109b858ddd1b4458563746d7b00c9.png)

关于应用 RANSAC 进行几何验证，有一点值得一提。我们希望确保所有匹配都符合全局几何变换；但是，有许多不正确的匹配。以下图为例，在没有几何验证的情况下，有许多不一致的匹配，而在应用 RANSAC 后，我们可以同时估计几何变换和一致匹配集。

![ransac](img/7b02a9329cfc85d12d3603830feb0e1c.png)

在我们的查询演示中，在 K 最近邻搜索之后，我们总共聚集了 23 个试验性的数据库图像，而在对每个试验性的查询图像应用 RANSAC 之后，只剩下 13 个候选图像。

以下代码片段使用 RANSAC 和可视化执行几何验证。

管道最终通过索引为每个数据库候选生成内联器计数分数。

然后打印出最匹配图像的描述及其索引就很简单了

哪些输出:

```py
Best guess for this image: 18\. The Colosseum — Rome, Italy
```

## 总结和进一步阅读

我们首先简要介绍图像识别/检索任务和 TensorFlow Hub 的 DELF 模块，然后构建一个演示图像识别管道来检索 50 个世界著名建筑。

最初的 [DELF 论文](https://arxiv.org/abs/1612.06321)给了我写这篇文章最大的灵感。

虽然有一些相关的资源，你可能会觉得有帮助。

[Kaggle - Google 地标识别挑战赛](https://www.kaggle.com/c/landmark-recognition-challenge)

[Kaggle - Google 地标检索挑战赛](https://www.kaggle.com/c/landmark-retrieval-challenge)

张量低集线器上的 DELF 模块

[Coursera - RANSAC:随机样本共识](https://www.coursera.org/learn/robotics-perception/lecture/z0GWq/ransac-random-sample-consensus-i)

最后，别忘了在[我的 GitHub](https://github.com/Tony607/Landmark-Retrival) 上查看这篇文章的源代码，并尝试一下可运行的 [Colab 笔记本](https://drive.google.com/file/d/1d718D4tzhPkRd56z3-oasWkeuRW6IGKI/view?usp=sharing)。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/easy-landmark-image-recognition-with-tensorflow-hub-delf-module/&text=Easy%20Landmark%20Image%20Recognition%20with%20TensorFlow%20Hub%20DELF%20Module) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/easy-landmark-image-recognition-with-tensorflow-hub-delf-module/)

*   [← Keras +通用语句编码器=针对文本数据的迁移学习](/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/)
*   [利用深度方向可分离卷积的简单语音关键词检测→](/blog/simple-speech-keyword-detecting-with-depthwise-separable-convolutions/)