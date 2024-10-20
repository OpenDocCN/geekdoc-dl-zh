# 如何使用 Keras 进行无监督聚类

> 原文：<https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/>

###### 发帖人:[程维](/blog/author/Chengwei/) 4 年 7 个月前

([评论](/blog/how-to-do-unsupervised-clustering-with-keras/#disqus_thread))

![food-cluster](img/cae6ebf9379f58b84058919611b5d9b1.png)

深度学习算法擅长将输入映射到给定标记数据集的输出，这要归功于其表达非线性表示的非凡能力。这种任务被称为分类，而有人必须标记这些数据。无论是标记 x 射线图像还是新闻报道的主题，它都依赖于人的干预，并且随着数据集变大，成本会变得相当高。

聚类分析或聚类是一种不需要标记数据的无监督机器学习技术。它通过相似性对数据集进行分组来做到这一点。

为什么您应该关注聚类或聚类分析？让我给你看一些想法。

## 聚类的应用

*   推荐系统，通过学习用户的购买历史，聚类模型可以通过相似性对用户进行细分，帮助你找到志同道合的用户或相关产品。
*   在生物学中，[序列聚类](https://en.wikipedia.org/wiki/Sequence_clustering)算法试图将有某种关联的生物序列分组。蛋白质根据它们的氨基酸含量进行分类。
*   图像或视频聚类分析，根据相似性将它们分组。
*   在医学数据库中，每个患者对于特定测试(例如，葡萄糖、胆固醇)可能具有不同的实值测量。首先对患者进行聚类可以帮助我们理解应该如何对实值特征进行宁滨，以减少特征稀疏性并提高分类任务(如癌症患者的生存预测)的准确性。
*   一般用例，为分类、模式发现、假设生成和测试生成一个简洁的数据摘要。

无论如何，聚类对于任何数据科学家来说都是一项宝贵的资产。

## 什么是好的集群

一个好的聚类方法将产生高质量的聚类，这些聚类应该具有:

*   类内相似性高:在类内具有内聚性
*   低类间相似性:聚类之间有区别

### 用 K 均值设定基线

传统的 K-means 算法速度快，适用范围广。然而，它们的距离度量被限制在原始数据空间，并且当输入维数很高时，例如图像，它往往是无效的。

让我们训练一个 K-Means 模型来将 MNIST 手写数字聚类成 10 个簇。

评估的 K-Means 聚类精度为 **53.2%** ，稍后我们将与我们的深度嵌入聚类模型进行比较。

我们即将介绍的模型由几个部分组成:

*   一个自动编码器，被预先训练以学习未标记数据集的初始压缩表示。
*   堆叠在编码器上的群集层，用于将编码器输出分配给群集。聚类层的权重基于当前评估用 K-Means’聚类中心初始化。
*   训练聚类模型以联合改进聚类层和编码器。

寻找源代码？拿到我的 [GitHub](https://github.com/Tony607/Keras_Deep_Clustering) 上。

## 预训练自动编码器

Autoencoder 是一种数据压缩算法，其中有两个主要部分，编码器和解码器。编码器的工作是将输入数据压缩到较低的维度特征。例如，28x28 MNIST 图像的一个样本总共有 784 个像素，我们构建的编码器可以将其压缩到一个只有 10 个浮点数的数组，也称为图像的特征。另一方面，解码器部分将压缩的特征作为输入，并重建尽可能接近原始图像的图像。Autoencoder 本质上是一种无监督学习算法，因为在训练过程中，它只需要图像本身，而不需要标签。

![autoencoder_schema](img/4b94e146e7c5549d49d1af9d1e5ff267.png)

我们建立的自动编码器是一个完全连接的对称模型，对称于图像如何以完全相反的方式压缩和解压缩。

![Fully connected auto-encoder](img/f65604f13a5b7aa40aa9ae884eb08ac8.png)

我们将为 300 个时期训练自动编码器，并为以后保存模型权重。

## 聚类模型

通过训练 autoencoder，我们让它的编码器部分学会了将每张图像压缩成十个浮点值。你可能会想，既然输入维数减少到 10，K-Means 应该能够从这里进行聚类？是的，我们将使用 K-Means 来生成聚类质心，这是 10 维特征空间中的 10 个聚类的中心。但是我们还将构建我们的自定义聚类层，以将输入要素转换为聚类标签概率。

概率由[学生的 t 分布](https://en.wikipedia.org/wiki/Student%27s_t-distribution)计算得出。与 t-SNE 算法中使用的相同，t-分布测量嵌入点和质心之间的相似性。正如您可能猜到的那样，聚类层的作用类似于 K-means 聚类，层的权重表示可以通过训练 K-means 来初始化的聚类质心。

如果你是第一次使用[在 Keras](https://keras.io/layers/writing-your-own-keras-layers/) 中构建定制层，有三个方法是你必须实现的。

*   `build(input_shape)`，定义层的权重，在我们的例子中是 10 维特征空间中的 10 个簇，即 10x10 个权重变量。
*   层逻辑存在的地方，也是从特性到聚类标签的神奇映射发生的地方。
*   `compute_output_shape(input_shape)`，在此指定从输入形状到输出形状的形状转换逻辑。

下面是自定义聚类层代码，

接下来，我们在预训练的编码器之后堆叠一个聚类层，以形成聚类模型。对于聚类层，我们正在初始化其权重，聚类中心使用 k-means 对所有图像的特征向量进行训练。

![deep-clustering-model](img/5a20a461714ad73855df90795ea1723d.png)

## 训练聚类模型

#### 辅助目标分布和 KL 发散损失

下一步是同时改进聚类分配和特征表示。为此，我们将定义基于质心的目标概率分布，并最小化其相对于模型聚类结果的 KL 散度。

我们希望目标分布具有以下属性。

*   加强预测，即提高聚类纯度。
*   更加重视被赋予高置信度的数据点。
*   防止大簇扭曲隐藏特征空间。

通过首先将 q(编码的特征向量)提高到二次幂，然后通过每个聚类的频率进行归一化，来计算目标分布。

有必要在辅助目标分布的帮助下，通过从高置信度分配中学习来迭代地改进聚类。在特定次数的迭代之后，目标分布被更新，并且聚类模型将被训练以最小化目标分布和聚类输出之间的 KL 发散损失。训练策略可以被看作是自我训练的一种形式。在自我训练中，我们采用一个初始分类器和一个未标记的数据集，然后用分类器标记数据集，以训练其高可信度预测。

损失函数，KL 散度或[kull back–lei bler 散度](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)它是两种不同分布之间行为差异的度量。我们希望将其最小化，以便目标分布尽可能接近聚类输出分布。

在下面的代码片段中，目标分布每 140 次训练迭代更新一次。

在每次更新后，您将看到聚类的准确性稳步提高。

## 评估指标

该指标表示，它已经达到了 **96.2%** 的聚类精度，考虑到输入是未标记的图像，这已经相当不错了。让我们仔细看看它的精度是如何得出的。

该指标采用来自无监督算法的聚类分配和基础事实分配，然后找到它们之间的最佳匹配。

最佳映射可以通过[匈牙利算法](https://en.wikipedia.org/wiki/Hungarian_algorithm)有效计算，该算法在 scikit 学习库中实现为**线性分配**。

看混淆矩阵更直接。

![confusion-matrix](img/00c7ea290fd44bf7298fae989bd3da34.png)

在这里，您可以手动快速匹配聚类分配，例如，聚类 1 匹配真实标签 7 或手写数字“7”和 vise visa。

下面显示了混淆矩阵的绘图代码片段。

## 应用卷积自动编码器(实验)

因为我们处理的是图像数据集，所以值得尝试使用卷积自动编码器，而不是只使用完全连接的层。

值得一提的是，为了重建图像，你可以选择去卷积层( [Conv2DTranspose](https://keras.io/layers/convolutional/#conv2dtranspose) 在 Keras)或上采样( [UpSampling2D](https://keras.io/layers/convolutional/#upsampling2d) )层，以减少伪影问题。卷积自动编码器的实验结果可在我的 [GitHub](https://github.com/Tony607/Keras_Deep_Clustering) 上获得。

## 结论和进一步阅读

我们已经学习了如何构建一个 keras 模型来对未标记的数据集执行聚类分析。预训练的 autoencoder 在降维和参数初始化中发挥了重要作用，然后根据目标分布训练定制的聚类层以进一步提高精度。

### 进一步阅读

在 Keras 中构建自动编码器

[用于聚类分析的无监督深度嵌入](https://arxiv.org/abs/1511.06335)——启发我写了这篇文章。

完整的源代码在我的 [GitHub](https://github.com/Tony607/Keras_Deep_Clustering) 上，请阅读到笔记本的末尾，因为您将发现另一种替代方法来同时最小化聚类和自动编码器损失，这被证明有助于提高卷积聚类模型的聚类精度。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/&text=How%20to%20do%20Unsupervised%20Clustering%20with%20Keras) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/)

*   [←利用 Keras 中的新闻数据进行简单的股票情绪分析](/blog/simple-stock-sentiment-analysis-with-news-data-in-keras/)
*   [如何训练 Keras 模型生成颜色→](/blog/how-to-train-a-keras-model-to-generate-colors/)