# 使用 Keras 模型构建新张量流数据集和估计量的简单指南

> 原文：<https://www.dlology.com/blog/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-keras-model/>

###### 发布者:[程维](/blog/author/Chengwei/)五年零一个月前

([评论](/blog/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-keras-model/#disqus_thread))

![](img/5a46a1ebe43a1bdd75a32b8f0bc27535.png)

#### 更新:

*   2019 年 5 月 29 日:[源代码](https://github.com/Tony607/Keras_catVSdog_tf_estimator)更新运行在 TensorFlow 1.13 上。

TensorFlow r1.4 于不久前 10 月下旬发布。

如果你还没有更新，

```py
pip3 install --upgrade tensorflow-gpu
```

一些变化值得注意，

*   Keras 现在是核心 TensorFlow 包的一部分
*   **数据集** API 成为核心包的一部分
*   对**估算器**的一些增强允许我们将 Keras 模型转换为 TensorFlow 估算器，并利用其数据集 API。

在这篇文章中，我将向您展示如何将 Keras 图像分类模型转换为 TensorFlow estimator，并使用数据集 API 训练它来创建输入管道。

如果你没有读过 TensorFlow 团队的[介绍 TensorFlow 数据集和估值器](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html)的帖子。现在就阅读它，了解我们为什么在这里工作。

看到你刚好在一个你不能访问任何谷歌网站的地区，这很糟糕，所以我在这里为你总结了一下。

## TensorFlow 数据集 API 和估算器概述

### 数据集 API

您应该使用 Dataset API 为 TensorFlow 模型创建输入管道。这是最佳实践方式，因为:

*   数据集 API 提供了比旧 API(`feed_dict`或基于队列的管道)更多的功能。
*   性能更好。
*   它更干净，更容易使用。
*   它的图像模型管道可以从分布式文件系统中的文件聚集数据，对每个图像应用随机扰动，并将随机选择的图像合并成一批用于训练。
*   我还有用于文本模型的管道

### 估计量

Estimators 是一个高级 API，它减少了您以前在训练 TensorFlow 模型时需要编写的大量样板代码。

创建估算器的两种可能方式:预先制作估算器以生成特定类型的模型，另一种是使用其基类创建自己的估算器。

Keras 与其他核心 TensorFlow 功能顺利集成，包括估算器 API

好了，介绍够了，让我们开始构建我们的 Keras 估计器吧。

## 认真对待代码

为了简单起见，让我们为著名的狗与猫图像分类建立一个分类器。

作为 2013 年末计算机视觉竞赛的一部分，Kaggle.com 提供了猫和狗的数据集。您可以在 https://www.kaggle.com/c/dogs-vs-cats/download/train.zip的 [下载原始数据集](https://www.kaggle.com/c/dogs-vs-cats/download/train.zip)

我们只使用了一小部分训练数据。

下载并解压缩后，我们将创建一个包含三个子集的新数据集:一个包含每个类 1000 个样本的训练集，一个包含每个类 500 个样本的测试集。这里省略了这部分代码，看看我的 [GitHub](https://github.com/Tony607/Keras_catVSdog_tf_estimator) 就知道了。

### Build Keras model

我们正在利用预训练的 VGG16 模型的卷积层。又名模型的“卷积基础”。然后我们添加自己的分类器全连接层来做二元分类(猫 vs 狗)。

注意，由于我们不想触及“卷积基”中预训练的参数，所以我们将它们设置为不可训练。想深入了解这种模式的工作原理吗？看看这个伟大的  [jupyter 笔记本](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb) 作者 Keras。

### 张量流估计的 Keras 模型

model_dir 将是我们存储经过训练的 tensorflow 模型的位置。训练进度可以通过 TensorBoard 查看。
我发现我必须指定完整的路径，否则，否则 Tensorflow 会在以后的训练中抱怨它。

### 图像输入功能

当我们训练我们的模型时，我们需要一个读取输入图像文件/标签并返回图像数据和标签的函数。评估人员要求您创建以下格式的函数:

返回值必须是由两个元素组成的元组，如下所示:

-第一个元素必须是一个字典，其中每个输入特征都是一个键。这里我们只有一个“ **input_1** ”，它是模型的输入层名称，该模型将处理后的图像数据作为训练批次的输入。
-第二个元素是训练批次的标签列表。

这里有个重要的代码，它为我们的模型创建了输入函数。

此函数的参数

*   **文件名**，图像文件名的数组
*   **labels=None** ，模型的图像标签数组。为推断设置为“无”
*   **perform_shuffle=False** ，在训练时有用，读取 batch_size 记录，然后打乱(随机化)它们的顺序。
*   **repeat_count=1** ，在训练时有用，在每个历元重复输入数据几次
*   **batch_size=1** ，一次读取 batch_size 条记录

作为健全性检查，让我们试运行 imgs_input_fn()并查看它的输出。

它输出我们图像的形状和图像本身

```py
(20, 150, 150, 3)
```

![processed](img/27ae798584b5b769d18b02c4f9edf9ef.png)

对于我们的模型，看起来颜色通道“RGB”已经变成了“BGR ”,形状大小调整为(150，150)。这就是 VGG16 的“卷积库”所期望的正确的输入格式。

### 训练和评估

tensor flow 1.4 版本还引入了实用程序函数 `tf.estimator.train_and_evaluate` ，该函数简化了训练、评估和导出估计器模型。

此功能支持培训和评估的分布式执行，同时仍支持本地执行。

模型训练结果将被保存到**。/models/ catvsdog** 目录。有兴趣可以看看 TensorBoard 里的总结

```py
tensorboard --logdir=./models/catvsdog
```

### 预测

这里我们只预测 test_files 中的前 10 个图像。

为了进行预测，我们可以将**标签**设置为**无**，因为这就是我们将要预测的。 **dense_2** 是我们模型的输出层名称，`prediction['dense_2'][0]`将是一个介于 0~1 之间的浮点数，其中 0 表示猫的图像，1 表示狗的图像。

检查预测结果

它输出

```py
Predict dog: [False, False, True, False, True, True, False, False, False, False]
Actual dog : [False, False, True, False, True, True, False, False, False, False]
```

该模型正确地对所有 10 幅图像进行了分类。

## 摘要

我们构建一个 Keras 图像分类器，将其转换为张量流估计器，为数据集管道构建输入函数。最后，对模型进行训练和评估。请继续查看我的 GitHub repo 中这篇文章的[完整源代码。](https://github.com/Tony607/Keras_catVSdog_tf_estimator)

### 进一步阅读

[tensor flow 数据集和估算器简介](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html)-谷歌开发者博客

[宣布 tensor flow r 1.4](https://developers.googleblog.com/2017/11/announcing-tensorflow-r14.html)-谷歌开发者博客

[TensorFlow r1.40 发行说明](https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md)

[数据集 API 指南](https://www.tensorflow.org/programmers_guide/datasets)

[估算师 API 指南](https://www.tensorflow.org/programmers_guide/estimators)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-keras-model/&text=An%20Easy%20Guide%20to%20build%20new%20TensorFlow%20Datasets%20and%20Estimator%20with%20Keras%20Model) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-keras-model/)

*   [←如何对新闻类别进行多类多标签分类](/blog/how-to-do-multi-class-multi-label-classification-for-news-categories/)
*   [如何利用 TensorFlow 的 TFRecord 训练 Keras 模型→](/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/)