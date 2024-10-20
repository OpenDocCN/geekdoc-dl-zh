# 教老狗新招——训练面部识别模型理解面部情绪

> 原文：<https://www.dlology.com/blog/teach-old-dog-new-tricks-train-facial-identification-model-to-understand-facial-emotion/>

###### 发帖人:[程维](/blog/author/Chengwei/) 4 年 11 个月前

([评论](/blog/teach-old-dog-new-tricks-train-facial-identification-model-to-understand-facial-emotion/#disqus_thread))

![emotions](img/b330baa2fb9d3280eb7b1ea00ad842d4.png)

[上周](https://www.dlology.com/blog/live-face-identification-with-pre-trained-vggface2-model/)，我们探索了使用预先训练好的 VGG-Face2 模型通过面部识别一个人。因为我们知道模型被训练来识别 8631 个名人和运动员。

原始数据集足够大，则由预训练网络学习的空间特征层次可以有效地充当我们的面部识别任务的通用模型。尽管我们的新任务可能需要识别完全不同的人的面孔。

有两种方法来利用预训练的网络:特征提取和微调。

我们上周做的是一个 *特征提取*的例子，因为我们选择使用提取的一张脸的特征来计算到另一张脸的距离。如果距离小于阈值，则我们识别它们来自同一个人。

本周，我们将利用相同的预训练模型来识别面部表情，我们将对这 7 种情绪进行分类。

```py
{0:'angry',1:'disgust',2:'fear',3:'happy', 4:'sad',5:'surprise',6:'neutral'}
```

你可以从 Kaggle 下载 [fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) 数据集。每张照片是 48x48 像素的人脸灰度图像。

## 首次尝试使用要素提取图层

我能想到的最简单的方法是把 CNN 特征提取层的输出叠加到一个分类器上。

分类器将是两个密集的完全连接的层，最后的密集层具有表示 7 种情绪的概率的输出形状 7。

我们冻结了 CNN 层的重量，使他们不可训练。因为我们想保留以前由卷积基学习的表示。

结果是这个模型不能很好地概括，并且总是预测一张“快乐”的脸。为什么“快乐”的脸，是模型更喜欢快乐的脸比其他脸像我们吗？不真的。

如果我们仔细看看 fer2013 训练数据集。模型要训练的开心脸比其他情绪多。所以如果模型只能挑一种情绪来预测，当然是挑笑脸。

![fer2013-data](img/67b5630995e5db5cef20859a9b264c12.png)

但是为什么模特总是只选择一张脸呢？这一切都来自于训练前的模型。因为预先训练的模型被训练来识别一个人，而不管他/她穿着什么样的情绪。因此最终的特征提取层将包含抽象信息以告诉不同的人。这些抽象的信息包括一个人眼睛的大小和位置、肤色以及其他信息。但有一点可以肯定，预先训练好的模型并不关心这个人的情绪。

那么我们能做的是，我们还能利用预训练模型的权重为我们做点什么吗？

答案是肯定的，我们要对模型进行“微调”。

## 微调预训练模型

![fine-tune](img/e5666b37bd1910cea15a3066069e3af9.png)

微调包括解冻用于特征提取的冻结模型库的几个顶层，并联合训练模型的新添加部分(在我们的情况下，全连接分类器)和这些顶层。这被称为“微调”,因为它稍微调整了被重用的模型的更抽象的表示，以便使它们与我们手头的任务更相关。

像预训练的 resnet50 face 模型这样的大型卷积网络有许多一个接一个的“Conv 块”堆栈。

下图显示了一个残差块示例。

![resnets_1](img/4bceeee4628c7f290a1ae227ca7a4b23.png)

图片来源:[http://torch.ch/blog/2016/02/04/resnets.html](http://torch.ch/blog/2016/02/04/resnets.html)

卷积基础中的早期层编码更通用、可重用的特征。在我们的例子中，第一个 conv 块可以提取边缘，一些图像模式，如曲线和简单的形状。对于不同的图像处理任务，这些都是非常通用的功能。再深入一点，conv 块可以提取一个人的眼睛和嘴巴，这是更抽象和不太通用的特征，用于不同的图像处理任务。

![features](img/d35734bbdfab3a72279ef108aa0ecbff.png)

最后的 conv 块可以表示与预训练模型任务相关联的更抽象的信息，以识别人。记住这一点，让我们解冻它的权重，并在我们用面部情绪图像训练时更新它。

我们通过定位层来解冻模型权重。在我们的例子中，命名为“activation_46”的层如下所示。

![last-resnet50-conv-block](img/ab4e277e9c9d74d14ebdab1d9a037153.png)

## 准备和训练模型

我们得到的是 35887 48x48 像素的人脸灰度图像。并且我们的预训练模型期望 224x224 彩色输入图像。

将所有 35887 图像转换为 224x224 大小并存储到 RAM 将占用大量空间。我的解决方案是一次将一幅图像转换并存储到一个 TFRecord 文件中，稍后我们可以用 TensorFlow 轻松加载该文件。

使用 TFRecord 作为训练数据集格式，它的训练速度也更快。你可以看看我之前的[实验](https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/)。

下面是让对话发生的代码。

在上面的代码中，train_data[0]包含每个都具有形状(48，48)的面部图像阵列的列表，并且 train_data[1]是独热格式的实际情感标签的列表。

例如，一种情绪被编码为

```py
[0, 0, 1, 0, 0, 0, 0]
```

指数 2 为 1，我们映射中的指数 2 是情绪“恐惧”。

为了用 TFRecord 数据集训练我们的 Keras 模型，我们首先需要用`tf.keras.estimator.model_to_estimator`方法将其转化为 TF 估计量。

我们已经在我之前的二进制分类任务中介绍了如何为 TF Estimator 编写图像输入函数。这里我们有 7 类情绪要分类。所以输入函数看起来有点不同。

下面的代码片段显示了主要的区别。

现在我们准备通过调用`train_and_evaluate`来训练和评估我们的模型。

总共花了 2 分钟，实现了 0.55 的验证精度。考虑到一张脸可能同时包含不同的情绪，这并不坏。比如既惊讶又开心。

## 总结与进一步思考

在这篇文章中，我们尝试了两种不同的方法来训练情绪分类任务的 VGG-Face2 模型。特征提取和微调。

特征提取方法不能概括面部情绪，因为原始模型被训练来识别不同的人而不是不同的情绪。

微调最后一个 conv 模块达到了预期的效果。

您可能想知道，如果我们微调更多的 conv 模块，会提高模型性能吗？

我们训练的参数越多，过度拟合的风险就越大。所以试图在我们的小数据集上训练它是有风险的。

源代码可以在我的 [GitHub repo](https://github.com/Tony607/Fine-Tune-Emotion) 上获得。

我也在考虑让这成为一个现场演示。敬请关注。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/teach-old-dog-new-tricks-train-facial-identification-model-to-understand-facial-emotion/&text=Teach%20Old%20Dog%20New%20Tricks%20-%20Train%20Facial%20identification%20model%20to%20understand%20Facial%20Emotion) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/teach-old-dog-new-tricks-train-facial-identification-model-to-understand-facial-emotion/)

*   [←预训练 VGGFace2 模型的活体人脸识别](/blog/live-face-identification-with-pre-trained-vggface2-model/)
*   [过拟合模型的两个简单配方→](/blog/two-simple-recipes-for-over-fitted-model/)