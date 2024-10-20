# 设置 Keras 深度学习框架和构建旅游推荐引擎的简明指南(第 1 部分)

> 原文：<https://www.dlology.com/blog/gentle-guide-to-setup-keras-deep-learning-framework-and-build-a-travel-recommendation-engine/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零九个月前

([评论](/blog/gentle-guide-to-setup-keras-deep-learning-framework-and-build-a-travel-recommendation-engine/#disqus_thread))

![travel](img/7c425f0e6e5fe33df2d8d2641df1a2bd.png)

TL；这篇文章将带你开始使用 Keras 深度学习框架，而不会有安装上的麻烦。我将向您展示在云上免费运行您的代码是多么容易。

我知道在 Keras 有很多教程可以帮助你开始深度学习。它们通常带有图像分类器，用于 MNIST 手写数字或猫/狗分类。在这里，我想采取一种不同但可能更有趣的方法，向您展示如何构建一个模型，在给定您喜欢的源图像的情况下，推荐您可能感兴趣的地方。

我们开始吧！

## 在云上运行，无需安装

如果你来自一个普通的编程背景，你可能曾经遭受过当你第一次开始一件新事物时的痛苦。

安装 IDE、库依赖、硬件驱动程序支持...在“你好，世界”第一次成功运行之前，它们可能已经花了你很多时间。

深度学习也是一样，它依赖于许多东西来使模型工作。比如，为了有一个深度学习模型训练和运行得更快，你需要一个显卡。对于初学者来说，这可能很容易需要几个小时来设置，更不用说您必须选择和购买显卡本身，这可能非常昂贵。

今天你可以消除深度学习的初始学习曲线。现在可以完全在云中运行您的代码，并且已经为您预装了所有必要的依赖项。更重要的是，你可以免费在显卡上更快地运行你的模型。

在这一点上，我想介绍一下[Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb)因为我发现与他人分享我的深度学习代码非常有用，他们可以在几秒钟内复制结果。

你所需要的只是一个 Gmail 账户和一个互联网连接。繁重的计算将由 Google Colab 服务器处理。

您需要熟悉 Colab 上的 Jupyter 笔记本环境。这很简单，你点击单元格左边的播放按钮来运行里面的代码。如果需要，您可以多次运行一个单元。

让我们通过激活 colab 上的 GPU 来加速深度学习模型的运行速度。

点击 **运行时** 菜单按钮，然后 **改变运行时类型** ，在 **硬件加速器** 下拉列表中选择 GPU。

![runtime](img/b53ef5f1a20373bb1bab4b4b043cc0a0.png)

我们已经为旅行做好了准备！现在系好安全带，因为我们将要进入深度学习世界的狂野西部。

## 对地点进行分类的深度学习模型

我们介绍的模型可以分辨出一幅图像包含哪些地方。

或者更正式地描述，模型的输入是图像数据，输出将是具有不同概率的地点列表。概率越高，图像包含相应场景/地点的可能性越大。

该模型可以对 365 个不同的地方进行分类，包括咖啡店、博物馆、户外等。

这里是用于本教程的 Colab 笔记本，您可以在阅读本文的同时尝试使用它。[**Keras _ travel _ place _ re commendation-part 1 . ipynb**](https://drive.google.com/file/d/1pbQGPy-E2-e_J31trPdrTtMoRHyadfoc/view?usp=sharing)

我们模型最重要的积木是 **卷积网络** ，它将起到提取图像特征的作用。从更一般的低级特征，如边/角，到更特定领域的高级特征，如图案和零件。

该模型将由几个卷积网络块一个接一个地堆叠而成。卷积层越深，提取的特征越抽象，层次越高。

这里有一张展示这个想法的图片。

![conv-features](img/5440ee077eaf6291a659b0b702638065.png)

现在有了卷积网络的直觉就够了。让我们通过构建一个模型来实现它。

用 **Keras** 框架搭建一个定制的深度学习模型真的很容易。Keras 是为人类设计的，不是机器。也是目前最流行的深度学习库——tensor flow 的官方高级 API。如果你刚入门，寻找深度学习框架。Keras 是正确的选择。

### 建立模型

如果您是第一次看到下面的 Keras 型号代码，请不要惊慌。其实理解起来挺简单的。该模型有几个卷积层块。正如我们前面解释的，每个块提取不同级别的图像特征。例如，“块 1”处于输入级别，它提取像边缘和拐角这样的入门级特征。越深入，每个块提取的抽象特征越多。你也注意到了由两个完全连接的**层形成的最终分类块，它们负责做出最终预测。**

 **### 预测图像的标签

让模型预测图像的标签。

![beach](img/65fa924332326ab9e54fca506a987f00.png)

该模型预期图像输入的固定形状为 244 x 244 像素，具有三个颜色通道(RGB)。但是如果我们有另一个不同分辨率的图像呢？Keras 有一些助手功能派上用场。

下面的代码将一幅图像转换成数据数组，然后在输入到模型之前进行一些数据值标准化。

然后我们将处理过的 shape 数组(3，244，244)输入到模型中，‘**【preds】**’变量是一个 365 个浮点数的列表，对应 365 个地点/场景。

选择前 5 个预测，并将它们的索引映射到地点/场景的实际名称。

这是结果。

```py
['beach', 'lagoon', 'coast', 'ocean', 'islet']
```

请随意尝试其他图像。

## 总结和进一步阅读

我们已经了解到，使用 Google Colab 快速获得预测地点/场景的深度学习模型是多么容易。阅读教程的第二部分 ，我将向你展示如何从图像中提取原始特征，并使用它来构建一个旅游推荐引擎。

与此同时，查看一些可能有用的资源。

Keras 文档[](https://keras.io/)，尤其是 [时序模型 API](https://keras.io/getting-started/sequential-model-guide/) 部分。

如果你想上传你的自定义图像到 Colab，请阅读我以前的一篇文章中的“ [用自定义图像预测](https://www.dlology.com/blog/how-to-run-object-detection-and-segmentation-on-video-fast-for-free/#predict-with-custom-images) ”部分。

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/gentle-guide-to-setup-keras-deep-learning-framework-and-build-a-travel-recommendation-engine/&text=Gentle%20guide%20to%20setup%20Keras%20deep%20learning%20framework%20and%20build%20a%20travel%20recommendation%20engine%20%28Part%201%29) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/gentle-guide-to-setup-keras-deep-learning-framework-and-build-a-travel-recommendation-engine/)

*   [←如何处理 Keras 中的消失/爆炸渐变](/blog/how-to-deal-with-vanishingexploding-gradients-in-keras/)
*   [设置 Keras 深度学习框架和构建旅游推荐引擎的温和指南(第二部分)→](/blog/gentle-guide-to-setup-keras-deep-learning-framework-and-build-a-travel-recommendation-engine-part-2/)**