# YOLO 对象本地化如何与 Keras 协同工作的简明指南(第 1 部分)

> 原文：<https://www.dlology.com/blog/gentle-guide-on-how-yolo-object-localization-works-with-keras/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零九个月前

([评论](/blog/gentle-guide-on-how-yolo-object-localization-works-with-keras/#disqus_thread))

![magnifier](img/dab7c220cfa419fe7b8c84a5934c02ce.png)

博士，我们将深入一点，了解 YOLO 物体定位算法是如何工作的。

我见过一些令人印象深刻的对象本地化实时演示。其中一个是带有 [TensorFlow 物体检测 API](https://github.com/tensorflow/models/tree/master/research/object_detection) ，你可以自定义它来检测你的可爱宠物——一只浣熊。

玩了一会儿 API 后，我开始想知道为什么对象本地化工作得这么好。

很少有网上资源以明确易懂的方式向我解释。所以我决定为那些对物体定位算法 如何工作感兴趣的人写这篇文章。

这篇文章可能包含一些高级的主题，但是我会尽可能友好地解释它。

## 到对象定位

### 什么是对象定位，它与对象分类相比如何？

你可能听说过 ImageNet 模型，它们在图像分类方面做得很好。一个模型被训练来辨别在给定的图像中是否有特定的物体，例如汽车。

对象定位模型类似于分类模型。但是经过训练的定位模型也可以通过在物体周围绘制一个边界框来预测物体在图像中的位置。例如，一辆**汽车**位于下图中。边界框、中心点坐标、宽度和高度的信息也包括在模型输出中。

![localization](img/06810aba2a149bc9022afd6048b0ce0a.png)

让我们看看我们有 3 种类型的目标要检测

*   1-行人
*   两辆车
*   3 -摩托车

对于分类模型，输出将是代表每个类别概率的 3 个数字的列表。上图中只有一辆**汽车**，输出可能看起来像 `[0.001, 0.998, 0.001]`。第二类是汽车的概率最大。

本地化模型的输出将包括边界框的信息，因此输出将如下所示

![y output](img/1f63ee68df82c4172568b45266660d47.png)

*   p[c]:1 表示检测到任何物体。如果为 0，输出的其余部分将被忽略。
*   b[x]:x 坐标，物体的中心对应图像的左上角
*   b[y]:y 坐标，物体的中心对应图像的左上角
*   b[h] :边框的高度
*   b[w] :边框宽度

至此，你可能已经想出了一种简单的方法来进行物体定位，即在整个输入图像上应用一个滑动窗口。就像我们用放大镜一次看地图的一个区域，看看那个区域是否有我们感兴趣的东西。这种方法易于实现，甚至不需要我们训练另一个定位模型，因为我们可以使用流行的图像分类器模型，并让它查看图像的每个选定区域，并输出每类目标的概率。

但是这种方法非常慢，因为我们必须预测大量的区域，并尝试大量的盒子尺寸来获得更准确的结果。它是计算密集型的，因此很难实现像自动驾驶汽车这样的应用程序所需的良好的实时对象定位性能。

这里有一个小技巧可以让它变得更快一点。

如果你熟悉卷积网络的工作原理，它可以通过为你拿着**虚拟放大镜**来模拟**滑动窗口效应**。因此，它在网络的一次前向传递中为给定的边界框大小生成所有预测，这在计算上更有效。但是，基于我们如何选择步幅和尝试多少不同大小的边界框，边界框的位置不会非常精确。

![cnn slide window](img/dce048c25735d51110a2fcfedd19c731.png)

*鸣谢:Coursera deeplearning.ai*

在上图中，我们有形状为 16 x 16 像素和 3 个颜色通道(RGB)的输入图像。然后**卷积滑动窗口**显示在左上角蓝色方块中，大小为 14×14，步距为 2。意味着窗口一次垂直或水平滑动 2 个像素。如果在 14 x 14 区域中检测到 4 种类型的目标对象中的任何一种，输出的左上角会显示左上角的 14 x 14 图像结果。

如果你不完全确定我刚刚谈到的滑动窗口的卷积实现，没有问题，因为我们稍后解释的 YOLO 算法将处理它们。

### 为什么我们需要对象本地化？

一个明显的应用，自动驾驶汽车，实时检测和定位其他汽车，路标，自行车是至关重要的。

它还能做什么？

一个安全摄像头来跟踪和预测一个可疑的人进入你的财产？

或者在水果包装和配送中心。我们可以建立一个基于图像的体积传感系统。它甚至可以使用边界框的大小来接近传送带上的橙子大小，并进行一些智能分类。

![volumn-sensing](img/6f266a016d1183f86fbe63be725d440b.png)

你能想到物体定位的其他有用的应用吗？请在下面分享你的新鲜想法！

系列文章的第二部分“[YOLO 对象定位如何与 Keras 协同工作的简明指南(第二部分)](https://www.dlology.com/blog/gentle-guide-on-how-yolo-object-localization-works-with-keras-part-2/)”。

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/gentle-guide-on-how-yolo-object-localization-works-with-keras/&text=Gentle%20guide%20on%20how%20YOLO%20Object%20Localization%20works%20with%20Keras%20%28Part%201%29) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/gentle-guide-on-how-yolo-object-localization-works-with-keras/)

*   [←如何利用深度学习诊断电机健康状况](/blog/try-this-model-to-quickly-tell-if-it-is-a-faulty-motor-by-listening/)
*   你能相信一个 Keras 模型能区分非洲象和亚洲象吗？→