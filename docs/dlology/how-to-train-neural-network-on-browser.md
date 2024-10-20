# 如何在浏览器上训练神经网络

> 原文：<https://www.dlology.com/blog/how-to-train-neural-network-on-browser/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零五个月前

([评论](/blog/how-to-train-neural-network-on-browser/#disqus_thread))

![demo-pong-vid](img/f0103ed346f646a07948ca7e46eb9cbd.png)

无论你是深度学习的新手还是经验丰富的老手，建立一个训练神经网络的环境有时会很痛苦。让训练一个神经网络变得像加载一个网页，然后点击几下，你就可以马上用它进行推理一样简单，这是不是很棒？

在本教程中，我将向您展示如何使用浏览器上的框架 TensorFlow.js 以及从您的网络摄像头收集的数据和在您的浏览器上的训练来构建模型。为了让这个模型有用，我们将把一个网络摄像头变成传奇游戏“乒乓”的控制器。

## 让我们先玩游戏

指令在您的计算机上本地提供 web 应用程序，

*   下载 **[dist.zip](https://github.com/Tony607/webcam-pong/releases/download/V0.1/dist.zip)** 并解压到你的本地机器。
*   安装一个 HTTP 服务器，我的建议是 npm 全局安装 **http-server** ，

## 将预训练模型导出到 tfjs

如果只是想学习 web 应用部分，可以跳过这一节。

让我们首先将一个预先训练好的卷积网络导出为 TensorFlow.js(tfjs)格式。在本教程中，我选择了用 ImageNet 数据集训练的 DenseNet，但你也可以使用其他模型，如 MobileNet。尽量避免大型深度卷积网络，如 ResNets 和 VGGs，即使它们可能提供稍高的精度，但不适合边缘设备，如我们在浏览器上运行的情况。

第一步是用 Python 脚本将预先训练好的 DenseNet Keras 模型保存到一个 **.h5** 文件中。

然后，我们运行转换脚本，将. h5 文件转换为针对浏览器缓存优化的 tfjs 文件。在继续之前，通过 pip3 安装 tensorflowjs 转换脚本 python 包。

```py
pip3 install tensorflowjs
```

我们现在可以通过运行以下命令来生成 tfjs 文件，

你会看到一个名为 **model** 的文件夹，里面有几个文件。 **model.json** 文件定义了模型结构和权重文件的路径。并且预先训练的模型准备好为 web 应用服务。例如，您可以将**模型**文件夹重命名为 **serveDenseNet** 并复制到您的 web app served 文件夹，然后模型就可以这样加载了。

`window.location.origin`是 web 应用程序的 URL，或者如果您在 1234 端口本地提供它，它将是`localhost:1234`。`await`声明只是允许 web 应用程序在后台加载模型，而不冻结主用户界面。

此外，要认识到，由于我们加载的模型是一个图像分类模型，我们不需要顶部的图层，我们只需要模型的特征提取部分。解决方案是找到最顶层的卷积层，并截断前面代码片段中显示的模型。

## 从网络摄像头生成训练数据

为了准备回归模型的训练数据，我们将从网络摄像头中抓取一些图像，并使用 web 应用程序中的预训练模型提取它们的特征。为了简化用于采集训练数据的用户界面，我们仅使用三个值[-1，0，1]中的一个来标记图像。

对于通过网络摄像头获取的每幅图像，它将被输入预先训练的 DenseNet，以提取特征并保存为训练样本。在将图像通过特征提取器模型之后，224×224 的彩色图像将使其维数减少到形状为[7，7，1024]的图像特征张量。形状取决于你选择的预先训练好的模型，可以通过调用我们在上一节中选择的图层上的`outputShape`来获得，就像这样。

使用提取的图像特征而不是原始图像作为训练数据的原因有两个。首先，它节省了存储训练数据的内存，其次，它通过不运行特征提取模型来减少训练时间。

下面的片段展示了如何通过网络摄像头捕捉图像，并提取和汇总其特征。请注意，所有图像特征都以张量的形式保存，这意味着如果您的模型与浏览器的 WebGL 后端一起运行，它在 GPU 内存中一次可以安全包含多少训练样本是有限制的。所以不要指望用数千甚至数百个图像样本训练你的模型取决于你的硬件。

## 建立和训练神经网络

建立和训练你的神经网络，而不上传到任何云服务保护你的隐私，因为数据永远不会离开你的设备，看着它在你的浏览器上发生，使它更酷。

回归模型将图像特征作为输入，将其展平为向量，然后跟随两个完全连接的层，并生成一个浮点数来控制游戏。最后一个完全连接的层没有激活函数，因为我们希望它产生-1 到 1 之间的实数。我们选择的损失函数是训练期间的均方误差，以最小化损失。更多关于选择，看我的帖子- [如何选择最后一层激活和损失函数](https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/) 。

以下代码将构建、编译和拟合模型。看起来很像 Keras 的工作流程吧？

## 将网络摄像头变成乒乓控制器

正如您可能期望的那样，使用图像进行预测也类似于 Keras 语法。图像首先被转换成图像特征，然后被传递到训练好的回归神经网络，该网络输出-1 到 1 之间的控制器值。

一旦你训练好了模型，游戏开始运行，预测值就会传递下来，通过这个调用 `pong.updatePlayerSpeed(value)` 来控制玩家的球拍以可变速度向左或向右移动的速度。你可以通过调用、 来开始和停止游戏

## 结论和进一步的思考

在本教程中，您学习了如何使用 TensorFlow.js 在浏览器上训练神经网络，并将您的网络摄像头变成 Pong 控制器，识别您的动作。请随意检查我的源代码，并对它进行实验，修改它，看看结果如何，如激活函数，损失函数和交换到另一个预先训练的模型等。在具有即时反馈的浏览器上训练神经网络的美妙之处在于，它使我们能够尝试新的想法，更快地获得结果，因为我们的原型也使它更容易被大众接受。在我的 GitHub repo **[网络摄像头上查看完整源代码——pong](https://github.com/Tony607/webcam-pong)**。

![Pong-logo](img/262d0976452e442b81da9aa458a98cf0.png)

*   标签:
*   [keras](/blog/tag/keras/) ,
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-train-neural-network-on-browser/&text=How%20to%20train%20neural%20network%20on%20browser) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-train-neural-network-on-browser/)

*   [←如何用 CMSIS-NN 在微控制器上运行深度学习模型(第三部分)](/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn-part-3/)
*   [用神经计算棒制作 DIY 安全摄像机(第一部分)→](/blog/build-a-diy-security-camera-with-neural-compute-stick-part-1/)