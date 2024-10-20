# 如何在 Movidius 神经计算棒上运行 Keras 模型

> 原文：<https://www.dlology.com/blog/how-to-run-keras-model-on-movidius-neural-compute-stick/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零四个月前

([评论](/blog/how-to-run-keras-model-on-movidius-neural-compute-stick/#disqus_thread))

![keras-ncs](img/03a05da6e25a4bccc94a85cc2a9a4a56.png)

Movidius neural compute stick(NCS)以及其他一些硬件设备，如 UP AI Core、AIY vision bonnet 和最近发布的[谷歌 edge TPU](https://cloud.google.com/edge-tpu/) 正在逐步将深度学习引入资源受限的 IOT 设备。我们离制造你一直期待的 DIY 猎杀无人机又近了一步吗？过去用于实现严重深度学习图像处理的强大 GPU 可以缩小到更即插即用的大小，可以将其视为一种移动中的迷你神经网络。如果这开始听起来有点太“终结者”了，请阻止我。

作为一名熟悉 Keras 深度学习框架的开发者和程序员，你很有可能能够将你训练的模型部署到 NCS 上。在本教程中，我将向您展示训练一个简单的 MNIST Keras 模型并将其部署到 NCS 是多么容易，NCS 可以连接到 PC 或 Raspberry Pi。

有几个步骤，

1.  在 Keras (TensorFlow 后端)中训练模型
2.  在 Keras 中保存模型文件和权重
3.  Turn Keras model to TensorFlow
4.  将张量流模型编译成 NCS 图
5.  在 NCS 上部署和运行图形

让我们逐一看看。

#### *这篇文章的源代码可以在我的 GitHub repo-**[keras _ mnist](https://github.com/Tony607/keras_mnist)上找到。***

## 培训和保存 Keras 模型

就像“你好世界！”首先在你的控制台上打印，训练一个手写数字 MNIST 模型相当于深度学习程序员。

这是一个 Keras 模型，它有几个卷积层，后面有一个最终输出级。完整的 [train-mnist.py](https://github.com/Tony607/keras_mnist/blob/master/train-minst.py) 代码在我的 GitHub 上，这里有一个简短的片段向您展示这一点。

在 GPU 上训练可能需要 3 分钟，在 CPU 上可能需要更长时间，顺便说一句，如果你现在没有可用的 GPU 训练机，你可以免费查看[我之前的教程](https://www.dlology.com/blog/how-to-run-object-detection-and-segmentation-on-video-fast-for-free/)如何在谷歌的 GPU 上训练你的模型，你需要的只是一个 Gmail 帐户。

无论哪种方式，在训练之后，将模型和权重保存到两个独立的文件中，如下所示。

或者，您可以调用`model.save('model.h5', include_optimizer=False)`到将模型保存在一个文件中，注意，我们通过将`include_optimizer`设置为`False`， 来排除优化器，因为优化器仅用于训练。

## Turn Keras to TensorFlow model

由于 Movidius NCSDK2 只编译 TensorFlow 或 Caffe 模型，我们将剥离绑定到 TensorFlow 图的 Keras。下面的代码处理这项工作，让我们看看它是如何工作的，以防将来您可能想要定制它。

首先，我们关闭学习阶段，然后从我们之前保存的两个单独的文件中以标准的 Keras 方式加载模型。

通过从具有 TensorFlow 后端的 Keras 调用 `K.get_session()` ，一个默认 TensorFlow 会话将可用。您甚至可以通过调用 `sess.graph.get_operations()` 来进一步探索张量流图内部的内容，其中返回您的模型中的张量流操作列表。这对于搜索 NCS 不支持的操作很有用，您可以从列表中找到它。最后，TensorFlow Saver 类将模型保存到指定路径的四个文件中。

每个文件都有不同的用途，

1.  **检查点**定义了模型检查点路径，在我们的例子中是“tf_model”。
2.  **。meta** 存储图形结构，
3.  **。数据**存储图表中每个变量的值
4.  T1。索引标识检查点。

## 用 mvNCCompile 编译张量流模型

NCSDK2 工具包附带的 **mvNCCompile** 命令行工具将 Caffe 或 Tensorflow 网络转换为可由 Movidius 神经计算平台 API 使用的图形文件。在图形生成期间，我们将输入和输出节点指定为 **mvNCCompile** 的 TensorFlow 操作名称。通过 调用 `sess.graph.get_operations()` ，可以找到 TensorFlow 操作的列表，如前一节所示。在我们的例子中，我们将' **conv2d_1_input** '作为输入节点，将' **dense_2/Softmax** '作为输出节点。

最后，编译命令看起来像这样，

```py
mvNCCompile TF_Model/tf_model.meta -in=conv2d_1_input -on=dense_2/Softmax
```

将在当前目录下生成一个名为“graph”的默认图形文件。

## 展开图表并做出预测

NCSDK2 Python API 接管，找到 NCS 设备，连接，将图形分配到其内存中，并进行预测。

下面的代码展示了本质部分，而`input_img`是将预处理后的图像作为 numpy 数组的形状(28，28)。

输出与 Keras 相同，十个数字代表十个数字中每一个的分类概率，我们应用 `argmax` 函数 找到最可能预测的索引。

Keras 模型现在在 NCS 上运行！你可以在这里调用它，或者通过添加一个网络摄像头来读取实时图像并在 Raspberry Pi 单板计算机而不是 Ubuntu PC 上运行来进一步增强演示。点击查看[视频演示。](https://youtu.be/tlhwfjOs2Sk)

## 树莓派上的网络摄像头实时图像预测奖金

在 Pi 上安装 NCSDK2 可能需要几十分钟，这对那些没有耐心的人来说不是个坏消息。但是好消息是，您可以选择只在您的 Pi 上安装 NCSDK2 的必要部分，以便使用在您的 Ubuntu PC 上编译的图形来运行推断。

不要将 NCSDK2 库克隆到您的 Pi，这可能需要很长时间，下载 NCSDK2 的发布版本 zip 文件,这可以节省大量磁盘空间，因为所有 git 版本控制文件都被跳过了。

其次，通过修改 `ncsdk.conf` 文件中的，在 NCSDK2 安装过程中跳过 TensorFlow 和 Caffe 安装。

运行一个实时网络摄像头需要安装 OpenCV3，在终端中运行下面四行代码就可以了。

```py
sudo pip3 install opencv-python==3.3.0.10
sudo apt-get update
sudo apt-get install libqtgui4
sudo apt-get install python-opencv
```

一旦 NCSDK2 和 OpenCV 3 安装完成，将您生成的**图**文件复制到您的 Pi 中。请记住，由于我们跳过了相当多的东西， **mvNC**** 命令不会在您的 Pi 上运行，因为它们依赖于 Caffe 和 TensorFlow 安装。

训练 MNIST 模型以识别 28×28 分辨率的灰度图像的黑色背景中的白色手写数字，为了转换捕获的图像，一些预处理步骤是必要的。

1.  裁剪图像的中心区域
2.  进行边缘检测以找到图像的边缘，这一步也将图像转换为灰度
3.  扩张边缘，使边缘变得更厚，以填充两条紧密平行边缘之间的区域。
4.  将图像大小调整为 28 x 28

记住这一点，Python OpenCV 3 的类似实现可以在文件 [ImageProcessor.py](https://github.com/Tony607/keras_mnist/blob/master/ImageProcessor.py) 中找到。为了结束网络摄像头演示，对于捕获的每一帧，我们将其传递给图像预处理函数，然后馈送给 NCS graph，后者像以前一样返回最终的预测概率。从那里，我们将最终的预测结果显示在显示器上的图像上。

![mnist_cam_demo](img/3fddd7241416f2c150b3bcf928964769.png)

## 结论和进一步阅读

现在您已经向 NCS 部署了一个 Keras 模型。请记住，由于 NCS 是以“视觉处理单元”的目的构建的，它支持卷积层和其他一些层，而 LSTM 和 GRU 等递归神经网络层可能无法在 NCS 上工作。在我们的演示中，我们告诉 **mvNCCompile** 获取最终的分类输出节点，同时可以使用中间层作为输出节点，在这种意义上，使用模型作为特征提取器，这类似于 NCS 的 faceNet 面部验证演示的工作方式。

一些有用的资源，

[TensorFlow 保存和恢复](https://www.tensorflow.org/guide/saved_model)

[用神经计算棒系列制作 DIY 安全摄像机](https://www.dlology.com/blog/build-a-diy-security-camera-with-neural-compute-stick-part-1/)

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-keras-model-on-movidius-neural-compute-stick/&text=How%20to%20run%20Keras%20model%20on%20Movidius%20neural%20compute%20stick) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-keras-model-on-movidius-neural-compute-stick/)

*   [←用神经计算棒制作 DIY 安全摄像机(第二部分)](/blog/build-a-diy-security-camera-with-neural-compute-stick-part-2/)
*   [用树莓派 DIY 物体检测涂鸦相机(第一部分)→](/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-1/)