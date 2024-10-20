# DIY 对象检测涂鸦相机与树莓派(第一部分)

> 原文：<https://www.dlology.com/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-1/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零四个月前

([评论](/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-1/#disqus_thread))

![camera1](img/257242f30794ecc1095b997c0ccbdf5c.png)

#### 让我们创建一个可以创建和打印一些艺术作品的相机。查看[演示视频](https://www.youtube.com/watch?v=uGgog7ER9-Q)看看结果。

看完这个教程，把下面几块拼起来，你就知道怎么做这样一个相机了。

*   用豪猪的声音激活来触发图像捕捉。
*   在 Raspberry Pi 中捕捉网络摄像头图像。
*   使用 TensorFlow 对象检测 API 进行对象检测
*   涂鸦检测到的物体
*   用微型热敏收据打印机打印绘图
*   在你的 Pi 上添加一个快门按钮和一个 LED 指示灯

在开始之前，确保你已经准备好以下材料。

1.  树莓 Pi 型号 3 或以上，安装 [Raspbian 9(拉伸)](https://www.raspberrypi.org/downloads/raspbian/)。其他模型未经测试，可能需要对源代码进行一些调整。
2.  带有内置麦克风的 USB 网络摄像头，如 Logitech C920。
3.  一台迷你热敏收据打印机，就像[Adafruit.com](https://www.adafruit.com/product/597)上的那台。
4.  一个 USB 转 TTL 适配器，用于将迷你热敏收据打印机连接到 Pi，就像亚马逊上基于 CP2102 芯片组的适配器一样。

可选的东西，你可能没有，但可以使你的相机看起来光滑。

1.  7.4V 2c lipo 电池和额定 3A 或以上输出电流的 DC-DC 转换器模块将电压降低至 5V。像亚马逊上的 [LM2596 DC 到 DC 降压转换器](https://www.amazon.com/UPZHIJI-LM2596-Converter-3-0-40V-1-5-35V/dp/B07BLRQQK7/ref=sr_1_2?ie=UTF8&qid=1534665501&sr=8-2&keywords=DC+DC)。
2.  打开/关闭相机的电源开关
3.  单个 LED 模块显示当前状态。
4.  手动触发图像捕捉快门的单按钮模块。

所有的源代码和数据资源都打包到一个文件中，从我的 GitHub 版本下载，然后用下面的命令解压到您的 Pi 中。

```py
wget https://github.com/Tony607/voice-camera/archive/V1.1.tar.gz
tar -xzf v1.1.tar.gz
```

现在我们已经准备好把这些碎片拼在一起了。

## 豪猪语音激活

我们正在谈论的这只豪猪不是长着尖刺的可爱啮齿动物，而是一个由深度学习驱动的设备上唤醒单词检测引擎。如果你读过我以前的帖子- [如何用 Keras](https://www.dlology.com/blog/how-to-do-real-time-trigger-word-detection-with-keras/) 做实时触发词检测，你就会知道我在说什么。但是这一次它是如此的轻量级，甚至可以在树莓 Pi 上运行，具有非常好的精确度。Porcupine 是一个跨平台软件，可以在其他操作系统上运行，比如 Android、iOS、watchOS、Linux、Mac 和 Windows。不需要特殊的训练，通过将文本字符串传递到优化器来指定一个您希望引擎检测的关键字，并生成一个关键字文件。引擎可以采用多个激活字来触发程序中的不同动作。在我们的例子中，我们将使用语音命令“蓝莓”来触发对象检测涂鸦风格，并使用“菠萝”来触发另一种边缘检测涂鸦风格。

要安装 Porcupine 的依赖项，请在 Pi 的终端中运行以下命令。

如果一切顺利，你会看到这样的东西，它在等你说出关键词“菠萝”。

![pineapple](img/5388a96721af309926b562d870feb4fd.png)

在[中有一个您可以使用的其他预优化关键字的列表。/porcupine/resources/keyword _ files](https://github.com/Tony607/voice-camera/tree/master/porcupine/resources/keyword_files)文件夹。

一个边注，豪猪库本身是相当准确的，而结果会受到麦克风相当大的影响。一些只有一个内置麦克风的网络摄像头只能在有限的范围内清晰地捕捉语音，而由两个麦克风组成的罗技 C920 网络摄像头上的麦克风阵列可以消除环境中的噪音，即使在延长的距离内也能清晰地记录您的声音。

## TF 使用实时网络摄像头进行物体检测

一旦该应用程序被语音激活，该软件将让网络摄像头捕捉图像，并试图找到里面的物体。

从网络摄像头捕捉图像需要在您的 Pi 上安装 OpenCV3 库。人们过去常常花费数小时编译源代码并将其安装在 Pi 上，现在在终端上运行这三行代码就可以避免这种情况。

```py
pip3 install opencv-python==3.3.0.10
sudo apt-get install libqtgui4
sudo apt-get install python-opencv
```

捕获的照片进入 TensorFlow 对象检测 API，模型返回四条信息，

1.  图像上检测到的物体的边界框，
2.  每个箱子的检测置信度得分
3.  每个对象的类别标签
4.  检测的总数。

我们用于对象检测的模型是从 TensorFlow [检测模型 zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) 下载的 SSD lite MobileNet V2。我们使用它是因为它很小，即使在 Raspberry Pi 上也能快速运行。我一直在 Pi 上以每秒最多 1 帧的速度实时测试运行该模型，如果你想在你的 Pi 上将帧速率提高到 8 FPS 或更高，这对这个应用程序来说是多余的，请随时查看我的另一个[教程](https://www.dlology.com/blog/build-a-diy-security-camera-with-neural-compute-stick-part-1/)关于如何用 Movidius 神经计算棒做到这一点。

要将最新的预建 TensorFlow 1.9.0 和对象检测 API 依赖项安装到您的 PI，请在终端中运行这些命令。

```py
sudo apt install libatlas-base-dev protobuf-compiler python-pil python-lxml python-tk
pip3 install tensorflow
```

这里需要注意的是，没有必要下载 [TensorFlow 模型库](https://github.com/tensorflow/models)，因为您通常会使用对象检测 API，这会占用您 PI 的 SD 卡上大约 1GB 的空间，并且会浪费宝贵的下载时间。完成本教程所需的所有 Python 源代码都已经打包在你之前从我的 repo 下载的 27MB GitHub 版本中。

到 验证 你的安装和相机正在工作，运行这个命令行。

```py
python3 ./image_processor/examples/test_object_detection.py
```

模型的加载可能需要 30 秒左右，如果一切正常，您将会看到类似这样的内容。

![pi_objects](img/f697054d5bb7a33597e28a70fd6014db.png)

## 结论和进一步阅读

到目前为止，你已经了解了语音激活和物体检测在项目中的应用，在下一篇文章中，我将向你展示其余部分的组合。

这个项目在很大程度上受到了[danmacnish/cartonify](https://github.com/danmacnish/cartoonify)的影响，而你可以看到我的版本，增加了语音激活，紧凑的卡通绘图数据集，对象检测 API 和优化的 Python 热敏打印机库都在一个 27 MB 的发布文件中，相比之下，原作者的 ~5GB 卡通数据集，~100MB TensorFlow 模型和应用程序源代码。

#### 继续上 **[第二部分](https://www.dlology.com/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-2/)** 的教程。

### 有用的资源

[为您的树莓派安装操作系统映像](https://www.raspberrypi.org/documentation/installation/installing-images/README.md)

使用[](https://www.raspberrypi.org/documentation/remote-access/vnc/)访问您 Pi 的桌面

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-1/&text=DIY%20Object%20Detection%20Doodle%20camera%20with%20Raspberry%20Pi%20%28part%201%29) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-1/)

*   [←如何在 Movidius 神经计算棒上运行 Keras 模型](/blog/how-to-run-keras-model-on-movidius-neural-compute-stick/)
*   [用树莓派 DIY 物体检测涂鸦相机(第二部分)→](/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-2/)