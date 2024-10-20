# 用神经计算棒构建 DIY 安全摄像机(第 2 部分)

> 原文：<https://www.dlology.com/blog/build-a-diy-security-camera-with-neural-compute-stick-part-2/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零五个月前

([评论](/blog/build-a-diy-security-camera-with-neural-compute-stick-part-2/#disqus_thread))

![security-cams](img/d83924c445b22af51d7f69eb1070761a.png)

上一篇文章向你展示了如何构建一个由 Raspberry Pi 和 Movidius neural compute stick 驱动的安全摄像头，可以实时检测人并只保存有用的图像帧。这一次，我将向您展示如何理解足够的源代码来添加一个 Arduino 伺服转台，使相机自动跟随它检测到的人。

## 当前代码是如何工作的？

无论您是来自 Arduino 的 C 编程背景，还是来自 Python 的舒适编码背景，掌握代码如何工作都不会太难。考虑到我们的目的是修改代码并添加一些伺服逻辑。在深入研究代码并对其进行定制之前，没有对深度学习知识的预先要求，因为深度学习模型已经打包到计算图中，神经计算棒理解如何获取输入图像并生成对象检测输出的特殊格式的文件。前面调用`make run_cam`命令，后面调用`mvNCCompile`命令，计算图就编译好了。

进入 `video_objects_camera.py` ，启动时发现并连接 NCS。然后将编译后的计算图分配到它的存储器中。Python OpenCV 库打开了系统中第一个可用的摄像头，因为摄像头索引为 0。至此，所有设置都已完成。在主循环中，对于每个捕捉到的网络摄像头图像，都会按照特定的分辨率进行处理，并按照模型的预期对其值进行标准化。

接下来，所有繁重的目标检测都在两行中完成，图像进入 NCS，检测结果出来。在最后一节中，您将看到以非同步方式实现这一点的替代 API 调用。

此处应用的 SSD MobileNet 对象检测模型可以检测 20 种不同的对象，如人、瓶子、汽车和狗。由于我们只对像安全摄像机一样检测人感兴趣，我们告诉检测器忽略其他类，这可以在`object_classifications_mask`变量中完成。

输出是一个浮点数数组，格式如下。

从那里开始，代码根据您设置的最低分数和分类掩码对结果进行过滤，然后在图像上创建边界框的覆盖。

这基本上就是代码所做的。添加计算在图像中检测到多少人的功能并在这个数字改变时保存它是很简单的，因此我们知道一个人何时进入或离开我们的视野。

## 添加伺服系统

假设您的安全摄像头是电池供电的，支持 WIFI，安装在一个最佳位置，可以监控 300 度的宽视角，并且可以同时覆盖您的车库、院子和街道。一些安全摄像机有可控的转台来手动转动摄像机以面对不同的角度，但当对象移出其视野时，它们不会自己转动。我们现在正在建造的这个安全摄像头即使在没有互联网的情况下也能实时检测到人，它足够智能，可以知道人在哪里，以及如何转向面对目标，但是我们还缺少一个东西，移动的部分，一个炮塔！

### Arduino 草图与伺服系统对话

像 [SG90](https://learn.adafruit.com/mini-pan-tilt-kit-assembly) 这样的廉价业余伺服电机应该可以完成这项工作，因为它不需要太多的扭矩来转动摄像头。伺服系统是如何工作的？脉冲的长度决定了伺服电机的位置。伺服系统预计大约每 20 毫秒接收一次脉冲。如果该脉冲为高 1 毫秒，那么伺服角度将为零，如果为 1.5 毫秒，那么它将位于其中心位置，如果为 2 毫秒，则它将位于 180 度。

![servos](img/3d193371b32a4e2d876b532d4e4bcf14.png)

使用 Arduino 控制器来控制伺服角度有三个好处，即使 Raspberry Pi 已经带有 PWM 引脚。

1.  使用 Arduino 使主要的 Python 代码可以跨其他平台进行比较，例如，你可以连接到 Arduino 并在 Ubuntu PC 上运行相同的演示。
2.  我们可以每隔 20 毫秒对 Arduino 上的伺服系统进行精确的速度控制，这对于 raspberry 来说是一个开销。
3.  它为 raspberry Pi 的数字引脚提供了额外的一层保护。

伺服有三个引脚，通常带有棕色，红色和橙色，代表接地，电源和信号。将接地(棕色)连接到 Arduino 的 GND，将电源(红色)连接到 Arduino 的 5V 或外部 5V 电池，将信号(橙色)连接到 Arduino 的引脚 4，如我的草图所示。请记住，如果您有一个高扭矩伺服系统，它可能需要更高的电压和电流，因此更安全的做法是由外部电源供电，如 12V 锂电池组，其电压通过 DC-DC 转换器(如 LM2596 模块)降低到所需的水平。

我选择了一个 Arduino Pro 微型板，它基本上是 Arduino Leonardo 的最小版本，小巧但功能多样，而其他 Arduino 兼容板应该也可以工作。Arduino 的代码可以在我的 [GitHub](https://github.com/Tony607/CamGimbal) 上免费获得。它接受串行端口命令，以相对角度或绝对角度转动伺服机构。

它在后面的技巧是通过控制在特定间隔内转动多少度来使伺服的运动更平滑。请随意试一试，说明包含在自述文件中。

按照此图连接所有部件。

![schematic](img/7969ac7cdbb887652cb0804a83f4d6f6.png)

### 用蟒蛇皮转动炮塔

拥有 Arduino 协处理器让 Raspberry Pi 的生活变得更加轻松。它需要做的只是发送命令告诉 Arduino 将伺服系统向左或向右转动一些角度。左转或右转的决定来自于计算被检测的人或多人的中心相对于视图中心的偏移，如果偏移大于某个阈值，将发出调谐的命令。这就是逻辑的本质，超级简单，但是行得通。

Python 代码通过枚举可用串行端口的描述来自动定位 Arduino 串行端口。确保您的 Raspberry Pi 上安装了 pySerial 库。

```py
pip3 install pyserial
```

通过在 Pi 的控制台中键入以下命令来运行演示，并观察神奇的事情发生。

```py
make run_gimbal
```

![surprise](img/53f9afd0de610f25f7c6008f81610ddd.png)

## 结论和进一步阅读

由于您正在试验相机演示，它可能只有每秒 6 帧(fps)左右。然而，通过利用 NCSDK2 附带的异步 FIFO API，这个数字可以提高到大约 8.69 fps。这超出了这个简单教程的范围，虽然它在我的 GitHub repo-[video _ objects _ threaded](https://github.com/Tony607/video_objects_threaded)上可用，但它是基于文件夹`ncappzoo/apps/video_objects_threaded`中的官方示例。希望以类似于您在`video_objects`中运行之前演示的方式运行它。如果你还没有看过[演示](https://youtu.be/am2RBqRJYgk)视频，请点击这里观看。如果你被卡住了，不要犹豫发表评论。

### 有用的链接

【Arduino 和正版产品入门

ncsdkpython API 2 . x FIFO

### GitHub 仓库

ncsdk 2-[https://github . com/move ius/ncsdk/tree/ncsdk 2](https://github.com/movidius/ncsdk/tree/ncsdk2)

NCSDK2 分支 ncappzoo-[https://github.com/movidius/ncappzoo.git](https://github.com/movidius/ncappzoo.git)

我的视频对象检测报告-[https://github.com/Tony607/video_objects](https://github.com/Tony607/video_objects)

线程和 FIFO 的视频对象检测在 Pi-【https://github.com/Tony607/video_objects_threaded】的上实现 8.69 fps

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/build-a-diy-security-camera-with-neural-compute-stick-part-2/&text=Build%20a%20DIY%20security%20camera%20with%20neural%20compute%20stick%20%28part%202%29) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/build-a-diy-security-camera-with-neural-compute-stick-part-2/)

*   [←用神经计算棒制作 DIY 安全摄像机(第一部分)](/blog/build-a-diy-security-camera-with-neural-compute-stick-part-1/)
*   [如何在 Movidius 神经计算棒上运行 Keras 模型→](/blog/how-to-run-keras-model-on-movidius-neural-compute-stick/)