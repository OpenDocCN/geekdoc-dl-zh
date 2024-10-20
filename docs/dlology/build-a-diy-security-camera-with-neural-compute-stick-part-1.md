# 用神经计算棒制作一个 DIY 安全摄像机(第 1 部分)

> 原文：<https://www.dlology.com/blog/build-a-diy-security-camera-with-neural-compute-stick-part-1/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零五个月前

([评论](/blog/build-a-diy-security-camera-with-neural-compute-stick-part-1/#disqus_thread))

![fisher](img/6032689bcf16589aec9cf3a5a15b4298.png)

1933 年，一个养鸡人兼业余摄影师决定找到偷他鸡蛋的罪犯。自问世以来，如今安全摄像头无处不在，大多数所谓的“智能摄像头”都是通过将视频流回显示器或服务器来工作的，以便有人或一些软件可以分析视频帧，并有望从中找到一些有用的信息。它们消耗大量的网络带宽和电力来传输视频，尽管我们只需要 10 帧图像就可以知道谁在偷鸡蛋。当网络不稳定，图像无法分析，“智能”变成“哑”时，他们还面临着服务中断的困境。

边缘计算是一种网络模型，支持在摄像头所在的网络边缘进行数据处理，无需将视频发送到中央服务器进行处理。在边缘处理图像数据减少了系统等待时间、视频传输所消耗的功率和带宽成本，并且由于将传输较少的可能被黑客攻击的信息而提高了私密性。简单的概念，但为什么它还没有流行起来？简单的回答，硬件和软件还没有准备好。众所周知，图像处理一直渴望处理能力和高级算法，以便从中提取有用的信息。随着深度学习算法的最新进展，以及价格友好的推理硬件的出现，为相机上更高级的边缘计算打开了大门。

![edge-computing](img/600edee9e67e043cbc2e7ad9346f72e1.png)

自己做这么酷的相机怎么样？本教程的最终目标是向您展示如何构建这样一个安全摄像机，它使用高级对象检测算法在本地处理镜头，并实时从数小时的视频帧中过滤出重要图像。要进行构建，您需要以下基本工具和硬件。

1.  [树莓 Pi 3 型](https://www.arrow.com/en/products/raspberrypi3b/raspberry-pi-foundation?utm_source=google&utm_campaign=g-shp-us-20offdevboard&utm_medium=cpc&utm_term=PRODUCT+GROUP&gclid=CjwKCAjwtIXbBRBhEiwAWV-5nqWpyVLl5aZw9hBcAvT_x0CF9_NubtnxJl40QSJnc9Ds-E1DLjNZvxoCkJ8QAvD_BwE&gclsrc=aw.ds&dclid=CJzG94POy9wCFQNzYAodUaYBig)，5V 电源供电的单板电脑。
2.  英特尔的神经计算棒。
3.  [NCSDK2](https://github.com/movidius/ncsdk) :这款英特尔 Movidius Neural Compute 软件开发者套件 v2 来自 GitHub，免费下载。
4.  一个 USB 摄像头，任何品牌都应该工作，而一个广角镜头脱颖而出，因为我们正在建立一个安全摄像头。

可选使相机运行在电池上。

*   12V 锂电池模块和 DC-DC 模块将电压降低到 5V。
*   或者带有 USB 端口的电池组提供 5V/2A。

可选建立一个额外的炮塔来转动相机，覆盖更广泛的视野范围。

1.  业余爱好伺服电机
2.  Arduino Pro Micro 或其他 Arduino 可比主板
3.  一些焊丝

建造项目房屋的可选材料，纸板、实木板等。

在这篇文章中，你将学习如何为 Movidius NCS 在 Raspberry Pi 上安装必要的软件，并在网络摄像头帧上实时进行对象检测。

## 在 Raspberry Pi 上安装 NCSDK2

树莓 Pi 的[基本安装和配置](https://movidius.github.io/ncsdk/install.html)已经在 Movidius 文档中列出，注意树莓 Pi 3 必须运行 NCSDK2 的最新 Raspbian Stretch，建议使用 SD 卡大小的 16G +。

安装 [NCSDK2](https://github.com/movidius/ncsdk) 后，安装 NCAPI v2 的 [ncappzoo](https://github.com/movidius/ncappzoo/tree/ncsdk2) ，其中包含了 Movidius 神经计算棒的例子。

```py
git clone -b ncsdk2 https://github.com/movidius/ncappzoo.git
```

然后将CD 成 `ncappzoo/apps` ，删除或者重命名文件夹 **video_objects** 到别的地方，在那里克隆我的 **video_objects** repo。

```py
git clone https://github.com/Tony607/video_objects
```

## 以最简单的方式安装 OpenCV3

运行这个演示需要安装 opencv3。虽然官方的标准方法是从源代码构建，但我尝试了几次都没有成功，这导致了我的替代解决方案，在 raspberry Pi 上安装预构建的 OpenCV3 库只需几分钟，而从源代码构建需要几个小时。只需在终端中运行以下四行代码。

```py
sudo pip3 install opencv-python==3.3.0.10
sudo apt-get update
sudo apt-get install libqtgui4
sudo apt-get install python-opencv
```

你刚刚为树莓派安装了 [python wheels 的 opencv -python 包版本 3.3.10！下面的两个 apt-get 安装确保没有缺失的依赖项。](https://www.piwheels.hostedpi.com/)

要检查 python3 中的安装，可以像这样运行。期待看到 OpenCV3 版本字符串“3.3.0”。

```py
[email protected]:~/workspace/ncappzoo/apps/video_objects $ python3
Python 3.5.3 (default, Jan 19 2017, 14:11:04) 
[GCC 6.3.0 20170124] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>> cv2.__version__
'3.3.0'
>>> 
```

所以不要再抱怨“我的代码正在编译”。

![compiling](img/e6330fde879040a26f45b399bcd13e29.png)

## 运行对象检测演示

将您的 NCS 和网络摄像头插入您的 Raspberry Pi 的 USB 端口，连接 HDMI 监视器或使用 [VNC](https://www.raspberrypi.org/documentation/remote-access/vnc/) 访问您的 Pi 桌面。让我们开始演示吧。在终端中，将光盘放入 **video_objects** 目录中你刚刚从我的 GitHub repo 中克隆出来然后运行。

```py
make run_cam
```

第一次运行时，make 脚本将下载 SSD mobileNet 深度学习模型定义文件和权重，因此预计会有一些延迟。之后，会弹出一个新窗口，显示实时摄像机画面，以及探测到的物体的覆盖边框。软件只会将两帧之间识别人数变化的关键图像帧保存到 文件夹 `images` 中。在下一篇文章中会有更多的介绍。

![video_objects_camera](img/504f173a7cd4f5224d1f3a24a3da23de.png)

## 结论和进一步阅读

作为安全摄像机项目的第一篇文章，它展示了如何构建一个最小的“更智能”的摄像机，该摄像机可以实时检测人，并只保存有用的关键帧。上一节中介绍的在 Raspberry Pi 上安装 NCSDK2 是这个项目中最困难和最耗时的部分，希望我的一些技巧能帮助您加快这一过程。在下一篇帖子中，我将解释你刚才运行的代码，并向你展示如何添加一个 Arduino 炮塔来转动相机，相机跟随周围的人，如我上传的这个 **[视频](https://youtu.be/am2RBqRJYgk)** 所示。

最后一个提示，如果你计划在 Ubuntu 16.04 PC 上运行 NCS 和 NCSDK2，我的建议是使用物理机而不是虚拟机，我遇到了 NCS 设备未被检测到的问题，还没有解决方案。

### 有用的资源

[Movidius 神经计算棒简介](https://movidius.github.io/ncsdk/index.html)

[Movidius 官方博客](https://movidius.github.io/blog/)

[树莓 PI 文档](https://www.raspberrypi.org/documentation/)

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/build-a-diy-security-camera-with-neural-compute-stick-part-1/&text=Build%20a%20DIY%20security%20camera%20with%20neural%20compute%20stick%20%28part%201%29) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/build-a-diy-security-camera-with-neural-compute-stick-part-1/)

*   [←如何在浏览器上训练神经网络](/blog/how-to-train-neural-network-on-browser/)
*   [用神经计算棒制作 DIY 安全摄像机(第二部分)→](/blog/build-a-diy-security-camera-with-neural-compute-stick-part-2/)