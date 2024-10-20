# 用树莓派 DIY 物体检测涂鸦相机(第二部分)

> 原文：<https://www.dlology.com/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-2/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零四个月前

([评论](/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-2/#disqus_thread))

![camera_part2](img/25cbc65e2e0a070c359cd70732cd3020.png)

[上一篇](https://www.dlology.com/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-1/)走过了这个项目的前几个部分，即语音激活和物体检测。这一次，我们将总结其余的软件和硬件组件，以建立一个完整的对象检测涂鸦相机。

*   涂鸦检测到的物体
*   用微型热敏收据打印机打印绘图
*   在你的 Pi 上添加一个快门按钮和一个 LED 指示灯

## 绘制检测到的对象

在前面的对象检测步骤之后，每个检测的边界框、类和分数都将可用。对象的大小和中心位置可以进一步从边界框值中获得。创建几乎不重复的高分辨率绘图的诀窍是应用谷歌快速绘制数据集，其中每个绘图都被记录为画布上的笔画坐标。在这里看看网上真人做的[103031 张猫图](https://quickdraw.withgoogle.com/data/cat)。

![quick_draw_cats](img/2d9b4a419e9ddb9b9ebf91a0805297bf.png)

下面的代码片段向您展示了如何将第一个“猫”画的笔画转换成图像。

`dataset.get_drawing()`法位于**。/drawing _ dataset/drawing dataset . py**文件通过名称和索引获取绘图。在这种情况下，我们得到的名称“猫”的索引为 0，对应于第一只猫画的笔画。在每一个笔画里面，都有多个点，它们被连接在一起，由 **Gizeh** Python 库在画布上绘制成一条折线。

例如，这只猫的第二个笔画从右向左画了 3 个点。

![quick_draw_stroke](img/37fb6e8f048b491e6e2a5d03a8f688b8.png)

由于SSDliteMobileNet V2对象检测模型只能检测有限类别的对象，而快速绘制数据集上有 345 个类别的 5000 万个图形，我决定预处理原始数据集，并保留每个可识别类别的前 100 个图形。每个类的绘图都被保存为更容易阅读的 Python 3 pickle 格式的路径。/data/quick_draw_pickles/。

## 用微型热敏收据打印机打印

[热敏打印机](https://www.adafruit.com/product/597)也被称为收据打印机。它们是你去自动取款机或杂货店时得到的东西。它的工作原理是加热热敏纸上的点，热敏纸上涂有受热时会变色的材料，因此不需要墨水。该打印机是 Raspberry Pi 等小型嵌入式系统甚至微控制器的理想选择，因为它通过 TTL 串行端口与应用处理器通信，并在打印期间由 5-9 VDC @ 1.5Amp 供电。只要你的系统有一个额外的串口，就不需要额外的打印机驱动，这意味着你可以跨操作系统使用 Python 打印机库，无论是 Ubuntu PC，Windows PC 还是 Raspberry Pi。

与最初的 [Adafruit 库](https://github.com/adafruit/Python-Thermal-Printer)相比，我的 Python 打印机库在打印大型位图图像的速度和稳定性方面都有了一些改进。

图像将首先被转换成二进制位图，然后逐行打印。一行越密，打印机需要的加热时间就越长，否则就会出现空行。动态计算每行的打印时间可实现最佳速度和稳定性。

![binary_bitmap](img/8b173c2859f950f29380e810845e6db2.png)

以这个 8×8 二进制位图为例，第 1 行有 4 个点要打印，第 5 行只有 2 个点，打印第 1 行比第 5 行需要的时间稍长。由于热敏纸上可以打印的最大宽度是 384 像素，我们将首先将绘图图像翻转 90 度，以逐行垂直打印。

由于通过串行端口发送完整的 384×512 位图需要很大的带宽，因此还需要将其波特率从默认的 19200 或 9600 调整到 115200，以便在热敏打印机上实现更快的位图打印。我在我的 GitHub 上附加了一个[工具](https://github.com/Tony607/voice-camera/releases/download/V1.1/AClassTool-printer-setting-tool.zip)来完成这个。

## 添加快门按钮和指示灯 LED

没有任何响应或反馈的语音激活会破坏用户体验，这就是为什么我们添加了单个 LED 以不同的模式闪烁，以显示是否检测到语音命令。一个额外的按钮还提供了一个额外的选项来触发相机捕捉，对象检测，绘图和打印工作流程。

与 Raspberry Pi 系统一起提供的 gpio zero Python 库提供了一个与 IO 引脚接口的快速解决方案。它允许你以各种方式脉冲或闪烁发光二极管，并使用一个背景线程处理时间，而无需进一步的交互。一个按钮只需两行就可以触发一个函数调用。

所有的 GPIO 用法都在**中定义。/raspberry _ io/raspberry . py**文件，以防您想要更改 LED 闪烁和脉冲的方式。

## 不用电线，由电池供电

虽然有可能在 3 安培时获得强大的 5V 电源，但它可能太笨重，不适合放在我们的相机盒中。使用紧凑型[两节 7.4 v 脂电池](https://www.amazon.com/Kreema-1500mAh-Battery-Rechargeable-Charger/dp/B07D5WMZM3/ref=sr_1_29?s=toys-and-games&ie=UTF8&qid=1534832672&sr=1-29&keywords=7.4V+1500mAh+Lipo+Battery)同时为 Raspberry Pi 和热敏打印机供电似乎是一种可行的方式。你可以直接用 7.4 V 电池给热敏打印机供电，而树莓 Pi 采用 5V 电源，这就是 DC-DC 降压模块发挥作用的地方。它是一个开关电源从输入到输出逐步降低电压。我正在使用一个 [LM2596 DC 到 DC 降压转换器](https://www.amazon.com/UPZHIJI-LM2596-Converter-3-0-40V-1-5-35V/dp/B07BLRQQK7/ref=sr_1_2?ie=UTF8&qid=1534665501&sr=8-2&keywords=DC+DC) ，你可以在亚马逊上看到。

模块上有一个小电位器，用来调节输出电压。将电池连接到输入电源和接地引脚，转动电位计上的小旋钮，同时用万用表监控输出电压，直到输出电压达到 5V。

## 结论和进一步阅读

这是最终的完整图表，向您展示了各部分是如何连接的。

![complete_diagram](img/88076dadbc58359d816cbb2b44b18459.png)

如果你有关于建立你的物体探测和涂鸦相机的问题，请随意发表评论。

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-2/&text=DIY%20Object%20Detection%20Doodle%20camera%20with%20Raspberry%20Pi%20%28part%202%29) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-2/)

*   [←带树莓派的 DIY 物体检测涂鸦相机(第 1 部分)](/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-1/)
*   [《如果我是女孩》StarGAN 的魔镜→](/blog/if-i-were-a-girl-magic-mirror-by-stargan/)