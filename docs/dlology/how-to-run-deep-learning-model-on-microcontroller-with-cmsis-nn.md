# 如何用 CMSIS-NN 在微控制器上运行深度学习模型(上)

> 原文：<https://www.dlology.com/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零六个月前

([评论](/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn/#disqus_thread))

![nn-mcu](img/45822b5c60f7ad4af2c024a34d69e301.png)

*TL；DR 您将学习如何在 ARM 微控制器上运行 CIFAR10 图像分类模型，如 STM32F4 Discovery board 或类似产品上的微控制器。*

## 为什么要在微控制器上运行深度学习模型？

如果你以前玩过 Arduino，很容易有这样的印象:它们是计算和内存资源有限的小芯片，但在从各种传感器收集数据或控制机器人手上的伺服系统时却非常出色。

许多微控制器要么运行在 FreeRTOS 等实时操作系统上，要么运行在没有操作系统的裸机上。这两种方式都使它们非常稳定且响应迅速，尤其是在任务关键的情况下。

然而，随着传感器收集的数据越来越多，声音和图像这两种最常见的数据类型需要大量的计算资源来处理，以生成有用的结果。这项任务通常是通过要求微控制器将数据上传到网络连接的服务器来完成的，服务器将处理后的结果发送回边缘，然后微控制器将进行特定的行为，如问候和开灯。

你可能已经注意到这个计划的一些缺点。

*   敏感数据会传到云端、照片和录音中。
*   出售这些数据的公司可能会收取服务费，甚至出售你的私人数据。
*   如果没有与服务器的网络连接，它将无法工作。
*   数据在设备和服务器之间来回传输会带来延迟。
*   在电路设计上需要网络和无线硬件组件，这增加了成本。
*   发送无用的数据可能会浪费带宽。

如果一切都独立于微控制器之内，不仅节省带宽、功耗和成本，还具有低延迟、令人难以置信的可靠性和隐私性，那岂不是很好？

![use_cases1](img/bb1f2cebe686826351b78f59c23f7380.png)

## CMSIS-NN 概述

CMSIS-NN 是一组针对 ARM Cortex-M 内核微控制器优化的神经网络功能，支持将神经网络和机器学习推入物联网应用的终端节点。

它实现了流行的神经网络层类型，如卷积、深度可分离卷积、全连接、轮询和激活。利用它的效用函数，还可以构造更复杂的神经网络模块，如 LSTM 和格鲁

对于用 TensorFlow、Caffe 等流行框架训练的模型。权重和偏差将首先量化为 8 位或 16 位整数，然后部署到微控制器进行推理。

基于 CMSIS-NN 内核的神经网络推理据称在运行时间/吞吐量方面实现了 4.6 倍的提升，与基线实施相比实现了 4.9 倍的提升。最佳性能是通过利用 CPU 的 SIMD 指令特性来提高 Cortex-M4 和 Cortex- M7 内核微控制器的并行性实现的，尽管 Cortex-M0 和 Cortex-M3 的参考实现也可以在没有 DSP 指令的情况下实现。

## 在微控制器上运行模型

在本节中，我们将在一个[STM 32 f 4 Discovery board](https://www.st.com/en/evaluation-tools/stm32f4discovery.html)或类似的 Keil MDK-ARM 上运行 CIRAR10 图像分类模型。

在继续之前，您需要，

*   安装了基尔·MDK-ARM 的 Windows 电脑。您可以在 [my GitHub repo](https://github.com/Tony607/arm_nn_examples) 中找到获取和安装软件的说明。
*   本教程选择的 Cortex-M4 或 Cortex-M7 内核微控制器板最好是 [STM32F4 发现板](https://www.st.com/en/evaluation-tools/stm32f4discovery.html)。

第一次上电板

*   做 **不做** 连接板卡到 PC！转到**C:\ Keil _ V5 \ ARM \ STLink \ USB driver**，双击**STLink _ win USB _ install . bat**安装板载 USB ST-Link/V2 的驱动程序。
*   用迷你 USB 线将 USB 电源 USB ST-Link/V2 端口连接到电脑。Windows 识别 ST-Link/V2 设备并自动安装驱动程序。

板载 STM32F407VGT6 微控制器采用 32 位 ARM Cortex -M4，带 FPU 内核、1 兆字节闪存和 192 千字节 RAM，最大功耗上限为 465mW。

尽管我的 GitHub 中的[项目](https://github.com/Tony607/arm_nn_examples/tree/master/cifar10)已经配置好并准备好在板上运行，但是如果你想知道它是如何配置的或者你想在不同的目标上运行，这还是很有帮助的。

该项目基于官方的 CMSIS-NN CIFAR10 示例，因此请从 GitHub 下载整个 [CMSIS_5 repo](https://github.com/ARM-software/CMSIS_5) 。

您可以在以下位置访问示例项目

```py
.\CMSIS\NN\Examples\ARM\arm_nn_examples\cifar10
```

### 添加新目标

用基尔·MDK-ARM 打开`arm_nnexamples_cifar10.uvprojx`项目。该项目最初被配置为仅在模拟器上运行，我们将从添加一个新目标开始。该项目可以配置为在不同的微控制器/板上运行，基尔·MDK-ARM 正在通过“目标”组织它们。

右击当前目标，然后从菜单中点击“**管理项目项**按钮。

![1_new_target](img/a26fda1fa7d541852eb272dedbec6bf7.png)

创建一个新目标，并将其命名为“STM32F407DISCO ”,以帮助您记住其用途。突出显示您的新目标，点击“**设为当前目标**”，然后点击“**确定**”。

![2_new_target](img/72734da43ac5f4aabedc2a8a81b5d143.png)

### 配置目标选项

打开目标选项，转到“**设备**选项卡，选择目标微控制器。如果您无法搜索“STM32F407”，则有必要从 pack installer 中手动获取它，或者通过打开一个配置了 STM32F4 DISCO 板的现有项目，然后 IDE 将提示安装它。

![3_target_options](img/d7bc773038393cf576e5f3e8ab9d87ce.png)

![4_target_options](img/2e2bfce15c580636be53ef6c83b0e6f1.png)

![5_pack_installer](img/1e09abaa3681cc6748505ad937c10e0e.png)

![6_pack_installer](img/9fca50369ca8b8786d53661398e164ae.png)

转到“**目标**选项卡，将外部晶振频率更改为 8MHz，并将片内存储区更改为与板上的相匹配。

![7_target](img/2430cf2fee78e24c72a92a0a5ea7c983.png)

在“C/C++”选项卡中，添加“HSE_VALUE=8000000”作为预定义符号，告诉编译器外部晶振频率为 8MHz。这与在 C/C++源代码中定义下面的代码行是一样的，只是预定义的符号允许您在不修改源代码的情况下为不同的目标配置编译项目。

或者，关闭编译器优化以改善调试体验。更高级别的编译器优化一方面通过使软件消耗更少的资源来改进代码，但另一方面减少了调试信息并改变了代码的结构，这使得代码更难调试。

![](img/5557904f530a462d54aff1f7bd0121bd.png)

在 **Debug** 选项卡中，选择 STM32F4 DISCO 板上的“ST-Link Debugger”，点击按钮进行配置。

![9_debug](img/ea6d2c1ed21b3e997436db47391cd364.png)

如果您的主板插上电源，ST-LINK/V2 调试器适配器将出现在新窗口中，并检查端口“SW”是否被选中。

![10_debug](img/9eb09578db20aed6b56bcebea47ade37.png)

在【Trace】选项卡中，输入正确的 CPU **核心时钟** 速度，如您的项目中所指定。勾选 **追踪启用** 框。 Trace 允许您通过 SWO(单线输出)查看 **printf** 消息，其中是 Cortex-M3/M4/M7 上可用的单引脚异步串行通信通道，由主调试器探针支持。该功能类似于 Arduino 的" **Serial.printf** "功能，用于打印调试信息，但不需要 UART 端口。  

![11_trace](img/a3f6085ac6c87f6996327b7ed1c2b31c.png)

在 **Flash 下载**选项卡中添加“STM32F4xx Flash”编程算法，这样您就可以将二进制文件下载到其 Flash 存储器中。

![12_flash_download](img/2e334e9ea47e3956c97a19b525b899f1.png)

现在确认更改并关闭目标窗口的**选项。**

### 配置运行在 168MHz

在微控制器的启动文件中，还有一个步骤来配置微控制器以 168MHz 运行。

![13_startup](img/20ed4fd294ee8f64f864ad1b87d0961e.png)

简单的方法是用你的启动文件替换 GitHub 中我的启动文件，同时有几点值得一提，以了解微控制器系统时钟一般是如何工作的。

为了获得最终的 168MHz 系统时钟，我们需要将“PLL_M”参数设置为 8。

《美少女的谎言》缩写为 PLL，等等，我的意思是锁相环是微控制器中的一个时钟产生引擎，用来产生远高于内部或外部晶振频率的时钟速度。如果你玩过 Arduino 板，如 Leonardo，你就已经见过 PLL，尽管你的 Arduino 代码运行在 16MHz 系统时钟，但其 USB2.0 总线从其片内 PLL 提升了 48MHz。设置好 PLL 参数后，这里是 STM32F4 的整体时钟配置图。它显示了 168Mhz 系统时钟是如何从最初的 8Mhz 高语音外部(HSE)时钟获得的。

![14_pll](img/71f14a851c850837c0882bd1dd2cefc7.png)

### 构建和调试

现在我们已经准备好了，将开发板连接到您的 PC，只需构建和调试应用程序。

![15_build](img/af926af67a5c7f5973fe3c1a634cc30c.png)

![16_debug](img/205bd340850cb2a2fc9c97191b8fc2d7.png)

启用 printf 消息的跟踪视图并运行代码，您将看到 CIFAR10 模型的输出。

以一幅 32×32 像素的彩色图像作为输入，然后由模型将其分类为 10 个输出类别之一。

由于该值是 softmax 层的输出，每个数字表示 10 个图像类别之一的概率。在下面的例子中，标签 5 对应于具有最高数字的“狗”标签，这意味着模型在输入图像中找到了狗。

![17_debug_viewer](img/04a6ef6e2e1c30721fe4523461cd6563.png)

在下一节中，我将向您展示如何用您的自定义图像来填充模型。

### 创建新的输入图像

**中的**IMG _ 数据**定义了输入的图像数据。图像数组存储为 HWC 格式或高-宽-通道，这是一个 32x32 RGB 彩色图像，具有 32 x 32 x 3 = 3072 个值。**

我们可以有一个更高分辨率的图像作为输入吗？是的，但是必须首先调整图像的大小并进行裁剪，这可以通过下面的 python 代码片段来实现。

![18_img_data](img/76a87addf252c6860298b76dfbaa18c5.png)

该函数将返回一个包含 3072 个数字的 numpy 数组，然后该数组将被裁剪为 int8 数字，并以正确的格式写入一个头文件。

## 总结和进一步阅读

想要深入源代码并了解一切是如何工作的吗？这将是我的下一篇博文。

同时，这里有一些我认为有用的资源来学习 ARM Cortex-M 微控制器、STM32、CMSIS-NN 和 Keil-MDK 等。

[ARM Cortex-M WiKi](https://en.wikipedia.org/wiki/ARM_Cortex-M)

[STM 32 f 4-发现快速入门指南](https://keilpack.azureedge.net/content/Keil.STM32F4xx_DFP.2.13.0/MDK/Boards/ST/STM32F4-Discovery/Documentation/32F4-DISCOVERY_QSG.pdf)

[STM 32 f 4 发现板示例代码](https://www.st.com/content/st_com/en/products/embedded-software/mcus-embedded-software/stm32-embedded-software/stm32-standard-peripheral-library-expansion/stsw-stm32068.html)

[CMSIS NN 软件库文档](https://www.keil.com/pack/doc/CMSIS/NN/html/index.html)

[Arm 的项目 Trillium - Processors 机器学习](https://www.arm.com/products/processors/machine-learning)

别忘了从我的 GitHub 页面查看源代码。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn/&text=How%20to%20run%20deep%20learning%20model%20on%20microcontroller%20with%20CMSIS-NN%20%28Part%201%29) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn/)

*   [←利用深度方向可分离卷积的简单语音关键词检测](/blog/simple-speech-keyword-detecting-with-depthwise-separable-convolutions/)
*   [如何用 CMSIS-NN 在微控制器上运行深度学习模型(第二部分)→](/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn-part-2/)