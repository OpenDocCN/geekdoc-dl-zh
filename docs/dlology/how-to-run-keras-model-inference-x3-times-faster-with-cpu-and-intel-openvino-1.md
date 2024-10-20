# 如何利用 CPU 和英特尔 OpenVINO 将 Keras 模型推理运行速度提高 x3 倍

> 原文：<https://www.dlology.com/blog/how-to-run-keras-model-inference-x3-times-faster-with-cpu-and-intel-openvino-1/>

###### 发帖人:[程维](/blog/author/Chengwei/) 3 年 11 个月前

([评论](/blog/how-to-run-keras-model-inference-x3-times-faster-with-cpu-and-intel-openvino-1/#disqus_thread))

![openvino](img/c0224978e1e50be5678dd03508a3fb71.png)

在这个快速教程中，您将学习如何设置 OpenVINO，并在不增加任何硬件的情况下使您的 Keras 模型推断速度至少提高 3 倍。

虽然有多种选择可以加速你在边缘设备上的深度学习推理，举几个例子，

*   添加低端 Nvidia GPU，如 GT1030
    *   优点:易于集成，因为它也利用 Nvidia 的 CUDA 和 CuDNN 工具包来加速与您的开发环境相同的推理，不需要重大的模型转换。
    *   缺点:PCI-E 插槽必须存在于目标设备的主板上，以便与图形卡接口，这增加了边缘设备的额外成本和空间。
*   使用面向加速神经网络推理的 ASIC 芯片，如 [Movidius 神经计算棒](https://software.intel.com/en-us/movidius-ncs)、 [Lightspeeur 2801 神经加速器](https://www.gyrfalcontech.ai/solutions/2801s/)。
    *   优点:
        *   就像 u 盘一样，它们也工作在不同的主机上，无论是采用 Intel/AMD CPU 的台式电脑，还是采用 ARM Cortex-A 的 Raspberry Pi 单板电脑。
        *   神经网络计算被卸载到这些 u 盘上，使得主机的 CPU 只需担心更多的通用计算，如图像预处理。
        *   随着边缘设备上吞吐量需求的增加，扩展可以像插入更多 USB 棒一样简单。
        *   与 CPU/Nvidia GPU 相比，它们通常具有更高的性能功耗比。
    *   缺点:
        *   由于它们是 ASIC(专用集成电路),预计对某些 TensorFlow 层/操作的支持有限。
        *   它们还需要特殊的模型转换来创建特定 ASIC 可理解的指令。
*   嵌入式 SoC 带有 NPU(神经处理单元)，如 Rockchip RK3399Pro。
    *   NPU 类似于 ASIC 芯片，需要特殊的指令和模型转换。不同之处在于，它们与 CPU 位于同一个硅芯片中，这使得外形尺寸更小。

前面提到的所有加速选项都需要额外的成本。但是，如果一个边缘设备已经有了英特尔的 CPU，你还不如用英特尔的 OpenVINO toolkit 免费加速它的深度学习推理速度 x3 倍。

## OpenVINO 和设置简介

您可能想知道，如果没有额外的硬件，额外的加速从何而来？

首先，因为 OpenVINO 是英特尔的产品，所以它针对其处理器进行了优化。

OpenVINO 推理引擎可以推理具有不同输入精度支持的 CPU 或英特尔集成 GPU 模型。

CPU 支持 FP32 和 Int8，而其 GPU 支持 FP16 和 FP32。

CPU 插件利用面向深度神经网络(MKL-DNN)的英特尔数学内核库以及 OpenMP 来并行化计算。

下面是 OpenVINO 2019 R1.01 支持的插件和量化精度矩阵。

| **插件** | **FP32** | **FP16** | **I8** |
| CPU plugin | 支持和首选 | 不支持 | 支持 |
| GPU 插件 | 支持 | 支持和首选 | 不支持 |
| FPGA plugin | 支持 | 支持 | 不支持 |
| VPU 插件 | 不支持 | 支持 | 不支持 |
| GNA 插件 | 支持 | 不支持 | 不支持 |

您将在本教程的后面部分看到模型优化，在此过程中，会采取额外的步骤来使模型更加紧凑，以便进行推理。

*   群卷积的合并。
*   用 ReLU 或 eLU 融合卷积。
*   融合卷积+和或卷积+和+ ReLu。
*   移除电源层。

现在，让我们在你的机器上安装 OpenVINO，在[这一页](https://software.intel.com/en-us/openvino-toolkit/choose-download)上选择你的操作系统，按照说明下载并安装它。

**系统需求**

*   第六至第八代英特尔酷睿
*   英特尔至强 v5 家族
*   英特尔至强 v6 家族

**操作系统**

*   Ubuntu* 16.04.3 长期支持(LTS)，64 位
*   CentOS* 7.4，64 位
*   64 位 Windows* 10
*   macOS* 10.14

如果您已经安装了 Python 3.5+，则可以安全地忽略安装 Python 3.6+的通知。

安装完成后，查看 [Linux](https://docs.openvinotoolkit.org/2019_R1/_docs_install_guides_installing_openvino_linux.html) 、 [Window 10](https://docs.openvinotoolkit.org/2019_R1/_docs_install_guides_installing_openvino_windows.html) 或 [macOS](https://docs.openvinotoolkit.org/2019_R1/_docs_install_guides_installing_openvino_macos.html) 安装指南，完成安装。

## OpenVINO 中的 InceptionV3 模型推理

你可以从 [my GitHub](https://github.com/Tony607/keras_openvino) 下载本教程的完整源代码，它包括一个 all in one Jupyter 笔记本你通过转换一个用于 OpenVINO 的 Keras 模型，对所有三种环境——Keras、TensorFlow 和 open vino——进行预测以及基准推理速度。

运行`setupvars.bat`调用`jupyter notebook`来设置环境。

```py
C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat 
```

或者在 Linux 中添加下面一行到  `~/.bashrc`

```py
source /opt/intel/openvino/bin/setupvars.sh
```

以下是将 Keras 模型转换为 OpenVINO 模型并进行预测的工作流程概述。

1.  将 Keras 模型另存为单个 `.h5` 文件。
2.  加载`.h5`文件并将图形冻结为单个张量流 `.pb` 文件。
3.  运行 OpenVINO `mo_tf.py` 脚本将 `.pb` 文件转换为模型 XML 和 bin 文件。
4.  用 OpenVINO 推理引擎加载模型 XML 和 bin 文件，并进行预测。

### 将 Keras 模型保存为一个单独的`.h5`文件

对于本教程，我们将从 Keras 加载一个预训练的 ImageNet 分类 InceptionV3 模型，

### 将图形冻结到单个  张量流`.pb` 文件中

这一步将删除推理不需要的任何层和操作。

### OpenVINO 模型优化

下面的代码片段运行在 Jupyter 笔记本中，它根据您的操作系统(Windows 或 Linux)定位`mo_tf.py`脚本，您可以相应地更改`img_height`。`data_type`也可以设置为 FP16，以便在英特尔集成 GPU 上进行推理时获得额外的加速，但进动性能会有所下降。

运行脚本后，你会发现在 目录下生成了两个新文件`./model``frozen_model.xml`和 `frozen_model.bin` 。它们是基于训练好的网络拓扑、权重和偏差值的模型的优化中间表示(IR)。

### 使用 OpenVINO 推理引擎(IE)进行推理

如果环境设置正确， 路径 如同`C:\Intel\computer_vision_sdk\python\python3.5`或`~/intel/computer_vision_sdk/python/python3.5`将 存在于`PYTHONPATH`。这是在运行时加载 Python `openvino` 包所必需的。

以下代码片段使用 CPU 运行推理引擎，如果您之前选择使用 FP16 `data_type`，它也可以在英特尔 GPU 上运行。

## 速度基准

基准设置，

*   TensorFlow 版本:1.12.0
*   操作系统:Windows 10，64 位
*   CPU:英特尔酷睿 i7-7700HQ
*   计算平均结果的推断次数:20。

所有三种环境的基准测试结果——Keras、TensorFlow 和 OpenVINO 如下所示。

```py
Keras          average(sec):0.079, fps:12.5
TensorFlow     average(sec):0.069, fps:14.3
OpenVINO(CPU)  average(sec):0.024, fps:40.6
```

![benchmark](img/cca8c654474e7b4c3fa2b43382d2cd35.png)

结果可能因您正在试验的英特尔处理器而异，但与在 CPU 后端使用 TensorFlow / Keras 运行推理相比，预计会有显著的加速。

## 结论和进一步阅读

在本教程中，您学习了如何使用英特尔处理器和 OpenVINO toolkit 运行模型推理，其速度比股票 TensorFlow 快几倍。虽然 OpenVINO 不仅可以加速 CPU 上的推理，但本教程中介绍的相同工作流可以很容易地适应一个 Movidius neural compute 棒，只需做一些更改。

OpenVINO 文档可能对您有所帮助。

[安装英特尔 open vino toolkit for Windows * 10 分发版](https://software.intel.com/en-us/articles/OpenVINO-Install-Windows)

[安装英特尔发布的 OpenVINO toolkit for Linux *](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux)

[open vino-Advanced Topics-CPU Plugin](https://software.intel.com/en-us/articles/OpenVINO-InferEngine#inpage-nav-8-2-2)在这里你可以了解更多关于各种模型优化技术的知识。

#### 从 [my GitHub](https://github.com/Tony607/keras_openvino) 下载本教程的完整源代码。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-keras-model-inference-x3-times-faster-with-cpu-and-intel-openvino-1/&text=How%20to%20run%20Keras%20model%20inference%20x3%20times%20faster%20with%20CPU%20and%20Intel%20OpenVINO) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-keras-model-inference-x3-times-faster-with-cpu-and-intel-openvino-1/)

*   [←如何从 Keras 中的图像文件进行混音训练](/blog/how-to-do-mixup-training-from-image-files-in-keras/)
*   [如何免费训练一个简单的物体检测模型→](/blog/how-to-train-an-object-detection-model-easy-for-free/)