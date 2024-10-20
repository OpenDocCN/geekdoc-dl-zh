# 如何在捷成纳米上运行 Keras 模型

> 原文：<https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年零八个月前

([评论](/blog/how-to-run-keras-model-on-jetson-nano/#disqus_thread))

![keras-jetson-nano](img/3eb16b391671f75e565b8c40f01dcd8b.png)

[杰特森纳米开发者套件](https://developer.nvidia.com/embedded/buy/jetson-nano-devkit)在 2019 年 GTC 上宣布售价 99 美元，为边缘计算硬件领域带来了一个新的竞争对手，以及更昂贵的前辈杰特森 TX1 和 TX2。Jetson Nano 的到来使该公司比其他平价产品更具竞争优势，仅举几个例子， [Movidius 神经计算棒](https://software.intel.com/en-us/movidius-ncs)，[运行 OpenVINO 的英特尔显卡](https://www.dlology.com/blog/how-to-run-keras-model-inference-x3-times-faster-with-cpu-and-intel-openvino-1/)和[谷歌 edge TPU](https://cloud.google.com/edge-tpu/) 。

在这篇文章中，我将向你展示如何在 Jetson Nano 上运行 Keras 模型。

下面是如何实现这一点的分解。

1.  冻结 Keras 模型到 TensorFlow 图，然后用 TensorRT 创建推理图。
2.  在 Jetson Nano 上加载tensort 推理图并进行预测。

我们将在开发机器上完成第一步，因为它的计算和资源密集程度远远超出了 Jetson Nano 的处理能力。

让我们开始吧。

## 设置 Jetson Nano

按照[官方入门指南](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)来刷新最新的 SD 卡镜像、设置和引导。

需要记住的一点是，Jetson Nano 不像最新的 Raspberry Pi 那样配备 WIFI 无线电，所以建议准备一个像[这个](https://www.amazon.com/dp/B003MTTJOY//ref=cm_sw_su_dp)这样的 USB WIFI 转换器，除非你打算用硬连线的方式连接它的以太网插孔。

### 在 Jetson Nano 上安装 TensorFlow

Nvidia 开发者论坛上有一个关于官方支持 Jetson Nano 上 TensorFlow 的帖子，这里有一个如何安装它的快速运行。

启动到 Jetson Nano 的终端或 SSH，然后运行这些命令。

如果您遇到下面的错误，

```py
Cannot compile 'Python.h'. Perhaps you need to install python-dev|python-devel
```

试着跑

```py
sudo apt install python3.6-dev
```

Python3 将来可能会更新到更高的版本。你总是可以先用 **python3 - version** 检查你的版本，并相应地修改之前的命令。

安装 Jupyter Notebook 也很有帮助，这样您就可以从开发机器远程连接到它。

```py
pip3 install jupyter
```

另外，请注意 Python OpenCV 版本 3.3.1 已经安装，这减轻了交叉编译的痛苦。您可以通过从 Python3 命令行界面导入 **cv2** 库来验证这一点。

## 步骤 1:冻结 Keras 模型并转换成 TensorRT 模型

在你的开发机器上运行这个步骤，Tensorflow nightly builds 默认包含 TF-TRT，或者在 Colab 笔记本的免费 GPU 上运行。

首先让加载一个 Keras 模型。对于本教程，我们使用 Keras 附带的预培训 MobileNetV2，必要时可以随意用您的自定义模型替换它。

一旦将 Keras 模型保存为单个`.h5`文件，就可以将其冻结为一个张量流图用于推理。

记下输出中打印的输入和输出节点名称。我们在转换  `TensorRT` 图和预测图时会需要它们的推断。

对于 Keras MobileNetV2 车型，分别是，`['input_1'] ['Logits/Softmax']`。

通常，这个冻结的图表是您用于部署的。然而，它没有优化运行在 Jetson Nano 上的速度和资源效率。这就是 TensorRT 发挥作用的地方，它将模型从 FP32 量化到 FP16，有效地减少了内存消耗。它还将层和张量融合在一起，进一步优化了 GPU 内存和带宽的使用。所有这些都很少或没有明显降低准确性。

这可以在一次呼叫中完成，

结果也是一个张量流图，但经过优化，可以在装有 TensorRT 的 Jetson Nano 上运行。让我们将它保存为一个单独的`.pb`文件。

将 tensort graph`.pb`文件从 colab 或您的本地机器下载到您的 Jetson Nano 中。您可以使用 scp/ sftp 远程复制文件。对于 Windows，你可以使用 [WinSCP](https://winscp.net/eng/index.php) ，对于 Linux/Mac 你可以从命令行尝试 scp/sftp。

## 第二步:加载 张量图并进行预测

在你的 Jetson Nano 上，用命令`jupyter notebook --ip=0.0.0.0`启动一个 Jupyter 笔记本，在那里你已经将下载的图形文件保存到了`./model/trt_graph.pb`。下面的代码将加载 TensorRT 图，并为推理做好准备。

对于您选择的 Keras 型号而非 MobileNetV2，输出和输入名称可能会有所不同。

现在，我们可以用大象图片进行预测，看看模型是否正确。

## 基准测试结果

让我们运行几次推理，看看它能跑多快。

它得到了 27.18 FPS，可以认为是实时预测。此外，Keras 模型可以在 Colab 的 Tesla K80 GPU 上以 60 FPS 的速度推理，比 Jetson Nano 快一倍，但那是数据中心卡。

## 结论和进一步阅读

在本教程中，我们介绍了如何使用 TensorRT 转换、优化 Keras 图像分类模型，并在 Jetson Nano 开发套件上运行推理。现在，尝试另一个 Keras ImageNet 模型或您的定制模型，将 USB 网络摄像头/ Raspberry Pi 摄像头连接到它，并进行实时预测演示，请务必在下面的评论中与我们分享您的结果。

未来，我们将研究其他应用程序的运行模型，例如对象检测。如果你对其他负担得起的边缘计算选项感兴趣，请查看我的[上一篇文章](https://www.dlology.com/blog/how-to-run-keras-model-inference-x3-times-faster-with-cpu-and-intel-openvino-1/)，如何使用 CPU 和英特尔 OpenVINO 将 Keras 模型推理运行速度提高 x3 倍也适用于 Linux/Windows 和 Raspberry Pi 上的 Movidius neural compute stick。

#### *本教程的源代码可在[我的 GitHub repo](https://github.com/Tony607/tf_jetson_nano) 上获得。你也可以跳过第一步模型转换，直接从 GitHub repo 版本下载 [trt_graph.pb](https://github.com/Tony607/tf_jetson_nano/releases/download/V0.1/trt_graph.pb) 文件。*

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/&text=How%20to%20run%20Keras%20model%20on%20Jetson%20Nano) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/)

*   [←如何使用贝叶斯优化对 Keras 模型进行超参数搜索](/blog/how-to-do-hyperparameter-search-with-baysian-optimization-for-keras-model/)
*   [如何在 Jetson Nano 上运行 TensorFlow 对象检测模型→](/blog/how-to-run-tensorflow-object-detection-model-on-jetson-nano/)