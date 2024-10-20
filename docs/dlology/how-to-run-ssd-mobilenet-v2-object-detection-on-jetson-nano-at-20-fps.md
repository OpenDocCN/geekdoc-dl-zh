# 如何在 Jetson Nano 上以 20+ FPS 运行 SSD Mobilenet V2 物体检测

> 原文：<https://www.dlology.com/blog/how-to-run-ssd-mobilenet-v2-object-detection-on-jetson-nano-at-20-fps/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年前

([评论](/blog/how-to-run-ssd-mobilenet-v2-object-detection-on-jetson-nano-at-20-fps/#disqus_thread))

![jetson_nano_ssd_v2](img/fad1fbe9cd8e21a280457c71fbf114b5.png)

泰勒:博士

首先，确保你已经在你的 Jetson Nano 开发 SD 卡上闪存了最新的 JetPack 4.3。

然后你会看到类似这样的结果。

![ssd_v2_benchmark](img/eee7cf8592d734bb53c85f4afa14b2e0.png)

现在进行一个稍微长一点的描述。

大约 8 个月前，我发布了[如何在 Jetson Nano](https://www.dlology.com/blog/how-to-run-tensorflow-object-detection-model-on-jetson-nano/) 上运行 TensorFlow 对象检测模型，意识到仅仅在 Jetson Nano 上以大约 10FPS 的速度运行 SSD MobileNet V1 对于一些应用来说可能是不够的。此外，这种方法消耗了太多内存，没有空间让其他内存密集型应用程序并行运行。

这一次，更大的 SSD MobileNet V2 对象检测模型以 20+FPS 的速度运行。速度提高了一倍，内存消耗也减少到只有 Jetson Nano 上总 4GB 内存的 32.5%(即大约 1.3GB)。有足够的内存来运行其他的花哨的东西。您还注意到 CPU 使用率也很低，仅比四核高 10%左右。

![ssd_v2_benchmark_top](img/f2514c00885894022f1ee114abbbf980.png)

据我所知，有许多技巧有助于提高性能。

*   什么带来了 [JetPack 4.3](https://developer.nvidia.com/embedded/jetpack) ，TensorRT 版本 6.0.1 对比之前的 TensorRT 版本 5。
*   TensorFlow 对象检测图是在硬件上优化和转换的，我指的是我现在使用的 Jetson Nano 开发套件。这是因为 TensorRT 通过使用可用的 GPU 来优化图形，因此优化的图形在不同的 GPU 上可能表现不佳。
*   该模型现在被转换为一种更特定于硬件的格式，TensorRT 引擎文件。但缺点是它不够灵活，受到运行它的硬件和软件堆栈的限制。稍后会详细介绍。
*   一些节省内存和提高速度的技巧。

## 它是如何工作的？

您刚才运行的命令行启动了一个 docker 容器。如果您是 Docker 的新手，可以把它想象成一个超级 Anaconda 或 Python 虚拟环境，它包含了重现 mine 结果所需的一切。如果您仔细查看一下我的 GitHub repo 上的 [Dockerfile](https://github.com/Tony607/jetson_nano_trt_tf_ssd/blob/master/Dockerfile) ，它描述了容器映像是如何构建的，您可以看到所有的依赖项是如何设置的，包括所有的 apt 和 Python 包。

docker 映像是基于最新的 JetPack 4.3 - L4T R32.3.1 基础映像构建的。要使用 TensorRT 引擎文件进行推理，需要两个重要的 Python 包，TensorRT 和 Pycuda。在 Jetson Nano 上从源代码构建 Pycuda Python 包可能需要一些时间，所以我决定将预构建包打包到一个 wheel 文件中，并使 Docker 构建过程更加流畅。请注意，用 JetPack 4.3 预构建的 Pycuda 与旧版本的 JetPack 和 vers visa 不兼容。在 TensorRT python 包中，它来自目录`/usr/lib/python3.6/dist-packages/tensorrt/`中的 Jetson Nano。我所做的只是将该目录压缩到一个`tensorrt.tar.gz`文件中。你猜怎么着，推理时不需要 TensorFlow GPU Python 包。考虑一下，通过跳过导入 TensorFlow GPU Python 包，我们可以节省多少内存。

你可以在[我的 GitHub 库](https://github.com/Tony607/jetson_nano_trt_tf_ssd/tree/master/packages/jetpack4.3)找到用 JetPack 4.3 构建的名为**TRT _ SSD _ mobilenet _ v2 _ coco . bin**的 TensorRT 引擎文件。有时，您可能还会看到 TensorRT 引擎文件，其扩展名为`*.engine`,就像 JetBot 系统映像中那样。如果想自己转换文件，可以看看 JK Jung 的 [build_engine.py](https://github.com/jkjung-avt/tensorrt_demos/blob/master/ssd/build_engine.py) 脚本。

现在，对于 TensorRT 引擎文件方法的限制。它不能跨不同版本 JetPack 工作。原因在于引擎文件是如何通过搜索 CUDA 内核构建的，以获得可用的最快实现，因此有必要使用相同的 GPU 和软件堆栈(CUDA、CuDnn、TensorRT 等)。)用于构建优化引擎将在其上运行的那种结构。TensorRT engine file 就像一件专门为设置量身定制的衣服，但当安装在正确的人员/开发板上时，它的性能令人惊叹。

随着速度的提高和更低的内存占用带来的另一个限制是精度的损失，以下面的预测结果为例，一只狗被错误地预测为一只熊。这可能是从 FP32 到 FP16 的模型权重量化或其他优化折衷的结果。

![result](img/a00a05a20796d9988af15c1dabb21b72.png)

## 节省内存和提高速度的一些技巧

关闭 GUI 并在命令行模式下运行。如果您已经在 GUI 桌面环境中，只需按“Ctrl+Alt+F2”进入非 GUI 模式，从那里登录您的帐户并键入“`service gdm stop`”，这将停止 Ubuntu GUI 环境并为您节省大约 8%的 4GB 内存。

通过在命令行中键入“jetson_clocks”来强制 CPU 和 GPU 以最大时钟速度运行。如果你有一个 PWM 风扇连接到主板上，并被风扇的噪音所困扰，你可以通过创建一个新的设置文件来降低它。

## 结论和进一步阅读

本指南向您展示了重现我在 Jetson Nano 上以 20+ FPS 运行 SSD Mobilenet V2 物体检测的结果的最简单方法。解释它是如何工作的，以及在将它应用到实际应用程序之前需要注意的限制。

别忘了在[我的 GitHub](https://github.com/Tony607/jetson_nano_trt_tf_ssd) 上获取这篇文章的源代码。

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-ssd-mobilenet-v2-object-detection-on-jetson-nano-at-20-fps/&text=How%20to%20run%20SSD%20Mobilenet%20V2%20object%20detection%20on%20Jetson%20Nano%20at%2020%2B%20FPS) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-ssd-mobilenet-v2-object-detection-on-jetson-nano-at-20-fps/)

*   [←端到端深度学习自动缺陷检查](/blog/automatic-defect-inspection-with-end-to-end-deep-learning/)
*   [从你的浏览器加速深度学习推理→](/blog/accelerated-deep-learning-inference-from-your-browser/)