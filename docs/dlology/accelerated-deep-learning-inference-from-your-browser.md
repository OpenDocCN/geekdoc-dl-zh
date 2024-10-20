# 从你的浏览器加速深度学习推理

> 原文：<https://www.dlology.com/blog/accelerated-deep-learning-inference-from-your-browser/>

###### 发帖人:[程维](/blog/author/Chengwei/)两年零六个月前

([评论](/blog/accelerated-deep-learning-inference-from-your-browser/#disqus_thread))

![logo](img/c11e523cc436d5be112068247d23bd43.png)

数据科学家和 ML 工程师现在可以从他们的浏览器中使用 FPGA 加速器的能力来加速他们的深度学习应用程序。

FPGAs 是适应性强的硬件平台，可以为机器学习、视频处理、定量金融等应用提供出色的性能、低延迟和更低的运营支出。然而，对于没有 FPGA 相关知识的用户来说，轻松高效的部署是一个挑战。

应用加速领域的先驱 InAccel 让您可以从浏览器中获得 FPGA 加速的强大功能。数据科学家和 ML 工程师现在可以轻松部署和管理 FPGAs，加速计算密集型工作负载，并通过零代码更改降低总拥有成本。

InAccel 提供了一个 [FPGA 资源管理器](https://inaccel.com/)，允许对 FPGA 进行即时部署、扩展和资源管理，使 FPGA 在机器学习、数据处理、数据分析等应用中的应用比以往任何时候都更容易。用户可以从 Python、Spark、Jupyter 笔记本甚至终端部署他们的应用。

通过 JupyterHub 集成，用户现在可以享受 JupyterHub 提供的所有好处，例如轻松访问计算环境以即时执行 Jupyter 笔记本。同时，用户现在可以享受 FPGA 的优势，例如更低的延迟、更短的执行时间和更高的性能，而无需事先了解 FPGA。InAccel 的框架允许使用 Xilinx 的 [Vitis 开源优化库](https://www.xilinx.com/products/design-tools/vitis/vitis-libraries.html#libraries)或第三方^(IP 核(用于深度学习、机器学习、数据分析、基因组学、压缩、加密和计算机视觉应用。))

InAccel 的 FPGA orchestrator 提供的加速机器学习平台既可以在内部使用，也可以在云上使用。这样，用户可以享受 Jupyter 笔记本的简单性，同时体验应用程序的显著加速。

用户可以通过以下链接免费测试 InAccel 集群上的可用库:

[https://inaccel.com/accelerated-data-science/](https://inaccel.com/accelerated-data-science/)

### 加速推理 ResNet50 上的一个用例

任何用户现在都可以从浏览器中享受 FPGA 加速器的加速。在 DL 示例中，我们展示了用户如何从相同的 Keras python 笔记本中享受更快的 ResNet50 推理，而没有任何代码更改。

用户可以在[https://labs.inaccel.com:8000](https://labs.inaccel.com:8000)使用谷歌账户登录 InAccel 门户网站

他们可以在 Resnet50 上找到现成的 Keras 示例。

![notebook1](img/f8935d96eb9af79636a7f10ca9fd25f1.png)

用户可以看到 python 代码与在任何 CPU/处理器上运行的代码完全一样。然而，在本例中，用户可以在 ResNet50 上体验高达 2，000 FPS 的推断，而没有任何代码更改。

用户可以使用可用的数据集(22，000 幅图像)测试加速的 Keras ResNet50 推理示例，也可以下载自己的数据集。

他们还可以使用如下所示的验证码来确认结果是否正确。

![notebook2](img/d85fd5b9e57f238c0800ca124383c9c6.png)

注意:该平台可用于演示目的。多个用户可以使用 2 个 Alveo 卡访问可用集群，这可能会影响平台的性能。如果您有兴趣使用多个 FPGA 卡部署自己的数据中心，或者专门在云上运行您的应用程序，请致电 [【电子邮件保护】](/cdn-cgi/l/email-protection#244d4a424b644d4a45474741480a474b49) 联系我们。

![arch](img/27b1c3647f7a3a1b55ab31149ab8e5f0.png)

图一。使用 Jupyter 从浏览器加速 ML、vision、finance 和数据分析

你也可以在这里查看网络视频:【https://www.youtube.com/watch?v=42bsjdXVmFg】T2

[![screenshot](img/54723d71fb2278cd96f7b717b199ba37.png)](https://www.youtube.com/watch?v=42bsjdXVmFg)

*关于 InAccel 公司*

InAccel 通过使用自适应硬件加速器，帮助企业加快应用速度。它为无缝利用 Spark 和 Jupyter 等高级框架中的硬件加速器提供了一个独特的框架。InAccel 还为机器学习、压缩和数据分析等应用开发高性能加速器。欲了解更多信息，请访问[https://inaccel.com](https://inaccel.com)

可用代码:

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/accelerated-deep-learning-inference-from-your-browser/&text=Accelerated%20Deep%20Learning%20inference%20from%20your%20browser) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/accelerated-deep-learning-inference-from-your-browser/)

*   [←如何在 Jetson Nano 上以 20+ FPS 运行 SSD Mobilenet V2 物体检测](/blog/how-to-run-ssd-mobilenet-v2-object-detection-on-jetson-nano-at-20-fps/)