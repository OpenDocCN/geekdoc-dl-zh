# 如何在 Google Colab 上运行支持 GPU 和 CUDA 9.2 的 PyTorch

> 原文：<https://www.dlology.com/blog/how-to-run-pytorch-with-gpu-and-cuda-92-support-on-google-colab/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零三个月前

([评论](/blog/how-to-run-pytorch-with-gpu-and-cuda-92-support-on-google-colab/#disqus_thread))

![pytorch-colab](img/bcb28accdfd5a44b693a111ee9b9f7cc.png)

自从我写了第一篇关于在谷歌的 GPU 支持的 Jupyter 笔记本界面- Colab 上运行深度学习实验的教程以来，已经有一段时间了。从那以后，我的几个博客都是通过 GPU 加速在 Colab 上运行 Keras、TensorFlow 或 Caffe。

1.  “[如何免费快速运行视频上的对象检测和分割](https://www.dlology.com/blog/how-to-run-object-detection-and-segmentation-on-video-fast-for-free/)”——我在 Colab 上的第一个教程， [colab 笔记本直接链接](https://drive.google.com/file/d/11yXcMidH2rmnvy5GxFAr0M_0mABr1M_-/view?usp=sharing)。
2.  "[在 Google Colab 中运行 TensorBoard 的快速指南](https://www.dlology.com/blog/quick-guide-to-run-tensorboard-in-google-colab/)"，- [Colab 笔记本直达链接](https://drive.google.com/file/d/1afN2SALDooZIHbBGmWZMT6cZ8ccVElWk/view?usp=sharing)。
3.  在 Colab - [上运行caffe-cuda Colab 笔记本直连](https://drive.google.com/file/d/1jqBo2hpFY_xNeFHDf5l1h6q_VTcRTRlQ/view?usp=sharing)。

Colab 上缺少的一个框架是 PyTorch。最近，我正在检查一个需要在 Linux 上运行的视频到视频合成模型，此外，在我可以带着这个闪亮的模型兜风之前，还有数千兆字节的数据和预先训练的模型要下载。我在想，为什么不利用 Colab 惊人的下载速度和免费的 GPU 来尝试一下呢？

#### 享受本教程的 [Colab 笔记本链接](https://colab.research.google.com/drive/1ldg8DbTpe0M8PaioPBmwIwqs-DkH-ha9)。

让我们从在 Colab 上安装 CUDA 开始。

## 安装 CUDA 9.2

为什么不是其他 CUDA 版本？这里有三个原因。

1.  截至 2018 年 9 月 7 日，CUDA 9.2 是 Pytorch 在其网站[pytorch.org](https://pytorch.org/)上看到的官方支持的最高版本。
2.  有些人可能认为安装 CUDA 9.2 可能会与 TensorFlow 冲突，因为 TF 目前只支持 CUDA 9.0。放松，把 Colab 笔记本当成一个沙箱，即使你弄坏了它，也可以通过几个按钮轻松重置，让 TensorFlow 在安装 CUDA 9.2 后工作正常，因为我测试过。
3.  如果你安装了 CUDA 9.0 版本，你可能会在编译 Pytorch 的本地 CUDA 扩展时遇到问题。一些复杂的 Pytorch 项目包含[定制 c++ CUDA 扩展](https://pytorch.org/tutorials/advanced/cpp_extension.html#integrating-a-c-cuda-operation-with-pytorch)，用于定制层/操作，比它们的 Python 实现运行得更快。不利的一面是，您需要从单个平台的源代码中编译它们。在运行于 Ubuntu Linux 机器上的 Colab 案例中，使用 g++编译器来编译本地 CUDA 扩展。但是 CUDA 9.0 版在使用 g++编译器编译原生 CUDA 扩展时有一个错误，这就是为什么我们选择 CUDA 9.2 版来修复这个错误。

回到安装，[Nvidia 开发者网站](https://developer.nvidia.com/cuda-downloads)会询问你想要运行 CUDA 的 Ubuntu 版本。要找到答案，请在 Colab 笔记本中运行下面的单元格。

它会返回您想要的信息。

```py
VERSION="17.10 (Artful Aardvark)"
```

之后，您将能够浏览目标平台选择，使安装程序键入**“deb(local)”**，然后**右键**点击**“下载(1.2 GB)”**按钮复制链接地址。

回到 Colab 笔记本，粘贴链接后发出 **wget** 命令下载文件。在 Colab 上下载一个 1.2GB 的文件只需要大约 10 秒钟，这意味着没有咖啡休息时间-_-。

```py
!wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda-repo-ubuntu1710-9-2-local_9.2.148-1_amd64
```

运行以下单元来完成 CUDA 安装。

如果您在输出的末尾看到这些行，这意味着安装成功了。

```py
Setting up cuda (9.2.148-1) ... Processing triggers for libc-bin (2.26-0ubuntu2.1) ...
```

继续 Pytorch。

## 安装 PyTorch

非常简单，去[pytorch.org](pytorch.org)，这里有一个选择器，你可以选择如何安装 Pytorch，在我们的例子中，

*   OS: **Linux**
*   包管理器: **pip**
*   Python: **3.6** ，你可以通过在一个 shell 中运行 `python --version` 来验证。
*   CUDA: **9.2**

它将让您运行下面这一行，之后，安装就完成了！

```py
pip3 install torch torchvision
```

## 在演示时运行 vid 2

出于好奇，Pytorch 在 Colab 上启用 GPU 后表现如何，让我们尝试一下最近发布的 [视频到视频合成演示](https://github.com/NVIDIA/vid2vid)，这是一个 Pytorch 实现我们的高分辨率真实感视频到视频翻译方法。那个视频演示把姿势变成了一个看起来很诱人的舞蹈身体。

![pose](img/230e5fab4eafbdca95e9271fc9f4f0ee.png)

此外，演示还依赖于定制的 CUDA 扩展，提供了测试已安装的 CUDA 工具包的机会。

下面的单元完成了从获取代码到使用预先训练好的模型运行演示的所有工作。

生成的帧会转到目录**results/label 2 city _ 1024 _ G1/test _ latest/images**中，您可以通过调用下面的单元格来显示一个。

本教程到此结束。

## 结论和进一步的思考

这篇短文向你展示了如何让 GPU 和 CUDA 后端 **Pytorch** 在 Colab 上快速自由地运行。不幸的是， **vid2vid** 的作者还没有得到[可测试的侧脸，pose-dance demo 也还没有贴出](https://github.com/NVIDIA/vid2vid/issues/24#issuecomment-417463746)，我在焦急地等待着。到目前为止，它只是作为一个演示来验证我们在 Colab 上安装 Pytorch。请随时在社交媒体上与我联系，我会让你了解我未来的项目和其他实用的深度学习应用。

*   标签:
*   [教程](/blog/tag/tutorial/)，
*   [深度学习](/blog/tag/deep-learning/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-pytorch-with-gpu-and-cuda-92-support-on-google-colab/&text=How%20to%20run%20PyTorch%20with%20GPU%20and%20CUDA%209.2%20support%20on%20Google%20Colab) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-pytorch-with-gpu-and-cuda-92-support-on-google-colab/)

*   [↓《如果我是女孩》StarGAN 的魔镜](/blog/if-i-were-a-girl-magic-mirror-by-stargan/)
*   [如何使用 Keras 时间序列生成器生成时间序列数据→](/blog/how-to-use-keras-timeseriesgenerator-for-time-series-data/)