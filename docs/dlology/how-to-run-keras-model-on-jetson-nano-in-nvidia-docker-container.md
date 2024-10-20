# 如何在 Nvidia Docker 容器中的 Jetson Nano 上运行 Keras 模型

> 原文：<https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano-in-nvidia-docker-container/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年零四个月前

([评论](/blog/how-to-run-keras-model-on-jetson-nano-in-nvidia-docker-container/#disqus_thread))

![docker_nano](img/9fe14dc00b32cdefe05e6ac840c01e18.png)

不久前，我写了“[如何在 Jetson Nano](https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/) 上运行 Keras 模型”，其中模型在主机操作系统上运行。在本教程中，我将向您展示如何重新开始，并让模型在 Nvidia docker 容器内的 Jetson Nano 上运行。

你可能想知道为什么要为 Jetson Nano 上的 docker 费心？我想出了几个理由。

1.与自己安装依赖项/库相比，用 docker 容器来重现结果要容易得多。因为你从 docker Hub 下载的 Docker 映像已经预装了所有的依赖项，这可以节省你从源代码构建的大量时间。

2.它不太可能搞乱 Jetson Nano 主机操作系统，因为您的代码和依赖项与它是隔离的。即使你遇到麻烦，解决问题也只是重新启动一个新的容器。

3.您可以通过创建一个新的 docker 文件，以一种更加可控的方式，基于预安装了 TensorFlow 的我的基本映像构建您的应用程序。

4.您可以用一台更强大的计算机(如基于 X86 的服务器)交叉编译 Docker 映像，节省宝贵的时间。

5.最后，你猜对了，在 Docker 容器中运行代码几乎和在有 GPU 加速的主机操作系统上运行一样快。

希望你相信，这里有一个如何做到这一点的简要概述。

*   在 Jetson Nano 上安装新的 JetPack 4.2.1。
*   在 X86 机器上交叉编译 Docker 构建设置。
*   用 TensorFlow GPU 搭建一个 Jetson Nano docker。
*   构建覆盖 Docker 图像(可选)。
*   在 Docker 容器中运行冻结的 Keras TensorRT 模型。

## 在 Jetson Nano 上安装新的 JetPack 4.2.1

从 Nvidia 下载 [JetPack 4.2.1 SD 卡镜像](https://developer.nvidia.com/embedded/jetpack)。从 zip 中解压 **sd-blob-b01.img** 文件。用 [Rufus](https://rufus.ie/) 把它闪存到 10 级 32GB 最小 SD 卡。我的 SD 卡是 SanDisk class10 U1 64GB 型号。

![rufus](img/e0df5a22eb54349c9e507f6e72c6543d.png)

你可以试试另一个闪光器，比如蚀刻机，但是我用蚀刻机闪的 SD 卡不能在 Jetson Nano 上启动。我也试着用 SDK 管理器安装 JetPack，但是遇到了一个关于“系统配置向导”的问题。还有我在 Nvidia 开发者论坛上开的[线程](https://devtalk.nvidia.com/default/topic/1058116/jetpack-4-2-1-fails-to-boot-on-nano-failed-to-start-load-kernel-modules/?offset=1)，他们的技术支持很有反应。

插入 SD 卡，插上 HDMI 显示器电缆、USB 键盘和鼠标，然后打开主板电源。按照系统配置向导完成系统配置。

## 在 X86 机器上交叉编译 Docker 构建设置

即使 Nvidia Docker 运行时预装在操作系统上，允许您在硬件上构建 Docker 容器。然而，考虑到更大的处理能力和网络速度，在基于 X86 的机器上交叉编译 Docker 可以节省大量的构建时间。所以为交叉编译环境设置一次是非常值得的。docker 容器将在服务器上构建，推送到 Docker 注册中心，如 Docker Hub，然后从 Jetson Nano 中取出。

在您的 X86 机器上，可能是您的笔记本电脑或 Linux 服务器，首先按照官方说明安装 [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) 。

然后从命令行安装 **qemu** ，qemu 在构建 Docker 容器时会在你的 X86 机器上模拟 Jetson Nano CPU 架构(也就是 aarch64)。

```py
sudo apt-get install -y qemu binfmt-support qemu-user-static
wget http://archive.ubuntu.com/ubuntu/pool/main/b/binfmt-support/binfmt-support_2.1.8-2_amd64.deb
sudo apt install ./binfmt-support_2.1.8-2_amd64.deb
rm binfmt-support_2.1.8-2_amd64.deb
```

最后安装波德曼。我们将使用它来构建容器，而不是默认的 docker 容器命令行界面。

```py
sudo apt update
sudo apt -y install software-properties-common
sudo add-apt-repository -y ppa:projectatomic/ppa
sudo apt update
sudo apt -y install podman
```

## 使用 TensorFlow GPU 构建 Jetson Nano Docker

我们基于官方的**nvcr.io/nvidia/l4t-base:r32.2**映像构建我们的 TensorFlow GPU Docker 映像。

这里是 **Dockerfile** 的内容。

```py
FROM nvcr.io/nvidia/l4t-base:r32.2
WORKDIR /
RUN apt update && apt install -y --fix-missing make g++
RUN apt update && apt install -y --fix-missing python3-pip libhdf5-serial-dev hdf5-tools
RUN apt update && apt install -y python3-h5py
RUN pip3 install --pre --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu
RUN pip3 install -U numpy
CMD [ "bash" ]
```

然后，您可以像这样提取基本映像，构建容器映像并将其推送到 Docker Hub。

```py
podman pull nvcr.io/nvidia/l4t-base:r32.2
podman build -v /usr/bin/qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t docker.io/zcw607/jetson:0.1.0 . -f ./Dockerfile
podman push docker.io/zcw607/jetson:0.1.0
```

根据需要将 **zcw607** 更改为您自己的 Docker Hub 帐户名称，您可能需要在推送至注册表之前先执行`docker login docker.io`操作。

## 构建覆盖 Docker 图像(可选)

通过构建一个覆盖 Docker 映像，您可以基于以前的 Docker 映像添加您的代码依赖项/库。

例如，您想要安装 Python pillow 库并设置一些其他东西，您可以创建一个像这样的新 other 文件。

```py
FROM zcw607/jetson:0.1.0
WORKDIR /home
ENV TZ=Asia/Hong_Kong
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
 apt update && apt install -y python3-pil
CMD [ "bash" ]
```

然后运行这两行代码来构建和推送新的容器。

```py
podman build -v /usr/bin/qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t docker.io/zcw607/jetson:r1.0.1 . -f ./Dockerfile
podman push docker.io/zcw607/jetson:r1.0.1
```

现在您的两个 Docker 容器驻留在 Docker Hub 中，让我们在 Jetson Nano 上同步。

## 在 Docker 容器中运行 TensorRT 模型

在 Jetson Nano 命令行中，像这样从 Docker Hub 中拉出 Docker 容器。

```py
docker pull docker.io/zcw607/jetson:r1.0.1
```

然后用下面的命令启动容器。

```py
docker run --runtime nvidia --network host -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix zcw607/jetson:r1.0.1
```

检查是否安装了 TensorFlow GPU，然后在命令中键入“python3 ”,

如果一切正常，应该可以打印出来

![tf_gpu](img/bb6faf81c24f9a64dac0d8640f2ef09e.png)

要运行 TensorRT 模型推理基准，请使用 [my Python 脚本](https://raw.githubusercontent.com/Tony607/jetson_nvidia_dockers/master/overlay_example/test_trt_inference.py)。该模型由用于图像分类的 Keras MobilNet V2 模型转换而来。它在 244×244 彩色图像输入的情况下实现了 30 FPS。这是在 Docker 容器中运行的，与没有 Docker 容器运行的 27.18FPS 相比，它甚至略快。

![fps](img/0a986f94342a7d5dc3e8514305dc4c7b.png)

阅读[我之前的博客](https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/)了解更多关于如何从 Keras 创建 TensorRT 模型的信息。

## 结论和进一步阅读

本教程展示了在 Nvidia Docker 容器中获得运行在 Jetson Nano 上的 Keras 模型的完整过程。您还可以了解如何在 X86 机器上构建 Docker 容器，推送至 Docker Hub，以及从 Jetson Nano 拉取。查看我的 [GitHub repo](https://github.com/Tony607/jetson_nvidia_dockers) 获取更新的 Dockerfile、构建脚本和推理基准脚本。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano-in-nvidia-docker-container/&text=How%20to%20run%20Keras%20model%20on%20Jetson%20Nano%20in%20Nvidia%20Docker%20container) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano-in-nvidia-docker-container/)

*   [←如何为实例分割创建自定义 COCO 数据集](/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/)
*   [用于物体检测的深度学习的最新进展-第 1 部分→](/blog/recent-advances-in-deep-learning-for-object-detection/)