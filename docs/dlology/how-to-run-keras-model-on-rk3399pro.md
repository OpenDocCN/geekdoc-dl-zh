# 如何在 RK3399Pro 上运行 Keras 模型

> 原文：<https://www.dlology.com/blog/how-to-run-keras-model-on-rk3399pro/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年零八个月前

([评论](/blog/how-to-run-keras-model-on-rk3399pro/#disqus_thread))

![keras-tb](img/41ebca3ee6e2f0c91c213f5e911fb30c.png)

此前，我们已经推出了多种嵌入式边缘计算解决方案并对其进行了基准测试，包括用于英特尔神经计算棒的 [OpenVINO](https://www.dlology.com/blog/how-to-run-keras-model-inference-x3-times-faster-with-cpu-and-intel-openvino-1/) 、用于 ARM 微控制器的 [CMSIS-NN](https://www.dlology.com/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn/) 以及用于 [Jetson Nano](https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/) 的 TensorRT 模型。

它们的共同点是每个硬件提供商都有自己的工具和 API 来量化张量流图，并结合相邻层来加速推理。

这一次，我们将看看 RockChip RK3399Pro SoC，它内置 NPU(神经计算单元)，在 8 位精度下的推理速度为 2.4 次，能够以超过 28 FPS 的速度运行 Inception V3 模型。您将会看到，在电路板上部署 Keras 模型与前面提到的解决方案非常相似。

1.  将 Keras 模型冻结为张量流图，并用 RKNN 工具包创建推理模型。
2.  将 RKNN 模型加载到 RK3399Pro 开发板上并进行预测。

让我们开始第一次设置。

## 设置 RK3399Pro 板

任何带有 RK3399Pro SoC 的开发板，如 [Rockchip Toybrick RK3399PRO 板](https://www.amazon.com/Toybrick-Development-Artificial-Intelligence-Acceleration/dp/B07P3M7683)或 [Firefly Core-3399Pro](http://shop.t-firefly.com/goods.php?id=98) 都应该可以工作。我有一个 Rockchip Toybrick RK3399PRO 板，6GB 内存(2GB 专用于 NPU)。

该板带有许多类似于 Jetson Nano 的连接器和接口。值得一提的是，HDMI 连接器无法与我的显示器一起工作，但是，我可以让 USB Type-C 转 HDMI 适配器工作。

![tb-rk3399pro](img/426c925c35f04fb15c720becb4f08cfd.png)

它预装了 Fedora Linux release 28，默认用户名和密码为“toybrick”。

RK3399Pro 有 6 个内核 64 位 CPU，采用 [aarch64 架构](https://en.wikipedia.org/wiki/ARM_architecture#AArch64)与[杰特森纳米](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/)相同的架构，但与只有 ARMv7 32 位的树莓 3B+截然不同。这意味着任何针对 Raspberry Pi 的预编译 python wheel 包都不太可能与 RK3399Pro 或 Jetson Nano 一起工作。不过不要绝望，可以从我的[aarch 64 _ python _ packages](https://coding.net/u/zcw607/p/aarch64_python_packages/git)repo 下载预编译的 aarch64 python wheel 包文件包括 scipy、onnx、tensorflow 和 rknn_toolkit 从他们的[官方 GitHub](https://github.com/rockchip-toybrick/RKNPUTool/tree/master/rknn-toolkit/package) 下载。

将这些 wheel 文件传输到 RK3399Pro 板上，然后运行以下命令。

```py
sudo dnf update -y
sudo dnf install -y cmake gcc gcc-c++ protobuf-devel protobuf-compiler lapack-devel
sudo dnf install -y python3-devel python3-opencv python3-numpy-f2py python3-h5py python3-lmdb
sudo dnf install -y python3-grpcio

sudo pip3 install scipy-1.2.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install onnx-1.4.1-cp36-cp36m-linux_aarch64.whl
sudo pip3 install tensorflow-1.10.1-cp36-cp36m-linux_aarch64.whl
sudo pip3 install rknn_toolkit-0.9.9-cp36-cp36m-linux_aarch64.whl 
```

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-keras-model-on-rk3399pro/&text=How%20to%20run%20Keras%20model%20on%20RK3399Pro) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-keras-model-on-rk3399pro/)

*   [←如何在 Jetson Nano 上运行 TensorFlow 对象检测模型](/blog/how-to-run-tensorflow-object-detection-model-on-jetson-nano/)
*   [如何在 Jupyter 笔记本中运行 PyTorch 1.1.0 的 tensor board→](/blog/how-to-run-tensorboard-for-pytorch-110-inside-jupyter-notebook/)