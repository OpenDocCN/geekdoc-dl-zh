# 《如果我是个女孩》-斯塔根的魔镜

> 原文：<https://www.dlology.com/blog/if-i-were-a-girl-magic-mirror-by-stargan/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零四个月前

([评论](/blog/if-i-were-a-girl-magic-mirror-by-stargan/#disqus_thread))

![mirror](img/1ab57815a9820416bb599ebbb4da024a.png)

有没有想过如果你是女孩，你会是什么样子？

想象一下。我跳下床，看着镜子。我是金发的！

你问:“那是你作为一个女孩的样子吗？”

我说:“是的，天哪，是的，是的，是的！这是我一直想要的！

魔镜由 StarGAN 提供支持，这是一个用于多领域图像到图像翻译的统一生成对抗网络。这篇文章将向你展示这个模型是如何工作的，以及你如何建造魔镜。

![magic-mirror](img/a8d2f9e66021f312181ce313536f4bf0.png)

#### *在这里欣赏 YouTube 上的演示[。](https://youtu.be/PkWIalWnYUg)*

#### *我的 [GitHub](https://github.com/Tony607/DeepMagicMirror) 页面上有完整的源代码。*

## 星际之门简介

图像到图像的翻译是将一个给定图像的某一方面改变为另一方面，例如将一个人的性别从男性改变为女性。随着生成对抗网络(GANs)的引入，这项任务经历了重大改进，结果包括从边缘地图生成照片，改变风景图像的季节，以及从莫奈的画作 重建照片。【T8 ![gan-model](img/ff2a9e5a9607fc1bfc6812bbf1f979e6.png) 

给定来自两个不同领域的训练数据，这些模型学习以单向方式将图像从一个领域翻译到另一个领域。例如，一个生成模型被训练成将黑头发的人翻译成金发。任何单个现有的 GAN 模型都不能“向后”翻译，就像前面的例子中从金色头发到黑色头发。此外，单一模型无法处理灵活的多领域图像翻译任务。比如性别和头发颜色的可配置转换。这就是 StarGAN 脱颖而出的地方，一种新颖的生成式对抗网络，仅使用单个生成器和鉴别器来学习多个域之间的映射，从所有域的图像中进行有效训练。StarGAN 的模型不是学习一种固定的翻译(例如，从黑人到金发)，而是将图像和领域信息都作为输入，并学习将输入图像灵活地翻译到相应的领域。
预训练的 StarGAN 模型由两个网络组成，像其他 GAN 模型一样，生成网络和判别网络。虽然只需要生成网络来构建魔镜，但是理解完整的模型来自哪里仍然是有用的。

生成网络 以两条信息作为输入，原始 RGB 图像为 256×256 分辨率，目标标签生成一幅相同分辨率的假图像，判别网络 学习区分真假图像，并将真实图像分类到其对应的域。 

我们要使用的预训练模型是在 CelebA 数据集上训练的，该数据集包含 202，599 张名人的面部图像，每张图像都用 40 个二元属性进行了注释，而研究人员使用以下属性选择了七个域:头发颜色( 黑色 、 金发 、 棕色 )、性别( 男/女 

![star-gan](img/04d2f8c071fb1aab4d7046a5b4b6efc3.png)

## 建筑魔镜

StarGAN 的研究人员在我们的魔镜项目所在的 GitHub 上发布了他们的[代码](https://github.com/yunjey/StarGAN)。我也是第一次处理 PyTorch 框架，到目前为止进展顺利。如果你像我一样是 PyTorch 框架的新手，你会发现很容易上手，尤其是有了 Keras 或 TensorFlow 等深度学习框架的经验之后。

完成该项目只需要 PyTorch 框架的最基本知识，如 PyTorch 张量、加载预定义的模型权重等。

让我们从安装框架开始。在我的情况下，在 Windows 10 上，这是最新的 PyTorch 官方支持的。

要使魔镜实时运行，并将可察觉的延迟降至最低，请使用游戏电脑的 Nvidia 显卡(如果有)来加速模型执行。

从 Nvidia 开发者网站上的[这个链接安装 CUDA 9。](https://developer.nvidia.com/cuda-90-download-archive)

![cuda9](img/c2f1cfad1f63b816a29005e8af179b91.png)

之后安装 PyTorch 与 CUDA 9.0 支持以下[其官方网站](https://pytorch.org/)的指示。

![pytorch](img/db98417273997f59d17a116f1ca144a5.png)

当 PyTorch 和其他 Python 依赖项安装好后，我们就可以开始编写代码了。

为了实现简单的实时人脸跟踪和裁剪效果，我们将使用 Python 的 OpenCV 库中的轻量级 **CascadeClassifier** 模块。该模块从网络摄像头帧中获取灰度图像，并返回检测到的人脸的边界框信息。如果在给定的帧中检测到多个面部，我们将采用具有最大计算边界框面积的“主要”面部。

由于 StarGAN 生成网络期望图像的像素值在-1 到 1 之间，而不是在 0 到 255 之间，我们将使用 PyTorch 的内置图像转换实用程序来处理图像预处理。

生成网络子类 PyTorch 的 `nn.Module` 其中表示可以通过传入输入张量作为自变量直接调用。

`labels`变量是一个 PyTorch 张量，有 5 个值，每个值设置为 0 或 1，表示 5 个目标标签。

将代码包装成一个函数调用 `MagicMirror()` ，其中带几个可选参数。

*   videoFile:保留默认值 0 使用第一个网络摄像头，或者传入一个视频文件路径。
*   setHairColor:三者之一，“黑色”、“金发”、“棕色”。
*   setMale:变身男性？设置为真或假。
*   setYoung:变身年轻人？设置为真或假。
*   



## 结论和进一步的思考

本教程向您展示了使用 PyTorch 这样的新框架并用预先训练好的 StarGAN 网络构建一些有趣的东西是多么容易和有趣。

生成的图像可能看起来还不是非常真实，而 [StarGAN 的论文](https://arxiv.org/abs/1711.09020)显示了一个与 CelebA + RaFD 数据集联合训练的模型可以通过利用这两个数据集来改善面部关键点检测和分割等共享的低级任务，从而生成伪影更少的图像。你可以跟随[他们的官方 GitHub](https://github.com/yunjey/StarGAN) 下载数据集并训练这样一个模型，只要你有一台结实的机器和额外的周来运行训练。

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/if-i-were-a-girl-magic-mirror-by-stargan/&text=%22If%20I%20were%20a%20girl%22%20-%20Magic%20Mirror%20by%20StarGAN) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/if-i-were-a-girl-magic-mirror-by-stargan/)

*   [←带树莓派的 DIY 物体检测涂鸦相机(下)](/blog/diy-object-detection-doodle-camera-with-raspberry-pi-part-2/)
*   [如何在 Google Colab 上运行支持 GPU 和 CUDA 9.2 的 py torch→](/blog/how-to-run-pytorch-with-gpu-and-cuda-92-support-on-google-colab/)