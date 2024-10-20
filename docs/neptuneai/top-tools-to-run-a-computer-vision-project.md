# 运行计算机视觉项目的顶级工具

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/top-tools-to-run-a-computer-vision-project>

得益于似乎每天都在发布的最先进的工具和技术，计算机视觉已经获得了巨大的吸引力。这项技术正在应用于各个领域，包括自动驾驶汽车、机器人和医学图像分析。

计算机视觉是一个科学领域，涉及使用技术来理解图像和视频。目标是将原本由人类完成的视觉任务自动化。正在自动化的一些常见任务包括:

*   **图像分类**:这涉及到对图像内容的分类
*   目标检测:包括识别图像中的目标并在它们周围画出边界框
*   语义分割:这需要识别图像中每个像素的类别，并在每个对象周围画一个遮罩
*   **人体姿态估计**:识别视频图像中人的姿态

仅举几个例子。

深度学习和机器学习的最新进展使得这些任务的自动化成为可能。特别是，卷积神经网络在推进计算机视觉方面发挥了重要作用。这些网络能够检测和提取图像中的重要特征。这些特征然后被用于识别物体和分类图像。

在本文中，我们将了解一些实现这些计算机视觉应用的顶级工具和技术。

## 框架和库

有许多计算机视觉框架和库。这些工具在计算机视觉领域做不同的事情。各有利弊。我们来看看吧！

### neptune.ai

[neptune.ai](/web/20230216014251/https://neptune.ai/) 是一个可以用来记录和管理计算机视觉模型元数据的平台。您可以用它来记录:

*   型号版本，
*   数据版本，
*   模型超参数，
*   图表，
*   还有很多。

海王星托管在云上，不需要任何设置，随时随地都可以接入你的计算机视觉实验。你可以在一个地方组织计算机视觉实验，并与你的团队合作。你可以邀请你的队友来观看和研究任何计算机视觉实验。

了解更多关于在 Neptune 中记录和管理运行的信息。

### OpenCV

[OpenCV](https://web.archive.org/web/20230216014251/https://opencv.org/) (开源计算机视觉库)是一个开源的计算机视觉和机器学习库。它支持 Windows、Linux、Android 和 Mac OS。

该库还为 [CUDA](https://web.archive.org/web/20230216014251/https://opencv.org/platforms/cuda/) 和 [OpenCL](https://web.archive.org/web/20230216014251/https://opencv.org/opencl/) 提供接口。OpenCV 也可以在 C++、Java、MATLAB 和 Python 中使用。使用 OpenCV 可以完成的一些任务包括:

*   对视频中的人体动作进行分类
*   物体识别
*   跟踪移动物体
*   清点人数
*   检测和识别人脸

OpenCV 也可以用于计算机视觉的图像处理。一些受支持的任务包括:

*   更改色彩空间
*   平滑图像
*   使用图像金字塔混合图像
*   基于分水岭算法的图像分割

例如，下面是你如何对一幅图像进行[傅立叶变换](https://web.archive.org/web/20230216014251/https://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html)。

```py
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('f.jpg',0)
dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
```

### TensorFlow

[TensorFlow](https://web.archive.org/web/20230216014251/https://www.tensorflow.org/) 是谷歌开发的开源深度学习库。它通过使用 [Keras](https://web.archive.org/web/20230216014251/https://keras.io/) 作为其高级 API，使得深度学习模型的构建变得简单而直观。

可以使用 TensorFlow 构建的一些网络包括:

*   卷积神经网络
*   全连接网络
*   递归神经网络
*   长短期记忆网络
*   生成对抗网络

TensorFlow 在开发人员中很受欢迎，因为它易于使用，并为您提供了执行各种操作的多种工具。生态系统中的一些工具包括:

*   用于可视化和调试模型的**张量板**
*   **TensorFlow Playground** 用于在浏览器上修补神经网络
*   包含众多经过训练的计算机视觉模型的 TensorFlow Hub
*   **TensorFlow Graphics** ，一个处理图形的库

我们已经提到卷积神经网络(CNN)主要用于计算机视觉任务。在尽可能多地使用普通人工神经网络的情况下，CNN 已经被证明表现得更好。

让我们看一个构建简单神经网络的代码片段。在这个片段中，您可以看到:

*   使用 Keras 的“顺序”功能初始化层网络
*   “Conv2D”用 3×3 特征检测器定义卷积层。特征检测器在保持重要特征的同时减小图像的尺寸
*   “MaxPooling2D”进行池化。上面获得的特征通过池层传递。常见的池类型包括最大池和最小池。池确保网络能够检测图像中的对象，而不管其在图像中的位置
*   卷积层和池层是重复的。但是，您可以自由定义自己的网络架构。您也可以使用[常见的网络架构](https://web.archive.org/web/20230216014251/https://keras.io/api/applications/)
*   “Flatten”层将池层的输出转换为可以传递给完全连接的层的单个列
*   “致密”是完全连接的层。在这一层，应用对应于问题类型的激活函数
*   最后一层负责产生网络的最终输出

```py
model = tf.keras.Sequential(
    [
    tf.keras.layers.Conv2D(32, (3,3), activation="relu",input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
]
)
```

当建立一个像上面这样的网络时，可以使用海王星来跟踪实验。这是很重要的，这样你的实验是可重复的。花几天时间来训练一个你以后无法复制其表现的网络将是非常不幸的。

[TensorFlow 和 Neptune](https://web.archive.org/web/20230216014251/https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras) 集成让您:

*   每次运行时记录模型超参数
*   可视化模型的学习曲线
*   查看每次运行的硬件消耗
*   记录模型权重
*   记录模型工件，比如图像和模型本身

### PyTorch

[PyTorch](https://web.archive.org/web/20230216014251/https://pytorch.org/) 是基于 Torch 库的开源深度学习库。它还支持分布式训练以及为您的计算机视觉模型服务。

PyTorch 的其他特性包括:

*   在移动设备上部署
*   支持 ONNX
*   C++前端
*   在主要云平台上受支持
*   工具和库的丰富生态系统

PyTorch 及其生态系统也得到海王星机器学习实验平台的支持。这个 [PyTorch + Neptune 集成](https://web.archive.org/web/20230216014251/https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch)让你:

*   原木火炬张量
*   可视化模型损失和指标
*   将火炬张量记录为图像
*   日志训练代码
*   记录模型权重

定义卷积神经网络与张量流定义非常相似，只是做了一些表面上的修改。正如您所看到的，您仍然需要定义卷积层和池层以及指示激活功能。

```py
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
```

如果不喜欢用原始 PyTorch 代码编写网络，可以使用它的一个高级 API 库。一个这样的包是“fastai”图书馆。

### 法斯泰

[Fastai](https://web.archive.org/web/20230216014251/https://www.fast.ai/) 是一个基于 PyTorch 构建的深度学习库。

该库的一些功能包括:

*   GPU 优化的计算机视觉模块
*   可以访问数据任何部分的双向回调系统
*   来自其他库时易于入门
*   支持循环学习率

[Fastai](https://web.archive.org/web/20230216014251/https://docs.neptune.ai/essentials/integrations/deep-learning-frameworks/fastai) 指标也可以记录到 Neptune。该集成还允许您监控和可视化模型的训练过程。

我们来看一个使用 [Fastai](https://web.archive.org/web/20230216014251/https://docs.fast.ai/) 进行图像分类的例子。第一步通常是创建一个“学习者”。“学习者”结合了模型、数据加载器和损失函数。定义“学习者”时，您可以使用预先训练的模型，并在数据集上对其进行微调。这被称为[迁移学习](/web/20230216014251/https://neptune.ai/blog/transfer-learning-guide-examples-for-images-and-text-in-keras)，与从头开始训练模型相比，通常会产生更好的模型性能。预先训练的模型通常很难被击败，因为它们在数百万张图像上进行训练，而这些图像对于个人来说很难获得。此外，需要大量的计算资源来训练具有数百万图像的模型。

```py
from fastai.vision.all import *
dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```

然后，您可以使用这个“学习者”来运行预测。

```py
learn.predict(files[0])
```

Fastai 还提供下载样本数据和[增强](https://web.archive.org/web/20230216014251/https://docs.fast.ai/vision.augment.html)图像的工具。

检查如何利用 Neptune 来记录和跟踪 [fastai](https://web.archive.org/web/20230216014251/https://docs.neptune.ai/integrations-and-supported-tools/model-training/fastai) 指标。

### 咖啡

[Caffe](https://web.archive.org/web/20230216014251/https://caffe.berkeleyvision.org/) 是由 Berkeley AI Research (BAIR)开发的开源深度学习框架。

咖啡的一些特点包括:

*   能够在不同的后端之间轻松切换
*   快速图书馆
*   可用于图像分类和图像分割
*   支持 CNN、RCNN 和 LSTM 网络

像其他库一样，在 Caffe 中创建网络需要定义网络层、损失函数和激活函数。

```py
from caffe import layers as L, params as P

def lenet(lmdb, batch_size):

    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)
```

### 奥本维诺

[OpenVINO](https://web.archive.org/web/20230216014251/https://docs.openvinotoolkit.org/latest/index.html) (开放式视觉推理和神经网络优化)可以让用计算机视觉模型进行推理更快。

OpenVINO 提供的其他功能包括:

*   支持基于 CNN 的边缘深度学习推理
*   附带针对 OpenCV 和 OpenCL 优化的计算机视觉功能

## 特定任务的现成解决方案

除了开源计算机视觉工具之外，还有现成的解决方案可用于构建计算机视觉应用程序的各个阶段。这些解决方案通常是托管的，可以立即使用。他们还提供免费版本，你可以马上开始使用。让我们来看看其中的一些。

### 德西

Deci 是一个平台，你可以用它来优化你的计算机视觉模型并提高它们的性能。

该平台能够:

*   减少模型的延迟
*   增加模型的吞吐量，而不影响其准确性
*   支持来自流行框架的模型优化
*   支持在流行的 CPU 和 GPU 机器上部署
*   在不同的硬件主机和云提供商上测试您的模型

你可以使用 Deci 平台来优化你的机器学习模型。该过程从上传模型开始，并通过选择所需的优化参数对其进行优化。

下图显示了在 Deci 上运行 YOLO V5 优化后获得的结果。图像显示如下:

*   该型号的吞吐量增加了 2.1 倍
*   模型的尺寸缩小了 1.4 倍
*   该型号的吞吐量从 194.1 FPS 提高到 413.2 FPS。现在快了 2.1 倍。
*   该模型的延迟提高了 2.1 倍

### 克拉里菲

[Clarifai](https://web.archive.org/web/20230216014251/https://www.clarifai.com/) 提供可通过 API、设备 SDK 和内部部署访问的图像和视频解决方案。

该平台的一些功能包括:

*   视觉搜索
*   图像裁剪器
*   视觉分类模型
*   按图像搜索
*   视频插值
*   人口统计分析
*   空中监视
*   数据标记

### 监督地

[Supervise.ly](https://web.archive.org/web/20230216014251/http://supervise.ly/) 为计算机视觉模型提供图像标注工具。

该平台提供的其他功能包括:

*   视频标注
*   3D 场景的标记
*   计算机视觉模型的训练和测试

### 标签盒

[Labelbox](https://web.archive.org/web/20230216014251/https://labelbox.com/) 是一个可以用来为计算机视觉项目标注数据的平台。

支持的计算机视觉项目有:

*   创建边界框、多边形、直线和关键点
*   图像分割任务
*   图像分类
*   视频标注

### 片段

[Segments](https://web.archive.org/web/20230216014251/http://segments.ai/) 是一个用于标注和模型训练的计算机视觉平台。

该平台支持:

*   实例分割
*   语义分割
*   多边形
*   边框和
*   图像分类

### 纳米网络

[Nanonets](https://web.archive.org/web/20230216014251/http://nanonets.ai/) 为各行业的应用计算机视觉提供解决方案。

该平台提供的一些解决方案包括:

*   身份证验证
*   机器学习和光学字符识别在各种文档分析中的应用

### 感知

[Sentisight](https://web.archive.org/web/20230216014251/http://sentisight.ai/) 计算机视觉平台提供以下功能:

*   智能图像标注工具
*   分类标签、边界框和多边形
*   训练分类和对象检测模型
*   图像相似性搜索
*   预训练模型

### 亚马逊索赔案

[Amazon Rekognition](https://web.archive.org/web/20230216014251/https://aws.amazon.com/rekognition/?blog-cards.sort-by=item.additionalFields.createdDate&blog-cards.sort-order=desc) 无需任何深度学习经验即可用于图像和视频分析。

该平台提供:

*   能够识别数千个标签
*   添加自定义标签
*   内容审核
*   文本识别
*   人脸检测和分析
*   人脸搜索和验证
*   名人认可

### 谷歌云视觉 API

[Google Cloud](https://web.archive.org/web/20230216014251/https://cloud.google.com/vision#which-vision-product-is-right-for-you) 为 vision APIs 提供了以下特性:

*   使用预训练模型的图像分类
*   数据标记
*   在边缘设备上部署模型
*   检测和计数物体
*   检测人脸
*   内容审核
*   名人认可

### 德国人

[Fritz](https://web.archive.org/web/20230216014251/http://fritz.ai/) 平台可以用于建立计算机视觉模型，无需任何机器学习经验。

该平台提供以下功能:

*   从少量图像样本生成图像
*   图像标注
*   图像分类和目标检测模型
*   图像分割模型
*   在边缘设备上部署训练模型
*   基于机器学习的镜头工作室镜头

### 微软计算机视觉应用编程接口

[微软](https://web.archive.org/web/20230216014251/https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/#features)还提供了一个分析图像和视频内容的计算机视觉平台。

该服务提供的一些功能包括:

*   文本提取
*   图像理解
*   空间分析
*   在云和边缘部署

### IBM Watson 视觉识别

IBM Watson 视觉识别服务提供以下服务:

*   将模型导出到 CoreML
*   预训练分类模型
*   自定义图像分类模型
*   定制模型培训

### ShaipCloud

ShaipCloud 是一个计算机视觉平台，用于标记敏感的人工智能训练数据——无论是图像还是视频。

该平台支持:

*   **图像标注/标注:**语义分割、关键点标注、包围盒、3D 长方体、多边形标注、地标标注、线段分割
    - > **用例:**物体检测、人脸识别跟踪、图像分类
*   **视频标注/标注:**逐帧标注、基于事件的时间戳标注、关键点标注
    - > **用例:**物体/运动跟踪、面部识别、自动驾驶、安全监控、视频/剪辑分类
*   **行业迎合:**尽管他们是行业不可知论者，但他们的高吞吐量服务已经在汽车、医疗保健、金融服务、技术、零售和政府等各种垂直行业推动了下一代技术的发展。

## 最后的想法

在本文中，我们介绍了可以在计算机视觉项目中使用的各种工具。我们探索了几种开源工具，以及各种现成的计算机视觉平台。

您对工具或平台的选择将取决于您的技能和预算。例如，ready 平台可以在没有先验知识的情况下用于深度学习，但它们不是免费的。开源工具是免费的，但是需要技术知识和经验才能使用。

无论你选择哪个平台或者工具，最根本的是要保证它能解决你的问题。

### 资源