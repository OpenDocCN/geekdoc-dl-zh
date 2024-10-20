# 如何为实例分割创建自定义 COCO 数据集

> 原文：<https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年零五个月前

([评论](/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/#disqus_thread))

![anno_coco](img/15b26989c29e75e0d9302eb24d40b04d.png)

在这篇文章中，我将向您展示使用 Google Colab 的 GPU 免费快速创建自定义 COCO 数据集和训练实例分割模型是多么简单。

如果你只是想知道如何为对象检测创建自定义 COCO 数据集，可以查看我的[之前的教程](https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-object-detection/)。

实例分割不同于对象检测注释，因为它需要多边形注释而不是边界框。这里有很多免费的工具，比如 labelme 和 coco-annotator。 [labelme](https://github.com/wkentaro/labelme) 易于安装并在所有主流操作系统上运行，但是，它缺乏对导出 COCO 数据格式注释的原生支持，而这些注释是许多模型训练框架/管道所需要的。 [coco-annotator](https://github.com/jsbroks/coco-annotator) ，另一方面，是一个基于 web 的应用程序，它需要额外的努力才能在您的机器上运行。这样一来最省力了？

这里概述了如何为实例分割创建自己的 COCO 数据集。

*   下载 labelme，运行应用程序并在图像上标注多边形。
*   运行我的脚本，将 labelme 注释文件转换为 COCO 数据集 JSON 文件。

## 用标签标注数据

[labelme](https://github.com/wkentaro/labelme) 和[label mg](https://github.com/tzutalin/labelImg)在边界标注上颇为相似。所以任何熟悉 labelimg 的人，开始用 labelme 注释应该不会花太多时间。

你可以像下面这样安装 labelme，或者在[发布版](https://github.com/wkentaro/labelme/releases/tag/v3.14.2)中找到预构建的可执行文件，或者下载我之前构建的最新的[Windows 64 位可执行文件](https://github.com/Tony607/labelme2coco/releases/download/V0.1/labelme.exe)。

当你打开这个工具时，点击“打开目录”按钮，导航到你的图像文件夹，所有的图像文件都在这里，然后你就可以开始画多边形了。要画完一个多边形，按“Enter”键，工具会自动连接第一个和最后一个点。注释完一幅图像后，按键盘上的快捷键“D”将带您进入下一幅图像。我注释了 18 幅图像，每幅图像包含多个对象，花了我大约 30 分钟。

![labelme](img/b8a119989df29e48c1c755e3f3369a5e.png)

一旦对所有图像进行了注释，您就可以在 images 目录中找到一个具有相同基本文件名的 JSON 文件列表。这些是 labelimg 注释文件，我们将在下一步将它们转换成一个单独的 COCO 数据集注释 JSON 文件。(或者两个用于训练/测试分割的 JSON 文件。)

## 将 labelme 注释文件转换为 COCO 数据集格式

你可以在我的 GitHub 上找到 [labelme2coco.py](https://github.com/Tony607/labelme2coco/blob/master/labelme2coco.py) 文件。要应用转换，只需传入一个参数，即图像目录路径。

```py
python labelme2coco.py images
```

该脚本依赖于三个 pip 包:labelme、numpy 和 pillow。如果您缺少其中的任何一个，请继续使用 pip 安装它们。执行完脚本后，你会发现当前目录下有一个名为`trainval.json`的文件，那就是 COCO 数据集标注 JSON 文件。

然后可选地，您可以通过打开[COCO _ Image _ viewer . ipynb](https://github.com/Tony607/labelme2coco/blob/master/COCO_Image_Viewer.ipynb)jupyter 笔记本来验证注释。

如果一切正常，它应该显示如下。

![coco_viewer](img/5e9aebf1a1484aa0ff8a106edb2d0ef0.png)

## 用 mmdetection 框架训练实例分割模型

如果你对 mmdetection 框架不熟悉，建议试试我之前的帖子——“[如何用 mmdetection](https://www.dlology.com/blog/how-to-train-an-object-detection-model-with-mmdetection/) 训练一个物体检测模型”。该框架允许您通过同一管道使用可配置的主干网络训练许多对象检测和实例分割模型，唯一需要修改的是模型配置 python 文件，您可以在其中定义模型类型、训练时期、数据集的类型和路径等。例如分割模型，有几个选项可用，你可以用掩码 RCNN 或级联掩码 RCNN 和预训练的主干网络进行迁移学习。为了让初学者更加友好，只需使用免费的 GPU 资源在线运行[Google Colab 笔记本](https://colab.research.google.com/github/Tony607/mmdetection_instance_segmentation_demo/blob/master/mmdetection_train_custom_coco_data_segmentation.ipynb)，并下载最终的训练模型。笔记本和[之前的物体检测演示](https://github.com/Tony607/mmdetection_object_detection_demo/blob/master/mmdetection_train_custom_coco_data.ipynb)挺像的，我就让你运行它玩玩吧。

下面是一个 mask RCNN 模型训练 20 个历元后的最终预测结果，训练时用了不到 10 分钟。

![result2](img/79430a51880c0f14b7c5b783fadcfce6.png)

请随意尝试其他模型配置文件，或者通过增加训练时段来调整现有的模型配置文件，更改批处理大小，看看它如何改善结果。还要注意，为了演示数据集的简单性和小尺寸，我们跳过了训练/测试分割，您可以通过手动将 labelme JSON 文件分割成两个目录并为每个目录运行`labelme2coco.py`脚本来生成两个 COCO annotation JSON 文件。

## 结论和进一步阅读

训练实例分段可能看起来令人畏惧，因为这样做可能需要大量的计算和存储资源。但这并没有阻止我们用大约 20 张带注释的图片和 Colab 的免费 GPU 来创建一个。

#### *你可能会发现有用的资源*

我的 GitHub repo for the[label me 2 COCO](https://github.com/Tony607/labelme2coco)脚本，COCO image viewer 笔记本，以及我的演示数据集文件。

[labelme Github repo](https://github.com/wkentaro/labelme) 在这里你可以找到更多关于注释工具的信息。

[笔记本](https://github.com/Tony607/mmdetection_instance_segmentation_demo/blob/master/mmdetection_train_custom_coco_data_segmentation.ipynb)你可以运行在 Google Colab 上训练一个 mmdetection 实例分割模型。

去 [mmdetection GitHub repo](https://github.com/open-mmlab/mmdetection) 了解更多关于框架的信息。

我之前的帖子- [如何用 mmdetection 训练一个物体检测模型](https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-object-detection/)

*   标签:
*   [教程](/blog/tag/tutorial/)、
*   [深度学习](/blog/tag/deep-learning/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/&text=How%20to%20create%20custom%20COCO%20data%20set%20for%20instance%20segmentation) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/)

*   [←如何为对象检测创建自定义 COCO 数据集](/blog/how-to-create-custom-coco-data-set-for-object-detection/)
*   [如何在 Nvidia Docker 容器中的 Jetson Nano 上运行 Keras 模型→](/blog/how-to-run-keras-model-on-jetson-nano-in-nvidia-docker-container/)