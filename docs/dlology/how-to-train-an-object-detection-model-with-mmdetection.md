# 如何使用 mmdetection 训练对象检测模型

> 原文：<https://www.dlology.com/blog/how-to-train-an-object-detection-model-with-mmdetection/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年零六个月前

([评论](/blog/how-to-train-an-object-detection-model-with-mmdetection/#disqus_thread))

![mmdetection_colab](img/12f3c1801cdc03036cd5ab92586121a0.png)

不久前，你已经学习了如何使用 TensorFlow 对象检测 API 和 Google Colab 的免费 GPU 来训练对象检测模型，如果你还没有，请查看帖子中的[。TensorFlow 对象检测中的模型已经相当过时，缺少对 Cascade RCNN 和 RetinaNet 等最新模型的更新。虽然 Pytorch 有一个类似于](https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/) [mmdetection](https://github.com/open-mmlab/mmdetection) 的对应物，它包括更多预先训练的最先进的对象检测模型，供我们训练自定义数据，但是设置它需要花费大量的时间来安装环境、设置配置文件和正确格式的数据集。好消息是你可以跳过那些无聊的东西，直接跳到有趣的部分来训练你的模型。

这里有一个如何实现它的概述，

1.注释一些图像，并进行训练/测试分割。

2.运行[Colab 笔记本](https://colab.research.google.com/github/Tony607/mmdetection_object_detection_demo/blob/master/mmdetection_train_custom_data.ipynb)来训练你的模型。

## 步骤 1: 注释一些图像并进行训练/测试分割

如果你想使用你的图片而不是那些自带的图片，只需要我**[我的库](https://github.com/Tony607/mmdetection_object_detection_demo)** 。从分叉 [my repository](https://github.com/Tony607/mmdetection_object_detection_demo) 开始，删除项目目录中的`data`文件夹，这样您就可以使用您的定制数据重新开始。

如果你是用手机拍摄的，图像分辨率可能是 2K 或 4K，这取决于你手机的设置。在这种情况下，我们将缩小图像以减少整体数据集大小，并加快训练速度。

你可以使用[调整图片大小。存储库中的 py](https://github.com/Tony607/mmdetection_object_detection_demo/blob/master/resize_images.py) 脚本来调整图像的大小。

首先，将你所有的照片保存到项目目录之外的一个文件夹中，这样它们就不会被意外上传到 GitHub。理想情况下，所有照片都是用`jpg`扩展。然后运行这个脚本来调整所有照片的大小，并将它们保存到项目目录中。

```py
python resize_images.py --raw-dir <photo_directory> --save-dir ./data/VOCdevkit/VOC2007/ImageSets --ext jpg --target-size "(800, 600)"
```

你可能想知道为什么“VOC”在 path 中，那是因为我们使用的注释工具生成了 [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) 格式的注释 XML 文件。没有必要深究 XML 文件的实际格式，因为注释工具会处理所有这些。你猜对了，这就是我们之前使用的， **[LabelImg](https://tzutalin.github.io/labelImg/)** 的工具，在 Windows 和 Linux 上都可以工作。

[下载 LabelImg](https://tzutalin.github.io/labelImg/) 并打开，

1.确认选择了“ **PascalVOC** ”，这是默认的注释格式。

2.打开已调整大小的图像文件夹“`./data/VOCdevkit/VOC2007/ImageSets`”进行注释。

3.将 XML 注释文件的保存目录更改为“`./data/VOCdevkit/VOC2007/Annotations`”。

![labelimg](img/763a3994db23eab4c7c0f83c72c1b614.png)

*照常使用快捷键(`w`:画  框 ，`d`:下一档，`a`:上一档等等。)加速标注。*

一旦完成，你会发现这些 XML 文件位于`./data/VOCdevkit/VOC2007/Annotations`文件夹中，它们的文件名与你的图像文件的文件名相同。

对于训练/测试分割，您将创建两个文件，每个文件包含一个文件基础名称列表，每行一个名称。这两个文本文件将分别位于名为`trainval.txt`和`test.txt`的文件夹`data/VOC2007/ImageSets/Main`中。如果您不想手动键入所有文件名，请尝试将 cd 放入“`Annotations`”目录并运行 shell，

这将给你一个排序良好的文件库名列表，只需将它们分成两部分，然后粘贴到这两个文本文件中。

现在您有了类似于下面这个的`data`目录结构。

用您标记的数据集更新您在 [GitHub 存储库](https://github.com/Tony607/mmdetection_object_detection_demo)中的分支，这样您就可以用 Colab 克隆它。

## 在 Colab 笔记本上训练模型

我们准备推出 [Colab 笔记本](https://colab.research.google.com/github/Tony607/mmdetection_object_detection_demo/blob/master/mmdetection_train_custom_data.ipynb)并启动培训。类似于 TensorFlow 对象检测 API，不是从零开始训练模型，我们将从预先训练的主干进行迁移学习，比如模型配置文件中指定的 resnet50。

笔记本允许您选择模型配置并设置训练时期的数量。

目前，我只测试了两种型号配置， **faster_rcnn_r50_fpn_1x，**和 **cascade_rcnn_r50_fpn_1x** ，其他配置可以按照笔记本中的演示进行整合。

笔记本在训练模型之前处理几件事情，

1.  正在安装 **mmdetection** 及其依赖项。
2.  用您的自定义数据集类标签替换 **voc.py** 文件中的**类**。
3.  修改选定的模型配置文件。比如更新类的数量以匹配您的数据集，将数据集类型更改为 **VOCDataset** ，设置总训练历元数等等。

之后，它将重新运行 **mmdetection** 包安装脚本，因此对 **voc.py** 文件的更改将被更新到系统 python 包中。

由于您的数据目录位于 **mmdetection** 目录之外，我们在笔记本中有以下单元格，它创建了一个到项目数据目录的符号链接。

然后开始训练。

训练时间取决于数据集的大小和训练时期的数量，我的演示使用 Colab 的特斯拉 T4 GPU 需要几分钟才能完成。

训练之后，您可以使用测试集中的图像来试驾模型，如下所示。

这是你所期望的结果，

![result](img/518d23d1e56cc027d6afab1d5f6bbcb8.png)

## 结论和进一步阅读

本教程向您展示如何使用您的自定义数据集训练 Pytorch **mmdetection** 对象检测模型，并在 Google Colab 笔记本上进行最小的努力。

如果您正在使用我的 GitHub repo，您可能已经注意到 **mmdetection** 作为一个子模块被包含在内，为了在将来更新它，运行这个命令。

考虑用另一种模型配置进行训练？你可以在这里找到一个配置文件列表以及[它们的规格](https://github.com/open-mmlab/mmdetection/blob/master/MODEL_ZOO.md#baselines)比如复杂度( **Mem(GB)** )和准确度( **box AP** )。然后在[笔记本](https://colab.research.google.com/github/Tony607/mmdetection_object_detection_demo/blob/master/mmdetection_train_custom_data.ipynb)的开头将配置文件添加到 **MODELS_CONFIG** 中。

你可能会觉得有用的资源，

*   [mmdetection](https://github.com/open-mmlab/mmdetection) - GitHub 库。
*   [LabelImg](https://tzutalin.github.io/labelImg/) -本教程中使用的注释工具。
*   本教程的我的存储库。

在未来的帖子中，我们将研究这些定制模型的基准测试，以及它们在边缘计算设备上的部署，敬请关注，祝编码愉快！

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)，
*   [指针](/blog/tag/pytorch/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-train-an-object-detection-model-with-mmdetection/&text=How%20to%20train%20an%20object%20detection%20model%20with%20mmdetection) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-train-an-object-detection-model-with-mmdetection/)

*   [←如何用 Efficientnet 进行迁移学习](/blog/transfer-learning-with-efficientnet/)
*   [如何创建用于对象检测的自定义 COCO 数据集→](/blog/how-to-create-custom-coco-data-set-for-object-detection/)