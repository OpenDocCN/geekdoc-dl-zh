# 如何创建用于对象检测的自定义 COCO 数据集

> 原文：<https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-object-detection/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年零五个月前

([评论](/blog/how-to-create-custom-coco-data-set-for-object-detection/#disqus_thread))

![voc2coco](img/c6b0c8a93828312e4b6d5854b9e400a0.png)

[在之前的](https://www.dlology.com/blog/how-to-train-an-object-detection-model-with-mmdetection/)中，我们已经用 Pascal VOC 数据格式的自定义注释数据集训练了一个 mmdetection 模型。如果你的对象检测训练管道需要 COCO 数据格式，你就不走运了，因为我们使用的 labelImg 工具不支持 COCO 注释格式。如果你仍然想继续使用注释工具，然后将注释转换成 COCO 格式，这篇文章是为你准备的。

我们将首先简要介绍两种注释格式，然后介绍将 VOC 转换为 COCO 格式的转换脚本，最后，我们将通过绘制边界框和类标签来验证转换结果。

## Pascal VOC 和 COCO 注释

Pascal VOC 注释保存为 XML 文件，每个图像一个 XML 文件。对于由 labelImg 工具生成的 XML 文件。它包含了<路径>元素中图像的路径。每个边界框都存储在一个<对象的>元素中，示例如下。

如你所见，边界框由两个点定义，左上角和右下角。

对于 COCO 数据格式，首先，一个数据集中的所有注释只有一个 JSON 文件，或者数据集的每个分割(Train/Val/Test)都有一个 JSON 文件。

包围盒表示为左上角的起始坐标和盒子的宽度和高度，和`"bbox" :[x,y,width,height]`一样。

这是一个 COCO 数据格式 JSON 文件的例子，它只包含一个图像，如在顶级“图像”元素中看到的，在顶级“类别”元素中看到的总共 3 个唯一的类别/类，以及在顶级“注释”元素中看到的图像的 2 个带注释的边界框。

## 将 Pascal VOC 转换为 COCO 注释

一旦你有了一些带注释的 XML 和图像文件，把它们放在类似下面的文件夹结构中，

```py
data
 └── VOC2007
 ├── Annotations
 │ ├── 0.xml
 │ ├── ...
 │ └── 9.xml
 └── JPEGImages
 ├── 0.jpg
 ├── ...
 └── 9.jpg
```

然后你可以像这样从我的 GitHub 运行 [voc2coco.py](https://github.com/Tony607/voc2coco/blob/master/voc2coco.py) 脚本，这会为你生成一个 coco 数据格式的 JSON 文件。

```py
python voc2coco.py ./data/VOC/Annotations ./data/coco/output.json
```

一旦我们有了 JSON 文件，我们就可以通过在图像上绘制边界框和类标签来可视化 COCO 注释。打开 Jupyter 笔记本中的 [COCO_Image_Viewer.ipynb](https://github.com/Tony607/voc2coco/blob/master/COCO_Image_Viewer.ipynb) 。在笔记本中找到下面的单元格，它调用`display_image`方法在笔记本中生成 SVG 图。

第一个参数是图像 id，对于我们的演示数据集，总共有 18 个图像，因此您可以尝试将其设置为 0 到 17。

![vis_8](img/845a44fd2de5fb2b9a15bd949c320bde.png)

## 结论和进一步阅读

在这个快速教程中，您已经学习了如何坚持使用流行的 [labelImg](https://tzutalin.github.io/labelImg/) [](https://tzutalin.github.io/labelImg/)进行自定义数据集注释，并在稍后将 Pascal VOC 转换为 COCO 数据集，以训练需要 COCO 格式数据集的对象检测模型管道。

#### *您可能会发现以下链接很有用，*

如何用 mmdetection 训练一个对象检测模型——我之前的帖子关于创建自定义 Pascal VOC 注释文件，用 PyTorch mmdetection 框架训练一个对象检测模型。

[COCO 数据格式](http://cocodataset.org/#format-data)

[Pascal VOC 文档](https://pjreddie.com/media/files/VOC2012_doc.pdf)

下载 [labelImg](https://tzutalin.github.io/labelImg/) 用于边界框注释。

获取这篇文章的源代码，查看[我的 GitHub 报告](https://github.com/Tony607/voc2coco)。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-create-custom-coco-data-set-for-object-detection/&text=How%20to%20create%20custom%20COCO%20data%20set%20for%20object%20detection) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-object-detection/)

*   [←如何用 mmdetection 训练物体检测模型](/blog/how-to-train-an-object-detection-model-with-mmdetection/)
*   [如何为实例分割创建自定义 COCO 数据集→](/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/)