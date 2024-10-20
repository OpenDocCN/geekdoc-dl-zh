# 如何使用自定义 COCO 数据集训练 Detectron2

> 原文：<https://www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年零两个月前

([评论](/blog/how-to-train-detectron2-with-custom-coco-datasets/#disqus_thread))

![/detectron2-custom](img/8a809e0222f46a072bc8a06caa5c7e28.png)

随着最新的 PyTorch 1.3 发布而来的是对之前的对象检测框架的下一代彻底重写，现在称为 Detectron2。本教程将通过使用自定义 COCO 数据集训练实例分割模型来帮助您开始使用该框架。如果你想知道如何创建 COCO 数据集，请阅读我之前的帖子- [如何为实例分割创建自定义 COCO 数据集](https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/)。

作为一个快速的开始，我们将在一台 Colab 笔记本电脑上做我们的实验，因此在熟悉 Pytorch 1.3 和 Detectron2 之前，您不需要担心在您自己的机器上设置开发环境。

## 安装检测器 2

在 Colab 笔记本中，只需运行这 4 行即可安装最新的 Pytorch 1.3 和 Detectron2。

单击单元输出中的“重新启动运行时”,让安装生效。

![restart](img/56fb855b134b9c7fb97c525ff07c7f51.png)

## 注册 COCO 数据集

为了告诉 Detectron2 如何获取数据集，我们将对其进行“注册”。

为了演示这个过程，我们使用  [水果坚果分割数据集](https://github.com/Tony607/mmdetection_instance_segmentation_demo) ，它只有 3 个类:数据、无花果和榛子。我们将从现有的在 COCO 数据集上预训练的模型中训练一个分割模型，该数据集可在 detectron2 的模型动物园中获得。

你可以像这样下载数据集。

或者您可以从这里上传自己的数据集。

![upload](img/18b7c94118454c111d27c6860267d05c.png)

按照 [detectron2 自定义数据集教程](https://github.com/facebookresearch/detectron2/blob/master/docs/tutorials/datasets.md) 将 **fruits_nuts** 数据集注册到 detectron2。

每个数据集都与一些元数据相关联。在我们的例子中，可以通过调用 `fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")`来访问它，你将得到

要获得关于数据集的目录存储信息的实际内部表示以及如何获得它们，可以调用`dataset_dicts = DatasetCatalog.get("fruits_nuts")`。内部 fformat 使用一个 dict 来表示一个图像的注释。

为了验证数据加载是否正确，让我们可视化数据集中随机选择的样本的注释:

其中一张图片可能会显示这一点。

![vis_annotation](img/012fec72b41d913643970a3d7da628ef.png)

## 训练模型

现在，让我们在 fruits_nuts 数据集上微调 coco 预训练的 R50-FPN 掩模 R-CNN 模型。在 Colab 的 K80 GPU 上训练 300 次迭代需要~6 分钟。

如果您切换到自己的数据集，请相应地更改类的数量、学习率或最大迭代次数。

![train](img/a3a4e3a29029fafab9081a45d16aeb77.png)

## 做一个预测

现在，我们用训练好的模型在 fruits_nuts 数据集上进行推理。首先，让我们使用刚刚训练的模型创建一个预测器:

然后，我们随机选择几个样本来可视化预测结果。

这是你得到的预测覆盖的样本图像。

![prediction](img/57fd93f4cc51be850019e4ffb31b13b5.png)

## 结论和进一步的思考

你可能已经读过我的上一篇教程，它是关于一个类似的物体检测框架，名为 MMdetection，也是基于 PyTorch 构建的。那么 Detectron2 和它相比怎么样呢？以下是我的几点想法。

这两个框架都很容易配置，只需一个描述如何训练模型的配置文件。Detectron2 的 YAML 配置文件效率更高有两个原因。首先，您可以通过首先创建一个“基础”配置来重用配置，并在这个基础配置文件的基础上构建最终的训练配置文件，从而减少重复代码。第二，可以首先加载配置文件，并允许在 Python 代码中根据需要进行任何进一步的修改，这使得它更加灵活。

推理速度呢？简单来说，对于相同的掩模 RCNN Resnet50 FPN 模型，Detectron2 比 MMdetection 稍快。MMdetection 的速度为 2.45 FPS，而 Detectron2 的速度为 2.59 FPS，即推理单幅图像的速度提高了 5.7%。基于以下代码的基准测试。

所以，你有了它，Detectron2 让你用自定义数据集训练自定义实例分割模型变得超级简单。您可能会发现以下资源很有帮助。

我之前的帖子 - [如何为实例分割](https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/) 创建自定义 COCO 数据集。

我之前的帖子- [如何用 mmdetection 训练一个物体检测模型](https://www.dlology.com/blog/how-to-train-an-object-detection-model-with-mmdetection/)。

[Detectron2 GitHub 库](https://github.com/facebookresearch/detectron2)。

本帖可运行的 [Colab 笔记本](https://colab.research.google.com/github/Tony607/detectron2_instance_segmentation_demo/blob/master/Detectron2_custom_coco_data_segmentation.ipynb)。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [指针](/blog/tag/pytorch/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/&text=How%20to%20train%20Detectron2%20with%20Custom%20COCO%20Datasets) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/)

*   [←VS 代码远程开发入门](/blog/getting-started-with-vscode-remote-development/)
*   [采用端到端深度学习的自动缺陷检查→](/blog/automatic-defect-inspection-with-end-to-end-deep-learning/)