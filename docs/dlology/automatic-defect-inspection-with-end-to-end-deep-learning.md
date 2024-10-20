# 利用端到端深度学习的自动缺陷检查

> 原文：<https://www.dlology.com/blog/automatic-defect-inspection-with-end-to-end-deep-learning/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年零两个月前

([评论](/blog/automatic-defect-inspection-with-end-to-end-deep-learning/#disqus_thread))

![defect-detection](img/c7d093968b6a1e5940e7eacb57bc457a.png)

在本教程中，我将向您展示如何建立一个深度学习模型来查找表面上的缺陷，这是许多工业检测场景中的一个流行应用。

![industrial-applications](img/8dd080645ecc5187f3ca03c1bf3b439b.png)

*由英伟达提供*

## 建立模型

我们将使用 U-Net 作为 2D 工业缺陷检查的 DL 模型。当缺少标记数据，并且需要快速性能时，U-net 是一个很好的选择。基本架构是一个带跳跃连接的编码器-解码器对，用于将低级特征映射与高级特征映射相结合。为了验证我们模型的有效性，我们将使用 [DAGM 数据集](https://resources.mpi-inf.mpg.de/conference/dagm/2007/prizes.html)。使用 U-Net 的好处是它不包含任何密集层，因此训练的 DL 模型通常是缩放不变的，这意味着它们不需要跨图像大小重新训练，就可以对多种输入大小有效。

这是模型结构。

![u-net-model](img/e3f520f4bdedae20c7ef3b7ed8baa1b7.png)

如您所见，我们使用了四次 2x2 max 池操作进行下采样，从而将分辨率降低了一半，降低了四倍。在右侧，2 x2 conv 2d 转置(称为去卷积)将图像升采样回其原始分辨率。为了使下采样和上采样工作，图像分辨率必须能被 16 整除(或 2⁴)，这就是为什么我们将输入图像和蒙版的大小从原来的 500x500 大小的 DAGM 数据集调整为 512x512 分辨率。

来自网络中较早层的这些跳跃连接(在下采样操作之前)应该提供必要的细节，以便为分割边界重建精确的形状。事实上，通过添加这些跳跃连接，我们可以恢复更精细的细节。

在 Keras functional API 中构建这样的模型很简单。

## 损失和指标

图像分割任务最常用的损失函数是**逐像素交叉熵损失**。这种损失单独检查每个像素，将类别预测(深度方向像素向量)与我们的独热编码目标向量进行比较。

因为交叉熵损失单独评估每个像素向量的分类预测，然后对所有像素进行平均，所以我们基本上主张对图像中的每个像素进行平等学习。如果您的各种类在图像中具有不平衡的表示，这可能是一个问题，因为训练可能由最普遍的类主导。在我们的例子中，这是前景到背景的不平衡。

用于图像分割任务的另一个流行的损失函数是基于**骰子系数**，其本质上是两个样本之间重叠的度量。该度量的范围从 0 到 1，其中 Dice 系数为 1 表示完全重叠。Dice 系数最初是针对二进制数据开发的，可以计算如下:

![dice-coefficient](img/1b56fd8ba86a8a1587e8dd12ba2a73fc.png)

其中|A∩B|表示集合 A 和 B 之间的公共元素，而|A|表示集合 A 中元素的数量(对于集合 B 也是如此)。

关于神经网络输出，分子与我们的预测和目标掩模之间的共同激活有关，而分母与每个掩模中单独激活的数量有关。这具有根据目标遮罩的大小来归一化我们的损失的效果，使得软骰子损失不会从图像中具有较小空间表示的类中学习。

这里我们使用加一或拉普拉斯平滑，它只是在每个计数上加一。加一平滑可以被解释为均匀的先验，这减少了过度拟合并使模型更容易收敛。

实现平滑的骰子系数损失。

这里我们比较了二进制交叉熵损失和平滑 Dice 系数损失的性能。

![binary-cross-entropy-loss](img/2c99c1c001f02504cb1d00a5f14eed75.png)

![dice-coefficient-loss](img/b76c098df02cfa5aa09aff082e566191.png)

可以看出，利用 Dice 系数损耗训练的模型收敛速度更快，最终 IOU 精度更高。关于最终的测试预测结果，用 D ice 系数损失训练的模型提供了比用交叉熵损失训练的模型更清晰的分割边缘。

## 结论和进一步阅读

在这个快速教程中，您已经学习了如何建立一个深度学习模型，该模型可以进行端到端的训练，并检测工业应用的缺陷。本文中使用的 DAGM 数据集相对简单，便于快速原型制作和验证。然而，在现实世界中，图像数据可能包含更丰富的上下文，这需要更深入和更复杂的模型来理解，实现这一点的一个简单方法是通过试验增加 CNN 层的内核数量。虽然有其他选择，如[本文](https://arxiv.org/abs/1611.09326)，但作者建议用密集块替换每个 CNN 块，这样在学习复杂的上下文特征时更有能力。

你可以用免费的 GPU 在 Google Colab 上运行[这个笔记本](https://colab.research.google.com/github/Tony607/Industrial-Defect-Inspection-segmentation/blob/master/Industrial_Defect_Inspection_with_image_segmentation.ipynb)来重现这篇文章的结果。

在我的 GitHub 上有源代码。

*   标签:
*   [keras](/blog/tag/keras/) ,
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/automatic-defect-inspection-with-end-to-end-deep-learning/&text=Automatic%20Defect%20Inspection%20with%20End-to-End%20Deep%20Learning) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/automatic-defect-inspection-with-end-to-end-deep-learning/)

*   [←如何使用自定义 COCO 数据集训练检测器 2](/blog/how-to-train-detectron2-with-custom-coco-datasets/)
*   [如何在 Jetson Nano 上以 20+ FPS 运行 SSD Mobilenet V2 物体检测→](/blog/how-to-run-ssd-mobilenet-v2-object-detection-on-jetson-nano-at-20-fps/)