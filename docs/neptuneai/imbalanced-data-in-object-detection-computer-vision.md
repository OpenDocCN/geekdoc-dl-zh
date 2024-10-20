# 目标检测计算机视觉项目中的不平衡数据

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/imbalanced-data-in-object-detection-computer-vision>

数据科学从业者面临的一个典型问题是**数据不平衡问题**。它困扰着每一个其他的 ML 项目，我们在处理一些分类问题时都面临着它。

[数据不平衡](https://web.archive.org/web/20221201164647/https://machinelearningmastery.com/what-is-imbalanced-classification/)可以有几种类型。例如，最经常讨论的问题之一是阶级不平衡。在收集真实世界的数据时，拥有一个类平衡的数据集的可能性真的很低。数据点数量较多的类往往会在模型中产生偏差，如果使用简单的准确度分数，这些有偏差的模型可能会被误解为表现良好。

比方说，如果在一个测试集中，97%被诊断的患者没有癌症，3%有。我们的有偏模型预测没有患者患癌症，准确率为 97%。因此，正确的准确性度量是处理类不平衡问题的重要方面之一。

同样，根据问题的性质和数据集的原始程度，数据集中可能会出现其他类型的不平衡。在本文中，**我们将在对象检测问题的背景下研究这些不同的不平衡，以及如何解决它们**。

## 对象检测模型中的不平衡

对象检测是在定位图片中感兴趣的对象的同时，将它分类到某个类别中。在深度学习出现之前，第一代对象检测算法主要依赖于手工制作的特征和线性分类器。随着深度学习，虽然对象检测取得了显著的进步，但新的问题也随之产生，即不平衡问题。

图像中有几个输入属性，如图像中的不同对象、前景和背景区域、不同任务的贡献(如网络中的分类和回归)等。如果这些属性的分布不均匀，就会导致不平衡问题。图 1 是对象检测模型数据不平衡的一个例子。

![Figure1_foreground_background_imbalance](img/217c42e1a230256a7b53949e329dc458.png)

*Figure 1 – Foreground Background Imbalance. | Source: Author*

现在让我们详细讨论对象检测中这些不同类型的不平衡。

## 阶级不平衡

如果数据集中一个对象的实例数多于另一个，就会导致类不平衡。在对象检测中，我们可以将图像中的区域分为前景和背景，如图 2 所示。

![Figure2_area_background_foreground](img/1387fd7fee4e2304390e4623a7ae91a6.png)

*Figure 2 – We can see the area of background is much more than that of the foreground. | Source: Author*

从对象检测的角度来看，类别不平衡可以细分为两种类型——前景-背景不平衡和前景-前景不平衡。

1.  **前景-背景不平衡**

一般来说，背景的面积比前景的面积大得多，因此大多数边界框被标记为背景。

![Figure3_retinanet_anchors_mscoco_background_vs_foreground](img/4303f73d2397b6be6b7efc4d2142058c.png)

*Figure 3 – These are the RetinaNet anchors on the MS-COCO dataset for background and foreground. | [Source](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1909.00169.pdf)*

在图 3 中，您可以看到背景锚和前景锚之间的巨大差异。现在，这很容易导致一个有偏见的模型给出更多的假阴性。

2.  **前景-前景不平衡**

在这种情况下，背景锚点不是感兴趣的对象。在图像中，一些对象可能被过度表现，而另一些对象可能被不足表现。因此，当一个类支配另一个类时，这被称为前景-前景不平衡。例如在图 1 中，车辆的数量决定了人的数量。

![Figure4_frequency_graph_all_classes](img/7bf92c3503fd41df52ead66dc75a2360.png)

*Figure 4 – Frequency graph of 80 object categories in MS-COCO dataset normalized with the total number of images. | [Source](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1909.00169.pdf)*

在图 4 中，我们可以观察到前景-前景不平衡。

### 解决阶级不平衡

有几种有效的方法来处理这个问题。让我们从最简单的方法开始。

1.  **硬采样**

硬采样是直接丢弃某些边界框，这样它们对训练没有影响，有不同的启发式方法来这样做。

因此，如果 w 是赋予样本的权重，w 将定义其对训练的贡献。对于硬采样，要么 w=0，要么 w=1。

在训练时，从每幅图像的一批样本中，选择一定数量的阴性和阳性样本，其余的被丢弃。

我们不是随机抽样，而是从上到下挑选否定样本的列表(最上面的是最有可能是否定的)。我们选择负面的例子(最上面的),使得负面与正面的比例不超过 3:1。这种方法背后的假设是，具有更多硬样本(其给出更高的损失)的训练算法最终会导致更好的性能。

![Figure5_random_vs_hard_negative_mining](img/4f5e432cdb0ec8052d9f27e4096b3cfa.png)

*Figure 5 – Random Sampling vs Hard Negative Mining, Hard negative mining selects boxes with more negative confidence | Source: Author*

负样本被分配到 K 个箱中，然后从每个箱中均匀地选择。这促进了具有给出高损耗值的高 iou 的样本。这也遵循与硬负挖掘相同的假设。

2.  **软采样**

与硬采样不同，在硬采样中，正负样本的权重被二进制化(它们要么有贡献，要么没有贡献)，在软采样中，我们基于一些方法来调整每个样本的贡献。其中之一就是焦损失。

使用聚焦损失是软采样和处理类别不平衡的最佳方法之一。它赋予硬样品更大的权重。

***FL(p[t])=-α(1-p[t])^λlog(p[t])***

在上面的损失方程中，p [t] 是样本的估计概率。现在，如果 p [t] ≃1，损失将趋于零(FL≃0)，因此只考虑硬例子。

一个很好的焦损 PyTorch 实现可以在[这里](https://web.archive.org/web/20221201164647/https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py)找到。

3.  **生成方法**

取样方法减少和增加样品的重量。然而，生成方法可以生成新的人工样本来平衡训练。生成对抗网络(GANs)的出现推动了这种方法的发展。

论文[对抗性快速 RCNN](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1704.03414.pdf) 展示了如何分两个阶段使用。获取来自数据集的图像，然后分两个阶段生成变换后的图像。

1.  ASDN:对抗性空间流失网络造成图像闭塞
2.  ASTN:对抗性空间变换网络将频道从-10 度旋转到 10 度，以创建更难的示例。

![Figure6_generative_network_occlusion](img/c9a206fd2dbf80d13b19a48d36cb2c65.png)

*Figure 6 – This is the work of ASDN, which shows the occluded areas in the images. | [Source](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1704.03414.pdf)*

图 6 显示了遮挡区域对分类器的重要性。进行这些改变将允许生成更硬的样本和更健壮的分类器。

Caffe 中的官方实现文件可以在这里找到，很容易理解。

本文主要研究硬正生成。但是生成方法也可以用于前景-背景类别不平衡。

## 规模失衡

规模不平衡是训练目标检测网络时面临的另一个关键问题。比例失衡的发生是因为某一范围的物体尺寸或某些特定级别(高/低级别)的特征被过度表现或表现不足。

规模失衡可细分为盒子级规模失衡或特征级规模失衡。

1.  **箱级秤不平衡**

在某些对象的数据集分布中，大小可能被过度表示。在这种情况下，经过训练的模型会导致不良的感兴趣区域偏向于过度表示的尺寸。

![Figure7_normalized_width_height_area_across_datasets](img/ea646d2d5e5a8cead7b57901a01b141a.png)

*Figure 7 – Normalized width, height and area across different datasets. | [Source](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1909.00169.pdf)*

图 7 清楚地显示了小尺寸物体的较高频率。

2.  **特征等级比例失调**

低级和高级特征的不平衡分布会产生不一致的预测。我们举个例子来理解这一点。快速 RCNN 是一种相当流行的对象检测方法，它使用特征金字塔网络(FPN)进行区域提议。

![Figure8_feature_pyramid_netword](img/2d4bcd49998747ef82d391148430b285.png)

Figure 8 – Feature Pyramid Network (FPN). | [Source](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1612.03144v2.pdf)

在图 8 中，我们看到了 FPN 的基本架构。当我们将输入图像传递到自下而上的路径并进行卷积时，初始层由低级特征组成，后续层具有更复杂的高级特征。

请注意，自上而下路径中的每一层都提出了感兴趣的区域。所以所有的层都应该有平衡数量的低级和高级特性。但事实并非如此，正如我们在自上而下的路径中所看到的，我们将在最顶层拥有更多高级功能，在最底层拥有更多低级功能。因此，这可能导致不一致的预测。

### 解决规模失衡

为了解决规模失衡，你应该更多地处理网络架构的变化，而不是像我们在类失衡中所做的那样处理损失函数的变化。我们讨论了两种类型的规模不平衡，让我们看看如何解决它们。

1.  **箱级秤不平衡**

正如您在图 9 中看到的，在所有卷积和池化完成后，早期的对象检测器用于预测最后一层的感兴趣区域。这没有考虑到边界框比例的多样性。

![Figure9_RoIprediction](img/7c5a540d18c1f85f2d9128eb7f141ef9.png)

*Figure 9 – An example of basic RoI prediction network. | [Source](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1909.00169.pdf)*

这就是特征金字塔发挥作用的地方。FPN(图 8)包括自下而上的路径、自上而下的路径和横向连接。它给出了从高层到低层的丰富语义。

任何需要区域提案网络的地方都可以使用 FPN。你可以在这里阅读更多关于 FPN 的信息。

在 PyTorch，你可以简单地从 torchvision.ops 获得 FPN；

```py
from torchvision.ops import FeaturePyramidNetwork
net = FeaturePyramidNetwork([10, 20, 30], 5)
print(net)
```

输出:

```py
FeaturePyramidNetwork(
  (inner_blocks): ModuleList(
    (0): Conv2d(10, 5, kernel_size=(1, 1), stride=(1, 1))
    (1): Conv2d(20, 5, kernel_size=(1, 1), stride=(1, 1))
    (2): Conv2d(30, 5, kernel_size=(1, 1), stride=(1, 1))
  )
  (layer_blocks): ModuleList(
    (0): Conv2d(5, 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Conv2d(5, 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (2): Conv2d(5, 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)

```

FPN 也有不同的变体，我们将在下一节讨论。

2.  **特征等级比例失调**

我们之前讨论过 FPN 的缺点是如何导致功能级别不平衡的。让我们来看看有助于解决这个问题的网络之一；路径聚合网络(PANet)。

你可以在这里看到 PANet 的建筑。

![Figure10_PANet_architecture](img/0ac7a07f97b7a24d85373744bb4befc3.png)

*Figure 10 – PANet Architecture. | [Source](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1909.00169.pdf)*

除了 FPN 之外，PANet 还有两个组件，可以生成增强的功能。自底向上的路径增强允许低级别特征被包括在预测中，第二个是 PANet 使用自适应特征池将 ROI 关联到每个级别。

## 空间不平衡

边界框有某些空间方面，包括并集上的交集(IoU)、形状、大小和位置。很明显，在现实世界的例子中，这些属性会不平衡，这会影响训练。

现在，我们将了解边界框的这些不同属性如何影响训练:

1.  **回归损失不平衡**

对象检测网络中的回归损失通常用于收紧图像中对象的边界框。预测边界框和基础真值位置的微小变化都会导致回归损失的剧烈变化。因此，为回归选择一个稳定的损失函数来处理这种不平衡是非常重要的。

![Figure11_L1,L2_loss_calc](img/940d6141327b43d0d704c567265e00ac.png)

*Figure 11 – L1, L2 loss calculation for bounding boxes. The blue box represents ground truth and the rest are predictions. | [Source](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1909.00169.pdf)*

在图 11 中，我们可以观察到 L1 和 L2 损耗在相同 IoU 值下的不同表现。

2.  **欠条分配不平衡**

当边界框的 IoU 分布(跨数据集)偏斜时，称为 IoU 分布不平衡。

![Figure12_IoU_distribution](img/1809655e39ee5a9ada0d408f3f5b02e9.png)

*Figure 12 – This is IoU distribution of anchors on MS-COCO dataset in RetinaNet. | [Source](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1909.00169.pdf)*

IoU 分布也可以影响两级网络，例如，当在更快的 RCNN 的第二级中处理 IoU 分布不平衡时，它显示出显著的改善。

3.  **物体位置不平衡**

我们使用遵循滑动窗口方法的卷积网络。每个窗口都分配有一定比例的锚点，赋予每个窗口同等的重要性。然而，跨图像的对象位置分布是不均匀的，从而导致不平衡。

![Figure13_Object_location_distribution](img/451e9bddb87200d31ac7a4b3b09882de.png)

Figure 13 – Above are the object location distribution for different datasets, starting from left; Pascal-Voc, MS-Coco, Open Images and Objects 365\. | [Source](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1909.00169.pdf)

我们可以在图 13 中看到，我们可以观察到数据集中的对象位置分布并不均匀，而是正态分布(高斯分布)。

### 解决空间失衡

下面就来逐一说说空间不平衡的三个不平衡的解决方案。

1.  **回归损失不平衡**

某些损失函数有助于稳定回归损失。请看下表:

| 损失函数 | 说明 |
| --- | --- |
|  | 

用于早期的深层物体探测器。对小错误稳定，但严重惩罚异常值。

 |
|  | 

因微小误差而不稳定。

 |
|  | 

基线回归损失函数。与 L1 的损失相比，对我们的员工来说更加稳健。

 |
|  | 

与平滑 L1 损失相比，增加内点的贡献。

 |
|  | 

基于 KL 散度预测关于输入包围盒的置信度。

 |
|  | 

使用 IoU 的间接计算作为损失函数。

 |
|  | 

固定 IoU 定义中输入框的所有参数，除了在反向传播过程中估计其梯度的参数。

 |
|  | 

根据 IoU 输入的最小外接矩形扩展 IoU 的定义，然后直接使用 IoU 和扩展 IoU，称为 GIoU，作为损失函数。

 |
|  | 

扩展了 IoU 的定义，增加了关于长宽比差和两个方框间中心距离的附加惩罚项。

 |

在表中，我们可以看到不同的损失函数。我个人更喜欢平稳的 L1 损失。它很健壮，在大多数情况下都能很好地工作。

2.  **欠条分配不平衡**

有些架构可以解决 IoU 分配不平衡的问题。

1.  级联 RCNN——[论文的作者](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1906.09756.pdf)在级联流水线中创建了三个具有不同 IoU 阈值的检测器。并且每个检测器使用来自前一级的盒而不是重新采样它们的事实表明，IoU 分布可以从左偏转变为均匀甚至右偏。
2.  分级镜头检测器——网络法在回归盒子后运行其分类器，而不是使用级联管道。这使得 IoU 分布更加平衡。您可以在[论文](https://web.archive.org/web/20221201164647/https://openaccess.thecvf.com/content_ICCV_2019/papers/Cao_Hierarchical_Shot_Detector_ICCV_2019_paper.pdf)中阅读该架构的更多细节。
3.  IoU Uniform RCNN——这种[架构](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1912.05190.pdf)通过向回归分支而不是分类分支提供统一边界框的方式增加了变量。

所有这些网络已被证明在处理 IoU 分布方面是有效的，提高了检测的整体性能。

**3。物体位置不平衡**

一个叫做[导向锚定](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1901.03278.pdf)的概念在一定程度上处理了这个问题。这里，来自特征金字塔的每个特征图在到达预测层之前都通过锚定引导模块。导向锚定模块由两部分组成；锚生成部分预测锚形状和位置，特征适应部分将具有锚信息的新特征地图应用于由特征金字塔生成的原始特征地图。

![Figure15_guided_anchoring_network](img/e02d5602c8b4abd6940b8a2be4f98d6f.png)

*Figure 15 – Guided Anchoring Module. | [Source](https://web.archive.org/web/20221201164647/https://arxiv.org/pdf/1901.03278.pdf)*

## 客观不平衡

正如我们之前讨论的，对于对象检测，有两个问题我们需要优化；分类和回归。现在，同时计算这两个问题的损失会带来一些问题:

1.  由于梯度范围的差异，其中一个任务可以支配训练。
2.  在不同任务中计算的损失范围的差异可以不同。例如，为回归而收敛的损失函数可能在 1 左右，而为分类而收敛的损失函数可能在 0.01 左右。
3.  任务的难度可以不同。例如，对于某个数据集，达到分类收敛可能比回归更容易。

### 解决客观不平衡

1.  **卡尔**

一种称为分类感知回归损失(CARL)的方法假设分类和回归是相关的。

*L[C](x)= C[I]L1[S](x)*

这里 L1S 是用作回归损失函数的平滑 L1 损失。ci 是基于由分类层计算的分类概率的分类因子。这样，回归损失提供了具有梯度信号的分类，从而部署了分类和回归之间的相关性。

2.  **引导损失**

回归损失仅由前景实例组成，并且仅通过前景类别的数量进行归一化，这一事实可用作分类损失的归一化器。

引导损失通过将损失的总幅度考虑为–,对分类部分进行加权

wregLregLcls

## 一些有用的提示和技巧

正如我们之前所讨论的，类别不平衡是训练对象检测模型时最令人头疼的问题之一。除了改变损失函数和修改架构之外，我们可以使用一些常识和玩弄数据集来减少类的不平衡。在本节中，我们将尝试通过一些数据预处理来缓解这个问题。

让我们考虑一个带有地理数据的图像(如图 16 所示)。我们想要识别建筑物、树木、汽车等物体。图 16 中的图像是我们的对象检测网络的训练数据集中的图像之一。

![Figure16_satellite_picture](img/885a8b7b0ce5e2443e71023196178413.png)

*Figure 16 – A Satellite picture consisting of buildings, canopy, river, cars, etc. | [Source](https://web.archive.org/web/20221201164647/https://earth.google.com/web/search/Spa,+Belgium/@50.44795575,30.51214238,186.93009383a,5379.28236026d,35y,360h,0t,0r/data=CigiJgokCVAWE6UtZ0lAERAzNE6cBUlAGbwpwG5UDhdAIXSS7ZbWrwtA)*

这是一张高分辨率的卫星照片。现在我们要识别建筑物、汽车、树木等。然而，我们可以清楚地看到这幅画的大部分是由建筑物组成的。你会在几乎所有现实生活的数据集中看到巨大的阶级不平衡。那么我们如何解决这个问题呢？

![Figure16_satellite_picture_zoom](img/4cf9bf5a813f63dbe0f6e4298f680aa5.png)

*Figure 17 – A zoomed-in picture from figure 16\. | [Source](https://web.archive.org/web/20221201164647/https://earth.google.com/web/search/Spa,+Belgium/@50.44795575,30.51214238,186.93009383a,5379.28236026d,35y,360h,0t,0r/data=CigiJgokCVAWE6UtZ0lAERAzNE6cBUlAGbwpwG5UDhdAIXSS7ZbWrwtA)*

首先，如果分辨率特别高，我们可以从中提取如图 17 所示的详细图像，然后尝试以下任何方法。

### 合并类

如果可能的话，合并相似的类。例如，如果我们在数据集中标注了“汽车”、“卡车”和“公共汽车”，将它们合并到“汽车”类别中会增加实例的数量(对于标签“汽车”)并减少类的数量。所有三个标签的平均像素直径几乎相同。

这是一个简单而通用的例子，但是理想情况下，这应该由具有领域知识的人来完成。

### 分割图像

您可以将高分辨率图像分成一定数量的小块，并使用每个小块作为检测模型的输入，而不是缩小高分辨率图像。这将有助于网络学习更小的物体。

虽然缩小会减少训练时间，但也会使小对象消失。切片数据集肯定会有所帮助。

![Figure17_split_image_15](img/d628bb21156e3dc9a23bccec056f35ac.png)

*Figure 18 – Splitted high-resolution image, each tile in the grid can be used as an input for the object detection model. | [Source](https://web.archive.org/web/20221201164647/https://earth.google.com/web/search/Spa,+Belgium/@50.44795575,30.51214238,186.93009383a,5379.28236026d,35y,360h,0t,0r/data=CigiJgokCVAWE6UtZ0lAERAzNE6cBUlAGbwpwG5UDhdAIXSS7ZbWrwtA)*

```py
import cv2
from matplotlib.pyplot import plt

a, b = 8, 10

img = cv2.imread('image_path')
M = img.shape[0]//a
N = img.shape[1]//b
tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]

plt.imshow(tiles[0])

```

你可以用这段简单的代码来分割你的图片。

### 欠采样和过采样

欠采样和过采样是处理类不平衡的最基本的方法。但以防万一，如果你有足够的时间和数据，你可以尝试这些东西。

#### 欠采样

欠采样是指从过度表示的类中丢弃数据点。

现在您已经在图 18 中制作了图块。您可以丢弃仅包含建筑物的图块，这将平衡数据集中的其他对象，如天篷、汽车等。

![Figure19_undersampling](img/c616e6fdb76d08177134596af505a260.png)

*Figure 19 – Under sampling. | Source: Author*

缺点是，这可能会导致一个坏的模型。建筑物的概化程度较低，当您根据模型进行推断时，可能无法识别特定类型的建筑物。

#### 过采样

在过采样中，你为代表性不足的类生成合成数据。在上一节中，我们已经看到了一个使用生成算法(如 GANs)的例子。

我们也可以通过简单地使用 CV 算法来进行过采样。例如，你有一个充满汽车的瓷砖，你扩大它，但旋转，垂直和水平翻转，改变对比度，添加包装，剪切图像等。

我们在 PyTorch 中有可用的转换，或者您可以只使用 OpenCV 来完成这样的任务。

![Figure20_oversampling](img/ac7e16fd741461a622732a17432a4d89.png)

*Figure 20 – Over sampling. | Source: Author*

过采样的缺点是，你的模型可能会过拟合你过采样的数据。

## 未决问题和正在进行的研究

深度学习网络中的不平衡所产生的问题是一个相当新的研究课题，随着网络越来越复杂，新的不平衡问题越来越频繁地被发现。

在回归损失不平衡中，我们看到了许多损失函数解决不同类型的问题，这仍然是一个开放的问题，人们正在努力将不同损失函数的所有优势整合为一个。

像这样需要额外工作的例子有很多。仍然没有一个统一的方法来处理所有不同的不平衡。不同架构的工作正在进行中。

好消息是有大量的资源可以阅读和解决这些不平衡。

### 进一步阅读