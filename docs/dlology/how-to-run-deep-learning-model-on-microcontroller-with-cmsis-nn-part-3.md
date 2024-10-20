# 如何用 CMSIS-NN 在微控制器上运行深度学习模型(三)

> 原文：<https://www.dlology.com/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn-part-3/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零五个月前

([评论](/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn-part-3/#disqus_thread))

![nn-mcu3](img/507f0ceb471786c59e2dbaca7ecc1659.png)

您已经学习了如何在 ARM 微控制器上运行图像分类模型，以及 CMSIS-NN 框架的基础知识。这篇文章展示了如何从头开始训练和部署一个新的模型。

## 建立并训练一个咖啡馆模型

当选择深度学习框架时，Keras 是我最喜欢的，因为它简单而优雅，但是这次我们将使用 Caffe，因为 ARM 的团队已经发布了两个有用的脚本来为我们生成为 Caffe 模型构建的代码。如果你像我一样是咖啡新手，不用担心。模型结构和训练参数都以易于理解的文本文件格式定义。

Caffe 的安装可能会很有挑战性，尤其是对初学者来说，这就是为什么我用 Caffe 的安装和教程代码创建了这个可运行的 Google Colab [笔记本](https://drive.google.com/file/d/1jqBo2hpFY_xNeFHDf5l1h6q_VTcRTRlQ/view?usp=sharing)。

Caffe 图像分类模型在文件[cifar 10 _ M4 _ train _ test _ small . proto txt](https://gist.githubusercontent.com/Tony607/f3797c737abdedcde20e4d48622f9c95/raw/cifar10_m4_train_test_small.prototxt)中定义，其模型结构图如下所示。它包含三个卷积层，由 ReLU 激活层和 max-pooling 层穿插，最后是一个全连接层，用于生成十个输出类之一的分类结果。

![CIFAR10_CNN](img/f5b54bfef7af092e7828254e2513cb19.png)

在**cifar 10 _ M4 _ train _ test _ small . proto txt**模型定义文件中

*   具有“数据”类型的层必须命名为“数据”，因为我们稍后使用的代码生成脚本是通过名称来定位层的。这一层 产生两个“斑点”，一个是`data`斑点包含图像数据，一个是`label`斑点代表输出类标签。
*   `lr_mult` s 是该层可学习参数的学习速率调整。在我们的例子中，它会将权重学习率设置为与运行时求解器给出的学习率相同，并将偏差学习率设置为两倍大——这通常会导致更好的收敛率。
*   全连通层在 Caffe 中称为an`InnerProduct`层 。
*   图层定义可以包括关于是否以及何时将其包含在网络定义中的规则，如下所示:

```py
layer {
  // ...layer definition...
  include: { phase: TRAIN }
} 
```

在上面的例子中，这一层将只包括`TRAIN`阶段中的。

检查求解器文件[cifar 10 _ small _ solver。prototxt](https://gist.githubusercontent.com/Tony607/79463f2f002768c198a50c05187647ff/raw/cifar10_small_solver.prototxt) 它定义了训练模型的迭代次数，我们用测试数据集评估模型的频率等等。

最后运行脚本[train _ small _ colab . sh](https://gist.githubusercontent.com/Tony607/5569923d09e1c1ce389f2c0958aa6bc9/raw/train_small_colab.sh)将开始训练，当它完成时权重将被保存。在我们的例子中，脚本运行两个解算器文件，对于第二个解算器文件中定义的最后 1000 次训练迭代，学习率降低了 10 倍。最终训练的权重将被保存到文件**cifar 10 _ small _ ITER _ 5000 . caffemodel . H5**表示模型已经被训练了 5000 次迭代。如果你来自 Keras 或其他不同的深度学习框架背景，这里的一次迭代不是说模型已经用整个训练数据集训练过一次，而是一批大小为 100 的训练数据，如**cifar 10 _ M4 _ train _ test _ small . proto txt**中所定义。

很简单，对吧？构建和训练 Caffe 模型不需要编码。

## 量化模型

关于 q 的快速事实量化，

*   Q 将 32 位浮点权重量化为 8 位定点权重进行部署，将模型大小缩小了 4 倍、
*   典型微控制器中定点整数运算比浮点运算运行速度快得多
*   在推理过程中，具有量化整数权重和偏差的模型不会表现出任何性能损失(即准确性)。

由于训练后权重是固定的，我们知道它们的最小/最大范围。使用它们的范围将它们量化或离散化为 256 级。下面是一个快速演示，将权重量化为定点数。假设一个层的权重最初只包含 5 个浮点数。

它输出，

```py
quantization format: 	 Q5.2
Orginal weights:   [-31.63  -6.54   0.45   0.9   31\.  ]
Quantized weights: [-127\.   -26\.    2\.     4\.    124\. ]
Recovered weights: [-31.75  -6.5    0.5    1\.    31\.  ]
```

在本演示中，权重被量化为 Q5.2 定点数格式，这意味着用 8 位表示带符号的浮点数，

*   一位作为符号(正/负)，
*   5 位表示整数部分
*   小数部分 2 位。

Qm.n 格式的**m**和**n**一般可以用前面演示的最小/最大范围来计算，但是如果权重矩阵中包含一个离群值呢？

如果您使用这个新的权重值重新运行前面的脚本，量化 Q8，-1 恢复的权重将如下所示，不太好，小的权重值丢失了！

```py
array([-32.,  -6.,   0.,   0.,  32., 200.])
```

这就是为什么 ARM 团队开发了一个助手脚本，在测试数据集上以最小的准确度损失进行权重量化，这意味着它还会运行模型，以在最初的  计算的值周围搜索最佳 Q m 和 n 值。

[nn _ quantizer . py](https://gist.githubusercontent.com/Tony607/3b7ba419609cb7918394299c5a4a68da/raw/nn_quantizer.py)脚本取模型  定义(cifar 10 _ M4 _ train _ test _ small . proto txt)文件和训练好的模型文件(cifar 10 _ small _ ITER _ 5000 . caffemodel . H5)然后逐层迭代做三件事。

*   量化权重矩阵值
*   量化层的激活值(包括值范围在 0~255 之间的输入图像数据)
*   量化偏差矩阵值

该脚本最终将网络图的连通性、量化参数转储到 pickle 文件中，以供下一步使用。

## 生成代码

![generate_code](img/2d71b8e1c90ff1078412cbb05ce6dcb4.png)

如果有“代码生成器”，谁还需要写代码？`code_gen.py` 从上一步得到量化参数和网络图连通性，生成由 NN 个函数调用组成的代码。【T2

目前支持以下层:卷积、InnerProduct(全连接)、Pooling(最大/平均)和 ReLu。它生成三个文件、、、

1.  `weights.h` : 模型权重和偏差。
2.  `parameter.h`:量化范围，从 Qm 计算的偏置和输出偏移值，n 格式的权重、偏置和激活、
3.  `main.cpp` : 网络代码。

生成器相当复杂，它根据前一篇文章中讨论的各种约束选择最佳的层实现。

## 部署到微控制器

如果模型结构不变，我们只需要更新`weights.h`和`parameter.h`中的那些数据。这些是偏差和输出偏移值，用于替换项目源文件中的值。如果你的项目是基于  [官方 CMSIS-NN cifar10 例子](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN/Examples/ARM/arm_nn_examples/cifar10)像我的，那些值都是在 文件`arm_nnexamples_cifar10_weights.h`里面定义的。

一些定义的命名略有不同，但很容易理清。

现在，在微控制器上构建并运行它！

## 结论和进一步的思考

![use_cases](img/7fa508db1074b4acd25ee2d3260ccea2.png)

到目前为止，您使用的是纯预定义的输入数据来运行神经网络，这在考虑各种传感器选择时并不有趣，仅举几个例子，摄像头、麦克风、加速度计都可以轻松地与微控制器集成，以从环境中获取实时数据。当这个神经网络框架被用来处理这些数据并提取有用的信息时，有无限的可能性。让我们讨论一下你想用这种技术构建什么样的应用。

不要忘记查看这个可运行的 Google Colab [笔记本](https://drive.google.com/file/d/1jqBo2hpFY_xNeFHDf5l1h6q_VTcRTRlQ/view?usp=sharing)来学习本教程。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn-part-3/&text=How%20to%20run%20deep%20learning%20model%20on%20microcontroller%20with%20CMSIS-NN%20%28Part%203%29) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn-part-3/)

*   [←如何用 CMSIS-NN 在微控制器上运行深度学习模型(第二部分)](/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn-part-2/)
*   [如何在浏览器上训练神经网络→](/blog/how-to-train-neural-network-on-browser/)