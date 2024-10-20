# 如何用 CMSIS-NN 在微控制器上运行深度学习模型(下)

> 原文：<https://www.dlology.com/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn-part-2/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零五个月前

([评论](/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn-part-2/#disqus_thread))

![nn-mcu2](img/6546f9a8bdd87f2ebbb784481d4ca38e.png)

[在之前的](https://www.dlology.com/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn/)中，我已经向您展示了如何在 ARM 微控制器上运行图像分类模型，这一次让我们更深入地了解该软件是如何工作的，并熟悉 CMSIS-NN 框架。

## 处理内存限制

源文件`arm_nnexamples_cifar10.cpp`，为了在计算期间将数据放入 192 千字节的 RAM 中，创建了两个缓冲变量，并在各层之间重复使用。

*   `col_buffer`为卷积层存储 im2col(图像到列)输出，
*   `scratch_buffer`存储激活数据(中间层输出)

在下面的代码片段中，你会发现这两个缓冲区是如何应用于 3 个连续的层的。

`arm_convolve_HWC_q7_fast` 函数创建一个卷积层，将 `img_buffer2` 的内容作为输入数据，并将输出到 `img_buffer1` 。 它也使用 `col_buffer` 作为它的内部存储器来运行卷积图像到列的算法。下面的 ReLu 激活层作用于`img_buffer1`本身 ，然后相同的缓冲区将作为下面的 max pool 层的输入，其中该层输出 到 `img_buffer2` 等等。

由于 RAM 空间有限，我们无法慷慨地将一大块内存分配给这两个缓冲区。

相反，我们只分配足够模型使用的内存空间。

为了找出所有卷积层所需的`col_buffer`大小，应用下面的公式。

```py
2*2*(conv # of filters)*(kernel width)*(kernel height)
```

在我们的例子中，我们有三个卷积层和 `conv1` 层需要最大数量的 `2*2*32*5*5 = 3200` 字节缓冲空间来进行其图像到列的计算，因此我们将该数量的空间分配给 `col_buffer` 以及在所有其他卷积层之间共享的。

对于分成两部分的暂存缓冲区，对于给定的图层，一部分可用作输入，另一部分可用作输出。

类似地，它的最大尺寸可以通过迭代所有层来确定。

![scratch_buffer](img/7022803637458e9f38763229b5b68c39.png)

上图显示了一个等效的模型结构，其中我们在所有的层中搜索 `img_buffer1` 和 `img_buffer2` 所需的最大尺寸，然后将两个缓冲区连接在一起，形成总尺寸`scratch_buffe`。

## 选择神经网络层函数

在 CMSIS-NN 中，2D 卷积层有几种选择

1.  arm _ 卷积 _ HWC _ Q7 _ 基本
2.  arm _ 卷积 _HWC_q7_fast
3.  arm _ 卷积 _HWC_q7_RGB
4.  arm _ 卷积 _ HWC _ Q7 _ 快速 _ 非方

它们中的每一个都在不同程度上针对速度和大小进行了优化，但是也有不同的限制。

`arm_convolve_HWC_q7_basic`函数是设计用于任何方形输入张量和权重维度的最基本版本。

`arm_convolve_HWC_q7_fast`函数顾名思义运行速度比前一个快，但要求输入张量通道是 4 的倍数，输出张量通道(滤波器数量)是 2 的倍数。

`arm_convolve_HWC_q7_RGB`专为与输入张量通道等于 3 的卷积而构建，这通常应用于将 RGB 图像数据作为输入的第一个卷积层。

`arm_convolve_HWC_q7_fast_nonsquare`类似于到 `arm_convolve_HWC_q7_fast` ，但可以取非正方形的输入张量。

对于完全连接的层，有两种截然不同的选择，

1.  arm _ 完全连接 _q7
2.  arm _ 完全连接 _q7_opt

第一个使用常规权重矩阵，另一个使用后缀“_opt”来优化速度，但是层的权重矩阵必须预先以交错方式排序。重新排序可以在代码生成器脚本的帮助下无缝实现，我将在下一篇文章中讨论。

### 4.6 倍的速度提升从何而来？

简答，ARM Cortex-M4，M7 微控制器支持特殊的 SIMD 指令，尤其是 16 位乘加(MAC)指令(例如 SMLAD ) 加速矩阵乘法。当你看一看基本的全连接层实现源代码 **arm_fully_connected_q7.c** 时，这个实现会自我检讨。带有 DSP 指令的微控制器在特殊指令下运行更快。

要了解卷积层是如何加速的，必须了解 img2col 的基本原理，img 2 col 使用 im2col()函数将卷积转换为矩阵乘法，该函数以通过矩阵乘法实现卷积输出的方式排列数据。

![img2col](img/a80edd939102bca0b598aed81b0bd001.png)

*[CMU 15-418/618 双方](http://15418.courses.cs.cmu.edu/fall2017/lecture/dnn/slide_023)的功劳。*

**im2col** 通过使用微控制器的 SIMD 特性来提高卷积的并行性，但由于原始图像被放大了 `(numInputChannels * kernel width * kernel height)` 的因子，因此会引入内存开销。

## 结论和进一步的思考

这篇文章介绍了一些基本概念，比如重用缓冲区和 NN 层函数的不同实现，当你用 CMSIS-NN 框架构建一个应用时，你可能会发现这些很有用。在下一节课中，我将向您展示从培训到将模型部署到您的微控制器是多么容易。

*   标签:
*   [教程](/blog/tag/tutorial/)，
*   [深度学习](/blog/tag/deep-learning/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn-part-2/&text=How%20to%20run%20deep%20learning%20model%20on%20microcontroller%20with%20CMSIS-NN%20%28Part%202%29) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn-part-2/)

*   [←如何用 CMSIS-NN 在微控制器上运行深度学习模型(第一部分)](/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn/)
*   [如何用 CMSIS-NN 在微控制器上运行深度学习模型(第三部分)→](/blog/how-to-run-deep-learning-model-on-microcontroller-with-cmsis-nn-part-3/)