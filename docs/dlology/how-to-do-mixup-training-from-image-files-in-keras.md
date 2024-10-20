# 如何从 Keras 中的图像文件进行混音训练

> 原文：<https://www.dlology.com/blog/how-to-do-mixup-training-from-image-files-in-keras/>

###### 发帖人:[程维](/blog/author/Chengwei/) 3 年 11 个月前

([评论](/blog/how-to-do-mixup-training-from-image-files-in-keras/#disqus_thread))

![mixology](img/2482aa7657bc2e8823b5054bd9b1f997.png)

之前，我们在 Keras 中介绍了[一系列技巧](https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/)来提高卷积网络的图像分类性能，这一次，我们将更仔细地看看最后一个叫做 mixup 的技巧。

## 什么是mix训练？

论文[混音](https://arxiv.org/pdf/1710.09412.pdf)[:BEYONDE经验性的RISKM](https://arxiv.org/pdf/1710.09412.pdf)[去伪化](https://arxiv.org/pdf/1710.09412.pdf)为缩放和旋转等传统的图像增强技术提供了替代方案。B y 通过两个现有示例的加权线性插值形成新示例。

![mixup-function](img/081ab9951aad92fab864a2128d51451c.png)

(x[I]；y[I])(x[j]；y[j])是从我们的训练数据中随机抽取的两个例子，λ∈【0； 1】，实际中 λ是从贝塔分布中随机抽样的，即 贝塔(α；α ) 。

【α∈【0。1； 0。【4】导致性能提高，较小的 α产生较少的混合效应，反之，对于较大的 α 、导致欠拟合。 

正如您在下图中看到的，给定一个小的α = 0.2 ，beta 分布采样更多更接近 0 和 1 的值，使得混合结果更接近两个示例中的任何一个。

![beta](img/30dbcca1c7d64a2fe7e843f5e01cb866.png)

## 混搭训练有什么好处？

虽然像 Keras [ImageDataGenerator](https://keras.io/preprocessing/image/#imagedatagenerator-class) 类中提供的那些传统的数据扩充一直导致改进的泛化，但是该过程依赖于数据集，因此需要使用专家知识。

此外，数据扩充并不对不同类别的实例之间的关系进行建模。

反之，

*   Mixup 是一个数据不可知的数据扩充例程。
*   它使决策边界从一个类线性过渡到另一个类，提供了更平滑的不确定性估计。
*   它减少了对腐败标签的记忆，
*   增加了对对抗实例的鲁棒性， 稳定了生成对抗网络的训练。

## Keras 中的 Mixup 图像数据生成器

试图给 mixup 一个旋转？让我们实现一个图像数据生成器，它从文件中读取图像，并开箱即用Keras`model.fit_generator()`。

混合生成器的核心由一对迭代器组成，这些迭代器从目录中随机抽取图像，一次一批，混合在`__next__`方法中执行。

然后，您可以创建用于拟合模型的训练和验证生成器，请注意，我们在验证生成器中没有使用 mixup。

我们可以用 Jupyter 笔记本中的以下片段来可视化一批混淆的图像和标签。

下图说明了 mixup 的工作原理。

![mixup-example](img/795b81ca349622bc967a1e8bdfe42873.png)

## 结论和进一步的想法

你可能认为一次混合两个以上的例子可能会导致更好的训练，相反，三个或更多个例子的组合与从β分布的多元泛化中采样的权重并不能提供进一步的增益，而是增加了 混合的计算成本。此外，仅在具有相同标号的输入之间进行插值不会导致的性能增益。T12

在我的 [Github](https://github.com/Tony607/keras_mixup_generator) 上查看完整的源代码。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-do-mixup-training-from-image-files-in-keras/&text=How%20to%20do%20mixup%20training%20from%20image%20files%20in%20Keras) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-do-mixup-training-from-image-files-in-keras/)

*   [←Keras 中卷积神经网络图像分类的锦囊妙计](/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/)
*   [如何利用 CPU 和英特尔 OpenVINO 将 Keras 模型推理运行速度提高 x3 倍→](/blog/how-to-run-keras-model-inference-x3-times-faster-with-cpu-and-intel-openvino-1/)