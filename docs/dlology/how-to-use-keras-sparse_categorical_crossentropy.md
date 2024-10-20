# 如何使用 Keras 稀疏分类交叉熵

> 原文：<https://www.dlology.com/blog/how-to-use-keras-sparse_categorical_crossentropy/>

###### 发帖人:[程维](/blog/author/Chengwei/) 4 年 2 个月前

([评论](/blog/how-to-use-keras-sparse_categorical_crossentropy/#disqus_thread))

![colors](img/ca9e3f7a738f1b91634c32cf27704827.png)

在这个快速教程中，我将向您展示两个简单的示例，在编译 Keras 模型时使用`sparse_categorical_crossentropy`损失函数和`sparse_categorical_accuracy`度量。

## 例子一——MNIST 分类

作为多类、单标签分类数据集之一，任务是将手写数字的灰度图像(28 像素×28 像素)分类到它们的十个类别(0 到 9)。让我们构建一个 Keras CNN 模型来处理它，最后一层应用了“softmax”激活，它输出一个十个概率分数的数组(总和为 1)。每个分数将是当前数字图像属于我们的 10 个数字类之一的概率。

对于这种输出形状为(None，10)的模型，传统的方法是将目标输出转换为一位热码编码数组以匹配输出形状，但是，借助于`sparse_categorical_crossentropy`loss函数，我们可以跳过这一步，保留整数作为目标。

你需要的就是把 `categorical_crossentropy` 换成`sparse_categorical_crossentropy`这样编译模型。

之后，你可以用整数目标训练模型，也就是像一样的一维数组

```py
array([5, 0, 4, 1, 9 ...], dtype=uint8)
```

请注意，这不会影响模型输出形状，它仍会为每个输入样本输出十个概率得分。

## 示例双字符级序列到序列预测

我们将根据威廉·莎士比亚的综合作品训练一个模型，然后用它来创作一部风格相似的戏剧。

文本 blob 中的每个字符首先通过调用 Python 的内置 `ord()` 函数转换成一个整数，该函数返回一个表示字符的整数作为其 ASCII 值。例如，`ord('a')` 返回整数`97`。因此，我们有一个整数列表来表示整个文本。

给定序列长度为 100 的移动窗口，模型学习预测未来一个时间步长的序列。换句话说，给定序列中时间步长 T0~T99 的字符，该模型预测时间步长 T1~T100 的字符。

让我们在 Keras 中构建一个简单的序列对序列模型。

我们可以进一步可视化模型的结构，以分别理解其输入和输出形状。

![training_model](img/53f4f7d5f4bb15e865d79f4c6bbcf83d.png)

即使模型具有三维输出，当用损失函数 `sparse_categorical_crossentropy` ，编译时，我们可以将训练目标作为整数序列。与前面的例子类似，如果没有`sparse_categorical_crossentropy`中的的帮助，首先需要将输出的整数转换成 one-hot 编码形式以适应模型。

训练模式为，

*   无状态
*   seq_len =100
*   batch_size = 128
*   模型输入形状:(批处理大小，序列长度)
*   模型输出形状:(批处理大小，序列长度，最大令牌数)

一旦训练好模型，我们就可以让它“有状态”，一次预测五个字符。通过使其有状态，LSTMs 在一个批次中的每个样本的最后状态将被用作下一个批次中的样本的初始状态，或者简单地说，一次预测的那五个字符和下一个预测批次是一个序列中的字符。

预测模型加载训练好的模型权重并一次预测五个字符，

*   宏伟威严的
*   seq_len =1，一个字符/批次
*   batch_size = 5
*   模型输入形状:(批处理大小，序列长度)
*   模型输出形状:(批处理大小，序列长度，最大令牌数)
*   需要在预测之前调用 `reset_states()` 来重置 LSTMs 的初始状态。

关于模型的更多实现细节，请参考我的 [GitHub 库](https://github.com/Tony607/keras_sparse_categorical_crossentropy)。

## 结论和进一步阅读

本教程探讨了两个例子使用 `sparse_categorical_crossentropy` 到保持整数为字符/多类分类标签，而不转换为单热标签。因此，当标签是整数时，模型的输出将是类似 softmax one-hot 的形状。

要了解[keras . back end . sparse _ category _ cross entropy](https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/keras/backend.py#L3582)和[sparse _ category _ accuracy](https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/keras/metrics.py#L587)的实际实现，可以在 TensorFlow repository 上找到。别忘了在 [my GitHub](https://github.com/Tony607/keras_sparse_categorical_crossentropy) 上下载本教程的源代码。

*   标签:
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)，
*   [深度学习](/blog/tag/deep-learning/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-use-keras-sparse_categorical_crossentropy/&text=How%20to%20use%20Keras%20sparse_categorical_crossentropy) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-use-keras-sparse_categorical_crossentropy/)

*   [←如何用生成对抗网络进行 Keras 中的新颖性检测(下)](/blog/how-to-do-novelty-detection-in-keras-with-generative-adversarial-network-part-2/)
*   [如何免费用 TPU 训练速度快 20 倍的 Keras 车型→](/blog/how-to-train-keras-model-x20-times-faster-with-tpu-for-free/)