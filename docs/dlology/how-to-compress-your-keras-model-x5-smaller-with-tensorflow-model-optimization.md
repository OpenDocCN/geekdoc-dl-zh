# 如何使用 TensorFlow 模型优化将 Keras 模型 x5 压缩得更小

> 原文：<https://www.dlology.com/blog/how-to-compress-your-keras-model-x5-smaller-with-tensorflow-model-optimization/>

###### 发帖人:[程维](/blog/author/Chengwei/) 3 年 7 个月前

([评论](/blog/how-to-compress-your-keras-model-x5-smaller-with-tensorflow-model-optimization/#disqus_thread))

![prune](img/f063505f9e2dc46c94a5fa780928513e.png)

本教程将演示如何使用 [TensorFlow 模型优化](https://www.tensorflow.org/model_optimization)将 Keras 模型的大小减少 5 倍，这对资源受限环境中的部署尤为重要。

来自官方 TensorFlow 模型优化文档。 权重剪枝是指剔除权重张量中不必要的值。我们将神经网络参数的值设置为零，以移除我们估计的神经网络各层之间不必要的连接。这是在训练过程中完成的，以允许神经网络适应这些变化。

这里有一个如何采用这种技术的分解。

1.  一如既往地训练 Keras 模型以达到可接受的精度。
2.  使 Keras 层或模型准备好被修剪。
3.  创建修剪计划，并为更多时期训练模型。
4.  通过从模型中剥离修剪包装器来导出修剪后的模型。
5.  使用可选量化将 Keras 模型转换为 TensorFlow Lite。

## 修剪你预先训练好的 Keras 模型

您的预训练模型已经达到理想的精度，您希望在保持性能的同时缩减其大小。修剪 API 可以帮助你做到这一点。

使用剪枝 API，安装`tensorflow-model-optimization` 和 `tf-nightly` 软件包。

然后你可以加载你之前训练好的模型，让它“可修剪”。基于 Keras 的 API 可以在单个层或整个模型中应用。因为您已经预先训练了整个模型，所以将修剪应用到整个模型更容易。该算法将应用于所有能够进行权重修剪的图层。

对于修剪时间表，我们从 50%的稀疏度开始，并逐渐训练模型以达到 90%的稀疏度。X%的稀疏性意味着 X%的权重张量将被删除。

此外，我们在每个修剪步骤之后给模型一些时间来恢复，因此修剪不会在每个步骤上都发生。我们将修剪设置为 100。类似于修剪盆景，我们逐渐修剪它，以便树可以充分愈合修剪过程中产生的伤口，而不是在一天内砍掉 90%的树枝。

鉴于模型已经达到了令人满意的精度，我们可以立即开始修剪。因此，我们在这里将`begin_step`设置为 0，并且只训练另外四个纪元。

在给定训练样本数、批量大小和总训练期的情况下，计算结束步骤。

如果您在`new_pruned_model`摘要中发现更多可训练参数，请不要惊慌，这些参数来自我们稍后将移除的修剪包装器。

现在让我们开始训练和修剪模型。

修剪后的模型的测试损失和准确性应该与原始的 Keras 模型相似。

## 导出修剪后的模型

那些 剪枝包装器可以这样轻松的去掉，之后参数总数应该和你原来的模型一样。

现在你可以通过与零比较来检查权重被削减的百分比。

这是结果，正如你所看到的，90%的卷积、密集和批量规范层的权重被修剪。

| 名字 | 段落总计 | 修剪百分比 |
| conv2d _ 2/内核:0 | eight hundred | 89.12% |
| conv2d_2/bias:0 | Thirty-two | 0.00% |
| 批处理 _ 规范化 _ 1/伽玛:0 | Thirty-two | 0.00% |
| batch_normalization_1/beta:0 | Thirty-two | 0.00% |
| conv2d _ 3/内核:0 | Thirty-two | 0.00% |
| conv2d_3/bias:0 | Thirty-two | 0.00% |
| 密集 _ 2/内核:0 | Fifty-one thousand two hundred | 89.09% |
| 密集 _ 2/偏差:0 | Sixty-four | 0.00% |
| 密集 _ 3/内核:0 | Three million two hundred and eleven thousand two hundred and sixty-four | 89.09% |
| 密集 _ 3/偏差:0 | One thousand and twenty-four | 0.00% |
| batch _ normalization _ 1/移动平均值:0 | Ten thousand two hundred and forty | 89.09% |
| batch _ normalization _ 1/moving _ variance:0 | Ten | 0.00% |

现在，只需使用通用文件压缩算法(如 zip ), Keras 模型将缩小 5 倍。

这是你得到的，5 倍小的模型。

压缩前修剪模型的大小: **12.52 Mb**
压缩后修剪模型的大小: **2.51 Mb**

## Convert Keras model to TensorFlow Lite

Tensorflow Lite 是一种可用于部署到移动设备的示例格式。要转换成 Tensorflow Lite 图形，需要使用如下的`TFLiteConverter`:

然后，您可以使用类似的技术来压缩`tflite`文件，并将大小缩小 x5 倍。

训练后量化将权重转换为 8 位精度，作为从 keras 模型到 TFLite 的平面缓冲区的模型转换的一部分，导致模型大小又减少了 4 倍。在调用`convert()`之前，只需将下面一行代码添加到前面的代码片段中。

与原始 Keras 模型的 12.52 Mb 相比，压缩的 8 位 tensorflow lite 模型仅占用 0.60 Mb，同时保持相当的测试精度。这完全是 x16 倍的尺寸缩减。

您可以像这样评估转换后的 TensorFlow Lite 模型的准确性，其中您向`eval_model`提供测试数据集。

## 结论和进一步阅读

在本教程中，我们向您展示了如何使用 TensorFlow 模型优化工具包权重修剪 API 创建*稀疏模型。现在，这允许您创建占用磁盘空间少得多的模型。得到的模型也可以更有效地实现以避免计算；未来，TensorFlow Lite 将提供这样的功能。*

 *查看官方 [TensorFlow 模型优化](https://www.tensorflow.org/model_optimization)页面和他们的 [GitHub 页面](https://github.com/tensorflow/model-optimization)了解更多信息。

#### *这篇文章的源代码可以在[我的 Github](https://github.com/Tony607/prune-keras) 上获得，也可以在 [Google Colab 笔记本](https://colab.research.google.com/github/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/g3doc/guide/pruning/pruning_with_keras.ipynb)上运行。*

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-compress-your-keras-model-x5-smaller-with-tensorflow-model-optimization/&text=How%20to%20compress%20your%20Keras%20model%20x5%20smaller%20with%20TensorFlow%20model%20optimization) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-compress-your-keras-model-x5-smaller-with-tensorflow-model-optimization/)

*   [←如何在 Jupyter 笔记本内运行 PyTorch 1.1.0 的 Tensorboard】](/blog/how-to-run-tensorboard-for-pytorch-110-inside-jupyter-notebook/)
*   [如何使用 Efficientnet 进行迁移学习→](/blog/transfer-learning-with-efficientnet/)*