# 如何将训练好的 Keras 模型转换为单个张量流？pb 文件并进行预测

> 原文：<https://www.dlology.com/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零一个月前

([评论](/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/#disqus_thread))

![keras_tf_pb](img/292a8aaadea5928ef295f9756c084ba5.png)

您将一步一步地学习如何将您训练的 Keras 模型冻结并转换为单个 TensorFlow pb 文件。

与 TensorFlow 相比，Keras API 可能看起来不那么令人畏惧，而且更容易使用，特别是当您正在进行快速实验并使用标准图层构建模型时。而当您计划跨不同编程语言将模型部署到不同平台时，TensorFlow 更加通用。虽然有许多方法可以将 Keras 模型转换成它的 TenserFlow 对应物，但我将向您展示一种最简单的方法，您只需要在部署情况下使用转换后的模型进行预测。

这是将要涵盖的内容的概述。

*   Keras to single TensorFlow .pb file
*   加载。用 TensorFlow 做 pb 文件并做预测。
*   (可选)在 Jupyter 笔记本中可视化图形。

#### *这篇文章的源代码可以在我的 [GitHub](https://github.com/Tony607/keras-tf-pb) 上找到。*

## Keras to TensorFlow .pb file

训练好 Keras 模型后，最好先将其保存为单个 HDF5 文件，以便在训练后将其加载回来。

如果您在使用包含**批处理化**层的模型(如 DenseNet)时遇到了“与预期资源不兼容”问题，请确保在新会话中加载 Keras 模型之前将学习阶段设置为 0。

然后，您可以加载模型并找到模型的输入和输出张量的名称。

如您所见，我们的简单模型只有单个输入和输出，而您的模型可能有多个输入/输出。

我们跟踪它们的名字，因为我们将在推断过程中通过名字在转换的张量流图中定位它们。

第一步是获得 TensorFlow 后端的计算图，它表示 Keras 模型，其中包括前向传递和训练相关的操作。

然后，该图将被转换为 GraphDef 协议缓冲区，之后，它将被修剪，从而移除计算所请求的输出(例如训练操作)所不需要的子图。这个步骤被称为冻结图形。

`frozen_graph`是一个序列化的 GraphDef 原型，我们可以使用下面的函数调用将其保存为一个二进制 pb 文件。

## 加载。pb 文件并进行预测

现在，我们有了所有需要预测的东西，图表保存为一个单一的。pb 文件。

要重新加载它，可以通过重启 Jupyter Notebook 内核或在新的 Python 脚本中运行来启动一个新的会话。

以下几行代码反序列化来自。pb 文件并将其恢复为默认图形到当前运行的 TensorFlow 会话。

找到输入张量，这样我们就可以用一些输入数据来填充它，并从输出张量中获取预测，我们将按名称获取它们。唯一不同的是，所有张量的名称都以字符串“import/”为前缀，因此输入张量现在是名为 `"import/conv2d_1_input:0"` 的，而输出张量是 `"import/dense_2/Softmax:0"` 。

要做一个预测，可以很简单，

`predictions`是在这种情况下的 softmax 值，shape (20，10)代表 20 个样本，每个样本具有 10 个类的 logits 值。

如果您的模型有多个输入/输出，您可以这样做。

## 在笔记本中可视化图形(可选)

你想知道模型冻结步骤对你的模型做了什么吗，比如哪些操作被删除了？

让我们通过加载 TensorBoard 的最小版本来显示图形结构，从而在 Jupyter 笔记本中并排比较这两个图形。

这里我包含了一个 [show_graph.py](https://github.com/Tony607/keras-tf-pb/blob/master/show_graph.py) 模块来允许你这样做。

您可以运行这个块两次，一次在 Keras 模型训练/加载之后，一次在加载和恢复之后。pb 文件，这里是结果。

列车图表:

![train_graph](img/e84751e639110981f0de0a7c976a3721.png)

冻结图表:

![tensorboard](img/d0988fee410ade4296fa2e56401c9f66.png)

你可以很容易地看到相关的训练操作在冻结图中被删除。

## 结论和进一步阅读

您已经学习了如何将 Keras 模型转换为张量流。pb 文件仅用于推断目的。请务必在我的 [GitHub](https://github.com/Tony607/keras-tf-pb) 上查看这篇文章的源代码。

这里有一些相关的资源，你可能会觉得有帮助。

[tensor flow 模型文件工具开发者指南](https://www.tensorflow.org/extend/tool_developers/)

[导出和导入元图](https://www.tensorflow.org/api_guides/python/meta_graph)

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/&text=How%20to%20convert%20trained%20Keras%20model%20to%20a%20single%20TensorFlow%20.pb%20file%20and%20make%20prediction) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/)

*   [←如何在 Python 3 中加载 Python 2 PyTorch 检查点](/blog/how-to-load-python-2-pytorch-checkpoint-in-python-3-1/)
*   [如何免费在 TPU 上更快地执行 Keras 超参数优化 x3→](/blog/how-to-perform-keras-hyperparameter-optimization-on-tpu-for-free/)