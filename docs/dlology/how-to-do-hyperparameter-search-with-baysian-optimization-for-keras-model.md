# 如何用贝叶斯优化进行 Keras 模型的超参数搜索

> 原文：<https://www.dlology.com/blog/how-to-do-hyperparameter-search-with-baysian-optimization-for-keras-model/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年零九个月前

([评论](/blog/how-to-do-hyperparameter-search-with-baysian-optimization-for-keras-model/#disqus_thread))

![search](img/35bf368627b320a53af604794bd9e162.png)

与网格搜索和随机搜索等更简单的超参数搜索方法相比，贝叶斯优化建立在贝叶斯推理和高斯过程的基础上，试图以尽可能少的迭代次数找到未知函数的最大值。它特别适合于优化高成本函数，如深度学习模型的超参数搜索，或其他探索和利用之间的平衡很重要的情况。

我们要用的贝叶斯优化包是 [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) ，可以用下面的命令安装，

```py
pip install bayesian-optimization
```

首先，我们将指定要优化的函数，在我们的例子中是超参数搜索，该函数将一组超参数值作为输入，并输出贝叶斯优化器的评估精度。在该函数中，将使用指定的超参数构建一个新模型，对多个时期进行训练，并根据一组指标进行评估。每个新评估的准确度将成为贝叶斯优化器的新观察，其有助于下一次搜索超参数的值。

让我们首先创建一个助手函数，它用各种参数构建模型。

然后，这里是要用贝叶斯优化器优化的函数，**部分**函数负责两个参数-`fit_with`中的`input_shape`和`verbose`，它们在运行时具有固定值。

该函数采用两个超参数进行搜索，即“dropout_2”层的辍学率和学习率值，它为 1 个时期训练模型，并输出贝叶斯优化器的评估精度。

****贝叶斯优化**对象将开箱即用，无需太多调整。构造函数接受要优化的函数以及要搜索的超参数的边界。您应该知道的主要方法是`maximize`，它完全按照您的想法工作，在给定超参数的情况下最大化评估精度。**

 **这里有许多您可以传递给`maximize`的参数，然而，最重要的是:

*   `n_iter`:你要执行多少步贝叶斯优化。步数越多，你就越有可能找到一个好的最大值。
*   `init_points`:你要执行多少步**的随机探索。随机探索有助于探索空间的多样化。**

```py
|   iter    |  target   | dropou... |    lr     |
-------------------------------------------------
468/468 [==============================] - 4s 8ms/step - loss: 0.2575 - acc: 0.9246
Test loss: 0.061651699058711526
Test accuracy: 0.9828125
|  1        |  0.9828   |  0.2668   |  0.007231 |
468/468 [==============================] - 4s 8ms/step - loss: 0.2065 - acc: 0.9363
Test loss: 0.04886047407053411
Test accuracy: 0.9828125
|  2        |  0.9828   |  0.1      |  0.003093 |
468/468 [==============================] - 4s 8ms/step - loss: 0.2199 - acc: 0.9336
Test loss: 0.05553104653954506
Test accuracy: 0.98125
|  3        |  0.9812   |  0.1587   |  0.001014 |
468/468 [==============================] - 4s 9ms/step - loss: 0.2075 - acc: 0.9390
Test loss: 0.04128134781494737
Test accuracy: 0.9890625
|  4        |  0.9891   |  0.1745   |  0.003521 |
```

经过 4 次搜索，用找到的超参数建立的模型仅用一个历元的训练就达到了 98.9%的评估精度。

## 与其他搜索方法相比

与在有限数量的离散超参数组合中进行搜索的网格搜索不同，高斯过程的贝叶斯优化本质不允许以简单/直观的方式处理离散参数。

例如，我们想从选项列表中搜索一个密集层的神经元数目。为了应用贝叶斯优化，有必要在构建模型之前将输入参数显式地转换为离散参数。

你可以这样做。

在构建模型之前，密集层神经元将被映射到 3 个唯一的离散值，128、256 和 384。

在贝叶斯优化中，每下一个搜索值依赖于以前的观测值(以前的评估精度)，整个优化过程可能很难像网格或随机搜索方法那样分布式或并行化。

## 结论和进一步阅读

这个快速教程介绍了如何使用贝叶斯优化进行超参数搜索，与网格或随机等其他方法相比，它可能更有效，因为每次搜索都是从以前的搜索结果中"**引导的**。

### 一些你可能会觉得有用的材料

[贝叶斯优化](https://github.com/fmfn/BayesianOptimization)——本教程中使用的高斯过程全局优化的 Python 实现。

[如何在 TPU 上更快地免费执行 Keras 超参数优化 x3](https://www.dlology.com/blog/how-to-perform-keras-hyperparameter-optimization-on-tpu-for-free/)——我之前的关于用 Colab 的免费 TPU 执行网格超参数搜索的教程。

#### 在我的 [GitHub](https://github.com/Tony607/Keras_BayesianOptimization) 上查看完整的源代码。

*   标签:
*   [keras](/blog/tag/keras/) ,
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-do-hyperparameter-search-with-baysian-optimization-for-keras-model/&text=How%20to%20do%20Hyper-parameters%20search%20with%20Bayesian%20optimization%20for%20Keras%20model) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-do-hyperparameter-search-with-baysian-optimization-for-keras-model/)

*   [←如何在 Jupyter 笔记本中运行 tensor board](/blog/how-to-run-tensorboard-in-jupyter-notebook/)
*   [如何在 Jetson Nano 上运行 Keras 模型→](/blog/how-to-run-keras-model-on-jetson-nano/)****