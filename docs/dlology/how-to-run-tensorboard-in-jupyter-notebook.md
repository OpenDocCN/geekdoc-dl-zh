# 如何在 Jupyter 笔记本中运行 TensorBoard

> 原文：<https://www.dlology.com/blog/how-to-run-tensorboard-in-jupyter-notebook/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年零九个月前

([评论](/blog/how-to-run-tensorboard-in-jupyter-notebook/#disqus_thread))

![jtb](img/9d8ec4dc197dd6f45cc5d659b59ef4f3.png)

TensorBoard 是一个很好的工具，它提供了评估 TensorFlow 模型训练所需的许多指标的可视化。过去很难使用这个工具，尤其是在托管的 Jupyter 笔记本环境中，如 Google Colab、Kaggle notebook 和 Coursera 的 Notebook 等。在本教程中，我将向您展示如何无缝运行和查看 TensorBoard，就在一个托管的或本地的 Jupyter 笔记本电脑中，使用最新的 TensorFlow 2.0。

你可以在阅读这篇文章的同时运行这个 [Colab 笔记本](https://colab.research.google.com/gist/Tony607/7f55518ba7af13eb7e2e782b3b50a38b/tensorboard_in_notebooks.ipynb)。

开始安装 TF 2.0，加载 TensorBoard 笔记本扩展:

或者，要运行本地笔记本，可以创建一个 conda 虚拟环境，安装 TensorFlow 2.0。

然后你可以在训练前启动 TensorBoard 来监控它的进展:在笔记本内使用 [魔法](https://ipython.readthedocs.io/en/stable/interactive/magics.html)。

现在您可以看到一个空的 TensorBoard 视图，显示消息“当前数据集没有活动的仪表板”，这是因为日志目录当前是空的。

让我们用一个非常简单的 Keras 模型来创建、训练和记录一些数据。

现在回到之前的 TensorBoard 输出，用右上角的按钮刷新它，并观察更新视图。

![tb1](img/b415f6134ed80123d8dc522db58f3c10.png)

通过发出相同的命令，相同的 TensorBoard 后端被重用。如果选择不同的日志目录，将会打开一个新的 TensorBoard 实例。端口被自动管理。

任何值得一提的有趣的新特性是“**概念图**”。要查看概念图，请选择“keras”标签。对于本例，您将看到一个折叠的**序列** 节点。双击节点可以看到模型的结构:

![tag_k](img/d4d6d6be9a15af4dd2f6e8ca147cdc0a.png)

## 结论与延伸阅读

在本快速教程中，我们将介绍如何在 Jupyter 笔记本电脑中启动和查看盛开的 TensorBoard。有关如何在 TensorFlow 2.0 中利用 TensorBoard 的其他新功能的更多说明，请务必查看这些资源。

[张量板标量:在 Keras 中记录训练指标](https://www.tensorflow.org/tensorboard/r2/scalars_and_keras)

[使用 HParams 仪表盘调节超参数](https://www.tensorflow.org/tensorboard/r2/hyperparameter_tuning_with_hparams)

[使用假设分析工具仪表板了解模型](https://www.tensorflow.org/tensorboard/r2/what_if_tool)

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-tensorboard-in-jupyter-notebook/&text=How%20to%20run%20TensorBoard%20in%20Jupyter%20Notebook) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-tensorboard-in-jupyter-notebook/)

*   [←如何使用英特尔显卡更快地运行 TensorFlow 对象检测模型](/blog/how-to-run-tensorflow-object-detection-model-faster-with-intel-graphics/)
*   [如何使用贝叶斯优化对 Keras 模型进行超参数搜索→](/blog/how-to-do-hyperparameter-search-with-baysian-optimization-for-keras-model/)