# 如何在 Jupyter 笔记本中运行用于 PyTorch 1.1.0 的 Tensorboard

> 原文：<https://www.dlology.com/blog/how-to-run-tensorboard-for-pytorch-110-inside-jupyter-notebook/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年零八个月前

([评论](/blog/how-to-run-tensorboard-for-pytorch-110-inside-jupyter-notebook/#disqus_thread))

![tb_pytorch_nb](img/7750ed9ce83032823950d8d5ee3fa668.png)

脸书推出了支持 TensorBoard 的 PyTorch 1.1。让我们在 Colab 的 Jupyter 笔记本上快速尝试一下。

不需要在您的开发机器上本地安装任何东西。即使升级了特斯拉 T4 GPU，谷歌的 Colab 也能免费派上用场。

首先，让我们创建一个 [Colab 笔记本](https://colab.research.google.com)或者打开 **[这个我做的](https://github.com/Tony607/pytorch-tensorboard/blob/master/PyTorch_1_1_0_tensorboard.ipynb)** 。

在第一个单元格中键入以检查 PyTorch 的版本是否至少为 1.1.0

然后你要像这样安装尖端的张量板。

输出可能会提醒您重新启动运行时以使新的 TensorBoard 生效。可以点开`Runtime -> Restart runtime...`。

接下来，用这条魔线加载 TensorBoard 笔记本扩展。

之后你可以开始探索[火炬。UTILS.TENSORBOARD](https://pytorch.org/docs/stable/tensorboard.html) API，这些实用程序允许您将 PyTorch 模型和指标记录到一个目录中，以便在 TensorBoard UI 中可视化。PyTorch 模型和张量都支持标量、图像、直方图、图形和嵌入可视化。

`SummaryWriter`类是记录供 TensorBoard 使用和可视化的数据的主要入口。让我们为 MNIST 数据集和 ResNet50 模型运行这个官方演示。

你刚刚写了一张图片和模型图数据到 TensorBoard summary。编写器将输出文件写入到”。/runs "目录。

让我们运行张量板来可视化它们

就这样，你成功了！

![tb_pytorch](img/e49b8e6421aaaf4e8d775ba8688da5bb.png)

## 总结和进一步阅读

这个非常短的教程让你在 Jupyter 笔记本上用最新的 Pytorch 1.1.0 开始运行 TensorBoard。继续尝试 PyTorch TensorBoard 支持的其他功能。

点击这里阅读官方 API 文档- [TORCH。UTILS.TENSORBOARD](https://pytorch.org/docs/stable/tensorboard.html)

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-run-tensorboard-for-pytorch-110-inside-jupyter-notebook/&text=How%20to%20run%20Tensorboard%20for%20PyTorch%201.1.0%20inside%20Jupyter%20notebook) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-run-tensorboard-for-pytorch-110-inside-jupyter-notebook/)

*   [←如何在 RK3399Pro 上运行 Keras 模型](/blog/how-to-run-keras-model-on-rk3399pro/)
*   [如何使用 TensorFlow 模型优化将您的 Keras 模型 x5 压缩得更小→](/blog/how-to-compress-your-keras-model-x5-smaller-with-tensorflow-model-optimization/)