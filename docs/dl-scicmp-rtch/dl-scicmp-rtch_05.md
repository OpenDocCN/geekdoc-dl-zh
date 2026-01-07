# 2  关于 torch，以及如何获取它

> 原文：[`skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/what_is_torch.html`](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/what_is_torch.html)

## 2.1 在`torch`世界中

`torch`是 PyTorch 的 R 端口，PyTorch 是工业和研究中最常用的两个深度学习框架之一（截至本文撰写时）。按照其设计，它也是一个在各类科学计算任务中使用（其中一些你将在本书的最后一部分遇到）的出色工具。它完全用 R 和 C++（包括一些 C）编写。使用它不需要安装 Python。

在 Python（PyTorch）方面，生态系统呈现为一组同心圆。在中间，是 PyTorch 本身，这是一个核心库，没有它任何东西都无法工作。围绕它，我们有被称为框架库的内层圈，这些库专注于特定类型的数据（图像、声音、文本等），或者围绕工作流程任务，如部署。然后，是更广泛的生态系统，包括附加组件、专业化和库，对于这些库，PyTorch 是构建块或工具。

在 R 方面，我们有相同的“核心”——一切都取决于核心`torch`——我们确实有相同类型的库；但类别，“圆圈”，彼此之间的界限不太明显。没有严格的界限。只有一个充满活力的开发者社区，他们来自不同的背景，有不同的目标，致力于进一步发展和扩展`torch`，以便帮助越来越多的人完成他们的各种任务。生态系统发展如此迅速，我将避免列出单个包——随时访问[torch 网站](https://torch.mlverse.org/packages/)，以查看精选子集。

尽管如此，有三个包我在这里会提到，因为它们在书中被使用：`torchvision`、`torchaudio`和`luz`。前两个打包了特定领域的转换、深度学习模型、数据集以及用于图像（包括视频）和音频数据的实用工具。第三个是一个高级、直观、易于使用的`torch`接口，允许通过几行代码定义、训练和评估神经网络。与`torch`本身一样，所有这三个包都可以从 CRAN 安装。

## 2.2 安装和运行 torch

`torch`适用于 Windows、MacOS 和 Linux。如果你有一个兼容的 GPU，并且安装了必要的 NVidia 软件，你可以从显著的加速中受益，这种加速将取决于训练的模型类型。然而，本书中的所有示例都经过选择，可以在 CPU 上运行，而不会对你的耐心提出过高要求。

由于它们往往具有短暂的特征，我在本书中不会详细阐述兼容性问题；类似地，我也将避免列出具体的安装说明。在任何时候，你都可以在[vignette](https://cran.r-project.org/web/packages/torch/vignettes/installation.html)中找到最新的信息；如果你遇到问题或有疑问，欢迎在`torch` GitHub 仓库中提出问题。
