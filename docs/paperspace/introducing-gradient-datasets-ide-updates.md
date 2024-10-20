# 介绍渐变数据集和全新的渐变笔记本 IDE！

> 原文：<https://blog.paperspace.com/introducing-gradient-datasets-ide-updates/>

在渐变笔记本的最新更新中，我们引入了许多新功能和改进，包括渐变数据集、对交互式小部件的支持、更好的单元、文件和内核管理，等等！

让我们开始更新吧！

## 引入公共和私有梯度数据集

梯度数据集为本地笔记本存储提供了一种替代方案，可跨梯度团队和资源使用。在这个版本中，现在可以在笔记本中装载数据集。

数据集在 IDE 的`Datasets`选项卡中可用，文件存储在`/datasets`目录中。

作为此次发布的一部分，我们已经为所有笔记本电脑提供了许多公共数据集。这些数据集可以在 IDE 的`Datasets`菜单中的`Public`选项卡中找到。

<https://blog.paperspace.com/content/media/2022/04/Screen-Recording-2022-04-01-at-3.07.43-PM-1.mp4>



Mounting the `tiny-imagenet-200` dataset.

公共数据集经常更新。首批公共数据集包括:

*   **Tiny ImageNet 200**:200 类 10 万张图像(每类 500 张)缩小为 64×64 彩色图像。每个类有 500 幅训练图像、50 幅验证图像和 50 幅测试图像。
*   **OpenSLR**:LibriSpeech ASR 语料库由 Vassil Panayotov 在 Daniel Povey 的协助下准备的大约 1000 小时的 16kHz 朗读英语语音组成。这些数据来源于 LibriVox 项目的 read audiobooks，并经过仔细的分割和排列。
*   MNIST:一个手写数字的数据库有 60，000 个样本的训练集和 10，000 个样本的测试集。这是从 NIST 可获得的更大集合的子集。数字已经过大小标准化，并在固定大小的图像中居中。
*   **LSUN** :包含 10 个场景类别，如餐厅、卧室、小鸡、户外教堂等。对于训练数据，每个类别包含大量图像，从大约 120，000 到 3，000，000。验证数据包括 300 幅图像，测试数据对于每个类别有 1000 幅图像。
*   **FastAI** : Paperspace 的 Fast.ai 模板是为程序员建立起来并运行实用的深度学习而构建的。相应的公共数据集使使用 FastAI 运行时创建的笔记本能够快速访问所需的演示数据。
*   **COCO** :这个数据集是一个大规模的对象检测、分割、关键点检测和字幕数据集。数据集由 328K 图像、边界框及其标签组成。

如果您想创建自己的数据集，只需使用笔记本 IDE 中数据集菜单内的`Team`选项卡来上传和安装您自己的数据。

下面是一个如何上传历史艺术品数据集，然后使用命令`!ls ../datasets/historic_art/`在笔记本中访问数据集的示例:

<https://blog.paperspace.com/content/media/2022/04/Screen-Recording-2022-04-04-at-1.53.12-PM-1.mp4>



Uploading and mounting a dataset within a notebook.

对于大于 5 GB 的数据集，将提示您使用渐变工作流或渐变 CLI 上传数据。

有关 Gradient 的公共数据集和数据存储的更多信息，请通读相关的 [doc](https://docs.paperspace.com/gradient/data/) 文档。

## 交互式小工具

除了数据集，笔记本现在还支持交互式 [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) 开箱即用！

<https://blog.paperspace.com/content/media/2022/04/itsworking.mp4>



Displaying a GIF file with `ipywidgets`.

Jupyter 小部件或 ipywidgets 在各种上下文中都很有用，从修改笔记本的外观和可解释性到启用许多深度学习功能。小部件现在可以在本地使用了——只需导入`ipywidgets`模块！

ipywidgets 还支持笔记本中的许多其他小部件，从输入提示到滑块、多选、日历等等。

<https://blog.paperspace.com/content/media/2022/04/final.mp4>



Widgets also enable sliders, calendars, and more.

查看 [ipywidgets 文档](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html)获取完整的小部件列表。

## IProgress and PyTorch/TensorFlow dataloaders

笔记本现在也支持 IProgress 文本进度条。

<https://blog.paperspace.com/content/media/2022/04/progress2.mp4>



IProgress helps provide visual cues for operations that take time to complete.

IProgress 经常用于显示有用的信息，如一次训练中还有多少训练期。通过此次更新，IProgress 现在可以完全在笔记本中呈现。

<https://blog.paperspace.com/content/media/2022/04/Screen-Recording-2022-04-04-at-2.25.00-PM.mp4>



An example of IProgress used with TensorFlow Datasets.

这尤其扩展到 TensorFlow 和 PyTorch 的扩展库套件。TensorFlow 数据集和 Pytorch Lightning 等库中的函数现在可以正常工作了。

<https://blog.paperspace.com/content/media/2022/04/fixed.mp4>



An example using the PyTorch fit method.

此外，TensorFlow 和 PyTorch 数据加载器的进度条和培训功能现在可以在笔记本中自然显示。不再需要向通过终端执行的 Python 脚本中添加代码来运行包含这些函数的代码。

## plotly 和其他带有 HTML 输出的绘图库

有了 ipywidgets 促进 HTML 输出，我们现在可以充分利用交互式绘图库 [plotly](https://plotly.com/python/) 。

<https://blog.paperspace.com/content/media/2022/04/plotly.mp4>



Example using Plotly.

与 Matplotlib 或 T2 Seaborn 不同，plotly 允许对现有绘图进行交互式动态更改，如删除数据类或缩放。现在，我们可以创建 plotly 图形，甚至保存它们嵌入到其他网页。

我们期待着看到你可以做什么，与全面的小部件支持。

现在我们来谈谈 IDE 本身的更新。

## 单元管理更新

首先，我们极大地增强和扩展了笔记本中的单元操作，以包括来自 JupyterLab 的熟悉概念，如`join`、`split`、`insert`等等。

<https://blog.paperspace.com/content/media/2022/04/combinemd.mp4>



The new Gradient Notebooks IDE makes it easy to cut, copy, paste, join, split cells, and more.

更新后的笔记本 IDE 还可以轻松隐藏单元格输出，从而在笔记本中腾出更多空间。

<https://blog.paperspace.com/content/media/2022/04/Screen-Recording-2022-04-04-at-4.49.22-PM.mp4>



我们还扩展了命令面板，使其更容易访问有用的快捷键和命令。命令面板实体的列表正在快速增长，所以一定要检查一下！

<https://blog.paperspace.com/content/media/2022/04/cmd-p.mp4>



Creating a new notebook from the command palette.

命令面板可通过键盘快捷键`**command + p**`访问。

## 更新的文件管理器

我们还向文件管理器引入了拖放功能和一些其他有用的操作。

<https://blog.paperspace.com/content/media/2022/04/re.mp4>



Moving files with the drag-and-drop file manager.

此外，现在可以将多个文件上传到一个文件夹，并且当右键单击一个文件或文件夹时，有许多新的文件管理选项可用。

<https://blog.paperspace.com/content/media/2022/04/Screen-Recording-2022-04-04-at-3.35.21-PM.mp4>



Right click on a file to perform a number of actions like renaming, downloading, copying, and more.

## 内核管理

现在选择、停止和重启笔记本内核比以往任何时候都更容易。

<https://blog.paperspace.com/content/media/2022/04/kernel.mp4>



Using the kernel selection window to select a Python 3 kernel for a new notebook.

导航到笔记本 IDE 左下角的内核会话管理器，与每个笔记本内核实时交互。使用适当的按钮停止并重启内核。

## 额外收获:终端更新！

作为对专业用户和成长型用户的奖励，我们已经将终端作为分屏项目移动到笔记本 IDE 中！

<https://blog.paperspace.com/content/media/2022/04/new-terminal.mp4>



Pro and Growth plan users can now interact with their notebooks simultaneously while using the new terminal.

现在可以在不离开笔记本文件本身的情况下发出终端命令。

# 尝试一下

准备好试用新的渐变笔记本 IDE 了吗？

尝试[在 Gradient 中创建一个新的笔记本](https://docs.paperspace.com/gradient/tutorials/notebooks-tutorial)或者继续从 [ML Showcase 中派生一个项目](https://ml-showcase.paperspace.com/)。

请务必[让我们知道](https://twitter.com/HelloPaperspace)您的项目，如果您有任何问题，请不要犹豫[联系支持](https://docs.paperspace.com/contact-support/)！