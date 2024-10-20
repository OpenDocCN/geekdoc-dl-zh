# 机器学习的 19 个最佳 JupyterLab 扩展

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/jupyterlab-extensions-for-machine-learning>

Jupyter 的旗舰项目 JupyterLab 是数据科学领域最受欢迎和最有影响力的开源项目之一。Jupyter 生态系统的一大优点是，如果你缺少什么，要么有一个开源扩展，要么你可以自己创建它。

在本文中，我们将讨论 JupyterLab 扩展，这些扩展可以使您的机器学习工作流更好。

## 什么是 JupyterLab 扩展？

正如 JupyterLab 的人们所说:

> “JupyterLab 被设计成一个可扩展的环境”。

JupyterLab extension 只是一个即插即用的插件，它使您需要的更多东西成为可能。

从技术上讲，JupyterLab extension 是一个 JavaScript 包，可以向 JupyterLab 界面添加各种交互式功能。如果你想知道如何创建自己的扩展，请阅读本指南。

## 如何管理 JupyterLab 扩展

有大量的 JupyterLab 扩展可供您使用。

您如何管理所有这些扩展？

[**扩展管理器**](https://web.archive.org/web/20221203094057/https://jupyterlab.readthedocs.io/en/stable/user/extensions.html#using-the-extension-manager) (命令面板中的小拼图图标)让你直接从 JupyterLab 安装和禁用扩展。我强烈推荐。

如果你的扩展管理器需要添加插件，看看我们的机器学习最佳 JupyterLab 扩展列表。

## 最佳 JupyterLab 扩展

我们不能谈论 JupyterLab 对机器学习的扩展，而不提到我们建立的帮助人们跟踪实验、数据探索和在笔记本上发生的错误分析的扩展。

Neptune-notebooks 扩展让您:

*   点击一个按钮发送或下载笔记本电脑检查点从海王星
*   比较海王星的检查点，分享给团队。

让我展示给你看:

如果看起来有趣，请查看我们的文档。

JupyterLab TensorBoard 是 JupyterLab:)上 TensorBoard 的前端扩展。它使用 jupyter_tensorboard 项目作为 tensorboard 后端。它通过为 jupyter 界面中的 tensorboard 启动、管理和停止提供图形用户界面，帮助 jupyter 笔记本和 tool for tensorflow 的可视化工具)之间的协作。

以下是它的作用:

*   无需在命令行中键入 tensorboard 和长日志路径。
*   不需要额外的端口来服务 tensor board——这对远程 jupyter 服务器很有帮助。
*   多个 tensorboard 实例同时管理。

ML workspace 是一个基于 web 的一体化集成开发环境，专门用于机器学习和数据科学。

它易于部署，让您可以在自己的机器上高效地构建 ML 解决方案。该工作区是面向开发人员的通用解决方案，预装了各种流行的数据科学库(如 Tensorflow、PyTorch、Keras、Sklearn)和开发工具(如 Jupyter、VS Code、Tensorboard)，经过完美配置、优化和集成。

系统监视器是一个 JupyterLab 扩展，用于显示系统信息(内存和 cpu 使用情况)。它允许您监控自己的资源使用情况。

该扩展使您能够深入了解当前笔记本服务器及其子服务器(内核、终端等)使用了多少资源，从而优化您的 ML 实验并更好地管理工作。

LSP(语言服务器协议)是一个 JupyterLab 扩展，它使进程间通信能够支持您可能想要使用的多种语言。

LSP 集成有几个详细但有用的特性:

*   悬停显示带有函数/类签名、模块文档或语言服务器提供的任何其他信息的工具提示
*   诊断—严重错误、警告等的颜色。
*   跳转到定义—使用上下文菜单项跳转到定义
*   引用的突出显示—当光标放在变量、函数等上时，所有的用法都将突出显示。
*   触发时自动完成某些字符
*   自动签名建议
*   无需运行内核的高级静态分析自动完成
*   在笔记本和文件编辑器中重命名变量、函数等
*   诊断面板

调试器是一个 JupyterLab 扩展，作为 Jupyter 笔记本、控制台和源文件的可视化调试器。它可以帮助您识别和修复错误，以便您的机器学习模型可以正常工作。

您可以使用 JupyterLab 的 kernelspy 扩展来检查调试器 UI 和内核之间发送的调试消息。

当您使用 VS 代码时，JupyterLab 调试器也很有帮助，因为您可以检查调试消息以了解何时发出调试请求，并在 VS 代码中比较 JupyterLab 调试器和 Python 调试器的行为。

这是 Git 的 JupyterLab 扩展——一个免费的开源分布式版本控制系统。它允许你进行版本控制。您只需从左侧面板的 Git 选项卡中打开 Git 扩展来使用它。

这个扩展为您提供了使用的灵活性，因为它的行为可以通过不同的设置来修改。

该扩展将一些 Jupytext 命令添加到命令选项板中。您可以使用它来为您的笔记本选择所需的 ipynb/text 配对。这是一个小功能，但可以帮助你浏览你的笔记本。

nbgather 是一个 JupyterLab 扩展，它拥有用于清理代码、恢复丢失的代码以及在 Jupyter Lab 中比较代码版本的工具。该扩展为您保存了所有已执行代码的历史记录，以及它生成的笔记本元数据的输出。

下载扩展后，您可以清理和比较代码的版本。

nbgather 正处于开发的初级阶段，所以它仍然可能有一些小故障。无论如何，如果你想拥有整洁一致的笔记本，这是值得一试的。

变量检查器是 JupyterLab 的一个有用的扩展，可以显示当前使用的变量及其值。它的灵感来自 jupyter 笔记本的可变检查器扩展和 jupyterlab 中包含的检查器扩展。

至于现在，它仍在开发中，所以你可能会遇到一些小故障。以下是您可以用它做的事情:

检查 python 控制台和笔记本的变量

*   在数据网格查看器中检查矩阵，但是，它可能不适用于大型矩阵
*   以内联和交互方式检查 Jupyter 小部件

这个 JupyterLab 扩展为你提供了一些有助于区分和合并 Jupyter 笔记本的功能。它了解笔记本文档的结构，因此在区分和合并笔记本时可以做出明智的决策。

以下是主要功能的简短总结:

以终端友好的方式比较笔记本电脑

*   通过自动解决冲突，三方合并笔记本
*   查看笔记本的丰富渲染差异
*   为笔记本电脑提供基于网络的三向合并工具
*   以终端友好的方式查看单个笔记本

Voyager 是一个 JupyterLab MIME 渲染器扩展，用于在 Voyager 2 中查看 CSV 和 JSON 数据。这是一个简单的解决方案，允许您可视化数据。

这个扩展提供了与 Voyager 的最低限度的集成。

LaTeX 是一个 JupyterLab 扩展，允许您实时编辑 LaTeX 文档。

默认情况下，该扩展在服务器上的 xelatex 上运行，但是您可以通过自定义 jupyter_notebook_config.py 文件来自定义该命令。至于书目，它运行在 bibtex 上，但你也可以定制它。

您可以定制的另一个元素是通过触发外部 shell 命令来运行任意代码的能力。

这是一个 JupyterLab 扩展 mimerenderer，用于在 IFrame 选项卡中呈现 HTML 文件。它允许您通过双击来查看呈现的 HTML。文件浏览器中的 html 文件。文件在 JupyterLab 选项卡中打开。

Plotly 是一个 JupyterLab 扩展，用于呈现 Plotly 图表。

要观察扩展源代码的变化并自动重建扩展和应用程序，可以观察 jupyter-renderers 目录并在观察模式下运行 JupyterLab。

我们列表中的另一个位置是用于渲染散景可视化的 Jupyter 扩展。

JupyterLab 的目录扩展可能看起来不太像是一个技术问题，但是它可以在向下滚动和查找信息时为您省去很多麻烦。

当您打开笔记本或 markdown 文档时，它会在左侧区域自动生成目录。条目是可点击的，你可以将文档滚动到有问题的标题。

可折叠标题是一个有用的扩展，它让你可以折叠标题。通过点击标题单元格左侧的插入符号图标或使用快捷方式，可以折叠/取消折叠选定的标题单元格(即以某个数字“#”开头的降价单元格)。

Jupyter Dash 是一个库，可以轻松地从 Jupyter 环境(例如 classic Notebook、JupyterLab、Visual Studio Code notebooks、nteract、PyCharm notebooks 等)中构建 Dash 应用程序。).

它有许多有用的功能:

非阻塞执行

*   显示模式:外部、内嵌、JupyterLab
*   热重装:当应用程序的代码发生变化时，自动更新正在运行的 web 应用程序的能力。
*   错误报告:一个小的用户界面，显示由属性验证失败和回调中引发的异常导致的错误
*   Jupyter 代理检测
*   生产部署
*   Dash 企业工作区
*   最后一个是 jupyterlab-sql 扩展，它向 jupyterlab 添加了一个 sql 用户界面。它允许您使用点击式界面浏览您的表，并使用自定义查询读取和修改您的数据库。

结论

JupyterLab 的列表非常广泛，因此有许多工具可供选择。您可以使用一个、两个或全部。只要确保它们不会让你的 Jupyter 空间变得拥挤，不会减慢进程。

## 愉快地尝试扩展！

The list of JupyterLab is quite extensive so there are many tools to choose from. You can use one, two, or all of them. Just make sure they’re not cluttering your Jupyter space and slow down processes.

Happy experimenting with extensions!