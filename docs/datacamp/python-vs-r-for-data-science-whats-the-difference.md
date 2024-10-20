# Python vs. R for Data Science:有什么区别？

> 原文：<https://web.archive.org/web/20230101103328/https://www.datacamp.com/blog/python-vs-r-for-data-science-whats-the-difference>

[![](img/1c5de5c4d1b8d170eb45c0d68a395c2c.png)](https://web.archive.org/web/20220525171448/https://www.datacamp.com/courses/)

如果您是数据科学的新手，或者您的组织是新手，您需要选择一种语言来分析您的数据，并选择一种深思熟虑的方式来做出决策。完全公开:虽然我会写 Python，但我的背景主要是 R 社区——但我会尽最大努力做到无党派。

好消息是，您不需要为这个决定付出太大的努力:Python 和 R 都有庞大的软件生态系统和社区，因此这两种语言几乎都适合任何数据科学任务。

最常用的两个编程语言索引 [TIOBE](https://web.archive.org/web/20220525171448/https://www.tiobe.com/tiobe-index) 和 [IEEE Spectrum](https://web.archive.org/web/20220525171448/https://spectrum.ieee.org/computing/software/the-top-programming-languages-2019) 对最流行的编程语言进行排名。他们使用不同的流行度标准，这解释了结果的差异(TIOBE 完全基于搜索引擎结果；IEEE Spectrum 还包括社区和社交媒体数据源，如 Stack Overflow、Reddit 和 Twitter。在每个列表中常用于数据科学的语言中，两个索引都将 Python 列为最受数据科学欢迎的语言，其次是 r。MATLAB 和 SAS 分别排在第三和第四位。

既然我们已经确定 Python 和 R 都是好的、流行的选择，那么有几个因素可能会左右您的决定。

## 你的同事使用什么语言？

决定使用哪种编程语言的最重要因素是了解您的同事使用哪种语言，因为能够与同事共享代码和维护更简单的软件堆栈的好处超过了一种语言相对于另一种语言的任何好处。

## 谁在处理数据？

Python 最初是作为软件开发的编程语言开发的(数据科学工具是后来添加的)，所以具有计算机科学或软件开发背景的人通常会发现 Python 对他们来说更自然。也就是说，从其他流行的编程语言如 Java 或 C++过渡到 Python 比从这些语言过渡到 r 更容易。

r 有一组名为 Tidyverse 的包，这些包提供了用于导入、操作、可视化和报告数据的强大而简单易学的工具。使用这些工具，没有任何编程或数据科学经验的人(至少是传闻中的)可以比使用 Python 更快地变得高效。如果你想亲自测试一下，可以试试 Tidyverse 的[简介，其中介绍了 R 的](https://web.archive.org/web/20220525171448/https://www.datacamp.com/courses/introduction-to-the-tidyverse) [dplyr](https://web.archive.org/web/20220525171448/https://dplyr.tidyverse.org/) 和 [ggplot2](https://web.archive.org/web/20220525171448/https://ggplot2.tidyverse.org/) 包，以及[Python 的](https://web.archive.org/web/20220525171448/https://www.datacamp.com/courses/introduction-to-data-science-in-python)数据科学简介，其中介绍了 Python 的 [pandas](https://web.archive.org/web/20220525171448/https://pandas.pydata.org/) 和 [Matplotlib](https://web.archive.org/web/20220525171448/https://matplotlib.org/) 包，看看你更喜欢哪个。

结论:如果您组织中的数据科学主要由具有编程经验的专门团队进行，Python 有一点优势。如果你有许多没有数据科学或编程背景的员工，但他们仍然需要与数据打交道，R 就有一点优势。

## 你在执行什么任务？

虽然 Python 和 R 基本上都可以完成你能想到的任何数据科学任务，但在某些领域，一种语言比另一种语言更强。

| Python 擅长的地方 | 其中 R 占优势 |
| --- | --- |
| 大部分的深度学习研究都是用 Python 完成的，所以像 T2 的 Keras T3 和 T4 的 py torch T5 这样的工具都是“Python 优先”开发的。你可以在 Keras 的[深度学习简介](https://web.archive.org/web/20220525171448/https://www.datacamp.com/courses/deep-learning-with-keras-in-python)和 PyTorch 的[深度学习简介](https://web.archive.org/web/20220525171448/https://www.datacamp.com/courses/deep-learning-with-pytorch)中了解这些话题。 | 许多**统计建模**研究都是在 R 中进行的，因此有更多的模型类型可供选择。如果您经常对数据建模的最佳方式有疑问，R 是更好的选择。DataCamp 有大量关于 R 的[统计的课程可供选择。](https://web.archive.org/web/20220525171448/https://www.datacamp.com/search?facets%5Btechnology%5D%5B%5D=r&facets%5Btopic%5D%5B%5D=Probability+%26+Statistics) |
| Python 优于 R 的另一个领域是**将模型部署到软件的其他部分**。因为 Python 是一种通用编程语言，所以你可以用 Python 编写整个应用程序，然后无缝地包含你的基于 Python 的模型。我们涵盖了在[部署模型，用 Python 设计机器学习工作流](https://web.archive.org/web/20220525171448/https://www.datacamp.com/courses/designing-machine-learning-workflows-in-python)和[用 Python 构建数据工程管道](https://web.archive.org/web/20220525171448/https://www.datacamp.com/courses/building-data-engineering-pipelines-in-python)。 | R 袖子上的另一个大把戏是简单的**仪表板创建**使用[闪亮](https://web.archive.org/web/20220525171448/https://shiny.rstudio.com/)。这使得没有太多技术经验的人能够创建和发布仪表板，与他们的同事共享。Python 的[破折号](https://web.archive.org/web/20220525171448/https://plot.ly/dash/)是一种替代，但没有那么成熟。您可以在[使用 Shiny in R](https://web.archive.org/web/20220525171448/https://www.datacamp.com/courses/building-web-applications-with-shiny-in-r) 构建 Web 应用程序和[使用 Shiny in R 构建 Web 应用程序:案例研究](https://web.archive.org/web/20220525171448/https://www.datacamp.com/courses/building-web-applications-in-r-with-shiny-case-studies)中了解 Shiny。 |

这个列表并不详尽，专家们一直在争论哪种语言能更好地完成任务。同样，还有更多好消息:Python 程序员和 R 程序员互相借鉴了很多好的想法。比如 Python 的 [plotnine](https://web.archive.org/web/20220525171448/https://plotnine.readthedocs.io/en/stable/) 数据可视化包的灵感来源于 R 的 [ggplot2](https://web.archive.org/web/20220525171448/https://ggplot2.tidyverse.org/) 包，R 的 [rvest](https://web.archive.org/web/20220525171448/https://rvest.tidyverse.org/) 网页抓取包的灵感来源于 Python 的 [BeautifulSoup](https://web.archive.org/web/20220525171448/https://www.crummy.com/software/BeautifulSoup/) 包。所以最终两种语言中最好的想法都会进入另一种语言。

如果你迫不及待地想等待你选择的语言中的某个特定特性，也值得注意的是 Python 和 R 之间有很好的语言互操作性。也就是说，你可以使用 [rpy2](https://web.archive.org/web/20220525171448/https://rpy2.bitbucket.io/) 包从 Python 运行 R 代码，也可以使用 [reticulate](https://web.archive.org/web/20220525171448/https://rstudio.github.io/reticulate/) 从 R 运行 Python 代码。这意味着一种语言中的所有特性都可以从另一种语言中访问。比如深度学习包 [Keras](https://web.archive.org/web/20220525171448/https://keras.io/) 的 [R 版](https://web.archive.org/web/20220525171448/https://keras.rstudio.com/)居然调用 Python。同样，调用 [PyTorch](https://web.archive.org/web/20220525171448/https://pytorch.org/) 。

## 你的竞争对手用什么？

如果你在一家发展迅速的公司工作，并且想招聘顶尖员工，那么做一些对立的研究来看看你的竞争对手在使用什么技术是值得的。毕竟，如果你的新员工不需要学习一门新的语言，他们会更有效率。

## 摘要

编程语言战争大多是人们推广他们最喜欢的语言的借口，并以戏弄使用其他语言的人为乐。所以我想澄清一点，我对在互联网上开始另一场关于 Python 和 R 在数据科学上的争论不感兴趣。

我希望我已经说服了你，虽然 Python 和 R 都是数据科学的好选择，但员工背景、你所处理的问题和你所在行业的文化等因素可以指导你的决定。