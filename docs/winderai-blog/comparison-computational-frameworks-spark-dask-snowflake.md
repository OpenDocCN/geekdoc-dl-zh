# 计算框架的比较:Spark、Dask、Snowflake 等等

> 原文：<https://winder.ai/comparison-computational-frameworks-spark-dask-snowflake/>

云德。人工智能按照协议工作。人工智能评估通用计算框架。这项工作的总结包括:

*   评估每个工具的工作流程和性能的全面演示
*   包含基准和示例应用程序的 GitHub 存储库
*   Bacalhau 文档网站的文档和摘要视频

下面的视频总结了这项工作。

[https://s3.eu-west-2.amazonaws.com/assets.winder.ai/blog/2022/220624_CaseStudy_ProtocolLabs/220624_PL_ComputeOverData_LandscapeAnalysis_720p.mp4](https://s3.eu-west-2.amazonaws.com/assets.winder.ai/blog/2022/220624_CaseStudy_ProtocolLabs/220624_PL_ComputeOverData_LandscapeAnalysis_720p.mp4)

## 计算景观

传统计算环境中有几十种能够处理一般工作负载的框架。有些是专门设计来利用数据局部性的，方法是将计算带到数据所在的地方。该景观分析回顾了这些工具的选择，总结了它们的优缺点。

> [完整幻灯片组](https://docs.google.com/presentation/d/1wOh-ASGshgc1Ivkoyaz9zGpVGTxX9LDMZQB4-eXOBP4/edit?usp=sharing)包含计算框架的详细概述，并包含示例代码片段。

 

Python 数据栈包括像 Pandas 和 Dask 这样的工具，它们提供了一个非常方便的数据结构，名为 [Dataframe](https://en.wikipedia.org/wiki/Dataframe) ，特别适合处理表格数据。

数据库世界提供了针对不同用例优化的多种选择(例如，表格数据、实时时间序列等。).这项研究着眼于 Postgres 和 Snowflake，这是这一领域的两个相当通用的工具。

像 Apache Spark 和 Hadoop 这样的大数据工具也是这种分析的一部分。它们能够在非常大的集群中处理结构化和非结构化数据。这个类别首先引入了数据局部性的概念，以避免通过集群网络传输数据。

最后但同样重要的是，一些 web3 工具也是这一分析的一部分。他们的目标是支持分布式存储和计算。请注意，在撰写本文时，它们正在大量开发中。在许多情况下，仍然不清楚它们是如何工作的，以及它们未来的发展方向。

不幸的是，这些系统中的许多都不容易在本地主机上运行或大规模运行。传统框架受到大量运营开销的困扰，导致资源使用效率低下。此外，即使是运行入门指南，通常也有很大的设置负担，设置了相对较高的准入门槛。

下表总结了他们在不同要求方面的得分。该评级基于设置和运行下一节中描述的代码的经验，请在[幻灯片](https://docs.google.com/presentation/d/1wOh-ASGshgc1Ivkoyaz9zGpVGTxX9LDMZQB4-eXOBP4/edit#slide=id.g11f8b483676_0_87)中找到更多详细信息。

 

为什么 Bachalau 不在这个名单上？

这个分析不是 Bacalhau 和现有框架之间的直接比较。相反，这项研究旨在帮助 Bacalhau 社区了解传统系统的优点和缺点。

## 代码库

### 示例代码

浏览计算水域的一个很好的起点是查看[代码库](https://github.com/winderai/bacalhau-landscape-analysis-benchmarks)，在那里您会发现令人尴尬的并行工作负载的工作示例(例如[字数统计](https://en.wikipedia.org/wiki/Word_count)，数据集聚合等)。).查看专用文件夹，以笔记本格式观看演示，无需安装。或者，你可以在[幻灯片](https://docs.google.com/presentation/d/1wOh-ASGshgc1Ivkoyaz9zGpVGTxX9LDMZQB4-eXOBP4/edit#slide=id.g122e31b6546_0_0)上找到这些例子。

比较 API 之间的冗长性和复杂性很有帮助。例如，在 Pandas 中实现简单的[字数统计](https://en.wikipedia.org/wiki/Word_count)作业[很简洁，只需通过链接方法就可以实现，而](https://github.com/winderai/bacalhau-landscape-analysis-benchmarks/blob/main/sample-code/word-count/pandas.ipynb)[的 Hadoop 实现](https://github.com/winderai/bacalhau-landscape-analysis-benchmarks/blob/main/sample-code/word-count/hadoop.ipynb)远没有这么直观，主要是因为它必然会使用[Map-Reduce 范式](https://en.wikipedia.org/wiki/MapReduce)。

[安装说明](https://github.com/winderai/bacalhau-landscape-analysis-benchmarks#sample-code)指导您完成安装过程，如果您想自己运行示例，请尝试了解简单的单节点安装是如何工作的。

### 基准

[代码库](https://github.com/winderai/bacalhau-landscape-analysis-benchmarks)也发布了基准脚本，这些脚本在大型数据集上运行并行工作负载，对其执行进行计时，并记录资源使用情况。浏览相关章节，熟悉[的粗糙边缘，安装过程](https://github.com/winderai/bacalhau-landscape-analysis-benchmarks#benchmarks)。

您可以选择生成一个[单节点](https://github.com/winderai/bacalhau-landscape-analysis-benchmarks/blob/main/installation/SINGLE-NODE.md)或[多节点](https://github.com/winderai/bacalhau-landscape-analysis-benchmarks/blob/main/installation/MULTI-NODE.md)集群。尝试*这两个*选项对于直接体验本地到集群的障碍以及面对安装 Hadoop 等框架的复杂性尤其有指导意义。

基准测试任务是处理包含+1.7B 个单词的数据集的[单词计数](https://en.wikipedia.org/wiki/Word_count)作业。下图报告了每个框架的基准运行时间，缺少一个条形意味着该工具不支持完全成熟的多节点设置(即只能垂直扩展)。整个环境的性能可能相差 10 倍，这是意料之中的，因为 Pandas 不是一个大数据工具，而且 Hadoop 不是为在单节点设置上表现良好而设计的。然而，令人惊讶的是，只有 Spark 和 Snowflake 提供了相当简单的设置，并结合了快速处理和非常低的资源使用。

 

查看[幻灯片以获得关于基准测试结果的完整报告](https://docs.google.com/presentation/d/1wOh-ASGshgc1Ivkoyaz9zGpVGTxX9LDMZQB4-eXOBP4/edit#slide=id.g11c07429d0d_0_1091)或深入[代码库](https://github.com/winderai/bacalhau-landscape-analysis-benchmarks)以启动集群并亲自运行基准测试。

## 主要发现

**通用**

*   现代框架必须是直观的，不能强加类似于 [MapReduce](https://en.wikipedia.org/wiki/MapReduce) 的拜占庭式方法(参见 Hadoop)。高级 API 是必须的。
*   安装往往是一个障碍，必须被剥离下来，让任何人都可以完成一个入门。
*   在这些技术中，声明性管道并不重要。
*   需要支持可重复的容器化作业。所有的框架都没有提供一种有效的方式来封装依赖关系([见 Spark/Hadoop](https://docs.google.com/presentation/d/1wOh-ASGshgc1Ivkoyaz9zGpVGTxX9LDMZQB4-eXOBP4/edit#slide=id.g13641fb0d49_8_0) )。
*   [本地到多节点集群之旅](https://docs.google.com/presentation/d/1wOh-ASGshgc1Ivkoyaz9zGpVGTxX9LDMZQB4-eXOBP4/edit#slide=id.g11c07429d0d_0_1178)相当艰难，需要额外的安装、配置和不同的代码编写方法。在前进的道路上，这一点必须克服。

**基准**

在多节点设置中，您可能需要[优化集群大小](https://docs.google.com/presentation/d/1wOh-ASGshgc1Ivkoyaz9zGpVGTxX9LDMZQB4-eXOBP4/edit#slide=id.g13641fb0d49_8_6)，这是一个令人失望的问题(参见 Dask、Hadoop)

*   测试过的普通配置:调整旋钮提高了性能，但增加了复杂性。现代工具必须提供开箱即用的体验([见雪花设置](https://docs.google.com/presentation/d/1wOh-ASGshgc1Ivkoyaz9zGpVGTxX9LDMZQB4-eXOBP4/edit#slide=id.g11cf604b99b_0_236))。
*   除了 Spark 之外，这些工具在处理非共享的大文件时性能很差(即努力实现并行化)，因此现实生活中的用例将需要额外的上游数据准备。
*   请记住，这些工具针对不同的使用情形进行了优化，不同的任务可能会产生不同的结果。不管怎样，有些人努力做到尽可能通用(见 Spark，Dask)。

## 接触

如果您的组织对这项工作感兴趣，我们很乐意与您交谈。请与温德公司的销售团队联系。我们可以聊聊如何帮助你。