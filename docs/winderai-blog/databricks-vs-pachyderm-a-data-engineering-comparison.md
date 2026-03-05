# 数据砖块与厚皮动物——数据工程的比较

> 原文：<https://winder.ai/databricks-vs-pachyderm-a-data-engineering-comparison/>

云德。AI 进行了一项研究，比较厚皮动物和数据砖块之间的差异。这两家公司在数据行业和机器学习行业都很突出。但是它们针对不同的用例提供不同的产品。现代的生产就绪型要求在数据不断发展、非结构化且庞大的情况下带来了重大挑战。本白皮书调查了他们各自主张的优势和劣势，以及他们如何应对这些挑战。

> 披露:这项工作部分由厚皮动物公司资助。

## 什么是厚皮动物？

Pachyderm 提供了数据基础，允许数据科学团队自动化和扩展他们的机器学习生命周期，同时保证可重复性。数据科学团队可以使用 Pachyderm，通过数据版本控制、管道和沿袭来自动化部分 ML 生命周期。Pachyderm 提供开源社区版和用户部署企业版。企业产品包括基于角色的访问控制(RBAC)、JupyterHub 集成以及云和本地部署能力。Pachyderm 的数据驱动自动化、Pb 级可扩展性和端到端可再现性使公司能够更快地进入市场，最大限度地降低数据处理和存储成本，并满足严格的数据治理需求。

Pachyderm 的核心主张是内置版本控制和血统的数据层。但是 Pachyderm 的管道也是有版本的。这不仅允许你[编排 ML 和数据工程管道](https://winder.ai/how-to-build-a-robust-ml-workflow-with-pachyderm-and-seldon/)，还允许你跟踪它们。通过合并管道，可以对任意有向无环图(DAG)进行编码。您还可以追踪数据流和执行，直到它们的源头。

Pachyderm 的技术是用 GO 编写的，通常部署在 Kubernetes 上。Pachyderm 可以在相同的环境中实例化您的管道容器。Pachyderm 通过将数据存储在各自的 blob 存储中来支持所有主要的云平台，如 S3、Azure Blob 存储、谷歌云存储等。

Pachyderm 提供了一个 S3 接口，以便数据科学家可以将 Pachyderm 视为一个数据湖，只是完整的数据版本控制功能可以减少意外事件。这有助于数据科学家，因为他们可以继续使用他们已经知道的相同工具。

Pachyderm 的血统特性有助于数据科学家和企业保持对数据的控制。Pachyderm 记录每个数据事件，包括数据源、执行代码和数据接收器。这些元数据提供了数据沿袭，实现了可再现性，并有助于审计。

你可以在 Pachyderm.com 了解更多。

![An overview of Pachyderm's architecture.](img/5d20fe715181f5df242ea7ada9de150c.png)

图片由 Pachyderm 公司提供。

## 什么是数据块？

Databricks 是 Apache Spark 创始人创建的基于 Apache Spark 的分析平台。Spark 是一个著名的全面的基于 Scala/JVM 的数据工程框架，能够处理大量数据。许多云提供商将 Databricks 集成到他们的产品中，包括 AWS、GCP 和 Azure。Databricks 提供一键式配置、简化的工作流程和交互式工作空间。Databricks 旨在简化数据科学家、管道工程师和业务分析师之间的协作。

Spark 的目标是通过依赖内存计算来超越 Hadoop，同时保持与 Hadoop 分布式文件系统(HDFS)的兼容性。因此，它特别适合在集群中分布许多小型 map-reduce 风格的作业。

随着时间的推移，Databricks 已经扩展了它的支持，以满足现代需求。它通过支持 Python 和 R 等语言扩展了 Spark 生态系统，并为 ML(即 MLLib)和数据湖功能(即 Delta Lake)构建了几个开源库。

虽然分析工作负载是核心重点，但 Databricks 已经扩展到其他领域。它们支持 ML 和数据分析项目的开发，尽管它们是以火花为中心的。即便如此，Spark 允许对大量数据运行强大的分析算法，即使是流数据。

得益于庞大的用户群，Spark 上的在线资源显著增长，社区也产生了许多资源。例如，[关于堆栈溢出的相关主题](https://stackoverflow.com/questions/tagged/apache-spark)现在已经越过了+72k 的问题标记。此外，围绕 Spark 还有各种各样的活动，如会议、聚会和网络研讨会[。自 2013 年以来，Databricks 成功举办了一年一度的会议，分享 Spark 的使用案例和经验。](https://databricks.com/events)

在大数据时代，Databricks 已经成为一个成功的企业分析系统。Spark 虽然功能强大，但管理起来很复杂。Databricks 通过提供由所有大型云供应商托管的托管服务，将 Spark 及其支持生态系统货币化。

找到更多关于 Databricks.com 的信息。

![An architectural overview of the Spark stack.](img/daf621d4c711afccd3124035d44e238f.png)

图片由 Databricks 公司提供。

## 平台比较

本文旨在强调厚皮动物和数据砖之间的主要差异。首先，两者都是稳定的、为企业客户提供良好支持的系统。此外，您可以在主要的云提供商和内部部署这两个平台。特别值得称赞的是 Pachyderm 在 Kubernetes 集群上的可部署性。虽然 Spark 也可以部署在 Kubernetes 上，但并不容易。

现代 ML 平台应该能够在数据以各种格式发展和出现的场景中表现良好。例如，非结构化数据变得越来越重要。不断发展的数据意味着需要拍摄不同版本的快照以及检查沿袭。此外，生产用例通常需要处理新的批次，而无需对整个数据集重新运行作业。因此，支持增量处理也很重要。最后但同样重要的是，系统应该提供令人惊叹的开发者体验。这里有一个比较的汇总表。

|  | **厚皮动物** | **数据块** |
| **结构化和非结构化数据** | 两者都完全支持 | 主要开发用于结构化数据 |
| **版本化&血统** | 内置的，硬血统，长期的 | 可选，元数据血统，非常短命 |
| **增量加工** | 内置的 | 附带警告 |
| **开发者体验** | 与语言无关，大量使用云 | 带 Python/R 包装器的 Scala/Java 核心，需要专业知识 |

## 预期使用案例

请务必注意每个解决方案的使用案例。Databricks 的目标是分析市场。Pachyderm 针对的是数据和数据处理市场。这些完全不同，你可能会说这是苹果和橘子的比较。你是对的。

### 迟钝的人

Pachyderm 的目标是可扩展的容器化机器学习工作负载，这些工作负载需要强大的血统保证。但是 Pachyderm 之前在 1.9 中通过 spouts 引入了流媒体支持。这允许您从 Kafka 和 RabbitMQ 等消息总线接收流数据，同时保留强大的血统。

如果你想要灵活性和出处，厚皮是一个很好的选择。

### 数据砖

Databricks 和 Spark 专为分析用例而设计。这就是为什么他们投资了如此多的 SQL 技术。但不要指望能够处理非结构化数据或机器学习。例如，Databricks 支持通过 Horovod 使用 Tensorflow/Pytorch，这是一个完全独立的集群管理器，与 Spark 无关。Spark 的机器学习库 MLLib 并不是很适合现代批量 ML 的需求。然而，它在流应用中是有用的。

如果您需要一个解决方案来处理复杂的分析需求，Databricks 是一个很好的选择。

## 结构化和非结构化数据

企业依赖[非结构化数据](https://www.komprise.com/glossary_terms/unstructured-data/)进行监管、分析和决策。非结构化数据还推动了分析、[机器学习](https://winder.ai/services/machine-learning/)、[强化学习](https://winder.ai/services/reinforcement-learning/)和商业智能的创新。根据 ITC 的数据，到 2025 年，非结构化数据量预计将增长到 1750 亿 TB。因此，对于拥有[人工智能产品](https://winder.ai/services/ai-product/)的企业来说，将正确的工具与他们的数据战略配对是非常重要的。

### 迟钝的人

Pachyderm 可以处理大规模的非结构化数据。它支持视频、音频、图像、非结构化文本或复杂的特定领域数据类型。它通过基于对象存储的混合方法、用于快照的写时复制风格的文件系统和用于索引的 Git 风格的数据库来实现这一点。

Pachyderm 利用云的 blob 存储的可扩展性为任何应用程序提供本机性能。

在管道中，Pachyderm 将数据存储库公开为本地目录。这使得在非结构化数据上引导转换管道更加容易，因为用户代码可以在本地访问文件。此外，这种方法减少了获取远程数据对外部库的依赖。因此，它简化了数据消耗，而无需使用任何特定的 API 或 SDK。

Pachyderm 可以管理由数十亿个文件和数 Pb 数据组成的大型数据集存储库。通过 incrementality 提供的重复数据删除仅存储更新的数据块，这使得它特别适合具有大量不断变化的数据的情况。

Pachyderm 也通过相同的接口处理结构化数据集，但非结构化文件是最大的区别。

### 数据砖

Delta Lake 是添加在本机云存储之上的存储层，用于提供版本控制和一致性。该工具检查所有传入的数据，并确保它与用户设置的模式相匹配。这有助于确保数据的可靠性和正确性。

Delta Lake 与 Apache Spark APIs 兼容。这使得它能够支持数据湖元数据等。Delta Lake 可以在本地服务器、云中或笔记本电脑等设备上运行。它与批处理或流数据源兼容。

然而，Databricks Delta Lake 是在考虑结构化数据的情况下开发的。它为用户提供了一个表格数据结构，用作数据框架。这适合所有类似表格的数据集，但对于非结构化数据来说是个问题。它将二进制数据存储为表记录，这导致了各种管理、可伸缩性和版本控制问题。

在 Delta Lake 很少使用非结构化数据，在撰写本文时，只有[个公开可用的示例](https://databricks.com/session_na21/image-processing-on-delta-lake)。

就数据连接性而言，Databricks spark 的一个好处是它有一大串[开箱即用的连接器](https://spark.apache.org/docs/latest/sql-data-sources.html)。与编写自己的连接器相比，这使得提取数据的任务变得非常容易。

## 版本和沿袭

### 迟钝的人

由于其类似 Git 的方法，Pachyderm 提供了一种自动和高效的方法来跟踪所有数据更改。它通过提交、分支和回滚来促进有效的协作。这实现了对跨管道阶段的所有数据和工件的完整审计跟踪，包括基于文件的版本控制的中间结果。

Pachyderm 将更改保存为分块格式的本机对象，而不是指针。这优化了存储成本，因为它只存储已更改的新区块。这种哲学也促进了“硬”血统。您不能更改提交历史记录，因为版本控制会被破坏。

它将更改快照成不可变的和持久的提交。和在 git 中一样，使用 Pachyderm，您可以扩展一个存储库并设计多方面的管道。也像 git 一样，您可以在提交过程中进行“时间旅行”,以查看以前的数据和运行。这在 ML 中很有价值，因为它允许您对实验元数据进行版本控制，比如数据旁边的超参数。

下面的屏幕截图描述了提交的详细信息。提交的唯一散列被链接到父提交散列以提供出处。

![An example of Pachyderm’s commit information.](img/ffc50872f34dde449c3a6e31cab10d4a.png)

厚皮动物提交信息的一个例子。

### 数据库

当您使用 Delta Lake 时，Databricks 默认启用版本控制，尽管您可以在没有它的情况下使用 Databricks Spark。Delta Lake 用时间戳和增量版本号标记每个写操作。您可以使用这些标记读取以前的时间点，如下面的代码片段所示。

```py
df = spark.read \
  .format("delta") \
  .option("timestampAsOf", "2022-02-01") \
  .load("/path/to/my/table") 
```

Delta Lake 将每次更改保存在[事务日志](https://github.com/delta-io/delta/blob/master/PROTOCOL.md)中，并且“描述历史”命令列出所有版本化的操作，指示它们的来源。这是一种元数据版本化方法，其中指针指向持久化数据中的完整行，并提供“软”血统。例如，可以强制改变元数据所指向的底层数据。

![An image depicting the output from Delta Lake's describe history command.](img/ecdd9845e8bc1761dd1ca0db02d09ee9.png)

Delta Lake 的描述历史命令的输出示例。

Delta Lake 的版本历史允许您回滚到数据表的前一个版本。这是撤销不想要的操作的理想方法，但是与厚皮动物相比，数据块中的历史范围在时间上是相当有限的。

虽然 Delta Lake 将提交元数据存储在一个单独的 JSON 日志文件中，但实际数据是跨分区保存的(在 parquet 文件中)。下图显示了增量表的内部结构。

![An image depicting the structure of Delta Lake's storage format.](img/5c2bf7d6966360772fb6062da5941f99.png)

三角洲湖数据存储格式

随着表的增长，内部文件的数量也会增长。随着分区数量的增加，Delta Lake 解析这些文件需要更长的时间。为了保持性能，Delta Lake 会定期压缩提交元数据，并在默认情况下删除对超过 30 天的提交的访问。因此，Delta Lake 将历史限制在过去 30 天内。可以更改此设置，但这会影响性能，不建议这样做。

另外，请注意，旧的数据拼花文件必须手动进行垃圾收集。默认情况下，这将删除七天前的所有数据。要在手动垃圾收集后访问 30 天的历史数据，您必须[将数据保留期从 7 天增加到 30 天](https://docs.databricks.com/delta/delta-batch.html#data-retention)。

由于缺少分块(见下一节),这可能导致大量重复数据，不仅增加分区的大小，还会占用过多的存储空间。出于性能原因，Databricks 建议尽量缩短数据保留期。

这些时间限制最终阻止了 Delta Lake 被用作数据的版本控制系统。对于真正的版本控制，只要项目需要，提交和数据就应该是持久的和可访问的。

## 增量处理

增量处理是一种允许您只处理新数据，而不是重新处理整个数据集的功能。这种方法消除了重新处理未更改数据的需要，从而提高了效率。

### 迟钝的人

在 Pachyderm 中，您通过配置构建一个多步骤数据摄取管道作为 DAG。您将指定各种设置，例如用户代码(或容器)和输入数据源。Pachyderm 管理[数据](https://docs.pachyderm.com/latest/concepts/pipeline-concepts/datum/)中数据的计算。

一个数据是一个厚皮抽象，使管道处理优化。它仅作为管道处理属性存在，描述在作业的每个实例化中处理的数据。通过控制数据的大小，Pachyderm 只需要处理已经改变的数据。

例如，如果您将每个数据想象为一个文件，那么 Pachyderm 会将每个文件视为一个独立的处理作业。这有助于可伸缩性，也允许 Pachyderm 通过只重新处理已更改的文件来生成新的输出。

得益于数据概念，厚皮动物只能辨别和处理最新添加的内容。增量处理最大限度地减少了计算时间，并允许更好地利用资源。

### 数据砖

Spark 批处理作业对不可变的内存数据结构(例如，RDD、数据帧)进行操作。这意味着 Delta Lake 必须将分区加载到内存中来执行任务。通过仔细管理数据，用户可以在一定程度上控制分区的大小，但是内存不足的异常是常见的。

用户还可以通过精心设计的查询在批处理过程中逐步实现。例如，如果表有时间戳，您可以只对有新时间戳的记录进行操作。但这并不是现成的，而是由用户来实现。

Spark streaming 提供了[个检查点](https://docs.databricks.com/spark/latest/structured-streaming/production.html#recover-from-query-failures),用于在任务停止时指向处理应该恢复的位置。它还提供了基于单个消息启动小型作业的触发器。

但这两者都不是递增的。检查点和触发器都不会对已经完成的工作进行重复数据消除。作为一个用户，你不能控制增量发生的水平。例如，如果二进制 blob 中的一个字符发生了变化，则必须重新计算整个记录。

三角洲湖方法需要更复杂的数据和软件工程来识别和利用增量数据。用户需要定义特定的查询来只处理增量数据，并且没有跟踪增量数据的特定子集是否已经被处理。

## 语言支持

在您的 [MLOps 平台](https://winder.ai/services/mlops/)上运行任何语言或代码的能力增强了灵活性。通过利用工作的最佳框架，它使您的团队能够更快地达到生产级别的可交付成果。当您的企业试图实现标准化时，它也会有所帮助。

### 迟钝的人

厚皮动物天生是语言不可知的。它执行任何容器化的程序。因此，您可以利用最适合您的用例的任何工具，甚至是 Spark！

### 数据砖

虽然官方的 Spark 库支持 Python 和 R，但它们只是底层 Scala/Java 核心的包装器。使用包装器降低了开发人员的体验，因为它将用户束缚在 Java 的工作方式上。它只实现了 Scala API 的一个子集。

排除错误带来了额外的困难。在使用 PySpark(Spark 的流行 Python 库)时，您可能遇到的大多数错误都会显示一个包含 Python 和 Java 异常的堆栈跟踪(在这里找到一些[示例](https://towardsdatascience.com/pyspark-debugging-6-common-issues-8ab6e7b1bde8))。这给 ML 从业者理解 Java API 增加了另一个负担。

## 开发者体验

开发人员体验是一个重要的特性，因为它定义了生产力的上限。如果常见的 ML 或 [MLOps](https://winder.ai/services/mlops/) 任务很难执行，这会给开发人员带来额外的工作负担。

### 迟钝的人

Pachyderm 推荐的部署目标是 Kubernetes，尽管您可以将其作为二进制文件运行。这意味着您可以部署到任何云或内部。企业版配有 UI 和托管笔记本，以提高可用性。

### 数据砖

Apache Spark 附带了一组 Web UI/用户界面，用于作业、阶段、任务、存储、环境、执行器和 [SQL](https://sparkbyexamples.com/spark/spark-web-ui-understanding/#sql) 。这允许对应用程序和集群的状态进行基本的监控。

与 Spark 集群的交互是通过 Spark CLI 进行的，这对快速开发非常有用。这也使得几乎不可能创建不可变的作业，除非您强制用户放弃使用本地 CLI。这也使得依赖管理变得困难，尤其是在使用 Python 时，因为您无法构建一个众所周知的工件。

您可以在独立模式下部署 Spark，这对于开发很有用，也可以使用集群管理器。Spark 通常与 Mesos 或 YARN 一起部署，但这些都过时了，而且难以操作——以至于它驱使大多数用户使用云管理的 Databricks 套件。Spark 还提供了一个 [Kubernetes 部署](https://spark.apache.org/docs/latest/running-on-kubernetes.html)模式，但是它的实现并不简单。

## 与其他工具的集成

### 迟钝的人

Pachyderm 的任意容器允许您开发可以与任何外部 ML 工具交互的代码。您甚至可以在 Pachyderm 中存储工件，并从外部工具引用它们。

过去，云德。人工智能已经展示了厚皮动物与谢顿的整合。单击此处了解有关该案例研究的更多信息。

我们将演示如何将 MLFlow 与 Pachyderm 集成，使用 Pachyderm 作为后台存储来增强血统。我们甚至有一个在厚皮动物体内运行 Spark 的例子！

Pachyderm 还在其存储库中维护了一组[示例集成，您可以通过搜索引擎找到更多。](https://github.com/pachyderm/pachyderm/tree/master/examples)

![A screenshot of the MLFlow UI with Pachyderm commit information.](img/933053341fa1411eb2222cad886aad81.png)

厚皮动物与 MLFlow 整合。

### 数据砖

MLFlow 是一个开源的元数据存储，带有一个用于日志记录和加载模型的专用 API 。默认情况下，它是作为一个库交付的，但是我们策划[我们自己的容器构建](https://github.com/combinator-ml/terraform-k8s-mlflow/pkgs/container/mlflow)。Databricks 在其云平台中提供了一个托管 MLFlow 实例，使其成为他们跟踪实验和管理 ML 工件的选定工具。

但是，只有 MLflow 和 Spark 不提供沿袭。例如，您可以在 MLFlow 不知道的情况下删除或修改工件。这需要用户建立和使用合理的工作实践。

也没有与数据或代码的直接链接，这意味着您必须自己管理可再现性。

单独来说，MLFlow 更适合单个数据科学家或者小团队做一系列实验。较大的团队会希望关注可重复性和沿袭特性。

尽管有这些缺点，MLFlow 还是一个非常有用的工具。它已经成为事实上的开源 ML 元数据管理解决方案，我们广泛使用它，甚至在 Pachyderm 上也使用它！

如果您愿意将 Spark 和 Databricks 套件的其余部分添加到 Spark 代码中，那么它们可以与其他工具集成。您还可以通过 MLFlow 的 API 与它进行交互。但是你不能像在 Pachyderm 上那样运行任意代码，所以它不够灵活。

## 加工图的规格

当工程师开发他们的数据管道架构时，他们将处理任务指定为有向无环图(DAG)。将工作负载分割成小的、分片的任务可以更容易地横向扩展工作。但是你如何在厚皮动物和数据砖中做到这一点呢？

### 迟钝的人

在厚皮动物中，你可以完全控制 DAG。您可以自由决定如何分割数据以及运行什么代码。这是简单而明显的——两个伟大的品质。

![An example of a Pachyderm DAG.](img/751294164cbc1d8fea0ad35e654a89a3.png)

来自我们的厚皮动物-谢顿整合案例研究的厚皮动物 DAG 示例。

### 数据砖

Spark 通过您编写的代码形成 DAG。所以虽然你有控制权，但是很难控制，因为编译器决定了 DAG 的样子。当你使用 PySpark 时，这个过程会更加混乱。

但这确实使 Spark 有了根本的不同。Spark 懒洋洋地跑着 DAG。它会等待，直到请求输出，并通过 DAG 自我反馈。

这在分析中很有用，因为这意味着它只运行 DAG 中有相关作业的部分，从而优化利用率。

但是它会给正在处理的数据增加显著的延迟。这也意味着，如果作业是批处理的，您仍然需要第三方调度程序来触发作业。在 Spark 流中，您可以使用触发器。

但更重要的是，这使得控制并行化变得非常困难。如果你有一个非常大的任务，购买一台更大的机器通常比试图修改代码以更好地共享数据更容易。内存管理是首要问题。

![An image of a Spark DAG.](img/5e3d93cebd4bb1d3034f9529f75b707c.png)

Spark DAG 可视化示例。

### 哲学上的根本差异

这里有两种根本不同的哲学。在厚皮动物中，数据驱动计算。在数据块中，输出驱动它。在厚皮动物中，你控制切割。在数据块中，编译器试图处理它。

如果您主要有类似 SQL 的工作负载，这些工作负载适合优化，那么我建议编译器控制的 DAG 生成是有意义的。但是对于特定的 ML 工作负载，我们更喜欢控制。

## 其他功能

### 笔记本托管

这两种产品现在都支持 Jupyter 笔记本托管，使得与平台的交互更加容易。

Pachyderm 笔记本电脑利用在内部运行的定制 JupyterHub 部署，并根据您的 Pachyderm 集群进行验证。因此，Pachyderm 使其用户能够从数据版本和血统中受益，从而实现可再现性。同时，ML 工程师可以更快、更有效、更安全地生产这些管道。

Databricks 还提供共享和交互式 Databricks 笔记本，用于实时共同创作、评论和自动化版本控制，在提供控制的同时简化了协作。此外，其 API 和作业调度程序使数据工程师能够快速自动化复杂的管道，而业务分析师可以通过交互式仪表盘直接访问结果。

### SQL 优化

Databricks 针对 [SQL 工作负载](https://databricks.com/product/photon)进行了优化。Photon 增加了 CPU 级别的数据和指令的并行处理。其他增量引擎组件包括增强的查询优化器和缓存层。另外，Photon 可以连接到任何一个 [SQL 数据源](https://docs.databricks.com/runtime/photon.html)。

## 摘要

Pachyderm 和 Databricks 都提供了大量功能来简化您的数据科学项目，并生成生产级别的高质量交付成果。但两者都有其独特之处。使用下面的指南来决定哪一个适合你。

如果您符合以下条件，厚皮动物可能是最佳选择:

*   处理非结构化数据
*   重视长期、艰苦的传承
*   首选容器形式的通用任意处理器
*   首选云原生解决方案
*   你有更多通用的 ML 用例
*   想要使用任意的外部框架

如果您符合以下条件，Databricks 可能是最佳选择:

*   具有分析工作负载，并且正在处理结构化数据
*   通过 SQL 完成大量工作
*   您希望通过 GCP/AWS/Azure 提供的托管服务减轻管理负担
*   热爱 Scala/Java