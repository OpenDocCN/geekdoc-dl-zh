# 最佳数据沿袭工具

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/best-data-lineage-tools>

数据是每个机器学习模型的主要部分。你的模型只有在有数据的情况下才是好的，没有数据你根本无法建立一个模型。因此，只使用准确可信的数据来训练模型是有意义的。

在运营过程中，许多组织都需要改进数据跟踪和传输系统。这种改进可能会导致:

*   发现数据中的错误，
*   实施有效的变革以降低风险。
*   创建更好的数据映射系统。

一些作家说数据是新的石油。就像石油成为汽车的燃料一样，数据也必须经历从原始数据到模型组件甚至简单可视化的过程。

数据科学家、数据工程师和机器学习工程师依靠数据来构建正确的模型和应用程序。它有助于理解数据在用于构建精确模型之前的必要旅程。

这个数据之旅的概念实际上有一个真实的名字— [数据血统](https://web.archive.org/web/20221206071337/https://www.imperva.com/learn/data-security/data-lineage/#:~:text=Data%20lineage%20is%20the%20process,%2C%20what%20changed%2C%20and%20why.)。在本文中，我们将探索数据血统在机器学习中的意义，并查看几个付费和开源的数据血统工具。

## 机器学习中的数据血统是什么？

机器学习中的数据谱系描述了数据从收集到使用的过程。它展示了从最终消费前理解、记录、可视化变化和转换数据的过程。它是数据如何转换、具体转换了什么以及为什么转换的详细过程。

在机器学习中，模型通常是对其进行训练的数据的压缩版本。当您知道数据集的数据谱系时，就更容易实现可重复性。

数据沿袭工具帮助您可视化和管理数据的整个过程。选择数据沿袭工具时，您应该寻找一些核心特性:

1.  **可追溯性**:追踪和验证数据历史的能力。这非常重要，它有助于确保您拥有高质量的数据。
2.  **不变性**:不变性给数据沿袭工具带来了信任。不变性意味着您可以在做出更改后返回到数据集的先前版本。
3.  开源:开源数据沿袭工具的优点是可以免费使用，并且在不断改进。
4.  **集成**:数据之旅涉及许多阶段和工具，从存储到不同的转换(争论、清理、摄取等。).因此，数据沿袭工具应该能够轻松地与第三方应用程序集成。
5.  **版本化**:一个好的数据沿袭工具应该跟踪数据的不同版本，并在各种转换和调优过程中对变化进行建模。
6.  **协作**:对于远程数据科学团队来说，在共享数据上[协作很重要](/web/20221206071337/https://neptune.ai/blog/best-software-for-collaborating-on-machine-learning-projects)。此外，了解谁对数据进行了更改以及为什么要进行更改也很重要。
7.  **元数据存储**:数据沿袭工具应该包括一个[元数据存储](/web/20221206071337/https://neptune.ai/blog/ml-metadata-store)。
8.  **大数据处理**:很多机器学习模型都需要大数据，因此数据血统工具应该能够高效地处理和加工大数据。

现在，让我们看看机器学习中一些最好的数据谱系工具。

让我们从不同工具的汇总表开始，我们将在下面更详细地讨论每一个工具。

海王星

### [Neptune](/web/20221206071337/https://neptune.ai/product) 是 MLOps 的元数据存储，为运行大量实验的研究和生产团队而构建。

它为您提供了一个中心位置来记录、存储、显示、组织、比较和查询机器学习生命周期中生成的所有元数据。个人和组织使用 Neptune 进行实验跟踪和模型注册，以控制他们的实验和模型开发。

**特性:**

Neptune 允许您显示和探索数据集的元数据。使用 Neptune 的名称空间或基本日志记录，您可以[存储和跟踪数据集元数据](https://web.archive.org/web/20221206071337/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#data-versions)，例如数据集的 md5 散列、数据集的位置、类列表、特性名称列表(针对结构化数据问题)。

*   Neptune 允许你通过[记录模型检查点](https://web.archive.org/web/20221206071337/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#model-checkpoints)来版本化你的模型。
*   Neptune 有一个[可定制的 UI](https://web.archive.org/web/20221206071337/https://docs.neptune.ai/you-should-know/core-concepts#web-interface-neptune-ui) ，允许你比较和查询所有的 MLOps 元数据。
*   要了解更多关于 Neptune 的数据血统和版本，请点击这里查看。

MLFlow

### MLflow 是一个用于构建、部署和管理机器学习工作流的开源平台。它旨在标准化和统一机器学习过程。它有四个主要部分来帮助组织 ML 实验:

**MLflow Tracking** :这允许您记录您的机器模型训练会话(称为运行),并使用 Java、Python、R 和 REST APIs 运行查询。

1.  **MLFlow 模型**:模型组件提供了一个标准单元，用于封装和重用机器学习模型。
2.  **ml flow Model Registry**:Model Registry 组件让您集中管理模型及其生命周期。
3.  **MLflow 项目**:项目组件封装了数据科学项目中使用的代码，确保其可以轻松重用，实验可以重现。
4.  **特性:**

MLflow Model Registry 为组织提供了一套 API 和直观的 UI，用于注册和共享新版本的模型，以及对现有模型执行生命周期管理。

*   MLflow Model Registry 与 MLflow tracking 组件一起使用，它允许您追溯生成模型和数据工件的原始运行以及该运行的源代码版本，从而为所有模型和数据转换提供完整的生命周期谱系
*   MLflow 模型注册组件与 Delta Lake Time Travel 集成:
*   为大数据存储提供数据存储(数据湖)。
    *   当存储在增量表或目录中时，自动对存储在数据湖中的数据进行版本控制。
    *   允许您使用版本号或时间戳获取数据的每个版本。
    *   允许您在意外写入或删除错误数据时进行审核和回滚。
    *   你可以复制实验和报告。
    *   要了解更多关于 MLflow 的信息，请查看 [MLflow 文档](https://web.archive.org/web/20221206071337/https://www.mlflow.org/docs/latest/index.html)。

检查一下[海王星和 DVC](/web/20221206071337/https://neptune.ai/vs/dvc) 有什么不同。

迟钝的人

### [Pachyderm](https://web.archive.org/web/20221206071337/https://www.pachyderm.com/) 是一个数据平台，它将数据谱系与 Kubernetes 上的端到端管道混合在一起。它为数据科学项目和 ML 实验带来了数据版本控制的管道层。它进行数据采集、摄取、清理、管理、争论、处理、建模和分析。

Pachyderm 有三种版本:

社区版:一个你可以在任何地方使用的开源平台。

1.  企业版:完整的版本控制平台，具有高级功能、无限的可扩展性和强大的安全性。
2.  Hub Edition:社区版和企业版的结合。它减轻了你自己管理 Kubernetes 的工作量。
3.  **特性:**

像 Git 一样，Pachyderm 帮助您找到数据源，然后在模型开发过程中对数据进行处理时对其进行跟踪和版本化。

*   厚皮动物给出了从数据源到当前状态的日期。
*   Pachyderm 允许您快速审计数据版本跟踪和回滚中的差异。
*   Pachyderm 提供大数据的 dockerized MapReduce。
*   Pachyderm 将所有数据存储在 Minio、AWS S3 或谷歌云存储等中央存储库中，并拥有自己的专用文件系统，称为 Pachyderm 文件系统或 PFS。
*   当数据集发生变化时，Pachyderm 会对所有相关数据进行更新。
*   它管理和记录对数据进行的转换。
*   要了解更多关于厚皮动物的信息，请查阅[厚皮动物文档](https://web.archive.org/web/20221206071337/https://www.pachyderm.com/data-lineage/)。

查查[海王星和 MLflow](/web/20221206071337/https://neptune.ai/vs/mlflow) 有什么区别。

真实数据

### [Truedat](https://web.archive.org/web/20221206071337/https://www.truedat.io/) 是一个开源的数据治理工具，提供从模型开始到结束的端到端数据可视化。

**特性:**

Truedat 从业务和技术的角度提供了对数据的端到端(从起点到终点)理解。

*   Truedat 允许您通过可配置的工作流程组织和丰富信息，并监控政府活动。
*   使用数据湖管理选项，您可以请求、交付和使用受治理的数据。
*   Truedat 提供了对数据随时间变化的洞察。

*   Truedat 提供数据治理。
*   Truedat 提供了对象沿袭跟踪、沿袭对象过滤和数据的时间点可见性。
*   Truedat 提供数据的用户/客户端/目标连接可见性。
*   Truedat 提供了关于数据之旅的视觉和文本沿袭视图。
*   Truedat 提供数据库变更影响分析。
*   Truedat 可以集成在亚马逊 S3、亚马逊 RDS、Azure、大查询、Power BI、MySQL、PostgresSQL、Tableau 等上面。
*   要了解更多关于 Truedat 的信息，请查看 [Truedat 文档](https://web.archive.org/web/20221206071337/https://docs.truedat.io/)。

CloverDX

### [CloverDX](https://web.archive.org/web/20221206071337/https://www.cloverdx.com/) 支持多个数据流程的组织，改进并自动化透明的数据转换。CloverDx 集合了转换和工作流的设计，包括编码能力。

CloverDX 通过开发人员友好的可视化设计器为您的数据集提供数据沿袭。它有利于自动化与数据相关的任务，如数据迁移，而且速度很快。

**特性:**

CloverDX 为数据工作流提供了透明度和平衡。

*   CloverDX 托管其他数据质量工具。
*   CloverDX 为您的数据进行错误跟踪和解决。
*   可回收数据操作。
*   自给自足的数据操作。
*   CloverDX 可以独立使用，也可以嵌入使用。
*   CloverDX 集成了 RDBMS、JMS、SOAP、LDAP、S3、HTTP、FTP、ZIP 和 TAR。
*   SentryOne 文档

### SentryOne Document 为您提供强大的工具，确保您的数据库得到持续、准确的记录。此外，数据沿袭分析功能通过提供数据来源的可视化表示，帮助您确保合规性。

**特性:**

通过清晰显示整个环境中的数据依赖关系的可视化显示来跟踪数据沿袭。

*   记录数据源，包括 SQL Server、SQL Server Analysis Services (SSAS)、SQL Server Integration Services(SSIS)、Excel、Power BI、Azure Data Factory 等。
*   借助易于访问的云或软件解决方案，轻松管理文档任务和查看日志。
*   DVC

### [DVC](https://web.archive.org/web/20221206071337/https://dvc.org/) 是一个机器学习项目的开源版本控制系统。DVC 通过始终如一地维护最初用于运行实验的输入数据、配置和代码的组合来保证再现性。

它利用现有的工具，如 Git 和各种 CI/CD 应用程序。它可以被分组为组件。

**特性:**

DVC 提供了分布式版本控制系统的所有优势——无锁、本地分支和版本控制。

*   DVC 运行在任何 Git 存储库之上，兼容任何标准的 Git 服务器或提供商(GitHub、GitLab 等)。
*   数据管道描述了模型和其他数据工件是如何构建的，并提供了一种有效的方法来复制它们。认为“数据和 ML 项目的 Makefiles”做得对。
*   它可以与亚马逊 S3、微软 Azure Blob 存储、谷歌驱动、谷歌云存储、阿里云 OSS、SSH/SFTP、HDFS、HTTP、网络附加存储或磁盘集成来存储数据。
*   DVC 有一种内置的方式将 ML 步骤连接到 DAG 中，并端到端地运行整个管道。
*   DVC 处理中间结果的缓存，如果输入数据或代码是相同的，它不会再运行一个步骤。
*   查查[海王星和 MLflow](/web/20221206071337/https://neptune.ai/vs/mlflow) 有什么区别。

齿条

### Spline (SPark LINEage)是一个免费的开源工具，用于自动数据谱系跟踪和数据管道结构。它最初是为 Spark 设计的，但该项目已经扩展到容纳其他数据工具。

样条有三个主要部分:

Spline 服务器:它充当通过 Producer API 从 spline 代理获得的沿袭元数据的中央存储库，并将其存储在 ArangoDB 中。

1.  Spline 代理:Spline 代理监听 spark 活动，然后跟踪并记录从各种 Apache 数据管道(spark 活动)获得的沿袭信息，并使用 REST 或 Kafka 通过其生产者 API 以标准格式将其发送到 Spline 服务器。Spline 代理是一个 Scala 库，可以独立使用，不需要 Spline 服务器，这取决于您的使用情况。
2.  Spline UI:它是一个 docker 图像或 WAR 文件，用作 Spline 的 UI。Spline 消费者 API 端点可以从 Spline UI 上的用户浏览器直接访问。
3.  **特性:**

Spline 跟踪并记录作业依赖关系，然后继续创建它们如何交互以及每个转换中发生的转换的概述。

*   Spline 可以在 Azure 数据块中使用。
*   样条函数很好地处理大数据，并且易于使用。
*   Spline 有一个显示沿袭信息的可视化界面。
*   Spline 使与业务团队的沟通变得容易。
*   Spline 可以很好地处理结构化数据 API，例如 SQL、数据集、数据框等。
*   要了解更多关于 Spline 的信息，请查阅 [spline 文档](https://web.archive.org/web/20221206071337/https://absaoss.github.io/spline/)。

结论

## 如您所见，如果您想做可重复的、高质量的工作，数据血统是非常重要的。浏览本文中列出的工具，看看什么最适合您的用例。感谢阅读！

As you can see, data lineage is very important if you want to do reproducible, high-quality work. Look through the tools listed in this article and see what fits your use case best. Thanks for reading!