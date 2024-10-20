# 如何解决 MLOps 堆栈的数据接收和功能存储组件

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/data-ingestion-and-feature-store-component-mlops-stack>

正如*数据科学*领域的每个从业者所知，**数据** **是机器学习**的主要燃料。值得信赖的数据源和高质量的数据收集和处理可以支持大量潜在的 ML 用例。但是拥有一个治理良好的[数据仓库](https://web.archive.org/web/20230106144203/https://aws.amazon.com/data-warehouse/#:~:text=A%20data%20warehouse%20is%20a,typically%20on%20a%20regular%20cadence.)需要组织中的每个团队全心全意地照顾和管理他们产生、摄取、分析或利用的每个数据点。**数据质量责任分散到每个人身上**。它不仅仅依赖于数据工程团队。

![Data quality characteristics](img/c46001e8993e189536a7c80cce725506.png)

*Main properties of Data Quality | [Source](https://web.archive.org/web/20230106144203/https://www.aqclab.es/index.php/en/data-quality-iso-25012)*

如今组织中最常见的数据架构是 [Lambda 架构](https://web.archive.org/web/20230106144203/https://en.wikipedia.org/wiki/Lambda_architecture)。它的特点是有独立的批处理和流管道将数据接收到数据湖中，数据湖由一个*着陆*或*原始*阶段组成，其中 [ELT](https://web.archive.org/web/20230106144203/https://www.ibm.com/cloud/learn/elt) 进程转储原始数据对象，如事件或数据库记录转储。

这些原始数据随后被接收到更有组织的数据湖表(例如 Parquet 文件)中，然后被充实到*数据仓库*中。进入数据仓库的数据是不同业务领域的逻辑组织信息，称为[数据集市](https://web.archive.org/web/20230106144203/https://www.oracle.com/autonomous-database/what-is-data-mart/)。这些数据集市很容易被数据分析师查询和被业务涉众探索。每个数据集市可以与不同的业务单元或产品领域相关联(*营销、订阅、注册、产品、用户…)* 。

![Example of a typical Data Architecture](img/f612da0195de0711b192d3cdc3aec3be.png)

*Example of a typical Data Architecture in Google Cloud Platform | [Source](https://web.archive.org/web/20230106144203/https://blog.miraclesoft.com/data-foundation-with-modernized-data-lake-data-warehouse/)*

还有其他的参考架构模式，例如, [Kappa](https://web.archive.org/web/20230106144203/https://hazelcast.com/glossary/kappa-architecture/#:~:text=What%20Is%20the%20Kappa%20Architecture,with%20a%20single%20technology%20stack.) 或 Delta，后者得到了商业产品的大量支持，例如【】Databricks 和【】Delta Lake 。

这些基础数据架构模式为分析工作负载铺平了道路。 [OLAP](https://web.archive.org/web/20230106144203/https://en.wikipedia.org/wiki/Online_analytical_processing) 大数据数据库和处理引擎，如 [Spark](https://web.archive.org/web/20230106144203/https://spark.apache.org/) 和 [Dask](https://web.archive.org/web/20230106144203/https://www.dask.org/) 等，已经实现了存储和计算硬件的解耦，允许数据从业者与海量数据进行交互，以进行*数据分析*和*数据科学*。

随着 [MLOps](/web/20230106144203/https://neptune.ai/blog/mlops) 、DataOps 的兴起，以及*软件工程*在生产*机器学习*中的重要性，出现了不同的创业公司和产品来解决服务特征的问题，如 [Tecton](https://web.archive.org/web/20230106144203/https://www.tecton.ai/) 、 [HopsWorks](https://web.archive.org/web/20230106144203/https://www.hopsworks.ai/) 、[盛宴](https://web.archive.org/web/20230106144203/https://feast.dev/)、 [SageMaker 特征店](https://web.archive.org/web/20230106144203/https://aws.amazon.com/sagemaker/feature-store/)、 [Databricks 特征店](https://web.archive.org/web/20230106144203/https://docs.databricks.com/applications/machine-learning/feature-store/index.html)、[顶点 AI 特征店](https://web.archive.org/web/20230106144203/https://cloud.google.com/vertex-ai/docs/featurestore)

此外，每个大规模从事生产数据科学的公司，如果没有使用前面提到的工具之一，都已经建立了自己的内部功能商店(例如，[优步是第一个发布自己构建 ML 平台方法的公司](https://web.archive.org/web/20230106144203/https://www.uber.com/blog/michelangelo-machine-learning-platform/)，随后是 Airbnb)。

在本文中，我们将解释特性商店解决的一些概念和问题，就像它是一个内部平台一样。这是因为我们认为理解底层组件以及它们之间的概念和技术关系更容易。我们不会深究商业产品。

我们还将讨论构建和购买之间的紧张关系，这是当今业界从业者的热门话题，以及实现这一决策的最佳方式。

## 什么是功能商店？

去年，一些[博客和 ML 世界中有影响力的人将 2021](https://web.archive.org/web/20230106144203/https://www.datanami.com/2021/01/19/2021-the-year-of-the-feature-store/) 命名为功能商店年。我们将在下一节讨论这背后的原因。但是，什么是功能商店呢？

[Featurestore.org](https://web.archive.org/web/20230106144203/https://www.featurestore.org/)给出的简短定义是:

> *“一个用于机器学习的* ***数据管理层*** *，允许共享&发现功能并创建更有效的机器学习管道。”*

这很准确。简单地扩展一些细节，特征库由一组技术、架构、概念和语义组件组成，这些组件使 ML 从业者能够创建、摄取、发现和获取用于进行离线实验和开发在线生产服务的特征。

### 功能存储的组件

![Components of a Feature Store](img/2614ddb62321793cb34f134093acf150.png)

*Components of a feature store | [Source](https://web.archive.org/web/20230106144203/https://www.featureform.com/post/feature-stores-explained-the-three-common-architectures)*

我们应该开始定义什么是特征向量，因为它是特征库处理的核心实体。

*   **特征向量:**这是一个数据元素，包含一个实体标识符和一组在某个时间点描述该元素的属性或特征。例如，实体标识符可以是一个**用户 ID** ，并且属性可以包含以下值:(*时间 _ 自 _ 注册、n _ 购买、ltv _ 值、is _ 免费 _ 试用、平均 _ 购买 _ 每月、累计 _ 购买、最后 _ 购买 _ts 等)*

现在，我们来解释托管这些特征向量的不同存储组件:

*   **离线存储:**这是一个分析数据库，可以接收、存储和提供离线工作负载的特征向量，如数据科学实验或批量生产作业。通常，每一行都包含一个由实体 ID 和给定时间戳唯一标识的特征向量。这个组件通常具体化为 S3、红移、BigQuery、Hive 等。

*   **在线商店:**也称为*热数据*，该存储层旨在为低延迟预测服务提供功能。该数据库现在用于以毫秒级速度提取要素。Redis、DynamoDB 或 Cassandra 是扮演这一角色的常见候选人。键值数据库是最好的选择，因为在运行时不经常需要复杂的查询和连接。

*   **特征目录或注册表:**理想情况下，这是一个很好的 UI，可以发现特征和训练数据集。

*   **特性商店 SDK:** 这是一个 Python 库，抽象了线上和线下商店的访问模式。

*   **元数据管理:**该组件用于跟踪来自不同用户或管道的访问、摄取过程、模式更改以及这类信息。

*   **离线和在线服务 API:** 这是一个代理服务，位于 SDK 和在线及功能硬件之间，以方便功能访问。

在下面的时间顺序图中，我们可以看到自 2017 年优步发布其著名的[米开朗基罗](https://web.archive.org/web/20230106144203/https://www.uber.com/blog/michelangelo-machine-learning-platform/)以来围绕 feature store 的关键里程碑的总结。几年后，在几个商业和操作系统产品推出后，我们已经看到行业从业者广泛接受了功能商店的概念。一些组织如 featurestore.org 的 T2 和 T4 的 mlops.community 应运而生。

![ Feature Store Milestones chart](img/2f5841be99c631cb465c67041ef2c938.png)

*Feature Store Milestones | [Source](https://web.archive.org/web/20230106144203/https://medium.com/data-for-ai/feature-store-milestones-cca2bafe6e9c)*

在与 MLOps 的关系中，特性存储本身受到影响，并影响堆栈的其他组件，如数据仓库、数据湖、数据作业调度程序、生产数据库等。也是。我们将在后面详细讨论这种关系，即功能商店在 MLOps 框架的大图中处于什么位置？

现在，让我们讨论一下 ML 工程师在产品特征工程方面面临的一些主要问题。

## 围绕功能商店的争论

### 特征摄取和获取的标准化

在存在适当的要素存储之前，每个数据科学团队使用非常不同的工具存储和获取要素。这些工作传统上被视为数据工程管道的一部分。因此，围绕这些工作的库、SDK 和工具是数据工程师使用的。根据团队的专业知识、成熟水平和背景，他们可以是非常多样化的。

例如，您可以在同一组织中看到以下情况:

*   **团队 A:** 团队对数据工程不是很了解。他们使用裸露的熊猫和带有 psycopg 连接器的 SQL 脚本在 Redshift 中存储离线特性，使用 boto 在 DynamoDb 中存储在线特性。
*   **团队 B:** 团队成熟，自主。他们建立了一个库，使用 sqlalchemy 或 PySpark 进行大数据作业，抽象出与几个数据源的连接。他们还有自定义的包装器，用于向 DynamoDb 和其他热门数据库发送数据。

这在大型组织中非常典型，在这些组织中，ML 团队不是完全集中的，或者 ML 跨团队不存在。

在不同项目中使用相同数据库的团队倾向于围绕它们构建包装器，这样他们就可以抽象连接器并封装公共实用程序或领域定义。这个问题已经被团队 b 解决了。但是团队 A 不太熟练，他们可能会开发另一个内部库来以更简单的方式处理他们的特性。

这导致了团队之间的摩擦，因为他们想在整个组织中推行他们的工具。这也降低了团队的生产力水平，因为每个团队都在以自己的方式重新发明轮子，将开发人员与项目联系起来。

通过引入特性存储 SDK，两个团队可以利用相同的接口与 Redshift 和 DynamoDb 以及其他数据源进行交互。团队 A 的学习曲线会更陡，但他们会保持相同的操作标准。因此，总体生产率将会提高。这允许更好的特性治理。SDK 通常隐藏其他 API 调用来记录用户请求和版本数据集，允许回滚等。

大多数商业特性商店都提供特定的 SDK 来与他们的中心服务进行交互。例如，在下一个片段的[中，您可以看到如何构建一个从 Feast 获取特性的数据集。](https://web.archive.org/web/20230106144203/https://github.com/feast-dev/feast#5-build-a-training-dataset)

![Building a dataset ](img/4b88573516faf7e9548b1fc075e33acb.png)

*Build a dataset fetching features from Feast | [Source](https://web.archive.org/web/20230106144203/https://github.com/feast-dev/feast#5-build-a-training-dataset)*

这不仅对标准化特色商店运营有价值，而且对抽象线上和线下商店的硬件也有价值。数据科学家不需要知道离线存储是 BigQuery 还是 Redshift 数据库。这是一个很大的好处，因为您可以根据用例、数据等使用不同的源。

### 时间旅行数据

如果我们想预测一个用户是否会购买一个产品，我们必须建立一个包含直到那个特定时刻的特征的数据集。**我们需要非常小心不要引入未来的数据，因为这会导致** [**数据泄露**](https://web.archive.org/web/20230106144203/https://machinelearningmastery.com/data-leakage-machine-learning/) **。**但是如何？

如果我们将未来数据引入关于每个观察的训练数据集中，*机器学习*模型将学习不可靠的模式。当将模型投入实时生产时，它将无法访问相同的功能(除非你可以旅行到未来)，其预测能力将会退化。

回到产品购买预测的例子，假设您想要使用关于用户的特定特征，例如，保存在购物车中的商品数量。训练数据集将包含关于看到并购买产品的用户(正标签)和看到但没有购买产品的用户(负标签)的事件。如果您想使用购物车中的商品数量作为一个特性，那么您需要专门查询那些记录在同一个会话中添加到购物车中的每个商品的事件，并且就在 purchase/seen 事件之前。

![Time Travel in ML](img/5f7c4db67da23f4d512334afb0cac36e.png)

*Tecton: Time Travel in ML | [Source](https://web.archive.org/web/20230106144203/https://www.tecton.ai/blog/time-travel-in-ml/)*

因此，在构建这样一个数据集时，我们需要专门查询关于每个事件在那个时间点可用的**特征。有必要再现事件发生的世界。**

#### 如何有一个准确的图片？

*   **记录并等待**:你只需要记录特定的特性，比如*n _ cumulative _ items _ in _ the _ cart，*，然后我们就可以知道用户在那个时间点有多少商品。主要的缺点是这种特性收集策略需要时间来为用例收集足够的数据点。但另一方面，实现起来也很容易。

*   **回填** *:* 该技术基本上旨在在给定的时间点重建期望的特征。例如，通过查看记录的事件，我们可以在每次购买之前将所有商品添加到购物车中。然而，这可能会变得非常复杂，因为我们必须为每个要素选择时间窗口截止点。这些查询通常被称为**时间点**连接。

*   **快照 *:*** 基于定期转储生产数据库的状态。这允许在任何给定的时间点拥有特性，缺点是连续快照之间的数据更改将不可用。

### 用于生产的功能可用性

当一个新的 ML 用例被提出时，有经验的 ML 工程师倾向于在运行时(在线)考虑哪些特性是可用的。在大多数情况下，设计支持特性的系统是 ML 架构中最耗时的部分。

让最新的特征向量准备好被馈送给 ML 模型以进行预测不是一件容易的任务。涉及到许多组件，需要特别注意将它们粘合在一起。

生产中的特征可能来自非常不同的来源。它们可以在请求体参数中提供给算法，可以从特定的 API 中获取，从 SQL 或 NoSQL 数据库中检索，从 Kafka 主题事件中检索，从键值存储中检索，或者可以从其他数据中即时计算和导出。每一个都意味着不同程度的复杂性和资源容量。

#### 这些来源是什么？

1.  **请求车身参数**

这是接收用于预测的特征的最简单方式。获取这些特性并将它们传递给 ML 模型的责任被委托给推理 API Web 服务的客户或消费者。然而，这并不是最常见的特征供给方式。事实上，请求参数往往包含从其他来源获取特征向量所需的唯一标识符。这些通常是用户 id、内容 id、时间戳、搜索查询等。

2.  **数据库**

根据特性模式和延迟的可发展性需求，特性可以存在于不同的数据库中，如 Cassandra、DynamoDb、Redis、PostgreSQL 或任何其他快速 NoSQL 或 SQL 数据库。从在线服务获取这些功能非常简单。您可以使用任何 Python 库，如用于 DynamoDb 的 boto、用于 redis 的 pyredis、用于 PostgreSQL 的 psycopg2、用于 mysql 的 mysql-connector-python、用于 cassandra 的 cassandra-driver 等等。

数据库中的每一行都有一个主键或索引，在运行时可用于每个预测请求。其余的列或值将是您可以使用的功能。

为了填写这些表格，我们可以根据要计算的特征的性质使用不同的方法:

*   **批处理作业:**这些是计算密集型、繁重且“缓慢”的，这就是为什么它们只提供由它们需要多*新鲜*所定义的特定类型的功能。当构建不同的用例时，你会意识到不是每个模型都需要实时特性。如果你使用的是产品的平均评分，你不需要每秒计算一次平均值。像这样的大多数特征只需要一个日常计算。如果该功能需要高于 1 天的更新频率，您应该开始考虑批处理作业。

![ Batch processing example](img/59c1b1adf8ede3545e8da11f45f40523.png)

*An example of a batch processing | [Source](https://web.archive.org/web/20230106144203/https://datawhatnow.com/batch-processing-mapreduce/)*

谈到常见的技术堆栈，老朋友们开始为不同的目的和规模服务:

*   气流+ DBT 或 Python 是调度和运行这些作业的良好开端。
*   如果在分布式内存方面需要更大的规模，我们可以开始考虑 Kubernetes 集群来执行 Spark 或 Dask 作业。

一些流程编排工具的替代品是 Prefect、Dagster、Luigi 或 Flyte。看看[数据科学编排和工作流工具](/web/20230106144203/https://neptune.ai/blog/best-workflow-and-pipeline-orchestration-tools)的对比。

*   **流接收:** [](https://web.archive.org/web/20230106144203/https://medium.com/data-for-ai/building-real-time-ml-pipelines-with-a-feature-store-9f90091eeb4)需要流或(近)实时计算的功能对时间很敏感。需要实时功能的常见用例有欺诈检测、实时产品推荐、预测性维护、动态定价、语音助手、聊天机器人等。对于这样的用例，我们需要一个非常快速的数据转换服务。

![Building ML pipeline with Feature](img/c6e03fa1b5bb2973cc9db53e328bd3df.png)

*Building ML pipeline with Feature | [Source](https://web.archive.org/web/20230106144203/https://medium.com/data-for-ai/building-real-time-ml-pipelines-with-a-feature-store-9f90091eeb4)*

这里有两个重要的维度需要考虑——**频率**和**复杂度**。例如，计算单个交易的“当前价格与月平均价格的标准差”既是实时的，也是复杂的汇总。

![Feature Store Streaming Ingestion](img/fc411d4faea2a2f74b4397eb17d1f51c.png)

*Amazon SageMaker Feature Store Streaming Ingestion | [Source](https://web.archive.org/web/20230106144203/https://aws.amazon.com/es/blogs/aws-spanish/ingesta-de-streaming-con-amazon-sagemaker-feature-store-para-tomar-decisiones-respaldadas-por-ml-casi-en-tiempo-real/)*

除了有一个收集事件的流工具(Kafka)之外，我们还需要一个高速和可伸缩的(每秒处理任意数量的事件)功能即服务(如 AWS Lambda)来读取和处理这些事件。更重要的是，转换服务需要支持聚合、分组、连接、自定义函数、过滤器、滑动窗口，以便在给定的时间段内每隔 X 分钟或小时计算数据，等等。

## 特性存储在 MLOps 架构中处于什么位置？

特征库是 ML 平台的固有部分。如前所述，自从第一批 ML 车型投入生产以来，它就一直是其中的一部分，但直到几年前，这一概念才在 MLOps 世界中获得了自己的身份。

可以使用 Neptune、MLFlow 或 SageMaker Experiments 等实验跟踪工具来跟踪要素数据源。也就是说，假设您正在训练一个欺诈检测模型，并且您使用了另一个团队构建的一些共享功能。如果您将这些要素元数据记录为参数，则在跟踪实验时，它们将与您的实验结果和代码一起被版本化。

![ Orchestrating Spark ML Pipelines and MLflow for Production](img/e95e651e74ed5bbeec53828ea6719e74.png)

The Killer Feature Store: Orchestrating Spark ML Pipelines and MLflow for Production | [Source](https://web.archive.org/web/20230106144203/https://www.slideshare.net/databricks/the-killer-feature-store-orchestrating-spark-ml-pipelines-and-mlflow-for-production)

此外，当模型处于生产阶段时，它们成为关键部分。现场直播时，有几个组件需要同步和密切监控。如果其中一个失败，预测可能会很快降级。这些组件是功能计算和摄取管道以及来自生产服务的功能消费。计算管道需要以特定的频率运行，以便特征的新鲜度不会影响在线预测。例如:如果一个推荐系统需要知道你昨天观看的电影，那么在你再次进入媒体流服务之前，应该运行特征管道！

## 如何实现功能商店？

在本节中，我们将讨论不同的架构，这些架构可以针对数据科学团队的不同阶段和规模来实施。在[这篇非常好的文章](https://web.archive.org/web/20230106144203/https://eugeneyan.com/writing/feature-stores/)中，你可以看到作者如何使用*需求层次*来非常明确地展示哪些是你需要解决的主要问题。他将 ***访问*** 需求，包括透明性和血统，作为比 ***服务*** 更基础性的需求。我不完全同意，因为产品中的特性可用性释放了更高的商业价值。

下面给出的建议将基于 AWS 服务(尽管它们可以很容易地与其他公共云服务互换)。

### 最简单的解决方案

这种架构基于托管服务，托管服务需要较少的维护开销，更适合能够快速运行的小型团队。

我的初始设置是 Redshift 作为离线存储，DynamoDB 作为在线键值存储，Airflow 管理批量特征计算作业。此外，Pandas 作为两种选择的数据处理引擎。在这种架构中，所有要素计算管道都在 Airflow 中进行调度，并且需要使用 Python 脚本来接收数据，这些脚本从红移或 S3 中提取数据，对其进行转换，并将其放入 DynamoDB 以用于在线服务，然后再次放入红移以用于离线要素存储。

![The initial setup chart ](img/2ae92301157bf7d3b4b6da8da28b128f.png)

*The initial setup | Source: author*

### 中型功能商店

如果您已经在处理大数据、接近实时的功能需求以及跨数据科学团队的可重用性需求，那么您可能正在寻求跨功能管道的更多标准化和某种程度的可重用性。

在这种情况下，当数据科学团队规模相对较大时(比如说，超过 8-10 名数据科学家)，我会建议开始使用第三方功能商店供应商。首先，我会探索 Feast，因为它是最常用的开源解决方案，并且可以在现有的基础设施上工作。您可以将 Redshift 用作离线特性存储，将 DynamoDB 或 Redis 用作在线特性存储。对于延迟要求较低的在线预测服务，后者速度更快。 [Feast](https://web.archive.org/web/20230106144203/https://feast.dev/) 将通过他们的 SDK 和网络用户界面(尽管仍处于试验阶段)帮助分类和提供功能。如果你想要一个全面管理的商业工具，我恳求你尝试一下[泰克顿](https://web.archive.org/web/20230106144203/https://www.tecton.ai/product/)。

如果有大数据需求，现在可以使用普通 Python 或 Spark 开发特性计算管道，利用 [Feast SDK](https://web.archive.org/web/20230106144203/https://rtd.feast.dev/en/master/) 管理数据摄取。

![](img/3f3227b62b8ef978d7d82153ab2afb35.png)

*Running Feast in production | [Source](https://web.archive.org/web/20230106144203/https://docs.feast.dev/how-to-guides/running-feast-in-production)*

在这种规模下，也很可能有一些实时特性和新鲜度需求的用例。在这种情况下，我们需要一个流媒体服务，将功能直接吸收到在线功能商店中。我们可以使用 Kinesis 服务和 AWS Lambda 将特征向量直接写入 Redis 或 DynamoDB。如果需要窗口聚合，那么 Kinesis 数据分析、KafkaSQL 或 Spark Streaming 可能是合理的选择。

### 企业级功能商店

在这个阶段，我们假设该公司有大量的数据科学家为不同的业务或技术领域创建不同类型的模型。为这种规模的开发团队设置架构的一个关键原则是提供一个可靠的、可伸缩的、安全的、标准化的数据平台。因此，SLA、GDPR、审计和访问控制列表是必须实施的强制性要求。在每个组织规模中，这些都是需要考虑的要点，但在这种情况下，它们扮演着重要的角色。

![Feature Store](img/16efc40d7321b56d4722dd16ce776507.png)

feature store explained | [Source](https://web.archive.org/web/20230106144203/https://www.featureform.com/post/feature-stores-explained-the-three-common-architectures)

技术领域的大多数大公司都根据自己的需求、安全原则、现有基础设施和托管可用性建立了自己的功能商店，以避免在服务完全托管的情况下出现单点故障。

但如果情况并非如此，并且你正在运行一个公共云繁重的工作负载，那么使用 AWS [SageMaker 功能商店](https://web.archive.org/web/20230106144203/https://aws.amazon.com/sagemaker/feature-store/)或 [GCP Vertex AI 功能商店](https://web.archive.org/web/20230106144203/https://cloud.google.com/vertex-ai/docs/featurestore)可能是不错的选择。他们的 API 与开源产品非常相似，如果你已经在使用 SageMaker 或 Vertex，那么设置他们的特性存储服务应该很简单。

![Amazon SageMaker Feature Store ](img/0918088d3295a875d34de8e14d323a86.png)

*Amazon SageMaker Feature Store for machine learning | [Source](https://web.archive.org/web/20230106144203/http://amazon%20sagemaker%20feature%20store%20for%20machine%20learning%20(ml)/)*

Databricks 还提供嵌入式功能存储服务，这也是一个很好的选择，可以与像 [MLFlow](https://web.archive.org/web/20230106144203/https://mlflow.org/) 这样的工具完美兼容。

![](img/ad3f4cbd76eaffc0259358d26b500ffb.png)

*Databricks Feature Store | [Source](https://web.archive.org/web/20230106144203/https://www.databricks.com/it/product/feature-store)*

## 购买还是制造的问题

MLOps 的格局一直由脸书、网飞、优步、Spotify 等大玩家主导和塑造。这些年来，通过他们非常有影响力的工程师和博客。但是 ML 团队应该能够认识到他们在自己的组织、团队和业务领域中运作的环境。一个 20 万用户的应用程序不需要 2000 万用户的规模、标准化和僵化。这就是为什么[合理规模的 m lops](/web/20230106144203/https://neptune.ai/blog/mlops-at-reasonable-scale)成为不在 FAANG 类公司工作的资深从业者的热门话题。

![Graphic explanation of a feature store](img/5ca493b5c0265ef1acaebc03a371ced7.png)

*Explanation of a feature store | [Source](https://web.archive.org/web/20230106144203/https://towardsdatascience.com/do-you-need-a-feature-store-35b90c3d8963)*

### 谁应该建立一个功能商店？

正如本文开头所提到的，在内部构建一个类似于商店的功能平台和购买一个商业或开源产品，如 T2 盛宴、霍普斯沃斯、T3 或 T4 泰克顿 T5 之间，存在着一场持续的争斗。这种紧张关系的存在主要是因为这些产品在其架构和 SDK 中可能会有某种程度的固执己见。因此，这些工具中的大多数需要有一个中央服务来处理在线商店上的功能服务，这成为生产 ML 服务的单点故障。

此外，一些其他产品是完全 SaaS，成为一些团队不确定的关键部分。因此，ML 工程师怀疑是否在他们的 MLOps 旅程中过早地使用这些工具。

在中小型公司或初创公司中，ML 和数据工程团队共享相同的技术堆栈是非常常见的。出于这个原因，迁移到一个特性商店可能会引起很大的麻烦，并暴露一些隐藏的成本。在计划、遗产维护、操作性、两面性等方面。，它成为另一个具有不同于传统数据工程的特定 SDK 的基础设施。

### 谁应该购买功能商店？

为了从商业功能商店中获取最大价值，您的用例以及数据科学团队的设置需要与它们提供的核心优势保持一致。严重依赖实时复杂 ML 用例的产品，如推荐系统、动态定价或欺诈检测，是最能利用这些工具的产品。

一个庞大的数据科学家团队也是拥有特征库的一个很好的理由，因为它将提高生产率和特征的可重用性。除此之外，它们通常提供一个很好的 UI 来发现和探索特性。尽管如此，商业特性库 SDK 和 API 提供了一组标准，以更为一致的方式获取和检索特性。作为副产品，数据被治理，可靠的元数据总是被记录。

在各种各样的 ML 团队领域中，上述情况并不总是满足的，并且建立这些新的商业栈有时仅仅是工程师的个人开发愿望，以跟上新技术的发展。

这就是为什么仍然有团队没有迁移到一个完整打包的特性库，而是仍然依赖现有的数据工程栈来运行他们的产品特性工程层。在我看来，这是完全正确的。

总而言之，特性存储只是在现有的数据工程堆栈上添加了一个方便的外壳，以提供统一的访问 API，一个发现和管理特性集的漂亮 UI，保证在线和特性存储之间的一致性，等等。但是所有这些特性并不是每个 ML 团队用例的关键。

## 结论

我希望这篇文章提供了一个关于什么是特性存储的广阔的视野。但更重要的是,它们之所以必要的原因，以及构建时需要解决的关键组件。

功能存储对于提升数据科学行业的生产服务是必要的。但是你需要他们背后的工程师。ML 工程师角色对于处理特征管道是至关重要的，因为它们只是一种特定类型的数据转换和摄取过程。像这样的混合角色使数据科学家能够更加专注于实验方面，并保证高质量的交付成果。

此外，我特别注意解释了*构建与购买*的困境。从我个人的经验来看，这个问题在任何一个成熟的 ML 团队中迟早都会出现。我试图描述它们对实现速度和标准化至关重要的情况，但也留下了一些思考，即为什么环境意识对于实现这项新技术是必要的。有经验的高级角色应考虑他们所处的 MLOps 旅程阶段。

功能商店(商业和开源)世界还很年轻，还没有一个统一的和公认的方法来处理所有不同的用例和需求。所以，在和一个人安定下来之前，尝试所有的方法。

### 参考

1.  [特色商店——需求层次](https://web.archive.org/web/20230106144203/https://eugeneyan.com/writing/feature-stores/)
2.  [功能商店讲解:三种常见架构|功能形式](https://web.archive.org/web/20230106144203/https://www.featureform.com/post/feature-stores-explained-the-three-common-architectures)
3.  [特征存储-现代数据堆栈](https://web.archive.org/web/20230106144203/https://www.moderndatastack.xyz/category/feature-store)
4.  [回到未来:解决机器学习中的时间旅行问题|泰克顿](https://web.archive.org/web/20230106144203/https://www.tecton.ai/blog/time-travel-in-ml/#:~:text=Time-travel)
5.  [数据集时间旅行概述 Max Halford](https://web.archive.org/web/20230106144203/https://maxhalford.github.io/blog/dataset-time-travel/)
6.  [“用 Apache Samza 彻底改变数据库”,作者 Martin Kleppmann](https://web.archive.org/web/20230106144203/https://www.youtube.com/watch?v=fU9hR3kiOK0)
7.  [时态数据库–wiki wand](https://web.archive.org/web/20230106144203/https://www.wikiwand.com/en/Temporal_database)
8.  [利用特征库构建实时 ML 管道| Adi Hirsch tein 著](https://web.archive.org/web/20230106144203/https://medium.com/data-for-ai/building-real-time-ml-pipelines-with-a-feature-store-9f90091eeb4)
9.  [生产中的奔跑盛宴](https://web.archive.org/web/20230106144203/https://docs.feast.dev/how-to-guides/running-feast-in-production)
10.  [黑仔特色店:协调生产用 Spark ML 管道和 ML flow](https://web.archive.org/web/20230106144203/https://www.slideshare.net/databricks/the-killer-feature-store-orchestrating-spark-ml-pipelines-and-mlflow-for-production)
11.  [需要特色店吗？](https://web.archive.org/web/20230106144203/https://towardsdatascience.com/do-you-need-a-feature-store-35b90c3d8963)