# Kedro vs ZenML vs Metaflow:应该选择哪个管道编排工具？

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/kedro-vs-zenml-vs-metaflow>

在本文中，我将对 Kedro、Metaflow 和 ZenML 进行比较，但在此之前，我认为值得后退几步。为什么还要费心使用 ML 编排工具，比如这三个？启动一个机器学习项目并没有那么难。你安装一些 python 库，初始化模型，训练它，瞧！准备好了。也有很多人告诉你**这是数据科学**，在本地运行的 Jupyter 笔记本上键入 *model.fit* 将保证你得到梦想中的工作，并解决宇宙中的所有问题。但这只是事实的一小部分。

先说清楚，用 Jupyter 笔记本做实验和报道没有错。问题是他们无法提供一个可以在大多数行业 ML 项目中使用的软件开发框架。

实现机器学习项目(和创建基于 ML 的产品)和任何其他软件开发一样复杂。例如，在现实世界的 ML 项目中，您必须能够:

*   以多种格式接收数据
*   预处理数据、清理数据集等。
*   做特征工程
*   训练你的模特
*   测试它们
*   部署您的模型
*   实施 CI/CD
*   使用 git 跟踪您的代码
*   跟踪您的模型的指标

还有很多其他的东西，[用笔记本](/web/20221206150330/https://neptune.ai/blog/should-you-use-jupyter-notebooks-in-production)是做不到的(如果有人这么说，他们可能是在误导你)。因此，您的代码必须是可维护的、可伸缩的、可跟踪的，并且能够在多人团队中工作。

这就是为什么数据科学家必须学习机器学习编排工具，这篇文章旨在深入探讨一些最常用的 MLOps 编排工具-- Kedro、MetaFlow 和 ZenML。我们将看看他们的:

*   主要目的
*   代码结构
*   主要特征
*   赞成和反对

我们还将展示:

*   开始使用其中的每一项是多么容易(以及如何开始)
*   展示他们真实的能力

为了解决之前提出的问题，已经实现了很多框架。我们选择了 Kedro、Metaflow 和 ZenML，因为它们是一些最常用的编排机器学习管道的工具:

*   他们有免费使用的许可证:你不需要支付高额费用来构建你的 ML 项目
*   它们分布广泛:有很多人在使用它们，这样在需要的时候很容易找到人来帮助你
*   他们在网上有很多学习材料:这样你(以及加入你团队的其他人)可以投入并开始你的第一个项目，或者维持现有的项目
*   他们与行业中最常用的工具进行了许多集成:您将能够将它们集成到现有的云框架中，甚至扩展它来做更多的事情

因此，这些是让你的机器学习离开你的 Jupyter 笔记本并让它面对真实世界为真实的人解决真实问题的最简单的方法。

选择流程编排工具时，需要考虑许多因素。这将是你的机器学习项目的基础。例如，这种选择会影响:

*   启动一个项目所需的时间
*   更改项目的特定部分(如数据源、代码片段等)所需的时间
*   是时候培训新员工了解正在运行的项目了
*   项目成本，包括(但不限于)云、处理能力和人力

这就是为什么关注您使用的编排工具如此重要。这就是我们要帮你找到的。

### TL；速度三角形定位法(dead reckoning)

如果你没有时间全部读完，你可以[跳到这里快速总结](#kedro-vs-zenml-vs-metaflow-summing-it-all-up)。

现在，让我们开始吧！

## 原始文件

### 定义

首先，让我们看一下每个文档是如何定义其工具的:

*[**Kedro**](https://web.archive.org/web/20221206150330/https://kedro.readthedocs.io/en/stable/index.html)是一个开源的 Python 框架，用于创建可复制、可维护和模块化的数据科学代码。它从软件工程中借用概念，并将它们应用到机器学习代码中；应用的概念包括模块化、关注点分离和版本控制。Kedro 由 LF AI &数据基金会托管。”*

在此[环节](https://web.archive.org/web/20221206150330/https://kedro.readthedocs.io/en/stable/01_introduction/01_introduction.html)可用。

*[**ZenML**](https://web.archive.org/web/20221206150330/https://www.zenml.io/home)是一个可扩展的开源 MLOps 框架，用于创建生产就绪的机器学习管道。它专为数据科学家打造，具有简单、灵活的语法，与云和工具无关，并且具有面向 ML 工作流的接口/抽象。”*

在此[环节](https://web.archive.org/web/20221206150330/https://docs.zenml.io/)可用。

*[**Metaflow**](https://web.archive.org/web/20221206150330/https://metaflow.org/)是一个人性化的 Python 库，帮助科学家和工程师构建和管理现实生活中的数据科学项目。Metaflow 最初是在网飞开发的，旨在提高数据科学家的工作效率，这些科学家从事从经典统计到最先进的深度学习的各种项目。”*

在此[环节](https://web.archive.org/web/20221206150330/https://docs.metaflow.org/)可用。

至此，一切看起来都非常相似。但是这些工具之间有什么不同呢？是的，有一些重要的区别需要注意。

### 谁建的？

这三个都是开源免费使用的框架，但都是由公司开发的:

## 

*   Kedro 是由麦肯锡公司 QuantumBlack 创建的，最近捐赠给了 Linux 基金会
*   另一方面，ZenML 是由德国慕尼黑的一家小公司开发的
*   Metaflow 是在网飞开发的

### 它们是否有完整的文件记录？

所有 3 个工具都有文档，提供可用作其实施基础的信息。代码文档对于软件开发尤其重要，因为它是由实际创建软件的人编写的指南，所以没有其他更好的媒介来解释如何使用它。

当维护一个正在运行的项目时，这变得更加重要，因为总是需要检查框架的某个部分的一些特性和代码样本，例如理解语法或首字母缩写词。

#### Kedro

Kedro 有大量的文档，其中包含教程和示例项目。软件的每个部分都有自己的部分，包含代码片段和示例，从最简单的功能到高级的功能。您可以安装一个模板项目，并使用样本数据集启动您的第一个项目，并在您的终端中用几行代码训练您的第一个模型。

#### ZenML

ZenML 文档也非常完整。一切都写得很好，有有用的图像帮助我们了解那里正在发生什么。还有一些复制粘贴教程，可以立即投入使用，并查看 ZenML 在实践中的工作。不过，以我的拙见，它缺少代码片段来演示如何应用一些概念。

#### 元流

元流文档是三者中最短的。有一个模板项目，但是文档的覆盖面有点太短。所以，最难理解的是到底发生了什么。Metaflow 是三种语言中唯一可以在 R 中运行的，所以如果这是你首选的编码语言，也许你已经选择了工具。

> **文件:Kedro = ZenML >元流量**

## 主要特征

现在是时候以更实际的方式来理解这些工具是如何工作的了。我们将看到如何学习它们，以及如何将它们用于现实世界的项目中。

### 学习如何使用

每一个软件工具都是有用的，就像使用它有多容易一样。让我们来看看每个工具的一些可用材料。

#### Kedro

Kedro 有一个很大的社区，它也提供了很多关于如何用它创建项目的信息。除了官方文档(附带一个模板项目)，YouTube 上还有一个完整的免费 [kedro 课程](https://web.archive.org/web/20221206150330/https://www.youtube.com/watch?v=rf8yBHsDOj4&list=PLTU89LAWKRwEdiDKeMOU2ye6yU9Qd4MRo)，由 DataEngineerOne 频道主办。它从非常初级的水平(像安装 kedro)到大多数高级功能。在 medium 中还有 4 篇逐步的帖子(我写的)展示了如何使用 kedro、MLflow 和 fastAPI 部署模型([第 1 部分](https://web.archive.org/web/20221206150330/https://medium.com/indiciumtech/how-to-build-models-as-products-using-mlops-part-1-introduction-a93eb5b6a96)、[第 2 部分](https://web.archive.org/web/20221206150330/https://medium.com/indiciumtech/how-to-build-models-as-products-using-mlops-part-2-machine-learning-pipelines-with-kedro-10337c48de92)、[第 3 部分](https://web.archive.org/web/20221206150330/https://medium.com/indiciumtech/how-to-build-models-as-products-using-mlops-part-3-versioning-models-with-mlflow-1e7bde445833)和[第 4 部分](https://web.archive.org/web/20221206150330/https://medium.com/indiciumtech/how-to-build-models-as-products-using-mlops-part-4-deploying-models-in-apis-with-kedro-fast-api-15f13e67c080))。

#### ZenML

YouTube 上有很多 ZenML 项目视频，其中很多都是由他们的官方个人资料保存的，你可以在这里找到。还有一些实现的教程项目，可以从命令界面本身下载然后抓取指南来实现自己的项目，这里的 how-to 是。

基本上，您可以运行“zenml 示例列表”,并通过键入“zenml 示例快速启动”选择一个可用的示例下载到您的当前目录中。

#### 元流

Metaflow 是目前可用材料最少的一个。文档很短，几乎没有代码示例。Youtube 上有一些视频展示了一些实现，就像这个，但似乎没有完整的课程。材料中也有拆分，因为 Metaflow 是这里唯一可以在 R 和 Python 中运行的，所以两者中都有学习材料插图实现。

> **学习如何使用:Kedro > ZenML > Metaflow**

### 代码结构

#### Kedro:节点和管道

Kedro 有一个由节点和管道定义的代码结构。每个节点都是一个 Python 函数。管道是节点的编排，指定管道的所有输入、输出和参数。它基本上描述了哪些数据进入每个节点，输出是什么，以及接下来的步骤是什么。

[![Kedro: nodes and pipelines](img/b48b36a1fa4aba155f17bec293dbf2e2.png)](https://web.archive.org/web/20221206150330/https://neptune.ai/kedro-vs-metaflow-vs-zenml4)

**Kedro nodes (squares), datasets (round-edge rectangles), and pipelines (the interconnection between them)* | [Source](https://web.archive.org/web/20221206150330/https://kedro.readthedocs.io/en/0.17.1/03_tutorial/06_visualise_pipeline.html)*

不需要指定哪个节点在下一个节点之前。Kedro 自动编排整个管道的执行顺序(如果执行中没有冲突，这样的数据源可以是同一个节点的输入和输出)。这样，如果您的数据结构保持不变，您的源代码可能会发生变化，而无需更改主代码。此外，它还允许您查看管道的每个步骤中发生了什么，因为中间数据正在被存储。

所有代码都位于“src”文件夹中，所有函数都在节点和管道 python 文件中创建。可以(强烈建议)将您的代码拆分到子文件夹中(如预处理、特征工程、模型训练等)以保持其组织性，这样就可以单独运行管道的每一步。整个代码架构，包括代码开发生命周期，可以在这里看到[。](https://web.archive.org/web/20221206150330/https://kedro.readthedocs.io/en/stable/faq/architecture_overview.html)

Kedro 还提供了一个环境接口。Kedro 环境旨在分离管道中使用的代码源，这样您就可以将开发从生产中分离出来，甚至可以在项目中创建其他分区。你可以在这里阅读更多相关信息[。它会自动创建本地和基础环境，但是您可以根据需要创建任意多个环境。](https://web.archive.org/web/20221206150330/https://kedro.readthedocs.io/en/stable/kedro_project_setup/configuration.html)

当向特定节点(或所有节点)添加附加特性时，kedro 使用了钩子结构。例如，您可以配置特定的行为，如记录数据集加载发生的时间。钩子的结构可以在这里找到。Kedro 也有一个秘密的管理结构，它使用一个证书 yaml 文件，你可以在这里[了解更多。](https://web.archive.org/web/20221206150330/https://kedro.readthedocs.io/en/stable/kedro_project_setup/configuration.html#credentials)

#### ZenML:步骤和管道

ZenML 是建立在步骤之上的。它为它定义了一个装饰器(@step)，然后管道的每一步都写成一个 Python 函数。它包含用于接收数据、数据清理、模型训练甚至度量评估的代码。可以使用管道来组织这些步骤，管道将构建所有的步骤，以允许管道同时运行。对步骤、输入和输出使用 ZenML 内置类的主要优点是所有集成都变得更容易。

另一方面，管道是指定步骤执行顺序的 python 函数，以及它们相应的输入和输出。所有代码都构建在 run.py 文件中，该文件将包含步骤和管道本身。

ZenML 项目基础设施也定义了一个栈，它定义了你的管道将如何运行。它通常包括:

*   **协调器**:协调管道执行的框架(比如气流和库伯流)
*   **工件存储**:数据的实际存储，主要用于中间管道步骤的数据(因为摄取通常是它自己的一个步骤)
*   **元数据存储库**:跟踪关于管道运行本身的数据的存储库

由于其结构，改变堆栈的组件更容易，如数据存储、云基础架构或编排。这里也有对每个可用堆栈组件[的解释。例如，您可以使用本地开发栈(在您的机器上运行)和生产栈(在云服务器上运行)。](https://web.archive.org/web/20221206150330/https://docs.zenml.io/advanced-guide/stacks-components-flavors)

[![An example of ZenML pipeline with two stacks, local and cloud](img/93995b2f2135c3fc7e2600fd3bdc38d1.png)](https://web.archive.org/web/20221206150330/https://neptune.ai/kedro-vs-metaflow-vs-zenml3)

*An example of ZenML pipeline with two stacks, local and cloud | [Source](https://web.archive.org/web/20221206150330/https://zenml.io/home)*

在冗余非常重要的项目中，堆栈特别有用。如果您的主系统出现故障(比如依赖于特定的云提供商)，您可以切换到不同的基础架构，几乎不需要交互，甚至可以实现自动化。它还可以包括一个工件注册表，它甚至可以管理项目的秘密。

与 Kedro 不同，ZenML 允许您定制整个管道的创建，因此我们强烈建议您遵循一些最佳实践。你可以在这里找到指南[。](https://web.archive.org/web/20221206150330/https://docs.zenml.io/links-and-resources/resources/best-practices)

#### 元流:有 UI(也有步骤)的那个

Metaflow 与 Airflow 非常相似，因为它是基于 DAGs(有向无环图)的，所以 Airflow 用户更容易理解它。

就像 ZenML 一样，Metaflow 也使用 steps 来修饰函数，所以可以和有 ZenML 体验的用户相当。然而，它使用了一个在步骤本身中指定流程的结构，这看起来有点混乱。此外，为了创建流，您应该首先创建一个包含其步骤的类(例如，MyFlow)。

大多数功能都是用 decorators 构建的。例如，您可以使用资源装饰器来指定计算资源(比如内存、CPU 和 GPU 的使用)、每个步骤应该使用的 Python 环境、重试次数、负责的用户等等。

使用 Metaflow 的主要优点是它有一个内置的 UI，用于跟踪项目中每次运行的度量。在这里可以找到一个示例 UI[。](https://web.archive.org/web/20221206150330/https://demo.public.outerbounds.xyz/?timerange_start=1406160000000)

Metaflow 自动跟踪一些指标，比如运行实验的用户、实验所用的时间、实验的步骤以及运行状态。

> **代码结构:Kedro > ZenML >元流**

### 摄取数据

#### Kedro:数据目录和参数文件

Kedro 有一个很好的方法将数据源从代码中抽象出来:数据目录文件。这些是 YAML 文件，其中描述了所有源(输入和输出数据)，以及关于它们的保存路径、格式和其他参数的信息。这样，它们将作为参数传递给管道，这使得更新更加容易。数据可以以多种格式、多种路径和许多不同的参数存储。

你可以在这里阅读更多相关信息[。这个 yaml 示例展示了如何导入位于本地的 CSV 数据集。](https://web.archive.org/web/20221206150330/https://kedro.readthedocs.io/en/stable/data/data_catalog.html)

```py
```
cars:
  type: pandas.CSVDataSet
  filepath: data/01_raw/company/cars.csv
  load_args:
    sep: ','
  save_args:
    index: False
    date_format: '%Y-%m-%d %H:%M'
    decimal: .

```py
```

将项目的数据集作为参数传递的主要优点是，如果您必须更改数据源(其路径、源或文件)，您不需要更新所有代码来匹配它。Catalog 是一个默认的 kedro 类，有几个集成可供使用。在最坏的情况下，数据清理节点需要更新，而管道的中间步骤不会改变。

当我说它可以有任何格式时，我的意思是，它可以是一段 SQL 代码(来自多个源和云)、CSV 表、parquet 表、Spark 数据集、MS Excel 表、JPEG 图像等。它可以位于云(AWS、GCP 和 Azure)、Hadoop 文件系统甚至 HTML 网站中。该目录还用于保存模型和度量文件，使得实现它们和管理所有来源变得更加容易。

Kedro 还使用 YAML 文件作为 params 文件。它可能包含模型或管线需要的所有参数。这样，在需要时可以很容易地跟踪和更改它们。

#### ZenML

前面说过，ZenML 是基于 steps 的。因此，举例来说，数据摄取就是您可以在一个步骤中以 python 脚本的形式编写的东西。

ZenML 的大部分功能来自于它的集成。它提供了大量工具来集成来自多个来源的数据，以及特性存储、工件存储等等。我们将在下一节进一步讨论 ZenML 集成。

因为 ZenML 是高度可定制的，您可以创建自己的代码结构，包括参数和配置文件，以便使您的项目看起来更有组织性(和更少的硬编码)和更容易维护。

#### 元流

它与 ZenML 摄取非常相似，尽管 Metaflow 只支持 AWS 集成，这在与不同的云供应商合作时可能是一个障碍。因此，基本上，为了集成数据源和其他特性，您可能必须用 Python(或 R)进行一些硬编码。

> **摄取数据:凯卓> ZenML >元流**

### 集成

#### Kedro:插件

kedro 的大部分功能来自它的插件。它可以通过本机运行第三方工具来扩展 kedro 的功能，并且它们可以一起工作以充分发挥它们的潜力。虽然 kedro 将它们作为插件添加，但它们实际上是作为集成到 kedro 结构中的独立组件来工作的。其中有很多，比如:

*   Kedro-neptune 一个连接 Kedro 和 Neptune.ai 的插件，允许实验跟踪和元数据管理。

[![Kedro experiment tracking using kedro-neptune](img/988d4863925379b6170e736d387c074b.png)](https://web.archive.org/web/20221206150330/https://github.com/neptune-ai/kedro-neptune)

*Kedro experiment tracking using kedro-neptune | [Source](https://web.archive.org/web/20221206150330/https://github.com/neptune-ai/kedro-neptune)*

*   **[Kedro-mlflow](https://web.archive.org/web/20221206150330/https://kedro-mlflow.readthedocs.io/en/stable/) :** 允许实现 MLflo 进行模型跟踪和模型服务
*   **[Kedro-viz](https://web.archive.org/web/20221206150330/https://github.com/kedro-org/kedro-viz) :** 一个神奇的工具，可以让你的整个 ML 管道可视化

[![Kedro pipeline visualization using kedro-viz](img/ece38119e0b625087860200161c520db.png)](https://web.archive.org/web/20221206150330/https://neptune.ai/kedro-vs-metaflow-vs-zenml6)

*Kedro pipeline visualization using kedro-viz | [Source](https://web.archive.org/web/20221206150330/https://github.com/topics/kedro-plugin)*

*   Kedro-docker 一个让使用 docker 创建、部署和运行 kedro ML 应用程序变得更加容易的工具；

这些插件的主要优点是它们都可以很容易地用 PyPI 安装，并且它们与 kedro 结构兼容，因此很容易在现有项目中安装和运行它们。您可以查看官方文档，找到一个对解决您的问题最有用的文档。

#### ZenML

ZenML 支持大量第三方集成。它们包含编排工具、分布式处理工具、云提供商、部署、特性存储、监控等等。ZenML 就像胶水一样把整个管道粘在一起(就像他们自己声称的那样)。在 ZenML Python 文件内部调用这些工具的方式与在 ZenML 外部调用它们的方式非常相似，例如，使用“ZenML . integrations . sklearn”来导入 sk learn。

[![Original Documentation](img/0fe1b6ec0695f7a7d1613468c343a0e4.png)](https://web.archive.org/web/20221206150330/https://neptune.ai/kedro-vs-metaflow-vs-zenml9)

*Original documentation |* [*Source*](https://web.archive.org/web/20221206150330/https://docs.zenml.io/advanced-guide/integrations)

在 ZenML 中有一个 Python 集成包，它包含了许多不同工具的所有接口。你可以在这里阅读更多关于已经集成的工具(以及他们目前正在开发的工具)[。说到安装集成，离你只有四个字的命令界面，比如‘zenml integration install sk learn’。](https://web.archive.org/web/20221206150330/https://docs.zenml.io/advanced-guide/integrations)

#### 元流

众所周知，网飞与 AWS 有着密切的关系。因此，Metaflow 与大多数 AWS 服务很好地集成，使用 S3 存储、AWS Batch 和 Kubernetes 进行计算、AWS Fargate 和 RDS 进行元数据存储、AWS Sagemaker 笔记本和 AWS Step 函数、Eventbridge 和 Argo 工作流进行调度。你可以在这里阅读更多相关信息[。](https://web.archive.org/web/20221206150330/https://docs.metaflow.org/metaflow-on-aws)

> **集成:Kedro = ZenML >元流**

## 现实世界的应用

我们一直在谈论的所有工具都可以用来将模型部署到生产中，并协调它们的培训和评估。然而，它们的一些特性可能更适合某些用例。

#### Kedro

由于 Kedro 具有高度组织化的代码结构，并抽象了数据和模型源和路径，因此对于那些应该由大型团队构建并需要长期维护的项目来说，它是一个很好的选择，因此基于插件和挂钩的新功能可以添加到项目中，而无需更改整个管道结构。此外，由于它可以与 PySpark 无缝协作，因此使用大数据或任何云平台进行部署都没有问题。

#### ZenML

ZenML 的代码结构比 Kedro 更具可定制性，它可能非常适合原型项目。ZenML 的主要优势之一是它可以在不改变代码的情况下处理不同的栈。因此，如果项目需要改变云提供商，例如，由于不可用，它可以在几秒钟内完成，无需太多设置。这提供了健壮性和可伸缩性。

#### 元流

Metaflow 是三个中唯一一个可以使用 r 的。因此，如果有一些模型还没有在 Python 中使用，Metaflow 可以使用 r 将其部署到生产中。此外，Metaflow 的另一个优点是它可以自动跟踪每个运行指标。那么它可能非常适合原型化和测试不同的项目方法。此外，由于它与 AWS 的兼容性，如果您对 AWS 有一些依赖，它应该是您的选择。

## 优点和局限性

#### 凯德罗

##### 优势

*   Kedro 的结构对于维护复杂项目中的秩序特别有用，而使用其他工具很难做到这一点。
*   它的插件可以提供广泛的操作，提供 kedro 原生不做的东西；
*   强社区:社区开发频繁，所以总是有 bug 修复和新插件发布；
*   节点和管道架构避免了代码重复，因为它允许功能重用，从而清洁编码；
*   可以使用不同的环境，比如 dev 和 prod 来分离数据和模型源。

##### 限制

*   因为它结构良好，如果你的项目太简单，kedro 可能是一个大材小用；

#### ZenML

##### 优势

*   ZenML 非常简单，易于实现。你可以很快地开始你的第一个项目，并随着时间的推移逐渐发展成一个复杂的项目。
*   它有许多与第三方工具的集成，这允许你实现广泛的项目；
*   栈结构允许你以多种不同的方式实现你的代码，并使 dev 和 prod 分离成为可能。

##### 限制

*   因为代码很简单，所以随着项目的增长，保持所有的东西都有组织是很棘手的。在有多个步骤和复杂数据流的项目中，需要遵守纪律。因此，阅读和使用建议的[最佳实践](https://web.archive.org/web/20221206150330/https://docs.zenml.io/links-and-resources/resources/best-practices)非常重要；
*   它是由一家公司举办的，因此很少有围绕 bug 修复或新功能创建可能性的社区互动。

#### 元流

##### 优势

*   Metaflow 自动跟踪度量和管道运行信息，这在生产中运行时相当有用；
*   在 R 中可用:即使不是唯一的，也是唯一的一个可以在 R 中运行的框架；
*   它有一个内置的用户界面来显示所有跟踪的指标；
*   基于 DAGs:这使得 Metaflow 对于熟悉图形编程的人特别有用，比如气流。

##### 限制

*   缺少文档和其他在线学习材料；
*   代码很容易与许多装饰器的使用混淆；
*   与 ZenML 类似，随着项目的增长，保持一切有序可能会很棘手。
*   对于初学者来说可能有点难学。

## 结论

在查看了所有这些方法之后，很明显每个工具之间都有很大的差异。每一种都有自己的优点和局限性，它可以更好或更差地适应每个项目的需求。简而言之:

*   Kedro 有一些有趣的功能，在处理复杂的项目时尤其有用。它抽象了数据源和模型，并且可以与许多其他工具很好地集成。它可以被设置为适合大多数机器学习现代项目，包括所有当前的行业最佳实践。以我的拙见，他们处理和抽象源代码和模型源代码的方式是令人惊奇的。

*   ZenML 也是一个强有力的竞争对手，与 Kedro 相比有一些优势。它更具可定制性，可以在不需要任何代码更改的情况下处理多个堆栈，并且它还可以与许多其他工具很好地集成。但是，它缺少一些功能，主要是在处理数据和模型源时。

*   元流是目前为止我们将在这里看到的最不结构化的方法。它主要关注编排步骤和跟踪步骤执行度量。因此，您的项目需要的其他特性可能需要通过使用其他 Python 功能手工编码来包含。可能不是不可以，但是肯定有一些弊端。然而，自动指标跟踪和 UI 在生产环境中非常酷。

最后，我希望您现在有更多的信息来选择这些工具中哪一个更适合您的用例，只是请记住，在每个工具的功能之间总是有一个权衡。谢谢你一直看完这篇博客，下次再见！