# 如何使您的 MLflow 项目易于共享和协作

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/mlflow-projects-easy-to-share-and-collaborate-on>

如果您使用 MLflow，您将会大开眼界！因为，在本文中，您将看到如何使您的 MLflow 项目更容易共享，并实现与您的队友的无缝协作。

为你的机器学习项目创建一个无缝的工作流程是极具挑战性的。

典型的机器学习生命周期包括:

*   数据收集+预处理
*   根据数据训练模型
*   将模型部署到生产中
*   测试+用新数据改进模型

这四个步骤看起来相当简单，但是每一层都有新的障碍。您可能需要为每个步骤使用不同的工具——Kafka 用于数据准备，Tensorflow 作为模型训练框架，Kubernetes 作为部署环境，等等。

每次使用新工具时，您都必须重复整个过程，可能是通过 Scikit-learn 运行相同的漏斗并部署到 Amazon SageMaker。随着 API 和组织的扩张，这显然是不可持续的。

另外，调整超参数对于创建一个非凡的模型是至关重要的；应该有超参数历史、源代码、性能指标、日期、人员等的完整记录。机器学习生命周期可能是一个令人生畏的平台开发挑战:你应该能够轻松地复制、重新访问和部署你的工作流到生产中，你还需要一个标准化生命周期的平台。

幸运的是，有 MLflow，这是一个很棒的开源解决方案，围绕 3 个支柱构建:跟踪、项目和模型。

围绕您的模型创建一个广泛的日志框架，分配特定的指标来比较运行。

创建一个 MLflow 管道来确定模型如何在云上运行。

以标准格式打包您的机器学习模型，以便在各种下游工具中使用。比如用 REST API 实时服务，或者用 Apache Spark 批量推理。

MLflow 为大型组织提供了可再现性和可扩展性。相同的模型可以在云中、本地或笔记本中执行。您可以使用任何 ML 库、算法、部署工具或语言，也可以添加和共享以前的代码。

但是，有些东西是 MLflow 没有的:一种简单的组织工作和协作的方式。

您需要托管 MLflow 服务器，精心组织团队成员访问，存储备份，等等。再加上 MLflow 的 UI，让你对比实验的 MLflow 追踪模块一点都不好用，尤其是大型团队。

不要担心！**我们可以用海王 AI 来解决这个问题。**

Neptune 的直观 UI 让你**跟踪实验并与队友**合作，同时也让你最喜欢的部分远离 MLflow。

## Neptune 和 MLflow 集成简介

Neptune 是一个轻量级的 ML 实验管理工具。它灵活且易于与所有类型的工作流集成。您的队友可以使用不同的 ML 库和平台，共享结果，并与 Neptune 在单个仪表板上协作。您甚至可以使用他们的 web 平台，这样您就不必在自己的硬件上部署它。

海王星的主要特征是:

*   实验管理:跟踪你团队的所有实验，并对它们进行标记、过滤、分组、排序和比较
*   **笔记本版本和区分:**比较两个笔记本或同一笔记本中的检查点；与源代码类似，您可以进行并排比较
*   **团队协作:**添加评论，提及队友，比较实验结果

[![](img/05c9bd307c88058327d9dba3f06e465a.png) ️ ](https://web.archive.org/web/20221206083720/https://docs.neptune.ai/integrations/tensorboard.html) [海王 vs ml flow](/web/20221206083720/https://neptune.ai/vs/mlflow)——他们有什么不同？

Neptune 和 MLflow 可以通过一个简单的命令进行集成:

```py
neptune mlflow
```

现在，你可以把所有这些 MLrun 物体推向海王星实验:

*   实验 id +名称
*   运行 id +名称
*   韵律学
*   因素
*   史前古器物
*   标签

## 与 Neptune 的组织和协作

现在，让我们来看看您将如何通过 Neptune 漂亮直观的用户界面分享和协作 MLflow 的实验。

### Neptune 设置(如果您已经有一个 Neptune 帐户，请跳过)

1.先注册一个[海王 AI](/web/20221206083720/https://neptune.ai/register) 账号。它对个人和非组织都是免费的，你可以获得 100 GB 的存储空间。

2.通过点击右上角的菜单获取您的 API 令牌。

3.  创建一个 NEPTUNE_API_TOKEN 环境变量，并在控制台中运行它。

```py
export NEPTUNE_API_TOKEN=’your_api_token’
```

4.创建一个项目。在您的项目仪表板中，单击“新建项目”并填写以下信息。注意隐私设置！

### 同步 Neptune 和 MLflow

首先安装 Neptune-MLflow:

```py
pip install neptune-mlflow
```

接下来，将 NEPTUNE_PROJECT 变量设置为用户名/项目名:

```py
export NEPTUNE_PROJECT=USER_NAME/PROJECT_NAME
```

最后，将您的 mlruns 目录与 Neptune 同步:

```py
neptune mlflow
```

### 与海王星合作

您的实验元数据现在应该存储在 Neptune 中，您可以在您的实验仪表板中查看它:

您可以通过添加标签和使用自定义过滤器对实验进行分组来自定义仪表板。

Neptune 让你只需发送一个链接就可以分享 ML 实验。它可以是:

Neptune 还带有[工作区](https://web.archive.org/web/20221206083720/https://docs.neptune.ai/you-should-know/core-concepts#workspace)，一个你可以管理项目、用户和订阅的中心枢纽；有个人和团队工作空间。

在团队工作区中，团队成员可以浏览与其分配的角色相关的内容。您可以在项目和工作区中分配各种角色。在团队工作区中，您可以邀请管理员或成员，每个人都有不同的权限。

可以在顶栏上的工作区名称中更改工作区设置:

在*概述、项目、人员和订阅*选项卡下，您可以看到工作场所设置:

项目中有三个角色:所有者、贡献者和查看者。根据角色的不同，用户可以运行实验、创建笔记本、修改以前存储的数据等。

*更多详情参见- >* [*用户管理*](https://web.archive.org/web/20221206083720/https://docs.neptune.ai/administration/user-management)

## 了解更多关于海王星的信息

如您所见，MLflow 和 Neptune 并不相互排斥。您可以从 MLflow 中保留您最喜欢的功能，同时使用 Neptune 作为管理您的实验并与您的团队合作的中心位置。

如果你想了解更多关于 Neptune 的知识，请查阅官方[文档](https://web.archive.org/web/20221206083720/https://docs.neptune.ai/)。如果你想尝试一下，[创建你的账户](/web/20221206083720/https://neptune.ai/register)，开始用 Neptune 跟踪你的机器学习实验。