# 最佳端到端 MLOps 平台:每个数据科学家都需要了解的领先机器学习平台

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/end-to-end-mlops-platforms>

每个机器学习模型在幕后总是有一个模型生命周期——它从数据版本化开始，然后是数据验证，通常是一些预处理(将现实世界的数据映射到我们的模型理解的东西)，模型训练，架构，一些调整，分析验证，最后是部署。

如果你是一名数据科学家，你将会一遍又一遍地重复这个循环，所以让这个过程自动化是有意义的。另外，对于数据科学家来说，另一个大问题是大多数 ML 项目从未超出实验阶段。

所以，作为这些问题的解决方案，出现了一个新的趋势: **MLOps** 。

在本文中，我们将探索 MLOps，并比较流行的 MLOps 平台，包括托管平台和开源平台。

## 什么是 MLOps？

MLOps 这个名字是“机器学习”和“操作”的融合。MLOps 是机器学习、DevOps 和数据工程的交集。

这是一套在生产中自动化 ML 算法生命周期的方法。通过这种方式，您可以在 ML 系统构建的所有步骤中实现自动化和监控，从最初的模型训练到针对新数据的部署和再训练。

借助 MLOps，数据科学家和 IT 团队可以协作并结合数据工程、机器学习和 DevOps 中使用的技能、技术和工具。它通过强大的机器学习生命周期管理来促进快速创新。

最大的问题是:有工具管理机器学习生命周期的所有部分吗？嗯，是也不是。

事实上，在本文中，我们将关注端到端 MLOps 平台。这些平台中的大多数都提供了强大的工具来管理从模型训练到部署等各种 ML 管道，但重要的是要提到，数据收集和标记是留给专门为这些任务构建的其他工具的。

介绍够了。让我们看看一些非常棒的端到端系统来管理机器学习管道。

* * *

* * *

## MLOps 平台概述

有几个 MLOps 框架用于管理机器学习的生命周期。以下是 11 大端到端 MLOps 平台:

| 名字 | 简短描述 |
| --- | --- |
|  | 

通过健康的 ML 生命周期安全地管理您的机器学习操作。

 |
|  | 

端到端的企业级平台，供数据科学家、数据工程师、开发人员和管理人员管理整个机器学习&深度学习产品生命周期。

 |
|  | 

一个端到端的机器学习平台，用于大规模构建和部署 AI 模型。

 |
|  | 

平台使数据访问民主化，使企业能够构建自己的人工智能之路。

 |
|  | 

AI 平台，使数据科学民主化，并大规模自动化端到端 ML。

 |
|  | 人工智能领域的开源领导者，其使命是为每个人实现人工智能的民主化。 |
|  | 

利用端到端的机器学习管道自动化 MLOps，将 AI 项目转化为现实世界的商业成果。

 |
|  | 

致力于让 Kubernetes 上的机器学习(ML)工作流部署变得简单、可移植、可扩展。

 |
|  | 

将数据谱系与 Kubernetes 上的端到端管道相结合，专为企业而设计。

 |
|  | 

Kubernetes 上可复制、可扩展的机器学习和深度学习的平台。

 |
|  | 

带你从 POC 到生产，同时管理整个模型生命周期。

 |

## 基于 MLOps 任务的比较

为了比较我们列表中的平台，我们将它们分为以下四类:

*   **数据和管道版本化:**通过对数据集、特征及其转换的版本控制进行数据管理
*   **模型和实验版本化:**管理模型生命周期(跟踪实验、参数、度量和模型训练运行的性能)
*   **超参数调整:**对给定模型的超参数值进行系统优化
*   **模型部署和监控:**管理生产中部署的模型并跟踪性能

我们的平台负责哪些任务？

某些平台如 [Kubeflow](https://web.archive.org/web/20230305040246/https://www.kubeflow.org/) 、 [Allegro.ai](https://web.archive.org/web/20230305040246/http://allegro.ai/) 覆盖了大范围的任务，其他如 [H2O](https://web.archive.org/web/20230305040246/https://www.h2o.ai/) 、[厚皮动物](https://web.archive.org/web/20230305040246/https://www.pachyderm.com/)专注于特定的任务。

这种比较提出了一个棘手的问题:**是选择一个完成所有任务的平台好，还是将多个专门的平台缝合在一起好？**

每个平台处理不同深度的任务并不会变得更容易。

## 基于受支持库的比较

每个数据科学家都使用不同的编程语言、库和框架来开发 ML 模型。因此，我们需要一个 MLOps 平台来支持您项目中的库。让我们基于一系列流行的库和框架来比较我们的平台。

在我看来，数据科学工程师最关心的是他们的编程语言的支持水平，在这种情况下，尤其是 Python 和/或 r。

该表告诉我们，与其他平台相比，Pachyderm 和 Algorithmia 涵盖了更多的库。TensorFlow 显然是最受支持的库。

## 基于 CLI 和 GUI 的 MLOps 平台比较

在这一部分，我们比较的重点将转移到数据科学家的专业知识上。

一些 MLOps 平台侧重于具有较少工程专业知识的能力来构建和部署 ML 模型。他们关注 GUI(图形用户界面)，一个允许通过 web 客户端访问的可视化工具。

其他平台面向具有工程专业知识的专家数据科学家。当将平台与现有工具集成时，这些平台倾向于使用命令行界面(CLI)或 API，而 web UI 对于专业用户来说可能并不重要。

下表包含了每个平台的使用基本上是围绕 CLI 还是 GUI 设计的近似和个人比较。

像 [cnvrg.io](https://web.archive.org/web/20230305040246/http://cnvrg.io/) 和 [Volahai](https://web.archive.org/web/20230305040246/https://valohai.com/) 这样的一系列平台都有更多的 CLI 焦点和进一步的 GUI 支持。其他平台，也就是 [Datarobot](https://web.archive.org/web/20230305040246/https://www.datarobot.com/) 带有 GUI 焦点。然而，大多数托管平台介于两者之间，例如 [Allegro.ai](https://web.archive.org/web/20230305040246/http://allegro.ai/) 和 [Iguazio](https://web.archive.org/web/20230305040246/https://www.iguazio.com/) 。

## MLOps 平台列表

Algorithmia 管理现有运营流程中 ML 生命周期的所有阶段。它将模型快速、安全且经济高效地投入生产。该平台自动化了 ML 部署，提供了最大的工具灵活性，优化了运营和开发之间的协作，利用了现有的 SDLC 和 CI/CD 实践，并包括高级安全和治理功能。

赞成的意见

*   轻松部署，无忧无虑
*   版本管理:对测试任何版本都有用。
*   GPU 支持

骗局

*   目前，Algorithmia 不支持 SAS。
*   创业成本高

Allegro 是一个开创性的端到端企业级平台，供数据科学家、数据工程师、开发人员和管理人员管理实验、编排工作负载和管理数据，所有这些都在一个简单的工具中完成，该工具集成了团队正在使用的任何工具链。该公司的平台支持本地、私有云、多云租户和定制配置。针对无限数量设备的持续学习和模型个性化。

优点:

*   基于对象存储的完全差异化的数据管理和版本控制解决方案(S3/GS/Azure/NAS)
*   自动实验跟踪、环境和结果
*   ML/DL 作业的自动化、管道和流程编排解决方案

缺点:

*   在可定制性方面稍有欠缺
*   不支持 R 语言

Cnrvg.io 是一个端到端的平台，管理、构建和自动化从研究到生产的整个 ML 生命周期。实际上，它是由数据科学家设计的，旨在组织数据科学项目的每个阶段，包括研究、信息收集、代码编写和模型优化。

优点:

*   允许用户只需点击几下鼠标就能构建紧凑的人工智能模型的平台
*   适用于大多数库和框架

缺点:

*   有一些缺失的功能，如可定制的模板，预测分析和问题管理等。

Dataiku 使数据访问民主化，并使企业能够以人为中心的方式建立自己的人工智能之路。它允许您创建、共享和重用利用数据和机器学习来扩展和自动化决策的应用程序。该平台为数据专家和探索者提供了一个共同的基础，一个最佳实践的存储库，机器学习和人工智能部署/管理的捷径，以及一个集中的受控环境。

优点:

*   根据不同的业务需求进行数据清理和转换的最佳工具。
*   用户界面非常直观，只需点击几下鼠标，即可将数据上传到项目中。

缺点:

*   无法很好地适应更多用户
*   可以在平台安装和维护方面获得更好的支持

DataRobot 是领先的端到端企业人工智能平台，可以自动化并加速您从数据到价值的每一步。它是在生产中部署、监控、管理和治理机器学习模型的中心枢纽，以最大限度地增加对数据科学团队的投资，并管理风险和法规合规性。

优点:

–易于 IT 组织使用，有良好的公司支持

–能够轻松构建从简单回归到复杂梯度提升树的机器学习模型算法

缺点:

–输入大数据可能需要很长时间

–缺少到 RDBMS 类型数据库(如 mysql 或 postgres)的数据源连接器

H2O.ai 是人工智能和自动机器学习领域的开源领导者，其使命是为每个人实现人工智能的民主化。它提供了一个平台，包括数据操作、各种算法、交叉验证、超参数调整的网格搜索、特性排序和模型序列化。此外，它还能帮助全球各行各业的数据科学家提高工作效率，并以更快、更简单、更经济的方式部署模型。

优点:

*   顶级的开源工具，包括 H2O-3 和 AutoML 系列。
*   R 和 Python 的接口支持将现有的工作流平稳过渡到 H2O 框架。
*   专有和开源工具的结合，无人驾驶人工智能和 H2O，提供了各种用例的工具。

缺点:

*   与 python pandas 或 pyspark 数据框架相比，H2O 框架的数据处理选项非常有限。
*   H20 错误不会返回人类可读的调试语句。

Iguazio 是一个自动化机器学习管道的数据科学平台。它通过 MLOps 和机器学习管道的端到端自动化，加速人工智能应用的大规模开发、部署和管理。这使得数据科学家能够专注于提供更好、更强大的解决方案，而不是将时间花在基础架构上。我们应该提到，它使用 Kubeflow 进行工作流编排。

优点:

*   能够在几秒钟内从笔记本电脑或 IDE 进行部署
*   与大多数流行的框架和 ML 库集成

缺点:

*   怀念 CI/CD 管道的场景

Kubeflow 是一个为想要构建和试验 ML 管道的数据科学家提供的平台。它也适用于希望将 ML 系统部署到各种开发、测试和生产级服务环境中的 ML 工程师和运营团队。Kubeflow 是一个开源的 Kubernetes-native 平台，用于促进 ML 模型的扩展。另外，它是一个基于谷歌内部 ML 管道的云原生平台。该项目致力于使在 Kubernetes 上部署 ML 工作流变得简单、可移植和可扩展。它可以作为补充工具与其他 MLOps 平台一起使用。

优点:

*   多框架集成
*   非常适合 Kubernetes 用户

缺点:

*   难以手动设置和配置。
*   高可用性不是自动的，需要手动配置。

参见[海王星和库伯流](/web/20230305040246/https://neptune.ai/vs/kubeflow)的对比。

Pachyderm 是一个强大的 MLOps 工具，允许用户控制端到端的机器学习周期。从数据沿袭，通过构建和跟踪实验，到可伸缩性选项。事实上，对于数据科学家和团队来说，这是一个简单的选择，因为它可以快速准确地跟踪知识和再现技能。它有助于开发可扩展的 ML/AI 管道，正如我们在基于受支持库的比较中看到的那样，它对大多数语言、框架和库都非常灵活。

赞成

*   与大多数流行的框架和 ML 库集成
*   当您测试新的转换管道时，它可以保留您的数据集的分支
*   基于容器，这使得您的数据环境可移植，并易于迁移到不同的云提供商。

缺点:

*   由于有如此多的移动部件，比如管理 Pachyderm 的免费版本所需的 Kubernetes 服务器，所以需要更多的学习曲线。

见[海王星和厚皮兽](/web/20230305040246/https://neptune.ai/vs/pachyderm)的对比。

Polyaxon 是一个在 Kubernetes 上自动化和复制深度学习和机器学习应用的平台。它让用户更快地迭代他们的研究和模型创建。该平台包括广泛的功能，从实验的跟踪和优化到模型管理和法规遵从性。它允许通过智能容器和节点管理进行工作负载调度，并将 GPU 服务器转变为团队或组织的共享自助服务资源。

优点:

*   可以根据自己的需求调整软件版本
*   端到端流程支持
*   使在 Kubernetes 集群上安排培训变得容易

缺点:

参见[海王星和多轴子](/web/20230305040246/https://neptune.ai/vs/polyaxon)的对比。

Valohai 是一个深度学习管理平台，帮助企业自动化深度学习基础设施。该平台使数据科学家能够管理机器编排、版本控制和数据管道。它使 DL 开发可审计，降低合规风险并削减劳动力和基础设施成本。

Valohai 提供了许多功能，包括并行 hypermeter 扫描、自定义脚本、培训课程可视化、数据探索、Jupyter 笔记本扩展、部署和生产监控。该平台允许用户在云或内部环境中使用多个中央处理单元(CPU)或图形处理单元(GPU)构建模型。此外，它兼容任何语言或框架，以及许多不同的工具和应用程序。Valohai 也是一款面向团队合作的软件，可以帮助团队领导管理协作、共享项目、分配成员、跟踪实验进度以及查看实时数据模型。

优点:

*   允许轻松管理深度学习
*   模型的全自动版本控制
*   有益的客户服务和每月检查

缺点:

## 结论

有几个 MLOps 平台用于管理机器学习的生命周期。确保在选择平台时考虑相关因素。

在整篇文章中，我探讨了在决策过程中要考虑的最符合您给定需求的不同因素。我希望这能帮助你做决定。

现在您已经有了最佳端到端平台的列表，这一切都归结到您的特定用例。

快乐学习！

### **资源**