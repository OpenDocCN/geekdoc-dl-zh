# MLOps 原则及其实施方法

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/mlops-principles>

在过去几年中，机器学习操作是一个越来越受关注的话题。随着公司在看到在其产品中使用 ML 应用程序的潜在好处后，不断投资于人工智能和机器学习，机器学习解决方案的数量正在增长。

此外，许多已经开始的项目，例如，半年或一年前，最终准备用于大规模生产。对于大多数项目和开发者来说，这意味着一个完全不同的世界、问题和挑战。

让一个 ML 项目生产准备就绪不再仅仅是模型做好它的工作和满足业务指标。这仍然是关键目标之一，但还有其他重要问题:

## 

*   模型能在一定时间内处理和响应请求吗？

*   2 如果输入数据的分布随时间变化，它会如何表现？

*   在你的项目中，你如何安全地测试一个全新版本的模型？

在某些时候，每个机器学习项目都会遇到这些问题，回答这些问题需要一套不同于研究阶段的技能和概念，无论是在哪个领域，无论是预测客户的行为或销售，检测或计算图像中的对象，还是复杂的文本分析。底线是，这些项目中的每一个都意味着产品化和维护，以便在某一点上，它开始得到回报，因此必然会遇到前面提到的打嗝。

![Graphics about MLOps](img/73bcc367365ae5a1f6f9a5e34aa3e5b7.png)

*Explanation of mlops | Source: Author*

数据科学项目的研究部分已经得到了很好的探索，即有一些标准的库、工具和概念(想想 Jupyter、pandas、实验跟踪工具等。).然而，与此同时，工程或“生产”部分对许多 ML 从业者来说仍然是一个谜——有许多灰色地带和不明确的标准，并且在很长一段时间内，没有一个设计良好、易于维护的 ML 项目的黄金路径。

这正是 MLOps(机器学习操作)应该解决的问题。在这篇文章中，我将解释:

*   它是关于什么的，
*   MLOps 的原则是什么，
*   以及如何在您当前或未来的项目中实现它们。

## MLOps 的原则:强大的 MLOps 战略的核心要素

现在，我们对 MLOps 及其在机器学习项目中的一般作用有了基本的了解，让我们更深入地了解哪些关键概念/技术将有助于您在现有或未来的项目中实施 MLOps 最佳实践。

我将介绍 ML 运营的几个“支柱”，并向您解释:

*   为什么这些对于使您的解决方案更加健壮和成熟非常重要
*   如何用可用的工具和服务实现它们

让我们更深入地了解细节，看看 MLOps 的关键原则是什么。

![The pillars of MLOps](img/c8ab6ff40cc4d036c8595b577733b0b9.png)

*The key principles of MLOps | Source: Author*

## 1.MLOps 原则:再现性和版本控制

一个成熟的机器学习项目的核心特征之一是能够重现结果。人们通常不会太关注这一点，尤其是在项目的早期阶段，他们主要是用数据、模型和各种参数进行实验。这通常是有益的，因为它可以让您找到(即使是偶然地)某个参数和数据分割率的一个好值。

然而，一个可能帮助您的项目变得更容易维护的良好实践是确保这些实验的可重复性。找到一个非常好的学习率是一个好消息。但是，如果你用相同的值再次进行实验，你会得到相同(或足够接近)的结果吗？

在某种程度上，你的实验的非确定性运行可能会给你带来一些运气，但是当你在一个团队中工作时，人们希望继续你的工作，他们可能会期望得到和你一样的结果。

在这种情况下，还有一件更重要的事情——为了让您的团队成员重现您的实验，他们还需要使用相同的配置(参数)执行完全相同的代码。你能向他们保证吗？

代码一天要修改几十次，您不仅可以修改参数的数值，还可以修改整个逻辑。为了保证项目的可再现性，您应该能够对您使用的代码进行版本控制。无论您使用的是 Jupyter Notebook 还是 Python 脚本，使用 git 这样的版本控制系统来跟踪变更应该是一件很容易的事情，这是您不能忘记的事情。

![Reproducibility and Versioning](img/bd6bc5cc73d7497a336dceb4850f8535.png)

*Reproducibility and Versioning | Source: Author*

还有什么可以被版本化以使你在项目中的工作可重复？

*   基本上，代码 EDA 代码、数据处理(转换)、培训等。
*   您的代码使用的配置文件，
*   **基础设施**！

*让我们在最后一点暂停一下——基础设施也可以而且应该进行版本控制。*

**但什么是“基础设施”？**基本上，任何种类的服务、资源和配置都托管在像 AWS 或 GCP 这样的云平台上。无论是简单的存储或数据库、一组 IAM 策略，还是更复杂的组件管道，当您需要在另一个 AWS 帐户上复制整个架构，或者需要从头开始时，对这些进行版本控制可以节省您大量的时间。

### 如何实现可再现性和版本控制原则？

至于代码本身，您应该使用 git 之类的版本控制(您可能已经这样做了)来提交和跟踪更改。根据同样的概念，您可以存储配置文件、小型测试数据或文档文件。

请记住，git 不是对大文件(如图像数据集)或 Jupyter 笔记本进行版本控制的最佳方式(这里，这与大小无关，而是比较特定版本可能会很麻烦)。

为了对数据和其他工件进行版本化，你可以使用类似于 [DVC](https://web.archive.org/web/20230313155329/https://dvc.org/) 或者[海王星](/web/20230313155329/https://neptune.ai/)的工具，这将使得存储和跟踪与你的项目或者模型相关的任何类型的数据或者元数据变得更加容易。至于笔记本——虽然将它们存储在 git 存储库中并不是一件坏事，但是你可能想要使用像 [ReviewNB](https://web.archive.org/web/20230313155329/https://www.reviewnb.com/) 这样的工具来使比较和审查变得更加容易。

[![Versioning artifacts](img/e51e5f190d22ff1c4f24cd64d2a960d8.png)](https://web.archive.org/web/20230313155329/https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/pillars-of-mlops-and-how-to-implement-them-4-1.gif?ssl=1)

*Versioning artifacts with Neptune | [Source](/web/20230313155329/https://neptune.ai/product/model-registry)*

版本化基础设施是一个普遍的问题(这整个概念被称为代码为的[基础设施)，通过众所周知的解决方案解决，如](https://web.archive.org/web/20230313155329/https://en.wikipedia.org/wiki/Infrastructure_as_code) [Terraform](https://web.archive.org/web/20230313155329/https://www.terraform.io/) 、 [Pulumi](https://web.archive.org/web/20230313155329/https://www.pulumi.com/) 或 [AWS CDK](https://web.archive.org/web/20230313155329/https://aws.amazon.com/cdk/) 。

## 2.MLOps 原则:监控

人们通常认为“监控”是顶部的樱桃，是 MLOps 或机器学习系统的最后一步。事实上，恰恰相反—**监控应该尽快实现，甚至在您的模型被部署到产品中之前**。

需要仔细观察的不仅仅是推理部署。你应该能够可视化和跟踪每一个训练实验。在每个培训课程中，您可以跟踪:

*   训练指标的历史，如准确性、F1、训练和验证损失等。,
*   训练期间脚本使用的 CPU 或 GPU、RAM 或磁盘的利用率，
*   在训练阶段之后产生的对维持集的预测，
*   模型的初始和最终权重，

以及与您的用例相关的任何其他指标。

现在，从训练转移到推理，这里也有很多东西需要监控。我们可以将这些分为两组:

1.  **部署服务本身的服务级监控**(Flask web Service，托管在 Kubernetes 上的微服务， [AWS](https://web.archive.org/web/20230313155329/https://aws.amazon.com/lambda/) Lambda 等。);了解处理来自用户的单个请求需要多长时间，平均负载大小是多少，您的服务使用了多少资源(CPU/GPU、RAM)等等，这一点很重要。

2.  模型级监控，即模型返回的预测以及模型收到的输入数据。前者可用于分析一段时间内的目标值分布，后者可以告诉您输入的分布，输入也可以随时间而变化，例如，财务模型可以将工资视为输入特征之一，其分布可以随时间的推移而改变，因为工资较高-这可能表明您的模型已经过时，需要重新培训。

![Training and inference](img/8073da3a17f1662cadb61bb606307ac6.png)

*Training and inference as a part of monitoring | Source: Author*

### 如何实施监测原则？

至于培训，你可以使用大量的实验跟踪工具，例如:

它们中的大部分可以很容易地集成到您的代码中(可以通过 pip 安装),并允许您在训练/数据处理期间实时记录和可视化指标。

[![Monitoring experiments with Neptune](img/3b2e0c0dbf80fc6891c2c44701b70355.png)](https://web.archive.org/web/20230313155329/https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/pillars-of-mlops-and-how-to-implement-them-6.png?ssl=1)

*Monitoring experiments with Neptune | [Source](/web/20230313155329/https://neptune.ai/product/experiment-tracking)*

关于推理/模型部署——这取决于您使用的服务或工具。如果是 AWS Lambda，它已经支持发送给 AWS CloudWatch 服务的相当广泛的日志记录[。另一方面，如果你想在 Kubernetes 上部署你的模型，可能最流行的栈是用于导出指标的](https://web.archive.org/web/20230313155329/https://docs.aws.amazon.com/lambda/latest/dg/monitoring-functions-access-metrics.html) [Prometheus](https://web.archive.org/web/20230313155329/https://prometheus.io/docs/introduction/overview/) 和用于创建你的定制仪表板和实时可视化指标和数据的 [Grafana](https://web.archive.org/web/20230313155329/https://prometheus.io/docs/visualization/grafana/) 。

## 3.MLOps 原则:测试

在机器学习团队中，很少有人提到“*测试*”、“*编写测试*”等。在传统的软件工程项目中，编写单元、集成或端到端测试更加常见(或者已经成为一种标准，希望如此)。那么 ML 中的测试是什么样子的呢？

有几件事您可能希望始终保持验证状态:

## 

*   1 输入数据的数量和质量，

*   2 输入数据的特征模式(预期值范围等。)，

*   3 你的加工(转换)作业产生的数据，以及作业本身，

*   4 您的功能和数据管道的合规性(如 GDPR)。

它将使你的机器学习管道更加健壮和有弹性。有了这样的测试，您就可以在数据或基础架构出现意外变化时立即检测出来，从而有更多的时间做出相应的反应。

### 如何实施测试原则？

让我再把它分成几个主题:

1.  对于数据验证，你可以使用开源框架，比如[远大前程](https://web.archive.org/web/20230313155329/https://github.com/great-expectations/great_expectations)或者[深度检查](https://web.archive.org/web/20230313155329/https://github.com/deepchecks/deepchecks)。当然，根据您的用例以及使用外部工具的意愿，您也可以自己实现基本的检查。一个最简单的想法是从训练数据中计算统计数据，并使用这些数据作为其他数据集(如生产中的测试数据)的预期。

2.  在任何种类的管道中，都可以测试转换，甚至是最简单的脚本，通常与测试典型软件代码的方式相同。相信我，如果您使用一个处理/ETL 作业来定期转换新的输入数据，那么在将数据进一步推送到训练脚本之前，您会希望确保它能够工作并产生有效的结果。

3.  特别是当涉及到工程或基础设施时，您应该总是倾向于将基础设施作为设置任何云资源的代码范例，我已经在关于可再现性的一节中提到了这一点。尽管还不常见，[基础设施代码也可以进行单元测试](https://web.archive.org/web/20230313155329/https://www.terraform.io/cdktf/test/unit-tests)。

4.  关于符合性测试，应该针对每个项目和公司具体地仔细实施。您可以在这里阅读更多关于模型治理的有用测试和过程[。](https://web.archive.org/web/20230313155329/https://ml-ops.org/content/model-governance)

## 4.MLOps 原则:自动化

最后但同样重要的是，MLOps 的一个重要方面。它实际上与我们到目前为止讨论的所有内容都有关系——版本控制、监控、测试等等。自动化的重要性已经在[ml-ops.org](https://web.archive.org/web/20230313155329/https://ml-ops.org/content/mlops-principles#automation)得到了很好的描述(必读):

> 数据、ML 模型和代码管道的自动化水平决定了 ML 过程的成熟度。随着成熟度的提高，新模型的训练速度也在提高。MLOps 团队的目标是将 ML 模型自动部署到核心软件系统中或作为服务组件。这意味着自动化完整的 ML 工作流步骤，无需任何手动干预。

如何以及在多大程度上**自动化你的项目是 MLOps 工程师的关键问题之一**。在理想的情况下(有无尽的时间，明确的目标，无限多的工程师等等。)，您几乎可以自动化管道中的每一个步骤。

![An automated ML pipeline](img/fb82329768dd39e3ca06b2a8a11c2031.png)

*An example of an automated ML pipeline | [Source](https://web.archive.org/web/20230313155329/https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#mlops_level_2_cicd_pipeline_automation)*

想象以下工作流程:

1.  新数据到达原始数据存储器，
2.  然后清理、处理数据，并创建特征，
3.  数据也得到测试的功能模式，GDPR；在计算机视觉项目中，它还可以包括特定的检查，例如图像质量，或者可能涉及面部模糊，
4.  如果适用，已处理的特征被保存到特征存储中以备将来重用，
5.  一旦数据准备就绪，您的训练脚本就会自动触发，
6.  所有的培训历史和指标都会被自然地跟踪和可视化，
7.  模型已经准备好了，并且证明非常有前景，这也是自动评估的，并且以相同的自动方式触发部署脚本，
8.  …

我可以继续进行故障处理、警报、自动化数据标记、检测模型中的性能衰减(或数据漂移)以及触发自动模型再训练。

关键是，它描述了一个几乎理想的系统，不需要人为干预。如果它必须适用于一个以上的模型/用例，那么它将花费大量的时间来实现。那么如何以正确的方式去做呢？

### 如何实现自动化原则？

首先，没有配方，也没有自动化的适量。我的意思是，这取决于你的团队和项目目标以及团队结构。

然而，有一些指导方针或示例体系结构可能会给你一些意义和问题的答案，“我实际上应该自动化多少？”。最常被引用的资源之一是 Google 的 [MLOps Levels。](https://web.archive.org/web/20230313155329/https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#mlops_level_2_cicd_pipeline_automation)

假设您已经知道您将自动化系统的哪些部分(例如，数据摄取和处理)。但是你应该使用什么样的工具呢？

这一部分可能是目前最模糊的，因为 MLOps 系统中的每个组件都有几十个工具。你必须评估和选择什么是适合你的，但是有像[州 MLOps](https://web.archive.org/web/20230313155329/https://ml-ops.org/content/state-of-mlops) 或 [MLOps 社区](https://web.archive.org/web/20230313155329/https://mlops.community/learn/)这样的地方会告诉你什么是最受欢迎的选择。

## Airbnb 的真实例子

现在让我们讨论 Airbnb 如何简化复杂的 ML 工作流程并设法将过多的不同项目纳入一个系统的例子。创建 Bighead 是为了确保模型的无缝开发及其全面管理。

![Airbnb’s End-to-End ML Platform](img/b5fd138516339ae51ed4a4f05ed6fb96.png)

*Airbnb’s End-to-End ML Platform | [Source](https://web.archive.org/web/20230313155329/https://medium.com/acing-ai/airbnbs-end-to-end-ml-platform-8f9cb8ba71d8)*

让我们看一下每个组件的解释，并了解它与原则的关系:

### ML 数据管理框架

Zipline 是一个用于定义、管理和共享功能的框架。它符合大多数条件——存储可以共享的功能定义(可能与其他项目共享),让 Airbnb 拥有了**再现性**和**版本化**(数据集和功能)的能力。不仅如此，正如作者所说，该框架还有助于实现更好的数据质量检查(**测试**原则)和**监控**ML 数据管道。

### Redspot(托管 Jupyter 笔记本服务)

图表中的下一个组件是 Redspot，它是一种“托管的、容器化的、多租户 Jupyter 笔记本服务”。作者说，每个用户的环境都以 Docker 映像/容器的形式存在。

这使得其他开发者在其他机器上复制他们的代码和实验变得更加容易。同时，这些用户环境可以在内部容器注册中心自然地**版本化**。

### 大头图书馆

再一次，关于**再现性**的附加点。Bighead Library 再次专注于存储和共享特性和元数据，这就像使用 Zipline 更容易一样，是解决**版本控制**和**测试** ML 数据的好办法。

### 沉思

> *Deep think 是一个共享的 REST API 服务，用于在线推理。它支持 ML 管道中集成的所有框架。部署完全由配置驱动，因此数据科学家不必让工程师参与推出新模型。然后，工程师可以连接到其他服务的 REST API 来获得分数。此外，还支持从 K/V 存储中加载数据。它还为模型性能的监控和离线分析提供了标准化的日志记录、警报和仪表板。*

Airbnb 平台的最后一个组件专注于另外两个原则:**自动化**(尽管根据图表，自动化可能也已经融入了之前的组件)和**监控**，通过提供*“标准化的日志记录、警报和仪表板以监控(……)模型性能*。

deep think 部署是“完全配置驱动的”,这意味着大多数技术细节对用户是隐藏的，并且很可能是自动化的。数据科学家用来部署新模型的这些配置文件的正确版本控制将允许其他开发者在另一个账户或另一个项目中复制部署

所有这些组件共同实现了一个运转良好的 MLOps 机器，并构建了一个流畅的工作流程，这是 Airbnb ML 功能不可或缺的一部分。

## 摘要

读完这篇文章后，你有希望知道这些原则(版本控制、监控、测试和自动化)是如何协同工作的，以及为什么它们对机器学习平台很重要。

如果你对这个话题更感兴趣，并且想阅读其他涉及这些原则的真实世界的 ML 平台，有许多由像[【优步】](https://web.archive.org/web/20230313155329/https://eng.uber.com/michelangelo/)、 [Instacart](https://web.archive.org/web/20230313155329/https://www.instacart.com/company/how-its-made/griffin-how-instacarts-ml-platform-tripled-ml-applications-in-a-year/) 和 [others(网飞、Spotify)](https://web.archive.org/web/20230313155329/https://towardsdatascience.com/lessons-on-ml-platforms-from-netflix-doordash-spotify-and-more-f455400115c7) 这样的公司写的[例子](https://web.archive.org/web/20230313155329/https://github.com/visenger/awesome-mlops#existing-ml-systems)和博客文章，在这些文章中，他们解释了他们的内部 ML 系统是如何构建的。

在其中一些文章中，您可能找不到任何关于“mlops 支柱”的明确信息，而是关于在该平台中使用或实现的特定组件和工具。您很可能会看到“特性库”或“模型注册”，而不是“版本控制和再现性”。类似地，“工作流程编排”或“ML 管道”为平台带来了“自动化”。记住这一点，好好读一读！