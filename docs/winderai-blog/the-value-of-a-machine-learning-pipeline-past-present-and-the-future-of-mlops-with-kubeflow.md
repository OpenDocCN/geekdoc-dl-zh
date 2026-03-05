# 机器学习流水线的价值:使用 Kubeflow 的 MLOps 的过去、现在和未来

> 原文：<https://winder.ai/the-value-of-a-machine-learning-pipeline-past-present-and-the-future-of-mlops-with-kubeflow/>

工业[机器学习咨询](https://winder.ai/services/machine-learning/machine-learning-consulting/)项目形式多样。

有时，客户会要求进行探索性数据分析，以评估他们的数据是否可以用于帮助解决人工智能问题。其他时候，我们使用机器学习(ML)算法来自动化决策，并提高企业或产品的效率。最近，我们重新关注[强化学习](https://winder.ai/services/reinforcement-learning/)，客户要求我们帮助控制一些复杂的多步骤流程。

在每种情况下，我们都从数据科学笔记本开始。

## 笔记本不是用来生产的

笔记本是记录研究的好工具。和保存结果。但是笔记本包含了顾名思义的内容，笔记。它们完全不适合生产使用。

其原因在于缺乏工程严谨性。代码不可伸缩。它未经测试。它有大量的边缘案例。结果是不可预测和不可复制的。

笔记本的设计目的是只产生一次结果，在结束时生成一个图像，向客户展示结果已经实现。

为了扩展、强化、分类，并使管道可复制，你需要一个自动化的机器学习管道。

## 机器学习管道的价值

这种管道的结果是一个被其他人或系统消费的工件。我强调建造这个词的重要性，因为像任何建筑一样，它可以持续很长时间，是一笔宝贵的财富。

假设您想通过扩建来更新您的建筑。

如果建筑处于失修状态，如果你不能进入，如果它是危险的，那么任何有意识的建筑检查员都不会在项目完成时签字，不管扩建有多好。

当你使用一个训练管道进行模型训练时也是一样的。如果在继续开发之前，您不能证明工件是稳定的，并且管道是可重复的、可靠的和可伸缩的，那么在完成之后，您如何能够签署您的模型呢？如果你对自己的基线没有信心，那么你就无法衡量进步。

ML 工作流中管道的价值在于，它代表了应用程序以及最终业务的可验证、可量化、可重复的基础。

## ML 管道是 Dag

事实证明，大多数 ML 管道代码看起来非常像有向无环图(DAG)。换句话说，流水线是由一系列的步骤构成的，这些步骤进入后续的步骤进行进一步的计算，等等。软件工程师对 Dag 的执行有很多经验，他们倾向于称这些为流水线。

ML 管道允许您构建将模型训练为 DAG 所需的步骤。

这很有用，因为 DAG 中的许多步骤会一遍又一遍地重复。因此，复杂的管道可以利用缓存来避免重复运行会产生相同输出的步骤。

它还使“扇出”训练管道中计算量大的部分变得简单得多，从而通过并行执行任务来减少训练机器学习模型所需的时间。如果任务不相互依赖，那么它们可以并行运行。这在 ML 中经常发生，比如在执行超参数搜索时，这是测试影响 ML 模型训练的不同设置的过程。

定义了 ML 管道(或 DAG)后，您需要做的就是告诉编排层您需要每个任务做什么，以及每个任务共享哪些数据。

## 库伯流管道很糟糕——但不是你想的那样

两年半前，当谷歌首次宣布 Kubeflow 时，我第一次写了它。从那以后，它变得非常受欢迎，并且是 MLOps 社区中事实上的标准。

Kubeflow pipelines (KFP)是一个库，它利用 Kubernetes 的卓越功能进行编排，利用 Python 的定制块完成任务，并利用一个自以为是的库来提供 DAG 和数据共享框架。

KFP 很受欢迎。我们已经将它用于各种客户的生产培训工作负载，从我参与的社区聊天中收集的信息来看，这是世界其他地方的一个缩影。

它受欢迎的原因是因为没有多少项目是完全开源的，与 Kubernetes 紧密集成，并且专门针对 ML。

当然，有数百家专注于持续集成/持续交付的供应商和工具，以及许多更通用的专注于任务的 DAG 库。但是没有一个能像 KFP 那样很好地应对曼梯·里提出的一些独特挑战。它密切关注 ML 需求，例如:数据管道共享、缓存、扩展数据预处理、特征工程、训练深度学习模型和超参数调整等常见任务的能力。

但是 KFP 的实施方式有一个很大的问题。这是许多其他项目中常见的问题。DAG 的规范和控制是通过 Python 库完成的。

这样做的一个经常被引用的原因是，它使人们，在这种情况下是数据科学家，更容易与项目进行交互。但是这立即导致一些明显的问题，例如:

*   您需要大部分从头开始重写您的管道和/或笔记本
*   你需要学习和理解 KFP API
*   您需要理解库所公开的概念

我听到你说“每个图书馆都一样”。但是等等，还有更多。

## DAG 是元处理它的方式

主要问题是 DAG 是一个元级概念。

它描述了解决问题所需的一个可能的步骤组合的结构。但是 KFP 直接将它与执行任务的代码结合起来——是的，即使你使用定制的容器。你不仅需要成为手头任务的领域专家，即数据科学博士和该领域的经验，还需要 Kubeflow。

这个瀑布。

如果你想成为 Kubeflow 的专家，那么你需要了解 Kubernetes，因为这个库是如此的非抽象。几乎所有基本的 Kubernetes 原语都被添加到了 Kubeflow pipeline API 中，因为您需要这种控制来有效地编排您的容器。

这导致了大量伪装成 Kubernetes 清单的新 Python 代码。

一个无益的事实是，要使用 Kubeflow，你不仅需要一个数据科学(或其他)的博士学位，还需要成为 Kubernetes 的专家，因此是云原生的，因此是一个现代的全栈软件工程师。

我们的经验表明，很少有博士数据科学家同时也是云原生领域的专家。在温德与我们联系。AI 如果你想和他们中的一些人说话！:-p)

## 不适合团队

然后还有 Kubeflow 这个项目。我喜欢，真的喜欢！

这是一个非常好的(主要)用户驱动的(主要)开放项目的例子——显然有商业利益，但这些主要是由社区驱动的。当然，它是有用的。

但是因为 Kubeflow 不在一个团队的控制之下，因为这么多公司和个人的贡献，也是它垮台的原因。

相当长一段时间以来，它试图做得太多了。例如，它试图实现多租户功能，但是以一种非常不可靠的方式实现，只适合一小部分用户。它期望拥有和控制名称空间，并假设人们是作为个人而不是团队工作的。它需要 Istio 来做任何有用的事情，这也很棒，但使用和操作起来很棘手。

最后，对于我们的一些客户来说，我们最终剥离了所有与名称空间相关的废话，并在每个团队的名称空间中提供完整、独立的 Kubeflow 安装。

## 还没有完成——可怕的库伯流安装

然后是可怕的安装过程。每当有人说“只要安装 Kubeflow”时，我的下巴就会绷紧，露出一些牙齿。

这个项目已经经历了书中的每一个安装过程。原始清单，这个二进制文件叫做 kfctl，Kustomize，现在通过 Argo，Helm &mldr;，它从来没有完全工作，主要是由于用户的认证设置和 Istio。我做了一个[快速 Github 问题查询](https://github.com/kubeflow/kubeflow/issues?q=is%3Aissue+install)。**高达 26%的问题与安装有关。**

如果我假设这些是个人，他们可能不是，可能有很多人有同样的问题，那么每四个人中就有一个人不能安装任何东西。

这是我们[组合器背后的一个关键原因。ML 项目](https://combinator.ml/stacks/kubeflow-mlflow/)。我们想让人们更容易上手并尝试这样的工具。

## 我的机器学习管道要求

我还可以挖掘更多的问题，但我想我今天已经做了足够多的 Kubeflow 抨击了！

我的意思是，我们仍然在许多项目中使用它，所以它并不是那么糟糕。

但我确实觉得有很大的改进空间，以下是我的一些高级建议:

*   集中精力做好一件事。同类最佳是这里的解决方案。KFP 是有用的，它应该从 Kubeflow 中分离出来，就像 KServe 一样(以前它与 Kubeflow 关系密切)。
*   相关:抛弃 UI。删除所有无意义的名称空间。删除身份验证。除掉 Istio。等等。让其他项目或公司将所有这些工具整合到一个漂亮的 UX 中。他们更倾向于把事情做好。这是对开发的一种拖累，因为每个人都希望 UI 能与每个新特性一起工作。
*   数据科学家不应该了解 Kubernetes 来创建和管理机器学习管道 DAG 的规范应该是独立的
*   争议:DAG 不应该(主要)用 Python 编码。它与 ML 代码无关，我认为数据科学家会感谢你隐藏了所有与 DAG 相关的代码。对于高级用户或编程访问，我会接受 Python 库。但关键是让数据科学家专注于数据科学。
*   改善开发/调试 UX。当 KFP 出错时，开发和调试它们真的很难。
*   医生们。
*   你是如何分享代码的？如何以 GitOps 友好的方式定义依赖关系？
*   典型的你怎么转换(不推荐！)`train.py --lots --of --options`变成数据科学家要用的管道？
*   作为一名数据科学家，我真的不想担心 CUDA/库不兼容，或者我需要在我的配额中请求什么随机字母集来获得 GPU(看看你 Azure！)

我们一直在与微软合作进行概念验证，试图解决其中的一些问题。

但是我们在等像你这样的人来找出答案。当你这样做时，[让我知道](mailto:phil@winder.ai)！

## 机器学习管道的未来

那么未来是什么样子的呢？

我希望当我想开发生产质量的机器学习管道时，技术是开放的和云原生的。我希望我不必成为 Kubernetes 专家来运行实验和构建生产工件。我想专注于手头的任务，而不是编排。我需要一种简单的方法来访问、处理和快照我的数据。

我想做更多的数据科学。

## 常见问题

[什么是机器学习管道？](#what-is-a-machine-learning-pipeline)

An ML pipeline codifies the process of building a production-ready machine learning model.

[机器学习管道由什么组成？](#what-do-machine-learning-pipelines-consist-of)

ML pipelines consist of steps that run in a predefined order. Coarse-grained examples of ML pipeline steps are data preparation, model training, and model deployment. Steps may be split if they become too big or if performance is a concern.

[什么是机器学习的 CI/CD 管道？](#what-is-a-ci-cd-pipeline-for-machine-learning)

All pipelines have the same goal; to assist in deploying models into production. But when people talk about as CI/CD pipeline for machine learning, they are often focusing more on the software-engineering aspects of the problem, like building containers or deploying to an orchestration platform, rather than the data science aspects.

[kube flow 可以部署模型吗？](#can-kubeflow-deploy-models)

Deploying machine learning models into production is the ultimate goal of any machine learning pipeline, but Kubeflow doesn’t actually do the deployment of the final model artifact. Kubeflow orchestrates (i.e. deploys containers) the steps within the machine learning pipeline. If you want to deploy your model you need another service or system to do that, or write a custom step to do the deployment for you. I’d suggest using another service, because deployment can get complicated.

[Kubeflow vs. Argo](#kubeflow-vs.-argo)

Argo is a more general-purpose pipelining tool; Kubeflow actually leverages the Argo library under-the-hood. The related Argo CD project has emerged as a leader in the Kubernetes deployment space. Kubeflow caters more towards data science. If you already use Argo in your organization, it may be worth using to minimize the number of services.

[库伯流量与气流](#kubeflow-vs.-airflow)

Apache Airflow is a general purpose pipelining tool. It is used throughout industry for a variety of tasks. It does not cater towards data science specific problems. If you already use Airflow in your organization, it may be worth using to minimize the number of services.