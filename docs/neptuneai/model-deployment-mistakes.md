# 可能让您付出巨大代价的 5 个模型部署错误

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/model-deployment-mistakes>

在数据科学项目中，模型部署可能是整个生命周期中**最关键的**和**复杂的**部分。

运营或任务关键型 ML 需要全面的设计。您必须考虑工件传承和跟踪、避免人为错误的自动部署、测试和质量检查、模型在线时的功能可用性…以及许多其他事情。

在本文中，我们整理了一份通常发生在生命周期最后阶段的常见错误列表。这些更关注软件架构，但在处理推理服务时扮演着非常重要的角色*。*

## 错误 1:手动部署您的模型

手动部署推理服务有很大的风险。大多数 ML 服务需要执行多个命令才能部署。假设我们正在部署一个 FastAPI web 服务，它是在线推理的标准。以下是成功部署需要执行的典型步骤:

## 

*   执行测试套件并获得代码覆盖率
*   2 获取 Docker 注册表的凭证
*   3 构建推理服务映像
*   4 调用 Kubernetes 从 Docker 注册中心获取图像并部署服务

想象一下手动执行这些步骤(以及您需要的所有设置)。很有可能发生人为错误——您可能忘记从您的模型注册中心更新模型路径，您可能忘记运行测试，您可能没有首先进入预生产环境就直接部署到生产环境中，等等。

### 您可以尝试的:自动持续集成和部署

持续集成工具

![Model Deployment - CI/CD Pipelines](img/6c4d772ca970c52ce3f10a23aae12a32.png)

*Model deployment – CI/CD pipelines I [Source](https://web.archive.org/web/20221201161359/https://hackernoon.com/understanding-the-basic-concepts-of-cicd-fw4k32s1) *

幸运的是，有各种各样的工具可以用来自动化这些步骤。Github 动作、[詹金斯](https://web.archive.org/web/20221201161359/https://www.jenkins.io/)和 [Gitlab CI/CD](https://web.archive.org/web/20221201161359/https://docs.gitlab.com/ee/ci/) 就是很好的例子。你可以在这篇文章中读到更多关于它们的信息[。](/web/20221201161359/https://neptune.ai/blog/continuous-integration-continuous-deployment-tools-for-machine-learning)

#### 这些工具允许您定义基于服务代码库中发生的某些事件触发的**工作流。当将一个分支合并到开发中时，您可以部署到集成环境中，当新特性到达主分支时，您可以部署到生产中。**

持续部署工具

对于持续部署步骤，有像 [ArgoCD](https://web.archive.org/web/20221201161359/https://argo-cd.readthedocs.io/en/stable/) 、 [Jenkins-X](https://web.archive.org/web/20221201161359/https://jenkins-x.io/) 或 [Flux](https://web.archive.org/web/20221201161359/https://fluxcd.io/) 这样的工具，它们是基于 GitOps 的 Kubernetes pods 部署器(如果你不知道那是什么，Gitlab 提供了一篇非常全面的文章解释它是什么[这里是](https://web.archive.org/web/20221201161359/https://about.gitlab.com/topics/gitops/))。这些工具将负责将您的变更发布到生产中。

#### 本质上，这些 CI/CD 管道是 UNIX 命令集**,它们自动执行上面定义的所有步骤**,并且总是在容器化的环境中执行。这保证了每个部署都是可重复的和确定的。

错误 2:忽视部署策略的使用

最简单的 ML 模型部署策略基本上是通过更新运行 API 的容器映像，将旧服务转换为新服务。这通常被称为 ***重建部署模式*** ，这是一个非常过时的策略，很少有公司还在使用。这种部署的主要缺点是它**会导致服务停机**一段特定的时间(只要服务需要启动)，这在某些应用中是不可接受的。除此之外，如果你没有一个更复杂的策略，你将无法利用一些强大的技术来提高可靠性、跟踪和实验。

## 您可以尝试的:使用一种部署策略

蓝绿色部署

### 该部署策略包括同时在生产环境中部署服务的两个版本(旧版本和新版本)。当部署新版本时，消费者的**流量通过负载均衡器逐渐被重定向**到这个版本。如果新服务产生任何类型的错误，流量负载将立即重定向到旧版本(作为一种自动回滚)。

#### 蓝绿色部署的好处是，您可以在很早的阶段发现错误，同时仍然为大多数消费者提供服务。它有一个嵌入式的灾难恢复功能，可以切换回之前的工作版本。在 ML 空间中，这种技术特别有用，因为模型容易由于各种原因产生错误。

![Blue-green model deployment](img/11bffaa50b53e08af71560c6875a7034.png)

*Blue-green model deployment | [Source](https://web.archive.org/web/20221201161359/https://martinfowler.com/bliki/BlueGreenDeployment.html) *

查看这个 [Martin Fowler 对蓝绿部署的解释](https://web.archive.org/web/20221201161359/https://martinfowler.com/bliki/BlueGreenDeployment.html)来了解更多关于这个策略的信息。

金丝雀部署

金丝雀部署类似于蓝绿色部署。主要区别在于，新旧版本之间的流量平衡不是基于百分比完成的，而是基于向新用户逐渐增加的模型版本。

#### 基本上，该模型首先发布给特定的生产用户群体，以便在早期捕捉错误和问题(您甚至可以在向公众推出之前向内部员工发布)。在确认该服务对他们有效后，该服务逐渐推广到越来越多的用户。部署受到密切监控，以捕捉潜在的问题，如错误、不良行为、高延迟、CPU 或 RAM 的过度使用等。这通常是一个缓慢的过程，但比其他类型的部署风险更小。如果出现问题，回滚相当容易。

影子部署

![Canary model deployment](img/c1d40e6db72e851e670e8dd16da7b5f2.png)

*Canary model deployment | [Source](https://web.archive.org/web/20221201161359/https://gowrishankar.info/blog/istio-service-mesh-canary-release-routing-strategies-for-ml-deployments-in-a-kubernetes-cluster/) *

这种部署不像以前的那样常见，而且被低估了。它提供了一个巨大的好处，不必直接将模型放归自然。

#### 它的工作方式是通过**将传入的请求**复制到另一个 sidecar 服务，该服务包含 ML 模型的新版本。这个新模型不会对消费者产生任何影响，也就是说，响应来自于唯一存在的稳定版本。

例如，如果您刚刚为在线交易构建了一个新的欺诈检测模型，但您不太愿意在没有使用真实数据进行测试的情况下将其发布到产品中，您可以将其部署为影子服务。旧的服务仍将与不同的系统交互，但您将能够实时评估新版本。如果您有一个定义良好的 ML 监控架构([进行 ML 模型监控的最佳工具](/web/20221201161359/https://neptune.ai/blog/ml-model-monitoring-best-tools))，您将能够通过引入一个具有基本事实的反馈回路来评估模型的准确性。对于这个用例，这意味着知道交易最终是否是欺诈性的。

这种类型的部署还需要配置负载平衡器，以便一次将请求复制到两个版本。根据使用案例，还可以将生产负载流量异步重放到影子版本中，以避免影响负载平衡器的性能。

![Shadow model deployment](img/700b0c311acafe00e96a70c865cd8ef8.png)

*Shadow model deployment | [Source](https://web.archive.org/web/20221201161359/https://christophergs.com/machine%20learning/2019/03/30/deploying-machine-learning-applications-in-shadow-mode/)*

错误#3:没有启用自动化(预测)服务回滚

想象一下，你有一个生产模型，负责你的应用程序主要服务的动态定价。高度依赖这种模式的公司有优步、Bolt AirBnB、亚马逊、Shopify 等等。,

## 然后，假设数据科学团队创建了 ML 模型的一个新的改进版本。但是部署失败了(由于任何原因)。应用程序将不得不切换到备用价格，因为模型 API 不会响应。现在，价格不是个性化的(在最好的情况下)，当然也不是动态的。

这个问题可能会导致收入大幅下降，直到服务得到解决，新模式得到部署。

您可以尝试的:为您的部署启用自动回滚

如果您的 ML 模型服务于您的应用程序的一个非常重要的特性，那么拥有一个健壮的回滚系统是至关重要的。回滚系统允许**将服务切换回之前的版本**，并减少应用程序运行不良的时间。正如我们之前已经看到的，这是蓝绿部署的重要部分。如果在渐进版本中没有出现任何错误，新版本只能接收 100%的流量。

### 手动回滚触发器

回滚到以前版本的另一种简便方法是启用**手动回滚触发器。**这对于在生产中部署的 ML 模型特别有用。有时 ML 服务不会失败，但是它们会开始返回异常输出，由于低效的模型编译和许多其他原因，需要很长时间来响应。这类问题通常不会被自动检测到，过一会儿就会被发现。通常，客户支持票开始到达，您会收到问题通知。

可以通过多种方式部署手动回滚触发器。比如 Github 允许设置 [*工作流调度*](https://web.archive.org/web/20221201161359/https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#workflow_dispatch) 事件。这些允许您通过提供一些输入，从您的服务存储库中手动运行 Github 工作流。您可以设置要回滚到的提交、标记或分支。

![Kubernetes deployment](img/f583488dab6812b28d56cbb16dad3fd6.png)

*Kubernetes deployment | [Source](https://web.archive.org/web/20221201161359/https://www.nclouds.com/blog/kubernetes-deployment-using-jenkins-with-auto-rollback-for-continuous-delivery/) *

#### 错误 4:忽略推理服务中的负载测试！

ML 服务往往比典型的后端服务要慢。ML 模型在做出预测时并不总是很快。这真的取决于你建立的模型的类型。例如，如果您使用一个[转换器模型](https://web.archive.org/web/20221201161359/https://arxiv.org/abs/1706.03762)进行文本分类，那么根据输入序列的长度，推断时间可能会花费一些时间。一般来说，神经网络也是高度 CPU 密集型的，在某些情况下，它们也会占用大量 RAM 内存。

你可以尝试的:想想交通高峰！做压力测试，把自动缩放策略

## 由于这些潜在的性能问题，设计一个**高效的硬件**基础设施对于及时返回响应至关重要。有必要了解哪种配置策略是根据硬件使用情况自动扩展系统、设置服务主机的基本内存和 CPU 能力、设置服务主机的初始数量等的最佳配置策略。

所有这些都可以在你的 [Kubernetes 配置 YAMLs](https://web.archive.org/web/20221201161359/https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/) 中定义，如果你正在部署一个 web 服务，或者在你的 [Lambda 配置](https://web.archive.org/web/20221201161359/https://docs.aws.amazon.com/lambda/latest/dg/invocation-scaling.html)中定义，如果你正在 AWS 中部署一个无服务器架构。(GCP 和 Azure 的无服务器*功能*也有类似选项)。获得这些数字的最好方法是进行压力和负载测试。您也可以跳过这一步，在服务已经投入生产的情况下校准配置，但是这样风险更大。

### 什么是负载测试？

负载测试由**模拟临时环境中的真实流量负载**组成。也就是说，尝试估计服务每秒将接收多少请求(也包括峰值),并从外部主机或云本地执行测试。有几个开源和付费的工具可以用来做这件事，比如[蝗虫](https://web.archive.org/web/20221201161359/https://locust.io/)(如果你使用 Python 就很容易使用)或者 [Apache JMeter](https://web.archive.org/web/20221201161359/https://jmeter.apache.org/) 。你可以在这里找到更多选项[。](https://web.archive.org/web/20221201161359/https://www.softwaretestinghelp.com/performance-testing-tools-load-testing-tools/)

您必须定义每秒请求数和以特定速率产生的用户数，以模拟您的服务。这些工具还允许您[定义定制负载](https://web.archive.org/web/20221201161359/https://docs.locust.io/en/stable/custom-load-shape.html)。例如，您可以模拟在一天的高峰时段有较高的流量，而在周末有较低的负载。

负载测试结果将显示服务返回 503 错误或返回高延迟响应的比率。你甚至可以用自己的监控系统来复查这种行为( [Datadog](https://web.archive.org/web/20221201161359/https://www.datadoghq.com/) 、 [Grafana](https://web.archive.org/web/20221201161359/https://grafana.com/) 等)。

负载测试结果将如何帮助您优化您的架构？

这些测试结果将允许您定义在正常的请求/秒速率下您需要有多少个 pod，并估计您的 T2 服务需要扩展多少。然后，您将能够设置 CPU 和 RAM 阈值来触发水平自动扩展策略。它还将为您提供延迟分布的感觉，并决定它对于手头的用例是否足够，或者在投入生产之前是否需要应用任何必要的优化。

然而，模拟真实行为总是困难的，极端的交通高峰也不总是可预测的。这将我们引向下一点，拥有一个健壮的和定义良好的监控系统将允许团队在任何问题发生时得到早期通知。您需要有警报来监控延迟、错误、日志模式中的异常等。

错误 5:没有监控你的 ML 系统！

#### 很明显，在您的生产模型上没有一个适当的监控层是一个很大的错误。这在过去几年变得更加重要，因为云环境中的技术堆栈越来越多。许多不同的组件相互作用，使得根本原因分析更加困难。当其中一个组件出现故障时，非常有必要准确地了解问题是如何发生的。

此外，在处理 ML 系统时，我们还有一个额外的挑战。机器学习模型依赖于它们接受训练的数据，但它们在生产中使用从未见过的数据。这提出了一个明显的问题，ML 模型本质上是**错误的，**但是正如常见的统计学格言所说，“……*有些是有用的”*。因此，随着时间的推移，监控我们的模型有多错误是至关重要的。

您可以尝试的:实现一个监控层

## ML 系统**非常复杂，难以监控**。为了获得完整的可观测性图层，您需要设置多个监控级别。这些是硬件&服务监控、反馈回路和模型退化。

硬件和服务监控

硬件和服务监控是绝对必须实施的，包括:CPU 和 RAM 使用情况、网络吞吐量、响应延迟、完整的端到端跟踪和日志记录。这将允许足够快地补救技术问题，以避免对用户体验以及公司的最终收入产生负面影响。这可以通过一些已经提到的工具很好的解决，比如[Datadog](https://web.archive.org/web/20221201161359/https://www.datadoghq.com/)**[Grafana](https://web.archive.org/web/20221201161359/https://grafana.com/)**或者其他的比如 [New Relic](https://web.archive.org/web/20221201161359/https://newrelic.com/) 或者 [Dynatrace](https://web.archive.org/web/20221201161359/https://www.dynatrace.com/) 。****

 ****还值得一提的是 Neptune 如何帮助监控训练指标和硬件消耗(在[文档](https://web.archive.org/web/20221201161359/https://docs.neptune.ai/how-to-guides/model-monitoring/monitor-model-training-live)中阅读更多相关信息)。如果您的模型需要定期重新训练，或者是您的生产服务的一部分(例如:训练+预测发生在同一作业中)，这将特别有用。

#### ​​

反馈回路

离线评估中使用的传统 ML 指标也可以在生产中进行评估，但前提是有实际数据可用。这就是通常所说的 [***反馈回路***](https://web.archive.org/web/20221201161359/https://www.clarifai.com/blog/closing-the-loop-how-feedback-loops-help-to-maintain-quality-long-term-ai-results) 。计算这些非常依赖于地面真实数据出现的方式。

![Monitoring dashboard in Datadog](img/f81f3acae3031fff9ac17e940b97ff97.png)

*Monitoring dashboard in Datadog | [Source](https://web.archive.org/web/20221201161359/https://www.capterra.es/software/135453/datadog-cloud-monitoring)*

在一些用例中，您需要等待一段特定的时间，直到您得到预测的真实结果(欺诈检测)，在另一些用例中，您得到一个模糊的结果(语法错误纠正)，在另一些用例中，它们甚至不可用。但大多数时候，你可以得出一个用户行为结果，作为检查模型是否支持你试图优化的业务指标的代理。

模型退化

[![Hardware monitoring in Neptune ](img/206ce80bec21c38ab1b3cf4901578c7f.png)](https://web.archive.org/web/20221201161359/https://app.neptune.ai/common/quickstarts/e/QUI-28177/monitoring)

*Hardware monitoring in Neptune | [Source](https://web.archive.org/web/20221201161359/https://app.neptune.ai/common/quickstarts/e/QUI-28177/monitoring)*

除了这两个(硬件、服务和反馈回路)，ML 模型暴露了一个新的基本复杂性。这叫 ***型号退化*** 。这种效应不会在系统中产生错误，但意味着预测的质量会逐渐下降，如果没有复杂的 ML 友好监控层，很难检测到这一点。

#### 模型退化主要是由**的数据漂移引起的。*这意味着你输入在线算法的数据相对于训练数据已经发生了某种程度的变化。这通常通过比较生产数据与训练数据的分布来进行统计测试。在[](https://web.archive.org/web/20221201161359/https://arxiv.org/abs/1810.11953)**的论文中，你可以读到更多关于这种效果的细节。***

 ***由于 MLOps 完全是自动化，因此检测这种类型的问题对公司来说是一个巨大的工程挑战。大多数提供 MLOps 工具的公司已经解决了监控培训阶段的问题，但是生产模型退化监控仍然是一个难以概括解决的概念( [SageMaker](https://web.archive.org/web/20221201161359/https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html) 、 [Whylabs](https://web.archive.org/web/20221201161359/https://whylabs.ai/?section=model-health) 和 [Aporia](https://web.archive.org/web/20221201161359/https://www.aporia.com/) 是少数几个已经提出了通用用例解决方案的工具)。

将 web 服务日志移动到一个数据存储区，在那里它们可以被批量提取和分析，这通常是通过使用一个**流管道**将记录放入一个流中来解决的，这些记录稍后将被写入对象存储区。例如，您可以使用 Kafka 主题和 Lambda 函数来接收要素和预测记录，并将它们保存在 S3。稍后，您可以设置一个定期的气流作业，提取 S3 的所有这些数据，并将它们与训练数据进行比较。如果差异很大，你可以向 ML 工程团队发送一个延迟通知。如果降级对您的系统至关重要，您可以触发模型训练并自动部署新模型。

#### 结束了！

在本文中，我们介绍了 ML 工程师在部署他们的第一个模型时犯的一些关键错误。所有这些对于基于 ML 的生产系统的长期成功都是至关重要的。作为一个提示，小心过度设计！迭代是解决 ML 项目的最佳方式，因为实现所有这些建议的成本和工作量都很高。您的模型用例 ROI 需要支持它。

如果您想了解更多关于本文中出现的所有主题的信息，请查看以下文章:

参考

Moving web services logs to a data store where they can be extracted and analyzed in bulk is usually solved by using a **streaming pipeline** to put records in a stream which are later written in object storage. For example, you could use a Kafka topic and a Lambda function to receive features and prediction records and save them in S3\. Later on, you can set up a periodic Airflow job that extracts all these data in S3 and compare them against the training data. If there’s a big difference, you could send a Slack notification to the ML engineering team. And if the degradation is critical for your system, you can trigger model training and deploy the new model automatically.

## Wrapping up!

In this article, we introduced some of the key mistakes that ML engineers make when deploying their first models. All of these are critical for the long-term success of an ML-based production system. And just as a note, beware of over-engineering! Iteration is the best way to solve ML projects because the cost and effort of implementing all of these suggestions are high. Your model use case ROI needs to support it. 

If you want to know more about all the topics surfaced in this article, check out these articles:

### References*******