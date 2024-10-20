# 模型部署策略

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/model-deployment-strategies>

近年来，大数据和机器学习已经被大多数主要行业采用，大多数初创公司也倾向于同样的做法。随着数据成为所有公司不可或缺的一部分，处理数据的方法，即获得有意义的见解和模式是必不可少的。这就是机器学习发挥作用的地方。

我们已经知道机器学习系统处理大量数据的效率有多高，并根据手头的任务，实时产生结果。但是这些系统需要正确地管理和部署，以便手头的任务高效地执行。本文旨在为您提供关于**模型部署策略**的信息，以及如何选择最适合您的应用的策略。

[![The entire pipeline of a data-science project](img/a7fa84cce24add10f6b4689cb5cc4a73.png)](https://web.archive.org/web/20230119095952/https://neptune.ai/model-deployment-strategies_5)

*The image above depicts the entire pipeline of a data-science project | [Source](https://web.archive.org/web/20230119095952/https://arxiv.org/pdf/2103.08937.pdf)*

**我们将介绍以下模型部署的策略和技术:**

## 

*   1 影子评估
*   2 A/B 测试
*   3 多臂土匪
*   4 蓝绿色展开
*   金丝雀测试
*   6 特征标志
*   7 滚动部署
*   8 再造战略

这些策略可以分为两类:

*   **静态部署策略**:这些是手动处理流量或请求分配的策略。例如影子评估、A/B 测试、金丝雀测试、滚动部署、蓝绿色部署等等。
*   **动态部署策略:**这些是自动处理流量或请求分配的策略。多臂强盗就是一个例子。

![Model deployment strategies](img/e6f7e6309fee48dc734634c4acd88418.png)

*Model deployment strategies | [Source](https://web.archive.org/web/20230119095952/https://www.coursera.org/lecture/ml-models-human-in-the-loop-pipelines/model-deployment-strategies-6icWT)*

首先，让我们快速了解一下什么是模型生命周期和模型部署。

## ML 模型的生命周期

机器学习模型的生命周期指的是构建整个数据科学或人工智能项目的整个过程。它类似于软件开发生命周期(SDLC ),但在一些关键领域有所不同，例如在部署之前使用实时数据来评估模型性能。ML 模型的生命周期或模型开发生命周期(MDLC)主要有五个阶段:

## 

*   1 数据采集
*   2 创建模型和培训
*   3 测试和评估
*   4 部署和生产
*   5 监控

[![Model development lifecycle (MDLC)](img/226c5921358f1ef05dce5f608a49a198.png)](https://web.archive.org/web/20230119095952/https://neptune.ai/model-deployment-strategies_12)

*Model development lifecycle (MDLC) | [Source](https://web.archive.org/web/20230119095952/https://towardsdatascience.com/the-machine-learning-lifecycle-in-2021-473717c633bc)*

现在，另一个你必须熟悉的术语是 **[MLOps](/web/20230119095952/https://neptune.ai/blog/mlops)** 。MLOps 通常是实现 ML 生命周期的一系列实践。它将机器学习和软件应用结合在一起。简而言之，这是数据科学家和运营团队之间的合作，负责并协调整个 ML 生命周期。MLOps 关注的三个关键领域是**持续集成**、**持续部署、**和**持续测试**。

## 什么是模型部署(或模型发布)？

模型部署(发布)是一个过程，使您能够将机器学习模型集成到生产中，以根据真实世界的数据做出决策。这实际上是在[监控](/web/20230119095952/https://neptune.ai/blog/ml-model-monitoring-best-tools)之前的 ML 生命周期的倒数第二个阶段。一旦部署完毕，还需要对模型进行监控，以检查数据摄取、特征工程、培训、测试等整个过程是否正确对齐，从而无需人工干预，整个过程是自动的。

但是在部署模型之前，必须评估和测试经过训练的 ML 模型是否适合部署到生产中。对模型的性能、效率、甚至错误和问题进行测试。在部署 ML 模型之前，可以使用各种策略。让我们探索它们。

## 模型部署策略

策略允许我们评估 ML 模型的性能和能力，并发现与模型相关的问题。要记住的一个关键点是，策略通常取决于手头的任务和资源。有些策略可能是很好的资源，但计算量很大，而有些策略可以轻松完成工作。我们来讨论其中的几个。

### 1.影子部署策略

在影子部署或影子模式中，新模型与实时模型一起部署新特性。在这种情况下，新部署的模型被称为**影子模型**。影子模型处理所有的请求，就像真实模型一样，只是它没有向公众发布。

这种策略允许我们通过在真实世界的数据上测试影子模型来更好地评估影子模型，同时不中断真实模型提供的服务。

![Shadow deployment strategy](img/02120186d95dd32d20bda776e24d0bbb.png)

*Shadow deployment strategy | [Sour](https://web.archive.org/web/20230119095952/https://alexgude.com/blog/machine-learning-deployment-shadow-mode/)[ce](https://web.archive.org/web/20230119095952/https://alexgude.com/blog/machine-learning-deployment-shadow-mode/)*

#### 方法论:冠军 vs 挑战者

在影子评估中，使用两个 API 端点将请求发送给彼此并行运行的两个模型。在推断过程中，来自两个模型的预测被计算和存储，但是在返回给用户的应用中仅使用来自活动模型的预测。

将来自实时和阴影模型的预测值与地面真实值进行比较。一旦有了结果，数据科学家就可以决定是否在生产中全面部署影子模型。

但是人们也可以使用[冠军/挑战者](https://web.archive.org/web/20230119095952/https://medium.com/decision-automation/what-is-champion-challenger-and-how-does-it-enable-choosing-the-right-decision-f57b8b653149)框架，以测试多个影子模型并与现有模型进行比较的方式。本质上，具有最佳准确性或关键性能指数(KPI)的模型被选择和部署。

**优点**:

*   模型评估是高效的，因为两个模型并行运行，对流量没有影响。
*   无论交通状况如何，都不能超载。
*   您可以监控影子模型，它允许您检查稳定性和性能；这降低了风险。

**缺点**:

*   因为支持影子模型所需的资源而昂贵。
*   影子部署可能是乏味的，尤其是如果您关注模型性能的不同方面，比如度量比较、延迟、负载测试等等。
*   不提供用户响应数据。

**什么时候用？**

*   如果你想相互比较多个模型，那么阴影测试是很棒的，尽管很乏味。
*   影子测试将允许您评估管道、延迟，同时产生结果以及承载能力。

### 2.A/B 测试模型部署策略

A/B 测试是一种基于数据的策略方法。它用于评估两个模型，即 A 和 B，以评估哪一个在受控环境中表现更好。它主要用于电子商务网站和社交媒体平台。通过 A/B 测试，数据科学家可以根据从用户那里收到的数据来评估和选择网站的最佳设计。

这两种型号在功能上略有不同，它们迎合不同的用户群。根据交互和从用户处收到的数据(如反馈)，数据科学家选择一个可以在全球范围内部署到生产中的模型。

#### 方法学

在 A/B 中，两个模型以不同的特征并行设置。目的是提高给定车型的**转化率**。为了做到这一点，数据科学家建立了一个假设。假设是基于对数据的抽象直觉的假设。这个假设是通过实验提出的，如果这个假设通过了测试，它就被接受为事实，这个模型就被接受，否则，它就被拒绝。

##### 假设检验

在 A/B 测试中，有两种假设:

## 

*   1 零假设是指模型中出现的现象纯属偶然，并不是因为某个特征。
*   2 替代假设通过陈述模型中发生的现象是由于某个特征而挑战原假设。

在[假设测试](https://web.archive.org/web/20230119095952/https://www.analyticsvidhya.com/blog/2021/09/hypothesis-testing-in-machine-learning-everything-you-need-to-know/)中，目标是通过设置类似 A/B 测试的实验并向少数用户展示具有特定功能的新模型来拒绝无效假设。新模型本质上是基于另一个假设设计的。如果替代假设被接受而无效假设被拒绝，那么该特征被添加并且新模型被全局部署。

要知道为了拒绝零假设你必须证明 [**统计显著性**](https://web.archive.org/web/20230119095952/https://www.investopedia.com/terms/s/statistical-significance.asp#:~:text=Statistical%20significance%20refers%20to%20the,attributable%20to%20a%20specific%20cause.) 的检验。

![A/B testing model deployment strategy](img/61885ea3b1aa8f20802e456f23390761.png)

*A/B testing model deployment strategy | [Source](https://web.archive.org/web/20230119095952/https://www.oreilly.com/library/view/building-machine-learning/9781492045106/)*

**优点**:

*   很简单。
*   快速产生结果，并有助于消除低性能模型。

**缺点**:

*   如果复杂性增加，模型可能不可靠。在简单假设检验的情况下，应该使用 A/B 检验。

**什么时候用？**

如前所述，A/B 测试主要用于电子商务、社交媒体平台和在线流媒体平台。在这种情况下，如果您有两个模型，您可以使用 A/B 来评估并选择要全局部署的模型。

### 3.多股武装匪徒

多臂土匪或 MAB 是 A/B 测试的高级版本。它也受到强化学习的启发，其思想是探索和利用使奖励函数最大化的环境。

MAB 利用机器学习来探索和利用收到的数据，以优化关键绩效指数(KPI)。使用这种技术的优点是根据两个或更多模型的 KPI 来转移用户流量。产生最佳 KPI 的模型在全球范围内部署。

![Multi Armed Bandit strategy](img/748d5d1d9d7b2b49745d9f0eaeb69d53.png)

*Multi Armed Bandit strategy | [Source](https://web.archive.org/web/20230119095952/https://vwo.com/blog/multi-armed-bandit-algorithm/)*

#### 方法学

MAB 在很大程度上依赖于两个概念:勘探和开发。

**探索:**这是一个概念，模型探索统计上有意义的结果，就像我们在 A/B 测试中看到的那样。A/B 测试的主要焦点是寻找或发现两个模型的转换率。

**利用**:这是一个概念，其中算法使用贪婪的方法，利用它在探索过程中获得的信息来最大化转化率。

与 A/B 测试相比，MAB 非常灵活。它可以在给定时间内与两个以上的模型一起工作，这增加了转化率。该算法根据发出请求的路由的成功情况，连续记录每个模型的 KPI 分数。这允许算法更新其最佳得分。

[![Building machine learning powered application](img/96ad898952dc5bcb33dd32106445a305.png)](https://web.archive.org/web/20230119095952/https://neptune.ai/model-deployment-strategies_8)

*Building machine learning powered application | [Source](https://web.archive.org/web/20230119095952/https://www.oreilly.com/library/view/building-machine-learning/9781492045106/)*

**优点**:

*   随着探索和利用 MAB 提供适应性测试。
*   资源不会像 A/B 测试那样被浪费。
*   更快更有效的测试方法。

**缺点**:

*   这是昂贵的，因为开发需要大量的计算能力，这在经济上是昂贵的。

**什么时候用？**

MAB 对于转化率是你唯一关心的，并且做决定的时间很短的情况非常有用。例如，在有限的时间内优化产品的优惠或折扣。

### 4.蓝绿部署策略

蓝绿色部署策略涉及两个生产环境，而不仅仅是模型。蓝色环境由实时模型组成，而绿色环境由模型的新版本组成。

![Blue-green deployment strategy](img/cf68d3a8df4a8cda5d4289b90192ff7f.png)

*Blue-green deployment strategy | [Source](https://web.archive.org/web/20230119095952/https://www.data4v.com/machine-learning-deployment-strategies/)*

绿色环境被设置为分级环境，即真实环境的精确复制品，但具有新的特征。让我们简单了解一下方法论。

#### 方法学

在蓝绿色部署中，两个相同的环境由相同的数据库、容器、虚拟机、相同的配置等组成。请记住，设置一个环境可能会很昂贵，所以通常情况下，一些组件(如数据库)会在两者之间共享。

包含原始模型的蓝色环境是活动的，并不断为请求提供服务，而绿色环境充当模型新版本的登台环境。它要经过部署和针对真实数据的测试的最后阶段，以确保它性能良好，并准备好部署到生产中。一旦测试成功完成，确保所有的错误和问题都得到纠正，新模型就可以投入使用。

一旦这个模型被激活，流量就从蓝色环境转移到绿色环境。在大多数情况下，蓝色环境充当备份，以防出现问题，请求可以被重新路由到蓝色模型。

**优点:**

*   它确保应用程序全天候可用。
*   回滚很容易，因为一旦出现问题，您可以快速将流量转移到蓝色环境。
*   由于两种环境相互独立，部署风险较小。

**缺点**:

*   由于两种模式都需要独立的环境，因此成本很高。

**什么时候用？**

如果您的应用程序无法承受停机时间，那么应该使用蓝绿色部署策略。

### 5.金丝雀部署策略

canary 部署旨在通过逐渐增加用户数量来部署新版本的模型。与我们看到的之前的策略不同，在之前的策略中，新模型要么对公众隐藏，要么建立一个小的控制组，金丝雀部署策略使用真实的用户来测试新模型。因此，在为所有用户全局部署模型之前，可以检测到错误和问题。

[![Canary deployment strategy](img/d6f0394f84502334a4deed05121106f3.png)](https://web.archive.org/web/20230119095952/https://neptune.ai/model-deployment-strategies_3)

*Canary deployment strategy | [Source](https://web.archive.org/web/20230119095952/https://cloud.google.com/architecture/application-deployment-and-testing-strategies#canary_test_pattern)*

#### 方法学

与 canary deployment 中的其他部署策略类似，新模型与当前的实际模型一起进行测试，但在这里，新模型在几个用户身上进行测试，以检查其可靠性、错误、性能等。

可以根据测试要求增加或减少用户数量。如果模型在测试阶段是成功的，那么模型可以被推出，如果不是，那么它可以在没有停机时间的情况下被回滚，但是只有一些用户将被暴露给新模型。

金丝雀部署策略可以分为三个步骤:

## 

*   1 设计一个新模型，并将一小部分用户请求发送到新模型。
*   检查新模型中的错误、效率、报告和问题，如果发现，则执行回滚。
*   3 在将所有流量路由到新模型之前，重复步骤 1 和 2，直到所有错误和问题都得到解决。

**优点**:

*   与蓝绿色部署相比更便宜。
*   易于根据真实数据测试新模型。
*   零停机时间。
*   在失败的情况下，模型可以很容易地回滚到当前版本。

**缺点:**

*   推出很容易，但速度很慢。
*   由于测试是针对真实数据进行的，用户很少，因此必须进行适当的监控，以便在失败的情况下，用户可以有效地路由到实时版本。

**什么时候用？**

当根据真实世界的实时数据评估模型时，必须使用 Canary 部署策略。此外，它比 A/B 测试更有优势，因为从用户那里收集足够的数据来找到具有统计意义的结果需要很长时间。金丝雀部署可以在几个小时内完成。

### 6.其他模型部署策略和技术

#### 特征标志

特性标志是一种技术，而不是一种策略，它允许开发人员将代码推入或集成到主分支中。这里的想法是保持特性休眠，直到它准备好。这允许开发人员在不同的想法和迭代上合作。一旦特性完成，就可以激活和部署它。

如前所述，特性标志是一种技术，因此它可以与前面提到的任何部署技术结合使用。

#### 滚动部署

滚动部署是一种逐渐更新和替换旧版本模型的策略。这种部署发生在正在运行的实例中，它不涉及阶段化，甚至不涉及私有开发。

上图展示了滚动部署的工作方式。如您所见，服务是水平扩展的，这是关键因素。

左上角的图像代表三个实例。下一步是部署 1.2 版。随着 1.2 版单个实例的部署，1.1 版的一个实例将被淘汰。所有其他实例都遵循相同的趋势，即每当部署新的实例时，旧的实例就被淘汰。

**优点**:

*   它比蓝/绿部署更快，因为没有环境限制。

**缺点**:

*   虽然这样更快，但是如果进一步的更新失败，回滚会很困难。

#### 重新制定战略

重建是一个简单的策略，即关闭模型的活动版本，然后部署新版本。

上图描述了重建策略的工作原理。实际上，旧实例(即 V1 实例)被关闭并丢弃，而新实例(即 V2 实例)被部署。

**优点**:

*   轻松简单的设置。
*   整个环境焕然一新。

**缺点**:

*   对用户有负面影响，因为它会停机并重启。

## 比较:使用哪种模型发布策略？

可以使用各种指标来确定哪种策略最适合他们。但这主要取决于项目的复杂性和资源的可用性。下面的比较表给出了一些何时使用哪种策略的想法。

[![Model Release (Deployment) Strategies](img/e6fe8242a0d6e973b4509dbb0058853a.png)](https://web.archive.org/web/20230119095952/https://neptune.ai/model-deployment-strategies_9)

*Model release (deployment) strategies | [Source](https://web.archive.org/web/20230119095952/https://cloud.google.com/architecture/application-deployment-and-testing-strategies)*

## 关键要点

部署策略通常有助于数据科学家了解他们的模型在给定情况下的表现。一个好的策略取决于产品的类型和目标用户。综上所述，以下是你应该记住的几点:

*   如果你想在真实世界的数据中测试模型，那么必须考虑影子评估策略或类似的策略。与使用用户样本的其他策略不同，影子评估策略使用实时和真实的用户请求。
*   检查任务的复杂性，如果模型需要简单或微小的调整，那么 A/B 测试是可行的。
*   如果有时间限制和更多的想法，那么你应该选择多臂强盗，因为它在这种情况下给你最好的结果。
*   如果你的模型很复杂，在部署之前需要适当的监控，那么蓝绿策略将帮助你分析和监控你的模型。
*   如果您不想停机，并且您可以向公众公开您的模型，那么选择 Canary 部署。
*   当您想要逐步部署模型的新版本时，必须使用滚动部署。

希望你们喜欢阅读这篇文章。如果您想了解更多关于这个主题的内容，可以参考附加的参考资料。继续学习！

### 参考

1.  [使用 Python 进行数据科学的 A/B 测试——数据科学家的必读指南](https://web.archive.org/web/20230119095952/https://www.analyticsvidhya.com/blog/2020/10/ab-testing-data-science/)
2.  [以影子模式部署机器学习模型](https://web.archive.org/web/20230119095952/https://christophergs.com/machine%20learning/2019/03/30/deploying-machine-learning-applications-in-shadow-mode/)
3.  [2021 年的机器学习生命周期](https://web.archive.org/web/20230119095952/https://towardsdatascience.com/the-machine-learning-lifecycle-in-2021-473717c633bc)
4.  机器学习生命周期解释！
5.  [机器学习模型的自动金丝雀释放](https://web.archive.org/web/20230119095952/https://towardsdatascience.com/automatic-canary-releases-for-machine-learning-models-38874a756f87)
6.  [部署策略介绍:蓝绿色、淡黄色等](https://web.archive.org/web/20230119095952/https://harness.io/blog/blue-green-canary-deployment-strategies/)
7.  [部署机器学习模型的策略](https://web.archive.org/web/20230119095952/https://www.alessandroai.com/strategies-to-deploy-your-machine-learning-models/)
8.  [机器学习部署策略](https://web.archive.org/web/20230119095952/https://www.data4v.com/machine-learning-deployment-strategies/)
9.  [什么是蓝绿部署？](https://web.archive.org/web/20230119095952/https://www.opsmx.com/blog/blue-green-deployment/)
10.  [安全推出量产车型](https://web.archive.org/web/20230119095952/https://towardsdatascience.com/safely-rolling-out-ml-models-to-production-13e0b8211a2f)
11.  [最大限度地减少因低性能变化导致的 A/B 测试损失](https://web.archive.org/web/20230119095952/https://vwo.com/blog/multi-armed-bandit-algorithm/)
12.  [部署机器学习模型的终极指南](https://web.archive.org/web/20230119095952/https://mlinproduction.com/deploying-machine-learning-models/)
13.  [机器学习部署:影子模式](https://web.archive.org/web/20230119095952/https://alexgude.com/blog/machine-learning-deployment-shadow-mode/)
14.  [多臂土匪](https://web.archive.org/web/20230119095952/https://www.optimizely.com/optimization-glossary/multi-armed-bandit/)
15.  [顺序 A/B 测试与多臂 Bandit 测试](https://web.archive.org/web/20230119095952/https://splitmetrics.com/blog/sequential-ab-testing-vs-multi-armed-bandit/)
16.  [什么是蓝绿部署？](https://web.archive.org/web/20230119095952/https://semaphoreci.com/blog/blue-green-deployment)
17.  [金丝雀发布的利与弊以及连续交付中的特征标志](https://web.archive.org/web/20230119095952/https://www.split.io/blog/canary-release-feature-flags/)
18.  [滚动部署](https://web.archive.org/web/20230119095952/https://docs.aws.amazon.com/whitepapers/latest/overview-deployment-options/rolling-deployments.html)
19.  [滚动部署:这是什么以及它如何消除软件部署的风险](https://web.archive.org/web/20230119095952/https://www.cloudbees.com/blog/rolling-deployment)
20.  [AWS 中机器学习模型的影子部署](https://web.archive.org/web/20230119095952/https://www.linkedin.com/pulse/shadow-deployments-machine-learning-models-aws-carlos-lara)
21.  [在亚马逊 SageMaker 中部署影子 ML 模型](https://web.archive.org/web/20230119095952/https://aws.amazon.com/blogs/machine-learning/deploy-shadow-ml-models-in-amazon-sagemaker/)
22.  [阴影 AB 测试模式](https://web.archive.org/web/20230119095952/https://mercari.github.io/ml-system-design-pattern/QA-patterns/Shadow-ab-test-pattern/design_en.html)
23.  [MLOps 解释](https://web.archive.org/web/20230119095952/https://www.arrikto.com/mlops-explained/)
24.  [MLOps:它是什么，为什么重要，以及如何实施](/web/20230119095952/https://neptune.ai/blog/mlops)
25.  [用亚马逊 SageMaker MLOps 项目对机器学习模型进行动态 A/B 测试](https://web.archive.org/web/20230119095952/https://aws.amazon.com/blogs/machine-learning/dynamic-a-b-testing-for-machine-learning-models-with-amazon-sagemaker-mlops-projects/)
26.  [多臂土匪对亚马逊评分数据集进行推荐和 A/B 测试](https://web.archive.org/web/20230119095952/https://abhishek-maheshwarappa.medium.com/multi-arm-bandits-for-recommendations-and-a-b-testing-on-amazon-ratings-data-set-9f802f2c4073)
27.  [自动化金丝雀测试，确保持续质量](https://web.archive.org/web/20230119095952/https://www.blazemeter.com/shiftleft/automate-canary-testing-continuous-quality)
28.  [CanaryRelease](https://web.archive.org/web/20230119095952/https://martinfowler.com/bliki/CanaryRelease.html)
29.  [最佳的 Kubernetes 部署策略是什么？](https://web.archive.org/web/20230119095952/https://www.cloudbolt.io/blog/what-is-best-kubernetes-deployment-strategy/)