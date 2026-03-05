# 构建云原生 PaaS

> 原文：<https://winder.ai/bulding-a-cloud-native-paas/>

# 行动纲要

云德。AI 与其合作伙伴 [Container Solutions](https://container-solutions.com) 合作，交付 Weave 云平台即服务(PaaS)的核心组件。

*   Kubernetes 和 Terraform 在 Google 云平台上的实现
*   提供了关键的计费组件来跟踪每秒的使用情况并进行计费
*   帮助发起、设计和交付 Weave Flux，这是一个 Git-Ops CI/CD 推动者

# 客户

Weaveworks 使开发人员和 DevOps 团队能够快速简单地构建和操作强大的容器化应用程序。它们通过提供自动化的连续交付管道、可观察性和监控，最大限度地降低了 Kubernetes 中运营工作负载的复杂性。Weaveworks 还参与了几个开源项目，包括 Weave Scope、Weave Cortex 和 Weave Flux。它是云计算原生计算论坛的首批成员之一。该公司成立于 2014 年，由谷歌风投和 Accel Partners 提供支持。欲了解更多信息，请访问 [www.weave.works](https://www.weave.works) 。

> 我们希望加快我们在 Weave Cloud 上的工作，特别是在我们针对 kubernetes 客户的 CICD 项目中。由于您的帮助，Weaveworks 能够更快地将 Weave Cloud 推向市场。您的云原生专业知识和公开演讲改善了我们的业务。时间和金钱用得其所。没抓到你。
> 
> ***Alexis Richardson**-Weaveworks 有限公司首席执行官*

# 问题

Weaveworks 获得了资金，需要快速上市。产品是功能性的，但是需要检测、跟踪和计费的能力。此外，为了获得市场牵引力，他们还需要加强其产品组合。

最大限度地缩短上市时间势在必行。Weaveworks 意识到有一个显著的先发优势。他们需要一家为快速部署而优化的多功能公司。

云德。选择 AI 来执行这项工作是因为我们可论证的[面向数据的](https://winder.ai/services/data-science/) [云原生的专业知识](https://winder.ai/services/)。

# 解决办法

对灵活性和快速开发的需求意味着[云原生技术](https://winder.ai/what-is-the-cloud/)非常适合他们的问题。使用 Kubernetes orchestrator，我们开发了一系列基于 Docker 的微服务，以提供一系列以用户为中心的功能。最重要的是在 S3 存储使用模式并将其转换为计费时间的计费原型。

和温德的一队人一起。AI 的合作伙伴 [Container Solutions](https://container-solutions.com) ，我们选定了由谷歌云平台管理的服务和定制微服务的组合。采用利益驱动的方法来确保开发时间和运行时间成本都得到优化。

在整个过程中，我们嵌入了客户工程师和产品管理团队来传递知识和经验。整个实施过程花费了大约六个月的时间。

# 结果

仅在两个月内，Weaveworks 就有了一个可以集成到产品中的计费系统。经过多次迭代后，他们最终得到了一个完全优化的系统，该系统可以根据他们的业务规模进行工作。

他们立即开始向客户收费(此前客户一直使用免费服务),并开始盈利。无限的利润。

账单到位后，温德。AI 的云原生专业知识随后被要求帮助设计和实现他们的核心产品之一 Flux。Flux 使团队能够使用一个叫做 Git-Ops 的过程进行操作。Git-Ops 通过强制要求所有操作和应用软件必须是代码，为云原生开发增加了一个迫切需要的过程。Flux 自动将 git 状态的变化应用到生产软件中，从而实现端到端的连续部署。