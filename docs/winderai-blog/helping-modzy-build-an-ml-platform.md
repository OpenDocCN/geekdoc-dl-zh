# 帮助 Modzy 建立一个 ML 平台

> 原文：<https://winder.ai/helping-modzy-build-an-ml-platform/>

云德。AI 与 [Modzy](https://modzy.com) 开发团队和 [MLOps 咨询](https://mlops.consulting)合作，提供了构成 Modzy 产品的各种解决方案，一个 ModelOps 和 MLOps 平台。这项工作的总结包括:

*   开发[开放式模型接口](https://openmodel.ml)
*   开源的[机箱](https://chassis.ml)，这个缺失的环节让数据科学家能够构建健壮的 ML 容器
*   模型监控和可观察性产品特性
*   MLOps 和模型管理产品功能

[https://winder.ai/helping-modzy-build-an-ml-platfoimg/modzy-demo.webm](https://winder.ai/helping-modzy-build-an-ml-platfoimg/modzy-demo.webm)

[https://winder.ai/helping-modzy-build-an-ml-platfoimg/modzy-demo.mp4](https://winder.ai/helping-modzy-build-an-ml-platfoimg/modzy-demo.mp4)

## 问题是:如何建立一个 ML 平台

Modzy 的目标是帮助大型组织协调和管理他们的机器学习(ML)模型。这是 ML 生命周期中的一个阶段，数据科学家通常退居二线，将他们的工作移交给 ML 工程师，ML 工程师负责将模型转化为可部署的工件，然后将这些工件移交给负责服务和监控的运营团队。

大多数公司没有资源或能力来承担如此紧密的角色，所以他们利用平台来完成大部分繁重的工作。Modzy 希望让数据科学家尽可能容易地发布他们自己的模型，这意味着他们需要专注于自动化和简化过程。

## 解决方案:利用绕线机。艾的经验和专长

莫奇问云德。AI 开发一系列 POC，生产 Modzy 产品的原型版本，从模型管理开始，到模型监控。我们交付了直接与他们的后端系统集成的工作微服务。这意味着他们可以演示和提供新的功能，只需要快速证明可行性所需时间的一小部分。

这项工作导致了进一步的开源项目，这些项目利用了我们与 MLOps.community 的紧密联系。Modzy 希望开始为开源工作做出贡献，以提高品牌的知名度。利用我们的 ML 经验，我们帮助定义了当前的 MLOps 问题，并创建了一个有助于缓解这些问题的策略。这项工作产生了两个开源项目，即[开放模型接口](https://openmodel.ml)和[底盘](https://chassis.ml)。

 

Modzy dashboard, courtesy of [Modzy](https://modzy.com).

## 开源合作伙伴

当 Modzy 要求我们在一个开源项目上进行合作时，我们从十年的战略、实施和营销经验中建立起来的社区专业知识受到了极大的考验。第一个问题是定义一个值得花时间去做的项目。

在我们的 ML 和 RL 应用程序工作中，有一个领域我们的东西在 ML 环境中没有得到充分的体现；构建和维护可服务的模型。许多供应商都提出了解决方案，但是在我们的许多工作中，运行在 Kubernetes 集群或类似集群上的简单的云原生容器通常就足够了，尤其是当项目面向内部时。但是数据科学家或从事纯数据工作的人通常没有 Docker 或 Kubernetes 或任何云原生环境的经验。

 

Open source project chassis empowers data scientists with a safe and fast way of building production-ready models.

因此，我们构建了 [chassis](https://chassis.ml) ，这是一个库和服务，允许数据科学家构建健壮、高性能、部署就绪的容器，只需一个命令，他们就可以从笔记本上运行。

作为这项工作的一部分，我们意识到没有很多以 ML 为中心的 API 标准，即使有，也缺少一些元素。所以我们也开源了[开放模型接口](https://openmodel.ml)，这是机箱容器符合的规范。它们快速、安全且可升级。在未来，我们希望将 OMI 兼容的容器部署到一系列云和供应商中成为可能；但是当然你现在就把你的 OMI 模型发布给了 Modzy。

 

The open model interface project aims to unify ML APIs to make it easier to deploy them to a variety of sources.

## 这项工作的价值

利用温德。AI 的经验，Modzy 能够快速原型新产品的想法，并验证他们的早期采用者。我们还为产品的未来方向奠定了基础，无论是技术上还是战略上，都比他们必须等待内部工程时间才能做到的要快得多，也便宜得多。

这种关系发展成了面向外部的开源工作，这为我们多年来建立的信任和跟踪记录提供了一些见解。这项工作有助于巩固 Modzy 在 ML 市场的地位，并建立其开源谱系。

### 接触

如果您的组织对这项工作感兴趣，我们很乐意与您交谈。请与温德公司的销售团队取得联系。我们可以聊聊如何帮助你。