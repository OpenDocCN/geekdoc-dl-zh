# 在生产中不应该使用头盔的 7 个理由

> 原文：<https://winder.ai/7-reasons-why-you-shouldnt-use-helm-in-production/>

Helm 被宣传为“ [Kubernetes](https://kubernetes.io/) 的包装经理”。目标是为 Kubernetes 提供类似于高级包管理的体验。这是所有主要集装箱化平台的目标。例如，Apache Mesos 有 [Mesos 框架](http://mesos.apache.org/documentation/latest/frameworks/)。并且给出了操作系统级的包管理的标准化(yum，apt-get，brew，choco 等。)和应用层(npm、pip、gem 等)。)，这完全说得通吧？

也许不是。

**过期警告**

这个帖子已经过时了。头盔 3 使下面的许多问题无效。我现在推荐使用头盔。云德。AI 在我们的 [MLOps 咨询](https://winder.ai/services/mlops/mlops-consulting/)中每天都在使用它。

## 价值主张

首先，让我们考虑一下 Helm 的价值主张。在思考这个问题的时候，我是从一个工程师的角度来说的，这个工程师主要负责私人客户的操作部署(即不是开源软件供应商)。它允许我们:

1.  安装应用程序
2.  管理这些应用程序的生命周期
3.  通过模板定制应用程序

我们已经可以通过 k8s 清单实现 **1** 了。 **2** 比较棘手，因为有很多 k8s 组件不符合标准的“应用”生命周期。例如 RBAC、PVC、名称空间、资源配额等。而对于 **3** ，我们可以用各种方式进行模板化，大部分都要简单得多。

Helm 还鼓励(无意中)动态的手动模板化。记住，我们的目标是把所有的东西都写成代码(这里我就不赘述了)。手动模板阻止了你对你的系统应该是什么样的有一个静态的，版本控制的声明。这会影响您测试、恢复和确保测试和生产环境等效的能力。

## 头盔的问题

因此，我认为标准 k8s 体现的价值主张已经站不住脚了。总结一下，以下是 Helm 可能不是一个好选择的七个原因:

1.  Tiller 默认在`configmaps`里面存储应用机密(即明文)。[有可能覆盖使用 k8s 秘笈，但还在 beta 中。](https://github.com/helm/helm/blob/master/docs/install.md#storage-backends)。

2.  RBAC 政策是针对每一个舵柄，而不是每一个用户。例如，任何可以访问 tiller 的受限用户都可以访问 tiller 可以访问的所有内容。所以这意味着你需要为每个角色/团队等安装一个单独的头盔。这大大增加了复杂性。这里的见[，这里的](https://engineering.bitnami.com/articles/helm-security.html)见[。](https://dzone.com/articles/securing-helm)

3.  默认情况下[分蘖豆荚可以被集群中的其他豆荚访问](https://engineering.bitnami.com/articles/helm-security.html)

4.  Helm 只在你安装社区组件时增加价值。否则无论如何你都需要写 yaml。

5.  导致模板中有太多的逻辑(对没有经验的 k8s 用户来说不好)

6.  违反了 gitop/infra structure 作为代码的最佳实践，因为您在之前对*进行了版本控制，它已经被模板化了。所以你不能得到真正的可重复构建。例如，考虑测试和生产环境的模板化。单独的模板可能会在两个环境之间产生差异(糟糕！).*

7.  如果您真的需要模板(例如，因为您支持多个集群)，那么考虑 kustomize 或其他模板工具。你得到了模板的好处，但是你保留了 GitOps 的好处。

请注意，这些都不妨碍利用头盔。您仍然可以使用 helm 作为静态生成器生成 yaml 清单。

还有，这忽略了赫尔姆擅长什么。如果您是一个公共项目的开发人员，并且希望您的用户能够轻松安装您的 k8s 应用程序，那么一个简单的 helm package 命令非常有吸引力。但是，还有其他方法可以为您的用户提供 oneliner。你可以从一个托管的 url(比如 [Docker](https://get.docker.com/) )或者一个单一连接清单的 oneliner(比如 [weave scope](https://www.weave.works/docs/scope/latest/installing/#kubernetes) )运行一个脚本。

## 进一步阅读

您可能感兴趣的其他资源:

*   使用头盔前请三思
*   [将现有舵柄设置轻松移植到 Kubernetes Secrets](https://dev.to/evilmartians/painless-migration-of-existing-helms-tiller-setup-to-kubernetes-secrets-d1p)