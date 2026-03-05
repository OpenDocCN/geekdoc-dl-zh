# 如何用厚皮动物和希洛建立一个健壮的 ml 工作流程

> 原文：<https://winder.ai/how-to-build-a-robust-ml-workflow-with-pachyderm-and-seldon/>

本文概述了 GitHub 上提供的 [Pachyderm-Seldon Deploy 集成背后的技术设计，并旨在突出演示的显著特点。想要深入了解，请观看 YouTube 上附带的](https://github.com/winderresearch/pachyderm-seldon)[视频](https://www.youtube.com/watch?v=91u2bUUIu9o)。

[https://www.youtube.com/embed/91u2bUUIu9o](https://www.youtube.com/embed/91u2bUUIu9o)

## 介绍

Pachyderm 和 Seldon 运行在 Kubernetes 之上，这是一个可扩展的编排系统；在这里，我解释它们的安装过程，然后我用一个用例来说明如何在一个活动的 ML 部署中操作一个发布、回滚、修复、重新发布周期。在整个演示中，我展示了 Pachyderm 如何在关键场景中方便地使用数据沿袭和自动化。

让我们先来鸟瞰一下主要组件。

### 什么是厚皮动物？

Pachyderm 是数据科学应用程序的数据层，具有内置的版本控制和沿袭。换句话说，它将数据版本控制和管道结合在一起，这样您就可以编排和跟踪复杂的 ML 工作流。通过合并管道，您可以构建一个自动版本化的有向无环图(DAG ),这样您就可以将任何运行追溯到其起点。输入新数据，拉出新训练的模型。数据版本控制是自动的，还有一个仪表板。

你可以在厚皮动物网站上了解更多信息。

 

### 什么很少部署？

Seldon Deploy 是基于开源的 Seldon 核心机器学习引擎的企业产品；它旨在通过仪表板或 API 大规模部署和监控 ML 图形。Python sdk 也是可用的，我在[演示笔记本](https://github.com/winderresearch/pachyderm-seldon)中展示了它的实际应用。

我说的 ML 图是什么意思？嗯，在谢顿，你可以在一个专用资源中组合多个模型。想想金丝雀或 A/B 测试的首次展示，在谢顿这是一个微不足道的设置。最重要的是，你可以部署几乎任何机器学习模型，因为它支持多种 ML 框架，并且由于 Seldon Alibi 模块，你可以不断地审计你的部署行为。更多信息参见[塞尔顿文档](https://deploy.seldon.io/en/latest/)。

 

## 集群设置

要开始，您需要设置一个具有一些先决条件的 Kubernetes 集群。我在知识库中为谷歌 Kubernetes 引擎(GKE) 和 [Minikube](https://github.com/winderresearch/pachyderm-seldon/blob/main/minikube_install/README.md) 提供了一步一步的说明，后者需要高规格的笔记本电脑才能流畅运行:6 核、32Gb 内存的 MacBook Pro 是最低配置。我推荐使用云安装，因此在本文的剩余部分，我将提到这个选项。如果您在一个私有的 GKE 实例上测试这个，那么您还必须添加一个定制的防火墙规则，它包含在安装说明中。

### 很少部署

Seldon Deploy 依赖于一组您必须安装的组件。下面是它们是什么以及为什么在这个设置中需要它们。

*   **Istio** 将入口流量路由至模型部署，并支持灵活的部署策略，如 canary、shadow、A/B 测试等。
*   Knative Serving 是 KFServing 的一个需求，它允许部署大量的 ML 框架(TensorFlow、Scikit-Learn、XGBoost、MLFlow 等)。).
*   **活动事件日志**预测请求以及预测后检测，如异常值、漂移、指标等。
*   **Elastic、Fluentd、Kibana** 从聚合日志并使其可搜索。这个依赖项是 UI 显示入站请求所必需的。
*   **Seldon Core Analytics** 收集监控指标；它是基于格拉法纳和普罗米修斯。
*   Seldon Core 是一个机器学习引擎，它提供了一个完全成熟的 Kubernetes 资源，名为 SeldonDeployment，能够定义灵活的 ML 图。

### 迟钝的人

对于 GKE，我使用 Pachyderm 的内置部署在谷歌云上。它使用云原生存储，因此您只需提供一个专用存储桶，Pachyderm 将使用它来保存您的管道数据。或者，Pachyderm 也支持 Minio(或 S3！)用于对象存储，尽管设置稍微复杂一些。

对于 Minikube，我建议使用简单的本地部署来节省资源。此选项使用磁盘上的本地存储，不适用于多节点群集。

注意，在这个设置中，我创建了两个集群角色绑定。由于 GKE 在默认情况下使用 RBAC，所以我向我的用户帐户授予集群管理特权，以便 Pachyderm deploy 可以成功启动。在生产环境中执行此操作之前，请咨询您的集群管理员。此外，我授予厚皮动物工作者编辑对象的权限，这样他们就可以管理秘密，谢顿将使用这些秘密通过[边车 S3 网关](https://docs.pachyderm.com/latest/deploy-manage/manage/s3gateway/deploy-s3gateway-sidecar/)获取训练有素的 ML 模型。

## 用例

本节探讨了 CreditCo 的案例，这是一家假设的信贷公司，建立了一个 ML 驱动的服务来预测人们的收入，当然它也适用于其他情况。

在实际部署中，您将多次训练和发布您的模型。也许当你的团队正在烘焙一个新的模型时，环境迫使你暂时回滚。只要您有正确的设置，这是非常好的。

我将在本文的剩余部分展示我用 Pachyderm & Seldon 实现的自动化程度和数据血统。对于一个完整的演练，你可能想看看相关的[笔记本](https://github.com/winderresearch/pachyderm-seldon/blob/main/repo/tutorial.ipynb)和[视频演示](https://www.youtube.com/watch?v=91u2bUUIu9o)。

在这个例子中，公司的任务是根据人口统计数据，如教育水平、就业类型、年龄等，预测个人的年收入是高于还是低于 50K。

起初，该公司发布了一个收入预测模型，该模型是在众所周知的[人口普查收入数据集](https://archive.ics.uci.edu/ml/datasets/census+income)上训练的。随着 CreditCo 业务的增长，预测模型变得过时，很快就需要一个新的版本。

为了快速推出新模型，该公司从一家供应商那里获得了一个数据集，但发现这受到了有争议的功能的影响，为了避免丑闻，CreditCo 迅速回滚到第一个模型。

当收集新数据和培训第三个模型时，该公司希望对新的部署保持谨慎，因此他们没有立即投入使用，而是采用了影子部署策略。

在下一节中，我将举例说明 Pachyderm 如何与 Seldon Deploy 集成，从而为这个例子创建一个端到端的 ML 部署。

### 解决办法

这个用例有一个数据入口点，公司会将数据集推送到这个入口点。然后，我将创建一个训练管道来迭代运行一组容器，以便在给定的数据集上训练 ML 模型。应在可访问的位置收集经过训练的模型，对其进行版本控制，并将其传递到部署阶段。这导致了以下管道阶段:

*   数据集存储库
*   培训渠道
*   模型库
*   部署管道
*   ML 部署和监控平台

您可以在[演示库](https://github.com/winderresearch/pachyderm-seldon/blob/main/repo/tutorial.ipynb)中找到这个实现的细节。为了保持简短，我将只说明生成的厚皮 DAG，如下图所示(来自厚皮 DAG)。

 

#### 数据存储库

最上面的存储库(浅蓝色)是我的 CSV 数据集的位置。Pachyderm 自动对我的数据集进行版本控制，类似于 git 对代码回购所做的:每当我推入一个新的 CSV 文件时，它都会生成一个提交散列并指向分支头。

我构建了这个 DAG，以便当一个新的数据集被添加到这个存储库的主分支时，Pachyderm 触发下游的训练管道。请注意，我没有直接将数据推送到 master，而是使用了侧分支来延迟它的处理([延迟处理数据-厚皮动物文档](https://docs.pachyderm.com/latest/concepts/advanced-concepts/deferred_processing/))，直到我希望它被触发。这是一个很好的做法，可以避免计算量很大的意外训练运行。

#### 培训渠道

下游的训练管道(绿色人字纹)监听“income_data”中的变化，并在新数据被推送时立即运行。其结果是一套人工制品，如 Scikit-learn 收入模型、Seldon-Alibi 解释器和其他监测模型。后者是检查和检测实时 ML 部署问题的强大算法，[点击](https://github.com/SeldonIO/alibi)了解更多关于 Seldon-Alibi 的信息。这些管道运行 Docker 容器中提供的定制用户代码[，在这里我使用 python 将相应的模型参数与给定的数据集相匹配。](https://github.com/winderresearch/pachyderm-seldon/tree/main/repo)

#### 部署管道

虽然培训阶段是完全自动化的，但是部署/发布应该有一定程度的人工监督来选择您正在部署的模型版本；毕竟，你希望对面向客户的机器学习模型进行人工监督。

`copy_models`管道是一个微小的层，其目的是收集所有的模型，并使它们在一个位置可用。这样做是为了方便:我现在可以通过访问一个存储库来查看所有可用的工件，而不是查询每一个源。为了最小化这个步骤的开销，这个管道使用了 input.pfs.empty_files 规范，它将文件公开为空，但允许我使用符号链接，这样我可以有效地将模型文件复制到一个位置。[阅读更多关于空文件](https://docs.pachyderm.com/latest/reference/pipeline_spec/)的信息。

为了进行部署，我分别为生产和暂存环境创建了一个管道。同样，我利用数据的[延迟处理](https://docs.pachyderm.com/latest/concepts/advanced-concepts/deferred_processing/)来手动控制何时部署什么环境，以便这些管道被设置为在`copy_models`的特定分支上创建提交时运行。

我想指出的是，这一步使用了一个[厚皮动物服务](https://docs.pachyderm.com/latest/concepts/pipeline-concepts/pipeline/service/)，这是一种特殊的管道，旨在向外部世界公开数据，而不是转换数据。因此，这个服务通过一个[边车 S3 网关](https://docs.pachyderm.com/latest/deploy-manage/manage/s3gateway/deploy-s3gateway-sidecar/)将模型传递给谢顿，并运行一个 Python 脚本来调用谢顿部署 REST api。这确保了与任何 S3 兼容的 ML 平台以及数据来源的兼容性:对上游数据的任何更改都可以追溯到任何给定的作业/输入提交。谈到起源，我将`copy_models`提交散列传递给部署脚本，以便它将它注入模型容器。这样，我可以随时检查部署了什么模型版本，并与模型回购历史进行交叉检查。

#### 很少部署

Seldon 将创建一个[selden deployment](https://docs.seldon.io/projects/seldon-core/en/latest/reference/seldon-deployment.html)，这是一个 Kubernetes 资源，能够表示由多个预测器和相关监控模型组成的复杂 ML 图。从现在开始，您可以通过 Seldon Deploy endpoint 查询 ML 模型，并通过 UI 监督实时部署。下面的屏幕截图显示了一个异常值检测器，它被用来标记非常规的传入请求。

 

## 摘要

在本文中，我描述了如何集成 Pachyderm 和 Seldon Deploy 来实现端到端的 ML 管道。在详细介绍了集群设置及其依赖关系之后，我给出了一个信贷公司在生产中使用 ML 的示例用例，并解释了 Pachyderm 和 Seldon 如何方便地管理复杂的现场情况。

要深入了解这个演示，请查看 GitHub 上的[演示笔记本，或者在 YouTube](https://github.com/winderresearch/pachyderm-seldon/blob/main/repo/tutorial.ipynb) 上观看[的附带视频。](https://www.youtube.com/watch?v=91u2bUUIu9o)

当然，温德。AI 已经准备好通过结合 [MLOps 咨询](https://winder.ai/services/mlops/mlops-consulting/)和 [ML 专业知识](https://winder.ai/services/machine-learning/)来帮助你改善你的 ML 工作流程。

[Talk to Sales](https://winder.ai/about/contact/)

## 参考

*   [Github 库](https://github.com/winderresearch/pachyderm-seldon)
*   [演示视频](https://www.youtube.com/watch?v=91u2bUUIu9o)

## 信用

这个项目是由厚皮动物资助的。