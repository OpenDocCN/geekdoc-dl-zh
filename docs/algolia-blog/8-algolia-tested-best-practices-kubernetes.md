# Kubernetes 最佳实践-来自过去的爆炸(radius )

> 原文：<https://www.algolia.com/blog/engineering/8-algolia-tested-best-practices-kubernetes/>

我们大约在四年前开始使用 Kubernetes。我们有新的服务要部署，即使我们是裸机的大用户，我们也需要更多的灵活性。因此，我们决定在新系统上测试和使用 Kubernetes。两年后，我们的大多数产品都部署在 Kubernetes 上，遵循 Kubernetes 的最佳实践。随着越来越多的团队开始在内部使用它，我们创建了一个内部培训。今天，我们很自豪地[将这个培训开源](https://github.com/algolia/kubernetes-hands-on)，这样任何人都可以从中学习并做出贡献。

实施两年后，我们从培训中提取了八个实践，我们认为这八个实践是正确使用 Kubernetes 的关键。我们重新发布这些 Kubernetes 最佳实践，作为过去的一个亮点，并为未来关于我们和 Kubernetes 在过去两年中如何发展的文章奠定基础。

## [](#1-do-not-use-root-user-in-your-containers)1。不要在容器中使用 root 用户

容器范式，以及它在 Linux 上的实现方式，并没有考虑到安全性。它的存在只是为了限制资源，比如 CPU 和 RAM，就像 Docker [的文档解释](https://docs.docker.com/engine/security/security/)一样。这意味着你的容器不应该使用“根”用户来运行命令。在容器中运行程序与在主机上运行程序几乎是一样的。如果你有兴趣了解更多，[查看这篇文章](https://medium.com/@mccode/processes-in-containers-should-not-run-as-root-2feae3f0df3b)了解原因。

因此，在所有图像上添加这些行，使您的应用程序由专门的用户运行。将“appuser”替换为与您更相关的名称。

```
ARG USER=appuser # set ${USER} to be appuser
addgroup -S ${USER} && adduser -S ${USER} -G ${USER} # adds a group and a user of it
USER ${USER} # set the user of the container
WORKDIR /home/${USER} # set the workdir to be the home directory of the user

```

这也可以通过 [pod 安全策略](https://kubernetes.io/docs/concepts/policy/pod-security-policy/)在集群级别得到保证。

## 2。处理“SIGTERM”信号

每当 Kubernetes 想要优雅地停止一个容器时，它就发送“SIGTERM”信号。您应该监听它，并在应用程序中做出相应的反应(通过关闭连接、保存状态等)。)一般来说，遵循[十二因素应用](https://12factor.net/)建议被认为是良好的实践。另外，不要忘记在您的 pod 上配置 [`terminationGracePeriodSeconds`](https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/#hook-handler-execution) 。默认值是 30 秒，但是您的应用程序可能需要更多(或更少)的时间来正确终止。

## [](#3-use-a-declarative-management-for-your-manifests)3。对您的清单使用声明式管理

使用声明性清单，这样可以有效地回滚代码和基础结构。这意味着您的源代码版本应该是您的清单的真实来源。

这意味着您只使用`kubectl apply`来更新或创建您的 Kubernetes 资源，但也意味着您不使用`latest`标签作为您的图像容器。容器的每个版本都应该是独一无二的，使用 Git 散列是一个很好的实践。当部署应用程序的新版本时，应该通过为容器指定新版本来更新清单，然后在源代码控制中提交清单，最后运行`kubectl apply`。

## [](#4-lint-your-manifests)4。Lint 你的清单

YAML 是一个[棘手的格式](https://docs.saltstack.com/en/latest/topics/troubleshooting/yaml_idiosyncrasies.html)。我们使用 [yamllint](https://github.com/adrienverge/yamllint) ，因为它支持单个文件中的多文档。

你也可以使用 Kubernetes-specifics 棉绒:

*   [kube-score](https://github.com/zegl/kube-score) 检查您的货物清单，并强制执行良好做法。
*   kubeval 也检查清单，但是只检查有效性。

在 Kubernetes 1.13 中， [`--dry-run`选项出现在`kubectl`](https://kubernetes.io/blog/2019/01/14/apiserver-dry-run-and-kubectl-diff/) 上，它让 Kubernetes 检查您的清单而不应用它们。您可以使用此功能来检查您的 YAML 文件对于 Kubernetes 是否有效。

## [](#5-configure-the-liveness-and-readiness-probes)5。配置活动和就绪探测器

活性和就绪性是应用程序向 Kubernetes 传达其健康状况的方式。配置这两者有助于 Kubernetes 正确处理您的 pod，并对状态变化做出相应的反应。

活性探测器在这里评估容器是否仍然是活性的；也就是说，如果容器没有处于中断状态、死锁或任何类似的状态。从那里，它可以作出决定，如重新启动它。

准备就绪探测器在这里检测容器是否准备好接受流量、阻止首次展示、影响 Pod 中断预算(PDB)等。当您的容器被 Kubernetes 设置为接收外部流量时(大多数情况下，当它是一个 API 时)，它特别有用。

通常，具有相同的就绪性和活性探针是可以接受的。但是在某些情况下，您可能希望它们有所不同。一个很好的例子是运行接受 HTTP 调用的单线程应用程序(如 PHP)的容器。假设您有一个需要很长时间处理的请求。您的应用程序不能接收任何其他请求，因为它被传入的请求阻塞了；因此它还没有“准备好”。另一方面，它正在处理一个请求，因此它是“活动的”。

另一件要记住的事情是，您的探测器不应该调用您的应用程序的依赖服务。这可以防止级联故障。

## 6。配置资源请求和限制

Kubernetes 允许您配置 pods 资源的“请求”和“限制”(CPU、RAM 和磁盘)。配置“请求”有助于 Kubernetes 更容易地调度您的 pod，并更好地在您的节点上打包工作负载。

大多数时候你可以定义`"request" = "limit"`。但是要小心，因为你的吊舱将被终止，如果它超过了`limit`。

除非您的应用程序被设计为使用多个内核，否则最佳实践通常是将 CPU 请求保持在`"1"`或更低。

## [](#7-specify-pod-anti-affinity)7。指定 pod 反关联性

当您部署一个具有大量副本的应用程序时，您很可能希望它们均匀地分布在 Kubernetes 集群的所有节点上。如果你所有的豆荚都在同一个节点上运行，而这个节点死了，这将杀死你所有的豆荚。为您的部署指定一个 pod 反关联性可以确保 Kubernetes 跨所有节点调度您的 pod。

一个好的做法是在节点的主机名上指定一个`podAntiAffinity`:

```
apiVersion: apps/v1
kind: Deployment
metadata:
 name: my-application
spec:
 replicas: 2
 selector:
   matchLabels:
     app: my-application
 template:
   metadata:
     labels:
       app: my-application
   spec:
     containers:
     - name: my-pod
       image: my-image:my-version
     affinity:
       podAntiAffinity:
         preferredDuringSchedulingIgnoredDuringExecution:
           - labelSelector:
               matchExpressions:
                 - key: app
                   operator: In
                   values:
                     - app: my-deployment
             topologyKey: kubernetes.io/hostname

```

这里我们有一个带有两个副本的部署“我的应用程序”，我们指定了一个带有软需求的`podAntiAffinity`规范(`preferredDuringSchedulingIgnoredDuringExecution`，更多细节请参见这里的)，所以我们没有在相同的主机名(`topologyKey: kubernetes.io/hostname`)上调度 pod。

## [](#8-specify-a-pod-disruption-budget-pdb)8。指定 Pod 中断预算(PDB)

在 Kubernetes，豆荚有一个有限的寿命，可以随时终止。这种现象被称为[【颠覆】](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/)。

中断可能是自愿的，也可能是非自愿的。顾名思义，非自愿中断是指任何人都无法预料到的事情(例如硬件故障)。主动中断是由某人或某事发起的，如节点升级、新部署等。

定义“Pod 中断预算”有助于 Kubernetes 在发生自愿中断时管理您的 Pod。Kubernetes 将努力确保与给定选择器匹配的足够多的资源同时保持可用。指定一个 [PDB](https://kubernetes.io/docs/tasks/run-application/configure-pdb/) 可以提高服务的可用性。

## [](#conclusion)结论

四年前，我们使用这些优秀的默认设置，并将它们应用到我们在 Kubernetes 的所有应用程序中。我们建议您根据应用程序和工作负载的具体情况调整实践。

您可以在培训的[专门部分找到关于这些良好实践的更多详细信息。](https://github.com/algolia/kubernetes-hands-on/tree/master/99-good-practices)