# OSX Minikube 上的本地 Jenkins 开发环境

> 原文：<https://winder.ai/local-jenkins-development-environment-on-minikube-on-osx/>

开发 Jenkinsfile 管道很难。我想我的世界纪录是尝试获取一个有效的 Jenkinsfile 的次数大约是 20 次。当您不得不在托管的 Jenkins 实例上不断推进和运行您的管道时，反馈周期会很长。开发人员生产力的主要瓶颈是反馈周期的长度。

所以我想部署一个本地版本的 Jenkins 来帮助开发。这将运行在你的笔记本电脑上的 Minikube，但就像一个管理版本。唯一的缺点是 Minikube 需要一段时间来设置和下载所有的容器。我建议只在你有任务要显著改变一个 Jenkinsfile(比如一个新的或者一个重构)的时候才使用这个设置。对于简单的更改，使用您的托管版本可能是最简单的。

## 先决条件

1.  安装 minikube。在 OSX 上做到这一点最简单的方法是通过`Preferences->Kubernetes`部分启用 Kubernetes。一个`Kubernetes failed to start`的错误通常只是一个超时。`kubectl get nodes`大概会管用。右击 Docker 菜单，然后按`Kubernetes->Disable local cluster`。然后转到首选项，再次重复启用。

2.  [安装舵](https://helm.sh/docs/using_helm/#install-helm)

## 舵安装

1.  确保当前的 kubeconfig 上下文指向您的本地服务器(不是生产服务器！)
2.  `helm init`

## 詹金斯构型

用下面的内容创建一个名为`values.yaml`的文件。

```py
Master:   ImageTag:  "lts"   ServiceType:  NodePort   InstallPlugins:   - kubernetes:1.14.0   - workflow-job:2.31   - workflow-aggregator:2.6   - credentials-binding:1.17   - git:3.9.1   - filesystem_scm:2.1  Agent:   Enabled:  true   volumes:   - type:  HostPath   hostPath:  /Users   mountPath:  /Users   - type:  HostPath   hostPath:  /var/run/docker.sock   mountPath:  /var/run/docker.sock  rbac:   install:  true  Persistence:   volumes:   - name:  source-code   hostPath:   path:  /Users   mounts:   - mountPath:  /Users   name:  source-code 
```

### 重要设置

值文件中有一些重要的设置。首先是已安装插件的列表。最重要的一个是`filesystem_scm`插件，它允许你从磁盘而不是存储库中读取数据。如果你需要的话，可以随意添加更多插件。

接下来是卷。您需要确保将源代码目录挂载到容器中。请注意，如果您使用的是 docker for mac，那么这些目录将通过 docker 应用程序共享。转到`Preferences->File Sharing`。这些是唯一可以共享的目录。默认情况下,`/Users`是共享的，所以如果您的主目录中有您的代码，您就可以开始了。

注意，您还需要在`Persistence`部分安装相同的路径。一个恼人的名字，但这是安装到詹金斯硕士。主服务器也需要这些文件，这样它就可以读取`Jenkinsfile`(如果你有的话)。

## 詹金斯装置

`helm install --name jenkins --values values.yaml stable/jenkins`

詹金斯集装箱的后续拉动和詹金斯的启动可能需要一段时间。用`kubectl get pods`看进度。你可以通过给 Docker 更多的 CPU/RAM 来加速这个过程。

如果 pod 启动失败，开始调试原因。检查磁盘使用情况。检查 RAM 使用情况。如果您得到一个模糊的错误消息`The node was low on resource: imagefs.`，很可能您需要将 Docker CPU 增加到 4，将 RAM 增加到 6GB，将 Swap 增加到 2GB。你的笔记本电脑可能成为一个移动加热设备。最坏的情况是，将 docker 重置为出厂设置，然后重试。

## 接近詹金斯

一旦一切都启动并运行，那么您应该能够浏览到运行机器的 IP 地址上的节点端口。对于使用 docker for mac 的用户，您可以通过以下方式获得该地址:

```py
open "http://localhost:$(kubectl get --namespace default -o jsonpath="{.spec.ports[0].nodePort}" services jenkins)" 
```

用户名为`admin`，密码在`helm status jenkins`输出中描述:

```py
printf $(kubectl get secret --namespace default jenkins -o jsonpath="{.data.jenkins-admin-password}" | base64 --decode);echo 
```

## 添加作业和凭据

添加一个使用文件系统 SCM 插件的定制管道作业，并将路径设置为保存代码的目录(`/Users/...`)。还要添加您可能需要的任何凭据。

## 运行 Jenkinsfile 声明性管道

Kubernetes 管道选项中有一个关键选项必须设置。这是`inheritFrom 'default'`。例如:

```py
pipeline {
  agent {
    kubernetes {
      defaultContainer 'jnlp'
      inheritFrom 'default'
      yaml """
---
apiVersion: v1
kind: Pod 
```

这是必要的，这样定制容器也可以访问`jnlp`容器正在使用的主机卷挂载(`/Users`)。点击查看继承文件[。](https://github.com/jenkinsci/kubernetes-plugin#pod-template-inheritance)

## 信用

这是由于@garunvagidov [here](https://medium.com/@garunski/local-development-with-kubernetes-and-jenkins-49cfb826ef65) 的工作而得到的强烈启发。