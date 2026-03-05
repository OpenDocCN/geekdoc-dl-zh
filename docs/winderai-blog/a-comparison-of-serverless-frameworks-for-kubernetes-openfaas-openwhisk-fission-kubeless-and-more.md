# Kubernetes 的无服务器框架比较:OpenFaas、OpenWhisk、裂变、Kubeless 等等

> 原文：<https://winder.ai/a-comparison-of-serverless-frameworks-for-kubernetes-openfaas-openwhisk-fission-kubeless-and-more/>

术语无服务器已经成为 AWS Lambda 的同义词。与 AWS 解耦有两个好处；它避免了锁定并提高了灵活性。

“无服务器”这个用词不当，它是一套将底层硬件完全抽象化的技巧和技术。显然，这些功能仍然在某个地方的“服务器”上运行，但关键是我们不在乎。开发者只需要提供代码作为功能。然后通过 API 使用或消费函数，通常是 REST，但也通过消息总线技术(Kafka、Kinesis、Nats、SQS 等)。).

这为 Kubernetes 平台的无服务器框架提供了比较和建议。

编辑:

*   2018 年 9 月 2 日 v1
*   v2 02/09/18:感谢 [@alexellisuk](https://twitter.com/alexellisuk) 、Richard Gee、 [@kenfdev](https://twitter.com/kenfdev) 对 OpenFaas 信息的重大编辑。感谢 [@sebgoa](https://twitter.com/sebgoa) 将无服务器从对照表中移除。
*   v3 10/09/18:关于 Fn 项目的编辑感谢 [@delabassee](https://twitter.com/delabassee%E2%80%8F) 。

无服务器专家

Winder 是云基础设施开发和运营方面的专家。我们所有的客户都依赖我们的专业知识来交付数据驱动的云原生产品。[联系我们获得我们的帮助](https://winder.ai/about/contact/)。

## 对照表

下表提供了 k8s 的不同无服务器框架的高级比较。它们根据流行度、稳定性、工具、技术和可用性进行分类。

像流行这样的观点的问题在于它很难量化。我们不得不求助于代理 metic。例如，我真的很感兴趣有多少人每天都在使用这个框架。但很难找到这些数字，所以我们使用 Github stars 或谷歌搜索等代理指标。这些被表示为`<desired_metric (proxy_metric)>`。关于项目知名度的更多信息见[。](https://opensource.guide/metrics/)

因此，请注意，这些指标中有许多都是粗略的估计。单个框架可能看起来比实际情况更好或更差。阅读时要对所有的证据和特征有一个整体的看法。

| 比较/框架 | [OpenFaas](https://github.com/openfaas/faas) | [open whish](https://github.com/apache/incubator-openwhisk) | [kub less](https://github.com/kubeless/kubeless) | [裂变](https://github.com/fission/fission) | [铁功能](https://github.com/iron-io/functions) | [联合国](https://github.com/fnproject/fn) |
| --- | --- | --- | --- | --- | --- | --- |
| 受欢迎程度(Github 明星) | ![](img/bb0963e9b3ac6bdd5e4faada7efdfa81.png) | ![](img/ceeecde481043f4f2842143b616848a2.png) | ![](img/5f408278a5e3638e0bc9f9a1f838d7e1.png) | ![](img/3b11e981c93136a9c4a5b5aebed4c41c.png) | ![](img/3efc2f45f925a4d3ff2915f5dbc0492b.png) | ![](img/84a9cb9341c322af8e0c021be24cfa9e.png) |
| 受欢迎程度(相对谷歌趋势- 100 表示最受欢迎) | Thirty-seven | Fifty-eight | Twenty-four | 不适用(与裂变冲突) | 不适用(与无服务器冲突) | three |
| 受欢迎程度(StackOverflow.com 帖子总数) | Twenty-one | Three hundred and fifty-nine | Fifteen | Two | five | nine |
| 稳定性(贡献者) | ![](img/7ec1088f615e7fc641cf9f74b1a604d2.png) | ![](img/29e79b67e85e743ced3822bf55f75cee.png) | ![](img/250dde1257b00135c710b736bc4c6b96.png) | ![](img/2120f28f04cc8e389c9b1c93b6d78037.png) | ![](img/e2e1a38968f0d7bfac49c1e7b6fcc9e0.png) | ![](img/66ff69f953379c5fd79dbb04205e47b8.png) |
| 稳定性(贡献者> 10 次提交) | Ten | Thirty-three | seven | six | nine | Nineteen |
| 稳定性(企业支持者) | [VMWare](https://www.vmware.com/) (1) | IBM 公司(阿帕奇基金会)(3) | [Bitnami](https://bitnami.com/) | [平台 9](https://platform9.com/) | [https://iron.io](https://iron.io) | [甲骨文](https://www.oracle.com/) |
| 稳定性(已启动) | 2016 年 12 月 | 2016 年 2 月 | 2016 年 11 月 | 2016 年 8 月 | 2016 年 2 月 | 2016 年 5 月 |
| 稳定性(开发语言) | 去 | 斯卡拉 | 去 | 去 | 去 | 去 |
| 工具(包装机制) | 码头集装箱 | 码头集装箱 | 码头集装箱 | 码头集装箱 | 码头工人 | 码头工人 |
| 工具(k8s 上的部署功能) | 带有自定义 yaml 的清单 | 带有自定义 yaml 的清单 | 显示 | 带有自定义 yaml 的清单 | 所有人 | 所有人 |
| 工具(通过无服务器部署？) | 是([在制品](https://github.com/openfaas/serverless-openfaas)) | [是](https://serverless.com/framework/docs/) | [是](https://serverless.com/framework/docs/) | 不 | [否](https://github.com/iron-io/functions/issues/509) | [是](https://serverless.com/framework/docs/) |
| 技术(基础技术) | Alertmanager /普罗米修斯，Nats | -伊甸园字幕组=-翻译:食梦貘 ts 校对:葱家的小毛毛时间轴:邦德猪 | 无(可选纳茨或卡夫卡) | fluentd (optional Nats) | 邮局，邮局 | DB (sqlite3、PostgreSQL、MySQL)、MQ (Bolt、Redis)、Prometheus |
| 可用性(开箱即用？) | 是 | 是 | [否](https://github.com/serverless/serverless-kubeless/issues/149)(无服务器插件失败) | 是的(但是[烦恼](https://github.com/fission/fission/issues/881) | 没试过 | 没试过 |
| 可用性(文档质量) | [好的](https://docs.openfaas.com/) | [好的](https://openwhisk.apache.org/documentation.html) | [一般](https://kubeless.io/docs/)(组织不良) | [糟糕的](https://docs.fission.io/0.10.0/)(没有组织好，奇怪的导航和丢失的文件) | [一般](https://github.com/iron-io/functions/tree/master/docs)(组织不良) | [好的](https://github.com/fnproject/docs/blob/master/fn/README.md#for-operators) |
| 可用性(松弛通道？) | [是，但通过电子邮件](https://docs.openfaas.com/community/#slack-workspace) | [是](https://openwhisk.apache.org/slack.html) | 是(k8s slack 的无库频道: [http://slack.k8s.io](http://slack.k8s.io/) ) | [是](http://slack.fission.io/) | [是](http://get.iron.io/functions-slack) | [是](http://slack.fnproject.io/) |

1.  VMWare 的开发团队全职开发 OpenFaas。
2.  根据 SDK 的建议，包括无服务器。
3.  在 IBM 的云功能产品中使用

## 无服务器框架建议

使用此表作为对比，我可以推荐:

*   使用[无服务器框架](https://github.com/serverless/serverless)作为软件开发工具包(SDK)
*   使用 [OpenFaas](https://github.com/openfaas/faas) 或 [OpenWhisk](https://github.com/apache/incubator-openwhisk) 作为 k8s 上函数的编排器。
*   OpenFaas 是成熟的，易于使用和可扩展的，但与 OpenWhisk 相比，它在核心项目上的活跃开发人员较少(根据我对活跃开发人员的定义)，也不太受欢迎(根据我对流行度指标的选择)。
*   OpenWhisk 很成熟，有很多活跃的开发者支持，很流行，但是很复杂，用 Scala 编写，有 IBM/Apache 支持(可能是好事也可能是坏事，取决于你的看法)

最终的技术堆栈如下所示:

![Recommended open source serverless stack](img/5c69cf15d1afa6de01e8b3443f49d7e8.png)

无服务器框架的额外建议允许开发者选择是部署到 lambda 还是部署到 k8s 上的无服务器平台。如果 lambda 上已经有函数，这有助于迁移。

## 框架注释

### [open whish](http://openwhisk.incubator.apache.org/)

OpenWhisk 是一个成熟的无服务器框架，由 Apache foundation 和 IBM 提供支持。OpenWhisk 是 IBM 云功能服务的基础。主要提交者是 IBM 员工。

有许多底层组件，这增加了复杂性。它利用了 CouchDB、Kafka、Nginx、Redis 和 Zookeeper。有利的一面是，开发人员已经明确关注可伸缩性和弹性。缺点是开发者和操作者需要这些工具的工作知识。另一个缺点是，它们复制了 Kubernetes 等 orchestrator 中的功能(例如自动缩放)。功能最终被捆绑到 Docker 容器中，与框架一起运行。

OpenWhisk 可以使用舵图安装，但不幸的是需要一些手动干预。可以使用 CLI 或无服务器框架部署应用程序。普罗米修斯指标是开箱即用的。

### [OpenFaas](https://docs.openfaas.com/)

OpenFaas 是一个流行的(根据我对流行度的估计，不如 OpenWhisk 流行)，易于使用的无服务器框架。它不像 OpenWhisk 那样流行，提交是基于个人的。除了个人贡献者在业余时间所做的大量工作之外，VMWare 还雇佣了一个团队全职开发 OpenFaas。一家名为 OpenFaas Ltd .的公司已在英国注册成立，但不清楚该公司与该项目的关系。

OpenFaas 的架构相对简单。API 网关可以通过 Kafka、SNS、CloudEvents、CRON 等触发器同步或异步调用。异步调用由 NATS 流处理。使用 Prometheus 和 Prometheus Alertmanager 执行自动缩放；但谢天谢地，这个[似乎可以换成](https://stefanprodan.com/2018/kubernetes-scaleway-baremetal-arm-terraform-installer/#horizontal-pod-autoscaling)来使用 Kubernetes 的`HorizontalPodAutoscaler`。

完全支持 Kubernetes 安装程序，可通过 Helm 或 kubectl 获得，包括允许使用 CRDs 的操作器，即`kubectl get functions`。还有一个 [Kubernetes 操作符是 WIP](https://github.com/openfaas-incubator/openfaas-operator) ，但是我发现这个很好用。

可以使用 CLI 或无服务器框架(WIP)部署应用程序。还提供了一个[“函数存储”](https://blog.alexellis.io/announcing-function-store/)，这是一个与 OpenFaas 一起使用的精选函数列表。普罗米修斯指标是开箱即用的。

openfans

### [kub less](https://kubeless.io/)

我对 Kubernetes 的功能原生框架 Kubeless 感到非常兴奋。它的工作原理是在 Kubernetes 中添加一个“函数”作为自定义资源定义(CRD)。加上一些聪明的代码，这意味着它把 Kubernetes 变成了一个功能机器，没有像其他框架那样的消息总线那样的附加复杂性。

我喜欢管理像标准 Kubernetes 对象这样的功能，这意味着所有常见的 Kubernetes 好东西都可以开箱即用(头盔、方舟等)。).

交互是通过标准的`kubectl`进行的，所以不需要额外的工具，而且它内置了无服务器支持。

听起来很完美！

但不幸的是，它还不够成熟，不能用于生产。社区不够大，文档不够好(不得不依赖博客帖子)，无服务器支持存在缺陷，这意味着它不能在 EKS 上工作。

鉴于积极的基础，我确信这将在未来六个月内成为事实上的 Kubernetes 无服务器框架。

### [裂变](https://fission.io/)

裂变很有趣，因为它位于 Kubeless 和 OpenWhisk 之间。它严重依赖于 Kubernetes 的特性，但并没有完全集成。这种方法的好处是，它可以利用 Kubernetes 擅长的东西，如自动缩放，但在需要时会做一些不同的事情来获得更好的性能。例如，它有一个相当复杂的冷启动池机制。

它由[平台 9](https://platform9.com/) 支持，可以通过 Helm 安装。它使用 Influxdb 来处理状态，并提供 FluentD 来注销。它使用 NAT 作为消息总线，Redis 作为缓存。正如您所看到的，其他框架不提供缓存和开箱即用，即使添加它会相当简单。

裂变有一个非常好的额外功能叫做[裂变工作流程](https://github.com/fission/fission-workflows/tree/master/Docs)。这是一个允许开发者*组合*功能的工具。函数式编程。这是一个非常有趣的方向，我很想看看它能做些什么。

但是用户很少(只有两个栈溢出问题——是说很好用吗？)和极少数认真的贡献者——只有 6 个人有 10 次以上的提交。有一些烦恼，但这主要是由于缺乏用户和开发者。文档组织得很差。这使得很难弄清楚这个框架是如何工作的。我也不喜欢代码如何负责模板化和启动 pods 本身；这可能会在将来导致破损。最后，我对裂变建造者非常困惑。我不知道它们是什么，也不知道为什么需要它们。

此外，这个名字很难搜索。

### [联合国](https://fnproject.io/)

另一个争夺傻名字桂冠的是 Fn。它是开源的，但主要贡献者为 Oracle 工作。主工作流使用 Fn CLI，但是底层的函数使用 Docker 容器。这篇博文中有一些信息[，文档](https://medium.com/fnproject/even-wider-language-support-in-fn-with-init-images-a7a1b3135a6e)[在这里](https://github.com/fnproject/docs/blob/master/cli/how-to/create-init-image.md)。并且有可能用 Helm 部署框架组件[。还有一个名为](https://github.com/fnproject/fn-helm) [Fn Flow](https://medium.com/fnproject/serverless-sagas-with-fn-flow-d8199b608b12) 的新功能，它编排了多个类似于 OpenWhisk 工作流的功能。

但是最重要的区别是你工作的方式。Fn 的重点是易于使用，但这使得它非常固执己见。它提供了热函数(所有其他框架也可以提供)和流函数(这是更独特的——不完全清楚它如何与其他框架一起工作)。

它始于 2016 年，这使它与 OpenWhisk 一样古老，并拥有相当数量的贡献者。尽管如此，我还是对强加给开发者的大量意见感到不快，而且这种意见不太适合 Kubernetes(你不能用 Kubernetes manifests AFAICT 进行部署)。这是对我的要求，所以我没有尝试。然而，它与无服务器框架兼容，可以稍微减轻这一点。

### [铁功能](https://github.com/iron-io/functions)

Iron Functions 由同名公司提供支持。这增加了一些初始的复杂性，因为 github readme 会将您重定向到“文档”，这实际上是 Iron 公司的主页。从那里如果你点击[“文件”](http://dev.iron.io/)，那么没有提到铁的功能。真正的文档在存储库的`docs`目录中。

像许多其他框架一样，它也是基于 Docker 的，一个有趣的特性是它对 AWS Lambda 函数的对等支持。你可以从 Lambda 获取代码，然后直接在 Iron 上运行；这对迁移来说很好。

不幸的是，它似乎不像 Fn 那样支持将清单部署到 Kubernetes，而且它也不受无服务器框架的支持。这足以使我不去尝试它。它也没有流行到出现在谷歌趋势上。

### [无服务器框架](https://serverless.com/framework/docs/)

我在文章中多次提到这一点，因此它本身就应该有一个章节。

它不是一个平台，它不运行任何功能。这是一个无服务器的软件开发工具包。事实上，它本质上只是一个包装机制。但美妙之处在于，一旦以无服务器的方式打包，你可以将相同的代码部署到 Lambda、Google Functions、Azure Functions、OpenWhisk、OpenFaas、Kubeless 或 Fn。

这种便携性非常吸引人。它允许开发人员以标准的方式构建他们的代码，但仍然允许标准、定价、功能集或可用性来决定在哪里部署它。

此外，它允许我们稍微推迟框架的选择。我之前提到过，我喜欢 Kubeless 在 Kubernetes 方面采取的方向，但它还不够成熟，无法使用。如果我们使用无服务器，那么我们现在可以在 OpenFaas 或 Lambda 上构建代码，以后可以很容易地移植到无服务器。

唯一不好的就是蠢名字的金刚。每一个。单身。时间。我必须加上后缀“无服务器是框架，而不是技术”。再加上支持的语言数量有限。但除此之外，我认为这是最安全的选择。

### [功能](https://funktion.fabric8.io/)

Funktion 是 RedHat 的一个废弃解决方案。