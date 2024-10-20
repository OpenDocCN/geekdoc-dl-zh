# 物联网边缘生态系统的 MLOps:在 AWS 上构建 MLOps 环境

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/building-mlops-environment-on-aws-for-iot-edge-ecosystems>

## 什么是 MLOps？

MLOps 或 DevOps for machine learning 是一种实践，旨在改善 ML 团队和运营专业人员在开发、部署和管理机器学习模型方面的项目管理、沟通和协作。MLOps 涉及使用工具和流程来自动构建、测试和部署机器学习模型，以及在生产中监控和管理这些模型，以提高生产率、可重复性、可靠性、可审计性、数据和模型质量。

虽然 MLOps 可以提供有价值的工具来帮助您扩展业务，但在您将 MLOps 集成到机器学习工作负载中时，您可能会面临某些问题。为了实施 MLOps，组织可以使用各种工具和技术，如版本控制系统、CI/CD 管道、构建或打包流程以及用于供应或配置的基础设施代码(IaC)。

### 是不是所有的事情都是用代码构建成一个过程？

CI/CD 是一个连续的工作流，旨在迭代产品的开发和部署，以改进和升级它。持续工作流(CX)概括了这一概念，并以相同的目标交付目的。为了管理机器学习产品的生命周期，我们必须定义以前不关心的新流程，例如，持续监控模型、评估新模型发布和自动数据收集。值得一提的是，您还需要考虑创建一个维护这些新流程的平台，类似于 Git 维护其工作流和操作，因为它们也有自己的生命周期。

### 物联网边缘项目的优势

总体而言，在物联网边缘公司中实施 MLOps 非常重要，因为物联网系统通常涉及相对大量的数据、高度的复杂性和实时决策，并且可以帮助确保高效地开发和部署机器学习模型，并确保它们随着时间的推移保持可靠和准确。MLOps 有助于确保每个人都朝着相同的目标努力，并且能够及时发现和解决任何问题或挑战。这可以使公司利用其物联网边缘设备生成的数据来推动业务决策，并获得竞争优势。

## 设计 MLOps 系统

值得注意的是，实施 MLOps 实践具有挑战性，可能需要在时间、资源和专业知识方面进行大量投资。考虑到这些因素，现在不同的云提供商提供的服务和工具有许多假设的场景和解决方案，然而，现实往往与你在博客和文章中读到的相去甚远。

[AWS](https://web.archive.org/web/20230122042108/https://aws.amazon.com/) 提供了一个三层的机器学习堆栈，可以根据您的技能组合和团队对实现工作负载以执行机器学习任务的要求进行选择。如果你的用例所需要的机器学习任务可以使用 AI 服务来实现，那么你就不需要 MLOps 解决方案。另一方面，如果您使用 ML 服务或 ML 框架和基础设施，建议您实施一个 MLOps 解决方案，该解决方案将由您的使用案例的细节在概念上确定，例如，如果您在容器中构建在 AWS Greengrass 上运行的代码或通过操作系统管理它们，环境会有多大的不同，无论哪种情况都部署在边缘设备上。

### 作为第一步，执行需求评估

长话短说，作为一名 MLOps 工程师，您需要为您公司的所有类型的 Ops 需求整合和设计一个环境，并根据团队内部和跨团队的工作和计划进行定制。在设计数据、机器学习模型和代码生命周期的新流程时，您还需要有一个良好的愿景，不仅要基于 IT 标准，还要考虑它们的可行性。

![The high-level overview of the MLOps environment architectural design](img/aadd9da94d39d0281d35229761768296.png)

Figure 1: The high-level overview of the MLOps environment architectural design built with AWS services listed in the left side | Source: Author

我与不同的团队进行了几轮采访和讨论，听取他们的期望，主要是 ML 团队，并与他们分享了其他团队的期望。该评估过程的结果导致概念化和设计一个框架，该框架提供了一个用于构建、管理和自动化过程或工作流的环境，通过该环境可以实现基于个人和跨团队需求的数据、模型和代码操作。

### 建筑设计

图 1 显示了环境架构及其所有资源。我在这里对它进行了分解，并解释了事物是如何被探测的，它是如何操作的，以及一个 MLOps 工程师如何使用它并在其中领导一个新过程的开发和部署。

作为第一步，我建议在 AWS 上创建一个新角色，并在这个角色下设置所有内容，这是一个好的实践。可以使用自定义虚拟专用云(VPC)并通过实施安全组和经由自定义 VPC 路由互联网流量来实施安全环境。此外，这个角色需要被授予适当的权利和权限来创建，例如，事件、Lambdas、层和日志的规则。让我们一步一步来看:

*   **Twin SageMaker notebooks** 拥有环境角色授予的所需权限:
    –*Development Twin*是一个用于原型化和测试工作流脚本的环境，它拥有创建/更新/删除一些资源(如事件、Lambdas 和日志)的权限。
    –*部署 twin* 严格地说是专门针对将在此组装在一起的 QA 流程和工作流。

*   **红移**，数据源可以后端连接到其他数据源和数据库，或者由任何其他类型的数据源(如 S3 桶或 DynamoDB)替代。孪生兄弟有权限对其进行读写(如果需要的话)。想象一下，作为一个用例，您需要通过编写一个复杂的查询来访问不同的数据库，然后将一个摘要写到另一个表中。

*   **工作流**由几个元素组成，它们协同工作以自动化流程:
    –*event bridge*作为 Lambda 函数的自动触发器，如 cron 作业或事件
    –*Lambda 函数*用于将脚本传递给部署 twin 执行
    –*备份 Lambda 函数*以防出现故障，它可以从开发 twin 或手动调用
    –cloud watch 来记录问题、跟踪报告或设置商业智能

*   **SNS** 服务，通过电子邮件进行自动订阅/发布报告。

*   [**Gitlab**](https://web.archive.org/web/20230122042108/https://about.gitlab.com/) 用于维护工作流脚本(也用于控制平面资源)的存储库，其设计方式是通过推送请求进行部署。

![ Development twin, “control plane”](img/079d0a9d3fd345631b4445582b813a6f.png)

*Figure 2: Development twin, “control plane”, with its functionalities and associated resources to manage the environment for development and deployment | Source: Author*

### 开发演练指南

*   作为一名开发人员，我可以登录 development twin，为任务或工作流编写脚本，从不同的表或桶中获取或推送一些数据，并使用 SageMaker SDK 来施展 orchestrator 的魔法[2]。
*   在完成开发、通过 QA 并最终用脚本更新存储库之后，是时候构建工作流的其余部分了，这意味着创建触发器、lambda 函数以及日志组和流。
*   图 2 显示了与 development twin 的角色策略相关的全部资源。我实际上称开发双胞胎为控制平面，因为你可以创建、更新和删除工作流所需的所有资源。“Boto3”，AWS python SDK，让你管理所有这些资源，相信我，使用 CloudFormation 或 Terraform 太麻烦了[1]。
*   我逐渐将我的脚本放在一起，并创建了一个工具箱来与这些资源进行交互，还为双胞胎指定了一个 S3 桶，并启用了版本控制来保存模型或数据集，这些模型或数据集是这些过程或工作流的结果。

### 部署演练指南

*   图 3 很好地展示了部署的工作流中涉及的所有部分。组装工作流后，由事件调用的 lambda 函数将传递工作流存储库，并告诉 SageMaker 笔记本需要执行哪个脚本或笔记本。
*   Lambda 功能可以通过 WebSocket 与笔记本进行通信(甚至“Boto3”也可以让你打开/关闭笔记本)。下面这段 Python 代码展示了如何与 notebook API 交互。但是，WebSocket 库应该作为一个层添加到 Lambda 函数中。在执行脚本时，任何问题都将被记录下来进行跟踪，最终报告将发送给订阅该主题的团队。

```py
 ```
sm_client = boto3.client("sagemaker", region_name=region)
Url = sm_client.create_presigned_notebook_instance_url(NotebookInstanceName)["AuthorizedUrl"]

url_tokens = url.split("/")
http_proto = url_tokens[0]
http_hn = url_tokens[2].split("?")[0].split("#")[0]

s = requests.Session()
r = s.get(url)
cookies = "; ".join(key + "=" + value for key, value in s.cookies.items())
terminal_resp = s.post(
        http_proto + "//" + http_hn + "/api/terminals",
        params={ "_xsrf": s.cookies["_xsrf"] }
)

ws = websocket.create_connection(
       "wss://{}/terminals/websocket/1".format(http_hn),
        cookie=cookies,
        host=http_hn,
        origin=http_proto + "//" + http_hn,
        header=["User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"]
)
commands = "pass commands here"
ws.send(f"""[ "stdin", {commands}]""")

```py 
```

![ Workflow plumbing ](img/fe9c850ae67446cef187cbdf8089a29d.png)

*Figure 3: Workflow plumbing in one glance | Source: Author*

## 作为 MLOps 工程师的典型一天

想象一下，你在一家维护公司工作，该公司提供的服务具有内置于 AWS Greengrass 中的机器学习模型，在数百台物联网边缘设备上运行，以对传感器记录进行推断。然后，这些设备将数据流和(主动)决策的结果提交给后端服务器，以存储在桶和表上。根据我的经验，这可能是你日常工作的样子。

*   您正在参加一个战略会议，团队领导正在讨论建立新工作流的计划，该工作流基于查询表格和使用机器学习模型进行某些分析，并最终通过电子邮件发送报告。团队人员不足，许多任务需要自动化。
*   ML 团队领导希望您每天安排他们评估新部署的机器学习模型，比较不同版本，并根据预定义的性能指标持续收集一些模型漂移数据。

*   产品团队似乎试图与一个大客户达成交易，他们希望您创建并向他们发送 ML 团队在设备上部署的新发布功能的每月分析。
*   QA 团队经理已经确定了在用于监控机器学习模型的分段设备上运行的实时流程所报告的问题。可能有必要仔细检查现有的系统和流程，以确定问题的根本原因。对这些问题进行故障排除的一种方法可以包括检查由机器学习模型生成的日志和度量，建立监控工作流以识别可能指示任何问题的任何模式或异常，并在问题发生时运行调试例程。

到目前为止，无非是建立一些工作流和连续工作流(CX)，你需要为每个工作流编写脚本，创建 cron 作业，Lambdas，并将一些指标推送到日志中进行每月分析。

## 面向物联网边缘的 MLOps:挑战和需要考虑的要点

最后但同样重要的是，是时候分享我在设置环境时遇到的一些挑战了。

*   我建议努力提高你的 Terraform 技能是一项很好的投资，不要止步于此，因为许多公司也采用 Terragrunt 来保持 IaC 代码库的干燥[3，4]。在使用 IaC 的过程中，让我印象深刻的是向 MLOps 授予跨环境权限，如“暂存”或“生产”,这些权限需要在目标环境中定义，然后授予 MLOps 角色策略。稍后，您可以通过在“Boto3”会话中指定配置文件名来访问这些资源。下面是在。hcl 配置文件。

```py
 ```
IAM role in target environment:
inputs = {
    iam_role_name = "MLOps-env"
    iam_assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::MLops-env-id:root"
      },
      "Action": "sts:AssumeRole",
      "Condition": {}
    }
  ]
}
EOF
}

IAM role policy in target environment:
inputs = {
  iam_role_policy_name = "MLOps-env"
  iam_role_policy = <<-EOF
{
    "Version": "2012-10-17",
    "Statement": [
			...       
    ]
}
EOF
}

IAM role policy in MLOps environment within the “Statement”:
{
    "Effect": "Allow",
    "Action": "sts:AssumeRole",
    "Resource": "arn:aws:iam::target-env-id:role/MLOps-env"
}

```py 
```

*   我意识到，尽管传统的 Git 的 CI 无法体现机器学习工件测试，但许多公司会在 Git 中为 CD 管道采用无服务器 bash 脚本。我决定建立一个集成的 Apache Airflow 来管理和执行 CI 测试过程和工作流。值得一提的是，我将开发 twin 用于培训任务，尽管如此，它也可以作为工作流自动化。
*   Lambda functions retries 选项应该固定为 1，超时应该为 1 分钟左右，否则，工作流可能会被调用多次。
*   图表仪表板对于可视化和呈现数字和一些统计数据非常方便，但是维护和更新它们似乎是一项持续的任务。
*   提示请记住，笔记本实例中除 SageMaker 文件夹之外的所有内容都不会持续存在，并且不会为工作流的失败设置警报，稍后您会感谢自己。

## 结论

我写这篇文章的目的是通过将 MLOps 视为一种架构设计来带您走一条不同的道路，这是对您公司提供的产品或服务的 Ops 需求进行评估的结果。它并不是实现 MLOps 的终极或通用解决方案，但是如果您对 It 标准和实践有很好的理解，它会为您提供一些关于如何将不同部分整合在一起的想法。

说到向平台添加特性，我可以想象将 Apache Airflow 集成到平台中，并使用它来自动化一些特定的任务。此外，Ansible 似乎有一些有用的功能，可以在“设备上”为一些任务部署流程或工作流。启动一个 EC2 实例，并将其集成到 MLOps 环境中来托管这两个实例。有很多选择，最终，由您根据自己的需求做出正确的选择。

### 参考

Boto3 官方文档

[2][SageMaker 官方文档](https://web.archive.org/web/20230122042108/https://sagemaker.readthedocs.io/en/stable/)

[3] [Terraform AWS 提供商](https://web.archive.org/web/20230122042108/https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

[4] [保持你的地形代码干燥](https://web.archive.org/web/20230122042108/https://terragrunt.gruntwork.io/docs/features/keep-your-terraform-code-dry/)