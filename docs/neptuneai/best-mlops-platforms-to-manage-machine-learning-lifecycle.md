# 管理机器学习生命周期的最佳 MLOps 平台

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/best-mlops-platforms-to-manage-machine-learning-lifecycle>

机器学习生命周期是以高效的方式开发机器学习项目的过程。建立和训练一个模型是一个艰难而漫长的过程，但这只是你整个任务中的一步。机器学习生命周期背后有一个漫长的过程:收集数据、准备数据、分析、训练和测试模型。

组织必须管理数据、代码、模型环境和机器学习模型本身。这需要一个过程，在这个过程中，他们可以部署他们的模型，监控它们，并对它们进行再培训。大多数组织在生产中有多个模型，即使只有一个模型，事情也会变得复杂。

## 典型的 ML 生命周期

1.  收集数据:ML 生命周期的第一步，这里的目标是从各种来源收集数据。
2.  数据准备:收集完所有数据后，您需要对数据进行清理和转换，以便进行处理和分析。这是重新格式化数据和进行修正的重要过程。
3.  分析数据:现在数据已经准备好了，可以用来建立模型，选择分析技术，等等。一旦完成，输出被审查以选择 ML 技术。
4.  训练模型:使用各种机器学习算法来训练数据集，需要理解流程、模式和规则。
5.  测试模型:一旦训练完成，您可以测试您的数据，检查模型的准确性，以及它的表现如何。
6.  部署:现在所有的事情都运行良好，是时候将机器学习模型部署到现实世界中了。您可以监视模型的执行情况，并查看如何改进工作流。

参见[关于机器学习生命周期](/web/20221203091531/https://neptune.ai/blog/life-cycle-of-a-machine-learning-project)的详细解释。

## 机器学习生命周期管理平台

在生产中管理机器学习模型是一项艰巨的任务，因此为了优化这一过程，我们将讨论几个最佳和最常用的机器学习生命周期管理平台。这些平台从小规模到企业级云和开源 ML 平台，将帮助您改进从收集数据到将应用程序部署到现实世界的 ML 工作流程。

### 1.亚马逊 SageMaker

Amazon SageMaker 是一个 ML 平台，它可以帮助您在生产就绪的 ML 环境中构建、训练、管理和部署机器学习模型。SageMaker 使用专门构建的工具加速您的实验，包括标记、数据准备、培训、调整、主机监控等等。

Amazon SageMaker 提供了大约 17 个与 ML 相关的内置服务，并且他们可能会在未来几年增加更多。确保你熟悉 AWS 的基础知识，因为你永远不知道分配这些服务器每小时要花多少钱。

**特性:**

Amazon Sagemaker 有许多特性，使 ML 模型的构建、培训、监控和部署变得简单。

*   Sagemaker 附带了许多用于训练数据集(大数据集)的 ML 算法。这有助于提高模型的准确性、比例和速度。
*   Sagemaker 包括监督和非监督的 ML 算法，如线性回归、XGBoost、聚类和客户细分。
*   端到端的 ML 平台加速了建模、标记和部署的过程。AutoML 特性将根据您的数据自动构建、训练和调整最佳 ML 模型。
*   SageMaker 为您提供了集成 API 和 SDK 的选项，使设置变得容易，您可以在任何地方使用机器学习功能。
*   它有 150 多个预构建解决方案，您可以快速部署。这有助于您开始使用 Sagemaker。

查看亚马逊 SageMaker 与 Neptune 相比如何，以及如何[将 Neptune 整合到亚马逊 sage maker 管道中](https://web.archive.org/web/20221203091531/https://docs.neptune.ai/execution-environments/amazon_sagemaker.html)。

**定价:**

SageMaker 不是免费的。如果你是新用户，前两个月可能是免费的。有了 AWS，您只需为您使用的内容付费。在 SageMaker 上构建、培训和部署您的 ML 模型，您将按秒计费。没有额外的费用或任何额外的收费。定价分为 ML 存储、实例和数据处理。

*参见* [*构建、训练和部署 ML 模型(ASM)*](https://web.archive.org/web/20221203091531/https://aws.amazon.com/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/)

### 2.Azure 机器学习

Azure ML 是一个基于云的平台，可用于训练、部署、自动化、管理和监控所有机器学习实验。就像 SageMaker 一样，它既支持监督学习，也支持非监督学习。

**特性:**

Azure 具有创建和管理 ML 模型的特性:

*   Azure ML 平台支持 Python、R、Jupyter Lab 和 R studios，具有自动机器学习功能。
*   拖放功能提供了一个无代码的机器学习环境，有助于数据科学家轻松协作。
*   你可以选择在本地机器或 Azure 机器学习云工作空间中训练你的模型。
*   Azure 机器学习支持许多开源工具，包括 Tensorflow、Scikit-learn、ONNX、Pytorch。它有自己的 MLops 开源平台([微软 MLOps](https://web.archive.org/web/20221203091531/https://github.com/microsoft/MLOps) )。
*   一些关键功能包括协作笔记本、AutoML、数据标签、MLOps、混合和多云支持。
*   强大的 [MLOps](https://web.archive.org/web/20221203091531/https://azure.microsoft.com/en-in/services/machine-learning/mlops/) 功能支持创建和部署。轻松管理和监控您的机器学习实验。

**定价:**

Azure 机器学习提供 12 个月的免费服务和少量积分来探索 Azure。与其他平台一样，Azure 也采用现收现付的定价模式。免费信用/服务结束后，您需要付费使用。

*蔚蓝—*[*入门*](https://web.archive.org/web/20221203091531/https://azure.microsoft.com/en-in/services/machine-learning/#documentation)

### 3.谷歌云人工智能平台

谷歌云是一个端到端的完全托管的机器学习和数据科学平台。它具有帮助您更快、更无缝地管理服务的功能。他们的 ML 工作流程使开发者、科学家和数据工程师的工作变得简单。该平台具有许多支持机器学习生命周期管理的功能。

**特性:**

谷歌云人工智能平台包括许多资源，可以帮助你有效地进行机器学习实验。

*   云存储和 bigquery 帮助您准备和存储数据集。然后，您可以使用内置功能来标记您的数据。
*   通过使用具有易于使用的 UI 的 Auto ML 特性，您可以在不编写任何代码的情况下执行任务。你可以使用 Google Colab，在那里你可以免费运行你的笔记本。
*   Google 云平台支持许多开源框架，如 KubeFlow、Google Colab 笔记本、TensorFlow、VM 镜像、训练模型和技术指南。
*   部署可以通过 Auto ML 特性来完成，它可以对您的模型执行实时操作。
*   通过管道管理和监控您的模型和端到端工作流。你可以用 AI 解释和 [What-if-too](https://web.archive.org/web/20221203091531/https://cloud.google.com/ai-platform/prediction/docs/using-what-if-tool) l 来验证你的模型，这有助于你了解你的模型输出，它的行为，以及改进你的模型和数据的方法。

**定价:**

谷歌人工智能云平台根据您的项目和预算，为您提供灵活的定价选项。谷歌对每个功能的收费都不一样。你可能很少看到完全免费的功能，也很少从你开始使用它们的那一刻起就开始收费。当您开始使用 GCP 时，您将获得 300 美元的免费点数。免费试用结束后，将根据您使用的工具按月收费。

*GCP-*[*入门*](https://web.archive.org/web/20221203091531/https://cloud.google.com/ai-platform/docs)

### 3.元流

Metaflow 是一个基于 Python 的库，帮助数据科学家和数据工程师开发和管理现实生活中的项目。这是一个专门用于 ML 生命周期管理的工作空间系统。元流最初是在网飞开发的，目的是提高科学家的生产率。

**特性:**

Metaflow 于 2019 年由网飞和 AWS 开源，它可以与 SageMaker，Python 和深度学习基础库集成。

*   Metaflow 为 stack 提供了一个统一的 API，从原型到基于生产的数据科学项目的执行都需要这个 API。
*   可以从数据仓库(可以是本地文件)或数据库中访问数据。
*   Metaflow 有一个图形用户界面，它帮助你把你的工作环境设计成一个有向无环图(D-A-G)。
*   部署到生产环境后，您可以跟踪所有的实验、版本和数据。

*元流-[入门](https://web.archive.org/web/20221203091531/https://docs.metaflow.org/introduction/what-is-metaflow)*

### 5.图纸空间

Gradient by Paperspace 是一个机器学习平台，从探索到生产都可以使用。它帮助您构建、跟踪和协作 ML 模型。它有一个云托管的设计来管理你所有的机器学习实验。大部分工作流都是围绕 NVIDIA GRID 构建的，因此您可以期待强大而更快的性能。

**特性:**

Paperspace Gradient 可帮助您探索数据、训练神经网络和部署基于生产的 ML 管道。

*   Paperspace Gradient 支持您可能正在使用或计划使用的几乎所有框架和库。
*   单一平台可训练、跟踪和监控您的所有实验和资源。
*   通过 jupyter 笔记本电脑支持的 GradientCI github 功能，您可以选择将您的机器学习项目与 github repo 集成。
*   你将获得免费的强大的图形处理器，你可以在一键启动。
*   用现代确定性过程开发 ML 管道。无缝管理版本、标签和生命周期。
*   你可以很容易地将你现有的实验转化为深度学习平台。
*   获得更多 GPU 选项，他们有 NVIDIA M4000，这是一种经济高效的卡，而 NVIDIA P5000 可以帮助您优化繁重的高端机器学习工作流。他们正计划添加 AMD 来优化机器学习工作流程。

**定价:**

根据您的用途，Paperspace Gradient 有许多方案。如果你是一名学生或初学者，它是免费的，使用和付费实例的费用有限。付费计划从每月 8 美元到 159 美元不等。你可以联系他们的销售团队来个性化你的计划。

*纸张空间渐变-[入门](https://web.archive.org/web/20221203091531/https://docs.paperspace.com/gradient/)*

### 6.MLflow

MLflow 是一个开源平台，用于管理机器学习生命周期——实验、部署和中央模型注册。它旨在与任何机器学习库、算法和部署工具一起工作。

**特性:**

MLflow 是用 REST APIs 构建的，这使得它的工作空间看起来很简单。

*   它可以与任何机器学习库、语言或任何现有代码一起工作。它在任何云中都以同样的方式运行。
*   它使用一种标准的格式来打包 ML 模型，该模型可以在下游工具中使用。
*   MLflow 主要由四个部分组成，MLflow 跟踪、MLflow 项目、MLflow 模型和 MLflow 注册。
*   MLflow 跟踪就是记录和查询你的代码和数据实验。
*   MLflow 项目是一个数据科学包，它以可重用和可复制的格式提供代码。它还包括用于运行 ML 和数据科学项目的 API 和 cmd 工具。
*   MLflow 模型帮助您部署不同类型的机器学习模型。每个模型都保存为包含任意文件的 dir。
*   MLflow registry 可以帮助您在一个中央存储库中存储、注释、探索和管理您的所有机器学习模型。

下面是一个 MLFlow 项目示例，它可以由一个名为 MLproject 的简单 YAML 文件定义:

```py
name: My Project
conda_env: conda.yaml
entry_points:
  main:
    parameters:
      data_file: path
      regularization: {type: float, default: 0.1}
    command: "python train.py -r {regularization} {data_file}"
  validate:
    parameters:
      data_file: path
    command: "python validate.py {data_file}"
```

*ml flow-*[*入门*](https://web.archive.org/web/20221203091531/https://www.mlflow.org/docs/latest/index.html)

### 7.算法 a

Algorithmia 是一个基于企业的 MLOps 平台，可加速您的研究并快速、安全且经济高效地交付模型。您可以部署、管理和扩展所有的 ML 实验。

**特性:**

Algorithmia 可帮助您安全地部署、服务、管理和监控所有机器学习工作负载。

*   Algorithmia 对工作空间有不同的规划。根据您的项目，您可以选择 Algorithmia 企业或 Algorithmia 团队。
*   Algorithmia 的平台可以帮助您构建和管理各种机器学习操作，并轻松提升您的 ML 能力。
*   它以经济高效的方式将模型交付速度提高了 12 倍。Algorithmia 将在你机器学习生命周期的所有阶段为你工作。
*   Algorithmia 使用自动化机器学习管道进行版本控制、自动化、日志记录、审计和容器化。您可以轻松访问 KPI、绩效指标和监控数据。
*   全面记录您的机器学习模型，并从您现有的实验中查询它们。Algorithmia 功能将向您清晰展示风险、合规性、成本和性能。
*   它支持超过 3900 种语言/框架。使用 MLOps 及其灵活的工具优化模型部署。
*   Algorithmia 提供了一些高级功能，可以轻松处理 CI/CD 管道并优化监控。
*   你可以将你的机器学习模型部署在云上、你的本地机器上或者任何其他类型的环境中。

![MLOps platforms - algorithmia](img/4479e33ba35a4903f1eafe07e2f9d78e.png)

**定价:**

Algorithmia 有三个计划:团队、企业专用和企业高级。团队计划采用现收现付的价格(PRO 为 299 美元/月)，对于企业级计划，您需要联系他们的销售团队。

*algorithm ia-*[*入门*](https://web.archive.org/web/20221203091531/https://algorithmia.com/developers)

### 8\. TensorFlow Extended (TFX)

Tensorflow Extended 是 Google 生产规模的 ML 平台。它提供了共享库和框架来集成到您的机器学习工作流中。

**特性:**

TFX 是一个在生产中开发和管理机器学习工作流的平台。

*   TensorFlow extended 允许您在许多平台上编排您的机器学习工作流，如 Apache、Beam、KubeFlow 等..
*   TFX 组件提供了帮助您轻松开始开发机器学习过程的功能。
*   它是实现机器学习管道的一系列组件，旨在帮助执行高端任务，建模，培训和管理您的机器学习实验。
*   TFX 流水线包括[示例生成](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/examplegen)、[统计生成](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/statsgen)、[模式生成](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/schemagen)、[示例验证器](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/exampleval)、[转换](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/transform)、[训练器](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/trainer)、[调谐器](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/tuner)、[评估器](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/evaluator)、[红外验证器](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/infra_validator)、[推动器](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/pusher)、[批量推断器](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/bulkinferrer)。
*   自动化数据生成来描述对数据的期望，这是一个查看和检查模式的特性。
*   TFX 图书馆包括:
*   [TensorFlow 数据验证](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/tfdv)分析和验证机器学习数据。这是一项高端设计，旨在改善 TFX 和 Tensorflow 的工作流程。
*   [张量流变换](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/tft)用张量流对数据进行预处理。这有助于通过均值和标准差方法归一化输入值。
*   [张量流模型分析](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/tfma)评估张量流模型。它以分布式方式提供大量数据的指标。这些指标可以在 jupyter 笔记本上计算。
*   [TensorFlow 元数据](https://web.archive.org/web/20221203091531/https://github.com/tensorflow/metadata)提供了在用 TF 训练机器学习模型时有用的元数据。可以手动生成，也可以在数据分析过程中自动生成。
*   [ML 元数据(MLMD)](https://web.archive.org/web/20221203091531/https://www.tensorflow.org/tfx/guide/mlmd) 是一个用于记录和检索机器学习工作流元数据的库。

### 9.塞尔顿

Seldon 是一个开源平台，用于在 Kubernetes 上部署机器学习模型。它帮助数据科学家无缝管理工作流、审计、ML 实验、部署等。

**特性:**

Seldon 为任何企业创建了一个从部署到治理的无缝管道。它附带了一个开源的 Python 库，用于检查和解释机器学习模型。

*   谢顿主要有三个产品:谢顿部署，谢顿代码和谢顿不在场证明。
*   它构建在 Kubernetes 上，可以运行在任何平台上(云或本地机器)
*   Seldon 支持大多数顶级的库、语言和工具包。
*   任何非 kubernetes 专家都可以部署机器学习模型，并在 ML 工作流中测试他们的模型。
*   您可以监控模型性能，并轻松检查和调试错误。
*   Seldon 附带了一个机器学习模型解释器，它可以检测你工作流程中的任何危险信号。
*   Seldon 的企业计划提供了更多功能，以实现更快的交付、适当的治理和全生命周期管理。

**定价:**

谢顿核心是免费部署到您的 Kubernetes 集群。塞尔登企业解决方案提供 14 天免费试用。您可以联系他们的销售团队或索取演示，以了解更多关于 Seldon Solutions 的信息。

### 10.HPE 埃兹迈拉 ML Ops

HPE·埃兹迈拉是惠普公司提供企业级机器学习操作的服务。从沙盒到模型训练、部署和跟踪。它可以与任何机器学习或深度学习框架无缝地执行。

**特性:**

它为 ML 生命周期的每个阶段的机器学习工作流提供了速度和敏捷性。

*   你可以在你的本地机器或者像 AWS、GCP 等云平台上运行 HPE·埃兹迈拉..
*   它为机器学习生命周期提供了基于容器的解决方案。构建、培训、部署和监控您的机器学习模型。
*   执行高端培训并获得对大数据的安全访问。
*   无缝跟踪型号版本、注册表。需要时更新您的 ML 模型。轻松监控模型性能。
*   在大规模部署之前，您可以运行 A/B 测试来验证模型。
*   您可以转变任何非云应用，无需重新架构。您可以构建应用程序，并将其部署到您想要的任何地方。

**定价:**

HPE Ezmeral 是一个企业级平台，所以你不会得到任何免费试用或现收现付的价格。你需要为你的项目得到个性化的报价。基本存储价格从 0.100 美元/GB 起，计算单价为 18.78 美元

## 比较

我们看到了许多平台，在这些平台上，你可以构建、训练、部署和管理你的机器学习模型。大多数平台彼此关系非常密切，具有相似的特性，但也有一些差异。我们将比较几个顶级平台及其最佳能力。

### 创造环境

亚马逊 Sage Maker——Sage Maker Studio 有一个很好的界面，跳过了所有的复杂性。你可以通过 jupyter 笔记本准备你的数据。

谷歌云平台–笔记本电脑云设置非常简单。只需在顶部搜索，您就可以部署所需的解决方案。GCP 主要运行在云壳特性上。你可以整合谷歌 Colab 的数据。

Microsoft Azure–Azure 允许您通过拖放选项导入保存的数据。您可以将数据集拖到实验画布上。

### 构建和培训模型

Amazon Sage Maker——借助 Python 代码，您可以将数据分成训练集和测试集。这是 SageMaker 的一个优点，它可以让你自动执行任务。使用预定的 XGBoost 算法，您可以通过在 jupyter 笔记本上运行，使用梯度优化来训练模型。

谷歌云平台——它没有预先构建、定制的机器学习算法，但它提供了一个使用 TF 运行模型的平台。通过进入数据集页面并定义模型详细信息，您可以用任何语言训练模型。创建自定义容器来安装培训工作流。

Microsoft Azure——在 Azure 中，如果你想拆分数据，你必须选择数据集中的列并拆分数据模块。Azure 允许你选择特性来训练你的算法。您可以简单地使用画布模块，拖动“火车模型”模块并连接您的数据。

### 测试、评分和部署模型

Amazon Sage Maker——要在服务器上部署模型，您需要在 Jupyter 笔记本上运行几行代码。请确保在测试后终止您的流程，以避免任何额外费用。

谷歌云平台 which 不提供自动化的超参数调整，相反，它有 Hypertune，可以让你精确地优化机器学习模型。基于 GCP 的模型被打包成一个 Python 模块，并将部署在 Google Cloud ML 上。

Microsoft Azure——为了测试和部署，您需要拖动并连接您的训练模型和您的分离数据和分数模型。完成后，您可以查看行和列中的输出。要评估您的模型执行情况，您可以将“评估模型”功能拖到画布中，并轻松连接它。

## 你应该选择哪一个？

大多数 ML 平台提供了一个健壮的基于 GUI 的工具来改进 ML 工作流程。不同的工具可能有不同的设计和工作流程。

有些平台对新手来说真的很容易。Azure 提供了一个拖动连接选项，对于访问、清理、评分和测试机器学习数据等任务来说非常简单。

当涉及到用 Python 编码和笔记本管理 ML 工作流时，很少有平台是真正非常复杂的。SageMaker、GCP 和其他一些公司是为了满足数据科学家和喜欢 Jupyter 笔记本的 ML 开发人员的需求而设计的。

每个平台都有优点和缺点。这是一个个人的选择，因为无论你选择什么平台，你的模型精度不会相差太多。工作流是不同的，但您可以导入您的算法。定价在这里是一个重要的话题，因为它们中的大多数都有一个现收现付的选项，允许您只为您使用的功能付费。我们讨论的大多数平台都有这个功能，所以定价部分没有问题。

如果你是一个单独的数据科学家、ML 工程师或有一个小团队，并且想在部署你的模型之前试用一个平台，你可以选择提供免费试用或免费积分的平台。它将帮助您了解事情是如何工作的，以及您和您的团队是否习惯使用一个平台。

### 额外研究和推荐阅读