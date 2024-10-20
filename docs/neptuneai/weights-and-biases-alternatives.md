# 最佳权重和偏差备选方案

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/weights-and-biases-alternatives>

[权重&偏差](https://web.archive.org/web/20230304041944/http://wandb.ai/)，也称为 WandB，是一个 [MLOps 工具](/web/20230304041944/https://neptune.ai/blog/best-mlops-tools)，用于机器学习模型的性能可视化和实验跟踪。它有助于 ML 模型的自动化、跟踪、培训和改进。

Weights & bias 是一项基于云的服务，允许您在一个中央存储库中托管您的实验，如果您有私有基础架构，Weights & bias 也可以部署在其上。

重量与偏差提供:

一个中央的，用户友好的和交互式的仪表板，您可以查看您的实验和跟踪他们的表现。

*   跟踪模型训练过程的每个部分，可视化模型，并比较实验。
*   使用[扫描](https://web.archive.org/web/20230304041944/https://docs.wandb.ai/guides/sweeps)进行自动超参数调整，提供了超参数组合的样本，有助于模型性能和理解。
*   [团队协作报告](https://web.archive.org/web/20230304041944/https://docs.wandb.ai/guides/reports)，您可以在其中添加可视化内容，组织、解释和分享您的模型性能、模型版本和进度。
*   机器学习管道的端到端工件跟踪，从数据准备到模型部署。
*   与 Tensorflow、Pytorch、Keras、Hugging Face 等框架轻松集成。
*   团队中的协作工作，具有共享、实验等多种功能。
*   这些都是 Weights & Biases 提供的有用功能，这使得它成为研究团队寻找发现、学习和获得对机器学习实验的见解的良好工具。

然而，当谈到交付时，重量和偏差并不总是最好的选择。以下是 Weights & Biases 目前没有提供的一些功能:

**笔记本托管**:从 Jupyter 笔记本到生产部署机器学习模型是每个数据研究人员的梦想，因为它允许快速迭代并节省时间。

*   ML 生命周期管理:管理一个模型的整个生命周期在研究期间是很重要的，即从数据源到模型部署，因为它允许他们在开发的任何阶段正确地监控、调试任何问题。
*   **生产用例**:对于基于生产的团队或项目，权重&偏差不是一个好的选择，因为它缺乏生产引擎。
*   **模型部署**:研究的一个重要部分是测试和执行实时推理。这就是为什么在构建和评估模型之后就需要模型部署的原因。
*   以下是一些可供选择的工具，您可以尝试一下:

Here are some alternative tools you can try out:

## 1 海王星

*   2 张量板
*   3 彗星
*   4 MLflow
*   库伯流
*   6 SageMaker Studio
*   Neptune 是 MLOps 的[元数据存储。它允许您在一个地方记录、存储、组织、显示、比较和查询所有的模型构建元数据。这包括元数据，如模型度量和参数、模型检查点、图像、视频、音频文件、数据版本、交互式可视化、](/web/20230304041944/https://neptune.ai/)[等等。](https://web.archive.org/web/20230304041944/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display)

Neptune 是为进行大量实验的研究和生产团队构建的，他们希望组织和重现实验，并希望确保将模型转移到生产的过程顺利进行。海王星的主要焦点围绕着[实验跟踪](/web/20230304041944/https://neptune.ai/product/experiment-tracking)和[模型注册](/web/20230304041944/https://neptune.ai/product/model-registry)，以及团队协作。

主要特点:

Neptune 允许您以任何想要的结构记录和显示模型元数据。无论是模型的嵌套参数结构，训练和验证度量的不同子文件夹，还是打包模型或生产工件的单独空间。怎么组织就看你自己了。

### Neptune 的[定价](/web/20230304041944/https://neptune.ai/pricing)是基于使用的。对于所有的 ML 团队来说，它可能更具成本效益，但对于那些偶尔根本不进行实验的团队，或者那些有许多利益相关者没有深入参与实验过程的团队来说尤其如此。

*   如前所述，Neptune 是为运行大量实验的团队构建的，因此它可以处理成千上万次运行，并且不会变慢。它随着你的团队和项目的规模而扩展。
*   Neptune 有 SaaS 版本，但也可以在内部托管。如果选择第二个选项，安装过程非常容易。
*   有了 Neptune，您可以[创建定制的仪表板](https://web.archive.org/web/20230304041944/https://docs.neptune.ai/you-should-know/displaying-metadata#how-to-create-a-custom-dashboard)以一种更好的方式组合不同的元数据类型。
*   定价
*   Neptune 可以用作托管解决方案，也可以部署在您的场所。有几个可用的计划:

### 个人:免费(超出免费配额的使用量)

学术界:免费

*   团队:付费
*   查看[海王星的定价](/web/20230304041944/https://neptune.ai/pricing)了解更多信息。
*   重量和偏差 vs 海王星

Neptune 和 Weights & Biases 都是托管服务，它们提供实验跟踪、模型管理和数据版本控制。

### Neptune 更加关注模型注册特性，而 Weights & Biases 也提供了自动优化超参数的工具

*   这两种工具提供的开箱即用集成存在一些差异；Neptune 支持 R 语言、DVC 或 Optuna，而 WandB 支持 Spacy、Ray Tune 或 Kubeflow。
*   总的来说，这两个工具非常相似，都是非常好的解决方案，所以主要的区别可以在定价结构中发现(基于使用和基于用户)。
*   TensorBoard 由 TensorFlow 团队开发，是一款用于机器学习实验的开源可视化工具。
*   它用于跟踪 ML 度量，如损失和准确性、可视化模型图、直方图、将嵌入投影到低维空间等等。

有了 TensorBoard，你还可以分享你的实验结果。

主要特点:

TensorBoard 允许您跟踪您的实验。

它还允许您跟踪不是基于 TensorFlow 的实验。

### TensorBoard 允许你通过一个可共享的链接与任何人分享你的实验结果，用于出版、合作等。

*   它提供对损失和准确性等指标的跟踪和可视化。
*   TensorBoard 拥有 What-If 工具(WIT)，这是一个易于使用的接口，用于解释和理解黑盒分类和回归 ML 模型。
*   TensorBoard 有一个强大的用户社区，提供了巨大的支持。
*   定价
*   它是免费使用的。
*   重量和偏差与 TensorBoard

### 如果你需要一个个人使用的工具，不打算花钱买，也不需要额外的功能，TensorBoard 可以是一个不错的选择。

TensorBoard 更适合处理张量流的可视化。

### Comet 是一个基于云的机器学习平台，开发者可以在这个平台上跟踪、比较、分析和优化实验。

*   Comet 安装很快，只需几行代码，您就可以开始跟踪您的 ML 实验，而无需任何库。
*   主要特点:

Comet 允许您为您的实验和数据创建定制的可视化。你也可以在[面板](https://web.archive.org/web/20230304041944/https://www.comet.ml/demo/gallery/view/new#select-panel?gallery-tab=Public)上使用社区提供的。

Comet 提供了关于您的实验的实时统计数据和图表。

您可以轻松地比较您的实验，包括代码、指标、预测、见解等等。

### 使用 comet，您可以调试模型错误、特定于环境的错误等。

*   Comet 还允许您监控您的模型，并在出现问题或错误时通知您。
*   它允许团队和业务涉众之间的协作。
*   它可以轻松地与 Tensorflow、Pytorch 等集成。
*   定价
*   Comet 提供以下定价方案:
*   个人:免费(超出免费配额的使用量)
*   学术界:免费

### 团队:付费

你可以在这里阅读他们的详细定价[。](https://web.archive.org/web/20230304041944/https://www.comet.ml/site/pricing/)

*   权重和偏差与 Comet
*   这两个工具都提供了用户管理特性、托管和本地设置、模型管理、超参数搜索和工件存储。
*   如果你需要一个自定义可视化的图库， [Comet](https://web.archive.org/web/20230304041944/https://www.comet.ml/demo/gallery/view/new#select-panel?gallery-tab=Public) 就有。

Comet 为开发提供了 Java 和 R SDK，这是 Weights & Biases 中所缺少的。

### MLflow 是一个开源平台，有助于管理整个机器学习生命周期。它有助于实验跟踪，再现性，部署，并给出了一个中央模型注册。

*   MLflow 包含四个主要功能:

**MLflow Tracking** :一个 API 和 UI，用于在运行机器学习代码时记录参数、代码版本、指标和工件，并用于以后可视化和比较结果。

**MLflow 项目**:将 ML 代码打包成可重用、可复制的形式，以便与其他数据科学家共享或转移到生产中。

<https://web.archive.org/web/20230304041944im_/https://neptune.ai/wp-content/uploads/2022/11/Alternatives-to-Weights-Biases_4.mp4>

[**Source**](https://web.archive.org/web/20230304041944/https://towardsdatascience.com/5-tips-for-mlflow-experiment-tracking-c70ae117b03f) 

**MLflow Models** :管理来自不同 ML 库的模型，并将其部署到各种模型服务和推理平台。

**MLflow Model Registry** :一个中央模型存储库，用于协作管理 MLflow 模型的整个生命周期，包括模型版本控制、阶段转换和注释。

1.  主要特点:
2.  MLflow Model Registry 为组织提供了一套 API 和直观的 UI，用于注册和共享新版本的模型，以及对现有模型执行生命周期管理。
3.  MLflow Model Registry 与 MLflow tracking 组件一起使用，它允许您追溯生成模型和数据工件的原始运行，以及该运行的源代码版本，从而为所有模型和数据转换提供生命周期的完整沿袭。
4.  当存储在增量表或目录中时，自动对存储在数据湖中的数据进行版本控制。

### 允许您使用版本号或时间戳获取数据的每个版本。

*   允许您在意外写入或删除错误数据时审核和/或回滚数据。
*   复制实验和报告。
*   要了解更多关于 MLflow 的信息，请查看 [MLflow 文档](https://web.archive.org/web/20230304041944/http://mlflow.org/docs/latest/index.html)。
*   定价
*   它是免费的。
*   重量和偏差与 MLflow

如果你的预算很低，MLflow 是一个更好的选择，因为它是免费的(开源的)实验性跟踪。

### MLflow 是语言不可知的，即它可以与 Python 或 r 中的任何机器学习库一起使用，而 Weights & Biases 仅适用于 Python 脚本。

Weights & Biases 提供托管和内部设置，而 MLflow 仅作为开源解决方案提供，需要您在服务器上进行维护。

### MLflow 提供端到端的 ML 生命周期管理，而 Weights & Biases 仅提供实验跟踪、模型管理和数据版本控制等功能。

*   Kubeflow 是一个免费的开源机器学习平台，用于在 Kubernetes 上构建简单、可移植(通过容器)和可扩展的模型。Kubeflow 负责跟踪、数据版本控制、模型版本控制和模型部署。
*   Kubeflow 是 Google 为数据科学家和 ML 工程师设计的，他们喜欢开发、测试和部署 ML 管道、模型和系统到各种环境中。
*   Kubeflow 由以下组件组成:
*   **[中央仪表盘](https://web.archive.org/web/20230304041944/https://www.kubeflow.org/docs/components/central-dash/overview/) :** 这个仪表盘提供了一个中央视图，可以快速访问您的所有操作。它容纳了集群中运行的作业和组件，如管道、Katib、笔记本等。

**[kube flow Pipelines](https://web.archive.org/web/20230304041944/https://www.kubeflow.org/docs/components/pipelines/):**kube flow pipeline 是一个平台，允许 ml 工程师构建和部署打包在 Docker 映像中的端到端 ML 工作流。它由用于跟踪实验和作业的 UI、用于管道操作的 SDK、多步调度引擎和用于构建 ML 模型的笔记本组成。

**[KFServing](https://web.archive.org/web/20230304041944/https://www.kubeflow.org/docs/components/kfserving/):**KFServing 组件是 Kubeflow 的模型部署和服务工具包。它通过在 Kubernetes 上启用无服务器推理来提供生产模型服务，并为在 TensorFlow、XGBoost、scikit-learn、PyTorch 和 ONNX 等框架上的部署提供抽象层。

**[Katib](https://web.archive.org/web/20230304041944/https://www.kubeflow.org/docs/components/katib/) :** Katib 是一个模型不可知的 Kubernetes-native 项目，为 AutoML 模型提供超参数调整、早期停止和神经架构搜索。它支持各种 AutoML 算法和框架，如 TensorFlow、MXNet、PyTorch 等。

*   **[培训操作员](https://web.archive.org/web/20230304041944/https://www.kubeflow.org/docs/components/training/) :** 该组件提供 Kubernetes 中 Tensorflow、PyTorch、MXNet、XGBoost、MPI 模型培训作业的操作员。
*   **[Kubeflow 笔记本](https://web.archive.org/web/20230304041944/https://www.kubeflow.org/docs/components/notebooks/) :** 这个 Kubeflow 的笔记本组件允许你在集群内部运行你的笔记本。您还可以在集群中创建笔记本，并在整个组织中共享它们。
*   主要特点:
*   Kubeflow 还在其工件存储中存储工件数据；它使用工件来理解各种 Kubeflow 组件的管道是如何工作的。
*   Kubeflow Pipeline 可以输出工件数据的简单文本视图和丰富的交互式可视化。
*   Kubeflow 有一个用户界面(UI ),用于管理和跟踪实验、作业和运行。

它为多步 ML 工作流提供调度。

### 它有一个用于定义和操作管道和组件的 SDK。

*   使用 SDK 与系统交互的笔记本电脑。
*   定价
*   它是免费的。
*   权重和偏差与 Kubeflow
*   使用 [Kubeflow 管道](https://web.archive.org/web/20230304041944/https://www.kubeflow.org/docs/components/pipelines/)或 [KF 服务于 Kubeflow 中的](https://web.archive.org/web/20230304041944/https://www.kubeflow.org/docs/components/kfserving/)组件，你可以在 docker 上部署机器学习模型，这是权重&偏差所缺少的。
*   Kubeflow 提供端到端的机器学习编排和管理，Weights & Biases 不提供。

### Kubeflow 为所有模型工件提供实验性跟踪和元数据跟踪。

对于不需要交互式可视化的用例，Kubeflow 是更好的选择。

### Amazon SageMaker Studio 是一个基于 web 的集成开发环境(IDE ),用于构建、培训、可视化、调试、部署和监控您的 ML 模型。您可以在一个集成的可视化界面中编写代码、跟踪实验、可视化数据以及执行调试和监控。

*   主要特点:
*   它提供了一个模型工件存储，该存储存储了包含模型类型和内容信息的模型的 s3 存储桶位置。
*   SageMaker studio 也为 AutoML 实验存储工件。
*   它允许您轻松创建和共享 Jupyter 笔记本。

它提供并管理模型环境的硬件基础设施，以便您可以快速地从一种硬件配置切换到另一种。

SageMaker Studio 支持 Tensorflow、PyTorch、MXNet 等框架。

### SageMaker Studio 有超过 150 个预打包的开源模型，用于各种用例。

*   SageMaker Studio 提供端到端的数据准备。它允许您使用自己选择的语言(SQL、Python 和 Scala)运行 Spark 作业，并且您还可以轻松连接到运行在 Amazon EMR 上的 Apache Spark 数据处理环境。
*   定价
*   有了亚马逊 SageMaker，你只需为你使用的东西付费。它提供两种付款方式:
*   按秒计费的按需定价，没有最低费用，也没有前期承诺
*   SageMaker 储蓄计划提供了一种灵活的基于使用量的定价模式，以换取对一致使用量的承诺。
*   您可以使用 [AWS 定价计算器](https://web.archive.org/web/20230304041944/https://calculator.aws/#/createCalculator/SageMaker)来计划您的账单。
*   重量与偏见 vs 亚马逊 SageMaker 工作室

### SageMaker Studio 的设置很简单，不像 Weights & Biases 需要一定水平的专业知识，因为它是一种托管和内部服务。

SageMaker studio 在实验跟踪期间提供实验日志和可视化。

1.  在 SageMaker studio 中，您可以设置一个排行榜，自动跟踪您的所有实验，然后对它们的表现进行排名。
2.  与 Weights & Biases 相比，SageMaker studio 以相对较低的价格出租计算资源。

SageMaker Studio 允许您交互式地查询、探索和可视化数据，除了实验跟踪，SageMaker studio 还提供数据注释、大量数据处理、调试以及模型和数据漂移检测。

### 结论

*   对于专注于研究的 ML 研究团队来说，Weights & Biases 是一个很好的工具，因为它擅长进行实验跟踪，但仅此还不够。本文中列出的替代工具有一些独特的价值主张，使它们适合可能不需要权重和偏差的用例。
*   对于开源的实验性跟踪工具，TensorBoard、MLflow、Kubeflow 会是很好的替代品。就元数据和工件的可视化和可伸缩存储而言，付费工具如 Neptune 和 Comet 是更好的选择。此外，它们还为企业团队提供高级别的安全性和支持。
*   因此，根据您的需求，您可以选择上述任何工具。
*   同时，我还会建议你总是寻找更多适合你的需求和任务的[工具](/web/20230304041944/https://neptune.ai/blog/best-ml-experiment-tracking-tools)，并给你足够的灵活性来最大限度地利用你的工作。
*   快乐实验！

## Conclusion

Weights & Biases is a great tool for ML research teams focusing on research because it is good at performing experimental tracking but that alone isn’t enough. The alternative tools listed in this article have some unique value propositions that make them fit into use-cases where Weights & Biases might not be needed. 

For open-source experimental tracking tools, TensorBoard, MLflow, Kubeflow would be good alternatives. In terms of visualizations and scalable storage for your metadata and artifacts, paid tools such as Neptune and Comet are better options. Additionally, they also provide high-level security and support for enterprise teams.

So, depending on your requirement, you may choose any of the aforementioned tools.

Simultaneously, I will also advise you to be always on the lookout for more [tools](/web/20230304041944/https://neptune.ai/blog/best-ml-experiment-tracking-tools) that suit your needs, tasks, and give you enough flexibility to get the most out of your work.

Happy experimenting!