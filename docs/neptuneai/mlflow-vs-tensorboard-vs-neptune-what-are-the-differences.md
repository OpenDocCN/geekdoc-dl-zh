# MLflow vs TensorBoard vs Neptune:有什么区别？

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/mlflow-vs-tensorboard-vs-neptune-what-are-the-differences>

你看到无穷无尽的列和行，随机的颜色，不知道在哪里找到任何值？啊，美丽混乱的实验数据表。机器学习开发者不应该经历这种痛苦。

在电子表格中跟踪和管理无数的变量和工件令人疲惫不堪。您必须手动处理:

*   **参数**:超参数，模型架构，训练算法
*   **作业**:预处理作业、培训作业、后处理作业—这些作业消耗其他基础架构资源，如计算、网络和存储
*   **工件**:训练脚本、依赖项、数据集、检查点、训练模型
*   **指标**:训练和评估准确性，损失
*   **调试数据**:权重、偏差、梯度、损耗、优化器状态
*   **元数据**:实验、试验和作业名称、作业参数(CPU、GPU 和实例类型)、工件位置(例如 S3 桶)

从长远来看，切换到专用的[实验跟踪工具](/web/20221206080949/https://neptune.ai/blog/best-ml-experiment-tracking-tools)是不可避免的。如果你已经在考虑哪种工具适合你，今天我们将比较 Neptune、Tensorboard 和 MLflow。以下是您将在本文中发现的内容:

*   快速概述 MLflow、Tensorboard、Neptune 以及它们的功能；
*   比较 MLflow、Tensorboard、Neptune 特征的详细图表；
*   当海王星是比 MLflow 和 Tensorboard 更好的替代方案；
*   海王星如何与 MLflow 和 Tensorboard 集成。

## MLflow、TensorBoard 和 Neptune 的快速概述

尽管您可以使用这三种工具来解决类似的问题，但是根据您的使用情况，它们的差异可能非常重要。

在 **[海王](/web/20221206080949/https://neptune.ai/)** 中，你可以追踪机器学习实验，记录度量，性能图表，视频，音频，文字，记录数据探索，有机的组织团队合作。Neptune 速度很快，您可以自定义 UI，并在本地环境或云环境中管理用户。管理用户权限和对项目的访问轻而易举。它监控硬件资源消耗，因此您可以优化您的代码以有效地使用硬件。

Neptune 具有广泛的框架集成，因此集成您的 ML 模型、代码库和工作流不会有问题。它是按比例构建的，因此您的实验不会有太大的问题。

[**MLflow**](https://web.archive.org/web/20221206080949/https://mlflow.org/) 是一个开源平台，通过跟踪实验来管理您的 ML 生命周期，提供可在任何平台上重复运行的打包格式，并将模型发送到您选择的部署工具。您可以记录运行，将它们组织到实验中，并使用 MLflow tracking API 和 UI 记录其他数据。

[](https://web.archive.org/web/20221206080949/https://www.tensorflow.org/tensorboard)**另一方面是专门从事视觉化。您可以跟踪包括损失和准确性在内的指标，并可视化模型图。Tensorboard 允许您查看权重、偏差或其他张量的直方图、项目嵌入，并且您可以合并描述图像、文本和音频。**

 **### 比较 MLflow、TensorBoard 和 Neptune 特性的详细图表

Neptune 在[实验跟踪](/web/20221206080949/https://neptune.ai/experiment-tracking)、框架和团队协作方面的灵活性，使其高于 MLflow 和 Tensorboard。

|  | MLflow | 海王星 | 张量板 |
| --- | --- | --- | --- |
|  |  | 

——个人免费，
非盈利及教育研究团队:

 |  |
|  |  |  |  |
| 

**实验跟踪功能**

 |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |
| 

规模达到百万次运行

 |  |  |  |
|  |  |  |  |
|  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

## 当海王星是比 MLflow 和 TensorBoard 更好的选择时

让我们来探索当你想要选择[海王星而不是 MLflow](/web/20221206080949/https://neptune.ai/vs/mlflow) 和 TensorBoard 的情况。稍后，您还将看到这三个工具可以一起使用，为您提供一个丰富的环境来满足您所有的 ML 实验需求。

### 哪个工具的可视化仪表板最容易为您的整个团队设置？

在 Neptune 中，您可以通过在托管服务器上备份或本地安装来保存实验数据。您可以轻松共享实验，无需任何开销。

说到 TensorBoard 和 MLflow，他们在本地存储和跟踪实验。他们的用户管理和团队设置能力非常有限，所以我绝对推荐 Neptune 用于大型协作项目。

### 可以在 MLflow 和 TensorBoard 中管理用户权限吗？

在 MLflow 和 TensorBoard 中协作非常有限。[在 Neptune 中，你拥有对用户和访问权限的完全控制权](https://web.archive.org/web/20221206080949/https://docs.neptune.ai/administration/user-management)。有三种用户规则:管理员、贡献者或查看者。

**您可以通过电子邮件邀请轻松添加团队成员:**

### MLflow 和 TensorBoard 上千跑快吗？

Neptune 是按比例构建的，以便支持前端和后端的数百万次实验运行。

MLflow 作为一个开源工具，并不是最快的工具；尤其是运行了 100 次或 1000 次之后，UI 会变得滞后。TensorBoard 是一个可视化工具包，它远没有 Neptune 快。

### 可以在 MLflow 和 TensorBoard 中保存不同的实验仪表板视图吗？

TensorBoard 和 MLflow 最适合个人工作，具有本地存储和本地 UI/仪表板。对于多用户(多租户)，这很快就会变得不舒服。

团队成员可以对如何设计实验仪表板有不同的想法。在 Neptune 中，每个人都可以随心所欲地定制、更改和保存实验仪表板视图。

### 能否在 MLflow 和 TensorBoard 中获取硬件指标？

不多。TensorBoard Profile 确实分析了代码的执行，但只针对当前运行。但是在 Neptune 中，您可以持续监控硬件和资源消耗(CPU、GPU、内存),同时训练您的模型。有了这些数据，您可以优化您的代码，以最大限度地利用您的硬件。

这些数据是自动生成的，您可以在 UI 的 monitoring 部分找到它:

### 在 MLflow 和 TensorBoard 中记录图像和图表有多容易？

现在，您可以在 UI“日志”部分的“预测”选项卡中浏览您的图像。

您甚至可以通过 neptunecontrib.api.log_chart 记录将在 UI 中交互呈现的交互式图表。

### MLflow 和 TensorBoard 会自动给你的 Jupyter 笔记本拍快照吗？

Neptune 与 Jupyter 笔记本集成，因此无论何时运行包含`neptune.create_experiment()`的单元格，您都可以自动拍摄快照。

不管您是否提交您的实验，所有的东西都将被安全地版本化，并准备好被探索。

### 你能用 MLflow 和 TensorBoard 追踪一个解释性分析吗？

在海王星，你可以版本化你的探索性数据分析或结果探索。保存在 Neptune 中后，您可以命名、共享和下载，或者在您的笔记本检查点中查看差异。

使用 Neptune，您可以自动将图像和图表记录到多个图像通道，浏览它们以查看模型训练的进度，并更好地了解训练和验证循环中发生的事情。

要将一个或多个图像记录到日志部分，您只需:

```py
neptune.log_image('predictions', image)
for image in validation_predications:
    neptune.log_image('predictions', image)

```

### MLflow 和 TensorBoard 允许你直接把你的实验仪表板拿到熊猫数据框吗？

mlflow.serach_runs()API 返回熊猫数据帧中的 mlflow 运行。

海王星允许你获取你或你的队友追踪和探索的任何信息。HiPlot 集成等探索性特性将帮助您做到这一点。

```py
neptune.init('USERNAME/example-project')

make_parallel_coordinates_plot(

     metrics= ['eval_accuracy', 'eval_loss',...],

     params = ['activation', 'batch_size',...])
```

## 与 MLflow 的 Neptune 集成

正如我们之前提到的，MLflow 的一个缺点是，您不能轻松地共享实验，也不能就它们进行协作。

为了增加组织和协作，您需要托管 MLflow 服务器，确认正确的人可以访问，存储备份，并通过其他环节。

实验对比界面有点欠缺，尤其是团队项目。

但是你可以把它和海王星结合起来。这样，您可以使用 MLflow 接口来跟踪实验，将 runs 文件夹与 Neptune 同步，然后享受 Neptune 灵活的 UI。

您不需要备份 *mlruns* 文件夹或在专用服务器上启动 MLflow UI 仪表板。多亏了海王星，你的 [MLflow 实验将被自动托管、备份、组织并支持团队合作。](https://web.archive.org/web/20221206080949/https://docs.neptune.ai/integrations-and-supported-tools/experiment-tracking/mlflow)

将工作流程从:

```py
mlflow ui
```

收件人:

```py
neptune mlflow

```

你可以像平常一样做任何事情。

## 海王星与张量板集成

您还可以[将 Neptune 与 TensorBoard](https://web.archive.org/web/20221206080949/https://docs.neptune.ai/integrations-and-supported-tools/experiment-tracking/tensorboard) 集成，让您的 TensorBoard 可视化托管在 Neptune 中，将您的 TensorBoard 日志直接转换为 Neptune 实验，并即时记录主要指标。

首先，安装库:

```py
pip install neptune - tensorboard

```

在用 Tensorboard 日志创建了一个简单的训练脚本并初始化 Neptune 之后，您可以用两行简单的代码进行集成:

```py
import neptune_tensorboard as neptune_tb
neptune_tb.integrate_with_tensorflow()

```

一定要创建实验！

```py
with neptune.create_experiment(name=RUN_NAME, params=PARAMS):
```

现在，你的实验将被记录到海王星，你也可以享受团队协作的功能。

了解有关海王星的更多信息…

## 如您所见，这些工具并不一定相互排斥。您可以从 MLflow 和 TensorBoard 的喜爱功能中受益，同时使用 Neptune 作为管理您的实验并与您的团队合作的中心位置。

你想更多地了解海王星吗？

你想马上开始追踪你的实验吗？

快乐实验！

Happy experimenting!**