# MLOps:你应该知道的 10 个最佳实践

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/mlops-10-best-practices>

> “……开发和部署 ML 系统相对来说既快又便宜，但是随着时间的推移对它们进行维护既困难又昂贵。”–d . Sculley 等人，“机器学习系统中隐藏的技术债务，NIPS 2015

每一位数据科学家都会认同这句话。也许你在寻找解决机器学习系统的许多移动部分之一的问题时遇到过它:数据、模型或代码。

拼凑一个解决方案通常意味着招致技术债务，这种债务会随着系统的老化和/或复杂性的增加而增加。更糟糕的是，您可能会损失时间、浪费计算资源并导致生产问题。

MLOps 可能令人生畏。成千上万的课程可以帮助工程师提高他们的机器学习技能。虽然开发一个模型来实现业务目标(项目分类或预测连续变量)并将其部署到生产中相对容易，但在生产中操作该模型会带来许多问题。

由于数据漂移等原因，模型性能在生产中可能会降低。您可能需要改变预处理技术。这意味着新的模型需要不断地投入生产，以解决性能下降的问题，或者提高模型的公平性。

除了持续集成和持续交付的 DevOps 实践之外，这需要持续的培训和持续的监控。因此，在本文中，我们将探索工程师需要的一些最佳实践，以一致地交付他们的组织所需的机器学习系统。

## 命名规格

命名约定并不新鲜。例如，Python 对命名约定的建议包含在[PEP 8:Python 代码的样式指南](https://web.archive.org/web/20221208062127/https://www.python.org/dev/peps/pep-0008/#naming-conventions)中。随着机器学习系统的增长，变量的数量也在增长。

因此，如果你为你的项目建立了一个清晰的命名约定，工程师们将会理解不同变量的角色，并且随着项目复杂性的增加而遵循这个约定。

这种做法有助于缓解“改变一切就改变一切”( CACE)原则的挑战。这也有助于团队成员快速熟悉您的项目。这是一个构建 Azure 机器学习管道的项目的例子。

```py
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep

intermediate_data_name_merge = "merged_ibroka_data"

merged_ibroka_data = (PipelineData(intermediate_data_name_merge, datastore=blob_datastore)
                      .as_dataset()
                      .parse_parquet_files()
                      .register(name=intermediate_data_name_merge, create_new_version=True)
                     )

mergeDataStep = PythonScriptStep(name="Merge iBroka Data",
                                 script_name="merge.py",
                                 arguments=[
                                         merged_ibroka_data,
                                         "--input_client_data", intermediate_data_name_client,
                                         "--input_transactions_data", intermediate_data_name_transactions
                                           ],
                                 inputs=[cleansed_client_data.as_named_input(intermediate_data_name_client),
                                         cleansed_transactions_data.as_named_input(intermediate_data_name_transactions)],
                                 outputs=[merged_liko_data],
                                 compute_target=aml_compute,
                                 runconfig=aml_runconfig,
                                 source_directory="scripts/",
                                 allow_reuse=True
                                ) 

print("mergeDataStep created")

intermediate_data_name_featurize = "featurized_liko_data"

featurized_ibroka_data = (PipelineData(intermediate_data_name_featurize, datastore=blob_datastore)
                    .as_dataset()
                    .parse_parquet_files()
                    .register(name=intermediate_data_name_featurize, create_new_version=True)
                    )

featurizeDataStep = PythonScriptStep(name="Featurize iBroka Data",
                                 script_name="featurize.py",
                                 arguments=[
                                     featurized_liko_data,
                                     "--input_merged_data", intermediate_data_name_merge,
                                           ],
                                 inputs=[merged_liko_data.as_named_input(intermediate_data_name_merge)],
                                 outputs=[featurized_liko_data],
                                 compute_target=aml_compute,
                                 runconfig=aml_runconfig,
                                 source_directory="scripts/",
                                 allow_reuse=True
                                )

print("featurizeDataStep created")
```

这里，流水线的两个步骤的中间输出被命名为**中间 _ 数据 _ 名称 _ 合并**和**中间 _ 数据 _ 名称 _ 特征化。**它们遵循易于识别的命名惯例。

如果在项目的另一个方面遇到了另一个这样的变量，比如说**intermediate _ data _ name _ clean**，这种命名约定可以很容易地理解它在更大的项目中所扮演的角色。

## 代码质量检查

Alexander Van Tol 关于代码质量的文章提出了高质量代码的三个合意标识:

*   它做它应该做的事情
*   它不包含缺陷或问题
*   易于阅读、维护和扩展

由于 CACE 原理，这三个标识符对于机器学习系统尤其重要。

通常，输入训练管道的真实世界数据中没有明确包含结果变量。例如，设想一个包含订阅事务的 SQL 数据库。可能没有说明特定订阅是否被续订的列。但是，很容易查看后续交易，并了解所述订阅是否在到期时终止。

这种结果变量的计算可以发生在训练流水线的一个步骤中。如果执行这种计算的函数有任何问题，模型将被拟合到错误的训练数据上，并且在生产中不会做得很好。代码质量检查(在这种情况下是单元测试)让像这样的关键功能做它们应该做的事情。

然而，代码质量检查超越了单元测试。您的团队将从使用 linters 和 formatters 在您的机器学习项目中强制执行特定的代码风格中受益。这样，您可以在 bug 进入产品之前将其消除，检测代码气味(死代码、重复代码等)。)，并加快代码审核速度。这对您的 CI 流程是一个促进。

将这种代码质量检查作为 pull 请求触发的管道的第一步是一个很好的实践。你可以在带有 AzureML 模板项目的 [MLOps 中看到这样的例子。如果你想让棉绒成为一个团队，这里有一篇很棒的文章可以帮助你开始——](https://web.archive.org/web/20221208062127/https://github.com/microsoft/MLOpsPython/blob/master/.pipelines/diabetes_regression-ci.yml#L27-L37)[棉绒不会妨碍你。他们站在你这边](https://web.archive.org/web/20221208062127/https://stackoverflow.blog/2020/07/20/linters-arent-in-your-way-theyre-on-your-side/)

## 实验——并跟踪您的实验！

特征工程、模型架构和超参数搜索都在不断发展。鉴于当前的技术状态和数据中不断发展的模式，ML 团队总是致力于交付尽可能最好的系统。

一方面，这意味着掌握最新的想法和基线。这也意味着尝试这些想法，看看它们是否能提高你的机器学习系统的性能。

实验可能包括尝试代码(预处理、训练和评估方法)、数据和超参数的不同组合。每一个独特的组合产生了你需要与你的其他实验进行比较的指标。此外，实验运行条件(环境)的变化可能会改变您获得的指标。

回忆什么提供了什么好处，什么有效，很快就会变得乏味。使用现代工具(海王星是一个伟大的！)当你尝试新的过程时，跟踪你的实验可以提高你的生产力，而且它使你的工作具有可重复性。

想入手[实验用海王星跟踪](/web/20221208062127/https://neptune.ai/experiment-tracking)？阅读本文—[ML 实验跟踪:它是什么，为什么重要，以及如何实施](/web/20221208062127/https://neptune.ai/blog/ml-experiment-tracking)。

[https://web.archive.org/web/20221208062127if_/https://www.youtube.com/embed/9jN7RuPNEyc?feature=oembed](https://web.archive.org/web/20221208062127if_/https://www.youtube.com/embed/9jN7RuPNEyc?feature=oembed)

视频

## 数据有效性

在生产中，数据可能会产生各种各样的问题。如果数据的统计属性不同于训练数据属性，则训练数据或采样过程是错误的。数据漂移可能会导致连续数据批次的统计属性发生变化。数据可能具有意外的特征，一些特征可能以错误的格式传递，或者像 Erick Breck 等人的[论文](https://web.archive.org/web/20221208062127/https://research.google/pubs/pub47967/)中的例子一样，一个特征可能被错误地固定到特定值！

服务数据最终成为训练数据，因此检测数据中的错误对于 ML 模型的长期性能至关重要。一旦发现错误，你的团队就可以进行调查并采取适当的行动。

Pandera 是一个数据验证库，可以帮助你完成这项工作，以及其他复杂的统计验证，如假设检验。这里有一个使用 Pandera 定义的数据模式的例子。

```py
import pandera as pa
from azureml.core import Run

run = Run.get_context(allow_offline=True)

if run.id.startswith("OfflineRun"):
    import os

    from azureml.core.dataset import Dataset
    from azureml.core.workspace import Workspace
    from dotenv import load_dotenv

    load_dotenv()

    ws = Workspace.from_config(path=os.getenv("AML_CONFIG_PATH"))

    liko_data = Dataset.get_by_name("liko_data")
else:
    liko_data = run.input_datasets["liko_data"]

df = liko_data.to_pandas_dataframe()

liko_data_schema = pa.DataFrameSchema({
    "Id": pa.Column(pa.Int, nullable=False),
    "AccountNo": pa.Column(pa.Bool, nullable=False),
    "BVN": pa.Column(pa.Bool, nullable=True, required=False),
    "IdentificationType": pa.Column(pa.String checks=pa.Check.isin([
        "NIN", "Passport", "Driver's license"
    ]),
    "Nationality": pa.Column(pa.String, pa.Check.isin([
        "NG", "GH", "UG", "SA"
    ]),
    "DateOfBirth": pa.Column(
        pa.DateTime,
        nullable=True,
        checks=pa.Check.less_than_or_equal_to('2000-01-01')
    ),
    "*_Risk": pa.Column(
        pa.Float,
        coerce=True,
        regex=True
    )
}, ordered=True, strict=True)

run.log_table("liko_data_schema", liko_data_schema)
run.parent.log_table("liko_data_schema", liko_data_schema)

```

该模式确保:

*   **Id** 是一个整数，不能为空
*   **BVN** 是一个布尔值，它可能在某些数据中不存在
*   **IdentificationType** 是列出的四个选项之一
*   **出生日期**为空或小于“2000-01-01”
*   包含字符串“ **_Risk** ”的列包含可强制转换为 float dtype 的数据。
*   新数据的列顺序与该架构中定义的顺序相同。这可能很重要，例如，当使用 XGBoost API 时，可能会因为列顺序不匹配而引发错误。
*   此模式中未定义的列不能作为服务数据的一部分传递。

这个简单的模式在项目中构建了许多数据验证功能。然后，可以在下游步骤中应用定义的模式，如下所示。

```py
liko_data_schema.validate(data_sample)
```

[Tensorflow](https://web.archive.org/web/20221208062127/https://www.tensorflow.org/) 还提供了全面的数据验证 API，这里记录了。

## 跨细分市场的模型验证

重用模型不同于重用软件。您需要调整模型以适应每个新的场景。为此，您需要培训渠道。模型也会随着时间而衰减，需要重新训练才能保持有用。

实验跟踪可以帮助我们处理模型的版本化和可再现性，但是在将模型提升到产品之前对其进行验证也是很重要的。

您可以离线或在线验证。离线验证包括在测试数据集上生成度量(例如，准确度、精确度、归一化均方根误差等)，以通过历史数据评估模型对业务目标的适合性。在做出促销决策之前，这些指标将与现有的生产/基准模型进行比较。

适当的实验跟踪和元数据管理为您提供了指向所有这些模型的指针，您可以无缝地进行回滚或升级。通过 A/B 测试进行在线验证，如本文[的文章](https://web.archive.org/web/20221208062127/https://mlinproduction.com/ab-test-ml-models-deployment-series-08/)中所探讨的，然后在实时数据上建立模型的适当性能。

除此之外，您还应该在不同的数据段上验证模型的性能，以确保它们满足需求。业界越来越注意到机器学习系统可以从数据中学习的偏见。一个流行的例子是 Twitter 的图像裁剪功能，该功能被证明对于某些用户来说表现不佳。为不同的用户群验证模型的性能可以帮助您的团队检测并纠正这种类型的错误。

## 资源利用:记住你的实验是要花钱的

在培训期间和部署后的使用中，模型需要系统资源— CPU、GPU、I/O 和内存。了解您的系统在不同阶段的需求可以帮助您的团队优化您的实验成本并最大化您的预算。

这是一个经常受到关注的领域。公司关心利润，他们希望最大限度地利用资源来创造价值。云服务提供商也意识到了这一点。Sireesha Muppala 等人在他们的[文章](https://web.archive.org/web/20221208062127/https://aws.amazon.com/blogs/machine-learning/identify-bottlenecks-improve-resource-utilization-and-reduce-ml-training-costs-with-the-new-profiling-feature-in-amazon-sagemaker-debugger/)中分享了关于在 Amazon SageMaker 调试器中降低培训成本的考虑。微软 Azure 还允许工程师在使用 SDK 部署之前[确定他们模型的资源需求](https://web.archive.org/web/20221208062127/https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-profile-model?pivots=py-sdk#run-the-profiler)。

这种分析使用提供的数据集测试模型，并报告资源需求的建议。因此，重要的是所提供的数据集能够代表模型投入生产时可能提供的服务。

剖析模型还提供了成本之外的其他优势。次优资源可能会降低培训工作的速度，或者给生产中的模型操作带来延迟。这些是机器学习团队必须快速识别和修复的瓶颈。

## 监控预测服务性能

到目前为止，上面列出的实践可以帮助您持续交付一个健壮的机器学习系统。在操作中，除了训练/服务数据和模型类型之外，还有其他指标决定您部署的模型的性能。可以说，这些指标与熟悉的项目指标(如 RMSE、AUC-ROC 等)一样重要。)来评估与业务目标相关的模型性能。

用户可能需要机器学习模型的实时输出，以确保能够快速做出决策。在这种情况下，监控运营指标至关重要，例如:

*   延迟:以毫秒为单位。用户能获得无缝体验吗？
*   可伸缩性:以每秒查询数(QPS)衡量。在预期延迟下，您的服务可以处理多少流量？
*   服务更新:在更新您的服务的基础模型的过程中引入了多少停机时间(服务不可用)？

例如，当时尚公司开展广告活动时，ML 推荐系统的不良服务性能会影响转化率。客户可能会对服务延迟感到失望，然后不再购买。这转化为商业损失。

Apache Bench 是 Apache 组织提供的一个工具，它允许您监控这些关键指标，并根据您组织的需求做出正确的规定。对于您的服务来说，跨不同地理位置测量这些指标非常重要。Austin Gunter 的[用 Apache Benchmark](https://web.archive.org/web/20221208062127/https://digwp.com/2012/04/measure-latency-apache-bench/) 测量延迟和这个[教程](https://web.archive.org/web/20221208062127/https://www.tutorialspoint.com/apache_bench/index.htm)也是对这个有用工具的很好的介绍。

## 仔细考虑你选择的 ML 平台

MLOps 平台可能很难比较。尽管如此，你在这里的选择可以决定你的机器学习项目的成败。您的选择应基于以下信息:

*   你所拥有的团队:经验的水平；主题专家还是技术专家？
*   你的项目是用传统的机器学习还是深度学习。
*   您将使用的数据类型。
*   你的商业目标和预算。
*   技术需求，比如你的模型监控需要有多复杂。
*   该平台的特性以及从长远来看它们将如何发展。

在线上有一些 ML 平台的比较来指导你的选择，比如[机器学习中的 12 大现场跟踪工具](/web/20221208062127/https://neptune.ai/blog/top-12-on-prem-tracking-tools-in-machine-learning)。海王星是讨论的平台之一。它使协作变得容易，并帮助团队管理和监控长期运行的实验，无论是在内部还是在 web UI 中。你可以在这里查看它的主要概念。

## 开放的通信线路很重要

长期实施和维护机器学习系统意味着各种专业人员之间的合作:数据工程师、数据科学家、机器学习工程师、数据可视化专家、DevOps 工程师和软件开发人员的团队。UX 设计师和产品经理也可以影响服务于您的系统的产品如何与用户交互。经理和企业所有者有控制如何评估和欣赏团队绩效的期望，而合规专业人员确保运营符合公司政策和法规要求。

如果您的机器学习系统要在不断发展的用户和数据模式及期望中不断实现业务目标，那么参与其创建、操作和监控的团队必须有效沟通。Srimram Narayan 探讨了这样的多学科团队如何在[敏捷 IT 组织设计](https://web.archive.org/web/20221208062127/https://www.amazon.com/gp/product/0133903354?ie=UTF8&tag=martinfowlerc-20&linkCode=as2&camp=1789&creative=9325&creativeASIN=0133903354)中采用面向结果的设置和业务目标方法。一定要把它加入到你的周末读物中！

## 定期对您的 ML 系统进行评分

如果您了解上述所有实践，很明显您(和您的团队)致力于在您的组织中建立最佳的 MLOps 实践。你应该得到一些掌声！

给你的机器学习系统打分既是你努力的一个很好的起点，也是随着你的项目逐渐成熟而进行持续评估的一个很好的起点。谢天谢地，这样的评分系统是存在的。Eric Breck 等人在他们的论文中提出了一个综合评分系统——[你的 ML 测试分数是多少？大规模生产系统的规则](https://web.archive.org/web/20221208062127/https://research.google/pubs/pub45742/)。评分系统包括功能和数据、模型开发、基础设施以及监控。

## **结论**

就是这样！您绝对应该考虑实施的 10 个实践是:

*   命名规格
*   代码质量检查
*   实验——并跟踪您的实验！
*   数据有效性
*   跨细分市场的模型验证
*   资源利用:记住你的实验是要花钱的
*   监控预测服务性能
*   仔细考虑你选择的 ML 平台
*   开放的通信线路很重要
*   定期对您的 ML 系统进行评分

尝试一下，你肯定会看到你在 ML 系统上的工作有所改进。