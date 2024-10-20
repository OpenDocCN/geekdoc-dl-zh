# 面向机器学习从业者的自动学习快速入门指南

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/a-quickstart-guide-to-auto-sklearn-automl-for-machine-learning-practitioners>

在现实世界中使用 AutoML 框架正成为机器学习从业者的一件常规事情。人们经常会问:自动化机器学习( [AutoML](https://web.archive.org/web/20221207184547/https://www.automl.org/automl/) )会取代数据科学家吗？

不完全是。如果您渴望了解 AutoML 是什么以及它是如何工作的，请加入我的这篇文章。我将向您展示 auto-sklearn，一个最先进的开源 AutoML 框架。

为此，我必须做一些研究:

*   阅读 auto-sklearn [V1](https://web.archive.org/web/20221207184547/https://link.springer.com/chapter/10.1007/978-3-030-05318-5_6) 和 [V2](https://web.archive.org/web/20221207184547/https://arxiv.org/abs/2007.04074) 的第一篇和第二篇论文。
*   深入研究了 [auto-sklearn 文档](https://web.archive.org/web/20221207184547/https://automl.github.io/auto-sklearn/master/index.html)和[示例](https://web.archive.org/web/20221207184547/https://automl.github.io/auto-sklearn/master/examples/index.html)。
*   查看了官方汽车 Sklearn 博客帖子。
*   我自己做了一些实验。

AutoML research 也是如此，到目前为止我已经学到了很多。读完这篇文章后，你会更了解:

*   AutoML 是什么，AutoML 是为谁服务的？
*   为什么 auto-sklearn 对 ML 社区很重要？
*   如何在实践中使用 auto-sklearn？
*   auto-sklearn 的主要特点是什么？
*   Neptune 中带有结果跟踪的 auto-sklearn 的一个用例。

开始吧！

## 自动化机器学习

AutoML 是一个年轻的领域。AutoML 社区希望建立一个自动化的工作流，它可以将原始数据作为输入，并自动生成预测。

这个自动化的工作流程应该自动执行预处理、模型选择、超参数调整以及 ML 过程的所有其他阶段。例如，看看下图，看看微软 Azure 是如何使用 AutoML 的。

AutoML 可以提高数据科学家的工作质量，但不会将数据科学家排除在循环之外。

专家可以使用 AutoML 通过专注于表现最好的管道来提高他们的工作表现，而非专家可以使用 AutoML 系统而无需广泛的 ML 教育。如果你有 15 分钟的空闲时间，下面的对话可能会帮助你理解 AutoML 是怎么回事。

[https://web.archive.org/web/20221207184547if_/https://www.youtube.com/embed/SEwxvjfxxmE?feature=oembed](https://web.archive.org/web/20221207184547if_/https://www.youtube.com/embed/SEwxvjfxxmE?feature=oembed)

视频

*[什么是 AutoML:约什·斯塔默与约安尼斯·察马迪诺的对话](https://web.archive.org/web/20221207184547/https://youtu.be/SEwxvjfxxmE)*

## AutoML 框架

有不同类型的 AutoML 框架，每一种都有独特的特性。他们每个人都自动化了完整机器学习工作流的几个步骤，从预处理到模型开发。在这张表中，我只总结了其中值得一提的几条:

自动 sklearn

auto-sklearn 是一个基于 [scikit-Learn](https://web.archive.org/web/20221207184547/https://docs.neptune.ai/integrations-and-supported-tools/model-training/sklearn) 的 AutoML 框架。这是最先进的，而且是开源的。

## auto-sklearn 结合了强大的方法和技术，帮助创作者赢得了第一届和第二届国际汽车挑战赛。

**auto-sklearn 基于将 AutoML 定义为现金问题。**

CASH =组合算法选择和超参数优化。简而言之，我们希望在一个巨大的搜索空间中找到数据集的最佳 ML 模型及其超参数，其中包括大量的分类器和大量的超参数。在下图中，您可以看到由作者提供的 auto-sklearn 的表示。

**auto-sklearn is based on defining AutoML as a CASH problem.** 

auto-sklearn 可以解决分类和回归问题。auto-sklearn 的第一个版本是在 2015 年第 28 届神经信息处理系统国际会议上，以一篇题为“[高效和健壮的自动化机器学习](https://web.archive.org/web/20221207184547/https://link.springer.com/chapter/10.1007/978-3-030-05318-5_6)”的文章推出的。第二个版本在 2020 年随论文“ [auto-sklearn 2.0:下一代](https://web.archive.org/web/20221207184547/https://arxiv.org/abs/2007.04074)”一同亮相。

Auto-sklearn 功能

auto-sklearn 能为用户做些什么？它有几个有价值的特性，对新手和专家都有帮助。

### 只需编写五行 Python 代码，初学者就可以看到预测，专家可以提高他们的工作效率。以下是 auto-sklearn 的一些主要功能:

用 Python 编写，在最流行的 ML 库(scikit-learn)之上。

适用于许多任务，如分类、回归、多标签分类。

*   由几种预处理方法组成(处理缺失值，规范化数据)。
*   在相当大的搜索空间(15 个分类器，超过 150 个超参数被搜索)中搜索最佳 ML 管线。
*   由于使用了元学习，贝叶斯优化，集成技术。
*   auto-sklearn 是如何工作的？
*   Auto-sklearn 可以解决分类和回归问题，但是怎么做呢？机器学习管道中有很多东西。总的来说，V1 自动 sklearn 有三个主要组成部分:

### 元学习

贝叶斯优化

1.  构建合奏
2.  因此，当我们想要对新数据集应用分类或回归时，auto-sklearn 首先提取其元特征，以找到新数据集与依赖元学习的知识库的相似性。
3.  在下一步中，当搜索空间通过元学习缩小到足够大时，贝叶斯优化将试图找到并选择表现优异的 ML 管道。在最后一步中，auto-sklearn 将基于搜索空间中的最佳 ML 工作流来构建集成模型。

Auto-sklearn v2:新一代

最近 auto-sklearn 第二版上市了。让我们来回顾一下新一代的变化。在官方[博文](https://web.archive.org/web/20221207184547/https://www.automl.org/auto-sklearn-2-0-the-next-generation/)和原[论文](https://web.archive.org/web/20221207184547/https://arxiv.org/abs/2007.04074)的基础上，有四点改进:

### 他们允许每个 ML 管道在整个搜索空间内使用提前停止策略；该特性提高了大型数据集的性能，但它主要用于基于树的分类器。

改进模型选择策略:auto-sklearn 的一个重要步骤是如何选择模型。在 V2 的 auto sklearn 中，他们使用了 BOHB 等多保真优化方法。然而，他们表明，单一的模型选择并不适合所有类型的问题，他们整合了几种策略。为了熟悉新的优化方法，你可以阅读这篇文章:“ [HyperBand 和 BOHB:了解最先进的超参数优化算法](/web/20221207184547/https://neptune.ai/blog/hyperband-and-bohb-understanding-state-of-the-art-hyperparameter-optimization-algorithms)”

*   建立一个投资组合，而不是使用元特征在知识库中查找相似的数据集。您可以在下图中看到这种改进。
*   在之前的改进基础上构建自动化策略选择，以选择最佳策略。
*   Auto-sklearn 主要参数
*   虽然 Auto-sklearn 可能能够在不设置任何参数的情况下找到一个表现更好的管道，但是有一些参数可以用来提高您的生产率。要检查所有参数，请访问[官方页面](https://web.archive.org/web/20221207184547/https://automl.github.io/auto-sklearn/master/api.html)。

参数名称

缺省值

| 描述 |  |  |
| --- | --- | --- |
| 

显示试衣后的模特是否

 | 显示试衣后的模特 |  |
| 

显示任务还剩多少秒。如果你增加它，获得更好性能的机会也会增加

 | 它显示任务还剩多少秒。如果您增加它，获得更好性能的机会也会增加 |  |
| 

显示每个 ML 模型应该花费多少秒

 | 它显示了每个 ML 模型应该花费多少秒 | 

initial _ configuration s _ viarn _ metal learning

 |
|  | 有多少通过元学习的配置考虑了超参数优化。如果设置为 0，此选项将处于非活动状态。此外，该参数在自动 sklearn V2 中不可用 |  |
| 

集合中模特的数量。要禁用该参数，将其设置为 1

T3 | 集合中模特的数量。要禁用此参数，请将其设置为 1 |  |
| 

并行作业的数量。要使用所有处理器，将它设置为-1

T3 | 并行作业的数量。要使用所有处理器，请将其设置为-1 |  |
| 

建立集合模型的最佳模型数。仅当 ensemble_size 大于 1

T3 时有效 | 建立集合模型的最佳模型数。仅在 ensemble_size 大于 1 时有效 |  |
| 

当没有估值器时，它将使用所有估值器。auto-sk learn V2

T3 不提供 | 当没有估值器时，它将使用所有估值器。不适用于 V2 auto-sk learn |  |
| 

你可以从搜索空间中排除一些估值器。auto-sk learn V2

T3 不提供 | 您可以从搜索空间中排除一些评估者。不适用于 V2 auto-sk learn |  |
| 

如果不定义指标，则根据任务选择。在本文中，我们将其定义为(autosklearn . metrics . roc _ AUC)

 | 如果不定义指标，将根据任务选择指标。在本文中，我们定义它(autosklearn.metrics.roc_auc) |  |

跟踪海王星上的 Auto-sklearn 实验

我做了一些[笔记本](https://web.archive.org/web/20221207184547/https://ui.neptune.ai/mjbahmani/auto-sklearn/notebooks)，你可以轻松下载并自己做实验。但是，要再次完成所有步骤，您需要:

## [检查海王星的所有实验](https://web.archive.org/web/20221207184547/https://ui.neptune.ai/mjbahmani/auto-sklearn)

首先，你需要在你的机器上安装 auto-sklearn。为此只需使用 pip3:

如果你得到一个错误，你可能需要安装依赖项，所以请检查官方的[安装页面](https://web.archive.org/web/20221207184547/https://automl.github.io/auto-sklearn/master/installation.html)。你也可以用我在海王星为你准备的[笔记本](https://web.archive.org/web/20221207184547/https://ui.neptune.ai/mjbahmani/auto-sklearn/notebooks)。然后运行以下命令以确保安装正确完成:

让我们解决一些分类和回归问题。

```py
pip3 install auto-sklearn

```

用于分类的 Auto-sklearn

```py
import autosklearn
print(autosklearn.__version__)

```

对于分类问题，我选择了一个珍爱的 Kaggle 竞赛——[桑坦德客户交易预测](https://web.archive.org/web/20221207184547/https://www.kaggle.com/c/santander-customer-transaction-prediction/data)。请下载数据集并随机选择 10000 条记录。然后按照第一个[笔记本里的实验](https://web.archive.org/web/20221207184547/https://ui.neptune.ai/mjbahmani/auto-sklearn/n/neptune-autosklearn-v1):

### 我们还需要定义一些配置，以便更深入地了解 auto-sklearn:

```py
import autosklearn
X_train=None
X_val=None
y_train=None
y_val=None
train=pd.read_csv("./sample_train_Santander.csv")
X=train.drop(["ID_code",'target'],axis=1)
y=train["target"]
X_train,X_val,y_train,y_val = train_test_split(X,y, stratify=y,test_size=0.33, random_state=42)

automl = autosklearn.classification.AutoSklearnClassifier()

automl.fit(X_train, y_train )

y_pred=automl.predict_proba(X_val)

score=roc_auc_score(y_val,y_pred[:,1])
print(score)

show_modes_str=automl.show_models()
sprint_statistics_str = automl.sprint_statistics()

```

配置

范围值

| 描述 |  |  |
| --- | --- | --- |
| 

我用 60 秒开始实验，然后对于每个实验，我把它增加到 5000

 | 我以 60 秒开始实验，然后对于每个实验，我把它增加到 5000 秒 |  |
| 

由于案例研究高度不平衡，那么我需要将指标改为 roc_auc

 | 由于案例研究非常不平衡，因此我需要将指标改为 roc_auc |  |
| 

在 auto-sklearn V1 中，如果我没有定义重采样 _ 策略，就无法得到好的结果。但在 auto-sklearn V2 中，它自动做到了

 | 在 auto-sklearn V1 中，如果我没有定义重采样 _ 策略，它不能得到一个好的结果。但是在 V2 的 auto-sklearn 中，它是自动完成的 | 

重采样 _ 策略 _ 自变量

 |
|  |  | 要使用上述配置，您可以如下定义 automl 对象: |

由于我使用了大量不同的配置，我只是在海王星上跟踪它们。你可以在图像中看到其中的一个，在[海王星](https://web.archive.org/web/20221207184547/https://ui.neptune.ai/mjbahmani/auto-sklearn/e/AUT1-17/logs)中查看全部。

当您安装 auto-sklearn 模型时，您可以使用 [PipelineProfiler](https://web.archive.org/web/20221207184547/https://pypi.org/project/pipelineprofiler/) (pip 安装 PipelineProfiler)检查所有表现最佳的管道。为此，您需要运行以下代码:

```py
TIME_BUDGET=60
automl = autosklearn.classification.AutoSklearnClassifier(
time_left_for_this_task=TIME_BUDGET,
metric=autosklearn.metrics.roc_auc,
n_jobs=-1,
resampling_strategy='cv',
resampling_strategy_arguments={'folds': 5},
)

automl.fit(X_train, y_train )

```

您的输出应该如下所示:

另一方面，我也运行了一些基于 auto-sklearn V2 的实验。结果令人着迷。您可以看到下面的结果:

```py
import PipelineProfiler

profiler_data= PipelineProfiler.import_autosklearn(automl)
PipelineProfiler.plot_pipeline_matrix(profiler_data)
```

要使用 auto-sklearn V2，您可以使用以下代码:

用于回归的自动 sklearn

auto-sklearn 可以解决的第二类问题是回归。我基于 auto-sklearn [文档](https://web.archive.org/web/20221207184547/https://automl.github.io/auto-sklearn/master/examples/index.html)中的官方示例运行了一些实验。

```py
TIME_BUDGET=60
automl = autosklearn.experimental.askl2.AutoSklearn2Classifier(
time_left_for_this_task=TIME_BUDGET,
n_jobs=-1,
metric=autosklearn.metrics.roc_auc,
)

```

### 我只是改变了时间预算来跟踪基于时间限制的性能。下图显示了结果。

最终想法

```py
TIME_BUDGET=60
automl = autosklearn.regression.AutoSklearnRegressor(
time_left_for_this_task=TIME_BUDGET,
n_jobs=-1
)
automl.fit(X_train, y_train, dataset_name='boston')
y_pred = automl.predict(X_test)
score=r2_score(y_test, y_pred)
print(score)
show_modes_str=automl.show_models()
sprint_statistics_str = automl.sprint_statistics()

print(show_modes_str)
print(sprint_statistics_str)

```

总的来说，auto-sklearn 还是一项新技术。因为 auto-sklearn 是建立在 scikit-learn 之上的，所以许多 ML 从业者可以很快尝试一下，看看它是如何工作的。

## 这个框架最重要的优点就是为专家节省了大量的时间。一个缺点是它像一个黑盒，并没有说明如何做决定。

总而言之，这是一个相当有趣的工具，所以值得给 auto-sklearn 看一看。

The most important advantage of this framework is that it saves a lot of time for experts. The one weakness is that it acts as a black box, and doesn’t say anything about how to make a decision.

All in all, it’s a pretty interesting tool, so it’s worth giving auto-sklearn a look.