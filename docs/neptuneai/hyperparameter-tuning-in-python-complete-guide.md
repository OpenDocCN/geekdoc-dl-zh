# Python 中超参数调优:完全指南

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/hyperparameter-tuning-in-python-complete-guide>

为机器学习或深度学习模型选择正确的超参数是从模型中提取最后汁液的最佳方式之一。在本文中，我将向您展示目前可用的一些最佳超参数调优方法。

## 参数和超参数有什么区别？

首先我们来了解一下机器学习中超参数和参数的[区别。](https://web.archive.org/web/20230304181723/https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)

*   **模型参数**:这些是模型从给定的数据中估计出来的参数。例如深度神经网络的权重。
*   **模型超参数**:这些是模型无法从给定数据中估计出来的参数。这些参数用于估计模型参数。比如深度神经网络中的学习速率。

| 因素 | 超参数 |
| --- | --- |
| 

它们是进行预测所需要的

 | 

它们是估算模型参数所需要的

 |
|  | 

它们是通过超参数调谐

来估计的 |
| 

它们不是手动设置的

 |  |
| 

训练后发现的最终参数将决定模型如何对未发现的数据执行

 | 

超参数的选择决定了训练的效率。在梯度下降中，学习率决定了优化过程在估计参数时的效率和准确性

 |

*模型参数 vs 模型超参数|来源:[GeeksforGeeks](https://web.archive.org/web/20230304181723/https://www.geeksforgeeks.org/difference-between-model-parameters-vs-hyperparameters/)*

## 什么是超参数调整，为什么它很重要？

[超参数调整](https://web.archive.org/web/20230304181723/https://towardsdatascience.com/hyperparameter-tuning-c5619e7e6624)(或超参数优化)是确定最大化模型性能的正确超参数组合的过程。它的工作原理是在一个训练过程中进行多次试验。每次试验都是使用您选择的超参数值(在您指定的范围内设置)完整执行您的训练应用程序。该过程一旦完成，将为您提供最适合模型的一组超参数值，以获得最佳结果。

不用说，这是任何机器学习项目中的重要一步，因为它会导致模型的最优结果。如果您希望看到它的实际应用，[这里有一篇研究论文](https://web.archive.org/web/20230304181723/https://arxiv.org/pdf/2007.07588.pdf)，它通过在数据集上进行实验，讲述了超参数优化的重要性。

## 如何进行超参数调谐？如何找到最佳超参数？

选择正确的超参数组合需要理解超参数和业务用例。但是，从技术上来说，有两种方法可以设置它们。

### 手动超参数调谐

手动超参数调整包括手动试验不同的超参数集，即您将执行一组超参数的每次试验。这项技术需要一个强大的实验跟踪器，它可以跟踪从图像、日志到系统指标的各种变量。

有几个实验跟踪器可以勾选所有的框。 [neptune.ai](/web/20230304181723/https://neptune.ai/) 就是其中之一。它提供了一个直观的界面和一个开源包 neptune-client 来方便你登录代码。您可以轻松记录超参数，并查看所有类型的数据结果，如图像、指标等。查看文档，看看如何[将不同的元数据记录到 Neptune](https://web.archive.org/web/20230304181723/https://docs.neptune.ai/logging/what_you_can_log/) 。

替代解决方案包括 W&B、Comet 或 MLflow。点击查看更多[实验跟踪工具&管理。](/web/20230304181723/https://neptune.ai/blog/best-ml-experiment-tracking-tools)

**手动超参数优化的优势**:

*   手动调整超参数意味着对过程的更多控制。
*   如果您正在研究调优以及它如何影响网络权重，那么手动进行调优是有意义的。

**手动超参数的缺点**优化**** :

*   手动调谐是一个乏味的过程，因为可能有许多尝试，并且跟踪可能证明是昂贵和耗时的。
*   当有许多超参数需要考虑时，这不是一个非常实用的方法。

点击了解[如何手动优化机器学习模型超参数。](https://web.archive.org/web/20230304181723/https://machinelearningmastery.com/manually-optimize-hyperparameters/)

### 自动超参数调谐

自动超参数调整利用现有的算法来自动化该过程。您需要遵循的步骤是:

*   首先，指定一组超参数和对这些超参数值的限制(注意:每个算法都要求这组参数是特定的数据结构，例如，在处理算法时，字典是常见的)。
*   然后算法会帮你完成繁重的工作。它会运行这些试验，并为您获取最佳的超参数集，以获得最佳结果。

在博客中，我们将讨论一些可以用来实现自动调优的算法和工具。我们开始吧。

## 超参数调谐方法

在这一节中，我将介绍当今流行的所有超参数优化方法。

### 随机搜索

在[随机搜索方法](https://web.archive.org/web/20230304181723/https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)中，我们为超参数创建一个可能值的网格。每次迭代尝试这个网格中超参数的随机组合，记录性能，最后返回提供最佳性能的超参数组合。

### 网格搜索

在[网格搜索方法](https://web.archive.org/web/20230304181723/https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e)中，我们为超参数创建一个可能值的网格。每次迭代以特定的顺序尝试超参数的组合。它在每个可能的超参数组合上拟合模型，并记录模型性能。最后，它返回具有最佳超参数的最佳模型。

### 贝叶斯优化

为您的模型调整和找到正确的超参数是一个优化问题。我们希望通过改变模型参数来最小化模型的损失函数。贝叶斯优化帮助我们在最少的步骤中找到最小的点。[贝叶斯优化](https://web.archive.org/web/20230304181723/https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f)还使用一个采集函数，该函数将采样指向可能比当前最佳观测值有所改进的区域。

### 树结构 Parzen 估计量(TPE)

[基于树的 Parzen 优化](https://web.archive.org/web/20230304181723/https://optunity.readthedocs.io/en/latest/user/solvers/TPE.html)的思想类似于贝叶斯优化。TPE 模型 P(x|y)和 p(y)不是寻找 p(y|x)的值，其中 y 是要最小化的函数(例如，验证损失), x 是超参数的值。树结构 Parzen 估计器的一个大缺点是它们没有模拟超参数之间的相互作用。也就是说，TPE 在实践中工作得非常好，并且在大多数领域都经过了实战考验。

## 超参数调整算法

这些是专门为进行超参数调整而开发的算法。

### 超波段

Hyperband 是随机搜索的一种变体，但是使用了一些[探索-利用](https://web.archive.org/web/20230304181723/https://en.wikipedia.org/wiki/Multi-armed_bandit#Empirical_motivation)理论来为每种配置找到最佳的时间分配。你可以查看这篇[研究论文](https://web.archive.org/web/20230304181723/https://arxiv.org/abs/1603.06560)以获取更多参考。

### 基于人口的培训

这种技术是两种最常用的搜索技术的混合:随机搜索和应用于神经网络模型的手动调整。

PBT 从用随机超参数并行训练许多神经网络开始。但是这些网络并不是完全相互独立的。

它使用来自其余群体的信息来改进超参数，并确定要尝试的超参数的值。你可以查看这篇[文章](https://web.archive.org/web/20230304181723/https://deepmind.com/blog/article/population-based-training-neural-networks)了解更多关于 PBT 的信息。

### BOHB

BOHB(贝叶斯优化和超带)混合了超带算法和贝叶斯优化。可以查看这篇[文章](https://web.archive.org/web/20230304181723/https://www.automl.org/blog_bohb/)进一步参考。

现在你知道了什么是方法和算法，让我们来谈谈工具，有很多这样的工具。

**一些最好的超参数优化库是:**

1.  [Scikit-learn](#scikit-learn)
2.  [sci kit-优化](#scikit-optimize)
3.  [Optuna](#optuna)
4.  [远视](#hyperopt)
5.  [雷.调](#ray)
6.  [塔罗斯](#talos)
7.  [贝叶斯优化](#bayesianoptimization)
8.  [度量优化引擎(MOE)](#moe)
9.  [留兰香](#spearmint)
10.  [GPyOpt](#gpyopt)
11.  [SigOpt](#sigopt)
12.  [法博拉斯](#fabolas)

### 1\. Scikit-learn

Scikit-learn 实现了网格搜索和随机搜索，如果您正在使用 sklearn 构建模型，这是一个很好的起点。

对于这两种方法，scikit-learn 在各种参数选择上以 k 倍交叉验证设置训练和评估模型，并返回最佳模型。

具体来说:

*   **随机搜索:**与 [`randomsearchcv`](https://web.archive.org/web/20230304181723/https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) 在一些随机参数组合上运行搜索
*   **网格搜索:** [`gridsearchcv`](https://web.archive.org/web/20230304181723/https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 对网格中的所有参数集进行搜索

用 scikit-learn 调优模型是一个好的开始，但是还有更好的选择，而且它们通常有随机搜索策略。

### 2.sci kit-优化

[Scikit-optimize](https://web.archive.org/web/20230304181723/https://scikit-optimize.github.io/stable/index.html) 使用基于[序列模型的优化](https://web.archive.org/web/20230304181723/https://ml.informatik.uni-freiburg.de/papers/11-LION5-SMAC.pdf)算法，在更短的时间内找到超参数搜索问题的最优解。

Scikit-optimize 提供了除超参数优化之外的许多功能，例如:

*   存储和加载优化结果，
*   收敛图，
*   比较代理模型

### 3.奥普图纳

[Optuna](https://web.archive.org/web/20230304181723/https://optuna.org/) 使用轨迹细节的历史记录来确定有希望的区域，以搜索优化超参数，从而在最短的时间内找到最佳超参数。

它具有修剪功能，可以在训练的早期阶段自动停止不被看好的轨迹。optuna 提供的一些主要功能有:

*   轻量级、通用且平台无关的架构
*   Pythonic 搜索空间
*   高效优化算法
*   易于并行化
*   快速可视化

关于如何开始使用 optuna 的教程，可以参考官方[文档](https://web.archive.org/web/20230304181723/https://optuna.readthedocs.io/en/stable/tutorial/first.html)。

### 4.远视

[Hyperopt](https://web.archive.org/web/20230304181723/http://hyperopt.github.io/hyperopt/) 是目前最流行的超参数调谐包之一。Hyperopt 允许用户描述一个搜索空间，在该搜索空间中，用户期望得到最佳结果，从而允许 hyperopt 中的算法更有效地搜索。

目前，hyperopt 中实现了三种算法。

要使用远视，您应该首先描述:

*   最小化的目标函数
*   要搜索的空间
*   存储搜索的所有点评估的数据库
*   要使用的搜索算法

这个[教程](https://web.archive.org/web/20230304181723/https://github.com/hyperopt/hyperopt/wiki/FMin)将带你了解如何构建代码并使用 hyperopt 包来获得最佳的超参数。

您也可以阅读[这篇](https://web.archive.org/web/20230304181723/https://mlwhiz.com/blog/2019/10/10/hyperopt2/)文章，了解更多关于如何使用 Hyperopt 的信息。

### 5.射线调谐

[射线调谐](https://web.archive.org/web/20230304181723/https://docs.ray.io/en/latest/tune/index.html)是实验和超参数调谐在任何规模的流行选择。Ray 使用分布式计算的能力来加速超参数优化，并在大规模上实现了几种最先进的优化算法。

“光线调节”提供的一些核心功能包括:

*   通过[利用 Ray](https://web.archive.org/web/20230304181723/https://github.com/ray-project/ray) 实现开箱即用的分布式异步优化。
*   易于扩展。
*   提供了 [ASHA](https://web.archive.org/web/20230304181723/https://ray.readthedocs.io/en/latest/tune-schedulers.html#asynchronous-hyperband) 、 [BOHB](https://web.archive.org/web/20230304181723/https://ray.readthedocs.io/en/latest/tune-searchalg.html#bohb) 、[基于人口的训练](https://web.archive.org/web/20230304181723/https://ray.readthedocs.io/en/latest/tune-schedulers.html#population-based-training-pbt)等 SOTA 算法。
*   支持 Tensorboard 和 MLflow。
*   支持 Sklearn、XGBoost、TensorFlow、PyTorch 等多种框架。

你可以参考这个[教程](https://web.archive.org/web/20230304181723/https://docs.ray.io/en/latest/tune/tutorials/overview.html)来学习如何针对你的问题实现光线调谐。

### 6\. Keras Tuner

[Keras Tuner](https://web.archive.org/web/20230304181723/https://keras.io/keras_tuner/) 是一个帮助您为 TensorFlow 程序选择最佳超参数集的库。当您为超参数调整构建模型时，除了模型架构之外，您还定义了超参数搜索空间。您为超参数调整设置的模型被称为*超模型*。

您可以通过两种方法定义超级模型:

*   通过使用模型构建器功能
*   通过子类化 Keras Tuner API 的超级模型类

对于计算机视觉应用，您还可以使用两个预定义的超级模型类——[hyperexception](https://web.archive.org/web/20230304181723/https://keras-team.github.io/keras-tuner/documentation/hypermodels/#hyperxception-class)和 [HyperResNet](https://web.archive.org/web/20230304181723/https://keras-team.github.io/keras-tuner/documentation/hypermodels/#hyperresnet-class) 。

进一步的实现细节可以参考这个[官方教程](https://web.archive.org/web/20230304181723/https://www.tensorflow.org/tutorials/keras/keras_tuner)。

### 7.贝叶斯最优化

[BayesianOptimization](https://web.archive.org/web/20230304181723/https://github.com/fmfn/BayesianOptimization) 是一个软件包，旨在最大限度地减少寻找接近最佳组合的参数组合所需的步骤。

这种方法使用了一个代理优化问题(寻找获取函数的最大值)，虽然这仍然是一个困难的问题，但在计算意义上它更便宜，并且可以使用常见的工具。因此，贝叶斯优化最适合对要优化的函数进行采样是一项非常昂贵的工作的情况。

点击这里访问 GitHub repo [查看它的运行情况。](https://web.archive.org/web/20230304181723/https://github.com/fmfn/BayesianOptimization)

### 8.度量优化引擎

[MOE(公制优化引擎)](https://web.archive.org/web/20230304181723/https://github.com/Yelp/MOE)当评估参数耗时或昂贵时，MOE 是优化系统参数的有效方法。

它是解决以下问题的理想选择

*   优化问题的目标函数是一个黑盒，不一定是凸的或凹的，
*   衍生品不可用，
*   我们寻求的是全局最优，而不仅仅是局部最优。

这种处理黑盒目标函数的能力允许我们使用 MOE 来优化几乎任何系统，而不需要任何内部知识或访问。

访问 GitHub [repo](https://web.archive.org/web/20230304181723/https://github.com/Yelp/MOE) 了解更多信息。

### 9.留兰香

[Spearmint](https://web.archive.org/web/20230304181723/https://github.com/HIPS/Spearmint) 是一个软件包，也执行贝叶斯优化。该软件被设计成自动运行实验(因此代号为 spearmint ),以迭代的方式调整多个参数，以便在尽可能少的运行中最小化一些目标。

阅读并实验 GitHub [repo](https://web.archive.org/web/20230304181723/https://github.com/HIPS/Spearmint) 中关于留兰香的内容。

### 10.GPyOpt

[GPyOpt](https://web.archive.org/web/20230304181723/https://github.com/SheffieldML/GPyOpt) 是使用 [GPy](https://web.archive.org/web/20230304181723/http://sheffieldml.github.io/GPy/) 的高斯过程优化。它使用不同的采集函数执行全局优化。

在其他功能中，可以使用 GPyOpt 来优化物理实验(顺序或批量)和调整机器学习算法的参数。它能够通过稀疏高斯过程模型处理大型数据集。

不幸的是，GPyOpt 的维护已经被 repo 的作者关闭了，但是你仍然可以在你的实验中使用这个包。

前往 GitHub repo [这里](https://web.archive.org/web/20230304181723/https://github.com/SheffieldML/GPyOpt)。

### 11.SigOpt

[SigOpt](https://web.archive.org/web/20230304181723/https://sigopt.com/) 将自动超参数调整与训练跑步跟踪完全集成，让您了解更广阔的前景和达到最佳模式的途径。

凭借高度可定制的搜索空间和多尺度优化等功能，SigOpt 可以在将模型投入生产之前，使用简单的 API 对其进行复杂的超参数调整。

访问文档[此处](https://web.archive.org/web/20230304181723/https://app.sigopt.com/docs/intro/intelligent_optimization)了解更多关于 SigOpt 超参数调整的信息。

### 12.法博拉斯

传统的贝叶斯超参数优化器将机器学习算法在给定数据集上的损失建模为要最小化的黑盒函数，而大型数据集上的快速贝叶斯优化(FABOLAS)对数据集大小上的损失和计算成本进行建模，并使用这些模型来执行具有额外自由度的贝叶斯优化。

你可以在这里查看实现 fabolas [的函数，在这里](https://web.archive.org/web/20230304181723/https://github.com/automl/RoBO/blob/master/robo/fmin/fabolas.py#L31)查看研究论文[。](https://web.archive.org/web/20230304181723/https://arxiv.org/pdf/1605.07079.pdf)

## 超参数调整资源和示例

在这一节中，我将分享一些为不同的 ML 和 DL 框架实现的超参数调优示例。

### 随机森林超参数调整

### XGBoost 超参数调整

### LightGBM 超参数调谐

### CatBoost 超参数调节

### Keras 超参数调谐

### PyTorch 超参数调谐

## 最后的想法

恭喜你，你成功了！超参数调优是任何机器学习项目不可或缺的一部分，因此这个主题总是值得深入研究。在这篇博客中，我们讨论了广泛使用和研究的不同超参数调优算法和工具。但即使如此，我们也涵盖了大量的技术和工具，正如一位智者曾经说过的，知识永无止境。

以下是该领域的一些最新研究，你可能会感兴趣:

就这些了，请继续关注更多，再见！