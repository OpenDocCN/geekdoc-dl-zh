# 何时选择 CatBoost 而不是 XGBoost 或 light GBM[实用指南]

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/when-to-choose-catboost-over-xgboost-or-lightgbm>

Boosting 算法已经成为对结构化(表格)数据进行训练的最强大的*算法*之一。为[赢得 ML 竞赛](https://web.archive.org/web/20230219051826/https://medium.com/kaggle-blog)提供了各种方法的三个最著名的 boosting 算法实现是:

## 

*   1 催化增强
*   2 XGBoost
*   3 灯 GBM

在本文中，我们将主要关注 CatBoost，它与其他算法相比如何，以及何时应该选择它。

## 梯度增强概述

为了理解 boosting，我们必须首先理解[集成学习](/web/20230219051826/https://neptune.ai/blog/ensemble-learning-guide)，这是一套结合来自多个模型(弱学习器)的预测以获得更好的预测性能的技术。它的策略就是团结的力量，因为弱学习者的有效组合可以产生更准确和更健壮的模型。集成学习方法的三个主要类别是:

*   **Bagging:** 该技术使用数据的随机子集并行构建不同的模型，并确定性地聚合所有预测者的预测。
*   **Boosting** :这种技术是迭代的、连续的、自适应的，因为每个预测器都修正了其前一个预测器的误差。
*   **堆叠**:这是一种元学习技术，涉及结合多种机器学习算法的预测，如 bagging 和 boosting。

1988 年，Micheal Kearns 在他的论文[关于假设提升的思考](https://web.archive.org/web/20230219051826/https://www.cis.upenn.edu/~mkearns/papers/boostnote.pdf)中，提出了一个相对较差的假设是否可以转化为非常好的假设的想法。本质上是一个学习能力差的人是否可以通过改造变得更好。从那以后，已经有许多成功的技术应用来开发一些强大的 boosting 算法。

![The most popular boosting algorithms: Catboost, XGBoost, LightGBM](img/ec21e27bce4f890da3f4bbf5ddd01eee.png)

*The most popular boosting algorithms: Catboost, XGBoost, LightGBM | Source: Author*

scope 中的三种算法(CatBoost、XGBoost 和 LightGBM)都是梯度增强算法的变体。随着我们的进展，对梯度推进的良好理解将是有益的。梯度推进算法可以是回归器(预测连续目标变量)或分类器(预测分类目标变量)。

这种技术涉及使用梯度下降优化过程，基于最小化弱学习者的微分损失函数来训练学习者，这与像自适应增强(Adaboost)那样调整训练实例的权重相反。因此，所有学习者的权重是相等的。梯度提升使用串联的决策树作为弱学习器。由于它的顺序架构，它是一个阶段式的加法模型，每次添加一个决策树，现有的决策树不变。

梯度增强主要用于减少模型的偏差误差。基于偏差-方差权衡，它是一种贪婪算法，可以快速地过拟合训练数据集。然而，这种过拟合可以通过收缩、树约束、正则化和随机梯度增强来控制。

## CatBoost 概述

[CatBoost](https://web.archive.org/web/20230219051826/https://catboost.ai/) 是一个开源的机器学习(梯度提升)算法，其名字由“ ***类别*** ”和“ ***提升*** ”杜撰而来它是由 [Yandex](https://web.archive.org/web/20230219051826/https://yandex.com/) (俄罗斯谷歌)在 2017 年开发的。Yandex 表示，CatBoost 已被广泛应用于推荐系统、搜索排名、自动驾驶汽车、预测和虚拟助手等领域。它是在 Yandex 产品中广泛使用的 [MatrixNet](https://web.archive.org/web/20230219051826/https://en.wikipedia.org/wiki/MatrixNet) 的继任者。

## CatBoost 的主要特性

让我们来看看使 CatBoost 优于其同类产品的一些关键特性:

1.  **对称树** : CatBoost 构建对称(平衡)树，不像 XGBoost 和 LightGBM。在每一步中，前一棵树的叶子都使用相同的条件进行分割。考虑到最低损失的特征分割对被选择并用于所有级别的节点。这种平衡树结构有助于有效的 CPU 实现，减少预测时间，使模型应用程序更快，并在该结构用作正则化时控制过拟合。

![Symmetric trees](img/42868a01925b71507284475d4c1e06f0.png)

*Asymmetric tree vs symmetric tree | Source: Author*

2.  **有序推进:**由于被称为预测偏移的问题，经典推进算法容易在小/有噪声的数据集上过度拟合。计算数据实例的梯度估计值时，这些算法使用构建模型时使用的相同数据实例，因此不会遇到看不见的数据。另一方面，CatBoost 使用有序提升的概念，这是一种排列驱动的方法，用于在数据子集上训练模型，同时在另一个子集上计算残差，从而防止目标泄漏和过度拟合。

3.  **原生特性支持:** CatBoost 支持所有类型的特性，无论是数字、分类还是文本，节省了预处理的时间和精力。

### 数字特征

CatBoost 像其他基于树的算法一样处理数字特征，即通过基于信息增益选择最佳可能的分割。

![Numerical features](img/57fa6e09e96e6e04c25b1e5dda581390.png)

*Numerical features | Source: Author*

### 分类特征

决策树基于类别而不是连续变量中的阈值来分割分类特征。分割标准是直观的，因为类被分成子节点。

![Categorical features](img/90c2bf1ea92a3051c879d63a17c90d44.png)

Categorical features *| Source: Author*

类别特征在高基数特征中可能更复杂，如' *id* 特征。每个机器学习算法都需要解析数字形式的输入和输出变量；CatBoost 提供了各种本地策略来处理分类变量:

*   **单热编码**:默认情况下，CatBoost 用单热编码表示所有二进制(两类)特征。通过改变训练参数***one _ hot _ max _ size****=****N .***CatBoost*通过指定分类特征和类别来处理一键编码，可以将该策略扩展到具有 **N** 数量类别的特征，以产生更好、更快、更高质量的结果。*

 **   **基于类别的统计:** CatBoost 应用随机排列的目标编码来处理类别特征。这种策略对于高基数列非常有效，因为它只创建了一个新的特性来说明类别编码。将随机排列添加到编码策略是为了防止由于数据泄漏和特征偏差导致的过拟合。你可以在这里详细了解[。](https://web.archive.org/web/20230219051826/https://catboost.ai/en/docs/concepts/algorithm-main-stages_cat-to-numberic)

*   **贪婪搜索组合:** CatBoost 还自动组合分类特征，大多数情况下是两个或三个。为了限制可能的组合，CatBoost 不会枚举所有的组合，而是使用类别频率等统计数据来枚举一些最佳组合。因此，对于每个树分割，CatBoost 会将当前树中先前分割已经使用的所有分类特征(及其组合)与数据集中的所有分类特征相加。

### 文本特征

CatBoost 还通过使用单词包(BoW)、Naive-Bayes 和 BM-25(用于多类)提供固有的文本预处理来处理文本特征(包含常规文本),以便从文本数据中提取单词、创建词典(字母、单词、grams)并将其转换为数字特征。这种文本转换是快速的、可定制的、生产就绪的，并且也可以用于其他库，包括神经网络。

4.  **排名:**排名技术主要应用于搜索引擎，解决搜索相关性问题。排名可以在三个目标函数下大致完成:[点方式、成对方式和列表方式](https://web.archive.org/web/20230219051826/https://medium.com/@nikhilbd/pointwise-vs-pairwise-vs-listwise-learning-to-rank-80a8fe8fadfd)。这三个目标函数在较高层次上的区别在于训练模型时考虑的实例数量。

CatBoost 有一个排名模式——[CatBoostRanking](https://web.archive.org/web/20230219051826/https://catboost.ai/docs/concepts/loss-functions-ranking)就像 [XGBoost ranker](https://web.archive.org/web/20230219051826/https://xgboost.readthedocs.io/en/stable/python/python_api.html) 和 [LightGBM ranke](https://web.archive.org/web/20230219051826/https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html) r 一样，但是，它提供了比 XGBoost 和 LightGBM 更强大的变化。这些变化是:

*   排名(yetirank，yetirankpairwise)
*   成对(PairLogit，PairLogitPairwise)
*   排名+分类(QueryCrossEntropy)
*   排名+回归(QueryRMSE)
*   选择前 1 名候选人(QuerySoftMax)

CatBoost 还提供了排名基准，通过不同的排名变化来比较 CatBoost、XGBoost 和 LightGBM，其中包括:

*   **CatBoost** : RMSE、QueryRMSE、pairlogit、pairlogitpairwise、yetirank、yetirankpairwise
*   **XGBoost** : reg:线性，xgb-lmart-ndcg，XG b-成对
*   **LightGBM** : lgb-rmse，LG b-成对

这些基准评估使用了四(4)个排名靠前的数据集:

1.  **百万查询数据集**来自 TREC 2008， [MQ2008](https://web.archive.org/web/20230219051826/https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/#!letor-4-0) ，(训练和测试折叠)。
2.  **微软 LETOR 数据集**(we b-10K)(第一套，训练，测试折叠)。
3.  **雅虎 LETOR 数据集** (C14)，[雅虎](https://web.archive.org/web/20230219051826/https://webscope.sandbox.yahoo.com/catalog.php?datatype=c&guccounter=1)(第一套，set1.train.txt 和 set1.test.txt 文件)。
4.  **Yandex LETOR 数据集**、[Yandex](https://web.archive.org/web/20230219051826/https://github.com/spbsu-ml-community/jmll/tree/master/ml/src/test/resources/com/expleague/ml)(features.txt.gz 和 featuresTest.txt.gz 文件)。

使用平均 [NDCG 度量](https://web.archive.org/web/20230219051826/https://en.wikipedia.org/wiki/Discounted_cumulative_gain)进行性能评估的结果如下:

可以看出，CatBoost 在所有情况下都优于 LightGBM 和 XGBoost。可以在 CatBoost 文档[这里](https://web.archive.org/web/20230219051826/https://catboost.ai/en/docs/concepts/loss-functions-ranking)找到更多关于分级模式变化及其各自性能指标的详细信息。这些技术可以在 CPU 和 GPU 上运行。

5.  **速度** : CatBoost 通过支持[多服务器分布式 GPU](https://web.archive.org/web/20230219051826/https://docs.neptune.ai/how-to-guides/neptune-api/distributed-computing)(支持多主机加速学习)和容纳旧 GPU 来提供可扩展性。它在大型数据集上设置了一些 CPU 和 GPU 训练速度基准，如 [Epsilon](https://web.archive.org/web/20230219051826/https://catboost.ai/en/docs/concepts/python-reference_datasets_epsilon) 和 [Higgs](https://web.archive.org/web/20230219051826/https://catboost.ai/en/docs/concepts/python-reference_datasets_higgs) 。它的预测时间比 XGBoost 和 LightGBM 快；这对于低延迟环境极其重要。

![Dataset Epsilon (400K samples, 2000 features). Parameters: 128 bins, 64 leafs, 400 iterations.](img/28b143b02881cf572a8f472a61076625.png)

*Dataset Epsilon (400K samples, 2000 features). Parameters: 128 bins, 64 leafs, 400 iterations | Source: Author*

![Dataset Higgs (4M samples, 28 features). Parameters: 128 bins, 64 leafs, 400 iterations.](img/19d3bb532b8ed4ab02ddec1f41c11dfd.png)

*Dataset Higgs (4M samples, 28 features). Parameters: 128 bins, 64 leafs, 400 iterations | Source: Author*

![Prediction time on CPU and GPU respectively on the Epsilon dataset](img/8144a2ccb3fb8110acee0150548dfe0e.png)

*Prediction time on CPU and GPU respectively on the [Epsilon dataset](https://web.archive.org/web/20230219051826/https://catboost.ai/en/docs/concepts/python-reference_datasets_epsilon) | Source: Author*

[](https://web.archive.org/web/20230219051826/https://catboost.ai/en/docs/concepts/python-reference_datasets_epsilon)

6.  **模型分析:** CatBoost 提供固有的模型分析工具，在高效的统计和可视化的帮助下，帮助理解、诊断和提炼机器学习模型。其中一些是:

### 特征重要性

CatBoost 有一些智能技术，可以为给定模型找到最佳特性:

*   **预测值变化:**显示预测值相对于特征值变化的平均变化量。由于特征导致的预测变化的平均值越大，重要性越高。要素重要性值被归一化以避免否定，所有要素的重要性都等于 100。这很容易计算，但可能会导致排名问题的误导性结果。

![Feature Importance based on PredictionValuesChange](img/9774933f3207f40c461ef92c699a1c49.png)

*Feature Importance based on PredictionValuesChange | Source: Author*

*   **LossFunctionChange:** 这是一种繁重的计算技术，通过获取包括给定特征的模型的损失函数与没有该特征的模型之间的差异来获得特征重要性。差异越大，该特征越重要。

![Feature Importance based on LossFunctionChange](img/d6d882f4ba6b365f1c8827b23886d36f.png)

*Feature Importance based on LossFunctionChange | Source: Author*

*   **InternalFeatureImportance:**该技术使用路径对称树叶上的节点中的分割值来计算每个输入要素和各种组合的值。

![Pairwise feature importance](img/0c56c339a7d1d5d174a0f4aebc7889b9.png)

*Pairwise feature importance | Source: Author*

*   **SHAP:** CatBoost 使用[**SHAP**](https://web.archive.org/web/20230219051826/https://github.com/slundberg/shap)(SHapley Additive exPlanations)将一个预测值分解成每个特征的贡献。它通过测量要素对单个预测值(与基线预测相比)的影响来计算要素重要性。这种技术提供了对模型决策影响最大的特征的可视化解释。SHAP 有两种应用方式:

#### 每个数据实例

![First prediction explanation (Waterfall plot)](img/5d558e22476ad0e31d5e29ad6fedab1f.png)

*First prediction explanation (Waterfall plot) | Source: Author*

上面的可视化显示了将模型输出从基础值(训练数据集上的平均模型输出)推至模型输出的要素。红色要素将预测值推高，而蓝色要素将预测值推低。这个概念可以用力图来形象化。

![First prediction explanation (Force plot)](img/6f8b3b8f4bc4d4a828a1b428f26412f9.png)

*First prediction explanation (Force plot) | Source: Author*

#### 整个数据集

SHAP 提供了绘图功能来突出显示模型的最重要的特征。该图按所有数据实例的 SHAP 值大小的总和对要素进行排序，并使用 SHAP 值来突出显示每个要素对模型输出的影响分布。

![Summarized effects of all the features](img/d2b5ecee89a372e6a2bb0e18a62c2c65.png)

*Summarized effects of all the features | Source: Author*

### 特征分析图表

这是 CatBoost 集成到其最新版本中的另一个独特功能。该功能提供计算和绘制的特定于要素的统计数据，并可视化 CatBoost 如何为每个要素拆分数据。更具体地说，统计数据是:

*   每个条柱(条柱对连续要素进行分组)或类别(当前仅支持一个热编码要素)的平均目标值。
*   每个箱的平均预测值
*   每个箱中数据实例(对象)的数量
*   各种特征值的预测

![Statistics of feature](img/5f172435b7f2ee46d2280c7582553cc6.png)

*Statistics of feature | Source: Author*

### CatBoost 参数

CatBoost 与 XGBoost 和 LightGBM 有共同的训练参数，但提供了一个非常灵活的参数调整界面。下表提供了三种增强算法提供的参数的快速比较。

| 功能 | CatBoost | XGBoost | LightGBM |
| --- | --- | --- | --- |
| 

控制过拟合的参数

 | 

–学习 _ 速率

–深度

–L2 _ reg

 | 

–学习 _ 速率

–最大 _ 深度

–最小 _ 孩子 _ 体重

 | 

–learning _ rate

–Max _ depth

–Num _ leaves

–min _ data _ in _ leaf

 |
| 

用于处理分类值的参数

 | 

–猫 _ 特性

–one _ hot _ max _ size

 |  |  |
| 

控制速度的参数

 |  | 

–col sample _ bytree

–子样本

–n _ estimates

 | 

–feature _ fraction

T5–bagging fraction

–num _ iterations

 |

此外，从下图可以明显看出，CatBoost 的默认参数提供了一个优秀的基线模型，比其他增强算法好得多。

你可以在这里阅读所有关于 CatBoost 的参数[。这些参数控制过度拟合、分类特征和速度。](https://web.archive.org/web/20230219051826/https://tech.yandex.com/catboost/doc/dg/concepts/parameter-tuning-docpage/)

![Log loss values (lower is better) for Classification mode. The percentage is metric difference measured against tuned CatBoost results.](img/c8bec76fa4c1a3f97d34a99b83dd9958.png)

*Log loss values (lower is better) for Classification mode. 
The percentage is metric difference measured against tuned CatBoost results | Source: Author*

其他有用的功能

### **过拟合检测器:** CatBoost 的算法结构抑制了梯度提升偏差和过拟合。此外，CatBoost 有一个过拟合检测器，如果发生过拟合，它可以在训练参数指示之前停止训练。CatBoost 使用两种策略实现过拟合检测:

*   Iter:考虑过拟合模型，使用具有最优度量值的迭代，在指定的迭代次数后停止训练。与其他梯度增强算法(如 LightGBM 和 XGBoost)一样，该策略使用 early_stopping_rounds 参数。
    *   IncToDec:当达到阈值时忽略过拟合检测器，并在用最佳度量值迭代后继续学习指定的迭代次数。通过在参数中设置“od_type”来激活过拟合检测器，以产生更一般化的模型。
    *   **缺失值支持** : CatBoost 为处理缺失值提供了三种固有的缺失值策略:
*   “禁止”:缺少的值被解释为错误，因为它们不受支持。
    *   “最小值”:缺失值被处理为所观察特征的最小值(小于所有其他值)。
    *   “最大值”:缺失值被视为所观察特征的最大值(大于所有其他值)。CatBoost 只有数值的缺失值插补，默认模式为最小。
    *   **CatBoost viewer** :除了 CatBoost 模型分析工具，CatBoost 还有一个[独立的可执行应用](https://web.archive.org/web/20230219051826/https://github.com/catboost/catboost-viewer)，用于在浏览器中绘制具有不同训练统计数据的图表。
*   **交叉验证** : CatBoost 允许对给定数据集进行交叉验证。在交叉验证模式中，训练数据被分成学习和评估两部分。
*   **社区支持** : CatBoost 有一个庞大且不断增长的开源社区，提供大量关于理论和应用的[教程](https://web.archive.org/web/20230219051826/https://github.com/catboost/tutorials)。
*   CatBoost 与 XGBoost 和 LightGBM:性能和速度的实际比较

## 前面几节介绍了 CatBoost 的一些特性，这些特性将作为选择 CatBoost 而不是 LightGBM 和 XGBoost 的有力标准。本节将有一个实践经验，我们将使用一个航班延误预测问题来比较性能和速度。

数据集和环境

### 该数据集包含 2015 年大型航空运营商运营的国内航班准点性能数据，由美国交通部(DOT)提供，可在 [Kaggle](https://web.archive.org/web/20230219051826/https://www.kaggle.com/usdot/flight-delays) 上找到。这种比较分析使用 CatBoost、LightGBM 和 XGBoost 探索和模拟具有可用独立特征的航班延误。该数据的一个子集(25%)用于建模，并且将使用 ROC AUC 评分评估各个生成的模型。在测量训练时间、预测时间和参数调整时间时，分析将涵盖默认和调整的设置。

为了便于比较，我们将使用 [neptune.ai](/web/20230219051826/https://neptune.ai/) ，这是一个用于 MLOps 的[元数据存储库](/web/20230219051826/https://neptune.ai/blog/ml-metadata-store)，为可能涉及大量实验的项目而构建。‌具体来说，我们将使用 neptune.ai 来:

所以，事不宜迟，我们开始吧！

通过 [Neptune-XGBoost 集成](https://web.archive.org/web/20230219051826/https://docs.neptune.ai/integrations/xgboost/)和 [Neptune-LightGBM 集成](https://web.archive.org/web/20230219051826/https://docs.neptune.ai/integrations/lightgbm/)，检查在使用 XGBoost 或 LightGBM 时如何跟踪您的模型构建元数据。

首先，我们必须安装所需的库。

导入已安装的库。

```py
!pip install catboost
!pip install xgboost
!pip install lgb
!pip install neptune-client
```

设置 Neptune 客户机来适当地记录项目的元数据。你可以在这里阅读更多相关信息[。](https://web.archive.org/web/20230219051826/https://docs.neptune.ai/setup/installation/)

```py
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import timeit
import pandas as pd
import numpy as np
import neptune.new as neptune

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
```

数据预处理和争论操作可以在[参考笔记本](https://web.archive.org/web/20230219051826/https://github.com/codebrain001/Catboost-tutorial/blob/main/Catboost_vs_LightGBM_vs_XGBoost.ipynb)中找到。我们将使用 30%的数据作为测试集。

```py
import neptune.new as neptune

run = neptune.init(project='<YOUR_WORKSPACE/YOUR_PROJECT>',
                  api_token='<YOUR_API_TOKEN>')
```

模型

```py
X = data_df.drop(columns=['ARRIVAL_DELAY'])
y = data_df['ARRIVAL_DELAY']

X_train, X_test,  y_train, y_test= train_test_split(X,y, random_state=2021, test_size=0.30)
```

### 接下来，让我们定义度量评估函数和模型执行函数。指标评估功能记录 ROC AUC 得分。

现在转到模型执行函数，它接受四个主要参数:

```py
def metrics(run, y_pred_test):
   score = roc_auc_score(y_test, y_pred_test)
   run['ROC AUC score'] = score
```

**模型:**生成的各个机器学习模型即 LightGBM、XGBoost 和 CatBoost

*   **描述:**模型执行实例的描述
*   **键:**键指定模型训练设置，尤其是要实现的分类特征参数
*   **cat_features:** 用作分类特征名称(对于 LightGBM)或索引(CatBoost)
*   该函数计算并记录元数据，包括描述、训练时间、预测时间和 ROC AUC 分数。

让我们在两种设置下运行相应模型的功能:

```py
def run_model(run, model, description, key, cat_features=''):
 if key =='LGB':

   run["Description"] = description

   start = timeit.default_timer()
   model.fit(X_train,y_train, categorical_feature=cat_features)
   stop = timeit.default_timer()
   run['Training time'] = stop - start

   start = timeit.default_timer()
   y_pred_test = model.predict(X_test)
   stop = timeit.default_timer()
   run['Prediction time'] = stop - start

   metrics(y_pred_test)

 elif key =='CAT':

   run["Description"] = description

   start = timeit.default_timer()
   model.fit(X_train,y_train,
             eval_set=(X_test, y_test),
             cat_features=cat_features,
             use_best_model=True)
   stop = timeit.default_timer()
   run['Training time'] = stop - start

   start = timeit.default_timer()
   y_pred_test = model.predict(X_test)
   stop = timeit.default_timer()
   run['Prediction time'] = stop - start

   metrics(y_pred_test)

 else:

   run["Description"] = description

   start = timeit.default_timer()
   model.fit(X_train,y_train)
   stop = timeit.default_timer()
   run['Training time'] = stop - start

   start = timeit.default_timer()
   y_pred_test = model.predict(X_test)
   stop = timeit.default_timer()
   run['Prediction time'] = stop - start

   metrics(y_pred_test)
```

1.CatBoost vs XGBoost vs LightGBM:默认超参数

### 可以在 Neptune 仪表板上查看基于 LightGBM、XGBoost 和 CatBoost 算法的默认设置的比较分析。

```py
model_lgb_def = lgb.LGBMClassifier()
run_model(model_lgb_def,'Default LightGBM without categorical support', key='LGB')

model_lgb_cat_def = lgb.LGBMClassifier()
run_model(model_lgb_cat_def, 'Default LightGBM with categorical support',key='LGB', cat_features=cat_cols)

model_xgb_def = xgb.XGBClassifier()
run_model(model_xgb_def, 'Default XGBoost', key='XGB')

model_cat_def = cb.CatBoostClassifier()
run_model(model_cat_def,'Default CatBoost without categorical support', key='CAT')

model_cat_cat_def = cb.CatBoostClassifier()
cat_features_index = [3,4,5]
run_model(model_cat_cat_def,'Default CatBoost with categorical support','CAT', cat_features_index)
```

结果:默认设置

#### 从仪表板上可以明显看出:

CatBoost 在没有分类支持的情况下具有最快的预测时间，因此在有分类支持的情况下显著增加。

*   对于测试数据，CatBoost 的 AUC 指标得分最高(AUC 得分越高，模型区分类别的性能越好)。
*   与 LightGBM 相比，XGBoost 在默认设置下的 ROC-AUC 得分最低，训练时间相对较长，但是其预测时间很快(在相应的默认设置运行中时间第二快)。
*   在训练时间上，LightGBM 的表现优于所有其他型号。
*   2.CatBoost vs XGBoost vs LightGBM:优化的超参数

### 以下是我们将在这次运行中使用的优化的超参数。三种算法中所选的参数非常相似:

“最大深度”和“深度”控制树模型的深度。

*   “learning_rate”说明了添加到树模型的修改量，并描述了模型学习的速度。
*   n_estimators 和 iterations 说明了树(轮)的数量，突出了增加迭代的数量。CatBoost‘l2 _ leaf _ reg’表示 L2 正则化系数，以阻止学习更复杂或灵活的模型，从而防止过拟合。
*   而 LightGBM num_leaves 参数对应于每棵树的最大叶子数，XGBoost‘min-child-weight’表示每个节点中所需的最小实例数。
*   这些参数被调整以控制过拟合和学习速度。

| LightBGM | XGBoost | CatBoost |  |
| --- | --- | --- | --- |
| 

max _ depth:7

learning _ rate:0.08

num _ leaves:100

n _ estimates:1000

 | 最大深度:7 | 最大深度:5 | 深度:10 |
|  |  |  |  |

超参数调谐部分可在[参考笔记本](https://web.archive.org/web/20230219051826/https://github.com/codebrain001/Catboost-tutorial/blob/main/Catboost_vs_LightGBM_vs_XGBoost.ipynb)中找到。

现在让我们用前面提到的调整设置运行这些模型。

同样，基于调优设置的比较分析可以在 Neptune 仪表盘上查看。

```py
params = {"max_depth": 7, "learning_rate" : 0.08, "num_leaves": 100,  "n_estimators": 1000}

model_lgb_tun = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metric='auc',**params)
run_model(model_lgb_tun, 'Tuned LightGBM without categorical support', 'LGB')

model_lgb_cat_tun = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metric='auc',**params)
run_model(model_lgb_cat_tun, 'Tuned LightGBM with categorical support', 'LGB', cat_cols)

params = {"max_depth": 5, "learning_rate": 0.8, "min_child_weight": 6,  "n_estimators": 1000}

model_xgb_tun = xgb.XGBClassifier(**params)
run_model(model_xgb_tun, 'Tuned XGBoost','XGB')

params = {"depth": 10, "learning_rate": 0.5, "iterations": 1000, "l2_leaf_reg": 5}

model_cat_tun = cb.CatBoostClassifier(**params)
run_model(model_cat_tun,'Tuned CatBoost without categorical support', key='CAT')

model_cat_cat_tun = cb.CatBoostClassifier(**params)
cat_features_index = [3,4,5]
run_model(model_cat_cat_tun,'Default CatBoost with categorical support','CAT', cat_features_index)

```

结果:调整设置

#### 从仪表板上可以明显看出:

在分类特征支持下，CatBoost 仍然保持了最快的预测时间和最佳的性能得分。CatBoost 对分类数据的内部识别允许它产生最慢的训练时间。

*   尽管进行了超参数调优，但默认结果和调优结果之间的差异并不大，这也凸显了 CatBoost 的默认设置会产生很好的结果这一事实。
*   XGBoost 性能随着调整的设置而提高，但是，它产生了第四好的 AUC-ROC 分数，并且训练时间和预测时间变得更差。
*   LightGBM 仍然拥有最快的训练时间和最快的参数调整时间。然而，如果你愿意在更快的训练时间和性能之间做出权衡，CatBoost 将是一个很好的选择。
*   结论

## CatBoost 的算法设计可能类似于“老”一代 GBDT 模型，但是，它有一些关键属性，如:

排序目标函数

*   自然分类特征预处理
*   模型分析
*   最快预测时间
*   CatBoost 还提供了巨大的性能潜力，因为它在缺省参数下表现出色，在调优时显著提高了性能。本文旨在通过讨论这些关键特性及其提供的优势，帮助您决定何时选择 CatBoost 而不是 LightGBM 或 XGBoost。我希望现在你对此有一个很好的想法，下次你面临这样的选择时，你将能够做出明智的决定。

如果你想更深入地了解这一切，下面的链接将帮助你做到这一点。暂时就这样吧！

If you would like to get a deeper look inside all of this, the following links will help you to do just that. That’s all for now!*