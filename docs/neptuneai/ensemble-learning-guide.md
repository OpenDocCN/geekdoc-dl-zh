# 集成学习综合指南:你到底需要知道什么

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/ensemble-learning-guide>

[集成学习技术](https://web.archive.org/web/20221115204110/https://www.toptal.com/machine-learning/ensemble-methods-machine-learning)已经被证明可以在机器学习问题上产生更好的性能。我们可以将这些技术用于回归和分类问题。

这些集合技术的最终预测是通过组合几个基本模型的结果获得的。平均、投票和堆叠是组合结果以获得最终预测的一些方式。

在本文中，我们将探索如何使用集成学习来提出最佳的机器学习模型。

## 什么是集成学习？

集成学习是一个问题中几个机器学习模型的组合。这些模型被称为弱学习者。直觉是，当你把几个弱学习者结合起来，他们就能成为强学习者。

每个弱学习器适合训练集，并提供获得的预测。通过组合来自所有弱学习器的结果来计算最终预测结果。

### 基本集成学习技术

让我们花点时间看看简单的集成学习技术。

#### 最大投票

在分类中，来自每个模型的预测是一个投票。在 max 投票中，最终的预测来自票数最多的预测。

让我们举一个例子，其中有三个分类器，它们具有以下预测:

*   分类器 1–A 类
*   分类器 2–B 类
*   分类器 3–B 类

最后的预测是 B 类，因为它拥有最多的票数。

#### 求平均值

在平均中，最终输出是所有预测的平均值。这适用于回归问题。例如，在[随机森林回归](/web/20221115204110/https://neptune.ai/blog/random-forest-regression-when-does-it-fail-and-why)中，最终结果是各个决策树预测的平均值。

让我们以预测商品价格的三个回归模型为例，如下所示:

*   回归变量 1–200
*   回归变量 2–300
*   回归变量 3–400

最终预测将是 200、300 和 400 的平均值。

#### 加权平均值

在加权平均中，具有更高预测能力的基础模型更重要。在价格预测示例中，将为每个回归变量分配一个权重。

权重之和等于 1。假设回归变量的权重分别为 0.35、0.45 和 0.2。最终模型预测可以计算如下:

0.35 * 200 + 0.45*300 + 0.2*400 = 285

## 高级集成学习技术

以上是简单的技术，现在让我们来看看集成学习的高级技术。

### 堆垛

叠加是组合各种估计量以减少其偏差的过程。来自每个估计器的预测被堆叠在一起，并被用作计算最终预测的最终估计器(通常称为*元模型*)的输入。最终评估者的训练通过交叉验证进行。

回归和分类问题都可以进行叠加。

可以认为堆叠发生在以下步骤中:

1.  将数据分成训练集和验证集，
2.  将训练集分成 K 个折叠，例如 10 个，
3.  在 9 个折叠上训练一个基础模型(比如说 SVM)并且在第 10 个折叠上进行预测，
4.  重复直到你对每个折叠都有一个预测，
5.  在整个训练集上拟合基础模型，
6.  使用该模型对测试集进行预测，
7.  对其他基础模型(例如决策树)重复步骤 3–6，
8.  使用来自测试集的预测作为新模型的特征—*元模型，*
9.  使用元模型对测试集进行最终预测。

对于回归问题，传递给元模型的值是数字的。对于分类问题，它们是概率或类别标签。

### 混合

混合类似于堆叠，但是使用定型集中的维持集来进行预测。因此，只对维持集进行预测。预测和维持集用于构建最终模型，该模型对测试集进行预测。

您可以将混合视为一种堆叠，其中元模型是根据基础模型对拒绝验证集所做的预测来训练的。

您可以将*混合*过程视为:

*   将数据分成测试和验证集，
*   根据验证集拟合基础模型，
*   对验证和测试集进行预测，
*   使用验证集及其预测来构建最终模型，
*   使用该模型进行最终预测。

混合的概念因 T2 Netflix 有奖竞赛而变得流行起来。获胜团队使用混合解决方案将网飞电影推荐算法的性能提高了 10 倍。

根据本 [Kaggle 组装指南](https://web.archive.org/web/20221115204110/https://mlwave.com/kaggle-ensembling-guide/):

> “混合是由网飞获奖者引入的一个词。这非常接近于堆叠概括，但是更简单，信息泄露的风险更小。一些研究人员交替使用“堆叠组合”和“混合”。
> 
> 使用混合，不是为训练集创建超出折叠的预测，而是创建一个小的维持集，比如说训练集的 10%。然后，堆叠器模型仅在此维持集上训练。"

### 混合与堆叠

混合比堆叠简单，可以防止模型中的信息泄漏。概化器和堆栈器使用不同的数据集。但是，混合使用的数据较少，可能会导致过度拟合。

交叉验证在堆叠上比混合更可靠。与在混合中使用小的保留数据集相比，它是在更多的折叠上计算的。

### 制袋材料

Bagging 随机抽取数据样本，构建学习算法，并使用平均值来计算 bagging 概率。它也被称为*引导聚合*。Bagging 汇总了几个模型的结果，以获得一个概括的结果。

该方法包括:

*   用替换从原始数据集创建多个子集，
*   为每个子集建立基础模型，
*   并行运行所有的模型，
*   组合所有模型的预测以获得最终预测。

### 助推

Boosting 是一种机器学习集成技术，通过将弱学习者转换为强学习者来减少偏差和方差。弱学习器以连续的方式应用于数据集。第一步是建立一个初始模型，并使其适合训练集。

然后拟合试图修正由第一模型产生的误差的第二模型。整个过程如下所示:

*   从原始数据创建一个子集，
*   用这些数据建立一个初始模型，
*   对整个数据集进行预测，
*   使用预测值和实际值计算误差，
*   给不正确的预测分配更多的权重，
*   创建另一个模型，尝试修复上一个模型的错误，
*   使用新模型对整个数据集运行预测，
*   创建多个模型，每个模型旨在纠正前一个模型产生的错误，
*   通过加权所有模型的平均值获得最终模型。

## 集成学习图书馆

介绍完之后，让我们来讨论一下可以用于集成的库。概括地说，有两类:

*   打包算法，
*   推进算法。

### 打包算法

装袋算法基于上述装袋技术。让我们来看看其中的几个。

#### Bagging 元估计量

Scikit-learn 让我们实现一个`[打包分类器](https://web.archive.org/web/20221115204110/https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)和一个`[打包分类器](https://web.archive.org/web/20221115204110/https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)`。bagging 元估计器根据原始数据集的随机子集拟合每个基本模型。然后，它通过聚合各个基础模型预测来计算最终预测。聚合是通过投票或平均来完成的。该方法通过在构造过程中引入随机化来减少估计量的方差。

有几种装袋方式:

*   将数据的随机子集绘制为样本的随机子集被称为*粘贴*。**
*   当用替换抽取样本时，该算法被称为*装袋*。
*   如果随机数据子集被视为特征的随机子集，则该算法被称为*随机子空间*。
*   当您从样本和特征的子集创建基本估计量时，它是*随机补丁*。

让我们看看如何使用 Scikit-learn 创建 bagging 估计器。

这需要几个步骤:

*   导入“打包分类器”,
*   导入一个基础评估器——决策树分类器，
*   创建“BaggingClassifier”的实例。

```py
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10, max_samples=0.5, max_features=0.5)
```

bagging 分类器有几个参数:

*   基础估计器——这里是一个决策树分类器，
*   集合中你想要的估计数，
*   “max_samples”定义将从每个基本估计量的训练集中抽取的样本数，
*   “max_features ”,以指示将用于训练每个基本估计量的特征的数量。

接下来，您可以在训练集上拟合这个分类器，并对其进行评分。

```py
bagging.fit(X_train, y_train)
bagging.score(X_test,y_test)
```

回归问题的过程是一样的，唯一的不同是你将使用回归估计器。

```py
from sklearn.ensemble import BaggingRegressor
bagging = BaggingRegressor(DecisionTreeRegressor())
bagging.fit(X_train, y_train)
model.score(X_test,y_test)
```

#### 随机树木的森林

[随机森林](/web/20221115204110/https://neptune.ai/blog/random-forest-regression-when-does-it-fail-and-why)是随机决策树的集合。每个决策树都是从数据集的不同样本中创建的。样品是替换抽取的。每棵树都有自己的预测。

在回归中，这些结果被平均以获得最终结果。

在分类中，最终结果可以作为得票最多的类别。

平均和表决通过防止过拟合来提高模型的准确性。

在 Scikit-learn 中，可以通过“RandomForestClassifier”和“ExtraTreesClassifier”实现随机树的森林。回归问题也有类似的估计。

```py
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
clf = RandomForestClassifier(n_estimators=10, max_depth=None,  min_samples_split=2, random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)
```

### 助推算法

这些算法基于前面描述的 boosting 框架。让我们来看看其中的几个。

#### adaboost 算法

AdaBoost 通过拟合一系列弱学习者来工作。它在随后的迭代中给不正确的预测更多的权重，给正确的预测更少的权重。这迫使算法关注更难预测的观察结果。最终的预测来自于对多数票或总数的权衡。

AdaBoost 可用于回归和分类问题。让我们花点时间看看如何使用 Scikit-learn 将该算法应用于分类问题。

我们使用“AdaBoostClassifier”。‘n _ estimators’表示集合中弱学习者的数量。每个弱学习者对最终组合的贡献由“learning_rate”控制。

默认情况下，决策树被用作基本估计器。为了获得更好的结果，可以调整决策树的参数。您还可以调整基本估计量的数量。

```py
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=100)
model.fit(X_train, y_train)
model.score(X_test,y_test)
```

#### 梯度树提升

梯度树提升还结合了一组弱学习器来形成强学习器。就梯度推进树而言，有三个主要事项需要注意:

*   必须使用微分损失函数，
*   决策树被用作弱学习器，
*   这是一个添加模型，所以树是一个接一个添加的。梯度下降用于在添加后续树时最小化损失。

可以使用 Scikit-learn 建立一个基于梯度树提升的模型。

```py
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model.fit(X_train, y_train)
model.score(X_test,y_test)
```

#### 极端梯度推进

极限梯度提升(eXtreme Gradient Boosting)，俗称 [XGBoost](/web/20221115204110/https://neptune.ai/blog/how-to-organize-your-xgboost-machine-learning-ml-model-development-process) ，是一个顶级的梯度提升框架。它基于弱决策树的集合。它可以在单台计算机上进行并行计算。

该算法对基础学习者使用回归树。它还内置了交叉验证。开发人员喜欢它的准确性、效率和可行性。

```py
import xgboost as xgb
params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)
model.fit(X_train, y_train)
model.score(X_test,y_test)
```

#### LightGBM

[LightGBM](https://web.archive.org/web/20221115204110/https://lightgbm.readthedocs.io/en/latest/) 是一种基于树学习的梯度推进算法。与其他使用深度生长的基于树的算法不同，LightGBM 使用叶子生长。逐叶增长算法往往比基于 dep 的算法收敛得更快。

通过设置适当的目标，LightGBM 可以用于回归和分类问题。

以下是如何将 LightGBM 应用于二进制分类问题。

```py
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {'boosting_type': 'gbdt',
              'objective': 'binary',
              'num_leaves': 40,
              'learning_rate': 0.1,
              'feature_fraction': 0.9
              }
gbm = lgb.train(params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=['train','valid'],
   )
```

#### CatBoost

[CatBoost](https://web.archive.org/web/20221115204110/https://github.com/catboost) 是由 [Yandex](https://web.archive.org/web/20221115204110/https://yandex.com/company/) 开发的深度梯度增强库。它使用遗忘决策树生成一棵平衡树。正如您在下图中看到的，在每个级别进行左右分割时使用了相同的功能。

研究人员需要 Catboost 的原因如下:

*   本机处理分类特征的能力，
*   模型可以在几个 GPU 上训练，
*   它通过使用默认参数提供很好的结果来减少参数调整时间，
*   可以将模型导出到 Core ML 用于设备上推理(iOS)，
*   它在内部处理丢失的值，
*   它可用于回归和分类问题。

以下是如何将 CatBoost 应用于分类问题的方法。

```py
from catboost import CatBoostClassifier
cat = CatBoostClassifier()
cat.fit(X_train,y_train,verbose=False, plot=True
```

## 帮助您在基础模型上进行堆叠的库

当叠加时，单个模型的输出被叠加，并且最终估计器被用于计算最终预测。估计量适用于整个训练集。最终估计量是根据基本估计量的交叉验证预测训练的。

Scikit-learn 可用于堆叠估计量。让我们来看看如何对分类问题的估计量进行叠加。

首先，您需要设置您想要使用的基本估计量。

```py
estimators = [
  ('knn', KNeighborsClassifier()),
   ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
   ('svr', LinearSVC(random_state=42))
]
```

接下来，实例化堆叠分类器。其参数包括:

*   上面定义的估计量，
*   你想用的最后一个估计量。默认情况下使用逻辑回归估计量，
*   ` cv '交叉验证生成器。默认情况下使用 5 k 倍交叉验证，
*   “stack_method ”,规定应用于每个估算器的方法。如果为“auto ”,它将依次尝试“predict_proba”、“decision_function”或“predict”。

```py
from sklearn.ensemble import StackingClassifier
clf = StackingClassifier(
 estimators=estimators, final_estimator=LogisticRegression()
 )
```

之后，您可以将数据拟合到训练集，并在测试集上对其进行评分。

```py
clf.fit(X_train, y_train)
clf.score(X_test,y_test)
```

Scikit-learn 还允许您实现投票估计器。它使用多数投票或基本估计值的概率平均值来进行最终预测。

这可以使用分类问题的“VotingClassifier”和回归问题的“VotingRegressor”来实现。就像堆叠一样，你首先必须定义一组基本估计量。

让我们看看如何实现它来解决分类问题。“投票分类器”允许您选择投票类型:

*   ‘软’意味着概率的平均值将被用于计算最终结果，
*   ` hard '通知分类器使用预测类别进行多数表决。

```py
from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(
    estimators=estimators,
    voting='soft')
```

投票回归器使用几个估计量，并将最终结果作为预测值的平均值返回。

### 使用 Mlxtend 堆叠

您还可以使用 [Mlxtend 的](https://web.archive.org/web/20221115204110/http://rasbt.github.io/mlxtend/)`[StackingCVClassifier](https://web.archive.org/web/20221115204110/http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/)进行堆叠。第一步是定义一个基本估计量列表，然后将估计量传递给分类器。

您还必须定义将用于汇总预测的最终模型。在这种情况下，它是逻辑回归模型。

```py
knn = KNeighborsClassifier(n_neighbors=1)
rf = RandomForestClassifier(random_state=1)
gnb = GaussianNB()
lr = LogisticRegression()
estimators = [knn,gnb,rf,lr]
stack = StackingCVClassifier(classifiers = estimators,
                            shuffle = False,
use_probas = True,
cv = 5, 
meta_classifier = LogisticRegression())

```

## 何时使用集成学习

当您想要提高机器学习模型的性能时，可以采用集成学习技术。例如增加分类模型的准确性或减少回归模型的平均绝对误差。集合也导致更稳定的模型。

当您的模型过度适应训练集时，您还可以采用集成学习方法来创建更复杂的模型。然后，集合中的模型将通过组合它们的预测来提高数据集的性能。

## 当集成学习效果最好时

当基本模型不相关时，集成学习效果最好。例如，您可以在不同的数据集或要素上训练不同的模型，如线性模型、决策树和神经网络。基础模型越不相关越好。

使用不相关模型背后的想法是，每一个都可能解决另一个的弱点。它们还具有不同的强度，当组合时，将产生性能良好的估计器。例如，创建仅基于树的模型的集合可能不如将树型算法与其他类型的算法相结合有效。

## 最后的想法

在本文中，我们探索了如何使用集成学习来提高机器学习模型的性能。我们还学习了各种工具和技术，您可以使用它们进行组装。你的机器学习技能有望增长。

组装愉快！

## 资源

### 德里克·姆维蒂

Derrick Mwiti 是一名数据科学家，他对分享知识充满热情。他是数据科学社区的热心贡献者，例如 Heartbeat、Towards Data Science、Datacamp、Neptune AI、KDnuggets 等博客。他的内容在网上被浏览了超过一百万次。德里克也是一名作家和在线教师。他还培训各种机构并与之合作，以实施数据科学解决方案并提升其员工的技能。你可能想看看他在 Python 课程中完整的数据科学和机器学习训练营。

* * *

**阅读下一篇**

## 机器学习中模型评估和选择的最终指南

10 分钟阅读|作者 Samadrita Ghosh |年 7 月 16 日更新

在高层次上，机器学习是统计和计算的结合。机器学习的关键围绕着算法或模型的概念，这些概念实际上是类固醇的统计估计。

然而，根据数据分布的不同，任何给定的模型都有一些限制。它们中没有一个是完全准确的，因为它们只是 ***(即使使用类固醇)*** 。这些限制俗称 ***偏差*** 和 ***方差*** 。

具有高偏差的**模型会因为不太注意训练点而过于简化(例如:在线性回归中，不管数据分布如何，模型将总是假设线性关系)。**

具有高方差的**模型将通过不对其之前未见过的测试点进行概括来将其自身限制于训练数据(例如:max_depth = None 的随机森林)。**

当限制很微妙时，问题就出现了，比如当我们必须在随机森林算法和梯度推进算法之间进行选择，或者在同一决策树算法的两个变体之间进行选择。两者都趋向于具有高方差和低偏差。

这就是模型选择和模型评估发挥作用的地方！

在本文中，我们将讨论:

*   什么是模型选择和模型评估？
*   有效的模型选择方法(重采样和概率方法)
*   流行的模型评估方法
*   重要的机器学习模型权衡

[Continue reading ->](/web/20221115204110/https://neptune.ai/blog/the-ultimate-guide-to-evaluation-and-selection-of-models-in-machine-learning)

* * *