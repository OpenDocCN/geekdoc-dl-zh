# 机器学习中的交叉验证:如何正确进行

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right>

在机器学习(ML)中，泛化通常指的是算法在各种输入中有效的能力。这意味着 [**ML** 模型](https://web.archive.org/web/20221225163138/https://towardsdatascience.com/all-machine-learning-models-explained-in-6-minutes-9fe30ff6776a)不会遇到来自相同训练数据分布的新输入的性能下降。

对人类来说，概括是最自然不过的事情了。我们可以即时分类。例如，我们肯定会认出一只狗，即使我们以前没有见过这个品种。然而，对于 ML 模型来说，这可能是一个相当大的挑战。这就是为什么在构建模型时，检查算法的泛化能力是一项需要大量注意力的重要任务。

为此，我们使用 [**交叉验证**](https://web.archive.org/web/20221225163138/https://scikit-learn.org/stable/modules/cross_validation.html) ( **CV** )。

在本文中，我们将涵盖:

*   什么是交叉验证:定义、使用目的和技术
*   不同的 CV 技术:保持、k 折叠、留一、留 p、分层 k 折叠、重复 k 折叠、嵌套 k 折叠、时间序列 CV
*   如何使用这些技术:sklearn
*   机器学习中的交叉验证:sklearn，CatBoost
*   深度学习中的交叉验证:Keras，PyTorch，MxNet
*   最佳实践和提示:时间序列、医疗和财务数据、图像

## 什么是交叉验证？

**交叉验证**是一种评估机器学习模型并测试其性能的技术。CV 常用于应用型 ML 任务。它有助于为特定的预测建模问题比较和选择合适的模型。

CV 易于理解，易于实施，并且与用于计算模型效率得分的其他方法相比，它倾向于具有较低的偏差。所有这些使得交叉验证成为为特定任务选择最佳模型的有力工具。

有很多不同的技术可以用来**交叉验证**模型。尽管如此，它们都有一个相似的算法:

1.  将数据集分成两部分:一部分用于训练，另一部分用于测试
2.  在训练集上训练模型
3.  在测试集上验证模型
4.  重复 1-3 步几次。该数字取决于您正在使用的CV 方法

你可能知道，有很多简历技巧。有些是常用的，有些只是理论上的。让我们来看看本文将涉及的交叉验证方法。

*   坚持
*   k 倍
*   留一个出来
*   漏接
*   分层 K 折叠
*   重复 K 倍
*   嵌套 K 折叠
*   时间序列 CV

## 保留交叉验证

**保留交叉验证**是最简单和最常见的技术。你可能不知道这是一种**坚持**的方法，但你肯定每天都在用。

保持技术的算法；

1.  将数据集分为两部分:训练集和测试集。通常，数据集的 80%用于训练集，20%用于测试集，但是您可以选择更适合您的任何拆分
2.  在训练集上训练模型
3.  在测试集上验证
4.  保存验证的结果

就是这样。

我们通常在大型数据集上使用 hold-out 方法，因为它只需要训练模型一次。

实施 hold-out 真的很容易。例如，您可以使用 sk learn . model _ selection . train _ test _ split 来完成。

```py
import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5, 2)), range(5)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=111)
```

尽管如此，拒绝合作还是有一个很大的缺点。

例如，分布不完全均匀的数据集。如果是这样的话，我们可能会在分拆后陷入困境。例如，训练集不会代表测试集。训练集和测试集可能差别很大，其中一个可能更容易或更难。

此外，我们只测试我们的模型一次的事实可能是这种方法的瓶颈。由于前面提到的原因，通过保持技术获得的结果可能被认为是不准确的。

## k 倍交叉验证

**k 倍交叉验证**是一种最大限度减少拒绝法缺点的技术。k-Fold 引入了一种分割数据集的新方法，有助于克服“只测试一次的瓶颈”。

k 倍技术的算法:

1.  选择折叠数–k。通常，k 为 5 或 10，但您可以选择小于数据集长度的任何数字。
2.  将数据集分成 k 个相等(如果可能)的部分(它们被称为折叠)
3.  选择 k–1 折叠作为训练集。剩余的折叠将是测试集
4.  在训练集上训练模型。在交叉验证的每次迭代中，您必须独立于前一次迭代中训练的模型来训练新的模型
5.  在测试集上验证
6.  保存验证的结果
7.  重复步骤 3-6k 次。每次使用剩余的折叠作为测试集。最后，你应该在你的每一次折叠中验证这个模型。
8.  要获得最终分数，请对第 6 步中得到的结果进行平均。

要执行 k-Fold 交叉验证，可以使用 sklearn.model_selection.KFold。

```py
import numpy as np
from sklearn.model_selection import KFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

```

总的来说，使用 k 折技术总比持有要好。在面对面的情况下，比较 k 倍给出了更稳定和可信的结果，因为训练和测试是在数据集的几个不同部分上执行的。如果我们增加折叠次数，在许多不同的子数据集上测试模型，我们可以使总得分更加稳健。

尽管如此，k-Fold 方法有一个缺点。增加 k 导致训练更多的模型，并且训练过程可能非常昂贵和耗时。

## 留一交叉验证

**留一法сRoss-validation**(**LOOCV**)是 **k 倍 CV** 的极端案例。想象一下如果 **k** 等于 **n** 其中 n 是数据集中样本的数量。这种 **k 折**情况相当于**留一法**技术。

LOOCV 技术的算法:

1.  从数据集中选择一个样本作为测试集
2.  剩余的 n-1 个样本将成为训练集
3.  在训练集上训练模型。在每次迭代中，必须训练一个新的模型
4.  在测试集上验证
5.  保存验证的结果
6.  重复步骤 1-5n 次，因为对于 n 个样本，我们有 n 个不同的训练和测试集
7.  要获得最终分数，请对第 5 步中得到的结果进行平均。

对于 LOOCV，sklearn 也有一个内置的方法。可以在 model_selection 库–sk learn . model _ selection . leave one out 中找到。

```py
import numpy as np
from sklearn.model_selection import LeaveOneOut

X = np.array([[1, 2], [3, 4]])
y = np.array([1, 2])
loo = LeaveOneOut()

for train_index, test_index in loo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

留一交叉验证的最大优点是不会浪费太多数据。我们只使用整个数据集中的一个样本作为测试集，而其余的是训练集。但与 k-Fold CV 相比，LOOCV 需要建立 n 个模型而不是 k 个模型，因为我们知道代表数据集中样本数量的 n 比 k 大得多。这意味着 LOOCV 比 k-Fold 的计算成本更高，使用 LOOCV 交叉验证模型可能需要大量时间。

因此，数据科学界有一个基于经验证据和不同研究的一般规则，这表明 5 倍或 10 倍交叉验证应优于 LOOCV。

## 遗漏交叉验证

***留一法交叉验证** ( **LpOC** )类似于**留一法 CV** ，它通过使用 **p** 样本作为测试集来创建所有可能的训练和测试集。所有提到的关于 **LOOCV** 的都是真的，对于 **LpOC** 。*

不过，值得一提的是，与 LOOCV 不同，如果 p 大于 1，LpOC 的 k 倍测试集将重叠。

LpOC 技术的算法；

1.  从将成为测试集的数据集中选择 p 个样本
2.  剩余的 n–p 个样本将成为训练集
3.  在训练集上训练模型。在每次迭代中，必须训练一个新的模型
4.  在测试集上验证
5.  保存验证的结果
6.  重复步骤 2–5c[p]n 次
7.  要获得最终分数，请对第 5 步中得到的结果进行平均

您可以使用 sk learn–sk learn . model _ selection . Leave pout 执行 Leave-p-out CV。

```py
import numpy as np
from sklearn.model_selection import LeavePOut

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])
lpo = LeavePOut(2)

for train_index, test_index in lpo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

```

LpOC 具有 LOOCV 的所有缺点，但无论如何，它和 LOOCV 一样强大。

## 分层 k 倍交叉验证

有时，我们可能会面临数据集中目标值的巨大不平衡。例如，在关于手表价格的数据集中，可能有大量价格高的手表。在分类的情况下，在猫和狗的数据集中，可能有很大的向狗类的转移。

**分层 k-Fold** 是标准 **k-Fold CV** 技术的一种变体，该技术被设计成在目标不平衡的情况下有效。

它的工作原理如下。分层 k 折叠将数据集分成 k 个折叠，使得每个折叠包含与完整集大致相同百分比的每个目标类的样本。在回归的情况下，分层 k-Fold 确保平均目标值在所有 Fold 中大致相等。

分层 k 折叠技术的算法；

1.  选择一些折叠–k
2.  将数据集分割成 k 个折叠。每个文件夹必须包含与完整集大致相同百分比的每个目标类别的样本
3.  选择 k–1 个折叠作为训练集。剩余的折叠将是测试集
4.  在训练集上训练模型。在每次迭代中，必须训练一个新的模型
5.  在测试集上验证
6.  保存验证的结果
7.  重复步骤 3-6k 次。每次使用剩余的折叠作为测试集。最后，你应该在你的每一次折叠中验证这个模型。
8.  要获得最终分数，请对第 6 步中得到的结果进行平均。

您可能已经注意到，分层 k 折叠技术的算法类似于标准 k 折叠。您不需要额外编写代码，因为该方法会为您做所有必要的事情。

分层 k-Fold 在 sklearn 中也有一个内置的方法——sk learn . model _ selection . stratifiedkfold。

```py
import numpy as np
from sklearn.model_selection import StratifiedKFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

```

上面提到的所有关于 k-Fold CV 对于分层 k-Fold 技术都是正确的。在选择不同的 CV 方法时，请确保您使用的是正确的方法。例如，您可能会认为您的模型性能很差，仅仅是因为您使用 k-Fold CV 来验证在具有[类不平衡](/web/20221225163138/https://neptune.ai/blog/how-to-deal-with-imbalanced-classification-and-regression-data)的数据集上训练的模型。为了避免这种情况，您应该始终对您的数据进行适当的探索性数据分析。

## 重复 k 倍交叉验证

**重复 k 倍交叉验证**或**重复随机子抽样 CV** 可能是本文所有 **CV** 技术中最稳健的。它是 **k-Fold** 的变体，但是在**重复 k-Fold****的情况下，k** 不是折叠的次数。这是我们训练模型的次数。

总的想法是，在每次迭代中，我们将在整个数据集中随机选择样本作为我们的测试集。例如，如果我们决定数据集的 20%将成为我们的测试集，那么将随机选择 20%的样本，剩下的 80%将成为训练集。

重复 k 倍技术的算法；

1.  选择 k–模型将被训练的次数
2.  挑选一些样本作为测试集
3.  分割数据集
4.  在训练器械上进行训练。在交叉验证的每次迭代中，必须训练一个新的模型
5.  在测试集上验证
6.  保存验证的结果
7.  重复步骤 3-6 k 次
8.  要获得最终分数，请对第 6 步中得到的结果进行平均。

重复 k 倍比标准 k 倍 CV 有明显的优势。首先，训练/测试分割的比例不依赖于迭代的次数。其次，我们甚至可以为每次迭代设置唯一的比例。第三，从数据集中随机选择样本使得重复 k-Fold 对选择偏差更加鲁棒。

尽管如此，还是有一些缺点。k-Fold CV 保证模型将在所有样本上进行测试，而重复 k-Fold 基于随机化，这意味着一些样本可能永远不会被选入测试集。同时，一些样本可能被选择多次。因此对于不平衡的数据集来说，这是一个糟糕的选择。

Sklearn 会帮你实现一个重复的 k 倍 CV。只需使用 sklearn . model _ selection . repeated kfold。在 sk learn 实现该技术时，您必须设置想要的折叠次数(n_splits)和将要执行的分割次数(n_repeats)。它保证您在每次迭代中都有不同的折叠。

```py
import numpy as np
from sklearn.model_selection import RepeatedKFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)

for train_index, test_index in rkf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

```

## 嵌套 k 折叠

与旨在评估算法质量的其他 CV 技术不同，**嵌套 k 重 CV** 用于训练模型，其中超参数也需要优化。它估计基础模型及其(超)参数搜索的泛化误差。

![Nested k-Fold ](img/ff7b2c5f405752272981a059822aeebe.png)

*Nested k-Fold cross-validation resampling | [Source](https://web.archive.org/web/20221225163138/https://stats.stackexchange.com/questions/292179/whats-the-meaning-of-nested-resampling)*

嵌套 k 折叠技术的算法；

1.  为当前模型定义一组超参数组合 C。如果模型没有超参数，C 就是空集。
2.  将数据分成 K 份，病例和对照的分布大致相等。
3.  (外循环)对于折叠 K，在 K 折叠中:
    1.  将折叠 k 设置为测试集。
    2.  对剩余的 K-1 个折叠执行自动特征选择。
    3.  对于 C 中的参数组合 C:
        1.  (内部循环)对于折叠 K，在剩余的 K-1 个折叠中:
            1.  将折叠 k 设置为验证集。
            2.  剩余 K-2 褶皱上的火车模型。
            3.  在折叠 k 上评估模型性能
        2.  计算参数组合 c 的 K-2 倍平均性能。
    4.  使用超参数组合在 K-1 个折叠上训练模型，该超参数组合在内部循环的所有步骤上产生最佳平均性能。
    5.  在折叠 k 上评估模型性能
4.  计算 K 倍的平均性能。

内部循环执行交叉验证，以使用外部循环的每次迭代中可用的 k-1 个数据折叠来识别最佳特征和模型超参数。该模型为每个外部循环步骤训练一次，并在保留的数据折叠上进行评估。这个过程产生模型性能的 k 个评估，每个数据折叠一个，并允许在每个样本上测试模型。

要注意的是，这种技术在计算上是昂贵的，因为大量的模型被训练和评估。不幸的是，sklearn 中没有内置的方法可以为您执行嵌套的 k-Fold CV。

可以自己实现，也可以参考这里的实现[。](https://web.archive.org/web/20221225163138/https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/)

## 时间序列交叉验证

传统的交叉验证技术不适用于时序数据，因为我们不能选择随机数据点并将它们分配给测试集或训练集，因为使用未来的值来预测过去的值是没有意义的。这主要有两种方法:

1.  **滚动交叉验证**

交叉验证是在滚动的基础上进行的，即从用于训练目的的一小部分数据开始，预测未来值，然后检查预测数据点的准确性。下图可以帮助你获得这种方法背后的直觉。

![Time-Series Cross-Validation](img/9a7104d0e5d8d31fbad3dc952a1c9e00.png)

*Rolling cross-validation | [Source](https://web.archive.org/web/20221225163138/https://goldinlocks.github.io/Time-Series-Cross-Validation/#:~:text=Blocked%20and%20Time%20Series%20Split,and%20another%20as%20a%20response.)*

2.  **交叉验证受阻**

第一种技术可能会将未来数据泄漏到模型中。该模型将观察未来的模式进行预测，并试图记住它们。这就是引入阻塞交叉验证的原因。

![Time-Series Cross-Validation](img/3e6f30151071cba62737fd67fefba9fd.png)

*Blocked cross-validation | [Source](https://web.archive.org/web/20221225163138/https://goldinlocks.github.io/Time-Series-Cross-Validation/#:~:text=Blocked%20and%20Time%20Series%20Split,and%20another%20as%20a%20response.)*

它的工作原理是在两个位置增加边距。第一个是在训练和验证折叠之间，以防止模型观察到两次使用的滞后值，一次作为回归变量，另一次作为响应。第二个是在每次迭代中使用的折叠之间，以防止模型从一次迭代到下一次迭代记住模式。

## 机器学习中的交叉验证

什么时候交叉验证是正确的选择？

尽管对你训练过的模型进行交叉验证从来都不是一个坏的选择，**在某些情况下，交叉验证是绝对必要的:**

1.  **有限数据集**

假设我们有 100 个数据点，我们正在处理一个有 10 个类的多类分类问题，平均每个类有 10 个例子。在 80-20 的训练测试划分中，这个数字甚至会进一步下降到每类 8 个样本用于训练。这里明智的做法是使用交叉验证，并利用我们的整个数据集进行训练和测试。

2.  **相关数据点**

当我们对数据进行随机训练测试分割时，我们假设我们的例子是独立的。这意味着知道一些实例不会帮助我们理解其他实例。然而，情况并不总是这样，在这种情况下，我们的模型熟悉整个数据集是很重要的，这可以进行交叉验证。

3.  **单一指标的缺点**

在没有交叉验证的情况下，我们只能得到一个单一的准确度或精确度值，或者回忆，这可能是一个偶然的结果。当我们训练多个模型时，我们消除了这种可能性，并获得了每个模型的指标，从而产生了强大的洞察力。

4.  **超参数调谐**

虽然有很多[方法可以调优你的模型的超参数](/web/20221225163138/https://neptune.ai/blog/hyperparameter-tuning-in-python-complete-guide)比如网格搜索，贝叶斯优化等等。这个练习不能在训练集或测试集上完成，因此需要一个验证集。因此，我们退回到上面讨论过的同样的分裂问题，交叉验证可以帮助我们解决这个问题。

## 深度学习中的交叉验证

深度学习(DL)中的交叉验证可能有点棘手，因为大多数 CV 技术需要至少训练模型几次。

在深度学习中，你通常会试图避免 CV，因为与训练 k 个不同模型相关联的成本。你可以使用训练数据的随机子集作为验证的依据，而不是使用 k-Fold 或其他 CV 技术。

例如， [**Keras**](https://web.archive.org/web/20221225163138/http://keras.io/models/) 深度学习库允许你为执行训练的 fit 函数传递两个参数中的一个。

1.  **validation_split** :应该等待验证的数据的百分比
2.  **validation_data** :应该用于验证的(X，y)元组。该参数覆盖 validation_split 参数，这意味着您一次只能使用其中一个参数。

在 PyTorch 和 MxNet 等其他 DL 框架的官方教程中也使用了同样的方法。他们还建议将数据集分成三部分:训练、验证和测试。

1.  训练–数据集的一部分，用于训练
2.  验证–训练时要验证的数据集的一部分
3.  测试–模型最终验证数据集的一部分

不过，如果数据集很小(包含数百个样本)，您可以在 DL 任务中使用交叉验证。在这种情况下，学习一个复杂的模型可能是一个不相关的任务，所以请确保不要使任务进一步复杂化。

## 最佳实践和技巧

值得一提的是，有时执行交叉验证可能有点棘手。

例如，分割数据集时很容易犯逻辑错误，这可能导致不可信的 CV 结果。

在交叉验证以下模型时，您可能会发现一些需要牢记的提示:

1.  拆分数据时要合乎逻辑(拆分方法有意义吗)
2.  使用适当的 CV 方法(这种方法对我的用例可行吗)
3.  当处理时间序列时，不要验证过去(见第一个提示)
4.  当处理医疗或财务数据时，记得按人进行分类。避免一个人的数据同时出现在训练集和测试集中，因为这可能会被认为是数据泄漏
5.  当从较大的图像中裁剪补丁时，记得按较大的图像 Id 进行分割

当然，提示因任务而异，几乎不可能涵盖所有的提示。这就是为什么在开始交叉验证模型之前执行**可靠的探索性数据分析**总是最佳实践。

## 最后的想法

交叉验证是一个强大的工具。每个数据科学家都应该熟悉它。在现实生活中，如果不交叉验证一个模型，你就无法完成这个项目。

在我看来，最好的 **CV** 技术是**嵌套 k 折**和标准 **k 折**。就我个人而言，我在欺诈检测任务中使用了它们。**嵌套的 k-Fold，as** 以及 **GridSeachCV** ，帮助我调整我的模型的参数。**另一方面，k 倍**用于评估我的模型的性能。

在本文中，我们已经弄清楚了什么是交叉验证，在野外有哪些 CV 技术，以及如何实现它们。在未来，最大似然算法的表现肯定会比现在更好。尽管如此，交叉验证总是需要的，以支持你的结果。

希望有了这些信息，你在为下一个机器学习项目设置简历时不会有任何问题！

## 资源

1.  [https://www . geeks forgeeks . org/cross-validation-machine-learning/](https://web.archive.org/web/20221225163138/https://www.geeksforgeeks.org/cross-validation-machine-learning/)
2.  [https://machinelearningmastery.com/k-fold-cross-validation/](https://web.archive.org/web/20221225163138/https://machinelearningmastery.com/k-fold-cross-validation/)
3.  [https://towards data science . com/cross-validation-in-machine-learning-72924 a 69872 f](https://web.archive.org/web/20221225163138/https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f)
4.  [https://towards data science . com/why-and-how-do-cross-validation-for-machine-learning-D5 BD 7 e 60 c 189](https://web.archive.org/web/20221225163138/https://towardsdatascience.com/why-and-how-to-do-cross-validation-for-machine-learning-d5bd7e60c189)
5.  [https://sci kit-learn . org/stable/modules/cross _ validation . html](https://web.archive.org/web/20221225163138/https://scikit-learn.org/stable/modules/cross_validation.html)