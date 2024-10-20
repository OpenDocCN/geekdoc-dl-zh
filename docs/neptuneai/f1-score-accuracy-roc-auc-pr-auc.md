# F1 评分 vs ROC AUC vs 准确性 vs PR AUC:应该选择哪种评价指标？

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc>

PR AUC 和 F1 分数是非常稳健的[评估指标](/web/20230215020452/https://neptune.ai/blog/the-ultimate-guide-to-evaluation-and-selection-of-models-in-machine-learning)，对于许多分类问题非常有效，但根据我的经验，更常用的指标是准确性和 ROC AUC。他们更好吗？不完全是。正如著名的“AUC 与准确性”的讨论一样:两者并用确实有好处。最大的问题是**什么时候**。

你现在可能有很多问题:

*   当准确性是比 ROC AUC 更好的评价指标时。
*   F1 成绩有什么用？
*   什么是 PR 曲线，实际如何使用？
*   如果我的问题高度不平衡，我应该使用 ROC AUC 还是 PR AUC？

一如既往地视情况而定，但在做出正确决策时，理解不同指标之间的权衡是至关重要的。

在这篇博文中，我将:

*   谈论一些最常见的**二元分类**T2 指标，如 F1 得分、ROC AUC、PR AUC 和准确性
*   **使用一个示例二进制分类问题来比较它们**
*   告诉你**在决定**选择一个指标而不是另一个指标**时，你应该考虑什么**(F1 得分与 ROC AUC)。

好，我们开始吧！

### 您将了解到:

## 评估指标摘要

我将从介绍这些分类标准开始。具体来说:

*   背后的**定义**和**直觉**是什么
*   **非技术性解释，**
*   **如何计算或绘制**，
*   **什么时候**你**应该用它**。

* * *

**^(提示)**
如果你读过我之前的博文[“二进制分类的 24 个评估指标(以及何时使用)](/web/20230215020452/https://neptune.ai/blog/evaluation-metrics-binary-classification/)，你可能想跳过这一节，向下滚动到[评估指标对比](#comparison)。

* * *

## 1.准确(性)

它衡量有多少正面和负面的观察结果被正确分类。

你**不应该在不平衡的问题上使用准确性**。那么，简单地把所有的观测值归为多数类，就很容易得到高精度的分数。

在 Python 中，您可以通过以下方式计算它:

```py
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred_class = y_pred_pos > threshold
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
accuracy = (tp + tn) / (tp + fp + fn + tn)

accuracy_score(y_true, y_pred_class)

```

由于准确性分数是根据预测的类别(而不是预测分数)计算的，因此我们**需要在计算之前应用某个阈值**。显而易见的选择是阈值 0.5，但这可能是次优的。

让我们看一个**精度如何取决于阈值**选择的例子:

![accuracy by threshold](img/cd07d5533e17137e601c2391a44b7a53.png)

*Accuracy by threshold*

您可以使用类似上面的图表来确定最佳阈值。在这种情况下，选择稍微高于标准 0.5 的值可能会使分数提高一点点 0.9686->0.9688，但在其他情况下，提高可能会更大。

那么，**什么时候**对**使用它**有意义吗？

*   当你的**问题得到平衡**时，使用准确度通常是一个好的开始。一个额外的好处是，向项目中的非技术涉众解释它非常容易，
*   当**每门课对你来说都同样重要**。

## 2.F1 分数

简而言之，它通过计算精确度和召回率之间的调和平均值，将精确度和召回率结合成一个指标。它实际上是更一般的函数**Fβ**的一个特例**:**

当在 F-beta 分数中选择 beta 时**，你越关心回忆**而不是精度**，你就应该选择更高的 beta** 。例如，对于 F1 分数，我们同样关心召回率和精确度；对于 F2 分数，召回率对我们来说是两倍重要。

![F beta by beta](img/4860e4e5e441cd417b3f02494cd34085.png)

*F beta threshold by beta*

在 0 <beta we="" care="" more="" about="" precision="" and="" so="" the="" higher="" threshold="" f="" beta="" score.="" when="">1 的情况下，我们的最佳阈值向更低的阈值移动，而在β= 1 的情况下，它处于中间位置。</beta>

可以通过运行以下命令轻松计算:

```py
from sklearn.metrics import f1_score

y_pred_class = y_pred_pos > threshold
f1_score(y_true, y_pred_class)
```

重要的是要记住，F1 分数是根据精确度和召回率计算的，而精确度和召回率又是根据预测类(而不是预测分数)计算的。

应该如何选择一个最优阈值？让我们在所有可能的阈值上绘制 F1 分数:

![f1 score by threshold](img/dfaf49cb048e548083cfc670eb3994df.png)

*F1 score by threshold*

我们可以**调整阈值来优化 F1 分数**。请注意，无论是精确度还是召回率，您都可以通过提高或降低阈值来获得满分。好的一面是，**你可以为 F1 的分数找到一个甜蜜点**。正如你所看到的，获得合适的阈值实际上可以提高你的分数，从 0.8077- > 0.8121。

你应该什么时候使用它？

*   几乎在每一个二元分类问题中，你更关心的是正类。在解决这些问题时，这是我的首要标准。
*   It **可以很容易地向业务利益相关者**解释，这在许多情况下可能是一个决定性因素。永远记住，机器学习只是解决商业问题的工具。

## 3.ROC AUC

AUC 是指曲线下的面积，因此要讨论 ROC AUC 得分，我们需要首先定义 ROC 曲线。

这是一个图表，它可视化了真阳性率(TPR)和假阳性率(FPR)之间的权衡。基本上，对于每个阈值，我们计算 TPR 和 FPR，并绘制在一个图表上。

当然，对于每个阈值，TPR 越高，FPR 越低越好，因此曲线越靠左上侧的分类器越好。

Tom Fawcett 的这篇[文章对 ROC 曲线和 ROC AUC 评分进行了广泛讨论。](https://web.archive.org/web/20230215020452/http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.10.9777&rep=rep1&type=pdf)

![roc curve](img/e8268f0230a2799f29cc6ddd4207a0b2.png)

*ROC curves*

我们可以看到一条健康的 ROC 曲线，无论是对积极类还是消极类，都被推向左上侧。不清楚哪一个在所有方面表现更好，因为 FPR < ~0.15，正类更高，从 FPR 开始~0.15，负类更高。

为了得到一个告诉我们曲线有多好的数字，我们可以计算 ROC 曲线下的面积，或 ROC AUC 得分。曲线越靠左上方，面积越大，ROC AUC 得分也越高。

或者，[可以显示【ROC AUC 分数等同于计算预测和目标之间的等级相关性。从解释的角度来看，它更有用，因为它告诉我们这个度量向**展示了你的模型**在预测排名方面有多好。它告诉你随机选择的正面实例比随机选择的负面实例排名更高的概率是多少。](https://web.archive.org/web/20230215020452/https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Area-under-curve_(AUC)_statistic_for_ROC_curves)

```py
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_true, y_pred_pos)

```

*   当你最终**关心排名预测**而不一定关心输出良好校准的概率时，你**应该使用它**(如果你想了解概率校准，请阅读杰森·布朗利的这篇[文章)。](https://web.archive.org/web/20230215020452/https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/)
*   当你的**数据严重不平衡**时，你**不应该使用它**。在这篇由 Takaya Saito 和 Marc Rehmsmeier 撰写的[文章中对此进行了广泛的讨论。直觉如下:由于大量真阴性，高度不平衡数据集的假阳性率被拉低。](https://web.archive.org/web/20230215020452/https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/)
*   你**应该在同等关心正负类**的时候使用。它自然地延伸了上一节关于不平衡数据的讨论。如果我们像关心真阳性一样关心真阴性，那么使用 ROC AUC 是完全有意义的。

## 4.平均精确度

类似于 ROC AUC，为了定义 PR AUC，我们需要定义精确回忆曲线。

这是一条在单一可视化中结合了精确度(PPV)和召回率(TPR)的曲线。对于每个阈值，你计算 PPV 和 TPR 并绘制它。y 轴上的曲线越高，模型性能越好。

当遇到经典的精确/召回难题时，你可以利用这个图做出明智的决定。显然，召回率越高，精确度越低。知道**在哪个回忆上你的精度开始快速下降**可以帮助你选择阈值并提供更好的模型。

![precision recall curve](img/4966c1378116ba28bbb60736b9d99ad3.png)

*Precision-Recall curve*

我们可以看到，对于负类，我们几乎在整个阈值范围内保持高精度和高召回率。对于阳性类别，一旦我们回忆起 0.2 的真阳性，精确度就开始下降，当我们达到 0.8 时，精确度下降到 0.7 左右。

与 ROC AUC 得分类似，您可以计算精确度-召回曲线下的面积，以获得描述模型性能的一个数字。

您也可以**将 PR AUC 视为针对每个回忆阈值**计算的精确度分数的平均值。如果需要，您还可以通过选择/剪裁召回阈值来调整此定义，以满足您的业务需求。

```py
from sklearn.metrics import average_precision_score

average_precision_score(y_true, y_pred_pos)

```

*   当您希望**向其他利益相关方传达精确/召回决策**时
*   当您想要**选择适合业务问题**的阈值时。
*   当你的数据**严重不平衡**的时候。如前所述，在 Takaya Saito 和 Marc Rehmsmeier 撰写的这篇[文章中对此进行了广泛的讨论。直觉如下:由于 PR AUC 主要关注积极类(PPV 和 TPR ),它不太关心频繁的消极类。](https://web.archive.org/web/20230215020452/https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/)
*   当**你更关心正面类而不是负面类**。如果你更关心阳性类别，因此 PPV 和 TPR，你应该选择精度-召回曲线和 PR AUC(平均精度)。

## 评估指标比较

我们将在真实用例中比较这些指标。基于最近的一次 [kaggle 竞赛](https://web.archive.org/web/20230215020452/https://www.kaggle.com/c/ieee-fraud-detection/overview)，我创建了一个欺诈检测问题示例:

*   我只选择了 **43 个特性**
*   我从原始数据集中取样了 66000 个观察值
*   我把正类的**分数调整为 0.09**
*   我用不同的超参数训练了一堆 lightGBM 分类器。

我想凭直觉判断哪些模型“真正”更好。具体来说，我怀疑只有 10 棵树的模型比有 100 棵树的模型更差。当然，随着更多的树和更小的学习率，这变得棘手，但我认为这是一个不错的代理。

因此，对于**学习率**和 **n 估计量**的组合，我做了以下工作:

*   定义的超参数值:

```py
MODEL_PARAMS = {'random_state': 1234,
                'learning_rate': 0.1,
                'n_estimators': 10}
```

```py
model = lightgbm.LGBMClassifier(**MODEL_PARAMS)
model.fit(X_train, y_train)
```

```py
y_test_pred = model.predict_proba(X_test)

```

*   记录每次运行的所有指标:

```py
y_test_pred = model.predict_proba(X_test)

```

要获得完整的代码库[，请访问这个库](https://web.archive.org/web/20230215020452/https://github.com/neptune-ml/blog-binary-classification-metrics)。

您也可以到这里[探索实验运行](https://web.archive.org/web/20230215020452/https://ui.neptune.ai/neptune-ml/binary-classification-metrics/experiments?viewId=817b46ba-103e-11ea-9a39-42010a840083)与:

*   评估指标
*   性能图表
*   阈值图度量

让我们看看我们的模型如何在不同的指标上得分。

在这个问题上，所有这些指标都是从最好到最差的排序模型，非常相似，但略有不同。此外，分数本身也可能有很大差异。

在接下来的部分中，我们将更详细地讨论它。

## 5.准确性与 ROC AUC

第一个大的区别是，你**计算预测类别**的准确性，而你**计算预测分数**的 ROC AUC。这意味着您必须为您的问题找到最佳阈值。

此外，准确性着眼于正确分配的正类和负类的分数。这意味着，如果我们的**问题是高度不平衡的**，我们通过简单地预测**所有的观察值都属于多数类，就可以得到一个真正的**高精度分数**。**

另一方面，如果你的问题是**平衡的**并且你**既关心正面预测又关心负面预测**，**准确性是一个很好的选择**，因为它真的简单且容易解释。

另一件要记住的事情是 **ROC AUC 特别擅长对**预测进行排名。正因为如此，如果你有一个问题，即排序你的观察是你关心的 ROC AUC 可能是你正在寻找的。

现在，让我们看看我们的实验结果:

第一个观察结果是，模型在 ROC AUC 和准确性上的排名几乎完全相同。

其次，精确度分数从最差模型的 0.93 开始，到最佳模型的 0.97。请记住，将所有观察值预测为多数类 0 将会给出 0.9 的准确度，因此我们最差的实验 [BIN-98](https://web.archive.org/web/20230215020452/https://ui.neptune.ai/neptune-ml/binary-classification-metrics/e/BIN-98/logs) 仅比这稍微好一点。然而，这个分数本身就很高，这表明**在考虑准确性时，你应该始终考虑到不平衡。**

![](img/8a71967c2a3ed34299dc79e420df5234.png)

有一个有趣的指标叫做 Cohen Kappa，它通过计算相对于“根据阶级不平衡抽样”模型的精确度改进，将不平衡考虑在内。

点击阅读更多关于[科恩卡帕的信息。](https://web.archive.org/web/20230215020452/https://neptune.ai/blog/evaluation-metrics-binary-classification#13)

6.F1 分数与准确性

## 这两个指标都将类别预测作为输入，因此无论选择哪一个，您都必须调整阈值。

记住 **F1 分数**在**正类**上平衡精度和召回，而**精度**查看正确分类的观测值**正类和负类**。这使得**差别很大**，特别是对于**不平衡问题**而言，在默认情况下，我们的模型将擅长预测真正的负值，因此准确性将很高。然而，如果你同样关心真阴性和真阳性，那么准确性是你应该选择的标准。

如果我们看看下面的实验:

在我们的示例中，这两个指标同样能够帮助我们对模型进行排序并选择最佳模型。1-10 **的**级不平衡**让我们的准确率**真的默认**高**。正因为如此，即使是最差的模型也有非常高的准确性，当我们登上积分榜前列时，准确性方面的改进并不像 F1 分数那样明显。

7.ROC AUC 与 PR AUC

## ROC AUC 和 PR AUC 之间的共同点是，它们都查看分类模型的预测分数，而不是阈值类别分配。然而不同的是 **ROC AUC 看的是**真阳性率**TPR**和假阳性率 **FPR** 而 **PR AUC 看的是**阳性预测值 **PPV** 和真阳性率 **TPR** 。

因此**如果你更关心正类，那么使用 PR AUC** 是一个更好的选择，它对正类的改进更敏感。一个常见的场景是高度不平衡的数据集，其中我们想要找到的肯定类的比例很小(就像在欺诈检测中一样)。我强烈推荐看一下这个 kaggle 内核，以便对不平衡数据集的 ROC AUC 与 PR AUC 进行更长时间的讨论。

**如果你同样关心正负类**或者你的数据集相当平衡，那么使用 **ROC AUC** 是个好主意。

让我们比较这两个指标的实验:

他们对模型的排名相似，但如果你看看实验 [BIN-100](https://web.archive.org/web/20230215020452/https://ui.neptune.ai/neptune-ml/binary-classification-metrics/e/BIN-100/logs) 和 [BIN 102](https://web.archive.org/web/20230215020452/https://ui.neptune.ai/neptune-ml/binary-classification-metrics/e/BIN-102/logs) ，就会发现略有不同。

然而，以平均精确度(PR AUC)计算的改善更大且更清晰。我们从 0.69 到 0.87，同时 ROC AUC 从 0.92 到 0.97。因为摇摆不定 AUC 可能会给人一种非常高性能的错误感觉，而实际上你的模型可能做得不太好。

8.F1 得分与 ROC AUC

## F1 分数和 ROC AUC 之间的一个很大的区别是，第一个以预测的类为输入，第二个以预测的分数为输入。因此，**对于 F1 分数，您需要选择一个阈值**，将您的观察结果分配给这些类别。通常，如果选择得当，您可以提高模型性能很多。

因此，**如果你关心排名预测**，不需要它们是正确校准的概率，并且你的数据集**不是严重不平衡的**，那么我会选择 **ROC AUC** 。

**如果你的数据集严重不平衡**和/或你主要关心正面类，我会考虑使用 **F1 分数**，或精确召回曲线和 **PR AUC** 。使用 F1(或 Fbeta)的另一个原因是这些指标更容易解释并与业务利益相关者沟通。

让我们来看看实验结果，以获得更多的见解:

实验在 F1 分数(*阈值=0.5* )和 ROC AUC 上排名相同。但是，F1 值较低，最差和最佳模型之间的差异较大。对于 ROC AUC 评分，值越大，差异越小。特别有意思的是[实验 BIN-98](https://web.archive.org/web/20230215020452/https://ui.neptune.ai/neptune-ml/binary-classification-metrics/e/BIN-98/logs) ，F1 分 0.45，ROC AUC 0.92。其原因是 0.5 的阈值对于一个尚未训练的模型(只有 10 棵树)来说是一个非常糟糕的选择。如果将 F1 值设置为 0.24，则 F1 值为 0.63，如下所示:

如果你想很容易地为每个实验记录这些图，我在这篇文章的结尾附上一个记录助手。

![f1 score by threshold](img/e18baaa6500b600ce5bbd3e39bc4c4e3.png)

*F1 score by threshold*

![](img/8a71967c2a3ed34299dc79e420df5234.png)

最后的想法

## 在这篇博文中，你已经**了解了用于评估二元分类模型的******几种常见的** **指标**。**

 **我们已经讨论了它们是如何定义的，如何解释和计算它们，以及何时应该考虑使用它们。

最后，**我们** **在一个实际问题上比较了那些评估指标**，并讨论了一些你可能面临的典型决策。

有了这些知识，你就有了为下一个二元分类问题选择一个好的评估指标的设备！

**奖金:**

### 为了让事情变得简单一点，我准备了:

看看下面这些吧！

记录功能

## 您可以**记录我们为您的机器学习项目覆盖的所有度量和性能图表**，并使用我们的 [Python 客户端](https://web.archive.org/web/20230215020452/https://docs.neptune.ai/usage/)和[集成](https://web.archive.org/web/20230215020452/https://docs.neptune.ai/integrations/)在 Neptune 中探索它们(在下面的示例中，我使用 [Neptune-LightGBM 集成](https://web.archive.org/web/20230215020452/https://docs.neptune.ai/integrations/lightgbm/))。

```py
pip install neptune-client neptune-lightgbm
```

```py
Import neptune.new as neptune

run = neptune.init(...)
neptune_callback = NeptuneCallback(run=run)

gbm = lgb.train(
       params,
       lgb_train,
       callbacks=[neptune_callback],
)

custom_score = ...

run["logs/custom_score"] = custom_score
```

您可以将不同种类的元数据记录到 Neptune，包括指标、图表、参数、图像等等。[查看文档](https://web.archive.org/web/20230215020452/https://docs.neptune.ai/logging/what_you_can_log/)了解更多信息。

二进制分类指标清单

## 我们已经为您创建了一个不错的备忘单，它将我在这篇博文中浏览的所有内容都放在一个几页长、易于理解的文档中，您可以在需要任何二进制分类指标相关的内容时打印和使用它。

获取您的二元分类指标清单

#### Get your binary classification metrics cheatsheet**