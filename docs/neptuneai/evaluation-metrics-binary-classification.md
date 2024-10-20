# 二元分类的 24 个评估指标(以及何时使用它们)

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/evaluation-metrics-binary-classification>

分类指标让你评估机器学习模型的[性能](/web/20221226161840/https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide)但是它们数量太多，每一个都有自己的优点和缺点，选择一个适合你的问题的评估指标有时真的很棘手。

在本文中，您将了解一些常见的和鲜为人知的评估指标和图表，以**理解如何为您的问题**选择模型性能**指标。**

具体来说，我将谈论:

*   大多数主要分类指标背后的**定义**和**直觉**是什么，
*   **非技术性解释**您可以向业务利益相关者传达二进制分类的指标，
*   **如何绘制**性能图表和**计算二元分类的通用指标，**
*   **什么时候**你**应该用**他们。

有了这些，你就会明白权衡取舍，从而更容易做出与指标相关的决策。

## 分类指标到底是什么？

简单地说，[分类指标](https://web.archive.org/web/20221226161840/https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce)是一个衡量机器学习模型在将观察值分配给特定类别时的性能的数字。

二元分类是一种特殊的情况，你只需要分类:积极的和消极的。

典型地，性能表现在从 0 到 1 的范围内(尽管不总是如此)，其中 1 分是为完美模型保留的。

不要用枯燥的定义来烦你，让我们基于最近的 [Kaggle 竞赛来讨论一个欺诈检测问题示例的各种分类指标。](https://web.archive.org/web/20221226161840/https://www.kaggle.com/c/ieee-fraud-detection/overview)

我选择了 **43 个特征**，并从原始数据集中采样了 **66000 个观察值**，将正类的**分数调整为 0.09** 。

然后我用不同的超参数训练了一堆 lightGBM 分类器。我只使用了 **learning_rate** 和 **n_estimators** 参数，因为我想对哪些模型“真正”更好有一个直觉。具体来说，我怀疑只有 10 棵树的模型比有 100 棵树的模型更差。当然，随着使用更多的树和更小的学习率，这变得棘手，但我认为这是一个不错的代理。

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

*   每次跑步的记录分数:

```py
run["logs/score"] = score
```

要了解更多关于记录分数和指标的信息，请访问 Neptune docs。

*   每次运行的记录 matplolib 数字:

```py
run["images/figure"].upload(neptune.types.File.as_image(fig))
```

要了解更多关于记录 matplotlib 数字的信息，请访问 Neptune docs。

[在这里，您可以通过以下方式探索实验运行](https://web.archive.org/web/20221226161840/https://ui.neptune.ai/neptune-ml/binary-classification-metrics/experiments?filterId=20f71748-85ad-499d-a72e-68962bcd36a0&viewId=817b46ba-103e-11ea-9a39-42010a840083):

*   评估指标
*   性能图表
*   阈值图度量

好了，现在我们可以开始讨论那些分类指标了！

## 了解以下评估指标

我知道一下子要检查的东西很多。这就是为什么你可以跳到你感兴趣的部分去读。

## 1.混淆矩阵

### 如何计算:

这是呈现真阳性(tp)、真阴性(tn)、假阳性(fp)和假阴性(fn)预测的常见方式。这些值以矩阵的形式显示，其中 Y 轴显示真实的类，而 X 轴显示预测的类。

它是基于类预测计算的，这意味着您的模型的输出需要首先进行阈值处理。

```py
from sklearn.metrics import confusion_matrix

y_pred_class = y_pred_pos > threshold
cm = confusion_matrix(y_true, y_pred_class)
tn, fp, fn, tp = cm.ravel()
```

### 看起来怎么样:

在这个例子中，我们可以看到:

*   **11918** 个预言被**真否定**，
*   **872** 被**真阳性**，
*   **82** 被**误报**，
*   333 个预测被**误判**。

此外，正如我们已经知道的，这是一个不平衡的问题。顺便说一句，如果你想阅读更多关于不平衡问题的文章，我推荐你看看这篇由汤姆·福西特写的文章。

### 何时使用:

*   差不多一直都是。我喜欢看到名义值，而不是标准化的值，以了解模型在不同的、通常不平衡的类上的表现。

## 2.假阳性率| I 型错误

当我们预测到某件事，而它并没有发生时，我们就增加了假阳性率。你可以把它看作是基于你的模型预测而产生的假警报的一小部分**。**

### 如何计算:

```py
from sklearn.metrics import confusion_matrix

y_pred_class = y_pred_pos > threshold
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
false_positive_rate = fp / (fp + tn)

run[“logs/false_positive_rate”] = false_positive_rate

```

### 模型在此指标中的得分情况(阈值=0.5):

对于所有模型，类型 1 错误警报非常低，但是通过调整阈值，我们可以获得更低的比率。因为我们在分母中有真正的负数，我们的误差会很低，只是因为数据集不平衡。

### 它如何取决于阈值:

显然，如果我们提高阈值，只有得分较高的观察结果才会被归类为阳性。在我们的例子中，我们可以看到，为了达到 0 的完美 FPR，我们需要将阈值增加到 0.83。然而，这可能意味着只有很少的预测被分类。

### 何时使用:

*   你很少会单独使用这个指标。通常作为其他度量的辅助指标，
*   如果处理警报的**成本很高**，您应该考虑提高阈值以获得更少的警报。

## 3.假阴性率|第二类错误

当我们没有预测到某件事情的发生时，我们就造成了假阴性率。你可以把它想成是你的模型允许错过的欺诈交易的**部分。**

### 如何计算:

```py
from sklearn.metrics import confusion_matrix

y_pred_class = y_pred_pos > threshold
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
false_negative_rate = fn / (tp + fn)

run[“logs/false_negative_rate”] = false_negative_rate

```

### 模型在此指标中的得分情况(阈值=0.5):

我们可以看到，在我们的例子中，第二类错误比第一类错误高得多。有趣的是，我们的 BIN-98 实验中第一类错误最低的第二类错误最高。有一个简单的解释，基于这样一个事实，即我们的数据集是不平衡的，并且对于类型 2 错误，我们在分母中没有真正的负数。

### 它如何取决于阈值:

如果我们降低阈值，更多的观察结果将被归类为阳性。在特定阈值，我们会将所有内容标记为肯定的(例如欺诈)。通过将阈值降至 0.01，我们实际上可以获得 0.083 的 FNR。

### 何时使用:

*   通常，它不单独使用，而是与其他度量一起使用，
*   如果让欺诈交易通过的成本很高，而你从用户那里获得的价值不高，你可以考虑关注这个数字。

## 4.真阴性率|特异性

它测量所有负面观察中有多少被我们归类为负面。在我们的欺诈检测示例中，它告诉我们在所有非欺诈交易中有多少交易被标记为干净的。

### 如何计算:

```py
from sklearn.metrics import confusion_matrix

y_pred_class = y_pred_pos > threshold
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
true_negative_rate = tn / (tn + fp)

run[”logs/true_negative_rate”] = true_negative_rate

```

### 模型在此指标中的得分情况(阈值=0.5):

所有模型的特异性都非常高。如果你想一想，在我们的不平衡问题中，你会想到这一点。将阴性病例归类为阴性比将阳性病例归类容易得多，因此得分高。

### 它如何取决于阈值:

阈值越高，我们能回忆起的真正负面的观察就越多。我们可以看到，从阈值=0.4 开始，我们的模型在将负面情况分类为负面方面做得非常好。

### 何时使用:

*   通常，您不会单独使用它，而是将其作为一个辅助指标，
*   当你说某样东西是安全的时候，你真的想确定你是对的。一个典型的例子是医生告诉病人“你很健康”。在这里犯一个错误，告诉一个病人他们是安全的，可以回家了，这可能是你想要避免的。

## 5.阴性预测值

它衡量所有负面预测中有多少是正确的。你可以认为它是负类的精度。在我们的例子中，它告诉我们在所有非欺诈性预测中，正确预测的干净交易的比例是多少。

### 如何计算:

```py
from sklearn.metrics import confusion_matrix

y_pred_class = y_pred_pos > threshold
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
negative_predictive_value = tn/ (tn + fn)

run[”logs/negative_predictive_value”] = negative_predictive_value

```

### 模型在此指标中的得分情况(阈值=0.5):

所有的模型得分都很高，这并不奇怪，因为对于一个不平衡的问题，预测负类是很容易的。

### 它如何取决于阈值:

阈值越高，被归类为阴性的病例越多，分数就越低。然而，在我们不平衡的例子中，即使在非常高的阈值下，阴性预测值仍然是好的。

### 何时使用:

*   当我们关心负面预测的高精度时。例如，假设我们真的不想有任何额外的过程来筛选预测为干净的事务。在这种情况下，我们可能要确保我们的阴性预测值很高。

## 6.错误发现率

它衡量所有积极预测中有多少预测是不正确的。你可以简单地认为它是 1-精度。在我们的示例中，它告诉我们在所有欺诈预测中，错误预测的欺诈交易所占的比例。

### 如何计算:

```py
from sklearn.metrics import confusion_matrix

y_pred_class = y_pred_pos > threshold
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
false_discovery_rate = fp/ (tp + fp)

run[“logs/false_discovery_rate”] = false_discovery_rate

```

### 模型在此指标中的得分情况(阈值=0.5):

“最佳模型”是令人难以置信的浅层 lightGBM，我们预计这是不正确的(更深的模型应该工作得更好)。

这是很重要的一点，单看精度(或召回率)会导致你选择一个次优的模型。

### 它如何取决于阈值:

阈值越高，正面预测越少。不太正面的预测，被分类为正面的预测具有较高的确定性分数。因此，错误发现率下降。

### 何时使用它

*   同样，单独使用它通常是没有意义的，而是要与其他指标结合使用，如召回率。
*   当发出错误警报的成本很高时，当您希望所有积极的预测都值得一看时，您应该优化精确度。

## 7.真实阳性率|回忆|灵敏度

它衡量在所有积极的观察中，我们将多少观察归类为积极的。它告诉我们我们从所有欺诈交易中收回了多少欺诈交易。

当你优化回忆时，你想把所有有罪的人关进监狱。

### 如何计算:

```py
from sklearn.metrics import confusion_matrix, recall_score

y_pred_class = y_pred_pos > threshold
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
recall = recall_score(y_true, y_pred_class) 

run[“logs/recall_score”] = recall

```

### 模型在此指标中的得分情况(阈值=0.5):

我们的最佳模型可以在阈值为 0.5 时召回 0.72 笔欺诈交易。我们的模型在召回率上的差异非常显著，我们可以清楚地看到更好和更差的模型。当然，对于每个模型，我们可以调整阈值来召回所有欺诈交易。

### 它如何取决于阈值:

对于阈值 0.1，我们将绝大多数交易归类为欺诈性交易，因此召回率非常高，为 0.917。随着阈值的增加，回忆下降。

### 何时使用:

*   通常，您不会单独使用它，而是将它与精度等其他指标结合使用。
*   也就是说，当你真的关心捕捉所有欺诈性交易时，即使以虚假警报为代价，召回也是一个必不可少的指标。对你来说，处理这些警报的成本可能很低，但当交易不为人知时，成本会非常高。

## 8.阳性预测值|精确度

它测量有多少预测为阳性的观察结果实际上是阳性的。以我们的欺诈检测为例，它告诉我们被正确归类为欺诈的交易的比例是多少。

当你优化精度时，你想确保你投入监狱的人是有罪的。

### 如何计算:

```py
from sklearn.metrics import confusion_matrix, precision_score

y_pred_class = y_pred_pos > threshold
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
precision = precision_score(y_true, y_pred_class) 

run[“logs/precision_score”] = precison

```

### 模型在此指标中的得分情况(阈值=0.5):

似乎所有的模型在这个阈值上都有相当高的精度。“最佳模式”是令人难以置信的浅光 GBM，这显然有鱼腥味。这是很重要的一点，单看精度(或召回率)会导致你选择一个次优的模型。当然，对于每个模型，我们可以调整阈值来提高精度。这是因为，如果我们采用一小部分高分预测，这些预测的精确度可能会很高。

### 它如何取决于阈值:

阈值越高，精度越好，阈值为 0.68 时，我们实际上可以获得非常精确的模型。超过这个阈值，模型不会将任何东西归类为积极的，所以我们不会绘制它。

### 何时使用:

*   同样，单独使用它通常是没有意义的，而是要与其他指标结合使用，如召回率。
*   当发出错误警报代价高昂时，当你希望所有积极的预测都值得一看时，你应该优化精确度。

## 9.准确(性)

它衡量有多少正面和负面的观察结果被正确分类。

你**不应该在不平衡的问题上使用准确性**。那么，简单地把所有的观测值归为多数类，就很容易得到高精度的分数。例如，在我们的案例中，通过将所有交易分类为非欺诈交易，我们可以获得超过 0.9 的准确度。

### 如何计算:

```py
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred_class = y_pred_pos > threshold
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
accuracy = accuracy_score(y_true, y_pred_class) 

run[“logs/accuracy”] = accuracy

```

### 模型在此指标中的得分情况(阈值=0.5):

我们可以看到，对于所有模型，我们都以很大的优势击败了虚拟模型(所有干净的交易)。此外，我们预期会更好的模型实际上位于顶部。

### 它如何取决于阈值:

准确地说，你真的可以使用上面的图表来确定最佳阈值。在这种情况下，选择稍微高于标准 0.5 的值可能会使分数提高一点点 0.9686->0.9688。

### 何时使用:

*   当你的问题得到平衡时，使用准确性通常是一个好的开始。一个额外的好处是，向项目中的非技术涉众解释它非常容易，
*   当每门课对你来说都同样重要的时候。

## 10.fβ分数

简而言之，它将精确度和召回率结合成一个指标。分数越高，我们的模型越好。你可以用下面的方法计算它:

当在 F-beta 分数中选择 beta 时**，你越关心回忆**而不是精度**，你就应该选择更高的 beta** 。例如，对于 F1 分数，我们同样关心召回率和精确度；对于 F2 分数，召回率对我们来说是两倍重要。

在 0 <beta we="" care="" more="" about="" precision="" and="" so="" the="" higher="" threshold="" f="" beta="" score.="" when="">1 的情况下，我们的最佳阈值向更低的阈值移动，而在β= 1 的情况下，它在中间的某个位置。</beta>

### 如何计算:

```py
from sklearn.metrics import fbeta_score

y_pred_class = y_pred_pos > threshold
fbeta = fbeta_score(y_true, y_pred_class, beta)

run["logs/fbeta_score"] = fbeta
```

## 11.F1 分数(β= 1)

这是精确度和召回率之间的调和平均值。

### 如何计算:

```py
from sklearn.metrics import f1_score
y_pred_class = y_pred_pos > threshold
f1= f1_score(y_true, y_pred_class)

run[“logs/f1_score”] = f1

```

### 模型在此指标中的得分情况(阈值=0.5):

正如我们所看到的，精确度和召回率的结合给了我们一个更真实的模型视图。我们得到 0.808 的最佳成绩，还有很大的提升空间。

好的一点是，它似乎正确地对我们的模型进行了排序，那些较大的 lightGBMs 在顶部。

### 它如何取决于阈值:

我们可以**调整阈值来优化 F1 得分**。请注意，无论是精确度还是召回率，您都可以通过提高或降低阈值来获得满分。好消息是，**你可以为 F1metric 找到一个甜蜜点**。正如你所看到的，获得合适的阈值实际上可以提高你的分数 0.8077- > 0.8121。

### 何时使用:

*   几乎在所有的二元分类问题中。这是我解决这些问题时的首要标准。这可以很容易地向商业利益相关者解释。

## 12.F2 分数(β= 2)

这是一个结合了精确度和召回率的指标，将**的召回率提高了两倍。**

### 如何计算:

```py
from sklearn.metrics import fbeta_score

y_pred_class = y_pred_pos > threshold
f2 = fbeta_score(y_true, y_pred_class, beta = 2)

run[“logs/f2_score”] = f2

```

### 模型在此指标中的得分情况(阈值=0.5):

所有车型的得分甚至低于 F1，但可以通过大幅调整阈值来提高。同样，它似乎对我们的模型进行了正确的排序，至少在这个简单的例子中是这样。

### 它如何取决于阈值:

我们可以看到，阈值越低，回忆起的真阳性越多，分数就越高。你通常可以**找到一个最佳切入点**。从 0.755 - > 0.803 的可能增益显示了**阈值调整在这里是多么重要。**

### 何时使用:

*   当回忆正面的观察(欺诈交易)比精确更重要时，我会考虑使用它

## 13.科恩卡帕度量

简而言之，Cohen Kappa 告诉你你的模型比基于类别频率预测的随机分类器好多少。

要计算它，需要计算两件事:**“观察一致”(po)** 和**“预期一致”(pe)** 。观察一致(po)就是我们的分类器预测如何与地面事实一致，这意味着它只是准确性。预期一致性(pe)是根据类别频率进行采样的**随机分类器的预测如何与基本事实或随机分类器的准确性相一致。**

从解释的角度来看，我喜欢它通过合并基线(虚拟)分类器，将一些很容易解释的东西(准确性)扩展到数据集不平衡的情况。

### 如何计算:

```py
from sklearn.metrics import cohen_kappa_score

cohen_kappa = cohen_kappa_score(y_true, y_pred_class)

run[“logs/cohen_kappa_score”] = cohen_kappa

```

### 模型在此指标中的得分情况(阈值=0.5):

基于这个指标，我们可以很容易地区分最差/最好的模型。还有，我们可以看到，我们最好的模型还有很大的改进空间。

### 它如何取决于阈值:

使用如上图所示的图表，我们可以找到优化 cohen kappa 的阈值。在这种情况下，它为 0.31，比标准的 0.5 提高了 0.7909 -> 0.7947。

### 何时使用:

*   这一指标在分类环境中并不常用。然而，它可以很好地解决不平衡的问题，似乎是准确性的一个伟大的伴侣/替代品。

## 14.马修斯相关系数

这是预测类和地面真相之间的关联。它可以基于混淆矩阵的值来计算:

或者，您也可以计算 y_true 和 y_pred 之间的相关性。

### 如何计算:

```py
from sklearn.metrics import matthews_corrcoef

y_pred_class = y_pred_pos > threshold
matthews_corr = matthews_corrcoef(y_true, y_pred_class)
run[“logs/matthews_corrcoef”] = matthews_corr

```

### 模型在此指标中的得分情况(阈值=0.5):

我们可以清楚地看到我们的模型质量有所提高，还有很大的发展空间，这是我非常喜欢的。此外，它对我们的模型进行合理的排名，并将您认为更好的模型放在最上面。当然，MCC 取决于我们选择的阈值。

### 它如何取决于阈值:

我们可以调整阈值来优化 MCC。在我们的例子中，最好的分数是 0.53，但我真正喜欢的是它对阈值变化不是超级敏感。

### 何时使用:

*   当处理不平衡的问题时，
*   当你想要一些容易理解的东西时。

## 15.受试者工作特征曲线

这是一个图表，它可视化了真阳性率(TPR)和假阳性率(FPR)之间的权衡。基本上，对于每个阈值，我们计算 TPR 和 FPR，并绘制在一个图表上。

当然，对于每个阈值，TPR 越高，FPR 越低越好，因此曲线越靠左上侧的分类器越好。

Tom Fawcett 的这篇[文章对 ROC 曲线和 ROC AUC 评分进行了广泛讨论。](https://web.archive.org/web/20221226161840/http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.10.9777&rep=rep1&type=pdf)

### 如何计算:

```py
from scikitplot.metrics import plot_roc

fig, ax = plt.subplots()
plot_roc(y_true, y_pred, ax=ax)

run[“images/ROC”].upload(neptune.types.File.as_image(fig))

```

### 看起来怎么样:

我们可以看到一条健康的 ROC 曲线，无论是正类还是负类都被推向左上侧。不清楚哪一个在所有方面表现更好，因为 FPR < ~0.15，正类更高，从 FPR 开始~0.15，负类更高。

## 16.ROC AUC 得分

为了得到一个告诉我们曲线有多好的数字，我们可以计算 ROC 曲线下的面积，或 ROC AUC 得分。曲线越靠左上方，面积越大，ROC AUC 得分也越高。

或者，[可以显示【ROC AUC 分数等同于计算预测和目标之间的等级相关性。从解释的角度来看，它更有用，因为它告诉我们这个度量向**展示了你的模型**在预测排名方面有多好。它告诉你随机选择的正面实例比随机选择的负面实例排名更高的概率是多少。](https://web.archive.org/web/20221226161840/https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Area-under-curve_(AUC)_statistic_for_ROC_curves)

### 如何计算:

```py
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_true, y_pred_pos)

run[“logs/roc_auc_score”] = roc_auc

```

### 模型在此指标中的得分情况:

我们可以看到改进，人们认为更好的模型确实得分更高。此外，分数独立于阈值，这很方便。

### 何时使用:

*   当你最终**关心排名预测**而不一定关心输出良好校准的概率时，你**应该使用它**(如果你想了解概率校准，请阅读杰森·布朗利的这篇[文章)。](https://web.archive.org/web/20221226161840/https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/)
*   当你的**数据严重不平衡**时，你**不应该使用它**。在这篇由 Takaya Saito 和 Marc Rehmsmeier 撰写的[文章中对此进行了广泛的讨论。直觉如下:由于大量真阴性，高度不平衡数据集的假阳性率被拉低。](https://web.archive.org/web/20221226161840/https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/)
*   你**应该在同等关心正负类**的时候使用。它自然地延伸了上一节关于不平衡数据的讨论。如果我们像关心真阳性一样关心真阴性，那么使用 ROC AUC 是完全有意义的。

## 17.精确回忆曲线

这是一条在单一可视化中结合了精确度(PPV)和召回率(TPR)的曲线。对于每个阈值，你计算 PPV 和 TPR 并绘制它。y 轴上的曲线越高，模型性能越好。

当遇到经典的精确/召回难题时，你可以利用这个图做出明智的决定。显然，召回率越高，精确度越低。知道**在哪个回忆上你的精度开始快速下降**可以帮助你选择阈值并提供更好的模型。

### 如何计算:

```py
from scikitplot.metrics import plot_precision_recall

fig, ax = plt.subplots()
plot_precision_recall(y_true, y_pred, ax=ax)

run[“images/precision_recall”].upload(neptune.types.File.as_image(fig))

```

### 看起来怎么样:

我们可以看到，对于负类，我们几乎在整个阈值范围内保持高精度和高召回率。因为当我们回忆起 0.2 的真阳性时，阳性分类的精确度就开始下降，当我们达到 0.8 时，精确度下降到 0.7 左右。

## 18.PR AUC 分数|平均精确度

与 ROC AUC 得分类似，您可以计算精确度-召回曲线下的面积，以获得描述模型性能的一个数字。

你也可以把 PR AUC 想成每个回忆阈值[0.0，1.0]计算出来的精度分数的平均值。如果需要，您还可以通过选择/剪裁召回阈值来调整此定义，以满足您的业务需求。

### 如何计算:

```py
from sklearn.metrics import average_precision_score

avg_precision = average_precision_score(y_true, y_pred_pos)

run[“logs/average_precision_score”] = avg_precision

```

### 模型在此指标中的得分情况:

我们怀疑“真正”更好的模型实际上在这个指标上更好，这绝对是一件好事。总的来说，我们可以看到高分，但远不如 ROC AUC 分数乐观(0.96+)。

### 何时使用:

*   当您希望**向其他利益相关方传达精确/召回决策**时
*   当您想要**选择适合业务问题**的阈值时。
*   当你的数据**严重不平衡**的时候。如前所述，在 Takaya Saito 和 Marc Rehmsmeier 撰写的这篇[文章中对此进行了广泛的讨论。直觉如下:由于 PR AUC 主要关注积极类(PPV 和 TPR ),它不太关心频繁的消极类。](https://web.archive.org/web/20221226161840/https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/)
*   当**你更关心正面类而不是负面类**。如果你更关心阳性类别，因此 PPV 和 TPR，你应该选择精度-召回曲线和 PR AUC(平均精度)。

## 19.原木损失

Log loss 经常被用作在机器学习模型的罩下被优化的目标函数。然而，它也可以用作性能指标。

基本上，我们计算每个观察值的真实值和预测值之间的差异，并对所有观察值的误差进行平均。对于一次观测，误差公式为:

我们的模型越确定一个观察结果是肯定的，当它实际上是肯定的时候，误差就越小。但这不是线性关系。看一下误差如何随着差异的增加而变化是有好处的:

所以当我们确定某事不真实时，我们的模型会受到很大的惩罚。例如，当我们给一个负面的观察值打 0.9999 分时，我们的损失就会大增。这就是为什么有时候为了降低发生这种情况的风险，删减你的预测是有意义的。

如果你想了解更多关于日志损失的知识，请阅读丹尼尔·戈多伊的文章。

### 如何计算:

```py
from sklearn.metrics import log_loss

loss = log_loss(y_true, y_pred)

run[“logs/log_loss”] = loss

```

### 模型在此指标中的得分情况:

很难真正看到强大的改进，也很难对模型的强大程度有一个直观的感受。还有，之前被选为最好的那款(BIN-101)在群里是中游。这表明使用日志损失作为性能指标可能是一个有风险的提议。

### 何时使用:

*   几乎总有一个性能**指标能更好地匹配你的**业务**问题。**因此，我会将对数损失作为您模型的一个目标，并结合一些其他指标来评估性能。

## 20.布赖尔乐谱

这是一个衡量你的预测与真实值相差有多远的指标。有一个观察结果是这样的:

基本上，它是概率空间中的均方误差，因此，它通常用于校准机器学习模型的概率。如果你想阅读更多关于概率校准的内容，我推荐你阅读 Jason Brownlee 的这篇[文章。](https://web.archive.org/web/20221226161840/https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/)

它可以很好地补充您的 ROC AUC 得分和其他关注其他事情的指标。

### 如何计算:

```py
from sklearn.metrics import brier_score_loss

brier_loss = brier_score_loss(y_true, y_pred_pos)

run[“logs/brier_score_loss”] = brier_loss

```

### 模型在此指标中的得分情况:

来自[实验 BIN-101](https://web.archive.org/web/20221226161840/https://ui.neptune.ai/neptune-ml/binary-classification-metrics/e/BIN-101/logs) 的模型具有最佳校准，对于该模型，我们的预测平均误差为 0.16 (√0.0263309)。

### 何时使用:

*   当你**关心校准概率**时。

## 21.累积收益图

简而言之，对于给定分数的最高得分预测，它可以帮助您衡量通过使用您的模型而不是随机模型获得了多少收益。

简单来说:

*   你把你的预测从最高到最低排序
*   对于每一个百分位数，你都要计算直到那个百分位数的真正正面观察的分数。

使用你的模型来定位给定的用户/账户/交易组的好处是显而易见的，尤其是如果你真的关心对它们进行排序的话。

### 如何计算:

```py
from scikitplot.metrics import plot_cumulative_gain

fig, ax = plt.subplots()
plot_cumulative_gain(y_true, y_pred, ax=ax)

run[“images/cumulative_gains”].upload(neptune.types.File.as_image(fig))

```

### 看起来怎么样:

我们可以看到，随着得分最高的预测样本的增加，我们的累积收益图表迅速上升。当我们到达第 20 个百分位数时，超过 90%的阳性病例被覆盖。您可以使用此图表来区分优先级，并筛选出可能的欺诈交易进行处理。

假设我们要使用我们的模型来分配可能的欺诈交易进行处理，并且我们需要进行优先级排序。我们可以用这个图表来告诉我们在哪里选择一个截止点最有意义。

### 何时使用:

*   每当您希望选择最有希望的客户或交易作为目标，并且希望使用您的模型进行排序时。
*   它可以很好地补充 ROC AUC 分数，ROC AUC 分数衡量模型的排名/排序性能。

## 22.升力曲线|升力图

它只是累积收益图的一种不同表现形式:

*   我们将预测从最高到最低排序
*   对于每个百分位数，我们计算我们的模型和随机模型在该百分位数上的真实正观察的分数，
*   我们计算这些分数的比率，然后画出来。

它会告诉您，对于给定百分比的最高得分预测，您的模型比随机模型好多少。

### 如何计算:

```py
from scikitplot.metrics import plot_lift_curve

fig, ax = plt.subplots()
plot_lift_curve(y_true, y_pred, ax=ax)

run[“images/lift_curve”].upload(neptune.types.File.as_image(fig))

```

### 看起来怎么样:

因此，对于前 10%的预测，我们的模型比随机模型好 10 倍以上，对于 20%的预测，比随机模型好 4 倍以上，以此类推。

### 何时使用:

*   每当您希望选择最有希望的客户或交易作为目标，并且希望使用您的模型进行排序时。
*   它可以很好地补充 ROC AUC 分数，ROC AUC 分数衡量模型的排名/排序性能。

## 23\. Kolmogorov-Smirnov plot

KS 图有助于评估正类和负类预测分布之间的分离。

为了创建它，您需要:

*   根据预测得分对您的观察结果进行排序，
*   对于排序数据集(深度)的每个截止点[0.0，1.0]，计算该深度中真阳性和真阴性的比例，
*   在 Y 轴上绘制这些分数，正(深度)/正(所有)、负(深度)/负(所有)，在 X 轴上绘制数据集深度。

因此，它的工作原理类似于累积收益图，但它不是只看正类，而是看正类和负类之间的分离。

Riaz Khan 的这篇[文章对 KS 图和 KS 统计数据进行了很好的解释。](https://web.archive.org/web/20221226161840/http://rstudio-pubs-static.s3.amazonaws.com/303414_fb0a43efb0d7433983fdc9adcf87317f.html)

### 如何计算:

```py
from scikitplot.metrics import plot_ks_statistic

fig, ax = plt.subplots()
plot_ks_statistic(y_true, y_pred, ax=ax)

run[“images/kolmogorov-smirnov”].upload(neptune.types.File.as_image(fig))

```

### 看起来怎么样:

因此，我们可以看到，最大的差异是在顶部预测的 0.034 的分界点。在这个阈值之后，随着我们增加顶部预测的百分比，它会以适中的速度下降。在 0.8 左右，情况变得非常糟糕。因此，即使最佳间隔是 0.034，我们也可以将它推得更高一点，以获得更积极的分类观察。

## 24.科尔莫戈罗夫-斯米尔诺夫统计量

如果我们想要绘制 KS 图并获得一个可以用作度量的数字，我们可以查看 KS 图中的所有阈值(数据集临界值),并找到真阳性和真阴性观察值的分布之间的距离(间隔)最大的阈值。

如果存在一个阈值，对于该阈值，上面的所有观察值都是真的正，下面的所有观察值都是真的负，则我们得到完美的 KS 统计值 1.0。

### 如何计算:

```py
from scikitplot.helpers import binary_ks_curve

res = binary_ks_curve(y_true, y_pred_pos)
ks_stat = res[3]

run[“logs/ks_statistic”] = ks_stat

```

### 模型在此指标中的得分情况:

通过使用 KS 统计数据作为衡量标准，我们能够将 BIN-101 列为最佳模型，我们确实希望它是“真正”最佳模型。

### 何时使用:

*   当你的问题是对最相关的观察进行排序/优先排序，并且你同样关心积极和消极的类别时。
*   它可以很好地补充 ROC AUC 分数，ROC AUC 分数衡量模型的排名/排序性能。

## 最后的想法

在这篇博文中，您了解了各种分类指标和性能图表。

我们复习了度量的定义和解释，学习了如何计算它们，并讨论了何时使用它们。

希望有了这些知识，您将完全有能力在未来的项目中处理与度量相关的问题。

![](img/9244906320bb3a7b4063de3b23b8f4a8.png)

为了帮助您最大限度地利用这篇博文中的信息，我准备了://r//n//r//n

看看下面这些吧！//r//n//r//n

## 记录功能

您可以**记录我们为您的机器学习项目覆盖的所有**指标** **和**性能**图表**，并使用我们的 [Python 客户端](https://web.archive.org/web/20221226161840/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display)在 Neptune 中探索它们。**

```py
pip install neptune-client
```

```py
Import neptune.new as neptune

run = neptune.init(...)

run[“logs/score”] = score

```

*   探索应用中的一切:

访问 Neptune [docs](https://web.archive.org/web/20221226161840/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display) 查看您可以在应用程序中记录和显示的内容。

## 二进制分类指标清单

我们已经为您创建了一个不错的备忘单，它将我在这篇博文中浏览的所有内容都放在一个几页长、易于理解的文档中，您可以在需要任何二进制分类指标相关的内容时打印和使用它。

#### 获取您的二元分类指标清单