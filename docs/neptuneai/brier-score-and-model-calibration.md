# Brier Score:了解模型校准

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/brier-score-and-model-calibration>

当你的天气应用程序中下雨的概率低于 10%时，你是否遇到过风暴？嗯，这很好地展示了一个没有[校准好的模型](https://web.archive.org/web/20221206143046/https://stats.stackexchange.com/questions/270508/meaning-of-model-calibration)(也称为校准不良模型，或 Brier 分数非常高的模型)会如何破坏你的计划。

在构建预测模型时，您通过计算[不同的评估指标](/web/20221206143046/https://neptune.ai/blog/evaluation-metrics-binary-classification)来考虑其预测能力。其中一些是常见的，如准确性和精确度。但是其他的，像上面天气预报模型中的 Brier 分数，经常被忽略。

在本教程中，您将获得对 **Brier 评分和校准的简单介绍性解释，这是用于评估统计预测性能的最重要的概念之一。**

什么是欧石南乐谱？

## Brier Score 评估[概率预测](https://web.archive.org/web/20221206143046/https://en.wikipedia.org/wiki/Brier_score)的准确性。

假设我们有两个正确预测晴天的模型。一个概率为 0.51，另一个概率为 0.93。它们都是正确的，并且具有相同的精度(假设阈值为 0.5)，但是第二个模型感觉更好，对吗？这就是 Brier score 出现的原因。

当我们处理只能取有限数量的值的变量时(我们也可以称它们为类别或标签)，它特别有用。

例如，紧急程度(采用四个值:绿色、黄色、橙色和红色)，或者明天是雨天、阴天还是晴天，或者是否将超过阈值。

Brier 评分更像是一个成本函数。较低的值意味着准确的预测，反之亦然。处理这个概念的主要目标是减少它。

Brier 分数的数学公式取决于预测变量的类型。如果我们正在开发二元预测，则得分由下式给出:

其中 p 是事件发生的预测概率，如果事件发生，oi 项等于 1，否则等于 0。

让我们举一个简单的例子来理解这个概念。让我们考虑一下事件 **A= "明天是晴天"**。

如果您预测事件 A 将以 100%的概率发生，并且该事件发生了(下一个是晴天，这意味着 o=1)，Brier 得分等于:

这是可能的最低值。换句话说:我们能达到的最好的情况。

如果我们以同样的概率预测了同样的事件，但是该事件没有发生，那么 Brier 的得分是:

假设你预测 A 事件会以另一种概率发生，假设 60%。如果事件在现实中没有发生，Brier 的分数将是:

您可能已经注意到，Brier 分数是概率域中的一个距离。也就是说:**这个分值越低，预测越好。**

完美的预测会得到 0 分。最差的分数是 1。这是一个综合标准，提供了关于预测模型的准确性、稳健性和可解释性的综合信息。

**虚线代表最坏情况**(如果事件发生，圆圈等于 1)。

什么是概率校准？

## 概率校准是对模型进行后处理，以提高其[概率估计](https://web.archive.org/web/20221206143046/https://www.webpages.uidaho.edu/niatt_labmanual/Chapters/traveldemandforecasting/professionalpractice/ModelCalibrationAndValidation.htm)。它有助于我们比较具有相同准确性或其他标准评估指标的两个模型。

当置信度为 p 的类的预测在 100%的时间内正确时，我们说模型被很好地校准。为了说明这种校准效果，让我们假设您有一个模型，该模型对每个患者的癌症预测得分为 70%(满分为 100 分)。如果你的模型校准良好，我们将有 70 名癌症患者，如果校准不当，我们将有更多(或更少)的患者。因此，这两种型号的区别在于:

模型的准确率为 70%，每个预测的置信度为 0.7 =校准良好。

*   一个模型的准确率为 70%，每次预测的置信度为 0.9 =校准不良。
*   对于完美的校准，预测概率和阳性分数之间的关系如下:

这种关系的表达式由下式给出:

上图是一个型号的**可靠性图。我们可以使用 scikit-learn 绘制它，如下所示:**

**绘制多个模型的可靠性曲线使我们能够选择最佳模型，不仅基于其准确性，还基于其校准**。

```py
import sklearn
from sklearn.calibration import calibration_curve
import matplotlib.lines as line
import matplotlib.pyplot as plt

x, y=calibration_curve(y_true, y_prob)

plt.plot(x,y)
ref = line.Line2D([0, 1], [0, 1], color='black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.suptitle('Calibration – Neptune.ai')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Fraction of positive')
plt.legend()
plt.show()

```

在下图中，我们可以排除 SVC (0.163)模型，因为它远未得到很好的校准。

如果我们想要一个数值来检查我们模型的校准，我们可以使用理论上给出的校准误差:

什么时候应该使用布赖尔乐谱？

## 评估机器学习模型的性能很重要，但评估[现实世界应用](https://web.archive.org/web/20221206143046/https://timvangelder.com/2015/05/18/brier-score-composition-a-mini-tutorial/)的预测是不够的。

我们经常担心:

模型对其预测的信心，

*   它的误差分布，
*   以及如何进行概率估计。
*   在这种情况下，我们需要使用其他性能因素。Brier score 就是一个例子。

这种类型的性能分数**专门用于高风险应用**。该分数允许我们不将模型结果视为真实概率，而是超越原始结果并检查模型校准，这对于避免错误决策或错误解释非常重要。

**需要校准好的概率/模型校准的例子**

### 假设您想要构建一个模型，通过用户点击新闻页面的机会向用户显示新闻页面。如果用户点击建议项目的机会很高，则该项目显示在主页上。否则，我们显示另一个机会更高的项目。

在这种问题中，我们并不真正关心点击的确切机会是多少，而只关心在所有现有项目中哪个项目的机会最高。**模型校准在这里并不重要**。重要的是哪一个被点击的概率(几率)最高。

另一方面，考虑这样一个问题，我们建立了一个模型，根据一些分析的结果预测患某种特定疾病的概率。概率的精确值在这里至关重要，因为它影响医生的决策和患者的健康。

通常，当结果显示模型有高概率的错误(或当模型不输出概率估计值时的预测分数，例如[随机森林](/web/20221206143046/https://neptune.ai/blog/random-forest-regression-when-does-it-fail-and-why))时，校准用于改进模型。

当然，你也可以看看其他将预测分数作为输入的指标，比如 [ROC AUC 分数](/web/20221206143046/https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc)，但它们通常不会关注正确校准的概率。比如 ROC AUC 侧重于排名预测。

好了，现在我们可以知道什么时候模型没有校准好，但是我们能做些什么呢？我们如何校准它？

概率校准方法

## 人们使用大量的[校准方法](https://web.archive.org/web/20221206143046/https://medium.com/@kingsubham27/calibration-techniques-and-its-importance-in-machine-learning-71bec997b661)。我们将重点介绍两种最流行的方法:

**普氏标度**

### 普拉特缩放通常用于校准我们已经建立的模型。该方法的原理是基于将我们的分类模型的输出转换成概率分布。

我们的模型不仅会给出一个分类结果(标签或类别)，还会给出结果本身一定程度的确定性。

不是返回类 1 作为结果，而是返回这个类预测正确的概率。不幸的是，一些分类模型(如 SVM)不返回概率值，或者给出很差的概率估计。

这就是为什么我们使用特定的转换来校准我们的模型，并将结果转换为概率。

**为了使用普拉特标度法**，我们通常训练我们的模型，然后训练附加 sigmoid 函数的参数，以将模型输出映射成概率。

你可以使用逻辑回归拟合模型的输出。

**保序回归**

```py
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(p_out, y_out)
calib_p=model.predict(p_test)[:,1]

```

### 保序回归与普拉特标度做同样的事情——它们都将模型输出转换成概率，因此对其进行校准。

有什么区别？

**Platt Scaling 使用 sigmoid 形状来校准模型**，这意味着我们的概率分布中存在 sigmoid 形状的失真。

**保序回归**是一种更强大的校准方法，可以校正任何单调失真。它**将一个非参数函数投射到一组递增函数**(单调)。

**如果数据集很小，不建议使用保序回归，**因为它很容易过度拟合。

为了实现这种方法，我们将再次使用 sklearn(我们假设您已经构建并训练了您的“未校准模型”):

```py
from sklearn.linear_model import IsotonicRegression

model=IsotonicRegression()
model.fit(p_out, y_out)
calib_p=model.transform(p_test)

```

在实际例子中使用模型校准

## 我们来练习一下吧！

为了保持它的唯一性，我们将使用来自 sklearn 的`make_classification`生成两个类。

之后，我们将在数据集上训练和拟合 SVM 分类器。

我们刚刚创建了独特的随机数据，并将其分为训练集和测试集。

```py
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_classification(n_samples=2500, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
```

现在，我们构建 SVC 分类器，并使其适合训练集:

**我们来预测一下结果:**

```py
from sklearn.svm import SVC

svc_model=SVC()
svc_model.fit(X_train, y_train)

```

接下来，让我们绘制之前讨论过的校准曲线:

```py
prob=svc_model.decision_function(X_test)

```

**上面代码的结果是校准曲线或可靠性曲线:**

```py
from sklearn.calibration import calibration_curve
x_p, y_p=calibration_curve(y_test, prob, n_bins=10, normalize=’True’)
plt.plot([0, 1], [0, 1])
plt.plot(x_p, y_p)
plt.show()

```

这表明我们的分类器校准不良(校准参考是蓝线)。

现在让我们来计算这个校准不当的模型的 Brier 分数:

为此，我们得到 ROC AUC 分数等于 0.89(意味着良好的分类)和 Brief 分数等于 0.49。相当高！

```py
from sklearn.metrics import brier_score_loss, roc_auc_score

y_pred = svc_model.predict(X_test)
brier_score_loss(y_test, y_pred, pos_label=2)
roc_auc_score(y_test, y_pred)

```

这解释了该模型校准不当的事实。我们该怎么办？

**让我们再次使用 sklearn 校准我们的模型。我们将应用普拉特标度**(使用 sigmoid 分布校准)。Sklearn 提供了一个预定义的函数来完成这项工作:`CalibratedClassifierCV`。

**模型校准或校准后的可靠性曲线如下:**

```py
from sklearn.calibration import CalibratedClassifierCV
calib_model = CalibratedClassifierCV(svc_model, method='sigmoid', cv=5) calib_model.fit(X_train, y_train)
prob = calib_model.predict_proba(X_test)[:, 1]

x_p, y_p = calibration_curve(y_test, prob, n_bins=10, normalize='True')
plt.plot([0, 1], [0, 1])
plt.plot(x_p, y_p)
plt.show()

```

可靠性曲线显示了向校准参考的趋势(理想情况)。为了进行更多的验证，我们可以使用与之前相同的数字指标。

在以下代码中，我们计算校准模型的 Brier 得分和 ROC AUC 得分:

Brier 评分在校准后下降(从 0.495 降至 0.35)，ROC AUC 评分增加，从 0.89 增至 0.91。

```py
from sklearn.metrics import brier_score_loss, roc_auc_score

y_pred = calib_model.predict(X_test)
brier_score_loss(y_test, y_pred, pos_label=2)
roc_auc_score(y_test, y_pred)

```

我们注意到，您可能想要在一个等待设置上校准您的模型。在这种情况下，我们将数据集分成三部分:

我们在训练集上拟合模型(第一部分)。

*   我们在校准集上校准模型(第二部分)。
*   我们在测试集上测试模型(第三部分)。
*   最后的想法

校准您的模型是提高其预测性能的关键步骤，尤其是如果您关心具有较低 Brier 分数的“好”概率预测。

## 然而，你应该记住，改进的校准概率是否有助于基于类别或概率的更好预测并不明显。

这可能取决于您用来测试预测的特定评估指标。根据一些论文，SVM、决策树和随机森林在校准后更有可能得到改进(在我们的示例中，我们使用了支持向量分类器)。

所以要一如既往地小心行事。

我希望，读完这篇文章后，你对 Brier score 和模型校准有一个很好的理解，并且你将能够在你的 ML 项目中使用它。

感谢阅读！

I hope that, after reading this article you have a good understanding of what Brier score and model calibration are and you’ll be able to use that in your ML projects.

参考

[1] Brier GW。“用概率表示的预测的验证”。周一天气 1950 年修订。

## [2]涅廷·T·拉夫特里·艾。严格适当的评分规则、预测和评估。J Am 统计协会 2007。

[3]你的模型为现实世界做好准备了吗？–Inbar Naor–2018 年以色列皮肯大会

[4]Python 中的校准图指南–Chang Hsin Lee，2018 年 2 月。

[3] Is your model ready for the real world? – Inbar Naor – PyCon Israel Conference 2018

[4] A guide to calibration plots in Python – Chang Hsin Lee, February 2018.