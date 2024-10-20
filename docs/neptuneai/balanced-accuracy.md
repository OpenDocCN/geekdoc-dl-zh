# 平衡精度:什么时候应该使用它？

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/balanced-accuracy>

当我们训练一个 ML 模型时，我们希望知道它的表现如何，这种表现是用度量来衡量的。直到性能足够好，度量标准令人满意，模型才值得部署，我们必须不断迭代，找到模型既不欠拟合也不过拟合的最佳点(完美的平衡)。

有很多不同的指标来衡量机器学习模型的性能。在本文中，我们将探索基本的度量标准，然后更深入地挖掘平衡的准确性。

## 机器学习中的问题类型

机器学习中有两个广泛的问题:

第一个处理离散值，第二个处理连续值。

分类可以细分为两个更小的类型:

### 多类分类

在多类分类中，类等于或大于三。许多二进制分类使用带有标签的两个类进行操作，并且许多分类器算法可以对其进行建模，而多类分类问题可以通过应用某种策略(即一对一或一对一)使用该二进制分类器来解决。

### 二元分类

二元分类有两个目标标签，大多数时候一个类别处于正常状态，而另一个类别处于异常状态。设想一个欺诈性交易模型，该模型预测交易是否是欺诈性的。这种异常状态(=欺诈性交易)有时在一些数据中表现不足，因此检测可能至关重要，这意味着您可能需要更复杂的指标。

## 什么是评估指标？

初学数据的科学家可能会犯的一个错误是，在构建模型后没有对其进行评估，也就是说，在部署之前不知道他们的模型有多有效，这可能是灾难性的。

评估标准衡量模型在训练后的表现。您建立一个模型，从度量中获得反馈，并进行改进，直到获得您想要的精确度。

选择正确的指标是正确评估 ML 模型的关键。选择单一指标可能不是最佳选择，有时最佳结果来自不同指标的组合。

不同的 ML 用例有不同的度量。在这里，我们将重点关注分类指标。

请记住，指标不同于损失函数。损失函数显示了模型训练期间模型性能的度量。度量用于判断和测量训练后的模型性能。

显示我们模型性能的一个重要工具是混淆矩阵——它不是一个指标，但与指标一样重要。

### 混淆矩阵

混淆矩阵是数据上分类器性能分布的表格。这是一个 N×N 矩阵，用于评估分类模型的性能。它向我们展示了模型执行得有多好，需要改进什么，以及它犯了什么错误。

其中:

*   TP–真阳性(模型正确预测的阳性类别结果)，
*   TN–真阴性(模型的正确预测的阴性类别结果)，
*   FP–假阳性(模型的错误预测阳性类别结果)，
*   FN–假阴性(模型的错误预测的阴性类别结果)。

现在让我们转到指标，从准确性开始。

### 准确(性)

准确性是一种度量标准，它通过将总正确预测除以模型的总预测来总结分类任务的性能。它是所有数据点中正确预测的数据点的数量。

这适用于在混淆矩阵上看到的预测类别，而不是数据点的分数。

***【准确度= (TP + TN) / (TP+FN+FP+TN)***

### 回忆

召回率是一个度量标准，它量化了模型可以做出的所有肯定预测中的正确肯定预测的数量。

***回忆= TP / (TP+FN)。***

召回率是多类分类中所有类的真阳性的总和，除以数据中所有真阳性和假阴性的总和。

***【回忆=总和(TP) /总和(TP+FN)***

回忆也叫敏感。

### 宏观召回

宏观召回衡量每类的平均召回。它用于具有两个以上目标类别的模型，是召回的算术平均值。

***宏调用=(Recall 1+Recall 2+——-Recall in)/n .***

### 精确

精度量化了模型做出的正面预测中正确的正面预测的数量。Precision 计算真阳性的准确度。

***精度= TP/(TP + FP。)***

### f1-分数

F1-score 保持了精确度和召回率之间的平衡。它通常用于类别分布不均匀的情况，但也可以定义为单个测试准确性的统计度量。

***F1 = 2 *([精度*召回]/[精度+召回])***

### ROC_AUC

ROC_AUC 代表“受试者操作者特征 _ 曲线下面积”。它总结了预测模型的真阳性率和假阳性率之间的权衡。当每个类之间的观察值平衡时，ROC 会产生好的结果。

无法从混淆矩阵中的汇总数据计算出此指标。这样做可能会导致不准确和误导的结果。可以使用 ROC 曲线来查看，该曲线显示了真阳性率和假阳性率之间每个可能点的变化。

## 平衡精度

平衡精度用于二进制和多类分类。它是灵敏度和特异性的算术平均值，其用例是当[处理不平衡数据](/web/20221206141839/https://neptune.ai/blog/how-to-deal-with-imbalanced-classification-and-regression-data)时，即当一个目标类比另一个出现得更多时。

### 平衡精度公式

**敏感度:**这也称为真阳性率或召回率，它衡量模型做出的总阳性预测中正确预测的真阳性的比例。

***敏感度= TP / (TP + FN)***

**特异性:**也称为真阴性率，它衡量正确识别的阴性占模型做出的总阴性预测的比例。

***【特异性=TN / (TN + FP)***

要在模型中使用这个函数，您可以从 scikit-learn 中导入它:

```py
from sklearn.metrics import balanced_accuracy_score
bal_acc=balanced_accuracy_score(y_test,y_pred)
```

### 平衡精度二进制分类

二元分类的平衡精度有多好？让我们看看它的用例。

在异常检测中，如处理欺诈性交易数据集，我们知道大多数交易都是合法的，即欺诈性交易与合法交易的比率很小，对于这种不平衡的数据，平衡的准确性是一个很好的性能指标。

假设我们有一个二元分类器，其混淆矩阵如下:

```py
Accuracy = (TP + TN) / (TP+FN+FP+TN) = 20+5000 / (20+30+70+5000)
Accuracy = ~98.05%.

```

这个分数看起来令人印象深刻，但它没有正确处理积极的一栏。

所以，让我们考虑平衡精度，这将说明类中的不平衡。下面是我们的分类器的平衡精度计算:

```py
Sensitivity = TP / (TP + FN) = 20 / (20+30) = 0.4 = 40%
Specificity = TN / (TN + FP) = 5000 / (5000 +70) = ~98.92%.

Balanced Accuracy = (Sensitivity + Specificity) / 2 = 40 + 98.92 / 2 = 69.46%
```

平衡的准确性做了一件伟大的工作，因为我们想要识别我们的分类器中存在的阳性。这使得分数低于准确性预测的分数，因为它给了两个类相同的权重。

### 平衡精度多类分类

正如二元分类一样，平衡精度对于多类分类也是有用的。这里，BA 是在每个类别上获得的回忆的平均值，即每个类别的回忆分数的宏观平均值。因此，对于一个平衡的数据集，分数往往与准确性相同。

让我们用一个例子来说明在不平衡数据中，平衡精度是一个更好的性能指标。假设我们有一个带有混淆矩阵的二元分类器，如下所示:

从每个类获得的 TN、TP、FN、FP 如下所示:

让我们计算一下精确度:

```py
Accuracy = TP + TN / (TP+FP+FN+TN)

TP = 10 + 545 + 11 + 3 = 569
FP = 175 + 104 + 39 + 50 = 368
TN = 695 + 248 + 626 + 874 = 2443
FN = 57 + 40 + 261 + 10 = 368

Accuracy = 569 + 2443 / (569 + 368 + 368 + 2443)
Accuracy = 0.803
```

分数看起来很棒，但是有个问题。集合 P 和 S 是高度不平衡的，模型在预测这一点上做得很差。

让我们考虑平衡精度:

***平衡精度=(recall p+recall q+recall r+recall s)/4。***

对数据中存在的每个类计算召回率(类似于二进制分类)，同时取召回率的算术平均值。

在计算召回时，公式是:

***【回忆= TP / (TP + FN)***

```py
For class P, given in the table above,

Recallp = 10 / (10+57) = 0.054 = 5.4%, 
```

如你所见，这个模型预测 P 类的真阳性率很低。

```py
For class Q
RecallQ = 545 / (545 + 40) = 0.932

For class R,
RecallR = 11 / (11 + 261) = 0.040

For class S,
RecallS = 3 / (3 + 10) = 0.231

Balanced Accuracy = (0.054 + 0.932 + 0.040 + 0.231) / 4 = 1,257 / 4 = 0.3143
```

正如我们所看到的，与准确性相比，这个分数确实很低，因为对所有存在的类应用了相同的权重，而不考虑每个集合中的数据或点。

所以这里我们知道要得到一个更好的分数，应该提供更多的关于 P S .和 R .的数据。

### 平衡精度与分类精度

*   如果我们在数据集中有一个相似的平衡，准确性可以是一个有用的度量。如果不是，那么平衡精度可能是必要的。一个模型可以有高精度但性能差，也可以有低精度但性能好，这可能与精度悖论有关。

考虑下面不平衡分类的混淆矩阵。

看看这个模型的准确性，我们可以说它很高，但是…它不会产生任何结果，因为它的预测能力为零(这个模型只能预测一个类别)。

```py
Binary Accuracy:

Sensitivity= TP / (TP + FN) = 0 / (0+10) = 0
Specificity = TN / (TN + FP) = 190 / (190+0) = 1

Binary Accuracy:
Sensitivity + Specificity / 2 = 0 + 1 / 2
Binary Accuracy = 0.5 = 50%
```

这意味着该模型不预测任何事情，而是将每个观察结果映射到一个随机猜测的答案。

准确性并不能让我们看到模型的问题。

在这里，模型积极因素得到了很好的体现。

所以，在这种情况下，平衡的准确性比准确性更好。如果数据集平衡良好，则精度和平衡精度往往会收敛于相同的值。

### 平衡准确度与 F1 分数

所以你可能想知道平衡准确度和 F1 分数之间的区别，因为两者都用于不平衡分类。所以，我们来考虑一下。

*   F1 保持了精确度和召回率之间的平衡

```py
F1 = 2 * ([precision * recall] / [precision + recall])
Balanced Accuracy = (specificity + recall) / 2

```

*   F1 评分并不关心有多少真阴性正在被分类。当处理不平衡的数据集时，需要注意底片，平衡精度比 F1 更好。
*   在积极因素和消极因素同样重要的情况下，平衡准确度是比 F1 更好的衡量标准。
*   当需要更多关注正面数据时，F1 是不平衡数据的一个很好的评分标准。

考虑一个例子:

在建模期间，数据有 1000 个负样本和 10 个正样本。模型预测 15 个阳性样本(5 个真阳性，10 个假阳性)，其余为阴性样本(990 个真阴性，5 个假阴性)。

f1-得分和平衡准确度将为:

```py
Precision = 5 / 15 = 0.33
Sensitivity = 5 / 10 = 0.5
Specificity = 990 / 1000 = 0.99

F1-score = 2 * (0.5 * 0.33) / (0.5+0.33) = 0.4
Balanced Accuracy = (0.5 + 0.99) / 2 = 0.745
```

你可以看到平衡精度仍然比 F1 更关心数据中的负数。

考虑另一种情况，数据中没有真正的负数:

```py
Precision = 5 / 15 = 0.33
Sensitivity = 5 / 10 = 0.5
Specificity = 0 / 10 = 0

F1-score = 2 * (0.5 * 0.33) / (0.5 + 0.33) = 0.4
Balanced Accuracy = (0.5 + 0) / 2 = 0.25
```

正如我们所看到的，F1 没有任何变化，而当真负值降低时，平衡精度会快速下降。

这表明 F1 分数更重视正面的数据点，而不是平衡的准确性。

### 平衡精度与 ROC_AUC

平衡精度和 roc_auc 有什么不同？

在制作模型之前，您需要考虑以下事项:

*   这是为了什么？
*   它有多少种可能性？
*   数据有多平衡？等等。

Roc_auc 类似于平衡精度，但有一些关键区别:

*   平衡准确度是根据预测类别计算的，roc_auc 是根据每个数据点的预测分数计算的，这些数据点不能通过混淆矩阵计算获得。

*   如果问题是高度不平衡的，平衡精度是比 roc_auc 更好的选择，因为 Roc_auc 是不平衡数据的问题，即当偏斜严重时，因为少量的正确/错误预测会导致分数的巨大变化。

*   如果我们希望在分类中观察到一系列可能性(概率)，那么最好使用 roc_auc，因为它是所有可能阈值的平均值。然而，如果类别是不平衡的，并且分类的目标是输出两个可能的标签，那么平衡的准确度更合适。

*   如果你既关心正负类，又关心稍微不平衡的分类，那么 roc_auc 更好。

## 使用二进制分类实现平衡的准确性

为了更好地理解平衡准确性和其他评分者，我将在一个示例模型中使用这些指标。代码将会在 Jupyter 笔记本上运行。数据集可以从[这里](https://web.archive.org/web/20221206141839/https://www.kaggle.com/c/frauddetectionchallenge/data?select=train.csv)下载。

我们将在这里使用的数据是欺诈检测。我们希望预测交易是否是欺诈性的。

我们的流程将是:

*   加载数据，
*   清理数据，
*   建模，
*   预测。

### 加载数据

像往常一样，我们从导入必要的库和包开始。

```py
Import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

import warnings
warnings.filterwarnings('ignore')

data_file_path='.../train.csv'
Train =pd.read_csv(data_file_path)
```

如您所见，数据既有数字变量，也有分类变量，一些操作将通过这些变量进行。

让我们看看目标中类的分布，即‘欺诈性列’。

让我们来看看剧情。

```py
sns.countplot(train['fradulent'])
plt.title('A plot of transaction')
plt.xlabel('target')
plt.ylabel('number of cases')
plt.show()
```

我们可以看到分布是不平衡的，因此我们进入下一个阶段——清理数据。

### 数据清理

这个数据没有 NAN 值，所以我们可以继续从时间戳中提取有用的信息。

我们将通过下面的代码提取交易的年份和时间:

```py
train['hour']=train['transaction time'].str.split('T',expand=True)[1].str.split(':',expand=True)[0]
train['year']=train['transaction time'].str.split('-',expand=True)[0]

```

#### 编码

下一步是将字符串(分类)变量编码成数字格式。我们将对其进行标记和编码。Sklearn 也为此提供了一个名为 LabelEncoder 的工具。

```py
encode={'account type':{'saving':0,'current':1},

       'credit card type':{'master':0,'verve':1},

       'occupation':{'clergy':0,'accounting':1,'lecturer':2,'banking':3,
                     'doctor':4,'farmer':5,'lawyer':6, 'musician':7},

        'marital_status':{'married':0,'single':1,'unknown':2},

        'year':{'2014':0,'2015':1,'2016':2,'2017':3}

       }
train=train.replace(encode)
```

由于现在已经进行了编码，数据应该如下所示:

True / False 值列不需要编码，因为它们是布尔值。

#### 设置索引和删除列

数据中不重要的列需要放到下面:

```py
train.drop(['transaction time','id'],axis=1,inplace=True)
```

#### 数据缩放

我们需要调整数据，以确保每个特征的权重相同。为了缩放这些数据，我们将使用 StandardScaler。

```py
X=train.drop(['fradulent'],axis=1)
y=train['fradulent']

from sklearn.preprocessing import StandardScaler
std= StandardScaler()
X=std.fit_transform(X)

```

### 建模

在拟合之前，我们需要将数据分为测试集和训练集，这使我们能够在部署之前了解模型在测试数据上的表现。

```py
run = neptune.init(project=binaryaccuracy,
                   api_token=api_token)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
```

在这种分割之后，我们现在可以在查看计算图时，用我们到目前为止讨论过的评分标准来拟合和评分我们的模型。

```py
for epoch in range(100,3000,100):
              gb=GradientBoostingClassifier(n_estimators=epoch,learning_rate=epoch/5000)
    gb.fit(x_train,y_train)
    y_pred=gb.predict(x_test)
    roc_auc=roc_auc_score(y_test,y_pred)

    acc=accuracy_score(y_test,y_pred)
    bal_acc=balanced_accuracy_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)

    run["train/accuracy"].log(acc)
    run['train/f1'].log( f1)
    run['train/roc_auc'].log(roc_auc)
    run['train/bal_acc'].log(bal_acc)
```

查看上面的图表，我们可以看到模型预测如何基于历元和学习率迭代波动。

*   虽然准确性最初很高，但与其他记分员相比，它逐渐下降，没有完美的下降。它没有很好地处理混淆矩阵上的数据表示。
*   F1 的分数在这里很低，因为它偏向于数据中的负值。然而，在上面的数据中，正面和负面都很重要。
*   roc_auc 得分是一个没有偏见的得分，数据中的两个标签被给予同等的优先级。与具有 1:100 比率的目标标记的一些数据相比，该数据偏斜度不是很大，因此 ROC_AUC 在这里表现得更好。
*   总的来说，平衡准确性在数据评分方面做得很好，因为模型并不完美，它仍然可以得到更好的预测。

要查看预测并存储在元数据中，请使用以下代码:

```py
df = pd.DataFrame(data={'y_test': y_test, 'y_pred': y_pred, 'y_pred_probability': y_pred_proba.max(axis=1)})

run['test/predictions'] = neptune.types.File.as_html(df)

```

记录元数据并查看绘图。图表中要记录和比较的指标有:acc(准确度)、f1(f1-得分)、roc_auc 得分、bal_acc(平衡准确度)。

此函数创建绘图并将其记录到元数据中，您可以从 scikitplot.metrics 中获得它处理的各种曲线。

```py
def plot_curve(graph):
    fig, ax = plt.subplots()
    graph(y_test, y_pred_proba,ax=ax)
    run['charts/{}'.format(graph)] = neptune.types.File.as_image(fig)

plot_curve(plot_roc)
```

## 平衡准确性的问题

我们已经讨论了很多关于平衡精度的问题，但是在一些情况下，即使是最简单的指标也是绝对合适的。

*   当数据平衡时。
*   当模型不仅仅是映射到(0，1)结果，而是提供广泛的可能结果(概率)时。
*   当模型更倾向于正面而非负面时。
*   在多类分类中，一些类的重要性不如另一些类，由于所有类都具有相同的权重而不考虑类的频率，因此可能会出现偏差。这样做可能会导致错误，因为我们的模型应该提供解决方案，而不是相反。

*   当有一个很高的偏差或者一些类比其他类更重要时，那么平衡的准确性就不是模型的完美评判。

## 摘要

研究和构建机器学习模型可能很有趣，但如果没有使用正确的度量标准，也会非常令人沮丧。正确的度量标准和工具非常重要，因为它们向您展示了您是否正确地解决了手头的问题。

这不仅仅是关于一个模型有多棒，更重要的是解决它被认为应该解决的问题。

平衡精度在某些方面是很好的，例如当类不平衡时，但是它也有缺点。深入理解它，会给你所需的知识，让你知道该不该用它。

感谢阅读！