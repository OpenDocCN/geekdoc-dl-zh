# 银行业的数据科学:欺诈检测

> 原文：<https://web.archive.org/web/20221129050302/https://www.datacamp.com/blog/data-science-in-banking>

## ![Fintech Concept Illustration](img/d3b961ca970a339eeea9df5ad8cfa07b.png)

*   [银行业中的数据科学用例:检测欺诈](#data-science-use-case-in-banking:-detecting-fraud)
*   [准备数据集](#preparing-the-dataset)
*   [计算数据集中的欺诈](#calculating-fraud-in-the-dataset)
*   [使用 SMOTE 重新平衡数据](#using-smote-to-re-balance-the-data)
*   [应用逻辑回归](#applying-logistic-regression)
*   [结论](#conclusion)

银行业是历史上收集大量结构化数据的幸运领域之一，也是最先应用数据科学技术的领域之一。

数据科学在银行业是如何运用的？如今，数据已经成为这个领域最有价值的资产。数据科学是银行跟上竞争对手、吸引更多客户、提高现有客户忠诚度、做出更高效的数据驱动型决策、增强业务能力、提高运营效率、改进现有服务/产品并推出新产品、增强安全性以及获得更多收入的必要条件。不足为奇的是，大部分数据科学工作需求来自银行业。

数据科学让银行业能够成功执行众多任务，包括:

*   投资风险分析
*   客户终身价值预测
*   客户细分
*   客户流失率预测
*   个性化营销
*   客户情绪分析
*   虚拟助手和聊天机器人

下面，我们将详细了解银行业中最常见的数据科学用例之一。

## 银行业中的数据科学用例:检测欺诈

欺诈活动不仅在银行业，而且在政府、保险、公共部门、销售和医疗保健等许多其他领域都是一个具有挑战性的问题。任何处理大量在线交易的企业都有很大的欺诈风险。金融犯罪有多种形式，包括欺诈性信用卡交易、伪造银行支票、逃税、洗钱、网络攻击、客户账户盗窃、合成身份、虚假申请和诈骗。

欺诈检测是一套主动措施，用于识别和防止欺诈活动和财务损失。其主要分析技术可分为两组:

*   统计:统计参数计算、回归、概率分布、数据匹配
*   人工智能(AI):数据挖掘、机器学习、深度学习

机器学习是欺诈检测的重要支柱。它的工具包提供了两种方法:

*   监督方法:k 近邻、逻辑回归、支持向量机、决策树、随机森林、时间序列分析、神经网络等。
*   无监督方法:聚类分析、链接分析、自组织映射、主成分分析、异常识别等。

欺诈检测没有通用可靠的机器学习算法。相反，对于真实世界的数据科学用例，通常会测试几种技术或它们的组合，计算模型预测准确性，并选择最佳方法。

欺诈检测系统的主要挑战是快速适应不断变化的欺诈模式和欺诈者的策略，并及时发现新的和日益复杂的计划。欺诈案件总是占少数，并且在真实交易中隐藏得很好。

## 准备数据集

让我们使用 Python 编程语言来探索信用卡欺诈检测的机器学习实现。我们将在 creditcard_data 数据集上工作，这是从关于[信用卡欺诈检测](https://web.archive.org/web/20220524184131/http://www.kaggle.com/mlg-ulb/creditcardfraud)的 Kaggle 数据集修改而来的样本。原始数据代表了 2013 年 9 月两天内欧洲持卡人的信用卡交易。

让我们导入数据并快速查看一下:

```py
import pandas as pd

creditcard_data = pd.read_csv('creditcard_data.csv', index_col=0)
print(creditcard_data.info())
print('\n')
pd.options.display.max_columns = len(creditcard_data)
print(creditcard_data.head(3))
```

```py
<class 'pandas.core.frame.DataFrame'>
Int64Index: 5050 entries, 0 to 5049
Data columns (total 30 columns):
#   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
0   V1      5050 non-null   float64
1   V2      5050 non-null   float64
2   V3      5050 non-null   float64
3   V4      5050 non-null   float64
4   V5      5050 non-null   float64
5   V6      5050 non-null   float64
6   V7      5050 non-null   float64
7   V8      5050 non-null   float64
8   V9      5050 non-null   float64
9   V10     5050 non-null   float64
10  V11     5050 non-null   float64
11  V12     5050 non-null   float64
12  V13     5050 non-null   float64
13  V14     5050 non-null   float64
14  V15     5050 non-null   float64
15  V16     5050 non-null   float64
16  V17     5050 non-null   float64
17  V18     5050 non-null   float64
18  V19     5050 non-null   float64
19  V20     5050 non-null   float64
20  V21     5050 non-null   float64
21  V22     5050 non-null   float64
22  V23     5050 non-null   float64
23  V24     5050 non-null   float64
24  V25     5050 non-null   float64
25  V26     5050 non-null   float64
26  V27     5050 non-null   float64
27  V28     5050 non-null   float64
28  Amount  5050 non-null   float64
29  Class   5050 non-null   int64 
dtypes: float64(29), int64(1)
memory usage: 1.2 MB

        V1        V2        V3        V4        V5        V6        V7  \
0  1.725265 -1.337256 -1.012687 -0.361656 -1.431611 -1.098681 -0.842274  
1  0.683254 -1.681875  0.533349 -0.326064 -1.455603  0.101832 -0.520590  
2  1.067973 -0.656667  1.029738  0.253899 -1.172715  0.073232 -0.745771  

        V8        V9       V10       V11       V12       V13       V14  \
0 -0.026594 -0.032409  0.215113  1.618952 -0.654046 -1.442665 -1.546538  
1  0.114036 -0.601760  0.444011  1.521570  0.499202 -0.127849 -0.237253  
2  0.249803  1.383057 -0.483771 -0.782780  0.005242 -1.273288 -0.269260  

        V15       V16       V17       V18       V19       V20       V21  \
0 -0.230008  1.785539  1.419793  0.071666  0.233031  0.275911  0.414524  
1 -0.752351  0.667190  0.724785 -1.736615  0.702088  0.638186  0.116898  
2  0.091287 -0.347973  0.495328 -0.925949  0.099138 -0.083859 -0.189315  

        V22       V23       V24       V25       V26       V27       V28  \
0  0.793434  0.028887  0.419421 -0.367529 -0.155634 -0.015768  0.010790  
1 -0.304605 -0.125547  0.244848  0.069163 -0.460712 -0.017068  0.063542  
2 -0.426743  0.079539  0.129692  0.002778  0.970498 -0.035056  0.017313  

  Amount  Class 
0  189.00      0 
1  315.17      0 
2   59.98      0 
```

数据集包含以下变量:

*   数字编码的变量 V1 到 V28 是从 PCA 变换中获得的主要成分。由于保密问题，没有提供关于原始功能的背景信息。
*   Amount 变量表示交易金额。
*   Class 变量显示交易是欺诈(1)还是非欺诈(0)。

就其性质而言，在所有交易中，欺诈事件幸运地是极少数。然而，当数据集中包含的不同类或多或少同等存在时，机器学习算法通常工作得最好。否则，很少有数据可以借鉴。这个问题叫做阶级不平衡。

## 计算数据集中的欺诈

让我们计算欺诈交易占数据集中交易总数的百分比:

```py
round(creditcard_data['Class'].value_counts()*100/len(creditcard_data)).convert_dtypes()
```

```py
0    99
1     1
Name: Class, dtype: Int64
```

并创建一个图表，将欺诈可视化为非欺诈数据点:

```py
import matplotlib.pyplot as plt
import numpy as np

def prep_data(df):
    X = df.iloc[:, 1:28]
    X = np.array(X).astype(float)
    y = df.iloc[:, 29]
    y = np.array(y).astype(float)
    return X, y

def plot_data(X, y):
    plt.scatter(X[y==0, 0], X[y==0, 1], label='Class #0', alpha=0.5, linewidth=0.15)
    plt.scatter(X[y==1, 0], X[y==1, 1], label='Class #1', alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    return plt.show()

X, y = prep_data(creditcard_data)

plot_data(X, y)
```

![Fraudulent Transactions](img/94bda1bf3c18545be32d0e651731748e.png)

## 使用 SMOTE 重新平衡数据

我们现在可以确认，欺诈交易的比例非常低，我们有一个阶级不平衡的问题。为了解决这个问题，我们可以使用合成少数过采样技术(SMOTE)来重新平衡数据。与随机过采样不同，SMOTE 稍微复杂一些，因为它不只是创建观察值的精确副本。相反，它使用欺诈案例的最近邻居的特征来创建新的合成样本，这些样本与少数类中现有的观察结果非常相似。让我们将 SMOTE 应用于我们的信用卡数据:

```py
from imblearn.over_sampling import SMOTE

method = SMOTE()
X_resampled, y_resampled = method.fit_resample(X, y)
plot_data(X_resampled, y_resampled)
```

![SMOTE Graph](img/e9f5014b36072bf01f7f26e3c7679c5a.png)

正如我们所看到的，使用 SMOTE 突然给了我们更多关于少数民族的观察。为了更好地了解这种方法的结果，我们将把它们与原始数据进行比较:

```py
def compare_plot(X, y, X_resampled, y_resampled, method):
    f, (ax1, ax2) = plt.subplots(1, 2)
    c0 = ax1.scatter(X[y==0, 0], X[y==0, 1], label='Class #0',alpha=0.5)
    c1 = ax1.scatter(X[y==1, 0], X[y==1, 1], label='Class #1',alpha=0.5, c='r')
    ax1.set_title('Original set')
    ax2.scatter(X_resampled[y_resampled==0, 0], X_resampled[y_resampled==0, 1], label='Class #0', alpha=.5)
    ax2.scatter(X_resampled[y_resampled==1, 0], X_resampled[y_resampled==1, 1], label='Class #1', alpha=.5,c='r')
    ax2.set_title(method)
    plt.figlegend((c0, c1), ('Class #0', 'Class #1'), loc='lower center', ncol=2, labelspacing=0.)
    plt.tight_layout(pad=3)
    return plt.show()

print(f'Original set:\n'
      f'{pd.value_counts(pd.Series(y))}\n\n'
      f'SMOTE:\n'
      f'{pd.value_counts(pd.Series(y_resampled))}\n')

compare_plot(X, y, X_resampled, y_resampled, method='SMOTE')
```

```py
Original set:
0.0    5000
1.0      50
dtype: int64

SMOTE:
0.0    5000
1.0    5000
dtype: int64
```

![Smote Method Graph](img/1505afa1edab5cfcde4b9baf158f94ff.png)

因此，SMOTE 方法完全平衡了我们的数据，现在少数类的大小与多数类相等。

我们将很快回到 SMOTE 方法的实际应用，但是现在，让我们回到原始数据并尝试检测欺诈案例。按照“老派”的方式，我们必须创建一些规则来捕捉欺诈。例如，这些规则可能涉及异常的交易地点或可疑的频繁交易。这个想法是基于普通统计定义阈值，通常基于观察的平均值，并在我们的特征上使用这些阈值来检测欺诈。

```py
print(creditcard_data.groupby('Class').mean().round(3)[['V1', 'V3']])
```

```py
 V1     V3
Class             
0      0.035  0.037
1     -4.985 -7.294
```

在我们的特殊情况下，让我们应用以下条件:V1 < -3 和 V3 < -5。然后，为了评估这种方法的性能，我们将标记的欺诈案例与实际案例进行比较:

```py
creditcard_data['flag_as_fraud'] = np.where(np.logical_and(creditcard_data['V1']<-3, creditcard_data['V3']<-5), 1, 0)
print(pd.crosstab(creditcard_data['Class'], creditcard_data['flag_as_fraud'], rownames=['Actual Fraud'], colnames=['Flagged Fraud']))
```

```py
Flagged Fraud     0   1
Actual Fraud          
0              4984  16
1                28  22
```

## 应用逻辑回归

我们检测到了 50 个欺诈案例中的 22 个，但无法检测到另外 28 个，并且得到了 16 个误报。让我们看看使用机器学习技术是否能击败这些结果。

我们现在将对我们的信用卡数据实施简单的逻辑回归分类算法，以识别欺诈事件，然后在混淆矩阵上显示结果:

```py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print(pd.crosstab(y_test, predictions, rownames=['Actual Fraud'], colnames=['Flagged Fraud']))
```

```py
Flagged Fraud   0.0  1.0
Actual Fraud           
0.0            1504    1
1.0             1      9
```

值得注意的是，这里我们在混淆矩阵中要查看的观察值较少，因为我们仅使用测试集来计算模型结果，即仅占整个数据集的 30%。

我们发现了更高比例的欺诈案件:90%(10 起中有 9 起)，而之前的结果是 44%(50 起中有 22 起)。我们得到的误报也比以前少得多，所以这是一个进步。

现在，让我们回到之前讨论的类别不平衡问题，并探索我们是否可以通过将逻辑回归模型与 SMOTE 重采样方法相结合来进一步增强预测结果。为了高效地一次性完成，我们需要定义一个管道，并在我们的数据上运行它:

```py
from imblearn.pipeline import Pipeline

# Defining which resampling method and which ML model to use in the pipeline
resampling = SMOTE()
lr = LogisticRegression()

pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', lr)])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print(pd.crosstab(y_test, predictions, rownames=['Actual Fraud'], colnames=['Flagged Fraud']))
```

```py
Flagged Fraud   0.0  1.0
Actual Fraud           
0.0            1496    9
1.0               1    9
```

正如我们所看到的，在我们的案例中，SMOTE 没有带来任何改进:我们仍然捕获了 90%的欺诈事件，而且，我们的误报数量略高。这里的解释是，重采样不一定在所有情况下都能得到更好的结果。当欺诈案例在数据中非常分散时，它们的最近邻居不一定也是欺诈案例，因此使用 SMOTE 会引入偏差。

## 结论

作为一种可能的方法，为了增加逻辑回归模型的准确性，我们可以调整一些算法参数。也可以考虑 k 倍交叉验证，而不仅仅是将数据集分成两部分。最后，我们可以尝试一些其他的机器学习算法(例如，决策树或随机森林)，看看它们是否能给出更好的结果。

如果您想了解更多关于欺诈检测模型实现的理论和技术方面的知识，您可以探索 Python 课程中的[欺诈检测的资料。](https://web.archive.org/web/20220524184131/https://app.datacamp.com/learn/courses/fraud-detection-in-python)