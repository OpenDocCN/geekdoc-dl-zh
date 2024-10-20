# 数据清理过程:应该是什么样子？

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/data-cleaning-process>

数据清理就是通过修改、添加或删除数据来为分析做准备的过程。这个过程通常也称为数据预处理。对于数据科学家和机器学习工程师来说，在数据清理领域非常熟练非常重要，因为他们或他们的模型将从数据中获得的所有见解都直接取决于对数据进行的预处理的质量。

在这篇博文中，就分类特征而言，我们将仔细研究数据清理的概念。

经过以上这几点，我尽量解释，给出图片参考来辅助理解，代码演示等等。

在本文结束时，您不仅会知道什么是数据清理，还会对数据清理中涉及的步骤、如何进行这些步骤以及最重要的是如何充分利用您的数据有一个完美的理解。所以，让我们继续前进。

## 为什么我们需要清理我们的数据？

有时，在我们能够从中提取有用的信息之前，需要对数据进行清理/预处理。大多数现实生活中的数据都有很多不一致的地方，比如缺失值、非信息特征等等，因此，在使用数据之前，我们需要不断地清理数据，以便从中获得最佳效果。

举个例子，机器学习过程只处理数字。如果我们打算使用具有文本、日期时间特征和其他非数字特征的数据，我们需要找到一种用数字表示它们而不丢失它们所携带的信息的方法。

比如说:

简单的逻辑回归模型使用线性函数简单地将因变量映射到自变量，例如: *y= wx + b*

其中:

w 是重量，可以是任何数字。

而 *b* 是 bias，也是一个数。

如果给你一组数据。【小，中，大】，你发现无法计算:*y = w *小+b*

但是如果你将小的编码为 1，中的编码为 2，大的编码为 3，从而将你的数据转换为[1，2，3]，你会发现你不仅能够计算=*w * 1+b，*，而且其中编码的信息仍然被保留。像这样的情况以及更多的情况是我们在数据清理和预处理方面的业务。

### 数据清洗有哪些步骤？

数据清理只是我们在准备数据进行分析的过程中对数据执行的一系列操作的统称。

数据清理的一些步骤是:

*   处理缺失值
*   编码分类特征
*   离群点检测
*   转换
*   等等。

## 处理缺失值

预处理数据时遇到的最常见的问题之一是数据中的[值缺失](https://web.archive.org/web/20221206100436/https://en.wikipedia.org/wiki/Missing_data#Missing_completely_at_random)的问题。它们非常常见，可能是由于:

1.  填写数据的人有意或无意的遗漏，或者根本不适用于他们。
2.  将数据输入计算机的人的疏忽。

数据科学家如果不小心，可能会从包含缺失值的数据中得出错误的推断，这就是为什么我们需要研究这种现象，并学会有效地解决它。

早期的机器学习库，如 scikit learn，不允许将缺失值传递给它们。这带来了一定的挑战，因为在将数据传递给 scikit learn ML 模型之前，数据科学家需要迭代许多方法来处理缺失值。最近的机器学习模型和平台已经消除了这一障碍，尤其是梯度增强机器，如 Xgboost，Catboost，LightGBM 等。

我个人喜欢的一种方法是 Catboost 的方法，它允许您在三个选项(禁止、最小和最大)中进行选择。“禁止”将缺失值视为错误，而“最小”将其设置为小于特定要素(列)中的所有其他值。通过这种方式，我们可以确定，在特征的基本决策树分割过程中，这些丢失的值也被[视为](https://web.archive.org/web/20221206100436/https://catboost.ai/docs/concepts/algorithm-missing-values-processing.html)。

LightGBM 和 [XGboost](https://web.archive.org/web/20221206100436/https://stackoverflow.com/questions/37617390/xgboost-handling-of-missing-values-for-split-candidate-search) 也以非常方便的方式处理缺失值。但是，在预处理的时候，尽量多尝试一些方法。处理你自己，使用一些我们上面讨论过的库，一些其他的工具，比如 automl，等等。

缺失值通常分为三类:

1.  **完全随机缺失(*****)**:缺失值完全随机缺失，如果我们没有任何信息、原因或任何可以帮助计算它的东西。例如，“我是一个非常困倦的研究生，不小心把咖啡打翻在我们收集的一些书面调查上，所以我们丢失了所有本来会有的数据。”*
2.  ***随机缺失(** ***MAR*** **)** :一个缺失的值，如果我们有信息、理由或任何东西(特别是从其他已知值)可以帮助计算它，那么它就是随机缺失的。例如，“我管理一项调查，其中包括一个关于某人收入的问题。女性不太可能回答关于收入的问题。”*
3.  ***不随意遗漏(*****)**。丢失的变量的值与它丢失的原因有关。例如，“如果我管理一项包含某人收入问题的调查。低收入人群回答这个问题的可能性要小得多”。因此，我们知道为什么这些数据点可能会丢失**

 **## 我们如何在 Python 中检测缺失值？

在我们处理丢失的值之前，我们应该学习如何检测它们，并根据丢失值的数量、我们拥有的数据量等等来决定如何处理我们的数据。我喜欢并经常使用的一个实现是为一个列设置一个阈值，以决定它是可修复的还是不可修复的，哈哈。(50–60%对我来说是这样)。

下面，您将看到一个函数，它实现了我们在上一段中讨论的思想。

```py
def missing_removal(df, thresh, confirm= None):
    holder= {}
    for col in df.columns:
        rate= df[col].isnull().sum() / df.shape[0]
        if rate > thresh:
            holder[col]= rate
    if confirm==True:
        df.drop(columns= [i for i in holder], inplace= True)
        return df
    else:
        print(f'Number of columns that have Nan values above the thresh specified{len(holder)}')
        return holder

```

***速记*** :

如果 confirm 设置为 true，将删除缺失值百分比高于阈值集的所有列，如果 confirm 设置为 None 或 False，该函数将返回数据中所有列的缺失值百分比列表。在您的数据上尝试一下。太酷了。

既然我们能够检测到数据中缺失的值，并丢弃不可恢复的值，保留可恢复的值，那么让我们继续讨论如何实际修复它们。

有两种方法可以解决这个问题:

**1。统计插补:**

随着时间的推移，这种方法已被证明是有效的。你所要做的就是用列的均值、中值、众数来填充列中缺失的数据。伙计们，相信我。

Scikit-learn 提供了一个名为[simple imputr](https://web.archive.org/web/20221206100436/https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)的子类，以这种方式处理我们丢失的值。

下面的简短描述将有助于更好地理解统计插补。

```py
from sklearn.impute import SimpleImputer

cols= df.columns

imputer= SimpleImputer(strategy= 'most_frequent')
df= imputer.fit_transform(df)

df= pd.DataFrame(df, columns= cols)

```

**2。链式方程的多重插补(小鼠):**

在这种方法中，一列及其缺失值被建模为数据中其他列的函数。重复该过程，直到先前计算的值和当前值之间的容差非常小并且低于给定的阈值。

Scikit-learn 提供了一个名为[iterative inputr](https://web.archive.org/web/20221206100436/https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html)的子类，以这种方式处理我们丢失的值。

下面是一个简短的描述，希望能让你巩固这个概念。

```py
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

cols= df.columns

imputer= IterativeImputer(max_iter= 3)
df= imputer.fit_transform(df)

df= pd.DataFrame(df, columns= cols)

```

希望到目前为止，我们讨论的内容足以让您有效地处理丢失的值。

想了解更多关于缺失值的信息，请查看 Matt Brems 的这篇精彩文章。

## 编码分类特征

### 什么是分类特征？

好吧，我的朋友们，分类特征是只取离散值的特征。它们不采用连续值，如 3.45、2.67 等。分类特征可以具有大、中、小、1- 5 作为等级、是和否、1 和 0、黄色、红色、蓝色等值。它们基本上代表了类别，例如年龄组、国家、肤色、性别等。

通常，它们以文本格式出现，其他时候，它们以数字格式出现(大多数人经常无法识别这种格式)。

### 如何识别分类特征？

这没什么大不了的，大多数时候，分类特征是以文本格式出现的，这就是你识别它们所需要的。

如果，这是一种排名形式，它们实际上是以数字的形式出现的。那么，要做的是检查一列中唯一值的数量，并将其与该列中的行数进行比较。比方说，一列有 2000 行，只有 5 或 10 个唯一值，您很可能不需要圣人告诉您该列是分类列。这没有规则，这只是直觉，你可能是对的，也可能是错的。所以要自由，尝试新事物。

### 编码类别

给定一个分类特征，正如我们在上面检查过的，我们面临一个问题，将每个特征中的唯一类别转换为数字，同时不丢失其中编码的信息。基于一些可观察的特征，有各种编码分类特征的方法。

还有两类分类特征；

1.  ***有序范畴特征*** :这种特定特征中的范畴之间存在着内在的顺序或关系，如大小(小、大、中)、年龄组等。
2.  ***名义分类特征:*** 特征的类别之间没有合理的顺序，例如国家、城市、名称等。

上面两个类的处理方式不同。我在研究名词性范畴特征时使用的一些方法有:

*   一键编码
*   频率/计数编码
*   目标均值编码
*   有序整数编码
*   二进制编码
*   省略一个编码
*   证据权重编码

对于有序分类特征，我们仅使用:

*   标签编码或顺序编码

现在，我们将尝试一个接一个地学习这些方法。我们也会尽可能用 python 实现它们，并使用一个叫做 category_encoders 的 python 库。

您可以使用 pip install category_encoders 安装库。

#### 一键编码

一键编码是对名词性分类特征进行编码的最有效的方法之一。此方法为列中的每个类别创建一个新的二进制列。理想情况下，我们会删除这些列中的一列，以避免列之间的共线性，因此，具有 K 个唯一类别的要素将在数据中产生额外的 K-1 列。

这种方法的一个缺点是，当一个特征有许多独特的类别或数据中有许多分类特征时，它会扩展特征空间。

上图真正解释了一键编码的概念。以这种方式对特征进行编码消除了所有形式的等级。

下面描述了该方法在 python 中的实现:

```py
import pandas as pd
data= pd.get_dummies(data, columns, drop_first= True)

```

#### 频率/计数编码

这个方法同样非常有效。它根据名词性分类特征在特征(列)中出现的频率为其引入层次结构。这与计数编码非常相似，因为计数编码可以采用任何值，而频率编码被规范化为范围 0 和 1。

下面描述了该方法在 python 中的实现:

```py

def freq_enc(df, cols):
    for col in cols:
        df[col]= df[col].map(round(df[col].value_counts()/len(df),4))
    return df

def count_enc(df, cols):
    for col in cols:
        df[col]= df[col].map(round(df[col].value_counts()))
    return df

```

#### 目标均值编码

这个方法背后的想法其实很高明。这种方法非常独特，因为它在计算中使用了目标列，这在一般的机器学习实践中很少见。分类特征中的每个类别都被替换为该类别的目标列的平均值。

这是一种非常好的方法，但是如果它编码了太多关于目标的信息，可能会导致过度拟合，因此，在使用这种方法之前，我们需要确保这种分类特征与目标列不是高度相关的。

通过映射，用来自训练数据的存储值对测试数据进行编码。

下面描述了该方法在 python 中的实现:

```py

def target_mean_enc(df, cols, target_col):
    mean_holder= {}
    for col in cols:
        col_mean= {}
        cat= list(df[col].unique())
        for i in cat:
            data= df[df[col]== i]
            mean= np.mean(data[target_col])
            col_mean[i]= mean
        mean_holder[col]= col_mean
    return mean_holder

```

上面的函数返回一个字典，其中包含被编码的列的平均值。然后将字典映射到数据上。见下文:

#### 有序整数编码

这种方法与目标均值编码非常相似，只是它更进一步，根据类别的目标均值大小对类别进行排序。

![Ordered integer encoding ](img/7381f6dfaaea5c5c777a54db4d29eeec.png)

实施有序整数编码后，数据帧如下所示:

![Ordered integer encoding ](img/bf69dab020b4dc712449cd0aef77950a.png)

下面描述了该方法在 python 中的实现:

```py
def ordered_interger_encoder(data, cols, target_col):
    mean_holder= {}
    for col in cols:
        labels =  list(enumerate(data.groupby([col])[target_col].mean().sort_values().index))
        col_mean= {value:order for order,value in labels}
        mean_holder[col]= col_mean
    return mean_holder
```

#### 省略一个编码

这种方法也非常类似于目标均值编码，除了在每个级别，它计算目标均值，而忽略特定级别。目标均值编码仍然用于测试数据(更多信息请查看)。

下面描述了该方法在 python 中的实现:

```py
from category_encoders import leave_one_out
binary= leave_one_out(cols= ['STATUS'])
binary.fit(data)
train= binary.transform(train_data)
test= binary.transform(test_data)

```

#### 二进制编码

二进制编码的工作方式很独特，在某种意义上，它也创造了额外的功能，几乎类似于一键编码。首先，它根据独特的特征在数据中出现的方式给它们分配等级(不要烦恼！，这个排名没有任何意义)。然后，等级被转换为以二为基数的数字系统(二进制)。然后，各个数字被拆分到不同的列中。

下面描述了使用 category_encoders 库的方法演示:

```py
from category_encoders import BinaryEncoder
binary= BinaryEncoder(cols= ['STATUS'])
binary.fit(data)
train= binary.transform(train_data)
test= binary.transform(test_data)
```

#### 证据权重编码

这是一种已经在信用风险分析中使用了 70 年的方法。它通常用于转换逻辑回归任务的特征，因为它有助于揭示我们以前可能看不到的特征之间的相关性。此方法只能在分类任务中使用。

它简单地通过将 *ln(p(好)/p(坏))*应用到列中的每个类别来转换分类列。

其中:

*p(好)*是目标列中的一类表示 *p(1)*

而 *p(坏)*是第二个类别，它可能就是 *p(0)。*

下面描述了使用 category_encoders 的这种方法的实现:

```py
from category_encoders import WOEEncoder
binary= WOEEncoder(cols= ['STATUS'])
binary.fit(data)
train= binary.transform(train_data)
test= binary.transform(test_data)

```

#### 标签编码或顺序编码

这种方法用于编码名义分类特征；我们只是简单地根据我们能从中推断出的数量来给每一类分配数字。

下面描述了该方法在 python 中的实现:

```py
df['size']= df['size'].map({'small':1, 'medium':2, 'big':3})
df
```

在这篇文章的过程中，参考了某些机器学习库，如 [scikit learn](https://web.archive.org/web/20221206100436/https://scikit-learn.org/stable/) 和 [category_encoders](https://web.archive.org/web/20221206100436/https://contrib.scikit-learn.org/category_encoders) 。其他一些有助于数据预处理的 python 库包括:Numpy、Pandas、Seaborn、Matplotlib、Imblearn 等等。

然而，如果你是那种不喜欢写太多代码的人，你可以看看诸如 [OpenRefine](https://web.archive.org/web/20221206100436/http://openrefine.org/) 、 [Trifacta 牧马人](https://web.archive.org/web/20221206100436/https://www.trifacta.com/products/wrangler/)和[等工具](https://web.archive.org/web/20221206100436/https://analyticsindiamag.com/10-best-data-cleaning-tools-get-data/)。

## 结论

到目前为止，我们讨论的大多数概念的代码实现都有清晰的例子，可以在这里找到。

我希望这篇文章能够让您深入了解分类编码的概念以及如何填充缺失值。

你应该使用哪一个？

数据清理是一个迭代的过程，因此尽可能多地尝试这些方法，并坚持使用最适合您的数据的方法。**