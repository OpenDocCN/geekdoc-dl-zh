# 如何为时间序列预测任务选择模型[指南]

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/select-model-for-time-series-prediction-task>

使用时间序列数据？这是给你的指南。在本文中，您将学习如何根据预测性能来比较和选择时间序列模型。

在第一部分中，你将被介绍到时间序列的众多模型。这一部分分为三个部分:

*   经典的时间序列模型，
*   监督模型，
*   和基于深度学习的模型。

在第二部分中，您将看到一个用例的应用，在这个用例中，您将为股票市场预测构建几个时间序列模型，并且您将了解一些时间序列建模技术。这些模型将互相比较，选出性能最好的一个。

## 时间序列数据集和预测简介

让我们从回顾时间序列到底是什么开始。**时间序列是一种特殊类型的数据集，其中一个或多个变量随着时间的推移而被测量。**

我们使用的大多数数据集都是基于独立的观察。例如，数据集的每一行(数据点)代表一个单独的观察值。例如，在一个网站上，你可以跟踪每个访问者。每个访问者都有一个用户 id，他或她将独立于其他访问者。

![Time Series Data Examples. A dataset with independent observations](img/5cc4dbd35bf1c90ae18620c4f5633849.png)

*Time series data examples: a dataset with independent observations | Source: Author*

然而，在时间序列中，观测值是随着时间推移而测量的。数据集中的每个数据点对应一个时间点。这意味着数据集的不同数据点之间存在关系。这对于可以应用于时间序列数据集的机器学习算法的类型具有重要的意义。

![Time Series Data Examples. A dataset with dependent observations](img/511aca90a7855f487f6b727719012724.png)

*Time series data examples: a dataset with dependent observations | Source: Author*

在本文的下一部分，您将更详细地了解时间序列数据的特性。

## 时间序列模型细节

由于时间序列数据的性质，时间序列建模有许多与其他数据集无关的特性。

### 单变量与多变量时间序列模型

时间序列的第一个特性是标识数据的时间戳具有内在意义。**单变量时间序列模型**是只使用一个变量(目标变量)及其时间变化来预测未来的预测模型。单变量模型特定于时间序列。

在其他情况下，您可能有关于未来的附加说明性数据。例如，假设您想将天气预报纳入您的产品需求预测，或者您有一些其他数据会影响您的预测。在这种情况下，可以使用**多元时间序列模型**。多元时间序列模型是适用于整合外部变量的单变量时间序列模型。你也可以使用监督机器学习来完成这项任务。

| 单变量时间序列模型 | 多元时间序列模型 |
| --- | --- |
|  |  |
|  |  |
| 

仅仅基于过去和现在的关系

 | 

基于过去与现在的关系，以及变量之间的关系

 |

如果要对时间序列数据使用时态变化，首先需要了解可能出现的不同类型的时态变化。

### 时间序列分解

**时间序列分解**是一种从数据集中提取多种类型变化的技术。时间序列的时态数据有三个重要组成部分:季节性、趋势和噪声。

*   **季节性**是出现在你的时间序列变量中的循环运动。例如，一个地方的温度在夏季会高一些，在冬季会低一些。您可以计算月平均温度，并使用这种季节性作为预测未来值的基础。
*   **趋势**可以是长期向上或向下的形态。在温度时间序列中，趋势可能是由于全球变暖而呈现的。例如，除了夏季/冬季的季节性之外，随着时间的推移，你很可能会看到平均气温略有上升。
*   **噪声**是时间序列中既不能用季节性也不能用趋势来解释的可变性部分。当构建模型时，您最终会将不同的组件组合成一个数学公式。这个公式的两个部分可以是季节性和趋势性。一个结合了两者的模型永远不会完美地代表温度值:误差永远存在。这用噪声系数来表示。

#### Python 中的时间序列分解示例

让我们看一个简短的例子来理解如何使用来自 statsmodels 库的 [CO2 数据集在 Python 中分解时间序列。](https://web.archive.org/web/20221213210225/https://www.statsmodels.org/dev/datasets/generated/co2.html)

您可以按如下方式导入数据:

```py
import statsmodels.datasets.co2 as co2
co2_data = co2.load(as_pandas=True).data
print(co2_data)
```

为了便于理解，数据集如下所示。它有一个时间索引(每周日期)，并记录二氧化碳测量值。

有几个 NA 值可以使用插值法移除，如下所示:

```py
co2_data = co2_data.fillna(co2_data.interpolate())
```

您可以使用以下代码查看 CO2 值的时间演变:

```py
co2_data.plot()
```

这将生成以下图:

![Time series decomposition in Python](img/bd0e187a453de6ea156089aa20ed7a46.png)

*Plot of the CO2 time series | Source: Author*

您可以使用 statsmodels 的季节性分解函数进行现成的分解。以下代码将生成一个图，将时间序列分为趋势、季节性和噪声(此处称为残差):

```py
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(co2_data)
result.plot()
```

![Time series decomposition in Python](img/50e9746fe02d2b74440404a10b1eea22.png)

*Plot of the decomposed CO2 time series | Source: Author*

二氧化碳数据的分解显示出上升趋势和很强的季节性。

### 自相关

让我们转到时间序列数据中可能存在的第二种类型的时间信息:**自相关**。

自相关是时间序列的当前值与过去值之间的相关性。如果是这种情况，您可以使用现值来更好地预测未来值。

自相关可以是正的也可以是负的:

1.  **正自相关**意味着现在的高值可能会在未来产生高值，反之亦然。你可以想想股票市场:如果每个人都在买股票，那么价格就会上涨。当价格上涨时，人们认为这是一个值得购买的好股票，他们也会购买，从而推动价格进一步上涨。然而，如果价格下跌，那么每个人都害怕崩溃，卖掉他们的股票，价格变得更低。
2.  **负自相关**则相反:今天的高值意味着明天的低值，今天的低值意味着明天的高值。一个常见的例子是自然环境中的兔子种群。如果一年的夏天有很多野兔，它们会吃掉所有可用的自然资源。到了冬天，就没什么可吃的了，所以很多兔子都会死掉，存活下来的兔子数量也会很少。在兔子数量很少的这一年里，自然资源会重新增长，并允许兔子数量在下一年增长。

两个著名的图表可以帮助您检测数据集中的自相关:ACF 图和 PACF 图。

#### ACF:自相关函数

自相关函数是一种有助于识别时间序列中是否存在自相关的工具。

您可以使用 Python 计算 ACF 图，如下所示:

```py
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(co2_data)
```

![ACF: the autocorrelation function](img/40f4ef076bdadd229baeafe3bd1c887a.png)

*Autocorrelation plot | Source: Author*

在 x 轴上，您可以看到时间步长(回到过去)。这也被称为滞后数。在 y 轴上，您可以看到每个时间步长与“当前”时间的相关性。很明显，在这张图上有明显的自相关。

#### PACF:自相关函数

PACF 是 ACF 的替代品。它给出的不是自相关，而是偏相关。这种自相关被称为部分自相关，因为随着过去的每一步，只列出额外的自相关。这与 ACF 不同，因为当可变性可以由多个时间点解释时，ACF 包含重复相关性。

例如，如果今天的值与昨天的值相同，但也与前天的值相同，则 ACF 将显示两个高度相关的步骤。PACF 只会在昨天出现，前天就消失了。

您可以使用 Python 计算 PACF 图，如下所示:

```py
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(co2_data)
```

![PACF: the autocorrelation function](img/592661d0d13a90210fef684b26e58473.png)

*Partial autocorrelation plot | Source: Author*

你可以在下面看到，这张 PACF 图更好地展示了二氧化碳数据的自相关性。滞后 1 有很强的正自相关:现在的高值意味着你很有可能在下一步观察到高值。因为这里显示的自相关是部分的，你看不到任何早期滞后的重复效果，使 PACF 图更整洁和清晰。

### 平稳性

时间序列的另一个重要定义是平稳性。平稳时间序列是没有趋势的时间序列。一些时间序列模型不能够处理趋势(稍后将详细介绍)。您可以使用 Dickey-Fuller 测试来检测非平稳性，并使用差分来消除非平稳性。

#### 迪基-富勒试验

Dickey-Fuller 检验是一种统计假设检验，允许您检测非平稳性。您可以使用以下 Python 代码对 CO2 数据进行 Dickey-Fuller 测试:

```py
from statsmodels.tsa.stattools import adfuller
adf, pval, usedlag, nobs, crit_vals, icbest =  adfuller(co2_data.co2.values)
print('ADF test statistic:', adf)
print('ADF p-values:', pval)
print('ADF number of lags used:', usedlag)
print('ADF number of observations:', nobs)
print('ADF critical values:', crit_vals)
print('ADF best information criterion:', icbest)

```

结果如下所示:

ADF 检验的零假设是时间序列中存在单位根。另一个假设是数据是稳定的。

第二个值是 p 值。如果这个 p 值小于 0.05，您可以拒绝零假设(拒绝非平稳性)并接受替代假设(平稳性)。在这种情况下，我们不能拒绝零假设，将不得不假设数据是非平稳的。由于你看到了数据，知道有趋势，所以这也印证了我们得出的结果。

#### 区别

您可以从时间序列中移除趋势。目标是只有季节性变化:这可以是使用某些模型的一种方式，这些模型只适用于季节性，而不适用于趋势。

```py
prev_co2_value = co2_data.co2.shift()
differenced_co2 = co2_data.co2 - prev_co2_value
differenced_co2.plot()
```

不同的 CO2 数据如下所示:

![Differencing](img/d644666d5246c38d537f20cfa82cad95.png)

*Differenced CO2 time series | Source: Author*

如果对差异数据重新进行 ADF 测试，您将确认该数据现在确实是稳定的:

```py
adf, pval, usedlag, nobs, crit_vals, icbest =  adfuller(differenced_co2.dropna())
print('ADF test statistic:', adf)
print('ADF p-values:', pval)
print('ADF number of lags used:', usedlag)
print('ADF number of observations:', nobs)
print('ADF critical values:', crit_vals)
print('ADF best information criterion:', icbest)
```

p 值很小，说明替代假设(平稳性)成立。

### 单步与多步时间序列模型

在开始建模之前，最后一个需要理解的重要概念是单步模型和多步模型的概念。

有些模型非常适合预测时间序列的下一步，但不具备同时预测多个步骤的能力。这些模型是单步模型。您可以通过在您的预测上设置窗口来使用它们创建多步模型，但这有一个风险:当使用预测值来进行预测时，您的误差可能会很快增加并变得非常大。

多步模型具有一次预测多个步骤的内在能力。它们通常是长期预测的更好选择，有时也是一步预测的更好选择。在开始构建模型之前，决定要预测的步数是很关键的。这完全取决于您的用例。

| 一步预测 | 多步预测 |
| --- | --- |
| 

旨在只预测未来 1 步

 | 

旨在预测未来的多步

 |
| 

可以通过加窗预测生成多步预测

 | 无需开窗预测 |
| 

对于多步预测来说性能较差

 | 

更适合多步预测

 |

时间序列模型的类型

## 既然您已经看到了时间序列数据的主要特性，那么是时候研究可用于预测时间序列的模型类型了。这项任务通常被称为预测。

经典时间序列模型

### 经典时间序列模型是传统上在许多预测领域中经常使用的模型族。它们强烈地基于一个时间序列内的时间变化，并且它们与单变量时间序列一起工作得很好。还有一些高级选项可以将外部变量添加到模型中。这些模型一般只适用于时间序列，对其他类型的机器学习没有用。

监督模型

### 监督模型是用于许多机器学习任务的模型族。当机器学习模型使用明确定义的输入变量和一个或多个输出(目标)变量时，它就会受到监督。

监督模型可以用于时间序列，只要你有办法提取季节性并将其放入变量中。示例包括为一年、一个月或一周中的某一天等创建变量。然后，这些被用作监督模型中的 X 变量，而“y”是时间序列的实际值。您还可以将 y 的滞后版本(y 的过去值)包含到 X 数据中，以便添加自相关效应。

深度学习和最新模型

### 过去几年深度学习越来越受欢迎，这也为预测打开了新的大门，因为已经发明了特定的深度学习架构，可以很好地处理序列数据。

云计算和人工智能作为一种服务的普及也在该领域提供了许多新发明。脸书、亚马逊和其他大型科技公司正在开源他们的预测产品，或者在他们的云平台上提供这些产品。这些新的“黑箱”模型的出现为预测从业者提供了尝试和测试的新工具，有时甚至可以击败以前的模型。

深入经典时间序列模型

## 在这一部分，你将深入发现经典的时间序列模型。

ARIMA 家族

### ARIMA 系列模型是一组可以组合的较小模型。ARMIA 模型的每个部分都可以作为独立的组件使用，也可以将不同的构建模块组合起来使用。当所有单独的组件放在一起时，您就获得了 SARIMAX 模型。现在，您将分别看到每个构造块。

1.自回归(AR)

#### 自回归是 SARIMAX 系列的第一个构件。您可以将 AR 模型视为一个回归模型，它使用变量的过去(滞后)值来解释变量的未来值。

AR 模型的阶数表示为 p，它代表模型中包含的滞后值的数量。最简单的模型是 AR(1)模型:它仅使用前一时间步的值来预测当前值。您可以使用的值的最大数量是时间序列的总长度(即，您使用所有以前的时间步长)。

2.移动平均线

#### 均线是更大的 SARIMAX 模型的第二个组成部分。它的工作方式与 AR 模型类似:它使用过去的值来预测变量的当前值。

移动平均模型使用的过去值不是变量的值。更确切地说，移动平均线使用先前时间步骤中的预测误差来预测未来。

这听起来有些反直觉，但背后有一个逻辑。当一个模型有一些未知但有规律的外部扰动时，你的模型可能有季节性或其他模式的误差。MA 模型是一种捕捉这种模式的方法，甚至不需要确定它来自哪里。

MA 模型也可以使用多个时间回溯步骤。这在称为 q 的阶数参数中表示。例如，MA(1)模型的阶数为 1，并且仅使用一个时间步长。

3.自回归移动平均(ARMA)

#### 自回归移动平均(ARMA)模型将之前的两个构建模块合并为一个模型。因此，ARMA 可以使用过去的值和预测误差。

对于 AR 和 MA 过程的滞后，ARMA 可以有不同的值。例如，ARMA(1，0)模型的 AR 阶为 1 ( p = 1)，MA 阶为 0 (q=0)。这其实只是一个 AR(1)模型。MA(1)模型与 ARMA(0，1)模型相同。其他组合也是可能的:例如，ARMA(3，1)具有 3 个滞后值的 AR 顺序，并对 MA 使用 1 个滞后值。

4.自回归综合移动平均(ARIMA)

#### ARMA 模型需要平稳的时间序列。正如你之前看到的，平稳性意味着时间序列保持稳定。您可以使用扩展的 Dickey-Fuller 测试来测试您的时间序列是否稳定，如果不稳定，则应用差分。

ARIMA 模型在 ARMA 模型中加入了自动差分。它有一个额外的参数，您可以将它设置为时间序列需要进行差分的次数。例如，需要进行一次微分的 ARMA(1，1)将产生以下符号:ARIMA(1，1，1)。第一个 1 用于 AR 订单，第二个 1 用于差额，第三个 1 用于 MA 订单。ARIMA(1，0，1)将与 ARMA(1，1)相同。

5.季节性自回归综合移动平均(SARIMA)

#### 萨里玛在 ARIMA 模型中加入了季节效应。如果季节性出现在你的时间序列中，在你的预测中使用它是非常重要的。

萨里玛符号比 ARIMA 符号要复杂得多，因为每个组件都在常规参数的基础上接收一个季节性参数。

例如，让我们考虑之前看到的 ARIMA(p，d，q)。在 SARIMA 符号中，这变成了 SARIMA(p，D，q)(P，D，Q)m。

m 就是每年观察的次数:月数据的 m=12，季度数据的 m=4，等等。小写字母(p、d、q)代表非季节性订单。大写字母(P，D，Q)代表季节性订单。

6.带有外生回归量的季节性自回归综合移动平均(SARIMAX)

#### 最复杂的变体是 SARIMAX 模型。它重新组合 AR、MA、差异和季节效应。除此之外，它还添加了 X:外部变量。如果你有任何可以帮助你的模型改进的变量，你可以用 SARIMAX 添加它们。

Python 中关于 CO2 的自动 Arima 示例

#### 现在，您已经看到了 ARIMA 家族的所有单个构件，是时候将它应用到一个示例中了。让我们看看是否可以使用该模型为二氧化碳数据建立一个预测模型。

ARIMA 或萨里马克斯模型的困难之处在于，你有许多参数(pp，D，Q)甚至(P，D，q)(P，D，Q)需要选择。

在某些情况下，您可以检查自相关图并确定参数的逻辑选择。您可以使用 SARIMAX 的 [statsmodels](https://web.archive.org/web/20221213210225/https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) 实现，并使用您选择的参数尝试性能。

另一种方法是使用 auto-arima 函数，它可以自动优化超参数。金字塔 Python 库正是这样做的:它尝试不同的组合，并选择具有最佳性能的组合。

您可以按如下方式安装金字塔:

安装后，有必要进行训练/测试分割。您将在后面看到更多关于这方面的内容，但现在我们先继续。

```py
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
```

然后，根据 CO2 训练数据拟合模型，并使用最佳选择的模型进行预测。

```py
train, test = train_test_split(co2_data.co2.values, train_size=2200)
```

您可以用这里创建的图向他们展示:

```py
model = pm.auto_arima(train, seasonal=True, m=52)
preds = model.predict(test.shape[0])
```

在该图中，蓝线是实际值(训练数据)，橙线是预测值。

```py
x = np.arange(y.shape[0])
plt.plot(co2_data.co2.values[:2200], train)
plt.plot(co2_data.co2.values[2200:], preds)
plt.show()
```

![Auto Arima in Python on CO2](img/67583b2a05659f84d3198737566f0845.png)

*Actual data in blue and forecasted data in orange* *| Source: Author*

更多关于金字塔的信息和例子，你可以查看他们的文档。

向量自回归(VAR)及其导数 VARMA 和 VARMAX

#### 你可以看到向量自回归，或 VAR 作为 Arima 的多变量替代。不是预测一个因变量，而是同时预测多个时间序列。当不同的时间序列之间有很强的关系时，这尤其有用。向量自回归和标准 AR 模型一样，只包含一个自回归分量。

**VARMA 模型**是 ARMA 模型的多元等价物。VARMA 对于 ARMA 就像 VAR 对于 AR 一样:它在模型中增加了一个移动平均分量。

如果想更进一步，可以用 **VARMAX** 。X 代表**外部(外生)变量**。外生变量是可以帮助您的模型做出更好预测的变量，但它们本身不需要预测。statsmodels [VARMAX](https://web.archive.org/web/20221213210225/https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_varmax.html) 实现是开始实现 VARMAX 模型的好方法。

存在更高级的版本，如季节性 VARMAX (SVARMAX ),但它们变得如此复杂和具体，以至于很难找到简单有效的实现。一旦模型变得如此复杂，就很难理解模型内部发生了什么，通常最好开始研究其他熟悉的模型。

缓和

### 指数平滑是一种基本的统计技术，可以用来平滑时间序列。时间序列模式通常有很多长期可变性，但也有短期(噪声)可变性。平滑允许你使你的曲线更平滑，这样长期的可变性变得更明显，短期的(有噪声的)模式被去除。

这个时间序列的平滑版本可以用于分析。

1.简单移动平均线

#### **简单移动平均线**是最简单的平滑技术。它包括用当前值和几个过去值的平均值替换当前值。要考虑的过去值的确切数量是一个参数。使用的值越多，曲线就越平滑。同时，你会失去越来越多的变异。

2.简单指数平滑(SES)

#### **指数平滑**是这种简单移动平均线的改编。它不是取平均值，而是取过去值的加权平均值。越往后的值越不重要，越近的值越重要。

3.双指数平滑

#### 当时间序列数据中存在趋势时，应避免使用简单的指数平滑法:在这种情况下，该方法效果不佳，因为模型无法正确区分变化和趋势。不过可以用**双指数平滑**。

在 DES 中，有一个指数滤波器的递归应用。这允许您消除趋势问题。对于时间零点，这使用以下公式:

以及后续时间步长的下列公式:

其中α是数据平滑因子，β是趋势平滑因子。

4.霍尔特·温特的指数平滑法

#### 如果想更进一步，可以用三重指数平滑法，也叫**霍尔特温特指数平滑法**。只有当您的时间序列数据中有三个重要信号时，您才应该使用它。例如，一个信号可以是趋势，另一个信号可以是每周季节性，第三个信号可以是每月季节性。

Python 中指数平滑的一个例子

#### 在下面的示例中，您将看到如何对 CO2 数据应用简单的指数平滑。平滑级别表示曲线应该变得多平滑。在本例中，它设置得非常低，表示曲线非常平滑。随意使用这个参数，看看不太平滑的版本是什么样子。

蓝线代表原始数据，橙线代表平滑曲线。由于这是一个简单的指数平滑法，它只能捕捉一个信号:趋势。

```py
from statsmodels.tsa.api import SimpleExpSmoothing
es = SimpleExpSmoothing(co2_data.co2.values)
es.fit(smoothing_level=0.01)
plt.plot(co2_data.co2.values)
plt.plot(es.predict(es.params, start=0, end=None))
plt.show()

```

![Exponential smoothing in Python](img/8754cd1a6793e01b2b2eede6c8ff16a3.png)

*Original data in blue and smoothed graph in orange | Source: Author*

深入监督机器学习模型

## 监督机器学习模型的工作方式与经典机器学习模型非常不同。主要区别在于，他们认为变量要么是因变量，要么是自变量。因变量或目标变量是您想要预测的变量。自变量是帮助你预测的变量。

有监督的机器学习模型不是专门为时间序列数据制作的。毕竟时间序列数据中往往没有自变量。然而，通过将季节性(例如，基于您的时间戳)转换为独立变量，使它们适应时间序列是相当简单的。

线性回归

### 线性回归可以说是最简单的监督机器学习模型。线性回归估计线性关系:每个独立变量都有一个系数，表明该变量如何影响目标变量。

简单线性回归是只有一个自变量的线性回归。非时间序列数据中简单线性回归模型的一个示例如下:取决于外部温度(摄氏度)的热巧克力销售额。

温度越低，热巧克力的销量越高。从视觉上看，这可能如下图所示。

在多元线性回归中，不是只使用一个自变量，而是使用多个自变量。您可以想象二维图形转换成三维图形，其中第三个轴代表可变价格。在这种情况下，您将构建一个线性模型，使用温度和价格来解释销售额。您可以根据需要添加任意数量的变量。

![Linear regression](img/1402bc00c3d9ec1d3ca54b86576e4099.png)

*Linear regression | Source: Author*

现在，当然，这不是一个时间序列数据集:没有时间变量。那么，如何将这种技术用于时间序列呢？答案相当简单。除了在这个数据集中只使用温度和价格，您还可以添加年、月、星期几等变量。

如果你在时间序列上建立一个监督模型，你有一个缺点，你需要做一点特征工程，以某种方式将季节性提取到变量中。然而，一个优点是增加外生变量变得容易多了。

现在让我们看看如何对 CO2 数据集应用线性回归。您可以按如下方式准备 CO2 数据:

这样你就有了三个独立变量:日、月和周。你也可以考虑其他季节性变量，比如星期几、星期几等等。，但是现在，我们先这样吧。

```py
import numpy as np

months = [x.month for x in co2_data.index]
years = [x.year for x in co2_data.index]
day = [x.day for x in co2_data.index]

X = np.array([day, months, years]).T

```

然后，您可以使用 scikit-learn 构建一个线性回归模型，并进行预测，以查看该模型了解了什么:

使用此代码时，您将获得下面的图，该图显示了与数据相对较好的拟合:

```py
from sklearn.linear_model import LinearRegression

my_lr = LinearRegression()
my_lr.fit(X, co2_data.co2.values)

preds = my_lr.predict(X)

plt.plot(co2_data.index, co2_data.co2.values)
plt.plot(co2_data.index, preds)
```

随机森林

![Linear regression](img/48047b27ffa2f19c68a4b8c5b4820064.png)

*Linear regression forecast | Source: Author*

### 线性模型非常有限:它只能拟合线性关系。有时这就足够了，但是在大多数情况下，最好使用更高性能的模型。随机森林是一种常用的模型，允许拟合非线性关系。还是很好用的。

scikit-learn 库有 RandomForestRegressor，您可以简单地使用它来替换前面代码中的 LinearRegression。

现在对训练数据的拟合甚至比以前更好:

```py
from sklearn.ensemble import RandomForestRegressor

my_rf = RandomForestRegressor()
my_rf.fit(X, co2_data.co2.values)

preds = my_rf.predict(X)

plt.plot(co2_data.index, co2_data.co2.values)
plt.plot(co2_data.index, preds)
```

目前来看，理解这个随机森林已经能够更好地学习训练数据就足够了。在本文的后面部分，您将会看到更多的模型评估的定量方法。

![Random forest](img/601832737898b428bb7d3b17fd79af94.png)

*Random forest forecast | Source: Author*

XGBoost

### XGBoost 模型是您绝对应该知道的第三个模型。还有许多其他模型，但随机森林和 XGBoost 被认为是监督机器学习家族中的绝对经典。

XGBoost 是一个基于梯度推进框架的机器学习模型。这个模型是弱学习者的集合模型，就像随机森林一样，但是有一个有趣的优点。在标准梯度提升中，各个树按顺序拟合，并且每个连续的决策树以最小化先前树的误差的方式拟合。XGBoost 获得了相同的结果，但是仍然能够进行并行学习。

您可以使用 XGBoost 包，如下所示:

如你所见，这个模型也非常符合数据。在本文的后面部分，您将学习如何进行模型评估。

```py
import xgboost as xgb

my_xgb = xgb.XGBRegressor()
my_xgb.fit(X, co2_data.co2.values)

preds = my_xgb.predict(X)

plt.plot(co2_data.index, co2_data.co2.values)
plt.plot(co2_data.index, preds)
```

![Xgboost model](img/a6944b0da4563769dff16194824ecc4b.png)

*XGBoost forecast | Source: Author*

深入研究高级和特定的时间序列模型

## 在这一部分中，您将发现两个更高级、更具体的时间序列模型，称为 GARCH 和 TBATS。

广义自回归条件异方差（GeneralizedAutoregressiveConditionalHeteroskedasticity）

### GARCH 代表广义自回归条件异方差。这是一种估计金融市场波动性的方法，通常用于此用例。它很少用于其他用例。

该模型很好地解决了这一问题，因为它假设时间序列的误差方差为 ARMA 模型，而不是实际数据。这样，您可以预测可变性而不是实际值。

GARCH 模型家族存在许多变体，例如，检查[这个](https://web.archive.org/web/20221213210225/https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity)出来。这个模型非常值得了解，但是应该只在需要预测可变性时使用，因此它与本文中介绍的其他模型相对不同。

TBATS

### TBATS 代表以下部件的组合:

三角季节性

*   博克斯-考克斯变换
*   ARMA 误差
*   趋势
*   季节性成分
*   该模型创建于 2011 年，作为预测具有多个季节周期的时间序列的解决方案。由于它相对较新，也相对较先进，所以不像 ARIMA 系列中的车型那样普及和使用。

TBATS 的一个有用的 Python 实现可以在 Python[sk time](https://web.archive.org/web/20221213210225/https://www.sktime.org/en/latest/api_reference/auto_generated/sktime.forecasting.tbats.TBATS.html)包中找到。

深入研究基于深度学习的时间序列模型

## 现在，您已经看到了两个相对不同的模型族，每个模型族都有其特定的模型拟合方式。经典的时间序列模型关注的是过去和现在之间的关系。监督机器学习模型专注于因果关系。

现在，您将看到另外三个可用于预测的最新模型。理解和掌握它们更加复杂，并且可能(也可能不会)产生更好的结果，这取决于数据和用例的细节。

LSTM(长短期记忆)

### LSTMs 是递归神经网络。神经网络是非常复杂的机器学习模型，通过网络传递输入数据。网络中的每个节点学习一个非常简单的操作。神经网络由许多这样的节点组成。该模型可以使用大量简单节点的事实使得整体预测非常复杂。因此，神经网络可以适应非常复杂和非线性的数据集。

RNNs 是一种特殊类型的神经网络，其中网络可以从序列数据中学习。这对于多种用例非常有用，包括理解时间序列(显然是一段时间内的值序列)，也包括文本(句子是单词序列)。

LSTMs 是一种特定类型的 rnn。事实证明，它们在多种情况下对时间序列预测非常有用。它们需要一些数据，学习起来比监督模型更复杂。一旦你掌握了它们，根据你的数据和你的具体使用情况，它们会被证明是非常强大的。

要深入 LSTMs，Python 中的 [Keras](https://web.archive.org/web/20221213210225/https://keras.io/api/layers/recurrent_layers/lstm/) 库是一个很好的起点。

先知

### Prophet 是一个时间序列库，由脸书开源。这是一个黑箱模型，因为它将在没有太多用户说明的情况下生成预测。这可能是一个优势，因为您几乎可以自动生成预测模型，而不需要太多的知识或努力。

另一方面，这里也有一个风险:如果您不够关注，您很可能会产生一个对自动化模型构建工具来说看起来不错的模型，但是实际上效果并不好。

当使用这种黑盒模型时，建议进行广泛的模型验证和评估，但是如果您发现它在您的特定用例中运行良好，您可能会发现这里有很多附加价值。

你可以在脸书的 GitHub 上找到很多资源。

更深

### DeepAR 是亚马逊开发的另一个这样的黑盒模型。内心深处的功能是不一样的，但是从用户体验来说，相对等于 Prophet。这个想法还是要有一个 Python 库来帮你完成所有的繁重工作。

同样，需要谨慎，因为你永远不能指望任何黑盒模型是完全可靠的。在下一部分中，您将看到更多关于模型评估和基准测试的内容，这对于如此复杂的模型来说是极其重要的。一个模型越复杂，错误就越多！

DeepAR 的一个伟大且易于使用的实现可以在 [Gluon](https://web.archive.org/web/20221213210225/https://ts.gluon.ai/api/gluonts/gluonts.model.deepar.html) 包中获得。

时间序列模型选择

## 在本文的前一部分，你已经看到了大量的时间序列模型，分为经典的时间序列模型，有监督的机器学习模型，以及最近的发展包括 LSTMs，Prophet 和 DeepAR。

时间序列预测任务的最终成果将是只选择一个模型。这必须是为您的用例交付最佳结果的模型。在本文的这一部分中，您将学习如何从大量可能的模型中选择一个模型。

时间序列模型评估

### 时间序列度量

#### 选择模型时，首先要定义的是您想要查看的指标。在上一部分中，您已经看到了不同质量的多重拟合(想想线性回归与随机森林)。

为了进一步选择模型，您需要定义一个度量来评估您的模型。预测中经常使用的模型是均方差。该指标测量每个时间点的误差，并取其平方。这些平方误差的平均值称为均方误差。一种常用的替代方法是均方根误差:均方误差的平方根。

另一个常用的度量是平均绝对误差:这里不是取每个误差的平方，而是取绝对值。平均绝对百分比误差是在此基础上的变化，其中每个时间点的绝对误差表示为实际值的百分比。这产生了一个百分比度量，非常容易解释。

时间序列训练测试分割

#### 评估机器学习时要考虑的第二件事是，考虑到在训练数据上工作良好的模型不一定在新的样本外数据上工作良好。这种模型被称为过拟合模型。

有两种常见的方法可以帮助您评估模型是否正确地概括:训练-测试-分割和交叉验证。

训练测试分割意味着在拟合模型之前删除一部分数据。例如，您可以从 CO2 数据库中删除最近 3 年的数据，并使用剩余的 40 年来拟合模型。然后，您预测三年的测试数据，并在您的预测和过去三年的实际值之间测量您选择的评估指标。

要进行基准测试和选择模型，您可以基于 40 年的数据构建多个模型，并对所有模型进行测试集评估。根据这个测试性能，您可以选择性能最好的型号。

当然，如果你正在建立一个短期预测模型，使用三年的数据是没有意义的:你应该选择一个与你在现实中预测的时期相当的评估时期。

时间序列交叉验证

#### 训练测试分割的一个风险是你只能在一个时间点进行测量。在非时间序列数据中，测试集通常由随机选择的数据点生成。然而，在时间序列中，这在许多情况下是行不通的:当使用序列时，你不能去掉序列中的一个点而仍然期望模型工作。

因此，最好通过选择最后一段时间作为测试集来应用时间序列训练测试分割。这里的风险是，如果你的最后一次月经不太可靠，这可能会出错。在最近的 covid 期间，你可以想象许多商业预测已经完全关闭:潜在的趋势已经发生了变化。

交叉验证是一种重复训练测试评估的方法。它不是进行一次列车测试分割，而是进行多次分割(精确的数字是用户定义的参数)。例如，如果您使用三重交叉验证，您将把数据集分成三个相等的部分。然后，您将在三分之二的数据集上拟合三次相同的模型，并使用另外三分之一进行评估。最后，您有三个评估分数(每个在不同的测试集上)，您可以使用平均值作为最终的度量。

通过这样做，您避免了偶然选择一个在测试集上工作的模型:您现在已经确保它在多个测试集上工作。

然而，在时间序列中，您不能应用随机选择来获得多个测试集。如果你这样做，你将会得到有很多缺失数据点的序列。

可以在时间序列交叉验证中找到解决方案。它所做的是创建多个训练测试集，但是每个测试集都是周期的结束。例如，第一次训练测试分割可以建立在前 10 年的数据上(5 次训练，5 次测试)。第二个模型将在前 15 年的数据上完成(10 次训练，5 次测试)，等等。这可以很好地工作，但是缺点是每个模型在训练数据中不使用相同的年数。

一种替代方法是进行滚动分割(总是 5 年训练，5 年测试)，但这里的缺点是你永远不能使用超过 5 年的训练数据。

时间序列模型实验

### 总之，在进行时间序列模型选择时，以下问题是在开始实验之前定义的关键:

您使用的是哪个指标？

*   您要预测哪个期间？
*   如何确保你的模型对模型没有看到的未来数据点起作用？
*   一旦您有了上述问题的答案，您就可以开始尝试不同的模型，并使用定义的评估策略来选择和改进模型。

时间序列建模的用例示例

## 在本部分中，您将制作标准普尔 500 第二天的天气预报。你可以想象每天晚上运行你的模型，然后第二天你就会知道股票市场是上涨还是下跌。如果你有一个非常准确的模型来做这件事，你可以很容易地赚很多钱(不要把它当作财务建议；)).

股票市场预测数据和评估方法的定义

### 获取股票市场数据

#### 您可以使用 Python 中的 Yahoo Finance 包来自动下载股票数据。

你可以在图中看到自 1980 年以来 S&P500 收盘价的演变:

```py
!pip install yfinance

import yfinance as yf

sp500_data = yf.download('^GSPC', start="1980-01-01", end="2021-11-21")
sp500_data = sp500_data[['Close']]
sp500_data.plot(figsize=(12, 12))
```

对于股票数据，绝对价格实际上并不那么重要。股票交易者更感兴趣的是知道价格是上涨还是下跌，以及上涨的百分比。您可以将数据更改为百分比增加或减少，如下所示:

![The evolution of the S&P500 closing prices since 1980](img/ba8eb256c935873c8a4f92f79e38fb48.png)

*The evolution of the S&P500 closing prices since 1980 | Source: Author*

定义实验方法

```py
difs = (sp500_data.shift() - sp500_data) / sp500_data
difs = difs.dropna()
difs.plot(figsize=(12, 12))
```

![Percentage of the price increase/ decrease](img/da620998c0ffeceb647428b5a969c855.png)

*Plot of the percentage difference of the S&P | Source: Author*

#### 模型的目标是对第二天股票价格的变化做出最好的预测。有必要决定一种方法，这样您就可以在这里稍微自动化这个过程。

因为您只想预测一天，所以可以理解测试集将会非常小(只有一天)。因此，最好创建大量的测试分割，以确保有一个可接受的模型评估量。

这可以通过前面解释的时间序列分割来获得。例如，您可以设置一个将生成 100 个训练测试集的时间序列拆分，其中每个训练测试集使用三个月的训练数据和一天的测试数据。这将有助于本例理解时间序列中的模型选择原则。

构建经典时间序列模型

### 让我们从这个问题的经典时间序列模型开始:Arima 模型。在这段代码中，您将设置 Arima 模型的自动创建，其顺序范围从(0，0，0)到(4，4，4)。每个模型都将使用具有 100 个分割的时间序列分割来构建和评估，其中训练规模最大为三个月，测试规模始终为一天。

因为涉及到大量的运行，结果被记录到 [neptune.ai](/web/20221213210225/https://neptune.ai/) 中以便于比较。为了跟进，你可以建立一个免费账户，从这个教程中获得更多信息[。](https://web.archive.org/web/20221213210225/https://docs.neptune.ai/usage/tutorial/)

您可以以表格形式查看结果:

```py
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import neptune.new as neptune

param_list = [(x, y, z) for x in range(5) for y in range(5) for z in range(5)]

for order in param_list:

    run = neptune.init(
        project="YOU/YOUR_PROJECT",
        api_token="YOUR_API_TOKEN",
    )

    run['order'] = order

    mses = []

    tscv = TimeSeriesSplit(n_splits=100,
                           max_train_size = 3*31,
                           test_size=1)
    for train_index, test_index in tscv.split(y):

        try:
          train = y[train_index]
          test = y[test_index]

          mod = sm.tsa.ARIMA(train, order=order)
          res = mod.fit()
          pred = res.forecast(1)[0]

          mse = mean_squared_error(test, pred)
          mses.append(mse)

        except:

          pass

    try:
        average_mse = np.mean(mses)
        std_mse = np.std(mses)
        run['average_mse'] = average_mse
        run['std_mse'] = std_mse

    except:
        run['average_mse'] = None
        run['std_mse'] = None

    run.stop()
```

平均 MSE 最低的模型是阶数为(0，1，3)的模型。但是，你可以看到，这个模型的标准差可疑地为 0。接下来的两个模型是 ARIMA(1，0，3)和 ARIMA(1，0，2)。它们非常相似，这表明结果是可靠的。这里的最佳猜测是将 ARIMA(1，0，3)作为最佳模型，其平均 MSE 为 0.00000131908，平均标准差为 0.00000197007。

如果您使用 Prophet， [Neptune-Prophet 集成](https://web.archive.org/web/20221213210225/https://docs.neptune.ai/integrations/prophet/)可以帮助您跟踪参数、预测数据帧、残差诊断图表和其他模型构建元数据。

构建有监督的机器学习模型

### 现在让我们转到一个监督模型，看看性能是否不同于经典的时间序列模型。

在用于预测的监督机器学习中，需要对特征工程做出决策。正如本文前面所解释的，监督模型使用因变量(预测变量)和自变量(预测变量)。

在一些用例中，你可能有很多关于未来的数据。例如，如果您想预测一家餐馆的顾客数量，您可以使用未来日期的预定数量的外部数据作为独立变量。

对于当前的股票市场用例，您没有这些数据:您只有一段时间内的股票价格。然而，监督模型不能仅使用目标变量来构建。你需要找到一种方法从数据中提取季节性，并使用特征工程来创建独立变量。众所周知，股票市场具有很多自相关效应，因此让我们尝试一个模型，该模型使用过去 30 天的值作为预测变量来预测第 31 天。

您可以创建一个数据集，其中包含 S&P500 的 30 个训练日和 1 个测试日(总是连续的)的所有可能组合，并且您可以通过以下方式创建一个巨大的训练数据库:

既然有了训练数据库，就可以使用常规的交叉验证:毕竟，数据集的行可以独立使用。它们都是 30 个训练日和 1 个“未来”测试日的集合。由于有了这些数据准备，您可以使用常规的 KFold 交叉验证。

```py
import yfinance as yf

sp500_data = yf.download('^GSPC', start="1980-01-01", end="2021-11-21")
sp500_data = sp500_data[['Close']]

difs = (sp500_data.shift() - sp500_data) / sp500_data
difs = difs.dropna()

y = difs.Close.values

X_data = []
y_data = []
for i in range(len(y) - 31):
    X_data.append(y[i:i+30])
    y_data.append(y[i+30])

X_windows = np.vstack(X_data)

```

下表显示了使用此循环获得的一些分数:

```py
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
import neptune.new as neptune
from sklearn.metrics import mean_squared_error

parameters={'max_depth': list(range(2, 20, 4)),
            'gamma': list(range(0, 10, 2)),
            'min_child_weight' : list(range(0, 10, 2)),
            'eta': [0.01,0.05, 0.1, 0.15,0.2,0.3,0.5]
    }

param_list = [(x, y, z, a) for x in parameters['max_depth'] for y in parameters['gamma'] for z in parameters['min_child_weight'] for a in parameters['eta']]

for params in param_list:

    mses = []

    run = neptune.init(
          project="YOU/YOUR_PROJECT",
          api_token="YOUR_API_TOKEN",
      )

    run['params'] = params

    my_kfold = KFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in my_kfold.split(X_windows):

        X_train, X_test = X_windows[train_index], X_windows[test_index]
        y_train, y_test = np.array(y_data)[train_index], np.array(y_data)[test_index]

        xgb_model = xgb.XGBRegressor(max_depth=params[0],gamma=params[1], min_child_weight=params[2], eta=params[3])
        xgb_model.fit(X_train, y_train)
        preds = xgb_model.predict(X_test)

        mses.append(mean_squared_error(y_test, preds))

    average_mse = np.mean(mses)
    std_mse = np.std(mses)
    run['average_mse'] = average_mse
    run['std_mse'] = std_mse

    run.stop()
```

本次网格研究中测试的参数如下表所示:

参数名称

| 测试值 | 描述 |  |
| --- | --- | --- |
|  | 

树越深，越复杂。设置该参数可以帮助您避免模型

过于复杂(过度拟合) | 树越深，就越复杂。设置此参数可以帮助您避免模型过于复杂(过度拟合) |
|  | 

如果树分裂创建了一个总和低于该值的节点，模型将停止分裂。这是避免过于复杂模型的另一种方法

 | 如果树拆分创建的节点的总和低于该值，则模型将停止拆分。这是避免过于复杂的模型的另一种方法 |
|  | 

用于防止过拟合的优化步长

 | 用于防止过度拟合的优化步长 |
|  | 

允许节点进一步分裂的最小损失减少:该值越高，在树中分裂越少

 | 允许进一步分裂节点的最小损失减少:该值越高，在树中进行的分裂越少 |

有关 XGBoost 调优的更多信息，请查看官方的 XGBoost 文档。

这个 XGBoost 获得的最佳(最低)MSE 是 0.000129982。有多个超参数组合获得此分数。正如您所看到的，XGBoost 模型的性能远不如经典的时间序列模型，至少在当前的配置中是这样。为了从 XGBoost 获得更好的结果，可能需要另一种组织数据的方法。

构建基于深度学习的时间序列模型

### 作为模型比较的第三个模型，让我们拿一个 LSTM，看看它是否能打败 ARIMA 模型。您也可以使用交叉验证进行模型比较。然而，这可能要运行相当长的时间。在这种情况下，您将看到如何使用训练/测试分割。

您可以使用以下代码构建 LSTM:

您将看到 10 个时期的以下输出:

```py
import yfinance as yf
sp500_data = yf.download('^GSPC', start="1980-01-01", end="2021-11-21")
sp500_data = sp500_data[['Close']]
difs = (sp500_data.shift() - sp500_data) / sp500_data
difs = difs.dropna()
y = difs.Close.values

X_data = []
y_data = []
for i in range(len(y) - 3*31):
    X_data.append(y[i:i+3*31])
    y_data.append(y[i+3*31])
X_windows = np.vstack(X_data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_windows, np.array(y_data), test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

from sklearn.model_selection import GridSearchCV
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
import neptune.new as neptune
from sklearn.metrics import mean_squared_error
archi_list = [
              [tf.keras.layers.LSTM(32, return_sequences=True,  input_shape=(3*31,1)),
               tf.keras.layers.LSTM(32, return_sequences=True),
               tf.keras.layers.Dense(units=1)
               ],
              [tf.keras.layers.LSTM(64, return_sequences=True,  input_shape=(3*31,1)),
               tf.keras.layers.LSTM(64, return_sequences=True),
               tf.keras.layers.Dense(units=1)
               ],
              [tf.keras.layers.LSTM(128, return_sequences=True,  input_shape=(3*31,1)),
               tf.keras.layers.LSTM(128, return_sequences=True),
               tf.keras.layers.Dense(units=1)
               ],
              [tf.keras.layers.LSTM(32, return_sequences=True,  input_shape=(3*31,1)),
               tf.keras.layers.LSTM(32, return_sequences=True),
               tf.keras.layers.LSTM(32, return_sequences=True),
               tf.keras.layers.Dense(units=1)
               ],
              [tf.keras.layers.LSTM(64, return_sequences=True,  input_shape=(3*31,1)),
               tf.keras.layers.LSTM(64, return_sequences=True),
               tf.keras.layers.LSTM(64, return_sequences=True),
               tf.keras.layers.Dense(units=1)
               ],

]

for archi in archi_list:
    run = neptune.init(
          project="YOU/YOUR_PROJECT",
          api_token="YOUR_API_TOKEN",
      )

    run['params'] = str(len(archi) - 1) + ' times ' + str(archi[0].units)
    run['Tags'] = 'lstm'

    lstm_model = tf.keras.models.Sequential(archi)
    lstm_model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanSquaredError()]
                      )
    history = lstm_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    run['last_mse'] = history.history['val_mean_squared_error'][-1]
    run.stop()

```

LSTM 的表现与 XGBoost 型号相同。同样，如果您想在这方面做更多的工作，还可以有很多东西需要进一步调整。你可以考虑使用更长或更短的训练时间。您可能还希望以不同的方式标准化数据:这通常会影响神经网络的性能。

选择最佳模型

### 作为本案例研究的结论，您可以说最佳性能是由 ARIMA 模型获得的。这是基于对三个月培训期和一天预测期的比较数据。

后续步骤

### 如果你想进一步发展这个模式，有很多地方你可以改进。例如，你可以尝试更长或更短的训练时间。您还可以尝试添加额外的数据，如季节性数据(星期几、月份等。)或其他预测变量，如市场情绪或其他。在这种情况下，您需要切换到 SARIMAX 模型。

我希望这篇文章向您展示了如何在时间序列数据的情况下进行模型选择。现在，您已经对可能感兴趣的不同模型和模型类别有了一个概念。您还看到了专门针对时间序列模型评估的工具，如窗口和时间序列分割。

对于更高级的阅读，我建议以下来源:

For more advanced reading, I suggest the following sources: