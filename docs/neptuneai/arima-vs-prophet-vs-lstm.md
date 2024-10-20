# ARIMA vs 预言家 vs LSTM 进行时间序列预测

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/arima-vs-prophet-vs-lstm>

假设我们同意对时间和因果关系的线性理解，正如 Sheldon Cooper [博士所说](https://web.archive.org/web/20221117203552/https://bigbangtheory.fandom.com/wiki/The_Financial_Permeability)，那么将历史事件表示为随着时间的推移观察到的一系列值和特征，为从过去学习提供了基础。然而，[时间序列与其他数据集](/web/20221117203552/https://neptune.ai/blog/time-series-prediction-vs-machine-learning)有些不同，包括像文本或 DNA 序列这样的序列数据。

时间组件提供了预测未来时有用的附加信息。因此，有许多专门为处理时间序列而设计的不同技术。这些技术从简单的可视化工具(显示趋势随时间演变或重复)到先进的机器学习模型(利用时间序列的特定结构)不等。

在本帖中，我们将讨论从时间序列数据中学习的三种流行方法:

## 

*   用于时间序列预测的经典 ARIMA 框架
*   2 脸书的内部模型 Prophet，它是专门为从商业时间序列中学习而设计的
*   3LSTM 模型，这是一种强大的递归神经网络方法，已被用于在序列数据的许多问题上取得最著名的结果

然后，我们将展示如何使用 Neptune 及其强大的功能来比较三个模型的结果。

我们先来简单概述一下这三种方法。

## 概述三种方法:ARIMA，先知和 LSTM

### ARIMA

ARIMA 是一类时间序列预测模型，名字是自回归综合移动平均的缩写。ARIMA 的主干是一个数学模型，它使用时间序列的过去值来表示时间序列的值。该模型基于两个主要特征:

1.  过去的值:很明显，过去的行为可以很好地预测未来。唯一的问题是我们应该使用多少过去的值。该模型使用最后 p 个时间序列值作为特征。这里 p 是我们设计模型时需要确定的超参数。
2.  **过去的错误:**模型可以使用关于它在过去表现如何的信息。因此，我们将模型产生的最近的 q 错误添加为特征。同样，q 是一个超参数。

这里的一个重要方面是，时间序列需要标准化，以便模型独立于季节或临时趋势。对此的正式术语是，我们希望模型在一个*平稳的*时间序列上进行训练。从最直观的意义上来说，平稳性意味着生成时间序列的过程的统计特性不会随时间而改变。这并不意味着序列不会随着时间的推移而改变，只是它改变的方式本身不会随着时间的推移而改变。

有几种方法可以使时间序列平稳，最常用的是差分法。通过用 n-1 个差异替换系列中的 n 个值，我们迫使模型学习更高级的模式。当模型预测一个新值时，我们只需将最后一次观察到的值加进去，就可以得到最终的预测值。平稳性如果第一次遇到这个概念会有些混乱，可以参考这个[教程](https://web.archive.org/web/20221117203552/https://machinelearningmastery.com/remove-trends-seasonality-difference-transform-python/)了解更多细节。

#### 因素

形式上，ARIMA 由描述模型三个主要组成部分的三个参数 p、d 和 q 定义。

*   **综合****(ARIMA 的 I):**达到平稳性所需的差数由参数 d 给出，设原特征为 Y [t] 其中 t 为序列中的索引。对于不同的 d 值，我们使用以下变换来创建一个平稳的时间序列。

##### 对于 d=0

在这种情况下，级数已经是静止的，我们无事可做。

##### 对于 d=1

这是最典型的转型。

##### 对于 d=2

请注意，差分可以被视为微分的离散版本。对于 d=1，新特征表示值如何变化。而对于 d=2，新特征表示变化率*，就像微积分中的二阶导数一样。以上也可以推广到 d > 2，但这在实践中很少使用。*

*   **自回归(AR):** 参数 p 告诉我们，对于当前值的表达式，要考虑多少个过去的值。本质上，我们学习一个预测时间 t 的值的模型:

*   **移动平均线(MA):** 要考虑过去的预测误差有多少。新值计算如下:

过去的预测误差:

这三个部分的组合给出了 ARIMA(p，d，q)模型。更准确地说，我们首先对时间序列进行积分，然后我们将 AR 和 MA 模型相加，并学习相应的系数。

### 先知

Prophet FB 是由脸书开发的一种算法，用于内部预测不同商业应用的时间序列值。因此，它是专门为预测商业时间序列而设计的。

这是一个附加模型，由四个部分组成:

让我们讨论一下每个组件的含义:

1.  **g(t):** 它代表*趋势*，目标是捕捉序列的总体趋势。例如，随着越来越多的人加入网络，脸书的广告浏览量可能会随着时间的推移而增加。但是增加的确切作用是什么呢？
2.  **s(t):** 是*季节性*成分。广告浏览的数量也可能取决于季节。例如，在北半球的夏季，人们可能会花更多的时间在户外，花更少的时间在电脑前。对于不同的商业时间序列，这种季节性波动可能非常不同。因此，第二个组成部分是一个模拟季节性趋势的函数。
3.  **h(t):***节假日*成分。我们使用对大多数商业时间序列有明显影响的假期信息。请注意，不同的年份、不同的国家，假期是不同的。因此信息需要明确地提供给模型。
4.  **误差项** ε [t] 代表模型无法解释的随机波动。通常，假设ε [t] 遵循正态分布 *N* (0，σ ² )，均值为零，未知方差σ必须从数据中得出。

### LSTM 递归神经网络

LSTM 代表长期短期记忆。LSTM 细胞用于递归神经网络，该网络学习从可变长度的序列预测未来。请注意，递归神经网络可以处理任何类型的序列数据，与 ARIMA 和预言家不同，它不限于时间序列。

LSTM 细胞背后的主要思想是学习到目前为止看到的序列中的重要部分，忘记不太重要的部分。这通过所谓的门来实现，即具有不同学习目标的函数，例如:

1.  到目前为止看到的时间序列的紧凑表示
2.  如何将新的输入与序列的过去表示相结合
3.  忘记这个系列的什么
4.  作为下一时间步的预测输出什么。

更多细节见图 1 和维基百科文章。

设计基于 LSTM 的最佳模型可能是一项艰巨的任务，需要仔细调整超参数。以下是基于 LSTM 的模型需要考虑的最重要参数列表:

*   用多少个 LSTM 单元来表示这个序列？请注意，每个 LSTM 单元将关注到目前为止处理的时间序列的特定方面。一些 LSTM 细胞不太可能捕获序列的结构，而太多的 LSTM 细胞可能导致过度拟合。
*   典型的是，首先，我们将输入序列转换成另一个序列，即值 *h* [t] 。这产生了一个新的表示，因为*h*t 状态捕获了到目前为止所处理的系列的结构。但是在某些时候，我们不需要所有的 htvalues，而是只需要最后一个 *h [t]* 。这将允许我们将不同的*h*t[t]馈入完全连接的层，因为每个*h*t[t]对应于单个 LSTM 单元的最终输出。设计精确的架构可能需要仔细的微调和多次试验。

![The structure of an LSTM cell](img/c8e6091136eb519c0609a39f5193b3f8.png)

*Figure 1: the structure of an LSTM cell | [Source](https://web.archive.org/web/20221117203552/https://en.wikipedia.org/wiki/Long_short-term_memory)*

最后，我们想重申，递归神经网络是从序列数据中学习的一类通用方法，它们可以处理任意序列，如自然文本或音频。

## 实验评测:ARIMA vs 先知 vs LSTM

### 资料组

我们将使用 Bajaj Finserv Ltd(一家印度金融服务公司)的股票交易数据来比较这三种模型。该数据集的时间跨度从 2008 年到 2021 年底。它包含每日股票价格(平均值、低值和高值)以及交易股票的总量和周转率。数据集的子样本如图 2 所示。

![The data used for evaluation](img/03f80bdacb67f44db01879ad3f4b102c.png)

*Figure 2: the data used for evaluation | Source: Author*

我们对预测每天结束时的成交量加权平均价格(VWAP)变量感兴趣。时间序列 VWAP 值的图表如图 3 所示。

![The daily values of the VWAP variable](img/0e81818d808cc9ce999d871222686ba1.png)

*Figure 3: the daily values of the VWAP variable | Source: Author*

为了进行评估，我们将时间序列分为训练和测试时间序列，其中训练序列由截至 2018 年底的数据组成(见图 4)。

**观察总数:** 3201

**训练观察:** 2624

**测试观察:** 577

![The train and test subsets of the VWAP time series](img/78515399a126e9de67ceac027e5e0b88.png)

*Figure 4: the train and test subsets of the VWAP time series | Source: Author*

### 履行

为了正常工作，机器学习模型需要良好的数据，为此，我们将做一点点**特征工程**。特征工程背后的目标是设计更强大的模型，利用数据中的不同模式。由于这三个模型学习了过去观察到的模式，我们创建了额外的特征来彻底描述股票运动的最近趋势。

特别是，我们跟踪 3、7 和 30 天内不同交易特征的移动平均线。此外，我们还考虑了月、周数和工作日等特性。因此，我们模型的输入是多维的。所用特征工程的一个小例子如下:

```py
lag_features = ["High", "Low", "Volume", "Turnover", "Trades"]
df_rolled_7d = df[lag_features].rolling(window=7, min_periods=0)
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index().astype(np.float32)
```

上面的代码摘录显示了如何添加描述股票销售的几个特征在上周的移动平均值。总的来说，我们创建了一组外生特征:

现在，让我们从主要型号开始:

#### ARIMA

我们从公开可用的包 [pmdarima](https://web.archive.org/web/20221117203552/http://alkaline-ml.com/pmdarima/) 中实现了 ARIMA 版本。函数 [auto_arima](https://web.archive.org/web/20221117203552/http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html#pmdarima.arima.auto_arima) 接受一列*外生*特征作为附加参数，其中我们提供了在特征工程步骤中创建的特征。auto_arima 的主要优点是，它首先执行几个测试，以确定时间序列是否是平稳的。此外，它采用智能电网搜索策略来确定上一节中讨论的 p、d 和 q 的最佳参数。

```py
from pmdarima import auto_arima
model = auto_arima(
	df_train["VWAP"],
	exogenous=df_train[exogenous_features],
	trace=True,
	error_action="ignore",
	suppress_warnings=True)
```

参数 p、d 和 q 的不同值的网格搜索如下所示。最后，返回具有最小 [AIC 值](https://web.archive.org/web/20221117203552/https://en.wikipedia.org/wiki/Akaike_information_criterion)的模型。(AIC 值是同时优化预测模型的准确性和复杂性的模型复杂性的度量。)

然后通过以下方式获得对测试集的预测

```py
forecast = model.predict(n_periods=len(df_valid),  exogenous=df_valid[exogenous_features])
```

#### 先知

我们使用 Prophet 的公开可用的 [Python 实现](https://web.archive.org/web/20221117203552/https://facebook.github.io/prophet/docs/quick_start.html)。输入数据必须包含两个特定字段:

1.  **日期**:应该是可以计算假期的有效日历日期
2.  **Y** :我们要预测的目标变量。

我们将模型实例化为:

```py
from prophet import Prophet
model = Prophet()
```

必须将特征工程期间创建的特征明确添加到模型中，如下所示:

```py
for feature in exogenous_features:
	model.add_regressor(feature)
```

最后，我们拟合模型:

```py
model.fit(df_train[["Date", "VWAP"] + exogenous_features].rename(columns={"Date": "ds", "VWAP": "y"}))
```

并且测试集的预测如下获得:

```py
forecast = model.predict(df_test[["Date", "VWAP"] + exogenous_features].rename(columns={"Date": "ds"}))
```

#### LSTM

我们使用 LSTMs 的 [Keras 实现](https://web.archive.org/web/20221117203552/https://keras.io/api/layers/recurrent_layers/lstm/):

```py
import tensorflow as tf
from keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import Sequential
```

该模型由以下函数定义。

```py
def get_model(params, input_shape):
	model = Sequential()
	model.add(LSTM(units=params["lstm_units"], return_sequences=True, input_shape=(input_shape, 1)))
	model.add(Dropout(rate=params["dropout"]))

	model.add(LSTM(units=params["lstm_units"], return_sequences=True))
	model.add(Dropout(rate=params["dropout"]))

	model.add(LSTM(units=params["lstm_units"], return_sequences=True))
	model.add(Dropout(rate=params["dropout"]))

	model.add(LSTM(units=params["lstm_units"], return_sequences=False))
	model.add(Dropout(rate=params["dropout"]))

	model.add(Dense(1))

	model.compile(loss=params["loss"],
              	optimizer=params["optimizer"],
              	metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

	return model
```

然后，我们用一组给定的参数实例化一个模型。我们使用时间序列中过去的 90 个观测值作为模型的输入序列。其他超参数描述了用于训练模型的架构和特定选择。

```py
params = {
	"loss": "mean_squared_error",
	"optimizer": "adam",
	"dropout": 0.2,
	"lstm_units": 90,
	"epochs": 30,
	"batch_size": 128,
	"es_patience" : 10
}

model = get_model(params=params, input_shape=x_train.shape[1])
```

以上结果产生了下面的 Keras 模型(参见图 5):

![A summary of the Keras LSTM model](img/da15449e689b260c8ac649c4300cd28e.png)

*Figure 5: a summary of the Keras LSTM model | Source: Author*

然后，我们创建一个回调来实现[提前停止](https://web.archive.org/web/20221117203552/https://en.wikipedia.org/wiki/Early_stopping)，即，如果对于给定数量的时期(在我们的示例中为 10 个时期)验证数据集没有产生改进，则停止训练模型:

```py
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                           	mode='min',
patience=params["es_patience"])
```

参数 *es_patience* 是指提前停止的次数。

最后，我们使用预定义的参数拟合模型:

```py
model.fit(
	x_train,
	y_train,
	validation_data=(x_test, y_test),
	epochs=params["epochs"],
	batch_size=params["batch_size"],
	verbose=1,
	callbacks=[neptune_callback, es_callback]
)
```

### 实验跟踪和模型比较

因为在这篇博文中，我们想回答一个简单的问题，即哪个模型对测试数据集产生最准确的预测，我们需要看看这三个模型之间的差异。

有许多不同的方法用于[模型比较](https://web.archive.org/web/20221117203552/https://docs.neptune.ai/you-should-know/comparing-runs)，例如创建记录不同指标评估的表格和图表，创建绘制测试集上预测值与真实值的图表，等等。然而，在这个练习中，我们将使用**海王星。**

* * *

#### 海王星是什么？

Neptune 是 MLOps 的[元数据存储库，为运行大量实验的团队而构建。‌](/web/20221117203552/https://neptune.ai/)

它为您提供了一个记录、存储、显示、组织、比较和查询所有模型构建元数据的单一位置。

‌Neptune 习惯了 for:‌

*   **实验跟踪**:在一个地方记录、显示、组织和比较 ML 实验。
*   **模型注册**:对训练好的模型进行版本化、存储、管理和查询，以及建模元数据。
*   **实时监控 ML 运行**:实时记录和监控模型培训、评估或生产运行

* * *

如本教程中的[所述，我们首先创建一个 Neptune 项目，并记录我们帐户的 API:](https://web.archive.org/web/20221117203552/https://docs.neptune.ai/getting-started/hello-world)

```py
run = neptune.init(project='<YOUR_WORKSPACE/YOUR_PROJECT>',
               	api_token='<YOUR_API_TOKEN>') 
```

变量 *run* 可以看作一个文件夹，我们可以在其中创建包含不同信息的子文件夹。例如，我们可以创建一个名为 model 的子文件夹，并在其中记录模型的名称:

```py
run["model/name"] = "Arima"
```

我们将根据两个不同的指标来比较这些模型的准确性:

1.  均方根误差(RMSE)

2.  平均绝对误差

请注意，可以通过设置相应的值将这些值记录到 Neptune 中，例如，设置:

```py
run["test/mae"] = mae
 run["test/rmse"] = mse
```

三个模型的均方误差和平均误差可以在运行表中并排显示:

[![The mean square error and the mean average error for the three models can be seen next to each other. (The tags for each project are at the top.)](img/31f9589958f846980c86ef2612105f57.png)](https://web.archive.org/web/20221117203552/https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Prophet-vs-ARIMA-vs-LSTM-for-Time-Series-Prediction_16.png?ssl=1)

*Figure 6\. the MSE and the MAE for the three models in the Neptune UI
(the tags for each project are at the top) | [Source](https://web.archive.org/web/20221117203552/https://app.neptune.ai/kutzkov/TimeSeries/experiments?compare=IzBMBpnAWI&split=tbl&dash=leaderboard&viewId=standard-view)*

这三种算法的比较可以在 Neptune 中并排看到，如图 7 所示。

[![Side by side comparison ARIMA Prophet LSTM](img/781d4ce608afc50414d4f3baef98373c.png)](https://web.archive.org/web/20221117203552/https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/ARIMA-vs-Prophet-vs-LSTM.png?ssl=1)

*Figure 7: The mean square error and the mean average error for the three models can be seen next to each other*
*(the tags for each project are at the top) | [Source](https://web.archive.org/web/20221117203552/https://app.neptune.ai/kutzkov/TimeSeries/experiments?compare=Iwg03I&split=cmp&dash=leaderboard&viewId=standard-view)*

我们看到 ARIMA 产生了最好的性能，即它在测试集上实现了最小的均方误差和平均绝对误差。相比之下，LSTM 神经网络在三个模型中表现最差。

根据真实值绘制的准确预测可以在下图中看到。我们观察到，所有三个模型都捕捉到了时间序列的总体趋势，但 LSTM 似乎落后于曲线，即它需要更多来调整自己以适应趋势的变化。和先知似乎失去了对 ARIMA 在最后几个月的考虑测试期间，它低估了真正的价值。

![ARIMA predictions](img/b67066282935f39f4acbecd4498a58d8.png)

*Figure 8: ARIMA predictions | Source: Author*

![Prophet predictions](img/cea112d1df59a6a88f79625470e30b6e.png)

*Figure 9: prophet predictions | Source: Author*

![LSTM prediction](img/e572c552414ef3c5f908fcbf2932dc16.png)

*Figure 10: LSTM prediction | Source: Author*

### 对模型性能的深入了解

#### ARIMA 网格搜索

当在 ARIMA 对 p、d 和 q 的不同值进行网格搜索时，我们可以绘制出各个值的均方误差。图 11 中的彩色点显示了不同 ARIMA 参数在验证集上的均方误差值。

[![Grid-search over the ARIMA parameters](img/c9e97ea832009952ba7e4ebff82fbae5.png)](https://web.archive.org/web/20221117203552/https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Prophet-vs-ARIMA-vs-LSTM-for-Time-Series-Prediction_27.png?ssl=1)

*Figure 11: grid-search over the ARIMA parameters | [Source](https://web.archive.org/web/20221117203552/https://app.neptune.ai/kutzkov/TimeSeries/experiments?compare=OwJgNAjJ1bP3RDlNS9bNA&split=cmp&dash=charts&viewId=standard-view&query=((%60sys%2Ftags%60%3AstringSet%20CONTAINS%20%22grid-search%22)%20OR%20(%60sys%2Ftags%60%3AstringSet%20CONTAINS%20%22arima%22))&sortBy=%5B%22sys%2Fcreation_time%22%5D&sortFieldType=%5B%22datetime%22%5D&sortFieldAggregationMode=%5B%22auto%22%5D&sortDirection=%5B%22descending%22%5D&suggestionsEnabled=true&lbViewUnpacked=true&chartFilter=mse)*

#### 预言家的趋势

我们在 Neptune 中收集参数、预测数据帧、残差诊断图表和其他元数据，同时使用 Prophet 训练模型。这是通过使用一个[单一函数来实现的，该函数捕获先知训练元数据](https://web.archive.org/web/20221117203552/https://docs.neptune.ai/integrations-and-supported-tools/model-training/prophet)并将其自动记录到 Neptune。

在图 12 中，我们展示了先知的不同组成部分的变化。我们观察到，趋势遵循线性增长，而季节性成分表现出波动。

[![The change of values of the different components in the Prophet over time](img/503521bd1cd9cb64d97f2a71c3446d9d.png)](https://web.archive.org/web/20221117203552/https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Prophet-vs-ARIMA-vs-LSTM-for-Time-Series-Prediction_28.png?ssl=1)

*Figure 12: the change of values of the different components in the Prophet over time | Source: Author*

#### 为什么 LSTM 表现最差？

当在几个时期内训练 LSTM 模型时，我们在海王星中收集平均绝对误差。这是通过使用一个 [Neptune 回调函数](https://web.archive.org/web/20221117203552/https://docs.neptune.ai/api-reference/integrations/tensorflow-keras)来实现的，该回调函数捕获训练元数据并将其自动记录到 Neptune。结果如图 13 所示。

观察到，虽然训练数据集上的误差在随后的时段中减小，但是验证集上的误差却不是这样，验证集上的误差在第二个时段中达到其最小值，然后波动。这表明，LSTM 模型对于一个相当小的数据集来说过于先进，并且容易过度拟合。尽管增加了正则项，如辍学，我们仍然无法避免过度拟合。

[![The evolution of train and test error over different epochs of training the LSTM model](img/cd7b3e0c4efb899ab653c64fb6c5a5b3.png)](https://web.archive.org/web/20221117203552/https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Prophet-vs-ARIMA-vs-LSTM-for-Time-Series-Prediction_29.png?ssl=1)

*Figure 13: the evolution of train and test error over different epochs of training the LSTM model | [Source](https://web.archive.org/web/20221117203552/https://app.neptune.ai/kutzkov/TimeSeries/e/TIM-117/charts)*

## 结论

在这篇博文中，我们展示并比较了三种不同的时间序列预测算法。正如所料，没有明确的赢家，每种算法都有自己的优势和局限性。下面我们总结了我们对每个算法的观察:

1.  **ARIMA** 是一个强大的模型，正如我们所见，它为股票数据取得了最好的结果。一个挑战是，它可能需要仔细的超参数调整和对数据的良好理解。
2.  **Prophet** 是专门为商业时间序列预测而设计的。它在股票数据上取得了非常好的结果，但是，从轶事来看，它在其他领域的时间序列数据集上可能会失败。特别是，这适用于时间序列，其中*日历日期*的概念不适用，我们无法学习任何季节模式。Prophet 的优势在于它需要较少的超参数调整，因为它是专门为检测业务时间序列中的模式而设计的。
3.  **基于 LSTM 的递归神经网络**可能是从序列数据中学习的最强大的方法，时间序列只是一个特例。当从大规模数据集学习时，我们可以检测复杂的模式，基于 LSTM 的模型的潜力就完全显现出来了。与 ARIMA 或 Prophet 不同，它们不依赖于关于数据的特定假设，如时间序列平稳性或日期字段的存在。一个缺点是 LSTM 的 rnn 很难解释，并且很难对它们的行为有直觉。此外，为了获得良好的结果，需要仔细调整超参数。

### 未来方向

所以我希望你喜欢阅读这篇文章，现在你一定对我们在这里讨论的时间序列算法有了更好的理解。如果你想深入了解，这里有一些有用资源的链接。快乐实验！

1.  PMD·ARIMA。各个 Python 包的文档。
2.  [先知](https://web.archive.org/web/20221117203552/https://facebook.github.io/prophet/)。脸书先知的文档和教程。
3.  凯拉斯·LSTM。喀拉斯 LSTM RNNs 的文档和示例。
4.  [海王星](https://web.archive.org/web/20221117203552/https://neptune.ai/)。有教程和文档的 Neptune 网站。
5.  一篇关于用海王星跟踪 ML 实验的博客文章。
6.  对 ARIMA 车型的更深入的了解。
7.  关于用 LSTM 神经网络进行时间序列预测的教程。
8.  [原先知研究](https://web.archive.org/web/20221117203552/https://peerj.com/preprints/3190.pdf)论文。