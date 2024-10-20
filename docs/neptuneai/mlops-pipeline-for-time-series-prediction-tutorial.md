# 为时序预测构建 MLOps 管道[教程]

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/mlops-pipeline-for-time-series-prediction-tutorial>

在本教程中，我们将展示一个基于时间序列的 ML 项目的简单示例，并为其构建一个 MLOps 管道。每一步都将按照 MLOps 的最佳实践来执行，整个项目将一步一步地解释。

这个时序项目是基于币安交易应用，但类似的逻辑也适用于其他 ML 项目。

**注意**:这篇文章不是为了财务建议，它只是为了教育目的而写的。此外，本文的主要目的是介绍具有端到端 ML 项目流程的 MLOps 架构，而不是介绍有利可图的交易策略。

## MLOps 101

MLOps 代表机器学习操作，它是管理 ML 项目管道的过程。MLOps 的作用是将 ML 项目的不同部分连接成一个结构，与其所有组件协调工作，并在将来保留该功能。为了实现最大的健壮性，MLOps 应用了 DevOps 的一些实践，尤其是持续集成和持续交付(CI/CD):

1.  **持续集成**确保每当一段代码或数据更新时，整个 ML 管道都能平稳运行。这是通过代码和数据版本化实现的，它允许代码在不同的团队之间共享和运行。重新运行可能包括培训和测试，但是拥有并运行代码测试以确保输入和输出数据遵循特定的格式并且一切按预期运行也是一个好的实践。

2.  **连续交付**允许自动部署新的 ML 模型。通过 CD 流程，可以使用触发器在现有环境中部署新模型，例如新数据集上的重新训练模型、新超参数或新模型架构。

总的来说，对于有编程经验的人来说，理解 MLOps 最简单的方法就是和 DevOps 对比。但在工具和流程上，它们并不完全相同。DevOps 和 MLOps 都试图整合开发、测试和操作原则；然而，DevOps 专注于常规软件开发，而 MLOps 只专注于 ML 项目。

在 ML 项目中，代码不是版本控制管理下的唯一组件。输入数据、超参数、元数据、日志和模型会随着时间而变化，因此需要控制和监控。另一个区别是传统软件不会退化，而 ML 模型会。一旦模型被部署到生产中，它就有可能开始产生错误和不准确的结果。这是因为输入数据随着时间的推移而变化，而模型保持不变，用旧数据进行训练。

因此，除了 CI/CD 之外，MLOps 还包括:

*   **持续培训(CT)**-在生产中自动重新培训 ML 模型的过程。
*   **持续监控(CM)**-持续监控生产中的数据和模型，以便注意潜在的数据漂移或模型过时。

### MLOps 阶段

没有解决 MLOps 问题的独特方法或架构，尤其是现在有数百个与 MLOps 相关的工具和软件包。这是因为 ML 项目的多样性和 MLOps 的概念相当年轻的事实，我们可以提出一些步骤来帮助建立 MLOps 管道，但最有可能的是，它们可能不是详尽的。

设计和范围

[![Some of the MLOps tools](img/364bf62f8fe4c860a25f5b53be3a4389.png)](https://web.archive.org/web/20221206144902/https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/image12.png?ssl=1)

*Some of the MLOps tools |* [*Sour*](https://web.archive.org/web/20221206144902/https://ml-ops.org/content/state-of-mlops)*[ce](https://web.archive.org/web/20221206144902/https://ml-ops.org/content/state-of-mlops)*

#### 在我们开始开发 ML 项目和编写代码之前，我们需要确保业务目标是明确的，并且我们有足够的能力解决问题。接下来是**设计和范围**的阶段。

在这个阶段，我们需要确保理解项目的**问题陈述和业务目标**。此外，我们需要检查**所有资源**的可用性，例如合适的架构、计算资源、有能力的团队等等。

发展

#### 在设计和确定范围之后，是**项目开发**。它包括:

**研究**–收集关于潜在输入特征、数据预处理步骤、ML 模型、新工具等的新信息。

*   **数据工程**——数据摄取、开发 ETL 管道、数据仓库、数据库工程等。
*   **探索性数据分析(EDA)**–使用数据分析和可视化技术了解我们的数据。
*   **实验开发**–可能包括数据预处理、特征工程、ML 模型开发、超参数调整等。
*   **实验跟踪**–最后，我们要比较所有实验并得出结论。
*   操作

#### 在一个模型被开发出来并准备好投入生产后，就进入了**运营**阶段。运营的主要目标是使用一些 **MLOps 实践**将开发的模型投入生产，如测试、版本控制、CI/CD、监控等。

时间序列 101

## 时间序列是按时间顺序排列的数据点序列。它是同一变量在不同时间点的一系列观察结果。时间序列数据通常表示为图表上的一条线，x 轴表示时间，y 轴表示每个数据点的值。此外，每个时间序列都由四个部分组成:

A time series is a sequence of data points that are ordered in time. It is a series of observations of the same variable at different points in time. A time-series data is often presented as a line on a graph with time on the x-axis and the value of each data point on the y-axis. Also, every time series is composed of four components:

## 1 趋势

*   2 季节变化

*   3 周期性变化

*   4 不规则或随机变化

时序项目示例

![Example of a time series ](img/f52617b44b104fc02307569a2f0ef680.png)

*Example of a time* *series |* [*Source*](/web/20221206144902/https://neptune.ai/blog/time-series-tools-packages-libraries)

### 时间序列在许多行业中都有表示，并且有大量时间序列项目的真实例子。这里，我们只提其中的一部分。

网站流量预测

#### 主要思想是预测特定网站的流量，并在此基础上优化资源分配。它可能有助于负载平衡器将网络或应用程序流量分布到多个服务器上。除此之外，为了发现潜在的黑客攻击，有可能为 web 流量中的异常检测开发 ML 解决方案。

许多组织，如脸书、亚马逊、易贝等，都使用类似的应用程序来预测和监控互联网流量。

医疗保健中的时间序列项目

#### 医疗保健中的时间序列数据，如电子健康记录(EHR)和注册表，代表了有关患者的有价值的信息来源，并可在许多方面用于患者福利。使用这些数据，可以开发 ML 模型，从而更深入地了解个体健康和疾病的轨迹，如癌症、阿尔茨海默病、心血管疾病、新冠肺炎和许多其他疾病。

利用时间序列和 ML 模型，有可能发现未来死亡率、复发和并发症的风险，并推荐相同的处方和预防措施。

股票市场预测

股票市场预测是一项非常具有挑战性的任务，其主要目标是创建各种策略来预测未来的股票价格。由于许多因素，如全球经济、财务报告、政治和其他因素，股票市场具有非常不稳定和混乱的行为。

![Time series and ML in medicine](img/9774b044276406fd39b3424f1c218883.png)

*Time series and ML in medicine |* [*Source*](https://web.archive.org/web/20221206144902/https://www.vanderschaar-lab.com/time-series-in-healthcare/)

#### 一般来说，股市预测方法的应用方式有很多种。其中一些是趋势预测，长期价格预测，波动预测，每日，每小时，或高频交易，等等。

其中大多数基于两种方法；基本面分析和技术面分析。基本面分析会考虑一些因素，如财务报告、行业趋势、通货膨胀率、GDP 等。技术分析使用从历史股票数据中计算出来的技术指标，并在此基础上预测股票未来的表现。

比特币交易

比特币是一种数字资产，其价格由供需决定。同样，作为股票市场，它具有非常不稳定和混乱的行为，而比特币价格预测是一个复杂的挑战。众所周知，比特币与一些科技公司的股票相关，因此可以使用许多股市预测技术。

#### 在本文中，我们将使用比特币价格交易作为时间序列项目的示例。这篇文章不是金融建议，它是为教育目的而写的，重点是时间序列和 MLOps 架构。因此，使用 Python 只能开发简单的交易策略。

为了编写一个能够自动下订单的 Python 程序，我们将使用币安交易所和币安 API 及其自己的包装包[Python-币安](https://web.archive.org/web/20221206144902/https://github.com/sammchardy/python-binance)。此外，币安提供了一个模拟账户，允许交易者在模拟环境中用“纸币”进行交易。关于这一点的更多内容将在下面的文章中介绍。

[使用 Python 和币安进行加密比特币交易简介](https://web.archive.org/web/20221206144902/https://medium.com/insiderfinance/introduction-to-crypto-bitcoin-trading-with-python-and-binance-743916258e5f)

时间序列预测的 MLOps 流水线:设计和范围

前面提到和简要描述的所有阶段将在下面使用我们的比特币交易示例一步一步地实际实现。

## 问题陈述

首先，我们需要确定我们清楚这个问题。在这种特殊的情况下，我们不需要与客户或利益相关者等外部因素进行沟通。为了更简单，我们将尝试预测每小时的比特币走势。意味着我们要预测比特币接下来一个小时是涨还是跌，这说明这是一个分类问题。基于这一预测，我们将建立多头或空头头寸(买入或卖出特定数量的比特币)。

### 例如，如果我们预测比特币的价格将在接下来的一个小时内上涨，如果它从 100 美元涨到 105 美元，我们的利润将是 5 美元。否则，如果价格从 100 美元降到 95 美元，我们将损失 5 美元。

粗略地说，预测将使用 XGBoost 模型，整个 ML 项目将遵循最佳 MLOps 实践。

在此阶段，我们可以大致提出 MLOps 结构的蓝图如下:

商业目标

这里的商业目标很明确，那就是盈利。这是一个非常投机的策略；除此之外，我们还需要定义我们可以处理什么样的损失。例如，在任何时候，如果该策略的累计回报率低于-20%，交易将被暂停。

可利用的资源

![Proposed MLOps architecture](img/8cd1926d41e02edcd3bed91b6d82a283.png)

*Proposed MLOps architecture | Source: Author*

### 由于我们的项目很小，而且本质上是实验性的，很可能我们不需要在工具上花任何钱。此外，不会投入实际资金，因为该项目将部署到币安模拟环境。

用于功能工程和培训的所有历史比特币价格数据将从 AWS S3 的币安下载。由于历史数据不是很大，我们也将它下载到本地，模型开发也将在本地完成。

### 最后，我们需要获得所有将在我们的项目中使用的工具，这将在途中完成。

时间序列预测的 MLOps 管道:模型开发

研究

正如我们之前提到的，ML 项目开发的一个好的实践是从研究开始。在我们的例子中，我们可以简单地谷歌一些与我们的项目相关的术语，如“机器学习比特币交易 python”或“比特币技术指标”。此外，youtube 是一个很好的信息来源，尤其是在一些更复杂的概念需要解释的时候。

## 一些高质量的资源可能不容易通过谷歌搜索获得。例如，来自 [Kaggle](https://web.archive.org/web/20221206144902/https://www.kaggle.com/) 的笔记本很少出现在顶级搜索结果中。此外，一些特定领域的高级课程可能会出现在 [Udemy](https://web.archive.org/web/20221206144902/https://www.udemy.com/) 、 [Udacity](https://web.archive.org/web/20221206144902/https://www.udacity.com/) 或 [Coursera](https://web.archive.org/web/20221206144902/https://www.coursera.org/) 上。

### 有一些流行的 ML 网站，像[机器学习大师](https://web.archive.org/web/20221206144902/https://machinelearningmastery.com/blog/)、[中级](https://web.archive.org/web/20221206144902/https://medium.com/)和[走向数据科学](https://web.archive.org/web/20221206144902/https://towardsdatascience.com/)。高质量的 MLOps 内容搜索的一个好地方肯定是 Neptune.ai 博客。Slack 上还有 MLOps 社区，这是一个超级活跃的从业者讨论和分享知识的团体。[你可以在这里加入](https://web.archive.org/web/20221206144902/https://go.mlops.community/slack)(如果你真的加入了，就来#neptune-ai 频道说说 MLOps，这篇文章，或者只是打个招呼！).

最后，对于一些最先进的解决方案，我们可以搜索[谷歌学术](https://web.archive.org/web/20221206144902/https://scholar.google.com/)或[研究之门](https://web.archive.org/web/20221206144902/https://www.researchgate.net/)。

数据工程

在我们的例子中，输入数据的唯一来源是币安。使用 python-finance 包，可以用现有的方法直接获得数据。此外，还有一种获取历史价格并直接保存的替代方法。每月整理的 csv 文件。

为此，我们需要使用币安公共数据库。下载压缩文件。csv 文件，我们使用命令:

### 其中“download-kline.py”是“python”目录中的脚本。关于这个 python 脚本及其参数的更多信息可以在[这里](https://web.archive.org/web/20221206144902/https://github.com/binance/binance-public-data/tree/master/python)找到。

由于我们将使用 AWS 云架构，数据将被摄取到 [S3 存储](https://web.archive.org/web/20221206144902/https://aws.amazon.com/s3/)中。新的 AWS 帐户可以免费获得 12 个月的 5GB S3 存储空间。为了上传数据，我们需要创建一个 bucket，它是存储在 S3 上的对象的容器。如何创建一个桶并在桶中上传一个文件在[这个视频](https://web.archive.org/web/20221206144902/https://www.youtube.com/watch?v=i4YFFWcyeFM)中有解释。

除了使用 AWS web 控制台上传数据之外，还可以从 [AWS EC2](https://web.archive.org/web/20221206144902/https://aws.amazon.com/ec2/) 实例上传数据。基本上，使用 EC2 可以创建一个实例或虚拟机，我们可以从那里下载数据，并使用一个简单的命令直接复制到 S3 存储桶:

```py
`python download-kline.py -s BTCUSDT -i 1h -startDate 2017-08-01 -endDate 2022-06-01`
```

有了一个新的 AWS 帐户，就有可能每月获得 750 小时的小实例，12 个月免费。此外，我们还将在 EC2 实例上部署我们的项目。

对于更复杂的项目，AWS 上有几个其他的数据工程服务，一些最常用的服务在[视频](https://web.archive.org/web/20221206144902/https://www.youtube.com/watch?v=tykcCf-Zz1M)中解释。

ML 模型开发

```py
`aws s3 cp LOCAL_DIRECTORY/ s3://BUCKET_NAME/DIRECTORY/ --recursive`
```

现在，当数据准备好了，我们可以开始探索性数据分析(EDA)。EDA 的主要目标是探索和可视化数据，以发现一些见解，并计划如何开发模型。通常，EDA 是使用 [Jupyter notebook](https://web.archive.org/web/20221206144902/https://jupyter.org/) 和一些用于数据操作和分析的包来完成的，例如[熊猫](https://web.archive.org/web/20221206144902/https://pandas.pydata.org/)、 [numpy](https://web.archive.org/web/20221206144902/https://numpy.org/) 、 [matplotlib](https://web.archive.org/web/20221206144902/https://matplotlib.org/) 等等。

接下来，我们可以从特征工程开始。出于本教程的考虑，我们将使用 9 个技术指标作为特征(5、10 和 20 小时的移动平均线，7、14 和 21 小时的 RSI 和 MFI)。所有技术指标将使用 [TA-Lib](https://web.archive.org/web/20221206144902/http://mrjbq7.github.io/ta-lib/) 进行计算，它们大多与金融时间序列相关，但一般来说，软件包 [tsfresh](https://web.archive.org/web/20221206144902/https://tsfresh.readthedocs.io/en/latest/) 是计算时间序列特征的一个好选择。

对于 ML 模型，我们将尝试使用 [XGBoost](https://web.archive.org/web/20221206144902/https://xgboost.readthedocs.io/en/stable/) 和 [Optuna](https://web.archive.org/web/20221206144902/https://optuna.org/) 来优化超参数。默认情况下，Optuna 使用树形结构的 Parzen 估计器算法，但也有更多的算法可供选择。优化算法将尝试最大化我们策略的累积回报。

![Top AWS data engineering services](img/cbf4f2188535ac3960e07d8dd284a22c.png)

*Top AWS data engineering services |* [*Source*](https://web.archive.org/web/20221206144902/https://www.youtube.com/watch?v=tykcCf-Zz1M)

### 数据集将分为两部分:

**样本内**(从 2018-01-01 到 2021-12-31，用于超参数搜索和回测)

**样本外**(从 2022 年 1 月 1 日到 2022 年 5 月 31 日，用于复查选择策略，以确保我们没有过度拟合模型)

回测将使用固定滑动窗口的时间序列交叉验证来完成，因为我们希望在每次迭代中保持我们的训练集大小相同。

实验跟踪

*   实验跟踪将使用 [neptune.ai 对 Optuna](https://web.archive.org/web/20221206144902/https://docs.neptune.ai/integrations-and-supported-tools/hyperparameter-optimization/optuna) 的集成来完成。这是一个非常方便的集成，让您只需几行代码，通过几个图就可以跟踪来自模型训练的所有元数据。
*   在我们的例子中，我们将使用 Neptune-Optuna 集成来记录和监控 XGBoost 模型的 Optuna 超参数调优。通常，与一些卷积神经网络相比，时间序列模型并不大，并且作为输入，具有几百或几千个数值，因此模型训练得相当快。

对于金融时间序列，有一种方便的方法来跟踪模型超参数尤其重要，因为我们需要运行许多不同的实验。这是因为金融中的时间序列往往具有非常混乱的运动，需要大量的调整。

通过使用合适的超参数跟踪工具，我们将能够通过观察**优化历史图**来识别优化是如何进行的。除了**运行时间和硬件消耗日志**之外，我们将能够断定我们是否需要增加优化试验(迭代)。

![Time series cross validation with sliding window](img/f72aab00bfd2151df65c95da2bd389a4.png)

*Time series cross validation with sliding window |* [*Source*](https://web.archive.org/web/20221206144902/https://www.kaggle.com/code/cworsnup/backtesting-cross-validation-for-timeseries/notebook)

### Neptune-Optuna 集成提供了对**超参数重要性**、**平行坐标图**的可视化，显示了不同超参数值和目标函数值之间的关系，以及许多更有用的功能。

为此，首先，我们将在主类中定义“objective”方法:

其中“trial”是 Optuna 中使用的参数，“apply_strategy”准备数据并训练模型，而“get_score”在我们的情况下返回累积回报，作为我们想要优化的度量。之后，我们需要连接 Optuna 和 neptune.ai

更多信息，请遵循这个[海王星-Optuna 整合指南](https://web.archive.org/web/20221206144902/https://docs.neptune.ai/integrations-and-supported-tools/hyperparameter-optimization/optuna)。

用于时间序列预测的 MLOps 流水线:自动化测试

在模型开发期间，我们将使用 [GitHub](https://web.archive.org/web/20221206144902/https://github.com/) 作为源代码管理工具。此外，基于一个项目及其需求，实现一些自动化测试是一个很好的实践。

```py
def objective(self, trial):

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 350, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.10),
        'subsample': trial.suggest_uniform('subsample', 0.50, 0.90),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.50, 0.90),
        'gamma': trial.suggest_int('gamma', 0, 20),
    }
    look_back = trial.suggest_int('look_back', 30, 180)
    self.apply_strategy(params)

    return self.get_score()

```

作为自动化测试的一个例子，我们将创建一个简单的冒烟测试，使用 [GitHub actions](https://web.archive.org/web/20221206144902/https://github.com/features/actions) 运行整个项目。这是一个简单的自动化测试，检查我们的主要 python 脚本是否可以成功运行。

```py
import optuna
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils

run = neptune.init(
	project="enes.zvornicanin/optuna-test",
	api_token="YOUR_API_TOKEN",
)

neptune_callback = optuna_utils.NeptuneCallback(run)

hb = HourlyBacktester(dt.data, trader_config)
n_trials = 20

study = optuna.create_study(direction="maximize")
study.optimize(hb.objective, n_trials=n_trials, callbacks=[neptune_callback])

```

为了测试这个项目，在我们的“main.py”脚本中，除了 main 函数之外，我们还将创建一个名称相同但前缀为“test_”的函数:

这样，如果我们以“pytest main.py”的形式运行“main.py”和 [Pytest](https://web.archive.org/web/20221206144902/https://docs.pytest.org/) ，Pytest 将自动运行所有带有“test_”前缀的函数。在“test_main”函数中，我们提供了一个“debug”参数，该参数控制我们是使用真实数据还是仅用于测试的虚拟数据。此外，当“debug=True”时，程序不登录币安帐户，而是创建一个模拟币安客户对象的一个方法的[模拟对象](https://web.archive.org/web/20221206144902/https://realpython.com/python-mock-library/):

## 其中 client.create_order 将在代码的后面使用。

接下来，我们将设置 GitHub，使它在每次推送到存储库后自动在 Ubuntu 虚拟机上运行我们的测试。第一步是在目录“. github/workflows”中创建一个. yml 文件，或者直接在 github 存储库中创建，方法是:

操作->新建工作流->自行设置工作流:

```py
def main():
	pt = PaperTrader(config)
	pt.execute_trade()

def test_main():
	pt = PaperTrader(config, debug=True)
	pt.test_execute_trade()

if __name__ == '__main__':
	main()

```

```py
def log_in(self):

    	if self.debug:
        	self.client = Mock()
        	self.client.create_order = lambda symbol, side, type, quantity:{
            	"executedQty":1,
            	"cummulativeQuoteQty":np.random.rand()+1
        	}
        	return

    	self.client = Client(api_key = os.environ.get('BINANCE_TESTNET_API'),
                         	api_secret = os.environ.get('BINANCE_TESTNET_SECRET'),
                         	tld = 'com',
                         	testnet = True)
```

之后，将出现一个新的工作流模板，其中包含一些基本命令:

更新的。为我们的用例创建的 yml 文件在[存储库](https://web.archive.org/web/20221206144902/https://github.com/eneszv/binance-trading-neptune/blob/production/.github/workflows/main.yml)中可用，它看起来像这样:

![Setting up workflow](img/617a2b545bcbde9784e35e1d35e3c09a.png)

*Setting up workflow | Source: Author*

这将使名为“Tests”的 GitHub 操作使用“master”分支在 push 上运行下面的命令。该工作流将在 Ubuntu 机器上运行，前两个步骤在大多数项目中都很常见，与检出存储库和 python 的设置相关。之后，它安装需求和需要特殊安装的包 talib(用于技术指标)。最后，它运行我们的测试。

如果操作成功完成所有步骤，操作选项卡下将出现绿色标志:

![New workflow template](img/f5d912f004c4ff71aab89397ba14d6e0.png)

*New workflow template | Source: Author*

用于时间序列预测的 MLOps 管道:部署和连续交付

将使用 [GitHub actions](https://web.archive.org/web/20221206144902/https://github.com/features/actions) 、 [Docker](https://web.archive.org/web/20221206144902/https://www.docker.com/) 和 [AWS ECS](https://web.archive.org/web/20221206144902/https://aws.amazon.com/ecs/) 按照 CI/CD 实践进行部署。弹性容器服务(ECS)是一种容器编排服务，可以轻松部署、管理和扩展容器化的应用程序。在我们的例子中，我们将把它用作 CI/CD 服务，它将托管 docker 将在其中运行的 [EC2](https://web.archive.org/web/20221206144902/https://aws.amazon.com/ec2/) 实例。

![The updated .yml file ](img/0868d4e6e3c1900517ea8074fb889c18.png)

*The updated .yml file | Source: Author*

将使用 [AWS 身份和访问管理(IAM)](https://web.archive.org/web/20221206144902/https://aws.amazon.com/iam/) 服务提供对 AWS 服务的访问。此外，在部署之前，我们将在 [AWS S3](https://web.archive.org/web/20221206144902/https://aws.amazon.com/s3/) 上接收输入数据。

简而言之，部署步骤如下:

当代码被推送到一个定义的分支时，GitHub 动作激活并启动 CD 进程。首先，操作配置 AWS 凭证，以便访问服务。

![Completed action](img/f9399f51c72bd982cb4fb74ac28c659d.png)

*Completed action | Source: Author*

## 动作建立 Docker 图像并将其推送到 [AWS ECR](https://web.archive.org/web/20221206144902/https://aws.amazon.com/ecr/) 上。Elastic Container Registry 是一个容器图像的存储库，其中的图像可以很容易地通过其他 AWS 服务进行推送、访问和分发。

操作在 ECS 服务上部署定义的映像。

ECS 服务将操作一个 EC2 实例，其中将部署一个 Docker 容器。在 Docker 容器中，cron 作业将使用 Miniconda 环境，每小时运行一次我们的主 python 脚本。

输出数据将存储在 [AWS S3](https://web.archive.org/web/20221206144902/https://aws.amazon.com/s3/) 上。此外，一些输出结果和元数据将作为监控服务存储在 [neptune.ai](https://web.archive.org/web/20221206144902/https://neptune.ai/) 上。

*   我和 S3
*   创建 IAM 用户的步骤
*   1.AWS -> IAM ->用户->添加用户

*   2.填写用户名，并在 AWS 访问类型下选择“访问密钥-编程访问”,然后单击权限选项卡旁边的。

3.选择以下策略:

![Production set-up](img/91862c7b13fb4897d8583b9eb3a28d94.png)

*Production set-up | Source: Author*

### AmazonS3FullAccess

#### 亚马逊 C2FullAccess

amazone C2 containerregistryfull access

AmazonECS_FullAccess

![Adding a user](img/dd729ca0e5b81d50e05c9ad1f59b6b6c.png)

*Adding a user | Source: Author*

EC 2 instanceprofileforimagebuilderecrcontainerbuilds

*   4.单击“下一步”按钮两次，然后单击“创建用户”。现在，用户凭证将会出现，**请确保下载并保存它们**，因为您将无法再次看到它们。
*   创建 S3 存储桶的步骤
*   1.转到 AWS ->亚马逊 S3 ->桶->创建桶
*   2.写入存储桶名称，并可选地启用存储桶版本控制。单击 create bucket，它应该已经创建好了。
*   3.现在点击你的桶名，试着从你的电脑上传一个文件。

ECR 和 ECS

![Polices to select](img/3a1b3fabc858ffbc7780eaddb29d2901.png)

*Selecting policies | Source: Author*

创建 ECR 存储库的步骤

1.转到 AWS ECR ->入门(创建存储库)

![Download and save user credentials](img/f53646ffe778ab71b07bd9156838e88f.png)

*Download and save user credentials | Source: Author*

#### 2.在可见性设置中，选择私有。

3.定义存储库名称。

4.单击创建存储库。

对于 ECS 服务，我们需要创建:

**任务定义**(需要在 Amazon ECS 中运行 Docker 容器)

![Creating the S3 bucket](img/2352f7d4ceef6d507ba63a250e57eaf0.png)

*Creating S3 bucket | Source: Author*

### **集群**(定义 Docker 将运行的基础设施)

#### 创建 ECS 任务定义的步骤

1.转到 AWS ECS ->任务定义->创建新任务定义

2.为启动类型兼容性选择 EC2。

3.指定任务定义的名称。

4.在容器定义下，单击添加容器按钮。

5.对于容器名称，添加先前定义的 ECR repo 名称。

![Creating ECR repository](img/2023f52430a53f35b92706c6a6a58046.png)

*Creating ECR repository | Source: Author*

6.为映像添加 ECR repo URI 链接(可以在 ECR 服务中找到，在存储库名称旁边)。内存限制和端口映射如下图所示。单击添加，然后单击创建按钮。

*   创建 ECS 群集的步骤

#### 1.转到 AWS ECS ->集群->创建集群

2.选择 EC2 Linux +网络，然后单击下一步。

3.定义一个集群名称，选择 EC2 实例类型(t2.micro 符合免费层的条件)并创建一个密钥对(或选择现有的密钥对)。点击创建按钮。

4.创建群集后，单击群集名称，然后在“Services”选项卡下，单击“Create”按钮。

5.选择 EC2 作为启动类型，定义服务名称和任务数(1)。

6.几次单击下一步，然后单击“创建服务”按钮完成。

![Creating ECS task definition](img/f3427e95ab3b2b20e337cf49ecf1a155.png)

*Creating ECS task definition | Source: Author*

#### Dockerfile

[Dockerfile](https://web.archive.org/web/20221206144902/https://github.com/eneszv/binance-trading-neptune/blob/production/Dockerfile) 可以在[这个资源库](https://web.archive.org/web/20221206144902/https://github.com/eneszv/binance-trading-neptune/tree/production)中找到，我们在这里简单解释一下。

对于 Docker 容器，我们将使用 Ubuntu 22.10 映像。我们为 miniconda 设置了环境变量 PATH、python 脚本将使用的环境变量，并为 Ubuntu 安装了一些包。之后，我们下载、安装并创建一个名为“env”的 miniconda python 环境。

命令收到。币安 _ 贸易/`复制我们项目的所有文件到 Docker“币安 _ 贸易”目录。下一组命令与[安装 Ta-Lib Python 包](https://web.archive.org/web/20221206144902/https://blog.quantinsti.com/install-ta-lib-python/)有关，我们用它来表示技术指标。之后，我们安装 requirements.txt 文件中的所有其他包。

![Creating ECS cluster](img/e3939633e5a5ce0566d2e14fded12211.png)

*Creating ECS cluster | Source: Author*

接下来的几个命令与在 Docker 中配置 cron 作业的[相关。为此，我们需要在存储库中有一个“cron-job”文件，我们将把它复制到 Docker 容器中。容器启动时将被激活的最后一个命令将环境变量重定向到/etc/environment 文件中，以便对 cron 作业可见。此外，它还会激活 cron 作业。](https://web.archive.org/web/20221206144902/https://stackoverflow.com/questions/37458287/how-to-run-a-cron-job-inside-a-docker-container)

Cron 作业文件是在我们的存储库中定义的，它应该每小时运行一次我们的主脚本。

![Creating ECS cluster](img/a88ae5effb24883e2f4a8ae90ff3a335.png)

*Creating ECS cluster | Source: Author*

GitHub 操作

接下来，我们将解释如何将我们的项目作为持续部署(CD)工作流的一部分部署到 ECS。为此，我们将使用 GitHub 动作。定义这个过程的 Yaml 文件可以在资源库中找到，在这里我们将一步一步地解释它。

### [我们将遵循 GitHub 文档中可用的框架](https://web.archive.org/web/20221206144902/https://docs.github.com/en/actions/deployment/deploying-to-your-cloud-provider/deploying-to-amazon-elastic-container-service)。该操作将在推送到主分支时被激活。

1.首先，我们定义一些稍后将会用到的环境变量。其中一些如存储库、服务、集群和容器名称是在前面的步骤中定义的。AWS 区域与 AWS 配置文件相关。

2.ECS 任务定义可以存储为。json 文件，位于 ECS ->任务定义->您的定义名称-> JSON 选项卡下

![Dockerfile](img/f2884f39bf2745f529db67d3dcbb29ed.png)

*Dockerfile | Source: Author*

*   之后，在 yaml 文件中，我们定义了将在最新的 Ubuntu 操作系统上运行的步骤，第一步是检查存储库。

*   4.接下来，我们需要配置 AWS 凭证。我们在 IAM 用户下创建了它们。此外，它们将被存储为动作秘密，我们可以在 GitHub 存储库中定义它们，位于:设置选项卡->秘密->动作->新存储库秘密

*   所有机密都可以使用$ { { secrets }访问。变量名称}}符号。

*   5.登录 Amazon ECR 后，我们构建 Docker 映像并将其推送到 ECR 存储库。这里我们从秘密中定义局部环境变量。其中一些在构建命令期间作为构建参数被处理到 Docker 容器中。这样，我们的 python 脚本就能够使用方法访问它们。

### 6.最后，我们在任务定义中填充一个新的映像 ID，并将其部署在 ECS 上。

现在，如果我们在我们的存储库中推送更改，GitHub actions 将启动这个过程。下面是一个成功运行的例子:

用于时间序列预测的 MLOps 管道:监控

我们可以使用 neptune.ai 轻松获得相同的功能，而不是建立一个 Flask 或 Fast API web 服务器来显示我们的结果、元数据和图表。这个想法是在每次脚本运行后，即每次交易后，我们在 neptune.ai 项目上上传带有图表的结果。通过几行代码和两分钟的工作，我们将能够存储和跟踪我们的 Matplotlib 图表和。neptune.ai 项目中的 csv 结果文件。

![Defining environment variables](img/6e2259d96f8b7e3fc945364c26785b12.png)

*Defining environment variables | Source: Author*

要开始监控性能指标，请运行以下代码:

![ECS task definition](img/008a7ad43ce81af4c07a8892a8a6a9c0.png)

*ECS task definition | Source: Author*

3.  定义 matplotlib 图:

![Checking repository](img/ded63b44417415d18a3802e14aebaffb.png)

*Checking repository | Source: Author*

并将其作为静态或交互式图像上传到项目中:

为了上传包含结果的 pandas 数据框，请再添加一行代码:

在这种情况下，这将自动记录元数据和结果的过程。同时，您还可以根据定义的指标，在不同的实验运行之间进行比较，如下一个屏幕截图所示。

```py
import os
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')

```

![Building and pushing the Docker image to the ECR repository](img/ab28098327c476605ec79c9aa38addcf.png)

*Building and pushing the Docker image to the ECR repository | Source: Author*

除了简单地跟踪我们的结果以便向我们的同事和客户展示之外，为 ML 项目建立监控系统还有更多好处:

![Successful run](img/e02a6c8e4998065e316db021460c85a7.png)

*Successful run | Source: Author*

## 例如，**模型陈旧**是金融时间序列中的常见问题。虽然在这个项目中，模型再训练的逻辑是在代码中实现的，并且在模型开始表现不佳时会被触发，但是直接监控模型的陈旧性可能是有用的。

类似地，**数据漂移**可以是监控的对象，可以创建仪表板，显示每个模型预测的实时生产数据的统计数据。

除了 Neptune 之外，还有一些更专用于 ML 监控的工具。例如， [Arize AI](https://web.archive.org/web/20221206144902/https://arize.com/) 通过帮助理解机器学习模型在现实世界中部署时的行为方式，使 ML 从业者能够更好地检测和诊断模型问题。

```py
import neptune.new as neptune
run = neptune.init(
        	project=os.environ.get('NEPTUNE_PROJECT'),
        	api_token=os.environ.get('NEPTUNE_API_TOKEN'),
    	)
```

类似地， [WhyLabs](https://web.archive.org/web/20221206144902/https://whylabs.ai/) 是一个模型监控工具，帮助 ML 团队监控数据管道和 ML 应用程序。更多关于 ML 监控工具的信息可以在本文中找到:[做 ML 模型监控的最佳工具](/web/20221206144902/https://neptune.ai/blog/ml-model-monitoring-best-tools)。

```py
fig = plt.figure(figsize =(4, 4))
```

结论

```py
run["static-img"].upload(neptune.types.File.as_image(fig))
run["interactive-img"].upload(neptune.types.File.as_html(fig))

```

在本教程中，我们介绍了一个简单的端到端 ML 时间序列项目，遵循 MLOps 实践。

```py
run['data/results'].upload(neptune.types.File.as_html(df_results))
```

我们的项目位于 [github 库](https://web.archive.org/web/20221206144902/https://github.com/eneszv/binance-trading-neptune)上，定义了在代码推送时触发的冒烟测试。这是 **CI 实践**的一个简单例子。除此之外， **CD 功能**是通过 GitHub actions、Docker 和 AWS 服务实现的。 **CM 示例**是使用 neptune.ai 简单提供的，而 **CT 逻辑**集成在 python 代码本身中，如果最后 N 次运行的结果低于 T 阈值，模型将被重新训练和优化。

关于这个项目的任何其他问题，请随时联系我。

参考

*   For instance, **model staleness** is a common issue in financial time series. Although in this project, the logic for model retraining is implemented in the code and will be triggered if the model starts to perform badly, it might be useful to monitor model staleness directly.

*   Similarly, **data drift** can be a subject for monitoring, where it’s possible to create dashboards that will show some statistics about real-time production data at every model prediction.

Besides Neptune, there are a few more tools that are more specialized for ML monitoring. For example, [Arize AI](https://web.archive.org/web/20221206144902/https://arize.com/) enables ML practitioners to better detect and diagnose model issues by helping understand why a machine learning model behaves the way it does when deployed in the real world.

Similarly, [WhyLabs](https://web.archive.org/web/20221206144902/https://whylabs.ai/) is a model monitoring tool that helps ML teams with monitoring data pipelines and ML applications. More about ML monitoring tools can be found in this article: [Best Tools to Do ML Model Monitoring](/web/20221206144902/https://neptune.ai/blog/ml-model-monitoring-best-tools).

## Conclusion

In this tutorial, we’ve presented a simple end-to-end ML time series project following MLOps practices. 

Our project is located on the [github repository](https://web.archive.org/web/20221206144902/https://github.com/eneszv/binance-trading-neptune) with defined smoke tests which are triggered on code push. That is a simple example of **CI practices**. In addition to that, **CD functionality** is implemented with GitHub actions, Docker, and AWS service. **CM example** is simply provided using neptune.ai and **CT logic** is integrated in the python code itself, where the model will be retrained and optimized if the results of the last N runs are below the T threshold.

For any additional questions regarding this project, feel free to reach out to me on Linkedin.

### References