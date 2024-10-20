# 如何有效地组织你的 ML 开发

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/how-to-organize-your-ml-development>

每个数据科学家和 ML 实践者最终都会遇到的一个主要问题是工作流管理。测试不同的场景和用例，记录信息和细节，共享和比较特定样本集的结果，可视化数据，跟踪洞察力。这些是数据科学工作流管理的关键组件。它们有助于业务，使您能够扩展任何数据科学项目。

数据科学家很清楚，测试一个版本的 ML 算法是不够的。我们的领域非常依赖经验主义，因此我们需要测试和比较具有不同超参数调整和特性选择的同一算法的多个版本。

所有这些都会产生元数据，需要正确存储。为了做到这一点，我使用了一个可以为我管理所有这些东西的平台——**Neptune**。它附带了一个完整的客户端库，可以无缝集成到您的代码中。它们还可以让您访问基于 web 的用户界面，在这里您的所有数据都被记录下来并可供使用。

为了让您了解 Neptune 所提供的功能，我用一个准备好的在线数据集模拟了一个真实的用例场景。我们将运行不同的分析和 ML 流程，看看海王星在日常工作中能为你提供多少支持。

## 设置 Neptune 环境

为了使您能够快速开始将 Neptune 集成到项目的所有方面，了解如何安装软件包和库，以及如何将 Jupyter 笔记本连接到 Neptune 帐户可能会很有用。

首先，让我们创建一个 conda 虚拟环境，我们将在其中安装所有需要的 neptune 库:

```py
conda create --name neptune python=3.6
```

安装 neptune 客户端库:

```py
pip  install neptune-client
```

安装 Neptune 笔记本，将我们所有的工作保存到 Neptune 的 web 客户端:

```py
pip install -U neptune-notebooks
```

使用以下扩展启用 jupyter 集成:

```py
jupyter nbextension enable --py neptune-notebooks
```

获取您的 API 密钥，并将您的笔记本与您的 Neptune 会话连接起来:

成功连接笔记本后，您需要创建一个个人项目，您的所有实验都将保存在该项目中:

要完成设置，请在笔记本中导入 neptune 客户端库，并调用 neptune.init()方法初始化连接:

```py
import neptune
neptune.init(project_qualified_name='aymane.hachcham/CaseStudyOnlineRetail')
```

你也可以去查一个海王星 AI 前数据科学家 Kamil 做的视频，里面把前面的细节解释的很透彻。

[https://web.archive.org/web/20221206091240if_/https://www.youtube.com/embed/37X5iXiVop8?list=PLKePQLVx9tOd8TEGdG4PAKz0Owqdv1aaw](https://web.archive.org/web/20221206091240if_/https://www.youtube.com/embed/37X5iXiVop8?list=PLKePQLVx9tOd8TEGdG4PAKz0Owqdv1aaw)

视频

***注**:我把代码放在最有启发性的地方，如果你想查看完整的代码版本和笔记本，请随意访问我的 Github repo—[Neptune-Retail](https://web.archive.org/web/20221206091240/https://github.com/aymanehachcham/Neptune-Retail)*

## 探索数据集

我们将看看在线零售数据集，在 [Kaggle](https://web.archive.org/web/20221206091240/https://www.kaggle.com/vijayuv/onlineretail) 公开发布。该数据集记录了世界各地使用在线销售平台的各种客户。每条记录都通知一个购买特定产品的订单。

数据集显示如下:

为了开始加载数据集，我创建了一个小的 python***data manager***类来下载 CSV 文件，提取主要特征并将它们转换成可用的熊猫数据帧:

```py
class DataETLManager:
    def __init__(self, root_dir: str, csv_file: str):
        if os.path.exists(root_dir):
            if csv_file.endswith('.csv'):
                self.csv_file = os.path.join(root_dir, csv_file)
            else:
                logging.error('The file is not in csv format')
                exit(1)
        else:
            logging.error('The root dir path does not exist')
            exit(1)

        self.retail_df = pd.read_csv(self.csv_file, sep=',', encoding='ISO-8859-1')

    def extract_data(self):
        return self.retail_df

    def fetch_columns(self):
        return self.retail_df.columns.tolist()

    def data_description(self):
        return self.retail_df.describe()

    def fetch_categorical(self, categorical=False):
        if categorical:
            categorical_columns = list(set(self.retail_df.columns) - set(self.retail_df._get_numerical_data().columns))
            categorical_df = self.retail_df[categorical_columns]
            return categorical_df
        else:
            non_categorical = list(set(self.retail_df._get_numerical_data().columns))
            return self.retail_df[non_categorical]

    def transform_data(self):
        data = self.retail_df

        data.drop_duplicates(keep='last', inplace=True)

        data['InvoiceNo'].fillna(value=0, inplace=True)
        data['Description'].fillna(value='No Description', inplace=True)
        data['StockCode'].fillna(value='----', inplace=True)
        data['Quantity'].fillna(value=0, inplace=True)
        data['InvoiceDate'].fillna(value='00/00/0000 00:00', inplace=True)
        data['UnitPrice'].fillna(value=0.00, inplace=True)

        data['CustomerID'].fillna(value=0, inplace=True)
        data['Country'].fillna(value='None', inplace=True)

        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

        self.data_transfomed = data
```

我们可以用来开始构建内部核心指标的重要栏目有:

*   ****数量****
*   ****单价****
*   ****CustomerID****
*   ****国家****

 *首先使用 DataETLManager 加载数据集:

```py
etl_manager = DataETLManager(root_dir='./Data', csv_file='OnlineRetail.csv')
etl_manager.extract_data()
etl_manager.transform_data()

dataset = etl_manager.data_transfomed
```

对于零售企业来说，核心价值依赖于平台通过客户订单产生的收入。我们可以将单价和数量结合起来形成月收入，并按发票日期汇总:

```py
dataset['Profit'] = dataset['Quantity'] * dataset['UnitPrice']
revenue = dataset.groupby(['InvoiceDate'])['Profit'].sum().reset_index()
```

我们还可以通过绘制下图来直观显示收入在几个月内的变化情况:

```py
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.offline as pyoff

pyoff.init_notebook_mode()

data = go.Scatter(
    x=revenuePerYear['InvoiceDate'],
    y=revenuePerYear['Profit']
)

layout = go.Layout(
    xaxis={"type": "category"},
    title='Monthly Revenue'
)

fig = go.Figure(data, layout)
pyoff.iplot(fig)
```

由于我们的主要目标是客户，一个值得关注的指标是我们平台保留的活跃客户数量。我们将专门针对英国客户进行实验，因为他们构成了数据样本的大部分。

为了研究主动客户保持，我们需要检查每个月有多少客户订单:

```py
uk_customers = dataset.query("Country=='United Kingdom'").reset_index(drop=True)
activeCustomers = dataset.groupby(['InvoiceDate'])['CustomerID'].nunique().reset_index()
```

分布似乎相当单调，在 2011 年 11 月达到峰值。

在我们的案例研究中，我们希望对这些客户进行适当的细分。通过这种方式，我们可以有效地管理投资组合，并剖析每个团队实际提供的不同价值水平。

我们还应该记住，随着业务规模的增长，不可能对每个客户都有一个直觉。在这个阶段，关于追求哪些客户的人类判断将不起作用，企业将不得不使用数据驱动的方法来建立适当的战略。

在下一部分，我们将更深入地挖掘不同的指标和分析，我们可以利用这些指标和分析对我们的客户群进行适当的细分。

## 提取指标并对数据进行分析

在这一部分，我们将彻底分析数据。我们希望根据财务标准对整个客户群进行细分。在本节结束时，我们应该能够描述和了解我们的客户购买行为。

由于我们已经在 Neptune 中初始化了我们的项目，我们将开始我们的第一个实验，记录我们将在本节中提取的统计数据。您可以将 Neptune 实验想象成一个命名空间，您可以在其中记录度量、预测、可视化以及您可能需要的任何其他内容。

首先初始化这个实验的参数，并调用 create _ experiment()方法。

```py
params = {
    'n_clusters':4,
    'max_iterations': 1000,
    'first_metric': 'Recency',
    'second_metric': 'Frequency',
    'third_metric': 'Monetary Value',
    'users': 'UK'
}

neptune.create_experiment(
    name='KMeans-UK-Users',
    tags=['KMeans-UK'],
    params=params
)
```

一旦你运行笔记本电池，你可以前往网站。如果您打开我们刚刚创建的实验，在**参数**下，您会发现值被正确记录，并准备好跟踪进一步的行动。

为了根据盈利能力和增长潜力对我们的客户群进行细分，我们将关注最终会影响我们客户金融行为的三个主要因素。该标准依赖于构成所谓的 **RFM 分数**的三个因素:

*   使用频率:监控用户活动最近的一个指标
*   使用频率:用户多久在平台上购买一次产品
*   一元价值:从字面上看，它们有多有利可图

首先，我们需要详细阐述数据集的指标。然后，我们将对这些数据点执行聚类，这样我们就可以根据相似性将它们从高价值客户到低价值客户分成不同的类别。从客户细分中获得的见解被用于开发量身定制的营销活动和设计营销策略。

对于这个任务， **K-Means 聚类**算法仍然是一个非常强大的工具。它的易用性和性能为我们的用例提供了完美的平衡。

关于 K-Means 如何工作的详细解释，我推荐这篇完美完成这项工作的文章: [K-Means 聚类——解释](https://web.archive.org/web/20221206091240/https://towardsdatascience.com/k-means-clustering-explained-4528df86a120)

### RFM 分数

这个想法是测量自上次购买以来的天数，从而测量平台上记录的不活动天数。我们可以将其计算为所有客户的最大购买日期减去该范围内的总最大日期。
**创建我们将在**工作的客户数据框架:

```py
customers = pd.DataFrame(dataset['CustomerID'].unique())
customers.columns = ['CustomerID']
```

**合计最大发票日期:**

```py
aggregatR = {'InvoiceDate': 'max'}
customers['LastPurchaseDate']=dataset.groupby(['CustomerID'],as_index=False).agg(aggregatR)['InvoiceDate']
```

**生成最近得分:**

```py
customers['Recency'] = (customers['LastPurchaseDate'].max() - customers['LastPurchaseDate']).dt.days
```

**客户最近:**

因为我们有相应的表，所以将它记录到我们的 Neptune 实验中是一个好主意。

为此，我们可以调用 neptune.log_table()方法，如下所示:

```py
from neptunecontrib.api import log_table
log_table('Recency English Users', recency_UK)
```

现在，您可以继续应用 K-Means 来对我们的最近分布进行聚类。在此之前，我们需要定义最适合我们需求的集群数量。一种方法是肘法。肘方法简单地告诉最佳惯性的最佳簇数。

```py
K-means_metrics = {}

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(customers['Recency'])
    customers["clusters"] = kmeans.labels_
    k-means_metrics[k] = k means.inertia_
```

让我们画出海王星的值，这样我们可以彻底检查曲线如何演变的细节:

```py
for val in kmeans_metrics.values():
    neptune.log_metric('Kmeans_Intertia_Values', val)
```

Neptune 自动记录日志部分的值，并相应地生成一个图表。

根据该图，最佳的最佳聚类数是 4。因此，我们将对三个指标使用 4 个集群。

**K-表示最近:**

```py
kmeans = KMeans(n_clusters=4)
kmeans.fit(customers[['Recency']])
customers['RecencyCluster'] = kmeans.predict(customers[['Recency']])
```

让我们记录最近的分布和预测的集群。

```py
for cluster in customers['RecencyCluster']:
    neptune.log_metric('UK Recency Clusters', cluster)

for rec in customers['Recency']:
    neptune.log_metric('Recency in days', rec)

```

如果您放大最近天数图表，您会注意到数值范围在 50 到 280 天之间。

我们可以通过查看一些常规统计数据来检查有关集群分布的更多信息:

我们可以注意到，聚类 2 中的客户比聚类 1 中的客户更新。

让我们通过分别计算频率和货币价值来推进我们的研究。我们将尝试在这三个指标之间进行更高层次的比较。

**汇总客户订单数:**

```py
customers = pd.DataFrame(dataset['CustomerID'].unique())
customers.columns = ['CustomerID']

aggregatF = {'InvoiceDate': 'count'}
freq = dataset.groupby('CustomerID', as_index=False).agg(aggregatF)
customers = pd.merge(customers, freq, on='CustomerID')
```

**K-频率得分的均值:**

```py
kmeans = KMeans(n_clusters=4)
kmeans.fit(customers[['Frequency']])
customers['FrequencyCluster'] = kmeans.predict(customers[['Frequency']]
```

当与前面的帧结合时，我们得到下表:

**合计每个客户产生的利润总和:**

```py
dataset['Profit'] = dataset['UnitPrice'] * dataset['Quantity']
aggregatMV = {'Profit': 'sum'}
mv = dataset.groupby('CustomerID', as_index=False).agg(aggregatMV)
customers = pd.merge(customers, mv, on='CustomerID')

customers.columns = ['CustomerID', 'lastPurchase', 'Recency', 'Frequency', 'MonetaryValue']

```

然后，我们将所有指标组合在一起，以便有一个总体概述。

**K-表示货币价值:**

```py
kmeans = KMeans(n_clusters=4)
kmeans.fit(customers[['MonetaryValue']])
customers['MonetaryCluster'] = kmeans.predict(customers[['MonetaryValue']])
```

为了得到一个综合考虑了我们刚刚收集的所有值的 RFM 分数，我们需要在一个唯一的总分数中总结不同的聚类。然后，我们根据获得的范围值对每个客户部分进行细分。

三个部分:

*   **高值:0-2 分**
*   **中间值:3-6 分**
*   **高值:6-9 分**

```py
customers['RFMScore'] = customers['RecencyCluster'] + customers['FrequencyCluster'] + customers['MonetaryCluster']
customers['UserSegment'] = 'Low'

customers.loc[customers['RFMScore'] <= 2, 'UserSegment'] = 'Low'
customers.loc[customers['RFMScore'] > 2, 'UserSegment'] = 'Mid'
customers.loc[customers['RFMScore'] > 5, 'UserSegment'] = 'High'
```

最精彩的部分是当我们绘制聚类图并可视化它们是如何分布的，将**频率**和**新近度**度量与它们产生的**货币价值**进行比较。

这两个指标都清楚地表明，最近的和频繁的用户更有利可图。因此，我们应该提高高价值用户(红色)的保留率，并根据该标准做出决策。此外，通过提高用户保留率，我们可以立即影响他们在平台上的频率和新近度。这意味着我们还应该在用户参与度上进行操作。

## 在 Neptune 组织 ML 开发

在这一节中，我们将利用 Neptune 提供的一个优秀特性，即 [ML 集成](https://web.archive.org/web/20221206091240/https://docs.neptune.ai/essentials/integrations)。在我们的例子中，我们将密切关注 **[XGBoost](https://web.archive.org/web/20221206091240/https://docs.neptune.ai/essentials/integrations/machine-learning-frameworks/xgboost)** ，因为海王星有助于所有的技术细节，比如:

*   每次提升迭代后记录指标
*   训练后的模型记录
*   特征重要性
*   最后一次提升迭代后的树可视化

e**X**treme**G**radient**B**oosting 是梯度增强的优化和并行化开源实现，由华盛顿大学的博士生陈天琦创建。XGBoost 使用决策树(像 random forest)来解决分类(二元&多类)、排序和回归问题。我们在监督学习算法领域。

本节的想法是预测客户终身价值，这是评估我们客户组合的另一个重要指标。平台对客户进行投资，进行获取成本、促销、折扣等。我们应该跟踪和密切关注当前的盈利客户，并预测他们在未来会如何发展。

在这个实验中，我们将在 **9** 个月的时间内锁定一组客户。我们将使用 **3** 个月的数据训练一个 XGBoost 模型，并尝试预测接下来的 **6** 个月。

### 分离数据

**3 个月用户:**

```py
from datetime import datetime, date

uk = dataset.query("Country=='United Kingdom'").reset_index(drop=True)
uk['InvoiceDate'] = pd.to_datetime(uk['InvoiceDate'])

users_3m = uk[(uk['InvoiceDate'].dt.date >= date(2010, 12, 1)) & (uk['InvoiceDate'].dt.date < date(2011, 4, 1))].reset_index(drop=True)
```

**6 个月用户:**

```py
users_6m = uk[(uk['InvoiceDate'].dt.date >= date(2011, 4, 1)) & (uk['InvoiceDate'].dt.date < date(2011, 12, 1))].reset_index(drop=True)
```

现在，在 3 个月的数据框架中，应用我们之前所做的相同汇总。关注频率、近期和货币价值。此外，计算与 K-Means 相同的聚类规则。

为了创建终身价值指标，我们将按 6 个月用户组每月产生的收入进行汇总:

```py
users_6m['Profit'] = users_6m['UnitPrice'] * users_6m['Quantity']
aggr = {'Profit': 'sum'}
customers_6 = users_6m.groupby('CustomerID', as_index=False).agg(aggr) customers_6.columns = ['CustomerID', 'LTV']
```

然后根据该度量生成 K 均值聚类:

```py
kmeans = KMeans(n_clusters=3)
kmeans.fit(customers_6[['LTV']])
customers_6['LTVCluster'] = kmeans.predict(customers_6[['LTV']])
```

### 开始培训过程

将 3 个月的表与 6 个月的表合并，您将拥有相同的数据框，以及我们将在后续步骤中使用的训练集和验证集。

```py
classification = pd.merge(customers_3, customers_6, on='CustomerID', how='left')
classification.fillna(0, inplace=True)
```

我们的目标是根据核心预测功能为 LTVCluster 提供分类细分，例如:MVCluster、FrequencyCluster、RFMScore 和 Monetary Value。

然而，我们还不知道它们的相关性和预测能力。就此而言，我们需要运行一些属性相关性分析。

### 属性相关性分析

运行**属性相关性分析**，我们将考虑两个重要的功能:识别对目标变量影响最大的变量，以及理解最重要的预测因子和目标变量之间的关系。为了运行这种分析，可以使用 ***信息值*** 和 ***证据权重*** 的方法。

***注*** *:为了更深入地回顾 WoE 和 IV，我强烈推荐这篇关于流失分析的中型文章:* [*使用信息价值和证据权重的流失分析*](https://web.archive.org/web/20221206091240/https://towardsdatascience.com/churn-analysis-information-value-and-weight-of-evidence-6a35db8b9ec5#9557) *，作者 Klaudia Nazarko* 。

在我们的例子中，我们将继续查看所有特性之间的相关性，并检查 **MVCluster** 和**RFM store**的信息值。

**相关矩阵:**

```py
classification.corr()['LTVCluster'].sort_values(ascending=False)
```

```py
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

corrMatrix = classification.corr()
sn.heatmap(corrMatrix, annot=False)
plt.show()
```

Neptune 与[可视化库](https://web.archive.org/web/20221206091240/https://docs.neptune.ai/essentials/integrations/visualization-libraries)(包括[熊猫](https://web.archive.org/web/20221206091240/https://docs.neptune.ai/essentials/integrations/visualization-libraries)、 [matplotlib](https://web.archive.org/web/20221206091240/https://docs.neptune.ai/essentials/integrations/visualization-libraries/matplotlib) 等等)的集成。

我们确实观察到与 LTVCluster 更相关的特征是频率、货币价值和新近性，这是有意义的。

同样，根据 WoE 和 IV 分析，MVCluster 和 RFMScore 似乎比其他的预测能力更强。

最后，为了进行进一步的训练，我们需要将分类变量转换成数字。一种快速的方法是使用 pd.get_dummies():

```py
classification = pd.get_dummies(customers)
```

*UserSegment* 列已经消失，但我们有新的数字来表示它。我们已经将它转换为 3 个不同的 0 和 1 列，并使其可用于我们的机器学习模型。

### 列车 XGBoost

#### 创造实验

首先在我们已经初始化的前一个项目中创建一个新的实验。在本节中，我们将使用 XGBoost 的多个版本来训练我们的数据。每个版本都将设置特定的超参数。

最后，我们将尝试比较不同的实验，以获得更多的见解。你可以随时查看 Neptune docs，找到任何相关的资源和文档，以备不时之需。

```py
params = {
    'max_depth':5,
    'learning_rate':0.1,
    'objective': 'multi:softprob',
    'n_jobs':-1,
    'num_class':3
}

neptune.create_experiment(
    name='XGBoost-V1',
    tags=['XGBoost', 'Version1'],
    params=params
)
```

根据 hyper-parameters，我们需要一个能够进行多标签分类的 XGBoost 模型(因此我们使用了 **multi:softprob** 目标函数)。我们专门针对客户 LTV 集群范围内的三个类别。

#### 拆分数据

将数据分成训练集和测试集:

```py
X = classification.drop(['LTV', 'LTVCluster', 'lastPurchase'], axis=1)
Y = classification['LTVCluster'] 
```

```py
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.05, random_state=56) 
```

实例化 ***XGB DMatrix*** 数据加载器，以便我们可以方便地将数据传递给模式:

```py
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
```

#### 使用 XGBClassifier Neptune 回调并记录所有指标

是时候让数据符合我们的模型了。我们将使用 XGBClassifier，并在实验仪表板中实时记录所有指标。利用 Neptune 与所有不同种类的梯度增强算法的紧密集成，我们能够非常容易地监控性能和进度。

```py
multi_class_XGB = xgb.XGBClassifier(**params3)
multi_class_XGB.fit(x_train, y_train, eval_set=[(x_test, y_test)], callbacks=[neptune_callback()])

neptune.stop()
```

有一个非常好的视频解释了 Neptune XGBoost 集成是如何工作的:[Integrations–XGBoost](https://web.archive.org/web/20221206091240/https://www.youtube.com/watch?v=xc5gsJvf5Wo&list=PLKePQLVx9tOd8TEGdG4PAKz0Owqdv1aaw&index=13&ab_channel=NeptuneAI)。

[https://web.archive.org/web/20221206091240if_/https://www.youtube.com/embed/xc5gsJvf5Wo?list=PLKePQLVx9tOd8TEGdG4PAKz0Owqdv1aaw](https://web.archive.org/web/20221206091240if_/https://www.youtube.com/embed/xc5gsJvf5Wo?list=PLKePQLVx9tOd8TEGdG4PAKz0Owqdv1aaw)

视频

如果我们回到 Neptune 并单击我们创建的实验，我们可以可视化损失图表、损失度量和特性重要性图表。

如果我们想看看我们的模型在测试集上的得分，我们可以使用 sklearn.metrics 包打印一个分类报告。

```py
from sklearn.metrics import classification_report,confusion_matrix
predict = multi_class_XGB.predict(x_test)
print(classification_report(y_test, predict))
```

尽管我们对之前的结果非常满意，但我们仍然可以创建另一个实验，并调整或更改一些超参数以获得更好的结果。

```py
params2 = {
    'max_depth':5,
    'learning_rate':0.1,
    'objective': 'multi:softprob',
    'n_jobs':-1,
    'num_class':3,
    'eta':0.5,
    'gamma': 0.1,
    'lambda':1,
    'alpha':0.35,
}

neptune.create_experiment(
    name='XGBoost-V2',
    tags=['XGBoost', 'Version2'],
    params=params2,
)
```

让我们来训练:

```py
multi_class_XGB = xgb.XGBClassifier(**params2)
multi_class_XGB.fit(
    x_train,
    y_train,
    eval_set=[(x_test, y_test)],
    callbacks=[neptune_callback()])

neptune.stop()
```

检查训练集和测试集的准确性:

```py
print('Accuracy on Training Set: ', multi_class_XGB.score(x_train, y_train))
print('Accuracy on Testing Set: ', multi_class_XGB.score(x_test[x_train.columns], y_test))
```

总的来说，结果相当不错，几乎与之前的实验相同。

#### 比较两个实验

Neptune 允许我们选择多个实验，并在仪表板上进行比较:

我们可以并排观察这两个实验，并比较每个列实验中的参数实际上如何影响训练集、测试集和验证集的损失。

#### 版本化您的模型

Neptune 的一个有趣特性是能够对模型二进制文件进行版本控制，这样我们就可以在进行实验时跟踪不同的版本。

```py
neptune.log_artifact('xgb_classifier.pkl')
```

但是，在训练过程中调用 neptune_callback()时，最后一次 boosting 迭代会毫不费力地自动记录到 neptune。

## 结论

本教程的主要目标是帮助您快速开始使用 Neptune。工具非常容易，很难在 UI 中迷失。

我希望这篇教程对你有用，因为我设计它是为了涵盖真实数据科学用例的不同方面。我给你留一些参考资料，看看你是否觉得自己的求知欲还需要淬火:

此外，不要忘了查看 Neptune 文档网站和他们的 Youtube 频道，那里有你开始更有效工作所需的所有工具的深度报道:

别忘了查看我的 Github repo 获取本教程的完整代码: [*海王星-零售*](https://web.archive.org/web/20221206091240/https://github.com/aymanehachcham/Neptune-Retail)*