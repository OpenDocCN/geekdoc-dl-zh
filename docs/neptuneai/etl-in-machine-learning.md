# 机器学习教程中的深度 ETL——Neptune 案例研究

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/etl-in-machine-learning>

大多数时候，作为数据科学家，我们认为我们的核心价值是我们找出解决任务的机器学习算法的能力。事实上，模型训练只是大量工作的最后一部分，主要是数据，这是开始构建模型所需要的。

在 ML 解决方案或高度复杂的神经网络出现之前，一个完整的数据基础设施应该已经就位、经过测试并准备就绪。不幸的是，我们常常认为整个数据管道上游部分所需的大量工作是理所当然的。

数据应该是我们的主要关注点，同时找到将其转化为可操作见解的方法。如今，对于任何寻求在快速发展的世界中领先的公司来说，数据都被视为高价值资源。

在本文中，我将介绍一组构建数据基础设施的概念和过程，以及成功的数据管道的不同构件。我将重点介绍:

*   **解释通用数据管道**
*   **涉及 ML 实践的 ETL(提取、转换、加载)过程**
*   **使用 Apache Airflow 实现数据工作流程自动化**
*   **在 Neptune 培训和管理不同的 ML 模型解决方案**

为了让这篇文章更加实用和有指导意义，我将使用一个**信用评分**数据集来运行所有的实验。我坚信边做边学，所以让我们马上开始吧！

***注*** : *完整的代码可以在我的 [Github repo](https://web.archive.org/web/20221117203556/https://github.com/aymanehachcham/ETL-Processing-ML) 中找到。不要犹豫，克隆它，自己尝试。*

## 数据管道和工作流

通常在商业中，我们经常为了各种各样的目的处理大量的数据。构建健壮的管道可能会变得相当复杂，尤其是在数据稀缺、转换过程涉及大量技术细节的时候。

高效的数据传输依赖于三个重要的模块:

1.  **数据生产者**

数据源指向原始数据准备好被获取的地方。

2.  **转换和运输工作流程**

ETL 子流程，涉及一堆*提取、转换和数据加载*层，将数据路由到相应的端点。

3.  **数据消费者**

利用干净和预处理信息执行高端 taks 的最终端点。

管道是非常通用的，根据商业计划有不同的用途。它们通常共享一般的概念，但是具体的实现会有所不同。

在我们的例子中，数据集已经准备好了。我们需要的是设计一个 ETL 过程，根据我们假装要做的事情来转换我们的数据。

## 探索信用评分数据集

在这一部分，我们将彻底分析信用评分数据集。数据集将帮助我们实现和测试我们的 ETL 工作流。

银行使用信用评分模型为客户分配信用评分。这些分数代表了客户的总体信用度。信用评分起源于 20 世纪 50 年代初的美国，供债权人评估有信用历史的客户的财务实力。今天，这项技术已经成为一个真正的金融机构。这一评级适用于所有拥有社会保障号码的美国人。

我们将使用的数据集“给我一些信任”，来自 [Kaggle](https://web.archive.org/web/20221117203556/https://www.kaggle.com/c/GiveMeSomeCredit) 。我将简要概述数据结构，解释每个特性的本质，并展示一组通用统计数据。

### 数据特征

*   **无担保额度的循环使用**:信用卡和个人信用额度的总余额，除了房地产，没有分期付款债务，如汽车贷款，除以信用额度的总和
*   **年龄**:借款人的年龄，以年为单位
*   **逾期 30-59 天未恶化的次数**:在过去 2 年中，借款人逾期 30-59 天未恶化的次数
*   **债务比率**:每月的债务支付、赡养费、生活费用除以每月总收入
*   **月收入**:人均月收入
*   **未结清的信用额度和贷款数量**:未结清的贷款(汽车贷款或抵押贷款等分期付款)和信用额度(如信用卡)数量
*   **逾期 90 天的次数**:借款人逾期 90 天或以上的次数
*   **房地产贷款数量或额度**:包括房屋净值信用额度在内的抵押和房地产贷款数量
*   **逾期 60-89 天的次数没有恶化**:在过去 2 年中，借款人逾期 60-89 天的次数没有恶化
*   **被赡养人数**:家庭中除自己以外的被赡养人数(配偶、子女等)。)

### 目标

*   **两年内严重拖欠**:逾期拖欠 90 天或更长时间的人

该目标表明在两年的时间窗内衡量的债务人的拖欠情况。值为 1 表示借款人拖欠贷款，并且在过去 2 年中一直拖欠贷款。值为 0 表示借款人是一个好客户，并且在最近两年中按时偿还了债务。

通常，大多数金融行业数据包含缺失值，或者对于特定特征没有意义的值。这个特殊的数据集也不例外，在债务和信贷余额比率方面存在不一致，价值远远超出了应该承认的范围。

因此，我们将应用数据转换来消除所有会改变建模和训练阶段结果的差异。

**现在，让我们专注于描述性分析**:

不出所料，目标特性高度不平衡，会导致模型训练出现严重问题。将 **86.3** %的债务人归类为不良付款人将导致模型过度适合这一特定类别，而完全忽略其他类别。

通过查看**年龄**分布，我们清楚地观察到，40 岁到 50 岁之间的年龄组包含大多数样本，其他剩余的组或多或少是平衡的。

![Age distribution](img/fa233fe81a1f82f29cd0f960d1246ec7.png)

*Age distribution regarding Frequency and Cumulative Frequency*

年龄分析表明，人口中负债最多的部分是 35 至 70 岁的人。最重要的是，月收入最低的人群的债务总是更高。

![Debt ratio](img/a21578e2ba274c3cfcd1cc4460c03d6a.png)

*Debt ratio distribution regarding Age and Monthly Income*

积累金融资源最多的人口段在 45 岁到 60 岁之间，这是相当符合逻辑和似是而非的。

![Monthly Income distribution by Age](img/6e86c2d2421a77f705e7431d96adc0a4.png)

*Monthly Income distribution by Age*

循环债务的趋势与长期债务非常相似，人口非常年轻，负债总是越来越多。数据中最明显的相关性是短期和长期债务、年龄组和工资之间的相关性。

![](img/ef3801fc29ac07e27585c9fdf51afe1f.png)

*Revolving credits ratio by Age and Monthly Income*

现在我们对数据集有了更多的了解和理解，我们可以继续进行数据提取和转换的 ETL 过程。

## 构建并自动化您的 ETL 过程

在这一节中，我们将实现一个 ETL 工作流，该工作流将数据从 csv 文件提取到一个可用的 pandas 数据框架中，我们将应用所有需要的转换来为 ML 模型准备一个干净的数据集。为了提高效率，我们将尝试模拟一个自动化循环来运行该流程。

本节将介绍一些概念，如用于过程自动化的气流定向非循环图，以及实现数据转换的因素分析程序。

### 数据析取

我们希望从 *csv* 文件中提取数据，并将其用于我们的实验目的。为此，首先我们创建一个小的 *Python 数据管理器*类，它将负责解析 csv、提取和格式化任何相关数据以供我们分析。

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

        self.credit_scoring_df = pd.read_csv(self.csv_file, sep=',', encoding='ISO-8859-1')
    def extract_data(self):
        return self.credit_scoring_df

    def fetch_columns(self):
        return self.credit_scoring_df.columns.tolist()

    def data_description(self):
        return self.credit_scoring_df.describe()

    def fetch_categorical(self, categorical=False):
        if categorical:
            categorical_columns = list(set(self.credit_scoring_df.columns) - set(self.credit_scoring_df._get_numerical_data().columns))
            categorical_df = self.credit_scoring_df[categorical_columns]
            return categorical_df
        else:
            non_categorical = list(set(self.credit_scoring_df._get_numerical_data().columns))
            return self.credit_scoring_df[non_categorical]
```

需要仔细研究的三种重要方法:

*   extract_data():返回我们刚刚创建的信用评分数据框架
*   fetch_columns():返回数据中的所有列
*   data_description():如果我们想快速浏览一下数据的结构，这很有用
*   fetch _ categorical():返回数据框中的分类值

我们将把列名改得更短:

```py
credit_df=credit_df.drop('Unnamed: 0', axis=1)
credit_df.columns = ['Target', 'Revolving', 'Age', '30-59PastDue', 'DbtRatio', 'Income', 'NumOpenLines', 'Num90DayLate', 'NumRealEstLines', '60-89PastDueNoW', 'FamMemb']
```

结果看起来像这样:

![Credit Scoring datatset](img/d07c193025eb5d627298b6e06c26bc85.png)

*Credit Scoring datatset*

### 数据转换

我们将关注两个转型阶段:

*   预处理转换
*   分析转换

这个想法是，我们绝对需要预处理传入的原始数据，消除重复，丢弃空值和缺失值。此外，进行单变量分析时，我们很快会发现许多样本的比率变量超出了范围。通常，我们需要检测和删除异常值。

有了可用的数据，我们将开始实施因子分析，以提取最能解释方差和相关性的深刻特征。

#### 预处理转换

**删除重复值和缺失值:**

```py
def transform_data(self):

        self.credit_scoring_df.drop_duplicates(keep='last', inplace=True)

        self.credit_scoring_df.dropna(how='all', inplace=True)
```

**去除异常值:**

*   年龄特征是从 0 到最大值 100 的连续变量。有某些记录，价值为零，没有意义，不能作为借款人的资格，该人必须是 18 岁的成年人。
*   检查值大于 1 的债务和周转比率的上限和顶部编码方法，这意味着所有高于上限的值都将被删除。
*   从模型中排除具有显著(超过 50%)缺失值的特征或记录，尤其是当缺失的程度对于数据不平衡率(相当高)来说足够重要时。

```py
clean_credit = self.credit_scoring_df.loc[self.credit_scoring_df['Revolving'] <= 1]
clean_credit = clean_credit.loc[clean_credit['DbtRatio'] <= 1]
clean_credit = clean_credit.loc[clean_credit['Age'] <= 100]
clean_credit = clean_credit.loc[clean_credit['Age'] >= 18]
clean_credit = clean_credit.loc[clean_credit['FamMemb'] < 20]

```

#### 分析转换

清理完数据后，我们将用**均值=0** 和**标准差=1** 对数据进行标准化，除了作为目标的二元因变量。

```py
def normalize(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))
    dataNorm["Target"]=dataset["Target"]
    return dataNorm

clean_scaled_df = normalize(clean_credit)
```

数据值范围从 0 到 1:

![Normalized data values](img/b95bc2c46b048e73bb42a4341cbf905b.png)

*Normalized data values*

### 从因子分析开始

在分离训练阶段的数据之前，我们希望检查所有变量之间的相关性，以了解变量将如何组合。

因子分析是一种线性统计模型。它用来解释观察变量之间的方差，将一组观察变量浓缩成*未观察变量，*称为因子。观察变量被建模为因子和误差项的线性组合。

首先，让我们运行一个充分性测试来检查数据集是否适合于因子分析。我们将进行[巴莱特球形度测试](https://web.archive.org/web/20221117203556/https://www.statology.org/bartletts-test-of-sphericity/#:~:text=Bartlett's%20Test%20of%20Sphericity%20compares,are%20orthogonal%2C%20i.e.%20not%20correlated.)。

```py
from scipy.stats import chi2, pearsonr
import numpy as np

def barlett_test(frame: pd.DataFrame):

    frame.info()

    col, row = frame.shape
    x_corr = frame.corr()

    corr_det = np.linalg.det(x_corr)
    chi_measure = -np.log(corr_det) * (col - 1 - (2 * row + 5) / 6)
    degrees_of_freedom = row * (row - 1) / 2
    p_value = chi2.sf(statistic, degrees_of_freedom)
    return chi_measure, p_value
```

```py
chi_square, p_value = barlett_test(clean_scaled_df)
(1003666.113272278, 0.0)
```

**高度适合** : *观察到的相关性不同于单位矩阵，H0 和 H1 假设得到验证。*

我们将使用 factor_analyzer python 包，通过以下命令安装它:

```py
pip install factor_analyzer
```

将数据拟合到 FactorAnalyzer 类，我们将运行 Kaiser criterion 内部统计以得出数据中的特征值。

```py
from factor_analyzer import FactorAnalyzer

fa = FactorAnalyzer()
fa.fit(clean_scaled_df)

ev, v = fa.get_eigenvalues()
eigen_values = pd.DataFrame(ev)
eigen_values
```

**原始特征值:**

有四个特征值大于 1 的主分量:

![Eigen Values](img/871b8715a587707b2883fe503a1c2a50.png)

*Eigen Values after Running Factor Analysis*

绘制碎石图我们可以很容易地想象出我们需要的四个相关因素:

```py
import matplotlib.pyplot as plt

plt.scatter(range(1,clean_scaled_df.shape[1]+1),ev)
plt.plot(range(1,clean_scaled_df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
```

![Scree plot for the 4 factors](img/ae4dd76a079c54bf8bcfffe8502c6cb6.png)

*Scree plot for the 4 factors*

检查如何将在 [matplotlib](https://web.archive.org/web/20221117203556/https://docs.neptune.ai/essentials/integrations/visualization-libraries/matplotlib) 中生成的图表记录到 Neptune。

让我们对这 4 个因素进行因素分析轮换，以获得更好的解释。旋转可以是正交的或倾斜的。它有助于在观察到的变量之间重新分配[公度](https://web.archive.org/web/20221117203556/https://www.datacamp.com/community/tutorials/introduction-factor-analysis)，具有清晰的载荷模式。

```py
fac_rotation = FactorAnalyzer(n_factors=4, rotation='varimax')
fac_rotation.fit(clean_scaled_df)

fac_rotation.get_factor_variance()
```

| 特征 | 因素 1 | 因素 2 | 因素 3 | 因素 4 |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

我们得出这四个分量解释的方差为 **51.4%** ，是令人满意的。基于此，我们决定用四个因素进行进一步分析。

显示 4 个因素及其各自的可观察特征，我们看到:

![Factors and Observable Features ](img/f3b8c16fc4b9974b459fb50444817554.png)

*Factors and Observable Features side by side*

**我的观察**

*   变量*30-59 过期*、*60-89 过期现在*和*num 90 延迟*在**工厂 1** 上加载高。这意味着这一因素导致借款人越来越拖欠。因此我们可以把这个因素命名为 ***财务斗争*** 。

**二观察**

*   变量 *DbtRatio* 、 *NumOpenLines* 和 *NumRealEstLines* 在 **Factor2** 上加载高。这意味着这一因素导致借款人承担更多的债务和更多的信贷额度。我们可以将其命名为 ***财务要求*** 。

**三观察**

*   变量*年龄*、*家庭成员*和*旋转*在**因子 3** 上负载高，年龄与因子 3 成间接比例。这个因素被命名为 ***消耗性收入*** ，因为随着年龄的增长，消耗性收入也会增加。

**四观察**

*   变量*收入*和*家庭成员*加载到**工厂 4** 上。所以我们可以很容易地将其命名为 ***行为生活方式*** 因为随着收入的增加，我们的生活方式和我们家庭成员的生活方式也会增加。

现在我们有了 ML 分析的四个因素:

*   财务斗争
*   财务要求
*   消耗性收入
*   行为生活方式

### 使用 Dag 自动化 ETL 流程

一旦提取和转换管道形成并准备好进行部署，我们就可以开始考虑将以前的数据存储在数据库中的方法。

为了简化和便于说明，我们将使用 Sql_Alchemy python 包将之前转换的数据框加载到本地 MySQL 数据库中。

```py
pip install SQLAlchemy
```

```py
from sqlalchemy.engine import create_engine

def load_data(self):
        database_config = {
            'username': 'your_username',
            'password': 'your_pass',
            'host': '127.0.0.1',
            'port':'3306',
            'database':'db_name'
        }

        engine = create_engine('mysql+mysqlconnector://{}:{}@{}:{}/{}'.format(
            database_config['username'],
            database_config['password'],
            database_config['host'],
            database_config['port'],
            database_config['database']
        ))

        data_to_load = type(pd.DataFrame())(self.credit_scoring_df)
        try:
            data_to_load.to_sql('Credit Scoring', con=engine, if_exists='append', index=False)
        except Exception as err:
            print(err)
```

#### 气流有向图:有向无环图

Airflow 计划自动化的数据工作流，包括共享特定依赖关系的多个任务或流程。气流文件对 Dag 的定义如下:

在 Airflow 中，DAG(或有向无环图)是您想要运行的所有任务的集合，以反映它们的关系和依赖性的方式组织。在 Python 脚本中定义了一个 DAG，它将 DAG 结构(任务及其依赖项)表示为代码

要安装并开始使用 Airflow，可以查看网站，网站对每一步都解释的很透彻: [AirFlow Doc](https://web.archive.org/web/20221117203556/https://airflow.apache.org/docs/apache-airflow/stable/installation.html) 。

在我们的例子中，我们将编写一个小的 DAG 文件来模拟 ETL 的自动化。我们将安排 DAG 从 2021 年 3 月 25 日开始每天运行。DAG 将有三个 python 运算符，分别代表提取、转换和加载功能。

```py
from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

```

```py
from etl_process import DataETLManager, DATA_PATH

default_dag_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 3, 9),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
}

etl_dag = DAG(
    'etl_retail',
    default_args=default_dag_args,
    description='DAG for ETL retail process',
    schedule_interval=timedelta(days=1),
    tags=['retail']
)
```

负责运行每个 ETL 过程的 PyhtonOperators:

```py
etl_manager = DataETLManager(DATA_PATH, 'OnlineRetail.csv')

extract = PythonOperator(
    task_id='extract_data',
    python_callable=etl_manager.extract_data,
    dag=etl_dag
)

transform = PythonOperator(
    task_id='transform_data',
    python_callable=etl_manager.transform_data,
    dag=etl_dag
)

load = PythonOperator(
    task_id='load_data',
    python_callable=etl_manager.load_data,
    dag=etl_dag
)
```

最后，我们定义任务相关性:提取，然后转换，然后加载到数据库中。

```py
extract >> transform >gt; load
```

## 运行预测和基准测试结果

在这一部分中，我们将把我们获得的经过转换和简化的数据分成训练集和测试集，并使用三种集成算法，使用我们之前提取的 4 个因素来预测严重的拖欠行为。

我们将把我们所有的 ML 开发与 Neptune 集成在一起，以跟踪和比较我们一路上得到的所有指标。

* * *

请注意，由于最近的 [API 更新](/web/20221117203556/https://neptune.ai/blog/neptune-new)，这篇文章也需要一些改变——我们正在努力！与此同时，请检查[海王星文档](https://web.archive.org/web/20221117203556/https://docs.neptune.ai/)，那里的一切都是最新的！

* * *

使用三个 ML 模型，我们将能够比较结果，并看到每种方法在哪里可以表现得更好。我们将为此特定任务测试三个 ML 模型:

*   逻辑回归
*   决策树
*   极端梯度推进

### 为发展设定海王星

安装所需的 neptune 客户端库，开始将您的代码与 Neptune 集成:

安装 neptune 库:

```py
pip  install neptune-client
```

安装 Neptune 笔记本，这样可以将我们所有的工作保存到 Neptune 网站

```py
pip install -U neptune-notebooks
```

通过安装以下扩展来启用 jupiter 集成

```py
jupyter nbextension enable --py neptune-notebooks
```

获取您的 api 密钥，将您的笔记本与您的 Neptune 会话连接起来:

![ML-development-API](img/fbebb4c833672af60262b6d9cd07ba05.png)

*Set up your Neptune Token from the website *

要完成设置，请在笔记本中导入 neptune 客户端库，并调用 neptune.init()方法初始化连接:

```py
import neptune
neptune.init(project_qualified_name='aymane.hachcham/CreditScoring')
```

### 逻辑回归

回归模型的主要目标是根据特定的数学标准找到最适合一组数据的预测值组合。

在分类反应数据问题中最常用的回归模型是逻辑回归。通常，逻辑回归使用逻辑函数的基本形式，根据各种预测值对二元因变量进行建模。

该模型的训练非常简单。我们使用著名的机器学习库 *SciKit-Learn* 来实现逻辑模型。之前在数据处理方面所做的努力将在这一部分对我们有很大帮助。

包含先前定义的因子的表格如下所示:

![ETL final_table](img/d2181a816a555649f03c9524d163d54f.png)

*Table that aggregates the values for the 4 factors*

```py
credit_scoring_final = pd.DataFrame(new_data_frame, columns=['Financial_Struggle', 'Finance_Requirements', 'Expendable_Income', 'Behavioral_LifeStyle'])
credit_scoring_final
```

**分离训练集和测试集中的数据:**

```py
X = credit_scoring_final
x_train, x_test, y_train, y_test = train_test_split(X, target, test_size=0.25, random_state=56)
```

**用逻辑回归训练:**

```py
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=0.00026366508987303583, class_weight=None, dual=False, max_iter=100, multi_class='auto', n_jobs=None, penalty='l1',
random_state=None, solver='saga')
model.fit(x_train, y_train)
```

**测试结果:**

```py
from sklearn.metrics import accuracy_score, classification_report

predictions = model.predict(x_test)
test_accuracy = accuracy_score(y_test, predictions)
```

| 准确(性) | 回忆 | 精确 | f1-分数 | 因素 4 |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

**ROC 曲线和混淆矩阵:**

![ROC Curve and Confusion Matrix](img/475bd039be647ec88f16c26a114b4b92.png)

*ROC Curve and Confusion Matrix for the Logistic Regression*

由于坏账在数据中所占的比例，该模型总是能预测出比好债务人更好的坏账。这是不可避免的，回归模型无法超越这一限制，无论涉及的工程水平如何。

### XGBoost

**EXtreme Gradient Boosting**(XGBoost)是华盛顿大学博士生陈天琦(Tianqi Chen)创建的梯度增强的优化和并行化开源实现。

XGBoost 使用决策树(像 random forest)来解决分类(二进制和多类)、排序和回归问题。所以，我们在这里的监督学习算法领域。

从初始化海王星实验开始:

```py
import neptune
from neptunecontrib.monitoring.xgboost import neptune_callback

params_attempt3 = {
    'max_depth':10,
    'learning_rate':0.001,
    'colsample_bytree': 0.7,
    'subsample':0.8,
    'gamma': 0.3,
    'alpha':0.35,
    'n_estimator': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'error'
}

neptune.create_experiment(
    name='CreditScoring XGB',
    tags=['XGBoost', 'Credit Scoring'],
    params=params
)
```

拆分数据并实例化 DMatrix 数据加载器:

```py
x_train, x_test, y_train, y_test = train_test_split(X, target, test_size=0.25, random_state=56)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
```

让我们开始训练模型，并使用[Neptune XGBoost integration](https://web.archive.org/web/20221117203556/https://docs.neptune.ai/essentials/integrations/machine-learning-frameworks/xgboost)跟踪每个指标。

```py
import xgboost as xgb
import neptune
from neptunecontrib.monitoring.xgboost import neptune_callback

xgb_classifer = xgb.XGBClassifier(**params_attempt3)
xgb_classifer.fit(
    x_train,
    y_train,
    eval_set=[(x_test, y_test)],
    callbacks=[neptune_callback(log_tree=[0, 1])])

neptune.stop()
```

回到 Neptune 查看损失指标和特性重要性图:

![Charts of train and eval loss](img/53aab21fb9043ed4430c21f3b8ada323.png)

*Charts of train and eval loss: Neptune web UI*

![Feature importance graph ](img/41ebf272918226cc3d83d49558514f58.png)

*Feature importance graph for the 4 factors*

我们甚至可以看看模型的内部估计器:XGBoost 内部使用的树的图形。

[![Internal XGBoost estimators](img/37637cd366efa264f208a0655c8288b4.png)](https://web.archive.org/web/20221117203556/https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/Internal-XGBoost-estimators.png?ssl=1)

*Internal XGBoost estimators (click to enlarge)*

XGBoost 结果:

准确(性)

| 回忆 | 精确 | f1-分数 |  |
| --- | --- | --- | --- |
|  |  |  | 决策图表 |

决策树的优点是解释简单、训练快速、非参数化，并且只需要很少的数据预处理。它们可以通过监督学习算法自动计算，该算法能够在非结构化和潜在的大数据中自动选择区别变量。

我们将为这个准备好的数据集测试决策树的一个轻量级实现，然后对三个模型进行基准测试，看看哪个性能更好。

![](img/2647dab2832f9818ce5ec01eb29166d3.png)

*ROC Curve and Confusion Matrix for the XGBoost performer*

### 训练模型

结果:

准确(性)

### 回忆

```py
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
```

精确

```py
preds = classifier.predict(x_test)
print('Accuracy Score: ', metrics.accuracy_score(y_test, preds))
```

f1-分数

|  |  |  |  |
| --- | --- | --- | --- |
| 三个模型的基准 | 在多次训练逻辑回归(LR)、决策树(DT)和极端梯度推进(XGBoost)以避免单一结果偏差之后，将使用每个模型的最佳执行迭代来进行比较。 |  |  |

我们看到所有的模型在准确性方面都非常接近，尽管总的来说 **XGBoost** 做得更好。最有趣的度量是 F1 分数，它更好地评估了模型在识别正确类别方面的混乱程度。 **XGBoost** 在预测正确的类别和区分“*坏*”和“*好*”债务人方面做得更好。

结论

![](img/3b55c843ba1c377bbd80d43ee6292f62.png)

*ROC Curve and Confusion Matrix for the Decision Tree model*

### 在本教程中，我们对数据科学工作中涉及的不同方面进行了全面考察，例如:

数据预处理和转换，

数据管道自动化，

统计分析和降维，

ML 开发和培训，

## 对多个模型变量进行基准测试并选择正确的一个，

使用 Neptune 的 ML 工作流。

*   我希望这篇教程对你有用，因为我已经把它设计成完全覆盖真实数据科学用例的不同方面。如果你觉得你对知识的渴望仍然需要冷却，请查看下面的参考资料。玩得开心！
*   参考
*   Statistical Analysis and Dimensionality reduction,
*   ML development and training,
*   Benchmarking multiple model variants and picking the right one,
*   ML workflow using Neptune.

I hope this tutorial was useful to you, as I’ve designed it to fully cover different aspects of real data science use-cases.  If you feel that your thirst for knowledge still needs quenching go ahead check the references below. Have fun!

### References