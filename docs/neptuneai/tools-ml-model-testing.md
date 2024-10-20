# 帮助您设置生产 ML 模型测试的 5 个工具

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/tools-ml-model-testing>

[开发机器学习或深度学习模型](/web/20230106144030/https://neptune.ai/categories/ml-model-development)似乎是一项相对简单的任务。它通常包括研究、收集和预处理数据、提取特征、建立和训练模型、评估和推理。大部分时间消耗在[数据预处理阶段](/web/20230106144030/https://neptune.ai/blog/data-preprocessing-guide)，随后是建模阶段。如果精度没有达到标准，我们就重复整个过程，直到我们找到满意的精度。

当我们想在现实世界中将模型投入生产时，困难就出现了。模型的表现通常不如在训练和评估阶段时好。这主要是因为[概念漂移](/web/20230106144030/https://neptune.ai/blog/concept-drift-best-practices)或数据漂移以及有关数据完整性的问题。因此，测试一个 ML 模型变得非常重要，以便我们可以了解它的优点和缺点，并采取相应的行动。

在本文中，我们将讨论一些可以用来测试 ML 模型的工具。这些工具和库有些是开源的，有些则需要订阅。无论是哪种方式，本文都将全面探讨对您的 MLOps 管道有用的工具。

## 为什么模型测试很重要？

在我们刚刚讨论的基础上，模型测试允许您查明可能导致模型预测能力下降的 bug 或关注区域。这可以随着时间的推移逐渐发生，也可以在瞬间发生。无论哪种方式，知道他们可能在哪个方面失败以及哪些特性会导致他们失败总是好的。它暴露了缺陷，也能带来新的见解。本质上，这个想法是要建立一个健壮的模型，可以有效地处理不确定的数据条目和异常。

模型测试的一些好处是:

## 

*   1 检测模型和数据漂移

*   2 在数据集中发现异常

*   3 检查数据和模型完整性

*   4 检测模型故障的可能根本原因

*   消除 bug 和错误

*   6 减少误报和漏报

*   7 鼓励在一定时间内对模特进行再培训

*   8 打造生产就绪型车型

*   9 确保 ML 模型的稳健性

*   10 在模型中寻找新的见解

### 模型测试和模型评估一样吗？

模型测试和评估类似于我们在医学上所说的诊断和筛选。

**模型评估**类似于诊断，根据 F1 分数或 MSE 损失等指标检查模型的性能。这些指标并没有提供关注的焦点。

**模型测试**类似于诊断，不变性测试和单元测试等特定测试旨在发现模型中的特定问题。

## 典型的 ML 软件测试套件包括什么？

机器学习测试套件通常包括测试模块来**检测不同类型的漂移**，如概念漂移和数据漂移，这可以包括协变漂移、预测漂移等。这些问题通常发生在数据集中。大多数情况下，数据集的分布会随时间而变化，从而影响模型准确预测输出的能力。您会发现我们将要讨论的框架将包含检测数据漂移的工具。

除了测试数据，ML 测试套件还包含测试**模型预测能力、**以及**过拟合、欠拟合、方差和偏差**等的工具。测试框架的想法是在开发的三个主要阶段检查管道:

*   数据摄取，
*   数据预处理，
*   和模型评估。

一些框架，如 Robust Intelligence 和 Kolena，在这些给定的领域中严格地自动测试给定的 ML 管道，以确保生产就绪的模型。

本质上，机器学习套件将包含:

1.  **单元测试**在代码库的层次上操作，
2.  **回归测试**复制模型上一次迭代中被修复的错误，
3.  **集成测试**模拟条件，通常是观察模型行为的长期运行测试。这些条件可以反映 ML 管道，包括预处理阶段、数据分布等等。

![A workflow of software development ](img/dac82513e118435e9a58eccddb8d604b.png)

*The image above depicts a typical workflow of software development | [Source](https://web.archive.org/web/20230106144030/https://www.jeremyjordan.me/testing-ml/)*

现在，让我们讨论一些测试 ML 模型的工具。本节分为三个部分:开源工具、基于订阅的工具和混合工具。

### 开源模型测试工具

#### 1.深度检查

[DeepChecks](https://web.archive.org/web/20230106144030/https://deepchecks.com/) 是一个用于测试 ML 模型&数据的开源 Python 框架。它基本上使用户能够在三个不同的阶段测试 ML 管道:

1.  **预处理阶段前的数据完整性测试**。
2.  **数据验证，**在训练之前，主要是在将数据分为训练和测试时，以及
3.  **ML 模型测试**。

![](img/c623bc70443b49d45e872be586823725.png)

*The image above shows the schema of three different tests that could be performed in an ML pipeline | [Source](https://web.archive.org/web/20230106144030/https://docs.deepchecks.com/stable/getting-started/when_should_you_use.html)*

这些测试可以同时进行，甚至可以独立进行。上图显示了可以在 ML 管道中执行的三个不同测试的模式。

##### 装置

可以使用以下 pip 命令安装 Deepchecks:

```py
pip install deepchecks > 0.5.0
```

Deepcheck 的最新版本是 0.8.0。

##### 框架的结构

DeepChecks 引入了三个重要的术语:**检查**、**条件**和**套件**。值得注意的是，这三个术语共同构成了框架的核心结构。

**检查**

它使用户能够检查数据和模型的特定方面。该框架包含各种类，允许您检查这两者。你也可以做全面检查。下面是一些这样的检查:

1.  ***数据检查*** 涉及围绕数据漂移、重复、缺失值、字符串不匹配、数据分布等统计分析的检查*。*您可以在检查模块中找到各种数据检查工具。校验模块允许您精确设计数据集的检查方法。这些是您可以找到的一些数据检查工具:

*   '数据重复'，
*   DatasetsSizeComparison '，
*   ' DateTrainTestLeakageDuplicates '，
*   DateTrainTestLeakageOverlap '，
*   '支配频率变化'，
*   '特征特征相关性'，
*   '功能标签相关性'，
*   ' FeatureLabelCorrelationChange '，
*   '标识符标签相关性'，
*   ' IndexTrainTestLeakage '，
*   ' IsSingleValue '，
*   '混合数据类型'，
*   MixedNulls '，
*   整体数据集漂移

在下面的例子中，我们将检查数据集是否有重复。我们将从 checks 模块导入类 DataDuplicates，并将数据集作为参数传递。这将返回一个包含数据集是否有重复值的相关信息的表。

```py
from deepchecks.checks import DataDuplicates, FeatureFeatureCorrelation
dup = DataDuplicates()
dup.run(data)

```

![Inspection of dataset duplicates ](img/a52f730f6a7739858b638e51430a2f09.png)

*An example of inspecting if the dataset has duplicates | Source: Author*

如您所见，上表给出了数据集中重复项数量的相关信息。现在让我们看看 DeepChecks 是如何使用可视化工具来提供相关信息的。

在以下示例中，我们将检查数据集中的要素之间的相关性。为此，我们将从 checks 模块导入 FeatureFeatureCorrelation 类。

```py
ffc = FeatureFeatureCorrelation()
ffc.run(data)

```

![ Inspection of feature-feature correlation](img/ff097a91b096307d02c68b6add1121b1.png)

*An example of inspecting feature-feature correlation within the dataset | Source: Author*

从这两个例子中可以看出，结果可以以表格或图形的形式显示，甚至可以两种形式都显示，以便向用户提供相关信息。

2.  ***模型检验*** 涉及过拟合、欠拟合等*。*与数据检查类似，您也可以在检查模块中找到各种模型检查工具。这些是您将找到的用于模型检查的一些工具:

*   ModelErrorAnalysis '，
*   '模型推理时间'，
*   模型信息'，
*   '多模型性能报告'，
*   '新标签培训测试'，
*   ' OutlierSampleDetection '，
*   绩效报告'，
*   回归误差分布'，
*   '回归系统错误'，
*   RocReport '，
*   分段性能'，
*   简单模型比较'，
*   SingleDatasetPerformance '，
*   特殊字符'，
*   StringLengthOutOfBounds '，
*   字符串匹配，
*   StringMismatchComparison '，
*   '训练测试特征漂移'，
*   ' TrainTestLabelDrift '，
*   '训练测试性能'，
*   '培训测试预测漂移'，

随机森林分类器的模型检查或检验示例:

```py
from deepchecks.checks import ModelInfo
info = ModelInfo()
info.run(RF)

```

![A model check or inspection on Random Forest Classifier](img/70a1c0c40596c541fe0e647c6cd21a06.png)

*An example of a model check or inspection on Random Forest Classifier | Source: Author *

**条件**

它是可以添加到支票中的功能或属性。本质上，它包含一个预定义的参数，可以返回通过、失败或警告结果。这些参数也可以相应地修改。按照下面的代码片段来理解。

```py
from deepchecks.checks import ModelInfo
info = ModelInfo()
info.run(RF)

```

![A bar graph of feature label correlation](img/c3f974e57abd4a88d743c157d8a81246.png)

*An example of a bar graph of feature label correlation | Source: Author*

上图显示了要素标注相关性的条形图。它本质上衡量的是一个独立特征本身能够预测目标值的预测能力。如上例所示，当您向检查添加条件时，该条件将返回附加信息，提及高于和低于该条件的特征。

在此特定示例中，您会发现条件返回了一条语句，表明算法"*发现 4 个特征中有 2 个特征的 PPS 高于阈值:{ '花瓣宽度(cm)': '0.9 '，'花瓣长度(cm)': '0.87'}* "这意味着具有高 PPS 的特征适合预测标签。

**组曲**

它是一个包含数据和模型检查集合的模块。这是一个有序的支票集合。所有的检查都可以在套件模块中找到。下面是框架的示意图以及它是如何工作的。

![Schematic diagram of suite of checks ](img/e7f6ea31676e173bd2667db3272fa55e.png)

*The schematic diagram of the suite of checks and how it works | [Source](https://web.archive.org/web/20230106144030/https://medium.com/@ptannor/new-open-source-for-validating-and-testing-machine-learning-86bb9c575e71) *

从上图中可以看出，数据和模型可以传递到包含不同检查的套件中。这些检查可以为更精确的测试提供条件。

您可以运行以下代码来查看 DeepChecks 提供的 35 个检查及其条件的列表:

```py
from deepchecks.suites import full_suite
suites = full_suite()
print(suites)
Full Suite: [
	0: ModelInfo
	1: ColumnsInfo
	2: ConfusionMatrixReport
	3: PerformanceReport
		Conditions:
			0: Train-Test scores relative degradation is not greater than 0.1
	4: RocReport(excluded_classes=[])
		Conditions:
			0: AUC score for all the classes is not less than 0.7
	5: SimpleModelComparison
		Conditions:
			0: Model performance gain over simple model is not less than
…]

```

总之，Check、Condition 和 Suites 允许用户在各自的任务中检查数据和模型。这些可以根据项目的需求和各种用例进行扩展和修改。

DeepChecks 允许以较少的努力实现 ML 管道的灵活性和即时验证。他们强大的样板代码可以让用户自动化整个测试过程，这可以节省很多时间。

![Graph with distribution checks](img/171bbb8d9048ace1360abf1d1af65c0c.png)

*An example of distribution checks | [Source](https://web.archive.org/web/20230106144030/https://deepchecks.com/)*

##### 为什么要用这个？

*   它是开源和免费的，并且有一个不断增长的社区。
*   结构非常好的框架。
*   因为它有内置的检查和套件，所以对于检查数据和模型中的潜在问题非常有用。
*   它在研究阶段是有效的，因为它可以很容易地集成到管道中。
*   如果您主要使用表格数据集，那么 DeepChecks 非常好。
*   您还可以使用它来检查数据、模型漂移、模型完整性和模型监控。

![Methodology issues](img/aea0541b81f8390d06cc1c6255b9d5a8.png)

*An example of methodology issues | [Source](https://web.archive.org/web/20230106144030/https://deepchecks.com/)*

##### 关键特征

## 1 它支持计算机视觉和表格数据集中的分类和回归模型。

*   只需一次调用，它就能轻松运行大量的检查。

*   它是灵活的，可编辑的，可扩展的。

*   它以表格和可视格式生成结果。

*   5 它不需要登录仪表板，因为包括可视化在内的所有结果都会在执行过程中即时显示。它有一个非常好的 UX。
*   主要缺点

![Performance checks ](img/37dd36025e6801d554ca867d1c78a61f.png)

*An example of performance checks | [Source](https://web.archive.org/web/20230106144030/https://deepchecks.com/)*

##### 它不支持 NLP 任务。

## 2 深度学习支持处于测试版，包括计算机视觉。所以结果会产生错误。

*   2.漂流者-ML
*   Drifter ML 是专门为 Scikit-learn 库编写的 ML 模型测试工具。它还可以用来测试类似于深度检查的数据集。它有五个模块，每个模块都与手头的任务密切相关。

#### **分类测试:**用于测试分类算法。

**回归测试:**测试分类算法。

1.  **结构测试:**这个模块有一堆允许测试聚类算法的类。
2.  **时间序列测试:**该模块可用于测试模型漂移。
3.  **列测试:**这个模块允许你测试你的表格数据集。测试包括健全性测试、均值和中值相似性、皮尔逊相关等等。
4.  装置
5.  框架的结构

##### Drifter ML 符合模型的 Scikit-Learn 蓝图，即模型必须包含. fit 和。预测方法。这实质上意味着您也可以测试深度学习模型，因为 Scikit-Learn 有一个集成的 Keras API。查看下面的[示例](https://web.archive.org/web/20230106144030/https://drifter-ml.readthedocs.io/en/latest/introduction.html)。

```py
pip install drifter-ml
```

##### 上面的例子显示了使用 drifter-ml 设计人工神经网络模型的简易性。类似地，你也可以设计一个测试用例。在下面定义的测试中，我们将尝试找到最低决策边界，通过该边界，模型可以轻松地对这两个类进行分类。

为什么要用这个？

```py

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import numpy as np
import joblib

def create_model():

   model = Sequential()
   model.add(Dense(12, input_dim=3, activation='relu'))
   model.add(Dense(8, activation='relu'))
   model.add(Dense(1, activation='sigmoid'))

   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   return model

df = pd.DataFrame()
for _ in range(1000):
   a = np.random.normal(0, 1)
   b = np.random.normal(0, 3)
   c = np.random.normal(12, 4)
   if a + b + c > 11:
       target = 1
   else:
       target = 0
   df = df.append({
       "A": a,
       "B": b,
       "C": c,
       "target": target
   }, ignore_index=True)

clf = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
X = df[["A", "B", "C"]]
clf.fit(X, df["target"])
joblib.dump(clf, "model.joblib")
df.to_csv("data.csv")

```

Drifter-ML 是专门为 Scikit-learn 编写的，这个库充当了它的扩展。所有的类和方法都是与 Scikit-learn 同步编写的，因此数据和模型测试变得相对容易和简单。

```py
def test_cv_precision_lower_boundary():
   df = pd.read_csv("data.csv")
   column_names = ["A", "B", "C"]
   target_name = "target"
   clf = joblib.load("model.joblib")

   test_suite = ClassificationTests(clf,
   df, target_name, column_names)
   lower_boundary = 0.9
   return test_suite.cross_val_precision_lower_boundary(
       lower_boundary
   )
```

##### 另外，如果你喜欢在开源库上工作，那么你也可以将这个库扩展到其他机器学习和深度学习库，比如 Pytorch。

*   关键特征

*   1 构建在 Scikit-learn 之上。

##### 2 提供深度学习架构的测试，但仅针对 Keras，因为它在 Scikit-learn 中进行了扩展。

## 3 开源库，开放投稿。

*   主要缺点
*   它不是最新的，它的社区也不太活跃。

*   它不能很好地与其他库一起工作。

##### 基于订阅的工具

## 1.跪下

*   Kolena.io 是一个基于 Python 的 ML 测试框架。它还包括一个可以记录结果和见解的在线平台。Kolena 主要关注大规模的 ML 单元测试和验证过程。
*   你为什么要用这个？

### Kolena 认为，分割测试数据集方法并不像看起来那么可靠。分割数据集提供了整个人口分布的全局表示，但无法捕获粒度级别的局部表示，对于标注或分类尤其如此。仍有一些隐藏的细微特征需要被发现。这导致了模型在现实世界中的失败，即使模型在训练和评估期间在性能指标中产生了好的分数。

#### 解决该问题的一种方法是创建一个更加集中的数据集，这可以通过将给定的类分解为更小的子类以获得集中的结果，甚至创建要素本身的子集来实现。这样的数据集可以使 ML 模型能够在更细粒度的水平上提取特征和表示。这也将通过平衡偏差和方差来提高模型的性能，从而使模型在现实世界的场景中具有良好的通用性。

例如，在构建分类模型时，可以将数据集中的给定类分解为多个子集，并将这些子集分解为更细的子集。这可以让用户在各种场景中测试模型。在下表中，CAR 类通过几个测试用例进行测试，以检查模型在各种属性上的性能。

![Kolena.io dashboard](img/444cad3583c9e073ab66cdbab050f953.png)

*Kolena.io dashboard example | [Source](https://web.archive.org/web/20230106144030/https://www.kolena.io/)*

##### 另一个好处是，每当我们在现实世界中面临一个新的场景时，一个新的测试用例可以立即被设计和测试。同样，用户可以为各种任务构建更全面的测试用例，并训练或构建模型。用户还可以在每一类测试用例中生成一个关于模型性能的详细报告，并在每次迭代中将其与之前的模型进行比较。

总而言之，Kolena 提供:

python 框架的易用性

自动化工作流测试和部署

![CAR class tested against several test cases](img/2770a61fc1297c85f6c2ee37447ec2e7.png)

CAR class tested against several test cases to check the model’s performance on various attributes | [Source](https://web.archive.org/web/20230106144030/https://medium.com/kolena-ml/best-practices-for-ml-model-testing-224366d3f23c)

更快的模型调试

更快的模型部署

*   如果你正在研究一个大规模的深度学习模型，这个模型很难监控，那么 Kolena 将是有益的。
*   关键特征
*   1 支持深度学习架构。

*   Kolena Test Case Studio 为模型提供可定制的测试用例。

3 它允许用户通过去除噪声和改进注释来准备质量测试。

##### 4 它能自动诊断故障模式，并能找到与之相关的确切问题。

## 5 无缝集成到 ML 管道中。

*   主要缺点
*   1 基于订阅的模式(未提及定价)。

*   2 基于订阅的模式(未提及定价)。

*   为了下载框架，你需要一个 CloudRepo 通行证。
*   2.强健的智能

![App Kolena.io ](img/5b53852e972ea0e34187efa1481eb029.png)

*View from the Kolena.io app | Source*

##### 这是一个 E2E 洗钱平台，提供洗钱完整性方面的各种服务。该框架是用 Python 编写的，允许根据您的需要定制您的代码。该框架还集成到一个在线仪表板中，该仪表板提供了对数据和模型性能的各种测试以及模型监控的见解。从培训到后期制作阶段，所有这些服务都以 ML 模型和数据为目标。

## 为什么要用这个？

*   该平台提供的服务包括:
*   **1。人工智能压力测试(AI stress testing，**)包括数百项测试，以自动评估模型的性能并识别潜在的缺陷。
*   **2。AI Firewall，**自动在训练好的模型周围创建一个包装器，以实时保护它免受不良数据的影响。包装器是基于模型配置的。它还会自动检查数据和模型，减少手动工作和时间。

```py
pip3 install --extra-index-url "$CR_URL" kolena-client
```

### **3。AI 连续测试**，其中监控模型并自动测试已部署的模型，以检查更新和重新训练。测试包括数据漂移、错误、根本原因分析、异常检测等。在持续测试中获得的所有见解都显示在仪表板上。

强大的智能支持模型测试、部署期间的模型保护以及部署后的模型监控。由于它是一个基于 e2e 的平台，所有阶段都可以很容易地自动化，对模型进行数百次压力测试，使其为生产做好准备。如果项目相当大，那么强大的智能会给你带来优势。

![Robust intelligence ](img/0883eb97178d04d7486b53483ea2b796.png)

*Robust intelligence features | [Source](https://web.archive.org/web/20230106144030/https://www.robustintelligence.com/platform/overview)*

##### 关键特征

1 支持深度学习框架

2 灵活易用

![AI stress testing](img/b0175f1b7ac0050c70e8676c3e97d16c.png)

*Evaluating the performance of the model | [Source](https://web.archive.org/web/20230106144030/https://www.robustintelligence.com/platform/overview)*

3 可定制

![AI Firewall](img/fb73b7a1959a8022de733803b7f8334c.png)

*Prevention of model failures in production | [Source](https://web.archive.org/web/20230106144030/https://www.robustintelligence.com/platform/overview)*

4 可扩展

![AI continuous testing](img/515b2ee6e082328449b70f0401906e16.png)

*Monitoring model in production | [Source](https://web.archive.org/web/20230106144030/https://www.robustintelligence.com/platform/overview)*

主要缺点

##### 1 仅针对企业。

## 网上几乎没有详细信息。

*   昂贵:一年的订阅费用约为 6 万美元。
*   *(* [*来源*](https://web.archive.org/web/20230106144030/https://aws.amazon.com/marketplace/pp/prodview-23bciknsbkgta) ***)***
*   混合框架
*   1.Etiq.ai

##### Etiq 是一个支持人工智能/人工智能生命周期的人工智能观察平台。像 Kolena 和 Robust Intelligence 一样，该框架提供了 ML 模型测试、监控、优化和可解释性。

## Etiq 被认为是一个混合框架，因为它同时提供离线和在线实现。Etiq 有四层用途:

*   **免费和公开**:包括免费使用图书馆和仪表盘。请记住，当您登录到平台时，结果和元数据将存储在您的 dashboard 实例中，但您将获得全部好处。
*   **免费和受限**:如果你想要一个免费但私密的项目测试环境，并且不想分享任何信息，那么你可以不用登录平台就可以使用平台。请记住，您不会像登录平台时那样获得全部权益。
*   订阅和私有:如果你想获得 Etiq.ai 的全部好处，那么你可以订阅他们的计划，并在你自己的私有环境中使用他们的工具。Etiq.ai 已经在 AWS 市场上推出，起价约为 3.00 美元/小时或 25，000 美元/年。

个性化请求:如果您需要 Etiq.ai 所不能提供的功能，比如可解释性、健壮性或团队共享功能，那么您可以联系他们并获得您自己的个性化测试套件。

### 框架的结构

#### Etiq 遵循类似于 DeepChecks 的结构。这种结构仍然是框架的核心:

**快照**:是生产前测试阶段数据集和模型的结合。

![Etiq.ai](img/a30e7ab117b77aa62545b89c883350e1.png)

*The dashboard of Etiq.ai | [Source](https://web.archive.org/web/20230106144030/https://docs.google.com/document/d/1oJ20eZeuuuFigdi4P4rqgcLONZ2ulimFn4XMBQ8M9Fw/edit#heading=h.upirzig57bbx)*

**扫描**:通常是应用于快照的测试。

1.  **Config** :它通常是一个 JSON 文件，包含一组参数，扫描将使用这些参数在快照中运行测试。
2.  **定制测试**:它允许您通过向配置文件中添加和编辑各种度量来定制您的测试。
3.  Etiq 提供两种类型的测试:**扫描**和**根本原因分析**或 RCA，后者是一个实验管道。扫描类型提供
4.  **准确度**:在某些情况下，高准确度可以像低准确度一样表明有问题。在这种情况下,“精确”扫描会有所帮助。如果精度太高，那么你可以进行泄漏扫描，或者如果精度太低，那么你可以进行漂移扫描。

##### **泄漏**:帮助你发现数据泄漏。

**漂移**:可以帮你找到特征漂移、目标漂移、概念漂移、预测漂移。

*   **偏差**:偏差指的是算法偏差，这种偏差可能是由于自动决策导致非故意歧视而产生的。
*   为什么要用这个？
*   Etiq.ai 提供了一个多步骤管道，这意味着您可以通过记录 ML 管道中每个步骤的结果来监控测试。这允许您识别和修复模型中的偏差。如果你正在寻找一个框架来完成你的人工智能管道的重担，那么 Etiq.ai 是一个不错的选择。
*   您应该使用 Etiq.ai 的其他一些原因:

Etiq offers two types of tests: **Scan** and **Root Cause Analysis** or RCA, the latter is an experimental pipeline. The scan type offers

*   这是一个 Python 框架

*   2 用于多视角和优化报告的仪表板设施

*   你可以管理多个项目。
*   以上各点对免费层使用有效。

##### Etiq.ai 的一个关键特性是，它允许您在模型构建和部署方法中非常精确和直接。它旨在为用户提供工具，帮助他们实现期望的模型。有时，开发过程会偏离最初的计划，主要是因为缺少塑造模型所需的工具。如果您想要部署一个与提议的需求相一致的模型，那么 Etiq.ai 是一个不错的选择。这是因为该框架在整个 ML 管道的每一步都提供了类似的测试。

关键特征

免费层中的许多功能。

## 2 测试每条管道，以便更好地监控

*   3 支持深度学习框架，如 PyTorch 和 Keras-Tensorflow

*   你可以申请一个个性化的测试库。
*   主要缺点

目前，在生产中，它们只提供批处理功能。

2 将测试应用于与细分、回归或推荐引擎相关的任务，他们必须与团队取得联系。

![Etiq.ai ](img/b056a981f159a7c64d85be0e744b9317.png)

*Steps of the process when to use Etiq.ai | [Source](https://web.archive.org/web/20230106144030/https://docs.etiq.ai/#why-use-etiq-for-ml-testing)*

##### 结论

## 我们讨论的 ML 测试框架是针对用户需求的。所有的框架都有各自的优缺点。但是你可以通过使用这些框架中的任何一个来获得。ML 模型测试框架在定义模型在部署到真实场景时的表现方面扮演着不可或缺的角色。

*   如果你正在为结构化数据集和较小的 ML 模型寻找一个免费且易于使用的 ML 测试框架，那么请选择 DeepChecks。如果你正在使用 DL 算法，那么 Etiq.ai 是一个很好的选择。但如果你能抽出一些钱，那么你一定要打听一下科勒娜。最后，如果你在一家中大型企业工作，寻找 ML 测试解决方案，那么毫无疑问，它必须是强大的智能。
*   我希望这篇文章为您提供了开始 ML 测试所需的所有初步信息。请把这篇文章分享给每一个需要的人。
*   感谢阅读！！！
*   参考

##### [https://www.robustintelligence.com/](https://web.archive.org/web/20230106144030/https://www.robustintelligence.com/)

## [https://AWS . Amazon . com/market place/PP/prod view-23 bciknsbkk GTA](https://web.archive.org/web/20230106144030/https://aws.amazon.com/marketplace/pp/prodview-23bciknsbkgta)

*   [https://etiq.ai/](https://web.archive.org/web/20230106144030/https://etiq.ai/)
*   [https://docs.etiq.ai/](https://web.archive.org/web/20230106144030/https://docs.etiq.ai/)

## [https://arxiv.org/pdf/2005.04118.pdf](https://web.archive.org/web/20230106144030/https://arxiv.org/pdf/2005.04118.pdf)

[https://medium . com/kolena-ml/best-practices-for-ml-model-testing-224366 D3 f23c](https://web.archive.org/web/20230106144030/https://medium.com/kolena-ml/best-practices-for-ml-model-testing-224366d3f23c)

[https://docs.kolena.io/](https://web.archive.org/web/20230106144030/https://docs.kolena.io/)

[https://www.kolena.io/](https://web.archive.org/web/20230106144030/https://www.kolena.io/)

[https://github.com/EricSchles/drifter_ml](https://web.archive.org/web/20230106144030/https://github.com/EricSchles/drifter_ml)

### [https://arxiv.org/pdf/2203.08491.pdf](https://web.archive.org/web/20230106144030/https://arxiv.org/pdf/2203.08491.pdf)

1.  [https://medium . com/@ ptan nor/new-open-source-for-validating-and-testing-machine-learning-86 bb 9 c 575 e 71](https://web.archive.org/web/20230106144030/https://medium.com/@ptannor/new-open-source-for-validating-and-testing-machine-learning-86bb9c575e71)
2.  [https://deepchecks.com/](https://web.archive.org/web/20230106144030/https://deepchecks.com/)
3.  [https://www . xenon stack . com/insights/machine-learning-model-testing](https://web.archive.org/web/20230106144030/https://www.xenonstack.com/insights/machine-learning-model-testing)
4.  [https://www.jeremyjordan.me/testing-ml/](https://web.archive.org/web/20230106144030/https://www.jeremyjordan.me/testing-ml/)
5.  [https://Neptune . ai/blog/ml-model-testing-teams-share-how-they-test-models](https://web.archive.org/web/20230106144030/https://neptune.ai/blog/ml-model-testing-teams-share-how-they-test-models)
6.  https://mlops . toys
7.  [https://docs.kolena.io/](https://web.archive.org/web/20230106144030/https://docs.kolena.io/)
8.  [https://www.kolena.io/](https://web.archive.org/web/20230106144030/https://www.kolena.io/)
9.  [https://github.com/EricSchles/drifter_ml](https://web.archive.org/web/20230106144030/https://github.com/EricSchles/drifter_ml)
10.  [https://arxiv.org/pdf/2203.08491.pdf](https://web.archive.org/web/20230106144030/https://arxiv.org/pdf/2203.08491.pdf)
11.  [https://medium.com/@ptannor/new-open-source-for-validating-and-testing-machine-learning-86bb9c575e71](https://web.archive.org/web/20230106144030/https://medium.com/@ptannor/new-open-source-for-validating-and-testing-machine-learning-86bb9c575e71)
12.  [https://deepchecks.com/](https://web.archive.org/web/20230106144030/https://deepchecks.com/)
13.  [https://www.xenonstack.com/insights/machine-learning-model-testing](https://web.archive.org/web/20230106144030/https://www.xenonstack.com/insights/machine-learning-model-testing)
14.  [https://www.jeremyjordan.me/testing-ml/](https://web.archive.org/web/20230106144030/https://www.jeremyjordan.me/testing-ml/)
15.  [https://neptune.ai/blog/ml-model-testing-teams-share-how-they-test-models](https://web.archive.org/web/20230106144030/https://neptune.ai/blog/ml-model-testing-teams-share-how-they-test-models)
16.  [https://mlops.toys](https://web.archive.org/web/20230106144030/https://mlops.toys/)