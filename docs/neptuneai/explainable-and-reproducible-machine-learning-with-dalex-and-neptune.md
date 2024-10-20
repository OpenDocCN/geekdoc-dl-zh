# 用 DALEX 和 Neptune 开发可解释和可再现的机器学习模型

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/explainable-and-reproducible-machine-learning-with-dalex-and-neptune>

机器学习模型开发很难，尤其是在现实世界中。

通常，您需要:

*   了解业务问题，
*   收集数据，
*   探索它，
*   建立适当的验证方案，
*   实现模型和调整参数，
*   以对业务有意义的方式部署它们，
*   检查模型结果只是为了找出你必须处理的新问题。

这还不是全部。

你应该有你运行的**实验**和你训练的**模型**版本**版本**，以防你或其他任何人将来需要检查它们或重现结果。从我的经验来看，这一时刻会在你最意想不到的时候到来，“我希望我以前就想到了”的感觉是如此的真实(和痛苦)。

但是还有更多。

随着 ML 模型服务于真实的人，错误分类的案例(这是使用 ML 的自然结果)正在影响人们的生活，有时会非常不公平地对待他们。这使得解释你的模型预测的能力成为一种需求，而不仅仅是一种美好。

那么你能做些什么呢？

幸运的是，今天有工具可以解决这两个问题。

最好的部分是你可以把它们结合起来**让你的模型版本化，可复制，可解释**。

**继续阅读了解:**

*   用 **DALEX** 讲解器讲解机器学习模型
*   使用 **Neptune** 让您的模型版本化，实验可重复
*   使用 **Neptune + DALEX 集成**自动保存每次训练运行的模型讲解器和交互式讲解图表
*   用**版本化解释器**比较、调试和审计你构建的每一个模型

让我们开始吧。

* * *

请注意，由于最近的 [API 更新](/web/20221031152438/https://neptune.ai/blog/neptune-new)，这篇文章也需要一些改变——我们正在努力！与此同时，请检查[海王星文档](https://web.archive.org/web/20221031152438/https://docs.neptune.ai/)，那里的一切都是最新的！🥳

* * *

## 用 DALEX 进行可解释的机器学习

亚当·赖德莱克 军情二处数据实验室的研究工程师，华沙理工大学数据科学专业的学生

przemysaw bie cekmi2 datalab 创始人，三星研发中心首席数据科学家&波兰

如今，在测试集上获得高分的模型通常是不够的。这就是为什么人们对可解释的人工智能( **XAI** )越来越感兴趣，这是一套让你理解模型行为的方法和技术。

在多种编程语言中有许多可用的 XAI 方法。机器学习中最常用的一些是*石灰*、 *SHAP、*或 *PDP* ，但还有更多。

人们很容易迷失在大量的技术中，这就是**可解释的人工智能金字塔**派上用场的地方。它将与模型探索相关的需求收集到一个可扩展的下钻图中。左边是与单个实例相关的需求，右边是与整个模型相关的需求。连续层挖掘关于模型行为(局部或全局)的越来越多的细节问题。

DALEX(在 R 和 Python 中可用)是一个工具，**帮助你理解**复杂模型是如何工作的。它目前只适用于表格数据(但将来会有文本和图像)。

它与大多数用于构建机器学习模型的流行框架相集成，如 *keras、sklearn、xgboost、lightgbm、H2O* 等等！

**DALEX** 中的核心对象是一个**讲解者**。它将训练或评估数据与已训练的模型联系起来，并提取解释这些数据所需的所有信息。

一旦有了它，您就可以创建可视化，显示模型参数，并深入了解其他与模型相关的信息。您可以与您的团队共享它，或者保存它供以后使用。

为任何模型创建一个解释器真的很容易，正如你在这个例子中看到的使用 *sklearn* ！

```py
import dalex as dx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = dx.datasets.load_titanic()
le = preprocessing.LabelEncoder()
for feature in ['gender', 'class', 'embarked']:
	data[feature] = le.fit_transform(data[feature])

X = data.drop(columns='survived')
y = data.survived

classifier = RandomForestClassifier()
classifier.fit(X, y)

exp = dx.Explainer(classifier, X, y, label = "Titanic Random Forest")
```

### **观测值的模型解释(局部解释)**

当你想理解**为什么你的模型做出了一个特定的预测**时，本地解释是你最好的朋友。

这一切都从预测开始，沿着金字塔的左半部分往下，你可以探索和了解发生了什么。

DALEX 为您提供了一系列方法来显示每个变量的局部影响:

*   [SHAP](https://web.archive.org/web/20221031152438/https://github.com/slundberg/shap) :使用经典的 Shapley 值计算特征对模型预测的贡献
*   [分解](https://web.archive.org/web/20221031152438/https://pbiecek.github.io/breakDown/):用所谓的“贪婪解释”将预测分解成可归因于每个变量的部分
*   [分解交互](https://web.archive.org/web/20221031152438/https://pbiecek.github.io/breakDown/reference/break_down.html):扩展“贪婪解释”来解释功能交互

沿着金字塔往下走，局部解释的下一个关键部分是**理解模型**对特征值变化的敏感性。

在 DALEX 中有一种简单的方法来绘制这些信息:

*   [其他条件不变](https://web.archive.org/web/20221031152438/https://github.com/pbiecek/ceterisParibus):显示模型预测的变化，仅允许单个变量存在差异，同时保持所有其他变量不变

按照我们在 Titanic 数据集上创建的示例随机森林模型，我们可以很容易地创建上面提到的图。

```py
observation = pd.DataFrame({'gender': ['male'],
                   	    'age': [25],
                   	    'class': ['1st'],
                   	    'embarked': ['Southampton'],
                       	    'fare': [72],
                   	    'sibsp': [0],
                   	    'parch': 0},
                  	    index = ['John'])

bd = exp.predict_parts(observation , type='break_down')
bd_inter = exp.predict_parts(observation, type='break_down_interactions')
bd.plot(bd_inter)

shap = exp.predict_parts(observation, type = 'shap', B = 10)
shap.plot(max_vars=5)

cp = exp.predict_profile(observation)
cp.plot(variable_type = "numerical")
cp.plot(variable_type = "categorical")
```

### **模型理解(全局解释)**

当你想了解**哪些特性对你的模型**通常是重要的，当它做决定时，你应该寻找全局的解释。

为了在全球层面上理解该模型，DALEX 为您提供了变量重要性图。变量重要性图，特别是[排列特征重要性](https://web.archive.org/web/20221031152438/https://christophm.github.io/interpretable-ml-book/feature-importance.html)，使用户能够从整体上理解每个变量对模型的影响，并区分最重要的变量。

这种可视化可以被视为 SHAP 的全球等价物，并分解描绘单次观察的相似信息的图。

沿着金字塔向下移动，在数据集层面上，有一些技术，如部分相关性分布图和累积局部相关性，使您能够**将模型的反应方式可视化为所选变量的函数。**

现在让我们为我们的例子创建一些全局解释。

```py

vi = exp.model_parts()
vi.plot(max_vars=5)

pdp_num = exp.model_profile(type = 'partial')
ale_num = exp.model_profile(type = 'accumulated')

pdp_num.plot(ale_num)

pdp_cat = exp.model_profile(type = 'partial', 
variable_type='categorical',
variables = ["gender","class"])
ale_cat = exp.model_profile(type = 'accumulated',
          variable_type='categorical',
          variables = ["gender","class"])

ale_cat.plot(pdp_cat)
```

### **可重用和有组织的解释对象**

一个干净的，结构化的，易于使用的 XAI 可视化集合是伟大的，但有更多的 DALEX。

在 **DALEX explainers** 中打包你的模型给你一个**可重用的和有组织的方式来存储和版本化**你用机器学习**模型**做的任何工作。

使用 DALEX 创建的 explainer 对象包含:

*   要解释的模型，
*   型号名称和类别，
*   任务类型，
*   将用于计算解释的数据，
*   对这些数据进行模型预测，
*   预测功能，
*   模型残差，
*   观察值的抽样权重，
*   其他型号信息(包、版本等。)

将所有这些信息存储在一个对象中使得创建局部和全局解释变得容易(正如我们之前看到的)。

它还使得在模型开发的每个阶段审查、共享和比较模型和解释成为可能。

## 使用 Neptune 进行实验和模型版本控制

 [雅各布·查孔 资深数据科学家](https://web.archive.org/web/20221031152438/https://www.linkedin.com/in/jakub-czakon-2b797b69) 

在完美的世界里，你所有的机器学习模型和实验都是以和你给软件项目版本化一样的方式来版本化的。

不幸的是，要跟踪您的 ML 项目，您需要做的不仅仅是将代码提交给 Github。

简而言之，为了正确地**版本化机器学习模型**你应该跟踪:

*   代码、笔记本和配置文件
*   环境
*   因素
*   数据集
*   模型文件
*   评估指标、性能图表或预测等结果

其中一些东西可以很好地配合。git(代码、环境配置),但其他的就没那么多了。

海王星通过让你记录你觉得重要的每件事和任何事情，让你很容易跟踪所有这些。

您只需在脚本中添加几行代码:

```py
import neptune
from neptunecontrib.api import *
from neptunecontrib.versioning.data import *

neptune.init('YOU/YOUR_PROJECT')

neptune.create_experiment(
          params={'lr': 0.01, 'depth': 30, 'epoch_nr': 10}, 
          upload_source_files=['**/*.py', 
                               'requirements.yaml']) 
log_data_version('/path/to/dataset') 

neptune.log_metric('test_auc', 0.82) 
log_chart('ROC curve', fig) 
log_pickle('model.pkl', clf) 

```

您运行的每个实验或模型训练都有版本，并在 Neptune 应用程序(和数据库)中等待您🙂 ).

[**见海王星**](https://web.archive.org/web/20221031152438/https://ui.neptune.ai/o/shared/org/dalex-integration/e/DAL-79/details)

您的团队可以访问所有的实验和模型，比较结果，并快速找到信息。

您可能会想:“好极了，所以我有我的模型版本，但是”:

*   如果我想在模型被训练后几周或几个月调试模型，该怎么办？
*   如果我想看到每次实验运行的预测解释或变量重要性，该怎么办？
*   如果有人让我检查这个模型是否有不公平的偏见，而我没有训练它的代码或数据，该怎么办？

我听到了，这就是 DALEX 集成的用武之地！

## DALEX + Neptune =版本化和可解释的模型

为什么不让你的 DALEX **解释器为每个实验**进行记录和版本化，并在一个漂亮的用户界面中呈现交互式解释图表，易于与你想要的任何人分享。

没错，为什么不呢！

通过 Neptune-DALEX 集成，您可以获得所有这些，只需增加 3 条线路。

此外，这也带来了一些实实在在的好处:

*   你可以**查看其他人创建的模型**并轻松分享你的模型
*   您可以**比较**任何已创建模型的行为
*   你可以**追踪和审计每个模型**不必要的偏差和其他问题
*   您可以**调试**并比较缺少训练数据、代码或参数的模型

好吧，这听起来很酷，但它实际上是如何工作的呢？

我们现在开始吧。

### **版本本地说明**

要记录本地模型说明，您只需:

*   创建一个观察向量
*   创建您的 DALEX 解释器对象
*   将它们从`neptunecontrib`传递给`log_local_explanations`函数

```py
from neptunecontrib.api import log_local_explanations

observation = pd.DataFrame({'gender': ['male'],
                   	    'age': [25],
                   	    'class': ['1st'],
                   	    'embarked': ['Southampton'],
                       	    'fare': [72],
                   	    'sibsp': [0],
                   	    'parch': 0},
                  	    index = ['John'])

log_local_explanations(expl, observation)

```

互动解释图表将在海王星应用程序的“文物”部分等着你:

[**见海王星**](https://web.archive.org/web/20221031152438/https://ui.neptune.ai/shared/dalex-integration/e/DAL-78/artifacts?path=charts%2F&file=SHAP.html)

将创建以下地块:

*   可变重要性
*   部分相关(如果指定了数字特征)
*   累积相关性(如果指定了分类特征)

### **版本全球解说**

有了全球模型解释，就更简单了:

*   创建您的 DALEX 解释器对象
*   从`neptunecontrib`传递给`log_global_explanations`函数
*   (可选)指定要绘制的分类特征

```py
from neptunecontrib.api import log_global_explanations

log_global_explanations(expl, categorical_features=["gender", "class"])
```

就是这样。现在，您可以转到“工件”部分，找到您当地的解释图表:

[**见海王星**](https://web.archive.org/web/20221031152438/https://ui.neptune.ai/o/shared/org/dalex-integration/e/DAL-78/artifacts?path=charts%2F&file=Variable%20Importance.html)

将创建以下地块:

*   崩溃，
*   在互动中崩溃，
*   shap，
*   对于数值变量，其他条件不变，
*   分类变量的其他条件不变

### **版本讲解器对象**

但是如果你真的想要版本化你的解释，你应该**版本化解释对象**本身。

保存它的好处？：

*   您可以在以后创建它的可视化表示
*   您可以用表格的形式深入了解细节
*   你可以随心所欲地使用它(即使你现在不知道如何使用)🙂)

这非常简单:

```py
from neptunecontrib.api import log_explainer

log_explainer('explainer.pkl', expl)
```

你可能会想:“我还能怎么使用解释器对象呢？”

让我在接下来的部分向您展示。

### **获取并分析训练模型的解释**

首先，如果你把你的解释器登录到 Neptune，你可以直接把它提取到你的脚本或者笔记本中:

```py
import neptune
from neptunecontrib.api import get_pickle

project = neptune.init(api_token='ANONYMOUS',
                       project_qualified_name='shared/dalex-integration')
experiment = project.get_experiments(id='DAL-68')[0]
explainer = get_pickle(filename='explainer.pkl', experiment=experiment)
```

现在您已经有了模型解释，您可以调试您的模型了。

一种可能的情况是，你有一个观察结果，而你的模型悲惨地失败了。

你想找出原因。

如果保存了 DALEX 解释器对象，您可以:

*   创建本地解释，看看发生了什么。
*   检查更改要素对结果的影响。

[**见海王星**](https://web.archive.org/web/20221031152438/https://ui.neptune.ai/shared/dalex-integration/n/6b9d8213-9d1c-4a7d-a448-d2d9e29f7878/66389c83-b397-4ec6-a1fc-8b5847980fe5)

当然，你可以做得更多，特别是如果你想比较模型和解释。

让我们现在就开始吧！

### **比较模型和解释**

如果您想:

*   将当前的模型想法与生产中运行的模型进行比较？
*   看看去年的实验想法是否能在新收集的数据上发挥更好的作用？

实验和模型有一个清晰的结构，并且有一个存放它们的地方，这真的很容易做到。

您可以在 Neptune UI 中基于参数、数据版本或指标来比较实验:

[**见海王星**](https://web.archive.org/web/20221031152438/https://ui.neptune.ai/o/shared/org/dalex-integration/compare?shortId=%5B%22DAL-78%22%2C%22DAL-77%22%2C%22DAL-76%22%2C%22DAL-75%22%2C%22DAL-72%22%5D&viewId=495b4a41-3424-4d01-9064-70be82716196)

您**可以通过两次点击**看到不同之处，并可以通过一两次点击深入查看您需要的任何信息。

好吧，当涉及到比较超参数和度量时，这确实很有用，但是解释者呢？

你可以进入每个实验，并[查看交互式解释图表](https://web.archive.org/web/20221031152438/https://ui.neptune.ai/o/shared/org/dalex-integration/e/DAL-78/artifacts?path=charts%2F&file=Break%20Down%20Interactions.html)，看看你的模型是否有可疑之处。

更好的是，Neptune 允许您访问您以编程方式记录的所有信息，包括模型解释器。
你可以**获取每个实验的讲解对象并进行比较**。只需使用`neptunecontrib`中的`get_pickle`函数，然后用 DALEX `.plot`可视化多个解释器:

```py
experiments =project.get_experiments(id=['DAL-68','DAL-69','DAL-70','DAL-71'])

shaps = []
for exp in experiments:
	auc_score = exp.get_numeric_channels_values('auc')['auc'].tolist()[0]
	label = f'{exp.id} | AUC: {auc_score:.3f}'

	explainer_ = get_pickle(filename='explainer.pkl', experiment=exp)

	sh = explainer_.predict_parts(new_observation, type='shap', B = 10)
	sh.result.label = label
	shaps.append(sh)

shaps[0].plot(shaps[1:])

```

[**见海王星**](https://web.archive.org/web/20221031152438/https://ui.neptune.ai/o/shared/org/dalex-integration/n/comparison-6b9d8213-9d1c-4a7d-a448-d2d9e29f7878/4bf30571-1ef1-4c63-9241-6d3b2cab65a7)

这就是 DALEX 情节的美妙之处。你可以通过多个解释者，他们会变魔术。

当然，您可以将之前训练过的模型与您当前正在研究的模型进行比较，以确定您的方向是否正确。只需将它添加到解释器列表中，并传递给`.plot`方法。

## **最终想法**

好了，总结一下。

在本文中，您了解了:

*   各种模型解释技术以及如何用 DALEX 解释器打包这些解释
*   如何用 Neptune 对机器学习模型和实验进行版本化
*   如何通过 Neptune + DALEX 集成为您运行的每个培训版本化模型解释器和交互式解释图表
*   如何比较和调试你用解释器训练的模型

有了这些信息，我希望您的模型开发过程现在会更有组织性、可再现性和可解释性。

快乐训练！

### 雅各布·查孔

大部分是 ML 的人。构建 MLOps 工具，编写技术资料，在 Neptune 进行想法实验。

* * *

**接下来的步骤**

## 如何在 5 分钟内上手海王星

##### 1.创建一个免费帐户

2.安装 Neptune 客户端库

[Sign up](/web/20221031152438/https://neptune.ai/register)

3.将日志记录添加到脚本中

##### 2\. Install Neptune client library

```py
pip install neptune-client

```

##### 3\. Add logging to your script

```py
import neptune.new as neptune

run = neptune.init_run("Me/MyProject")
run["parameters"] = {"lr":0.1, "dropout":0.4}
run["test_accuracy"] = 0.84
```

[Try live notebook](https://web.archive.org/web/20221031152438/https://colab.research.google.com/github/neptune-ai/examples/blob/master/how-to-guides/how-it-works/notebooks/Neptune_API_Tour.ipynb)

* * *