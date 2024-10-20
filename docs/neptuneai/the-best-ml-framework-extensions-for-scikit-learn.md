# Scikit 的最佳 ML 框架和扩展-学习

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/the-best-ml-framework-extensions-for-scikit-learn>

许多包实现了[sci kit-learn](https://web.archive.org/web/20230204025453/https://scikit-learn.org/)estimator API。

如果您已经熟悉 Scikit-learn，您会发现这些库的集成非常简单。

有了这些包，我们可以扩展 Scikit-learn 估算器的功能，我将在本文中向您展示如何使用它们。

## 数据格式

在这一节中，我们将探索可用于处理和转换数据的库。

您可以使用这个包将“DataFrame”列映射到 Scikit-learn 转换。然后，您可以将这些列组合成功能。

要开始使用该软件包，请通过 pip 安装“sklearn-pandas”。“DataFrameMapper”可用于将 pandas 数据框列映射到 Scikit-learn 转换。让我们看看它是怎么做的。

首先，创建一个虚拟数据帧:

```py
data =pd.DataFrame({
    'Name':['Ken','Jeff','John','Mike','Andrew','Ann','Sylvia','Dorothy','Emily','Loyford'],
    'Age':[31,52,56,12,45,50,78,85,46,135],
    'Phone':[52,79,80,75,43,125,74,44,85,45],
    'Uni':['One','Two','Three','One','Two','Three','One','Two','Three','One']
})

```

“DataFrameMapper”接受元组列表——第一项的名称是数据帧中的列名。

第二个传递的项是将应用于该列的转换类型。

例如，' [LabelBinarizer](https://web.archive.org/web/20230204025453/https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html) 可应用于' Uni '列，而' Age '列则使用' [StandardScaler](https://web.archive.org/web/20230204025453/https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) 进行缩放。

```py
from sklearn_pandas import DataFrameMapper
mapper = DataFrameMapper([
     ('Uni', sklearn.preprocessing.LabelBinarizer()),
     (['Age'], sklearn.preprocessing.StandardScaler())
 ])
```

定义映射器后，接下来我们用它来拟合和转换数据。

```py
mapper.fit_transform(data)
```

映射器的“transformed_names_”属性可用于显示转换后的结果名称。

```py
mapper.transformed_names_
```

![scikit-learn extensions](img/391fb770bcc0fc8b21f9dd19510149fc.png)

向映射器传递“df_out=True”将会以熊猫数据帧的形式返回您的结果。

```py
mapper = DataFrameMapper([
     ('Uni', sklearn.preprocessing.LabelBinarizer()),
     (['Age'], sklearn.preprocessing.StandardScaler())

 ],df_out=True)
```

![scikit-learn extensions ](img/70c3b45ccdaaf5ac5b91680f0a871d95.png)

这个包结合了来自 [xarray](https://web.archive.org/web/20230204025453/http://xarray.pydata.org/en/stable/) 的 n 维标签数组和 [Scikit-learn](https://web.archive.org/web/20230204025453/http://scikit-learn.org/stable/) 工具。

您可以将 Scikit-learn 估值器应用于“xarrays ”,而不会丢失它们的标签。您还可以:

*   确保 Sklearn 估算器与 xarray 数据阵列和数据集之间的兼容性，
*   使估计者能够改变样本数，
*   有预处理变压器。

Sklearn-xarray 基本上是 xarray 和 Scikit-learn 之间的桥梁。为了使用其功能，请通过 pip 或“conda”安装“sklearn-xarray”。

这个包有包装器，允许您在 xarray 数据数组和数据集上使用 sklearn 估计器。为了说明这一点，让我们首先创建一个“数据阵列”。

```py
import numpy as np
import xarray as xr
data = np.random.rand(16, 4)
my_xarray = xr.DataArray(data)
```

![scikit-learn extensions ](img/a92a3b0346a3f5494b178adad99bfefc.png)

从 Sklearn 中选择一个转换以应用于此“数据阵列”。在这种情况下，[让我们应用](https://web.archive.org/web/20230204025453/https://phausamann.github.io/sklearn-xarray/content/wrappers.html)“标准缩放器”。

```py
from sklearn.preprocessing import StandardScaler
Xt = wrap(StandardScaler()).fit_transform(X)

```

![scikit-learn extensions ](img/f9a37321c642d4969618707c719213b0.png)

包装估计器可以无缝地用于 Sklearn 管道中。

```py
pipeline = Pipeline([
    ('pca', wrap(PCA(n_components=50), reshapes='feature')),
    ('cls', wrap(LogisticRegression(), reshapes='feature'))
])

```

当安装这个管道时，您只需传入 DataArray。

类似地，DataArrays 可以用于交叉验证的网格搜索。

为此，您需要从“sklearn-xarray”创建一个“CrossValidatorWrapper”实例。

```py
from sklearn_xarray.model_selection import CrossValidatorWrapper
from sklearn.model_selection import GridSearchCV, KFold
cv = CrossValidatorWrapper(KFold())
pipeline = Pipeline([
     ('pca', wrap(PCA(), reshapes='feature')),
     ('cls', wrap(LogisticRegression(), reshapes='feature'))
 ])
gridsearch = GridSearchCV(
     pipeline, cv=cv, param_grid={'pca__n_components': [20, 40, 60]}
)

```

之后，您将使“gridsearch”适合“DataArray”数据类型中的 X 和 y。

## 自动毫升

有没有整合 Sklearn 的工具和库来更好的 Auto-ML？是的，这里有一些例子。

有了这个，你就可以用 Scikit-learn 进行自动化的机器学习了。对于设置，您需要手动安装一些依赖项。

```py
$ curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

```

接下来，通过 pip 安装“auto-sklearn”。

使用该工具时，您不需要担心算法选择和超参数调整。Auto-sklearn 会为您完成所有这些工作。

这要归功于贝叶斯优化、元学习和集成构建方面的最新进展。

要使用它，您需要选择一个分类器或回归器，并使其适合训练集。

```py
from autosklearn.classification import AutoSklearnClassifier
cls = AutoSklearnClassifier()
cls.fit(X_train, y_train)
predictions = cls.predict(X_test)
```

### [Auto _ ViML](https://web.archive.org/web/20230204025453/https://github.com/AutoViML)–自动变量可解释机器学习”(读作“Auto_Vimal”)

给定一个特定的数据集，Auto_ViML 尝试不同的模型和不同的特性。它最终选定了性能最佳的模型。

该软件包还在构建模型时选择尽可能少的特征。这给了你一个不太复杂和可解释的模型。该套件还:

*   通过建议更改缺少的值、格式和添加变量来帮助您清理数据。
*   自动分类变量，无论是文本，数据，还是数字；
*   当 verbose 设置为 1 或 2 时，自动生成模型性能图；
*   允许您使用“功能工具”进行功能工程；
*   当“不平衡标志”设置为“真”时，处理不平衡数据

要查看它的运行，请通过 pip 安装“autoviml”。

```py
from sklearn.model_selection import train_test_split, cross_validate

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=54)

train, test = X_train.join(y_train), X_val.join(y_val)
model, features, train, test = Auto_ViML(train,"target",test,verbose=2)
```

### [TPOT—](https://web.archive.org/web/20230204025453/http://proceedings.mlr.press/v64/olson_tpot_2016.pdf)基于采油树的管道优化工具

这是一个基于 Python 的自动 ml 工具。它使用遗传编程来优化机器学习管道。

它会探索多个管道，以便为您的数据集选择最佳管道。

通过 pip 安装“tpot ”,开始修改它。运行“tpot”后，可以将生成的管道保存在一个文件中。一旦浏览过程完成或您终止该过程，文件将被导出。

下面的代码片段展示了如何在 digits 数据集上创建分类管道。

```py
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_digits_pipeline.py')
```

这是一个自动化特征工程的工具。它的工作原理是将时态和关系数据集转换成特征矩阵。

通过 pip 安装“featuretools[complete]”开始使用它。

深度特征合成(DFS)可用于自动化特征工程。

首先，定义一个包含数据集中所有实体的字典。在“featuretools”中，实体是单个表格。之后，定义不同实体之间的关系。

下一步是将实体、关系列表和目标实体传递给 DFS。这将为您提供特性矩阵和相应的特性定义列表。

```py
import featuretools as ft

entities = {
   "customers" : (customers_df, "customer_id"),
  "sessions" : (sessions_df, "session_id", "session_start"),
   "transactions" : (transactions_df, "transaction_id", "transaction_time")
 }

relationships = [("sessions", "session_id", "transactions", "session_id"),
               ("customers", "customer_id", "sessions", "customer_id")]

feature_matrix, features_defs = ft.dfs(entities=entities,
                                                 relationships = relationships,
                                                  target_entity = "customers")
```

您可以使用 Neuraxle 进行超参数调整和 AutoML。通过 pip 安装“neuraxle”以开始使用它。

除了 Scikit-learn，Neuraxle 还兼容 Keras、TensorFlow 和 PyTorch。它还具有:

*   并行计算和串行化，
*   通过提供这类项目的关键抽象来处理时间序列。

要使用 Neuraxle 进行自动 ml，您需要:

*   定义的管道
*   验证拆分器
*   通过“计分回调”定义计分指标
*   选定的“超参数”存储库
*   选定的“超参数”优化器
*   “自动”循环

点击查看完整的[示例。](https://web.archive.org/web/20230204025453/https://www.neuraxle.org/stable/hyperparameter_tuning.html)

## 实验框架

现在是时候使用一些 SciKit 工具来进行机器学习实验了。

SciKit-Learn Laboratory 是一个命令行工具，可以用来运行机器学习实验。要开始使用它，请通过 pip 安装“skll”。

之后，您需要获得一个“SKLL”格式的数据集。
接下来，为实验创建一个[配置文件](https://web.archive.org/web/20230204025453/https://skll.readthedocs.io/en/latest/run_experiment.html#create-config)，并在终端中运行实验。

```py
$ run_experimen experiment.cfg
```

当实验完成时，多个文件将被存储在[结果](https://web.archive.org/web/20230204025453/https://skll.readthedocs.io/en/latest/run_experiment.html#results)文件夹中。您可以使用这些文件来检查实验。

### 海王星

Neptune 与 Scikit-learn 的集成让你可以使用 Neptune 记录你的实验。例如，您可以记录 Scikit-learn 回归器的摘要。

```py
from neptunecontrib.monitoring.sklearn import log_regressor_summary

log_regressor_summary(rfr, X_train, X_test, y_train, y_test)

```

查看本[笔记本](https://web.archive.org/web/20230204025453/https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/sklearn/docs/Neptune-Scikit-learn.ipynb#scrollTo=GvDSBSrOx-R4)获取完整示例。

型号选择

## 现在让我们换个话题，看看专注于模型选择和优化的 SciKit 库。

这个库实现了基于顺序模型的优化方法。通过 pip 安装“scikit-optimize ”,开始使用这些功能。

Scikit-optimize 可用于通过基于贝叶斯定理的贝叶斯优化来执行超参数调整。

您使用“BayesSearchCV”来获得使用该定理的最佳参数。一个 Scikit-learn 模型作为第一个参数传递给它。

拟合后，您可以通过' best_params_ '属性获得模型的最佳参数。

```py
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
regressor = BayesSearchCV(
    GradientBoostingRegressor(),

     {
         'learning_rate': Real(0.1,0.3),
         'loss': Categorical(['lad','ls','huber','quantile']),
   'max_depth': Integer(3,6),
    },
     n_iter=32,
     random_state=0,
      verbose=1,
      cv=5,n_jobs=-1,
 )
regressor.fit(X_train,y_train)
```

Sklearn-deap 是一个用来实现[进化算法](https://web.archive.org/web/20230204025453/https://en.wikipedia.org/wiki/Evolutionary_algorithm)的包。它减少了为模型寻找最佳参数所需的时间。

它不会尝试每一种可能的组合，而只会改进产生最佳性能的组合。通过 pip 安装“sklearn-deap”。

用于生产的模型出口

```py
from evolutionary_search import EvolutionaryAlgorithmSearchCV
cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                   params=paramgrid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(n_splits=4),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=4)
cv.fit(X, y)
```

## 接下来，让我们来看看 Scikit 工具，您可以使用这些工具来导出您的生产模型。

sklearn-onnx 支持将 sklearn 模型转换为 [ONNX](https://web.archive.org/web/20230204025453/https://onnx.ai/) 。

要使用它，您需要通过 pip 获得‘skl 2 onnx’。一旦你的管道准备好了，你就可以使用‘to _ onnx’函数将模型转换成 ONNX。

这是一个决策树集成的模型编译器。

```py
from skl2onnx import to_onnx
onx = to_onnx(pipeline, X_train[:1].astype(numpy.float32))

```

它处理各种基于树的模型，如随机森林和梯度增强树。

您可以使用它来导入 Scikit-learn 模型。这里，“模型”是一个 scikit-learn 模型对象。

模型检查和可视化

```py
import treelite.sklearn
model = treelite.sklearn.import_model(model)

```

## 在这一节中，让我们看看可用于模型可视化和检查的库。

dtreeviz 用于决策树可视化和模型解释。

eli5 是一个可以用来调试和检查机器学习分类器的包。你也可以用它来解释他们的预测。

```py
from dtreeviz.trees import dtreeviz
viz = dtreeviz(
              model,
               X_train,
               y_train,
               feature_names=boston.feature_names,
               fontname="Arial",
               title_fontsize=16,
               colors = {"title":"red"}
              )

```

例如，Scikit-learn 估计器权重的解释如下所示:

[dabl](https://web.archive.org/web/20230204025453/https://github.com/amueller/dabl)–数据分析基线库

```py
import eli5
eli5.show_weights(model)
```

![scikit-learn extensions ](img/09afe61e5389860cf9dbb08c21455ee7.png)

### dabl 为常见的机器学习任务提供了样板代码。它仍在积极开发中，所以不推荐用于生产系统。

Skorch 是 PyTorch 的 Scikit-learn 包装器。

```py
import dabl
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
sc = dabl.SimpleClassifier().fit(X_train, y_train)
print("Accuracy score", sc.score(X_test, y_test))
```

它允许你在 Scikit-learn 中使用 PyTorch。它支持多种数据类型，如 PyTorch 张量、NumPy 数组和 Python 字典。

最后的想法

```py
from skorch import NeuralNetClassifier
net = NeuralNetClassifier(
    MyModule,
    max_epochs=10,
    lr=0.1,
    iterator_train__shuffle=True,
)
net.fit(X, y)

```

## 在本文中，我们探索了一些扩展 Scikit-learn 生态系统的流行工具和库。

如您所见，这些工具可用于:

处理和转换数据，

*   实现自动化机器学习，
*   执行自动特征选择，
*   运行机器学习实验，
*   为您的问题选择最佳的模型和管道，
*   为生产导出模型…
*   …还有更多！

在您的 Scikit-learn 工作流程中试用这些包，您可能会惊讶于它们有多么方便。

Try out these packages in your Scikit-learn workflow, and you might be surprised how convenient they are.