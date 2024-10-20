# 如何组织你的 LightGBM ML 模型开发过程——最佳实践范例

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/how-to-organize-your-lightgbm-ml-model-development-process-examples-of-best-practices>

[LightGBM](https://web.archive.org/web/20221206214705/https://lightgbm.readthedocs.io/en/latest/) 是一个分布式高效的[梯度推进框架](https://web.archive.org/web/20221206214705/https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)，使用基于树的学习。它以快速训练、准确性和有效利用内存而闻名。它使用逐叶树生长算法，与逐深度生长算法相比，这种算法往往收敛得更快。

LightGBM 很棒，用 LightGBM 构建模型很容易。但是，当您使用不断变化的功能和超参数配置训练模型的许多版本时，很容易迷失在所有这些元数据中。

在 Excel 表格或文本文件中管理这些配置会很快变得一团糟。幸运的是，今天有许多工具和库可以帮助你跟踪这一切。

本文将介绍如何使用最流行的实验和模型管理库 Neptune 来处理各种版本的 ML 模型。我还将向您展示如何通过几个步骤将实验管理添加到您当前的工作流程中。

如何跟踪 LightGBM 模型构建元数据: [Neptune + LightGBM 集成](https://web.archive.org/web/20221206214705/https://docs.neptune.ai/integrations-and-supported-tools/model-training/lightgbm)

## 用 LightGBM 进行 ML 模型开发的现状

### **获取数据集**

任何模型开发过程都将从获取数据集开始。让我们使用 Scikit-learn 来生成一个回归数据集。之后，我们将它分成训练集和测试集。

```py
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100000, n_features=10, n_informative=8,  random_state=101)
import pandas as pd
X = pd.DataFrame(X,columns=["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"])
y = pd.DataFrame(y,columns=["Target"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
```

### **训练 LightGBM 模型**

此时，我们可以开始训练 [LightGBM](/web/20221206214705/https://neptune.ai/blog/lightgbm-parameters-guide) 模型的过程。然而，我们需要解决几个问题:

*   根据`train`方法的要求，将训练集和验证集定义为`lgb.Dataset`格式
*   定义培训参数

```py
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {'boosting_type': 'gbdt',
              'objective': 'regression',
              'num_leaves': 40,
              'learning_rate': 0.1,
              'feature_fraction': 0.9
              }
gbm = lgb.train(params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=['train','valid']],
   )

```

培训之后，我们应该保存模型，以便在部署过程中使用。

```py
gbm.save_model('mode.pkl')

```

我们现在可以运行预测并将它们保存在 CSV 文件中。

```py
import pandas as pd
pd.DataFrame(predictions, columns=["Predictions"]).to_csv("light_predictions.csv")
```

在构建机器学习模型时，需要对上述所有项目(代码、参数、数据版本、度量和预测)进行管理和版本化。您可以使用 git、电子表格、配置、文件系统等来完成这项工作。但是，今天我将向您展示如何使用 Neptune.ai 对所有内容进行版本控制。

[https://web.archive.org/web/20221206214705if_/https://www.youtube.com/embed/w9S5srkfSI4?list=PLKePQLVx9tOd8TEGdG4PAKz0Owqdv1aaw](https://web.archive.org/web/20221206214705if_/https://www.youtube.com/embed/w9S5srkfSI4?list=PLKePQLVx9tOd8TEGdG4PAKz0Owqdv1aaw)

视频

## 在 Neptune 组织 ML 开发

### **安装包并设置 Neptune**

首先，我们需要安装 Neptune 客户端包。

```py
pip install neptune-client

```

使用 Neptune 笔记本，我们可以将笔记本检查点保存到 Neptune。让我们也安装它:

```py
pip install neptune-notebooks

```

为了完成集成，我们需要启用这个扩展:

```py
jupyter nbextension enable --py neptune-notebooks

```

***注**:如果你不用笔记本，可以跳过这一部分。*

现在我们正在安装软件包，让我们也把 Neptune Contrib 软件包拿出来。这个包将使我们能够在训练 LightGBM 模型时将我们的度量记录到 Neptune。

```py
pip install neptune-contrib[monitoring]

```

### **将你的脚本连接到 Neptune**

为了让 Neptune 客户端与 [Neptune AI](/web/20221206214705/https://neptune.ai/login) 进行通信，我们需要设置一个帐户并获取 API 密钥。登录后，单击个人资料图片即可获得 API 密钥。

第一步是创建一个项目。当您登录到您的帐户时，这可以在“项目”选项卡下完成。

之后我们需要初始化我们和 Neptune AI 之间的通信。

第一步是通过点击 Neptune 图标将我们的笔记本连接到 Neptune。

现在将提示您输入 API 令牌。一旦连接成功，你就可以通过点击上传按钮将你的笔记本上传到 Neptune。

之后，我们使用`neptune.init`来初始化我们和 neptune.ai 项目之间的通信。

```py
import neptune
neptune.init(project_qualified_name='mwitiderrick/LightGBM, api_token='YOUR_API_KEY')
```

### **创建实验并保存超参数**

开始登录 Neptune 的第一件事是创建一个实验。这是一个命名空间，您可以在其中记录度量、预测、可视化和任何其他内容([查看您可以在 Neptune](https://web.archive.org/web/20221206214705/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display) 中记录和显示的所有元数据类型的完整列表)。

让我们创建一个实验并记录模型超参数。

```py
neptune.create_experiment('LightGBM',params=params)

```

运行`neptune.create_experiment`输出海王星实验的链接。

可以点开看看训练过程直播。

现在，没有记录太多，但是我们可以在 parameters 部分看到超参数。

“参数”选项卡显示用于训练 LightGBM 模型的参数。

### **创建 Neptune 回调并将其传递给“train”**

为了将训练指标记录到 Neptune，我们使用了来自`neptune-contrib`库的现成回调。这很酷，因为这是我们在训练阶段唯一需要添加的东西。

有了回调设置，海王星照顾其余的。

```py
import lightgbm as lgb
gbm = lgb.train(params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=['train','valid'],
    callbacks=[neptune_monitor()],
   )

```

***注意:**在笔记本上工作时，一旦完成运行实验，确保您的运行`neptune.stop()`完成当前工作(在脚本中实验自动停止)。*

点击 Neptune 上的项目，将显示与该特定项目相关的所有实验。

单击单个实验将显示该特定实验的图表和日志。

日志部分显示了用于生成上述图表的培训和验证指标。

有趣的是，我们可以在模型训练时监控 RAM 和 CPU 的使用情况。这些信息可以在实验的监控部分找到。

当我们看图表时，海王星允许我们放大和缩小不同的地方。这对于更深入地分析模型的训练是重要的。

此外，我们可以选择几个实验并比较它们的性能。

### **版本测试指标**

Neptune 还允许我们记录我们的测试指标。这是使用`neptune.log_metric`功能完成的。

```py
neptune.log_metric('Root Mean Squared Error', np.sqrt(mean_squared_error(y_test, predictions)))
neptune.log_metric('Mean Squarred Error', mean_squared_error(y_test, predictions))
neptune.log_metric('Mean Absolute Error', mean_absolute_error(y_test, predictions))
```

### **版本数据集**

在 Neptune 中对数据集哈希进行版本控制也非常有用。这将使您能够在执行实验时跟踪数据集的不同版本。这可以用 Python 的`hashlib`模块和 Neptune 的`set_property`函数来完成。

```py
import hashlib
neptune.set_property('x_train_version', hashlib.md5(X_train.values).hexdigest())
neptune.set_property('y_train_version', hashlib.md5(y_train.values).hexdigest())
neptune.set_property('x_test_version', hashlib.md5(X_test.values).hexdigest())
neptune.set_property('y_test_version', hashlib.md5(y_test.values).hexdigest())

```

之后，您可以在项目的 details 选项卡下看到版本。

你也可以使用一个数据版本化工具，比如 [DVC](https://web.archive.org/web/20221206214705/https://dvc.org/doc/command-reference/add#example-single-file) 来管理数据集的版本。此后，您可以记录。dcv 文件到 Neptune。

为了做到这一点，你首先要添加文件到 dvc。这是在当前工作目录下的终端上完成的。

```py
$ dvc add data.csv

```

这将创建。dvc 文件，您可以登录到 Neptune。

```py
neptune.log_artifact('data.csv.dvc')

```

### **版本型号二进制**

还可以使用`neptune.log_artifact()`将模型的各种版本保存到 Neptune。

### **修改你认为你还需要的东西**

Neptune 还提供了使用您最喜欢的绘图库记录其他东西的能力，例如模型解释器和交互式图表。

记录[解释者](https://web.archive.org/web/20221206214705/https://modeloriented.github.io/DALEX/)是使用`log_explainer`函数完成的。

```py
from neptunecontrib.api import log_explainer, log_global_explanations
import dalex as dx

expl = dx.Explainer(model, X, y, label="LightGBM")
log_global_explanations(expl, numerical_features=["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"])
log_explainer('explainer.pkl', expl)
```

这样做之后，这个实验的工件部分将会提供这个经过腌制的解释器和图表。

同样重要的是要注意，即使您使用 LightGBM Scikit-learn 包装器，日志也可以工作。你唯一要做的就是在模型的拟合阶段通过 Neptune 回调。请注意，您可以添加评估集以及评估指标。

```py
model.fit(X_test,y_test,eval_set=[(X_train,y_train),(X_test,y_test)],eval_metric=['mean_squared_error','root_mean_squared_error'],callbacks=[neptune_monitor()])
```

## 在仪表板中组织实验

有了 Neptune，您可以灵活地决定想要在仪表板上看到什么。

您可以随意在仪表板中添加或删除列。例如，您可以添加“在笔记本中创建”列，以便立即访问笔记本检查点。

您还可以按列降序或升序筛选仪表板。如果您想删除列，只需单击 x 按钮。

Neptune 还允许您将实验分组到视图中并保存它们。保存的视图可以共享或固定在仪表板上。

## 与您的团队合作进行 ML 实验

海王星实验可以通过邀请你的队友合作来分享。

可以通过首先公开项目来共享项目。一旦公开，你就可以通过链接与任何人自由分享。

![](img/46e6fbc8c192ce13a324dca2298693a0.png)

使用团队计划时，您可以与队友分享您的私人项目。该团队计划对研究、非营利组织和 Kagglers 也是免费的。

人们可以分享他们在 Neptune 上做的任何事情，例如，我可以通过发送一个[链接](https://web.archive.org/web/20221206214705/https://app.neptune.ai/mwitiderrick/LightGBM/experiments?split=bth&dash=leaderboard&viewId=standard-view)来分享我之前做的比较。

## 以编程方式下载模型工件

Neptune 还允许你下载任何实验的文件。这使您能够从 Python 代码中下载单个文件。例如，您可以使用`download_artifact`方法下载单个文件。例如，要下载我们之前上传的模型，我们只需要获取实验对象并使用它来下载模型。该模型存储在我们当前工作目录的模型文件夹中。

```py
project = neptune.init('mwitiderrick/LightGBM',api_token='YOUR_TOKEN')
my_exp = project.get_experiments(id='LIG-8')[0]
experiment.download_artifact("light.pkl","model")
```

当您想要将您的模型转移到生产中时，这很方便。然而，这是另一篇文章的主题。

结论

## 希望这已经向您展示了使用 Neptune 向您的 LightGBM 训练脚本添加[实验跟踪](/web/20221206214705/https://neptune.ai/experiment-tracking)和模型版本化是多么容易。

具体来说，我们讲述了如何:

设置海王星

*   使用 Neptune 回调来记录我们的 LightGBM 培训课程
*   分析和比较海王星的实验
*   海王星上各种项目版本
*   与团队成员协作
*   从海王星下载你的神器
*   希望有了这些信息，LightGBM 模型现在会更清晰，更易于管理。

感谢阅读！

Thanks for reading!