# 记录和管理 ML 模型构建元数据的最佳工具

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/best-tools-to-log-and-manage-ml-model-building-metadata>

当你在开发机器学习模型时，你绝对需要能够重现实验。如果你得到了一个结果很好的模型，但是你不能复制它，因为你没有记录实验，这将是非常不幸的。

你可以通过记录每一件事来使你的实验具有可重复性。有几个工具可以用来做这件事。在本文中，让我们看看一些最流行的工具，看看如何开始用它们记录实验的元数据。您将了解如何使用这些工具运行一个 [LightGBM 实验](/web/20221206203026/https://neptune.ai/blog/how-to-organize-your-lightgbm-ml-model-development-process-examples-of-best-practices)。

让我们开始吧。

## 海王星

[Neptune](/web/20221206203026/https://neptune.ai/) 是一个可以用来记录和管理 ML 建模元数据的平台。您可以用它来记录:

*   型号版本，
*   数据版本，
*   模型超参数，
*   图表，
*   还有很多。

海王星托管在云上，不需要任何设置，随时随地都可以访问你的实验。你可以在一个地方组织你所有的实验，并与你的团队合作。你可以邀请你的队友来观看和研究任何实验。

要开始使用 Neptune，[你需要安装](https://web.archive.org/web/20221206203026/https://docs.neptune.ai/getting-started/quick-starts/hello-world) `neptune-client`。你还需要在[设立一个项目](https://web.archive.org/web/20221206203026/https://neptune.ai/login)。这是您将从 Neptune 的 Python API 中使用的项目。

Neptune 的一个[新版本刚刚发布。新版本支持更多的工作流，如离线模式、ML 管道和恢复运行。然而，一些集成，如 LightGBM one，仍将被移植到新版本中。因此，在本文中，我使用了一个旧版本。敬请关注](/web/20221206203026/https://neptune.ai/blog/neptune-new)[这一页的新版本](https://web.archive.org/web/20221206203026/https://docs.neptune.ai/essentials/integrations/machine-learning-frameworks/lightgbm)。

下一步是初始化“neptune-client”来处理这个项目。除了项目，你还需要你的 API 密匙。你可以在你的个人资料图片下面得到这个，如下图。

有了这两件事，您现在可以初始化项目了。

```py
import neptune
neptune.init(project_qualified_name='mwitiderrick/LIGHTSAT', api_token='YOUR_TOKEN')
```

下一步是创建一个实验并命名它。这个实验的参数也在这个阶段传入。

```py
params = {'boosting_type': 'gbdt',
              'objective': 'regression',
              'num_leaves': 40,
              'learning_rate': 0.09,
              'feature_fraction': 0.8
              }

exp = neptune.create_experiment(name='LightGBM-training',params=param)
```

现在，您将开始训练 LightGBM 模型。训练时，您将使用 Neptune 的 LightGBM 回调来记录训练过程。你需要安装“Neptune-contrib[监控]”。

接下来，您将导入“neptune_monitor”回调并将其传递给 LightGBM 的“train”方法。

```py
from neptunecontrib.monitoring.lightgbm import neptune_monitor
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

gbm = lgb.train(params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=['train','valid'],
    callbacks=[neptune_monitor()],
   )
```

一旦训练过程结束，您可以回到 Neptune web UI 来查看实验并比较结果。

Neptune 也记录模型度量。例如，让我们看看如何记录平均绝对误差、均方误差和均方根误差。您可以使用“log_metric”功能记录各种指标。

```py
predictions = gbm.predict(X_test)
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
neptune.log_metric('Root Mean Squared Error', np.sqrt(mean_squared_error(y_test, predictions)))
neptune.log_metric('Mean Squarred Error', mean_squared_error(y_test, predictions))
neptune.log_metric('Mean Absolute Error', mean_absolute_error(y_test, predictions))
```

Neptune 还自动记录训练和验证学习曲线。您可以在 web 用户界面的图表菜单下找到这些内容。

记录你训练好的模型是非常重要的。通过这种方式，您可以快速将其投入生产。您可以使用“log_artifact”功能记录所有模型。

```py
gbm.save_model('model.pkl')
neptune.log_artifact('model.pkl')
```

Neptune 允许您访问这个实验并下载模型。使用“get_experiments”方法访问实验，并使用“download_artifact”功能下载模型。这个模型需要工件的名称以及您想要存储它的路径。

```py
project = neptune.init('mwitiderrick/LIGHTSAT',api_token='YOUR_TOKEN')
experiment = project.get_experiments(id='LIGHTSAT-5')[0]
experiment.download_artifact("model.pkl","model")
```

查看这款笔记本，了解 Neptune 的更多功能，并体验完整的 LightGBM 实验。

## MLflow

[MLflow](https://web.archive.org/web/20221206203026/https://mlflow.org/docs/latest/quickstart.html) 是一个开源平台，用于跟踪机器学习模型、记录和管理 ML 模型构建元数据。它还集成了流行的数据科学工具。让我们来看看 LightGBM 集成(这种集成在 MLflow 中仍处于试验阶段)。

使用 MLflow 运行实验时，我们要做的第一件事是启用参数和指标的自动记录。

```py
import mlflow
mlflow.lightgbm.autolog()
```

您也可以使用“log_param”功能手动记录参数。该方法记录当前运行下的参数。如果没有处于活动状态的管路，它将创建一个新管路。

```py
params = {'boosting_type': 'gbdt',
              'objective': regression,
              'num_leaves': 67,
              'learning_rate': 0.01,
              'feature_fraction': 0.8
              }

mlflow.log_param("boosting_type", params["boosting_type"])
mlflow.log_param("objective", params["objective"])
mlflow.log_param("num_leaves", params["num_leaves"])
mlflow.log_param("learning_rate", params["learning_rate"])
mlflow.log_param("feature_fraction", params["feature_fraction"])
```

也可以使用“log_model”功能手动记录经过训练的 LightGBM 模型。启用自动记录时，MFlow 会自动记录此情况。

```py
from mlflow.lightgbm import log_model
log_model(artifact_path='lightgbm-model',lgb_model=gbm)
```

然后，您可以使用该模型对新数据进行预测。加载模型，并使用“预测”功能进行预测。

```py
import mlflow
logged_model = 'file:///Users/derrickmwiti/Downloads/mlruns/0/56cb6b76c6824ec0bc58d4426eb92b91/artifacts/lightgbm-model'

loaded_model = mlflow.pyfunc.load_model(logged_model)

import pandas as pd
loaded_model.predict(pd.DataFrame(data))
```

如果你将模型作为[火花 UDF](https://web.archive.org/web/20221206203026/https://spark.apache.org/docs/latest/sql-ref-functions-udf-scalar.html) 加载，你也可以在[火花数据帧](https://web.archive.org/web/20221206203026/https://neptune.ai/blog/apache-spark-tutorial)上进行预测。

```py
import mlflow
logged_model = 'file:///Users/derrickmwiti/Downloads/mlruns/0/56cb6b76c6824ec0bc58d4426eb92b91/artifacts/lightgbm-model'

loaded_model = mlflow.pyfunc.spark_udf(logged_model)

df.withColumn(loaded_model, 'my_predictions')
```

您可以使用“mlflow.end_run()”函数结束活动的 MLflow 运行。可以随时从网络用户界面查看跑步记录。您可以通过在终端上执行“mflow ui”来启动 web UI。

网络用户界面使得比较不同的跑步变得容易。这将显示不同参数和指标的比较。

您还可以看到不同运行的训练和验证学习曲线的比较。

在 artifacts 部分，您将会找到记录的模型和图表。例如，下面是一次 LightGBM 训练运行的记录特征重要性。

点击查看完整的 [MLflow 示例。](https://web.archive.org/web/20221206203026/https://colab.research.google.com/drive/1LwwJIsQ5Zb9ETngBjqagwhhsm6kufXQF?usp=sharing)

检查 [MLflow 与海王星](/web/20221206203026/https://neptune.ai/vs/mlflow)相比如何。

## 权重和偏差

Weights and Biases 是一个平台，用于[实验跟踪](/web/20221206203026/https://neptune.ai/experiment-tracking)、模型、数据集版本化以及管理 ML 模型构建元数据。要开始使用它，你必须[创建一个账户和一个项目](https://web.archive.org/web/20221206203026/https://docs.wandb.ai/quickstart)。然后，您将在 Python 代码中初始化项目。

现在让我们导入‘wandb’并初始化一个项目。此时，您可以传递将用于 LightGBM 算法的参数。这些将被记录下来，您将在 web 用户界面上看到它们。

```py
import wandb
params = {'boosting_type': 'gbdt',
          'objective': 'regression',
          'num_leaves': 40,
          'learning_rate': 0.1,
          'feature_fraction': 0.9
          }
run = wandb.init(config=params,project='light', entity='mwitiderrick', name='light')
```

下一步是使用来自“wandb”的 LightGBM 回调来可视化和记录模型的训练过程。将“wandb_callback”传递给 LightGBM 的“train”函数。

```py
from wandb.lightgbm import wandb_callback

gbm = lgb.train(params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=['train','valid'],
    callbacks=[wandb_callback()],
   )
```

权重和偏差日志标量，如准确性和回归度量。让我们看看如何记录每次 LightGBM 运行的回归指标。使用“wandb.log”功能。

```py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
predictions = gbm.predict(X_test)
wandb.log({'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, predictions))})
wandb.log({'Mean Squared Error': mean_squared_error(y_test, predictions)})
wandb.log({'Mean Absolute Error': mean_absolute_error(y_test, predictions)})
```

网络用户界面还会自动记录培训和验证学习计划。

您还可以从运行中快速创建报告。

您可以保存训练好的 LightGBM 模型，并在每次运行时将其记录到权重和偏差中。实例化一个空的“工件”实例，然后用它来记录模型。数据集可以类似地被记录。

```py
gbm.save_model('model.pkl')
artifact = wandb.Artifact('model.pkl', type='model')
artifact.add_file('model.pkl')
run.log_artifact(artifact)
```

您将在 web UI 的工件部分看到记录的模型。

您可以使用' wandb.finish()'来结束特定的实验。点击查看完整的[带重量和偏差的 LightGBM 示例。](https://web.archive.org/web/20221206203026/https://colab.research.google.com/drive/16aYcVYEG3yWn3f9hXHtfAz-KkVP_KRO8?usp=sharing)

检查[权重&偏差与海王星](/web/20221206203026/https://neptune.ai/vs/wandb)相比如何。

## 神圣的

[神圣](https://web.archive.org/web/20221206203026/https://github.com/IDSIA/sacred)是一个开源的机器学习实验工具。该工具还可以用于记录和管理 ML 模型构建元数据。使用神圣时，首先需要创建一个实验。如果你在 Jupyter 笔记本上运行实验，你需要通过“interactive=True”。

```py
from sacred import Experiment
ex = Experiment('lightgbm',interactive=True)
```

接下来，使用` @ex.config `装饰器定义实验配置。该配置用于定义和记录算法的参数。

```py
@ex.config
def cfg():
    params = {'boosting_type': 'gbdt',
              'objective': 'regression',
              'num_leaves': 40,
              'learning_rate': 0.01,
              'feature_fraction': 0.9
              }
```

接下来，定义 run 函数。在交互模式下运行时，这个函数必须用` @ex.main `修饰。否则，使用` ex.automain`。这个修饰器负责计算主文件所在的文件名。带有这些装饰器的函数是在运行实验时执行的函数。对于这个实验，在“run”函数中会发生一些事情:

*   LightGBM 模型的训练，
*   保存模型，
*   使用该模型进行预测，
*   使用“log_scalar”方法记录回归度量，
*   使用“add_artifact”函数记录模型。

您还可以使用“添加资源”功能记录资源，如 Python 文件。

```py
import lightgbm as lgb

@ex.main
def run(params):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    gbm = lgb.train(params,
        lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=['train','valid'],
       )
    gbm.save_model('model.pkl')
    predictions = gbm.predict(X_test)

    ex.log_scalar('Root Mean Squared Error', np.sqrt(mean_squared_error(y_test, predictions)))
    ex.log_scalar('Mean Squared Error', mean_squared_error(y_test, predictions))
    ex.log_scalar('Mean Absolute Error', mean_absolute_error(y_test, predictions))
    ex.add_artifact("model.pkl")
    ex.add_resource("main.py")

```

下一步是进行实验。

```py
r = ex.run()
```

不幸的是，Sacred 没有提供可以用来查看实验的网络用户界面。为此，您必须使用外部工具。这就把我们带到了下一个图书馆，Omniboard。

## 总括

Omniboard 是一个基于网络的神圣用户界面。该工具连接到神圣使用的 MongoDB 数据库。然后，它将为每个实验收集的指标和日志可视化。要查看神圣收集的所有信息，您必须创建一个观察者。“MongoObserver”是默认的观察者。它连接 MongoDB 数据库并创建一个包含所有这些信息的集合。

```py
from sacred.observers import MongoObserver
ex.observers.append(MongoObserver())
```

准备就绪后，您可以从终端运行 [Omniboard](https://web.archive.org/web/20221206203026/https://vivekratnavel.github.io/omniboard/#/quick-start) 。

```py
$ omniboard -m localhost:27017:sacred
```

然后，您可以通过 127.0.0.1:9000 访问 web 用户界面。

点击跑步将显示更多相关信息。例如度量标准。

您还可以看到在运行过程中记录的模型。

跑步者配置也在那里。

使用神圣+ Omniboard 的[完整例子在这里](https://web.archive.org/web/20221206203026/https://colab.research.google.com/drive/1hDQ9ypq_Nr6xP_LB5lDbHT1k6KltmXQY?usp=sharing)。你必须在服务器或你的本地机器上运行笔记本，这样你就可以[完成所有需要的设置](https://web.archive.org/web/20221206203026/https://vivekratnavel.github.io/omniboard/#/quick-start)来运行 Omniboard。

查查[神圣+全能和海王星](/web/20221206203026/https://neptune.ai/vs/sacred-omniboard)相比如何。

## 最后的想法

在本文中，我们使用各种实验跟踪工具运行了机器学习实验。您看到了如何:

*   创建运行和实验，
*   日志模型和数据集，
*   捕获所有实验元数据，
*   比较不同的运行，
*   记录模型参数和度量。

希望你学到了新东西。感谢阅读！