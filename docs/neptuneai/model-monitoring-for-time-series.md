# 时间序列的模型监控

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/model-monitoring-for-time-series>

模型监控是 CI/CD 管道的重要组成部分。它确保了一致性，并为部署的应用程序提供了健壮性。深度学习模型的一个主要问题是，它可能在开发阶段表现良好，但在部署时，它可能表现不佳，甚至可能失败。时间序列模型尤其如此，因为数据集中的变化可能非常快。

![ML meme](img/7904b2437c4d147f0e6d44fd02feb271.png)

Model monitoring for time series | [Source](https://web.archive.org/web/20230124065351/https://nimblebox.ai/blog/mlops-maintain-scale) 

在本文中，我们将探讨时间序列预测模型，以实际了解我们如何监控它。这篇文章基于一个案例研究，它将使读者能够理解 ML 监控阶段的不同方面，并且同样地执行能够使 ML 模型性能监控在整个部署中保持一致的操作。

所以让我们开始吧。

## 模型监控过程:定义项目

在这篇文章的开始，我们将定义一个简单的项目或案例研究，在这里我们可以深入了解模型监控过程的细节。对于这个项目，我们将设计一个 ML 模型，它可以**预测**在不同商店出售的商品的单位销售额。我发现厄瓜多尔的大型杂货零售商 Corporación Favorita 在 [Kaggle](https://web.archive.org/web/20230124065351/https://www.kaggle.com/competitions/store-sales-time-series-forecasting) 为我们提供了他们的数据。

这个想法是利用一个准确的预测模型，帮助零售商通过在正确的时间提供正确的产品来取悦他们的客户。使用深度学习模型进行时间序列预测可以帮助零售商对其运营做出更明智的战略决策，并提高其在市场中的竞争力。

### 理解目标

该项目的目标是:

1.  使用 Pytorch-Forecasting 构建深度学习模型，这是一个完全致力于时间序列分析和预测的库
2.  在 [neptune.ai 仪表板](https://web.archive.org/web/20230124065351/https://docs.neptune.ai/tutorials/monitoring_training_live/)中监控训练期间的 ML 模型性能，包括其准确性和损失，例如
    *   对称平均绝对百分比误差
    *   mqf 2 分布损失
    *   分位数损失
3.  监控硬件性能
4.  在部署后监控模型的性能，我们将使用深度检查来监控:

### 描述数据

如前所述，我们将在 [Kaggle](https://web.archive.org/web/20230124065351/https://www.kaggle.com/competitions/store-sales-time-series-forecasting) 中使用 Corporación Favorita 提供的数据。该数据包括日期、商店、产品信息以及关于产品促销的信息。其他功能包括销售数字和补充信息。

[![The data provided by Corporación Favorita in Kaggle](img/7a1c84687c7e47ccf8c1b12cf2c79d91.png)](https://web.archive.org/web/20230124065351/https://i0.wp.com/neptune.ai/wp-content/uploads/2023/01/model-monitoring-for-time-series-2.png?ssl=1)

*Dataset | Source: Author*

数据很复杂，因为它包含不同类别的要素。有目标特征、静态分类特征、时变已知分类特征、时变已知真实特征、时变未知真实特征。

由于数据集中的复杂性，我们必须非常小心地选择适当的模型，在数据集中探索和发现模式和表示。

### 探索模型[理论上]

对于这个项目，我们将使用一个时间融合变压器或 TFT。TFT 是一种专门设计用于处理时序数据(如时间序列或自然语言)的神经网络架构。它结合了通常用于 NLP 任务的 transformer 架构。TFT 最初是在 2020 年的一篇论文中介绍的，该论文描述了一种用于时间序列分析的多时段预测方法，其中一个模型根据过去的数据进行训练，以预测未来。在多时段预测中，模型根据过去的数据进行训练，以预测未来。

![Multi-horizon forecasting ](img/ffffa9207716720cf3a4283c559f0d6e.png)

Multi-horizon forecasting  | [Source](https://web.archive.org/web/20230124065351/https://ai.googleblog.com/2021/12/interpretable-deep-learning-for-time.html)

TFT 的基本构建模块由四个组件组成:

1.  **门控机制:**用于跳过架构中任何未使用的组件。这里使用 gru，它提供有效的信息流，提供自适应的深度和网络复杂性。它还可以适应广泛的数据集和场景。
2.  **变量选择网络:**该模块用于选择每个时间步的相关输入特征。
3.  **静态协变量编码器:**该编码器用于将静态元数据集成到网络中。元数据被编码成上下文向量，并且它被用于调节时间动态。
4.  **时态处理:**该组件负责两种类型的处理:
    *   **时间相关处理**其中 LSTMs 被用于信息的本地处理。
    *   **长期依赖关系**被多头关注块捕获。
5.  **预测区间**:本质上是在给定的地平线上近似变量的可能性。这就是所谓的分位数预测。因此，TFT 不会产生回归模型中常见的特定值，而是在给定时间内提供一系列可能的结果。

也就是说，TFT 是预测该项目的销售的完美模型。数据本身有许多输入特征，一些是静态的，而另一些是时变的。监控这样一个系统可能具有挑战性，因为有许多功能会妨碍模型的性能。

![Illustration of multi-horizon forecasting with many inputs](img/20c841311628a1c7a9bd1b6012cf8bf5.png)

Multi-horizon forecasting with many inputs | [Source](https://web.archive.org/web/20230124065351/https://arxiv.org/pdf/1912.09363.pdf)

### 建立性能基线

模型性能必须在整个部署阶段保持一致。我们衡量模型一致性的方法是选择正确的性能指标，从而建立基线。

作者使用分位数损失来最小化分位数输出的误差。类似地，对于不同的层位长度 H **，可以使用**均方误差**和**平均绝对误差**来计算模型的精度。**

![The formula for mean squared error and mean absolute error](img/8fafdc742dc2913f981c2285677de8e6.png)

*Mean squared error and mean absolute error | [Source](https://web.archive.org/web/20230124065351/https://arxiv.org/pdf/2201.12886.pdf)*

有了度量标准，我们现在必须定义模型的基线性能。基准性能由两种方法定义:

1.  **持久性**:该基线使用前一时间点的变量值作为当前时间点的预测。这对于评估时间序列深度学习模型在预测任务中的性能非常有用，预测任务的目标是根据变量的过去值预测其未来值。

2.  **基准模型**:在某些情况下，将时间序列深度学习模型的性能与被视为该领域“基准”的成熟且广泛使用的模型进行比较可能是合适的。这可以给出模型相对于最先进的方法表现如何的感觉。

当我们一切从头开始时(在我们的情况下)，我们将使用第一种方法。在下一节中，我们将学习如何使用 Pytorch-Forecasting 定义基线模型。一般我们会用某些不同的配置对模型进行优化，这样可以得到不同的模型。然后，我们将选择最佳部署，其准确性将作为现有模型的基准。

部署后，我们将使用当前最佳模型监控模型性能，并检查数据漂移和模型漂移。

## 构建时间序列模型[py torch]

现在，让我们使用 [Pytorch 预测库](https://web.archive.org/web/20230124065351/https://pytorch-forecasting.readthedocs.io/en/stable/index.html)构建一个 TFT 时间序列模型。该库由 Jan Beitner 创建，用于预测时间序列，具有最先进的网络架构，如 TFT、NHiTS、NBeats 等。该库构建在 Pytorch-Lightning 之上，用于在 GPU 和 CPU 上进行可扩展的培训，并用于自动日志记录。

可以查看完整的[笔记本](https://web.archive.org/web/20230124065351/https://colab.research.google.com/drive/1uzWH9_y_9T3rDeUW0f2nNEzuW_gpLSsT?usp=sharing)。在本文中，我将只提供必要的组件。

让我们安装 Pytorch-Forecasting 并导入必要的库。

```py
 ```
!pip install pytorch-forecasting

import torch

import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping

from pytorch_forecasting import Baseline, TimeSeriesDataSet, TemporalFusionTransformer

from pytorch_forecasting.data import GroupNormalizer

from pytorch_forecasting.metrics import QuantileLoss
```py 
```

Pytorch-Forecasting 提供了一个名为“TimeSeriesDataSet”的数据集对象。其本质上根据模型的要求准备数据。该数据集包括以下功能。见下图。

[![The dataset ](img/4cbb3d59aa74952781fe1001a8150c8e.png)](https://web.archive.org/web/20230124065351/https://i0.wp.com/neptune.ai/wp-content/uploads/2023/01/model-monitoring-for-time-series-2.png?ssl=1)

*The dataset* | Source: Author

这些功能必须小心处理，以便模型可以提取和捕获信息。使用时间序列数据集，我们可以为训练目的准备数据集。

```py
 ```
training = TimeSeriesDataSet(
   df_train[lambda x: x.time_idx <= training_cutoff],
   time_idx="time_idx",
   target="sales",
   group_ids=["store_nbr", "family"],
   min_encoder_length=max_encoder_length // 2,  
   max_encoder_length=max_encoder_length,
   min_prediction_length=1,
   max_prediction_length=max_prediction_length,
   static_categoricals=["store_nbr",
                        "family",
                        "city",
                        "state",
                        "store_cluster",
                        "store_type"],
   time_varying_known_categoricals=["holiday_nat",
                                    "holiday_reg",
                                    "holiday_loc",
                                    "month",
                                    "dayofweek",
                                    "dayofyear"],
   time_varying_known_reals=["time_idx", "onpromotion", 'days_from_payday', 'dcoilwtico', "earthquake_effect"
],
   time_varying_unknown_categoricals=[],
   time_varying_unknown_reals=[
       "sales",
       "transactions",
       "average_sales_by_family",
       "average_sales_by_store",
   ],
   target_normalizer=GroupNormalizer(
       groups=["store_nbr", "family"], transformation="softplus"
   ),  
   add_relative_time_idx=True,
   add_target_scales=True,
   add_encoder_length=True,
   allow_missing_timesteps=True

```py 
```

您还可以查看准备好的数据集的参数。

```py
 ```
print(training.get_parameters())

>> {'time_idx': 'time_idx', 'target': 'sales', 'group_ids': ['store_nbr', 'family'], 'weight': None, 'max_encoder_length': 60, 'min_encoder_length': 30, 'min_prediction_idx': 0, 'min_prediction_length': 1, 'max_prediction_length': 16, 'static_categoricals': ['store_nbr', 'family', 'city', 'state', 'store_cluster', 'store_type'], 'static_reals': ['encoder_length', 'sales_center', 'sales_scale'], 'time_varying_known_categoricals': ['holiday_nat', 'holiday_reg', 'holiday_loc', 'month', 'dayofweek', 'dayofyear'], 'time_varying_known_reals': ['time_idx', 'onpromotion', 'days_from_payday', 'dcoilwtico', 'earthquake_effect', 'relative_time_idx'], 'time_varying_unknown_categoricals': [], 'time_varying_unknown_reals': ['sales', 'transactions', 'average_sales_by_family', 'average_sales_by_store'], 'variable_groups': {}, 'constant_fill_strategy': {}, 'allow_missing_timesteps': True, 'lags': {}, 'add_relative_time_idx': True, 'add_target_scales': True, 'add_encoder_length': True, 'target_normalizer': GroupNormalizer(
	method='standard',
	groups=['store_nbr', 'family'],
	center=True,
	scale_by_group=False,
	transformation='softplus',
	method_kwargs={}
), 'categorical_encoders': …}

```py 
```

如您所见，数据集被分成子样本，其中包括静态分类特征、时变已知分类特征、时变已知真实特征和时变未知真实特征。

现在我们建立一个 TFT 模型。

在 Pytorch-Forecasting 中构建模型非常简单，您只需要调用模型对象并根据您的需求配置它，类似于我们看到的数据集。您只需要调用 TemporalFusionTransformer 对象并相应地配置模型。

```py
 ```
tft = TemporalFusionTransformer.from_dataset(
   training,

   learning_rate=0.03,
   hidden_size=16,  

   attention_head_size=1,
   dropout=0.1,  
   hidden_continuous_size=8,  
   output_size=7,  
   loss=QuantileLoss(),

   reduce_on_plateau_patience=4)

```py 
```

## 训练和评估模型

在开始培训之前，让我们首先定义基线。如果你还记得的话，我们将使用持久方法来得到我们的基线分数。在 Pytorch 预测中，您可以调用 Baseline()。predict 函数根据最后一个已知的目标值来预测值。一旦生成了值，就可以计算 MAE 来找出误差差。

```py
 ```
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
print((actuals - baseline_predictions).abs().mean().item())

```py 
```

### ML 模型性能监控:可视化学习曲线

一旦基线值被设置，我们就可以开始我们的训练并监控模型。那么，为什么在培训阶段需要模型监控呢？

模型监控是培训阶段的一个重要方面，原因如下:

1.  **过度拟合**:当模型被训练时，它可能开始太好地拟合训练数据，导致验证数据的表现不佳，这可能导致过度拟合。模型监控允许您在早期检测过度拟合并采取措施防止它，例如**调整**或**提前停止**。

2.  **收敛**:有时候，在训练的时候，模型在一个数值范围内停滞不前。如果模型没有收敛到一个好的解决方案，就会发生这种情况。如果模型没有进展或者陷入次优解决方案，您可以调整模型的架构、学习率或其他超参数，以帮助它收敛到更好的解决方案。

为了监控模型，您可以使用类似于 [neptune](/web/20230124065351/https://neptune.ai/) .ai 的平台。neptune 提供了一个[实时监控仪表板](https://web.archive.org/web/20230124065351/https://docs.neptune.ai/tutorials/monitoring_training_live/)，使我们能够随时随地查看模型的性能。您可以使用下面的代码下载这个包。

```py
 ```
!pip install neptune-client

```py 
```

由于我们使用 Pytorch Lightning，我们可以使用以下代码导入 Neptune 记录器:

```py
 ```
from pytorch_lightning.loggers import NeptuneLogger

```py 
```

现在，让我们通过运行以下 Pytorch-Lightning 脚本来开始培训:

```py
 ```
trainer.fit(tft, train_dataloaders=train_dataloader, 
val_dataloaders=val_dataloader)

```py 
```

[![Examples of Neptune's dashboard ](img/6c0c704da243ef005539595f6f263d75.png)](https://web.archive.org/web/20230124065351/https://i0.wp.com/neptune.ai/wp-content/uploads/2023/01/model-monitoring-for-time-series-6.png?ssl=1)

*Neptune’s dashboard | Source: Author*

从图中可以看出，损失在下降，这意味着模型收敛得很好。

### 监控硬件指标

与监视模型的性能一样，监视硬件性能也很重要。为什么？

在训练 DL 模型期间监视硬件性能可以帮助识别系统中的瓶颈。例如，监控 GPU 内存使用可以确保模型不会耗尽内存而导致训练突然停止。它还可以确保硬件得到高效利用。

[![Example of monitoring hardware metrics](img/d99d81afd8887ca75a969b01f342c0da.png)](https://web.archive.org/web/20230124065351/https://i0.wp.com/neptune.ai/wp-content/uploads/2023/01/model-monitoring-for-time-series-7.png?ssl=1)

*Monitoring hardware metrics in Neptune | Source: Author*

上面的图像显示内存使用是最优的，训练是平滑和高效的。

## 生产中的 ML 模型性能监控

当模型投入生产时，我们必须确保我们必须持续地监控模型的性能，并将其与最近的性能指标进行比较。除此之外，我们还必须持续监控数据。

在本节中，我们将了解如何监控模型的性能、模型漂移和数据漂移。

### 模型漂移:根据新数据和看不见的数据检查模型的准确性

该模型可以在两个数据集上进行测试:没有任何新条目的原始数据集和具有新条目的新数据集。通常，模型会在新数据集上进行测试。但是，如果您在旧数据集上测试模型，并且准确性下降，那么可能有一个有效的理由来重新训练模型，因为模型的参数已经改变。

大多数情况下，在用旧数据集测试模型时，准确性会有一点波动，因此模型可以保持不变。但是，当使用新数据集测试模型时，准确性会显著下降，那么数据的分布就有可能发生了变化。这是您必须检查数据漂移的地方。

给定的代码片段可以帮助您在新数据集或现有数据集上评估模型。

```py
 ```
encoder_data = new_data[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

last_data = new_data[lambda x: x.time_idx == x.time_idx.max()]
decoder_data = pd.concat([last_data.assign(date=lambda x: x.date + 
    pd.offsets.MonthBegin(i)) for i in range(1,  
    max_prediction_length + 1)], ignore_index=True)

decoder_data["time_idx"] = decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
decoder_data["time_idx"] += encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()

decoder_data["month"] = decoder_data.date.dt.month.astype(str).astype("category")  

new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

new_raw_predictions, new_x = best_tft.predict(new_prediction_data, mode="raw", return_x=True)

for idx in range(10):  
   best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False)

```py 
```

![Model drift](img/f08f3b08d21b15d9e9d940b5cb33742c.png)

Source: Author

您还可以添加一些额外的技术来评估性能。例如，您可以评估模型如何对每个要素进行预测。

```py
 ```
predictions, x = best_tft.predict(val_dataloader, return_x=True)
predictions_vs_actuals = 
best_tft.calculate_prediction_actual_by_variable(x, predictions)

best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)

```py 
```

![Model predictions for different features](img/3e67a576d0cd3f382e028517a30e220c.png)

Model predictions for different features | Source: Author

从上面的图片中可以看出，该模型能够准确预测不同的特征。我想鼓励你测试和评估模型的每一个可能的方面。

我要举的另一个例子是检查模型的可解释性。例如:

```py
 ```
interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
best_tft.plot_interpretation(interpretation)

```py 
```

![Interpretability of the model](img/f5bbb20c043ca8be17ee50cf5aa0fa24.png)

Checking the interpretability of the model | Source: Author

可解释性确保人类能够理解深度学习模型做出决定的原因。从上面的图像中，您可以看到销售规模和销售额是模型中的顶级预测因素。

确保前几个预测值在两个数据集中保持相同，即原始数据集和新数据集。

### 检查数据漂移

我们将使用 apparent . ai 监控数据漂移。如果我们遇到任何漂移，我们将看到采取什么必要的步骤。为了检查数据漂移，我们将首先安装并导入所有必要的函数。这里有一个简短的注释:

> 显然，ai 是一种监控工具，它使用户能够评估、测试和监控数据和机器学习模型。它为用户提供了一个交互式的仪表板，所有的结果和报告都在这里生成。？

```py
!pip install evidently
from evidently.dashboard import Dashboard
from evidently.report import Report
from evidently.model_profile import Profile
from evidently.profile_sections import DataDriftProfileSection
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.dashboard.tabs import (
   DataDriftTab,
   DataQualityTab,
   CatTargetDriftTab,
   ClassificationPerformanceTab,
   ProbClassificationPerformanceTab,
) 
```

我将向您展示生成数据漂移报告的两种方法:

## 

*   1 使用报表对象
*   2 使用仪表板对象

#### 使用报告对象

报告对象将指标作为参数之一，并生成关于数据的完整报告。很好用，也挺有效的。

```py
 ```
report = Report(metrics=[DataDriftPreset()])

```py 
```

一旦初始化了对象，就需要有两个样本数据集。其中一个将用作基准的参考数据集，另一个将用作当前数据集。实质上，这两个数据集将用于相互比较统计特性的漂移。

**注**:参考数据集是用于初始训练的原始数据，而当前数据集是新数据集。在现实世界中，我们必须比较这两个数据集。

在本例中，我们将从原始数据集创建两个样本数据集。

```py
 ```
reference = df_train.sample(n=5000, replace=False)
current = df_train.sample(n=5000, replace=False)
```py 
```

一旦创建了两个样本，您就可以在下面的函数中传递它们并查看报告。

```py
 ```
report.run(reference_data=reference, current_data=current)

```py 
```

[![Comparison of the two datasets distribution](img/f2ea1247797033475522ebb937c5d1b3.png)](https://web.archive.org/web/20230124065351/https://i0.wp.com/neptune.ai/wp-content/uploads/2023/01/model-monitoring-for-time-series-11.png?ssl=1)

*Comparison of the two datasets distribution | Source: Author*

生成的仪表板将表示所有特征/列。它将比较两个数据集的分布。每个特性都可以扩展，这将提供分布图和其他相关信息。

[![Report object - the drift summary](img/7286f1d4ab0a5f2d8b543f0846072753.png)](https://web.archive.org/web/20230124065351/https://i0.wp.com/neptune.ai/wp-content/uploads/2023/01/model-monitoring-for-time-series-12.png?ssl=1)

The drift summary | Source: Author

你还会发现漂移总结。

![](img/89f0bc4c4a86574c458fca97b484cf6a.png)

Dataset drift | Source: Author

**免责声明**:在这个数据集中，你不会发现任何漂移，因为数据集没有新的条目。

你可以在这篇由[撰写的](https://web.archive.org/web/20230124065351/https://www.evidentlyai.com/)[文档](https://web.archive.org/web/20230124065351/https://docs.evidentlyai.com/presets/data-drift)中了解更多

#### 使用仪表板和列映射对象

仪表板和列映射对象类似于报告，但它不是自动检查漂移，而是允许您指定类型列来检查数据漂移。这是因为列类型会影响一些测试、度量和可视化。通过指定列的类型，您显然能够产生准确的结果。

以下是使用仪表板和列映射的示例:

```py
 ```
column_mapping = ColumnMapping()
column_mapping.prediction = None  
column_mapping.id = "id" 
column_mapping.datetime="date"
column_mapping.target="sales",

column_mapping.numerical_features=["city",
                                  "dcoilwtico",
                                  "transactions",
                                  "earthquake_effect",
                                  "days_from_payday",
                                  "average_sales_by_family",
                                  "average_sales_by_store",
                                  "onpromotion"]

column_mapping.categorical_features=["store_nbr",
                                     "family",
                                     "city",
                                     "state",
                                     "store_cluster",
                                     "store_type",
                                    "holiday_nat",
                                     "holiday_reg",
                                     "holiday_loc",
                                     "month",
                                     "dayofweek",
                                     "dayofyear"]

column_mapping.task = "regression"

```py 
```

在上面的代码中，您会发现列(我觉得这个任务很有趣)是在一个列表中指定的，该列表由数字、分类、id、日期、时间等组成。这是一项繁琐的任务，但是您可以使用这个 df_train.info()函数来获取列并为特定的类别创建一个列表。

```py
>>> <class>Int64Index: 3000888 entries, 0 to 3000887
Data columns (total 23 columns):
 #   Column                   Dtype         
---  ------                   -----         
 0   id                       int64         
 1   date                     datetime64[ns]
 2   store_nbr                category      
 3   family                   category      
 4   sales                    float64       
 5   onpromotion              int64         
 6   city                     category      
 7   state                    category      
 8   store_type               category      
 9   store_cluster            category      
 10  holiday_nat              category      
 11  holiday_reg              category      
 12  holiday_loc              category      
 13  dcoilwtico               float64       
 14  transactions             float64       
 15  earthquake_effect        float64       
 16  days_from_payday         int64         
 17  average_sales_by_family  float64       
 18  average_sales_by_store   float64       
 19  dayofweek                category      
 20  month                    category      
 21  dayofyear                category      
 22  time_idx                 int64</class> 
```

一旦初始化了 ColumnMapping，就可以将它传递给下面的函数。

```py
 ```
datadrift_dashboard = Dashboard(tabs=[DataDriftTab(verbose_level=1)])
datadrift_dashboard.calculate(reference, current, column_mapping=column_mapping)
datadrift_dashboard.show()
```py 
```

[![Detection of the drift ](img/07691c0b60f3c458795931e00bbca248.png)](https://web.archive.org/web/20230124065351/https://i0.wp.com/neptune.ai/wp-content/uploads/2023/01/model-monitoring-for-time-series-14.png?ssl=1)

Detection of the drift | Source: Author

## 模型监控之后的下一步是什么？

既然我们已经了解了如何监控我们的模型，这里有一些提示可以帮助您采取后续步骤:

1.  定期监控模型的性能，在添加新技术的同时使用相同的方法。例如，您可以使用新的度量或比较来评估损失，如对称平均绝对百分比误差(SMAPE)，并将其与分位数损失进行比较。SMAPE 描绘了模型有预测问题的区域。它衡量所有预测范围内预测值和实际值之间的平均百分比差异。

下面是一个实现 SMAPE 的示例:

```py
 ```
from pytorch_forecasting.metrics import SMAPE

predictions = best_tft.predict(val_dataloader)
mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
indices = mean_losses.argsort(descending=True)  
for idx in range(10):  
   best_tft.plot_prediction(
       x, raw_predictions, idx=indices[idx], add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles)
   )

```py 
```

![Example of SMAPE implementation ](img/99f2581869ea3d4084e331ac9f9e077f.png)

Source: Author

如您所见，该模型可以很好地处理分位数损失，但不能处理 SMAPE。

2.  跟踪数据统计特征的任何变化。这将有助于您尽早发现数据漂移。
3.  使用诸如**特征工程**和**数据扩充**之类的技术来提高你的模型的健壮性。这可以帮助您的模型更好地处理与定型数据具有不同统计特征的数据。
4.  在新的数据集和最新数据上重新训练您的模型，这些数据与您期望在测试时看到的数据具有相似的统计特征。这可以帮助您的模型即使在存在数据漂移的情况下也能保持良好的性能。
5.  使用迁移学习或微调等技术来使预训练模型适应新数据集，而不是从头开始训练它，因为这样可以节省时间，并且可以更快、更有效。
6.  **使用在线学习算法**:另一种解决方案是使用在线学习算法，这种算法能够适应数据分布随时间的变化。这可以通过不断向模型输入新数据并定期重新训练来实现。
7.  **集成学习**:另一种解决方案是使用集成学习，它涉及训练多个模型并结合它们的预测来做出最终预测。这有助于减轻模型漂移的影响，因为集合的整体性能对任何单个模型的性能不太敏感。
8.  **使用领域知识**:另一个解决方案是使用领域知识来识别模型漂移的最可能来源，并相应地设计模型或训练过程。例如，如果您知道数据的某些特征可能会随着时间的推移而改变，则可以在模型中降低这些特征的权重，以减少模型漂移的影响。
9.  注意季节性趋势，并快速适应和重新训练模型。
10.  **监控和警报系统**:最后，另一个解决方案是建立监控和警报系统来检测模型漂移何时发生，以便您可以在它成为问题之前采取措施解决它。

请记住，数据漂移和模型漂移是机器学习中的常见问题，解决它们是一个持续的过程，需要定期监控和维护您的模型。

### 重新训练模型——是还是不是？

重新训练模型是必须的。但是时机很重要。考虑再培训时，请记住以下几点:

## 

*   如果数据分布每周频繁变化，您必须每周对模型进行微调。
*   2 如果您正在处理一项数据随季节变化且添加了新功能的任务，那么请遵循一个时间表，即每月三次及时微调模型，并在新数据集上从头开始重新训练一个新模型。
*   基于以上两点，迁移学习在许多方面也有帮助。这包括使用预先训练的模型作为起点，然后通过冻结原始模型中的一些层并仅训练层的子集，在新的任务上训练它。如果数据集很小，并且希望防止模型忘记在原始任务中学习到的信息，这可能是一个不错的选择。

### 更新数据管道

更新培训渠道时，遵循一些最佳实践会有助于确保平稳过渡，并最大限度地减少错误和部署延迟。以下是一些你可以考虑的策略:

1.  **提前计划**:在上一节课中，我提到数据集的分布可能会偶尔、频繁和季节性地发生变化。在对您的培训渠道进行任何更改之前，提前计划并考虑您正在进行的更改的影响是非常重要的。这可能包括评估对模型性能的潜在影响，确定进行更改所需的资源，以及估计实现更新所需的时间。
2.  利用领域知识:当在一个特定的领域工作时，你会知道什么时候需要什么时候不需要。组织和分离将在特定季节使用的数据集格式。
3.  **测试增量变化**:建立在上述基础上。与其一次完成所有的更改，不如在继续之前对它们进行增量测试，并验证它们是否按预期工作。这有助于尽早发现任何问题，并在必要时更容易回滚更改。
4.  **使用版本控制**:使用版本控制来跟踪你的培训管道的变化是一个好主意。*这使得在必要时回滚更改变得更加容易*，并且它还可以提供一个随着时间的推移所做的修改的记录。版本控制也将帮助您找到问题的答案。就像他们说的“历史重演？，所以版本控制是个好主意。
5.  **记录您的修改**:确保记录您对培训渠道所做的任何更改，包括**更改背后的原因**以及预期的影响。这有助于确保将来可能需要使用管道的其他人了解您的更新。
6.  **监控绩效**:在对您的培训渠道进行更改后，确保监控您的模型的绩效，以确保它不会受到更新的负面影响。这可能包括跟踪诸如准确性和损失之类的度量，并在做出更改之前将它们与模型的性能进行比较。

## 承认

特别感谢[路易斯·布兰奇、](https://web.archive.org/web/20230124065351/https://github.com/LuisBlanche)T2、卡迪普·辛格、[简·贝特纳](https://web.archive.org/web/20230124065351/https://www.linkedin.com/in/janbeitner/)。这篇文章的代码就是受他们的启发，没有他们，这篇文章就不可能完成。

### 参考

1.  型号:
2.  指标:
3.  图书馆:
4.  实验: