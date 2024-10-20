# 如何在您的项目中跟踪机器学习模型指标

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/how-to-track-machine-learning-model-metrics>

跟踪机器学习模型的评估指标至关重要，以便:

*   了解您的模型做得如何
*   能够将它与以前的基线和想法进行比较
*   了解你离项目目标有多远

“如果你不衡量它，你就不能改进它。”

但是你应该跟踪什么呢？

我从来没有发现自己在这样的情况下，我认为我已经为我的机器学习实验记录了太多的指标。

此外，在现实世界的项目中，您所关心的指标可能会由于新的发现或不断变化的规范而改变，因此记录更多的指标实际上可以在将来为您节省一些时间和麻烦。

不管怎样，我的建议是:

“记录比你认为需要的更多的指标。”

好吧，但是你具体是怎么做的呢？

## 跟踪单一数字的指标

在许多情况下，您可以为机器学习模型的性能分配一个数值。您可以计算保留验证集的准确度、AUC 或平均精度，并将其用作模型评估指标。

在这种情况下，您应该跟踪每次实验运行的所有这些值。

借助 Neptune，您可以轻松做到这一点:

```py
neptune.log_metric('train_auc', train_auc)
neptune.log_metric('valid_auc', train_auc)
neptune.log_metric('valid_f1', train_auc)
neptune.log_metric('valid_accuracy', train_auc)

```

![](img/c3f798d0d0722b36a787440f47f3a16c.png)

跟踪训练数据集和验证数据集的度量可以帮助您评估模型在生产中表现不佳的风险。差距越小，风险越低。Jean-Fran ois Puget 的《kaggle days》是一个很好的资源。

也就是说，有时候，一个单一的值不足以告诉你你的模型是否运行良好。

这就是性能图表发挥作用的地方。

## 跟踪作为性能图表的指标

要了解您的模型是否有所改进，您可能需要查看图表、混淆矩阵或预测分布。

在我看来，这些仍然是指标，因为它们帮助你衡量你的机器学习模型的性能。

对于 Neptune 测井，这些图表是微不足道的:

```py
neptune.log_image('diagnostics', 'confusion_matrix.png')
neptune.log_image('diagnostics', 'roc_auc.png')
neptune.log_image('diagnostics', 'prediction_dist.png')

```

![](img/c3f798d0d0722b36a787440f47f3a16c.png)

如果您想要使用二进制分类指标，您可以记录:

![](img/c3f798d0d0722b36a787440f47f3a16c.png)

*   所有主要指标，如 f1、f2、brier_loss、准确性等

t

*   所有主要的性能图表，如混淆矩阵、ROC 曲线、精确召回曲线

![](img/c3f798d0d0722b36a787440f47f3a16c.png)

```py
import neptunecontrib.monitoring.metrics as npt_metrics//r//n//r//nnpt_metrics.log_binary_classification_metrics(y_test, y_test_pred)

```

## 跟踪迭代级别的度量(学习曲线)

大多数机器学习模型迭代收敛。深度学习模型、梯度提升树和许多其他模型都是如此。

您可能希望在每次迭代之后跟踪训练集和验证集的评估度量，以查看您的模型是否监控过度拟合。

监控这些学习曲线很容易实现，但却是重要的习惯。

对于简单的基于迭代的训练，它可能是这样的:

```py
for i in range(iterations):

   train_loss = loss(y_pred, y)
   neptune.log_metric('train_loss', train_loss)
```

在大多数深度学习框架中使用的回调系统的情况下:

```py
class NeptuneLoggerCallback(Callback):
    ...
    def on_batch_end(self, batch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(f'batch_{log_name}', log_value)
```

![](img/c3f798d0d0722b36a787440f47f3a16c.png)

Neptune 集成了大多数主要的机器学习框架，您可以毫不费力地跟踪这些指标。在此检查[可用的集成。](https://web.archive.org/web/20221206012745/https://docs.neptune.ai/)

## 在每个时期后跟踪预测

有时，您可能希望在每个时期或迭代后查看一下模型预测。

这在训练需要大量时间收敛的图像模型时尤其有价值。

例如，在图像分割的情况下，您可能希望在每个时期后绘制预测遮罩、真实遮罩和原始图像。

在 Neptune 中你可以使用`.log_image`方法来做到这一点:

```py
for epoch in epochs:
     …
     mask_preds = get_preds(model, images)
     overlayed_preds = overlay( images, masks_true, masks_pred)
     neptune.log_image('network_predictions', overlayed_preds)
```

## 培训完成后跟踪指标

在某些应用程序中，您无法跟踪培训脚本中所有重要的指标。

此外，在现实生活中的机器学习项目中，项目的范围以及您关心的度量标准可能会随着时间的推移而变化。

在这些情况下，您将需要更新实验指标，或者在您的培训工作已经完成时添加新的性能图表。

幸运的是，用海王星更新实验很容易:

```py
exp = project.get_experiments(id='PROJ-421')[0]

exp.log_metric('test_auc'; 0.62)
exp.log_image('test_performance_charts', 'roc_curve_test.png')
```

![](img/c3f798d0d0722b36a787440f47f3a16c.png)

请记住，为一个实验或模型引入新的指标意味着您可能需要重新计算和更新以前的实验。通常情况下，一个模型在一个度量上可能更好，而在另一个度量上可能更差。

最后的想法

## 在本文中，我们了解到:

你应该记录你的机器学习指标

*   如何跟踪单值指标并查看哪些模型表现更好
*   如何跟踪学习曲线以实时监控模型训练
*   如何跟踪性能图表以了解更多信息
*   如何在每个时期后记录图像预测，
*   如果在培训结束后计算评估指标，如何更新实验指标
*   快乐训练！

Happy training!