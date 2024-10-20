# Keras Metrics:您需要知道的一切

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/keras-metrics>

Keras 指标是用于评估深度学习模型性能的函数。为您的问题选择一个好的度量标准通常是一项困难的任务。

*   你需要了解【tf.keras 和 tf.keras 中哪些指标已经可用以及如何使用它们，
*   在许多情况下，您需要**定义您自己的定制指标**，因为您正在寻找的指标没有随 Keras 一起提供。
*   有时，您希望**通过在每个时期后查看 ROC 曲线或混淆矩阵**等图表来监控模型性能。

### 本文将解释一些术语:

*   keras 度量准确性
*   keras 编译指标
*   keras 自定义指标
*   回归的 keras 度量
*   keras 混淆矩阵
*   tf.kerac.metrics.meaniou
*   tf.keras.metrics f1 分数
*   tf.keras.metrics.auc

## Keras metrics 101

在 Keras 中，度量在编译阶段传递，如下所示。您可以通过逗号分隔来传递多个指标。

```py
from keras import metrics

model.compile(loss='mean_squared_error', optimizer='sgd',
              metrics=[metrics.mae,
                       metrics.categorical_accuracy])
```

您应该如何选择这些评估指标？

其中一些在 Keras 中可用，另一些在 tf.keras 中可用。

让我们回顾一下所有这些情况。

## Keras 中提供了哪些指标？

Keras 提供了丰富的内置指标。根据你的问题，你会使用不同的。

让我们来看看你可能正在解决的一些问题。

### 二元分类

二元分类度量用于只涉及两个类别的计算。一个很好的例子是建立深度学习**模型来预测猫和狗**。我们有两个类别要预测，阈值决定了它们之间的分离点。 ***binary_accuracy*** 和 ***accuracy*** 就是 Keras 中的两个这样的函数。

***binary_accuracy** ，*例如，计算二进制分类问题的所有预测的平均准确率。

```py
keras.metrics.binary_accuracy(y_true, y_pred, threshold=0.5)

```

***准确性*** 指标计算所有预测的准确率。 *y_true* 代表真实标签，而 *y_pred* 代表预测标签。

```py
keras.metrics.accuracy(y_true, y_pred)

```

***confusion_matrix*** 显示一个表格，显示真阳性、真阴性、假阳性和假阴性。

```py
keras.metrics.confusion_matrix(y_test, y_pred)

```

在上面的混淆矩阵中，模型做出了 3305 + 375 个正确的预测，106 + 714 个错误的预测。

你也可以把它想象成一个 matplotlib 图表，我们稍后会讲到。

### 什么是 Keras 精度？

这似乎很简单，但实际上并不明显。

> 术语“准确度”是一个表达式，让[训练文件](https://web.archive.org/web/20220926093651/https://github.com/keras-team/keras/blob/d8b226f26b35348d934edb1213061993e7e5a1fa/keras/engine/training.py#L651)决定应该使用哪个度量标准(**二进制准确度**、**分类准确度**或**稀疏分类准确度**)。该决定基于某些参数，如输出形状(由该层产生的张量的形状，并且将是下一层的输入)和损失函数。

因此，有时质疑甚至是最简单的事情也是好的，尤其是当您的度量发生了意想不到的事情时。

### 多类分类

这些度量用于涉及两个以上类别的分类**问题。扩展我们的动物分类例子，你可以有三种动物，猫、狗和熊。因为我们要对两种以上的动物进行分类，所以这是一个多类分类问题。**

*y_true* 的形状是条目数乘以 1，即(n，1 ),但是 *y_pred* 的形状是条目数乘以类数(n，c)

***category _ accuracy***指标计算所有预测的平均准确率。

```py
keras.metrics.categorical_accuracy(y_true, y_pred)

```

***sparse _ categorial _ accuracy***与*categorial _ accuracy*类似，但大多在对稀疏目标进行预测时使用**。一个很好的例子是在深度学习问题中处理文本，如 word2vec。在这种情况下，一个人用**数千个类**工作，目的是预测下一个单词。这个任务会产生一种情况，y_true 是一个几乎全是零的巨大矩阵，这是使用稀疏矩阵的最佳位置。**

```py
keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

```

***top _ k _ category _ accuracy***计算 top-k 分类准确率。我们从我们的模型中取出前 k 个预测类，并查看正确的类是否被选为前 k 个。如果是，我们说我们的模型是正确的。

```py
keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

```

### 回归

回归问题中使用的度量包括**均方误差、平均绝对误差和平均绝对百分比误差。**这些指标用于预测房屋销售和价格等数值。查看这个参考资料，获得关于回归度量的[完整指南。](https://web.archive.org/web/20220926093651/https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

```py
from keras import metrics

model.compile(loss='mse', optimizer='adam', 
              metrics=[metrics.mean_squared_error, 
                       metrics.mean_absolute_error, 
                       metrics.mean_absolute_percentage_error])
                       metrics.categorical_accuracy])
```

## 如何在 Keras 中创建自定义指标？

正如我们前面提到的，Keras 还允许您定义自己的定制指标。

您定义的函数**必须将 *y_true* 和 *y_pred* 作为参数，并且必须返回单个张量值**。这些对象是具有 float32 数据类型的张量类型。对象的形状是行数乘以 1。例如，如果有 4，500 个条目，形状将是(4500，1)。

可以在深度学习模型的编译阶段传递函数来使用。

```py
model.compile(...metrics=[your_custom_metric])
```

### 如何计算 Keras 中的 F1 分数(精度，召回作为加分)？

让我们看看如何在 Keras 中计算 **f1 分数、精确度和召回率。**我们将为多类场景创建它，但您也可以将其用于二进制分类。

f1 分数是精确度和召回率的加权平均值。因此，为了计算 f1，我们需要首先创建计算精度和召回率的函数。请注意，在多类场景中，您需要查看所有的类，而不仅仅是正类(这是二进制分类的情况)

```py
def recall(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

```

下一步是在我们深度学习模型的编译阶段使用这些函数。我们还添加了默认可用的 Keras *准确性*指标。

```py
model.compile(...,metrics=['accuracy', f1_score, precision, recall])

```

现在让我们根据训练集和测试集来调整模型。

```py
model.fit(x_train, y_train, epochs=5)

```

现在您可以评估您的模型，并访问您刚刚创建的指标。

```py
(loss, 
accuracy, 
f1_score, precision, recall) = model.evaluate(x_test, y_test, verbose=1)

```

很好，您现在知道如何在 keras 中创建定制指标了。

也就是说，有时你可以使用已经存在的东西，只是在一个不同的库中，比如 tf.keras🙂

## tf.keras 中有哪些指标？

最近 **Keras 已经成为 TensorFlow** 中的标准 API，有许多有用的指标可供您使用。

让我们来看看其中的一些。与在 Keras 中只使用 *keras.metrics* 函数调用指标不同，在 tf.keras 中，您必须实例化一个*指标*类。

例如:

```py
tf.keras.metrics.Accuracy() 

```

keras 指标和 tf.keras 之间有很多重叠。但是，有一些指标你只能在 tf.keras 中找到。

让我们看看那些。

### tf.keras 分类指标

***TF . keras . metrics . AUC***通过[黎曼和](https://web.archive.org/web/20220926093651/https://www.khanacademy.org/math/ap-calculus-ab/ab-integration-new/ab-6-2/a/left-and-right-riemann-sums)计算 ROC 曲线的近似 AUC(曲线下面积)。

```py
model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.AUC()])

```

您可以使用 precision，回想一下我们以前在 tf.keras 中实现的开箱即用。

```py
model.compile('sgd', loss='mse', 
               metrics=[tf.keras.metrics.Precision(), 
                        tf.keras.metrics.Recall()])

```

### tf.keras 细分指标

***TF . keras . metrics . meaniou***–*Mean Intersection-Over-Union*是用于评估语义图像分割模型的度量。我们首先计算每个类的欠条:

所有班级的平均值。

```py
model.compile(... metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])

```

### tf.keras 回归度量

就像 Keras 一样，tf.keras 也有类似的回归度量。我们不会过多讨论它们，但有一个有趣的指标值得强调，叫做 ***表示相对误差*** 。

***表示相对误差*** 取一次观测的绝对误差，除以常数。这个常数，**归一化因子**，可以对所有观测值相同，也可以对每个样本不同。

因此，平均相对误差是相对误差的平均值。

```py
tf.keras.metrics.MeanRelativeError(normalizer=[1, 3, 2, 3])
```

## 如何在 tf.keras 中创建自定义指标？

在 *tf.keras* 中，您可以通过扩展 *keras.metrics.Metric* 类来创建一个自定义指标。

为此，您必须覆盖 update_state、result 和 reset_state 函数:

*   ***【update _ state()***对状态变量进行所有更新并计算度量，
*   ***result()*** 从状态变量中返回度量值，
*   ***reset_state()*** 将每个时期开始时的度量值设置为预定义的常数(通常为 0)

```py
class MulticlassTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='multiclass_true_positives', **kwargs):
        super(MulticlassTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):

        self.true_positives.assign(0.)
```

然后我们简单地在编译阶段传递它:

```py
model.compile(...,metrics=[MulticlassTruePositives()])

```

## 性能图表:Keras 中的 ROC 曲线和混淆矩阵

**有时，性能不能用一个数字**来表示，而是用性能图表来表示。这种图表的例子有 ROC 曲线或混淆矩阵。在这些情况下，您可能希望将这些图表记录在某个地方，以便进一步检查。

为了做到这一点，你需要创建一个回调函数，它将在每个时期结束时跟踪你的模型的性能。然后，你可以在一个文件夹中或者在[实验跟踪工具](https://web.archive.org/web/20220926093651/https://neptune.ai/blog/best-ml-experiment-tracking-tools)中查看改进。让我们开始吧。

首先，我们需要一个回调，在每个时期结束时创建 ROC 曲线和混淆矩阵。

```py
import os

from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from scikitplot.metrics import plot_confusion_matrix, plot_roc

class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data

        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]             
        y_pred_class = np.argmax(y_pred, axis=1)

        fig, ax = plt.subplots(figsize=(16,12))
        plot_confusion_matrix(y_true, y_pred_class, ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))

        fig, ax = plt.subplots(figsize=(16,12))
        plot_roc(y_true, y_pred, ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))

```

现在我们简单地将它传递给 *model.fit()* callbacks 参数。

```py
performance_cbk = PerformanceVisualizationCallback(
                      model=model,
                      validation_data=validation_data,
                      image_dir='performance_vizualizations')

history = model.fit(x=x_train,
                    y=y_train,
                    epochs=5,
                    validation_data=validation_data,
                    callbacks=[performance_cbk])
```

如果你愿意，你可以有多个回调。

现在，您将能够在模型训练时看到这些可视化效果:

你也可以像 Neptune 一样把一切记录到实验跟踪工具中。这种方法将让您在一个地方拥有所有的模型元数据。为此，您可以使用 [Neptune + TensorFlow / Keras 集成](https://web.archive.org/web/20220926093651/https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras):

```py
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
import neptune.new as neptune
from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

run = neptune.init(
        project='YOUR_WORKSAPCE/YOUR_PROJECT_NAME',
        api_token='YOUR_API_TOKEN')

```

请注意，您不需要为图像创建文件夹，因为图表将直接发送到您的工具。另一方面，你必须[创建一个项目](https://web.archive.org/web/20220926093651/https://docs.neptune.ai/api-reference/management#.create_project)来开始跟踪你的跑步。

一旦你做到了这一点，一切照常。

```py
neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

history = model.fit(x=x_train,
                    y=y_train,
                    epochs=5,
                    validation_data=validation_data,
                    callbacks=[neptune_cbk])

fig, ax = plt.subplots(figsize=(16, 12))
plot_confusion_matrix(y_train, y_train_pred, ax=ax)

run['confusion_matrix'].upload(neptune.types.File.as_image(fig))

```

您可以在[应用](https://web.archive.org/web/20220926093651/https://app.neptune.ai/common/tf-keras-integration/e/TFK-18/charts)中探索指标和性能图表。

## 如何绘制 Keras 历史对象？

每当调用 *fit()* 时，它都返回一个 ***历史*** 对象，该对象可用于可视化训练历史。**它包含一个字典，其中包含为训练和验证数据集计算的每个历元的损失和度量值**。

例如，让我们提取“*准确性*”指标，并使用 matplotlib 绘制它。

```py
import matplotlib.pyplot as plt

history = model.fit(x_train, y_train, 
                    validation_split=0.25, 
                    epochs=50, batch_size=16, verbose=1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_‘accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

## Keras 指标示例

好了，你已经走了很长一段路，学到了很多。为了唤起你的记忆，让我们把它们放在一个例子中。

我们将从 mnist 数据集开始，并创建一个简单的 CNN 模型:

```py
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
validation_data = x_test, y_test

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

```

我们将在 keras 中创建一个自定义指标、多类别 **f1 分数:**

```py
def recall(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

```

我们将创建一个定制的 tf.keras 度量:**multiclasstruepoints**确切地说:

```py
class MulticlassTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='multiclass_true_positives', **kwargs):
        super(MulticlassTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):

        self.true_positives.assign(0.)
```

我们将**用我们的指标编译 keras 模型**:

```py
import keras

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy',
                       keras.metrics.categorical_accuracy,
                       f1_score, 
                       recall_score, 
                       precision_score,
                       tf.keras.metrics.TopKCategoricalAccuracy(k=5),
                       MulticlassTruePositives()])
```

我们将实现 keras **回调，它将 ROC 曲线和混淆矩阵**绘制到一个文件夹中:

```py
import os

from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from scikitplot.metrics import plot_confusion_matrix, plot_roc

class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data

        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]             
        y_pred_class = np.argmax(y_pred, axis=1)

        fig, ax = plt.subplots(figsize=(16,12))
        plot_confusion_matrix(y_true, y_pred_class, ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))

        fig, ax = plt.subplots(figsize=(16,12))
        plot_roc(y_true, y_pred, ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))

performance_viz_cbk = PerformanceVisualizationCallback(
                                       model=model,
                                       validation_data=validation_data,
                                       image_dir='perorfmance_charts')
```

我们将**进行培训**并监控表现:

```py
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=5,
                    validation_data=validation_data,
                    callbacks=[performance_viz_cbk])
```

我们将**可视化来自 keras 历史对象的指标:**

```py
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

```

我们将在 TensorBoard 或 Neptune 等工具中监控和探索您的实验。

## 海王星

```py
run = neptune.init(
        project='YOUR_WORKSAPCE/YOUR_PROJECT_NAME',
        api_token='YOUR_API_TOKEN')

neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
history = model.fit(..., callbacks=[neptune_cbk])

```

如果您感兴趣，请查看此[文档](https://web.archive.org/web/20220926093651/https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras)和示例[实验运行](https://web.archive.org/web/20220926093651/https://app.neptune.ai/o/common/org/tf-keras-integration/e/TFK-35541/dashboard/metrics-b11ccc73-9ac7-4126-be1a-cf9a3a4f9b74):

👉了解更多关于 Neptune 与 Keras 的整合。

## 张量板

您只需要**添加另一个回调或者修改您之前已经**创建的回调:

```py
from  tf.keras.callbacks import TensorBoard

tensorboard_cbk = TensorBoard(log_dir="logs/training-example/")

history = model.fit(..., callbacks=[performance_viz_cbk, 
                                    tensorboard_cbk])
```

使用 TensorBoard，您需要启动本地服务器并在浏览器中浏览您的跑步记录。

```py
tensorboard --logdir logs/training-example/
```

## 最后的想法

希望这篇文章能给你一些 keras 中模型评估技术的背景知识。

我们涵盖了:

*   keras 和 tf.keras 中的内置方法，
*   实现您自己的定制指标，
*   如何在模型训练时可视化自定义性能图表。

欲了解更多信息，请查看 Keras 知识库和张量流度量文档。

快乐训练！

### 德里克·姆维蒂

Derrick Mwiti 是一名数据科学家，他对分享知识充满热情。他是数据科学社区的热心贡献者，例如 Heartbeat、Towards Data Science、Datacamp、Neptune AI、KDnuggets 等博客。他的内容在网上被浏览了超过一百万次。德里克也是一名作家和在线教师。他还培训各种机构并与之合作，以实施数据科学解决方案并提升其员工的技能。你可能想看看他在 Python 课程中完整的数据科学和机器学习训练营。

* * *

**阅读下一篇**

## 如何在您的项目中跟踪机器学习模型指标

3 分钟阅读| Jakub Czakon |发布于 2020 年 6 月 22 日

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

### 跟踪单一数字的指标

在许多情况下，您可以为机器学习模型的性能分配一个数值。您可以计算保留验证集的准确度、AUC 或平均精度，并将其用作模型评估指标。

在这种情况下，您应该跟踪每次实验运行的所有这些值。

[Continue reading ->](/web/20220926093651/https://neptune.ai/blog/how-to-track-machine-learning-model-metrics)

* * *