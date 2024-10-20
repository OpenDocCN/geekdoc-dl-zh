# 深入研究 TensorBoard:示例教程

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/tensorboard-tutorial>

有一个常见的商业说法是**你不能改进你没有测量的东西**。机器学习也是如此。有各种工具可以衡量深度学习模型的性能:Neptune AI、MLflow、Weights and Biases、Guild AI，仅举几个例子。在这篇文章中，我们将重点介绍 **TensorFlow 的**开源**可视化工具包 [TensorBoard](https://web.archive.org/web/20221207172226/https://www.tensorflow.org/tensorboard)** 。

该工具使您能够跟踪各种指标，如训练集或验证集的准确性和日志丢失。正如我们将在这篇文章中看到的，TensorBoard 提供了几个我们可以在机器学习实验中使用的工具。这个工具也很容易使用。

以下是我们将在本文中涉及的一些内容:

*   **在 TensorBoard 中可视化图像**
*   在张量板上检查**模型重量和偏差**
*   可视化**模型的架构**
*   将**混淆矩阵**的图像发送到 TensorBoard
*   **剖析**您的应用程序，以便查看其**性能**，以及
*   使用**张量板**与 **Keras** 、 **PyTorch** 和 **XGBoost**

我们开始吧。

## 如何使用 TensorBoard

本节将重点帮助您了解如何在您的机器学习工作流程中使用 TensorBoard。

### **如何安装**张量板****

在开始使用 TensorBoard 之前，您必须通过 pip 或 conda 安装它

```py
pip install tensorboard
conda install -c conda-forge tensorboard
```

### **使用 **TensorBoard** 搭配 Jupyter 笔记本和 Google Colab**

安装 TensorBoard 后，您现在可以将它加载到您的笔记本中。请注意，你可以在 **Jupyter 笔记本**或**谷歌的 Colab** 中使用它。

```py
%load_ext tensorboard
```

一旦完成，你必须设置一个**日志目录**。这是 TensorBoard 存放所有日志的地方。它将从这些日志中读取数据，以显示各种可视化效果。

```py
log_folder = 'logs'
```

如果你想重新加载 **TensorBoard 扩展**，下面的命令将会变魔术——没有双关语。

```py
%reload_ext tensorboard
```

您可能希望清除当前日志，以便可以将新日志写入该文件夹。你可以通过在 **Google Colab** 上运行这个命令来实现

```py
!rm -rf /logs/
```

在 Jupyter 笔记本上

```py
rm -rf logs
```

如果您正在运行多个**实验**，您可能想要存储所有日志，以便您可以比较它们的结果。这可以通过创建带有时间戳的**日志**来实现。为此，请使用下面的命令:

```py
import datetime
log_folder = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
```

### **如何运行 TensorBoard**

运行 Tensorboard 只需要一行代码。在本节中，您将看到如何做到这一点。

现在让我们看一个例子，在这个例子中，您将使用 TensorBoard 来可视化模型指标。为此，您需要构建一个简单的图像分类模型。

```py
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

model = tf.keras.models.Sequential([
   tf.keras.layers.Flatten(input_shape=(28, 28)),
   tf.keras.layers.Dense(512, activation='relu'),
   tf.keras.layers.Dropout(0.2),
   tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='sgd',
   loss='sparse_categorical_crossentropy',
   metrics=['accuracy'])
```

接下来，加载 **TensorBoard 笔记本扩展**并创建一个指向**日志文件夹**的变量。

```py
%load_ext tensorboard
log_folder = 'logs'
```

### **如何使用 TensorBoard 回调**

下一步是在模型的拟合方法中指定 TensorBoard 回调。为了做到这一点，你首先要导入 **TensorBoard 回调**。

该回调负责记录事件，例如**激活直方图、** **度量概要图**、**剖析图**和**训练图可视化图**。

```py
from tensorflow.keras.callbacks import TensorBoard
```

准备就绪后，您现在可以创建 **TensorBoard 回调**并使用 **log_dir** 指定日志目录。TensorBoard 回调还接受其他参数:

*   **histogram_freq** 是计算模型层的激活和权重直方图的频率。将此项设置为 0 意味着将不计算直方图。为了实现这一点，您必须设置**验证数据**或**验证分割**。
*   **write_graph** 指示图形是否将在 TensorBoard 中可视化
*   **write_images** 设置为 true 时，模型权重在 TensorBoard 中显示为图像
*   **update_freq** 决定如何将**损失**和**指标**写入 TensorBoard。如果设置为整数，比如 100，则每 100 批记录一次损耗和指标。当设置为批处理时，损失和指标在每次批处理后设置。当设置为纪元时，它们在每个纪元后被写入
*   **profile_batch** 决定要评测哪些批次。默认情况下，会分析第二批。例如，您也可以设置为从 5 到 10，以分析批次 5 到 10，即 profile_batch='5，10 '。将 profile_batch 设置为 0 将禁用分析。
*   **embeddings_freq** 嵌入层可视化的频率。将此项设置为零意味着嵌入不会被可视化

```py
callbacks = [TensorBoard(log_dir=log_folder,
                         histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
                         profile_batch=2,
                         embeddings_freq=1)]
```

下一项是拟合模型并传入**回调**。

```py
model.fit(X_train, y_train,
          epochs=10,
          validation_split=0.2,
          callbacks=callbacks)
```

### **如何启动冲浪板**

如果您通过 pip 安装了 TensorBoard，您可以通过命令行启动它

```py
tensorboard -- logdir=log
```

在笔记本电脑上，您可以使用以下方式启动它:

```py
%tensorboard -- logdir={log_folder}
```

TensorBoard 也可通过以下网址通过**浏览器**获得

```py
http://localhost:6006
```

### **远程运行 TensorBoard】**

在远程服务器上工作时，可以使用 SSH 隧道将远程服务器的端口转发到本地机器的端口(在本例中是端口 6006)。这看起来是这样的:

```py
ssh -L 6006:127.0.0.1:6006 your_user_name@my_server_ip
```

有了它，你就可以用正常的方式运行 TensorBoard 了。

请记住，您在 tensorboard 命令中指定的端口(默认为 6006)应该与 ssh 隧道中的端口相同。

```py
tensorboard --logdir=/tmp  --port=6006
```

**注意:**如果您使用默认端口 6006，您可以丢弃–port = 6006。您将能够在本地计算机上看到 TensorBoard，但 TensorBoard 实际上是在远程服务器上运行的。

## 张量板仪表板

现在让我们看看 TensorBoard 上的各个选项卡。

### **张量板标量**

**[标量](https://web.archive.org/web/20221207172226/https://www.tensorflow.org/tensorboard/scalars_and_keras)** 选项卡显示了各时期的损耗和指标变化。它可用于跟踪其他标量值，如学习率和训练速度。

### **张量板图像**

这个仪表盘有显示重量的 **[图像](https://web.archive.org/web/20221207172226/https://www.tensorflow.org/tensorboard/image_summaries)** 。调整滑块显示不同时期的权重。

### **张量图**

此选项卡显示模型的层。您可以使用它来检查模型的架构是否符合预期。

### **张量板分布**

“分布”选项卡显示张量的分布。例如，在下面的密集层中，您可以看到每个时期的权重和偏差分布。

### **张量板直方图**

**直方图**显示了张量随时间的分布。例如，查看下面的 dense_1，您可以看到**偏差**在每个时期的分布。

## 使用 TensorBoard 投影仪

您可以使用 **TensorBoard 的投影仪**来可视化任何**矢量表示**，例如文字**嵌入**和[**图像**](https://web.archive.org/web/20221207172226/https://medium.com/@kumon/visualizing-image-feature-vectors-through-tensorboard-b850ce1be7f1) 。

单词嵌入是捕获它们的语义关系的单词的数字表示。投影仪帮助你看到这些图像。你可以在**非活动**下拉列表中找到它。

## 使用 TensorBoard 绘制训练示例

您可以使用 **TensorFlow 图像摘要 API** 来可视化训练图像。这在处理像这样的图像数据时特别有用。

现在，为图像创建一个新的日志目录，如下所示。

```py
logdir = "logs/train_data/"
```

下一步是创建一个**文件写入器**，并将其指向这个目录。

```py
file_writer = tf.summary.create_file_writer(logdir)
```

在本文开始时(在“如何运行 TensorBoard”一节中)，您指定图像形状为 28 x 28。在将图像写入 TensorBoard 之前对其进行整形时，这是非常重要的信息。您还需要将通道指定为 1，因为图像是灰度的。然后，使用 file_write 将图像写入 TensorBoard。

在本例中，索引为 10 到 30 的图像将被写入 TensorBoard。

```py
import numpy as np

with file_writer.as_default():
    images = np.reshape(X_train[10:30], (-1, 28, 28, 1))
    tf.summary.image("20 Digits", images, max_outputs=25, step=0)
```

## 在 TensorBoard 中可视化图像

除了可视化图像张量，您还可以在 TensorBoard 中可视化实际图像。为了说明这一点，您需要使用 Matplotlib 将 MNIST 张量转换为图像。之后，您需要使用' tf.summary.image '在 Tensorboard 中绘制图像。

从清除日志开始，或者您可以使用带有时间戳的日志文件夹。之后，指定日志目录并创建一个“tf.summary.create_file_writer ”,用于将图像写入 TensorBoard

```py
!rm -rf logs 
```

```py
import io
import matplotlib.pyplot as plt

class_names = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine']
logdir = "logs/plots/"
file_writer = tf.summary.create_file_writer(logdir)
```

接下来，创建一个包含图像的网格。在这种情况下，网格将容纳 36 位数字。

```py
def image_grid():
    figure = plt.figure(figsize=(12,8))

    for i in range(36):
        plt.subplot(6, 6, i + 1)
        plt.xlabel(class_names[y_train[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i], cmap=plt.cm.coolwarm)

    return figure

figure = image_grid()
```

现在将这些数字转换成一个单独的图像，在张量板上可视化。

```py
def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    digit = tf.image.decode_png(buf.getvalue(), channels=4)
    digit = tf.expand_dims(digit, 0)

    return digit
```

下一步是使用 writer 和‘plot _ to _ image’在 TensorBoard 上显示图像。

```py
with file_writer.as_default():
    tf.summary.image("MNIST Digits", plot_to_image(figure), step=0)

```

```py
%tensorboard -- logdir logs/plots
```

## 将混淆矩阵记录到张量板上

使用相同的示例，您可以记录所有时期的混淆矩阵。首先，定义一个函数，该函数将返回一个 Matplotlib 图，其中保存着**混淆矩阵**。

```py
import itertools

def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Accent)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure
```

接下来，清除以前的日志，为混淆矩阵定义日志目录，并创建一个写入日志文件夹的 writer 变量。

```py
!rm -rf logs
```

```py
logdir = "logs"
file_writer_cm = tf.summary.create_file_writer(logdir)
```

接下来的步骤是创建一个函数，该函数将根据模型进行预测，并将混淆矩阵记录为图像。

之后，使用“文件写入器 cm”将混淆矩阵写入日志目录。

```py
from tensorflow import keras
from sklearn import metrics

def log_confusion_matrix(epoch, logs):
    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)

    cm = metrics.confusion_matrix(y_test, predictions)
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)

    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)
```

接下来是 TensorBoard 回调和`LambdaCallback`的定义。

`LambdaCallback`将记录每个时期的混淆矩阵。最后使用这两个回调函数来拟合模型。

由于您之前已经拟合了模型，建议您重新启动运行时，并确保只拟合一次模型。

```py
callbacks = [
   TensorBoard(log_dir=log_folder,
               histogram_freq=1,
               write_graph=True,
               write_images=True,
               update_freq='epoch',
               profile_batch=2,
               embeddings_freq=1),
   keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
]

model.fit(X_train, y_train,
          epochs=10,
          validation_split=0.2,
          callbacks=callbacks)
```

现在运行 TensorBoard 并检查**图像**选项卡上的混淆矩阵。

```py
%tensorboard -- logdir logs
```

## 用张量板调整超参数

你可以用 TensorBoard 做的另一件很酷的事情是用它来可视化**参数优化**。以同一个 MNIST 为例，您可以尝试调整模型的超参数(手动或使用自动超参数优化)并在 TensorBoard 中可视化它们。

这是你期望得到的最终结果。仪表板位于**参数**选项卡下。

为此，您必须清除以前的日志并导入 hparams 插件。

```py
!rm -rvf logs
```

```py
logdir = "logs"

from tensorboard.plugins.hparams import api as hp
```

下一步是定义要调整的参数。在这种情况下，密集层中的单位、辍学率和优化器函数将被调整。

```py
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([300, 200,512]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1,0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))

```

接下来，使用 tf.summary.create_file_writer 定义存储日志的文件夹。

```py
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],)
```

这样一来，您需要像以前一样定义模型。唯一的区别是，第一个密集层的神经元数量、辍学率和优化器函数不会被硬编码。

这将在稍后运行实验时使用的函数中完成。

```py
def create_model(hparams):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS],  activation='relu'),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer=hparams[HP_OPTIMIZER],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5)
    loss, accuracy = model.evaluate(X_test, y_test)

    return accuracy
```

您需要创建的下一个函数将使用前面定义的参数运行上面的函数。然后它会记录精确度。

```py
def experiment(experiment_dir, hparams):

    with tf.summary.create_file_writer(experiment_dir).as_default():
        hp.hparams(hparams)
        accuracy = create_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
```

之后，您需要对上面定义的所有参数组合运行该函数。每个实验都将存储在自己的文件夹中。

```py
experiment_no = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                HP_NUM_UNITS: num_units,
                HP_DROPOUT: dropout_rate,
                HP_OPTIMIZER: optimizer,}

            experiment_name = f'Experiment {experiment_no}'
            print(f'Starting Experiment: {experiment_name}')
            print({h.name: hparams[h] for h in hparams})
            experiment('logs/hparam_tuning/' + experiment_name, hparams)
            experiment_no += 1
```

最后，运行 TensorBoard 来查看您在本节开始时看到的可视化效果。

```py
%tensorboard -- logdir logs/hparam_tuning
```

在 **HPARAMS 选项卡上，****表格视图**显示所有的模型运行及其相应的准确性、丢失率和密集层神经元。**平行坐标视图**将每次运行显示为一条穿过每个超参数和精度指标轴的直线。

单击其中一个将显示试验和超参数，如下所示。

**散点图视图**将超参数和指标之间的比较可视化。

## TensorFlow Profiler

您还可以使用 **[分析器](https://web.archive.org/web/20221207172226/https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)** 跟踪 TensorFlow 模型的性能。分析对于理解 TensorFlow 操作的硬件资源消耗至关重要。在此之前，您必须安装 profiler 插件。

```py
pip install -U tensorboard-plugin-profile
```

安装完成后，它将出现在非活动下拉列表中。这是侧写器上众多视觉效果之一的快照。

现在您唯一要做的事情就是定义一个回调，并包含将要分析的批处理。

之后，当你符合模型时，你通过回调。别忘了给 TensorBoard 打电话，这样你就可以看到可视化效果。

```py
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_folder,
                                            profile_batch='10,20')]

model.fit(X_train, y_train,
          epochs=10,
          validation_split=0.2,
          callbacks=callbacks)
```

```py
%tensorboard --logdir=logs
```

### **概览页面**

**档案选项卡**上的**概览页面**显示了该型号性能的高级概览。从下图可以看出，**性能总结**显示了:

*   编译内核所花费的时间，
*   读取数据所花费的时间，
*   启动内核所花费的时间，
*   生产产出所花费的时间，
*   设备上的计算时间，以及
*   主机计算时间

**步进时间图**显示了所有已采样步进的器件步进时间。图表上的不同颜色描述了花费时间的不同类别:

*   **红色**部分对应于器件在等待输入数据时空闲的步进时间。
*   绿色的部分显示设备实际工作的时间。

不过，在概览页面上，您可以看到运行时间最长的 **TensorFlow 操作**。

**运行环境**显示使用的主机数量、**设备类型**、**设备内核数量**等环境信息。在这种情况下，您可以看到在 Colab 的运行时，有一台主机的 GPU 包含一个内核。

从这一页你可以看到的另一件事是**优化**模型性能的建议。

### **跟踪查看器**

**跟踪查看器**可用于了解输入管道中的性能瓶颈。它显示了在**评测**期间 GPU 或 CPU 上发生的不同事件的时间线。

纵轴显示各种事件组，横轴显示事件轨迹。在下图中，我使用了快捷键 **w** 来放大事件。要缩小，使用键盘快捷键 **S** 。 **A** 和 **D** 可分别用于向左和向右移动。

您可以单击单个事件来进一步分析它。使用**浮动工具栏**上的光标或使用键盘快捷键 **1** 。

下图显示了对显示开始和墙壁持续时间的**SparseSoftmaxCrossEntropyWithLogits**事件(一批数据的损失计算)的分析结果。

您还可以通过按住 **Ctrl 键**并选择它们来检查各种事件的摘要。

### **输入管道分析器**

**输入管道分析器**可用于分析模型输入管道中的低效问题。

该功能显示输入流水线分析的**摘要、设备端分析细节**和**主机端分析细节**。

输入管道分析总结显示了**总输入管道**。它是通知应用程序是否被输入绑定以及绑定多少的部分。

**器件侧分析细节**显示器件步进时间和器件等待输入数据的时间。

**主机端分析**显示主机端的分析，如主机上输入处理时间的分解。

在**输入流水线分析器上，**你还可以看到关于单个**输入操作**、花费的**时间**及其**类别**的统计。以下是各列所代表的内容:

*   **输入操作** —输入操作的张量流操作名
*   **Count** —分析期间操作执行的实例数
*   **总时间** —在上述每个实例上花费的累计时间总和
*   **总时间%** —是花费在操作上的总时间占花费在输入处理上的总时间的百分比
*   **总自我时间** —在每个实例上花费的自我时间的累计总和。
*   **总自我时间%** —总自我时间占输入处理总时间的百分比
*   **类别** —输入操作的处理类别

### **张量流统计**

该仪表板显示了在主机上执行的每个 TensorFlow 操作的性能。

*   **第一张** **饼状图**展示了主机上每个操作自执行时间的分布。
*   **第二个**显示主机上每个操作类型的自执行时间分布。
*   第**第三**显示设备上每个操作的自执行时间分布。
*   第四个显示设备上每个操作类型的自执行时间分布。

饼图下方的表格显示了**张量流操作**。每个**行**都是一个操作。**栏**显示了每个操作的各个方面。您可以使用任何列对表进行过滤。

在上表下方，您可以看到按类型分组的各种张量流操作。

### **GPU 内核统计数据**

该页面显示了**性能统计数据**以及每个 GPU 加速内核的原始操作。

内核统计数据下面是一个表格，其中显示了内核和各种操作花费的时间。

### 内存配置文件页面

该页面显示了在分析期间内存的**利用率。它包含以下几个部分:内存配置文件摘要、内存时间线图和内存细分表。**

*   **内存配置文件摘要**显示 TensorFlow 应用程序的内存配置文件摘要。
*   **内存时间线图**显示了内存使用量(以 gib 为单位)和碎片百分比(以毫秒为单位)与时间的关系图。这
*   **内存细分表**显示在性能分析间隔内存使用率最高的点的活动内存分配。

## 如何在 TensorBoard 上启用调试

您也可以将[调试](https://web.archive.org/web/20221207172226/https://www.tensorflow.org/tensorboard/debugger_v2)信息转储到您的 TensorBoard。要做到这一点，你必须启用调试——它仍然处于实验模式

```py
tf.debugging.experimental.enable_dump_debug_info(
   logdir,
   tensor_debug_mode="FULL_HEALTH",
   circular_buffer_size=-1)
```

仪表板可以在**调试器 V2** 的非活动下拉菜单下查看。

调试器 V2 GUI 有**告警**、 **Python 执行时间线**、**图形执行、**和**图形结构**。警报部分显示程序的异常情况。Python 执行时间线部分显示了操作和图形的热切执行的历史。

图形执行显示所有在图形中计算过的浮点型张量的历史。图形结构部分包含源代码和堆栈跟踪，它们是在您与 GUI 交互时填充的。

## 将 TensorBoard 与深度学习框架结合使用

你不局限于单独使用 TensorFlow 的 TensorBoard。您还可以将它与其他框架一起使用，如 Keras、PyTorch 和 XGBoost 等。

### **py torch 中的张量板**

您首先通过[定义一个 writer](https://web.archive.org/web/20221207172226/https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) 来指向您想要写入日志的文件夹。

```py
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='logs')
```

下一步是使用 summary writer 添加您希望在 TensorBoard 上看到的项目。

```py
from torch.utils.tensorboard import SummaryWriter
import numpy as np

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
```

### **喀拉斯的 tensor board**

由于 [TensorFlow 使用 Keras](https://web.archive.org/web/20221207172226/https://keras.io/api/callbacks/tensorboard/) 作为官方高级 API，TensorBoard 的实现类似于它在 TensorFlow 中的实现。我们已经看到了如何做到这一点:

创建回拨:

```py
from tensorflow.keras.callbacks import TensorBoard

tb_callback = TensorBoard(log_dir=log_folder,...)
```

将它传递给“model.fit ”:

```py
model.fit(X_train, y_train,
          epochs=10,
          validation_split=0.2,
          callbacks=[tb_callback])
```

### **XG boost 中的 tensor board**

使用 XGBoost 时，还可以将事件记录到 TensorBoard。为此需要使用 [tensorboardX](https://web.archive.org/web/20221207172226/https://github.com/lanpa/tensorboardX) 包。例如，要记录度量和损失，您可以使用“SummaryWriter”和日志标量。

```py
from tensorboardX import SummaryWriter

def TensorBoardCallback():
    writer = SummaryWriter()

    def callback(env):
        for k, v in env.evaluation_result_list:
            writer.add_scalar(k, v, env.iteration)

    return callback

xgb.train(callbacks=[TensorBoardCallback()])
```

## Tensorboard.dev

Tensorboard.dev 是一个托管的 Tensorboard 平台，可以轻松托管、跟踪和共享 ML 实验。它允许人们发布他们的 TensorBoard 实验，排除故障以及与团队成员合作。一旦你有了一个 TensorBoard 实验，把它上传到 TensorBoard.dev 是非常简单的。

```py
tensorboard dev upload --logdir logs
    --name "(optional) My latest experiment"
    --description "(optional) Simple comparison of several      hyperparameters"
```

一旦你运行这个命令，你会得到一个提示，要求你用谷歌账户授权 TensorBoard.dev。一旦你这样做，你会得到一个验证码，你将进入认证。

这将产生一个独特的张量板。开发链接给你。这里有一个这样的[链接](https://web.archive.org/web/20221207172226/https://tensorboard.dev/experiment/Yf4oVs9bS7mUBZPTV8KcPQ/)的例子。如你所见，这非常类似于在本地主机上查看 TensorBoard，只是现在你是在线查看。

一旦你在这里着陆，你就可以和冲浪板互动，就像你在这个作品的前几部分一样。

需要注意的是，这个 TensorBoard 对互联网上的每个人都是可见的，所以请确保您没有上传任何敏感数据。

## 使用 TensorBoard 的限制

正如你所看到的，TensorBoard 给了你很多很棒的功能。也就是说，使用 TensorBoard 并非一帆风顺。

它有一些限制:

*   难以在需要协作的团队环境中使用
*   没有用户和工作区管理:大型组织通常需要这些功能
*   您不能执行数据和模型版本化来跟踪各种实验
*   无法将其扩展到百万次运行；运行太多次，你会开始遇到 UI 问题
*   用于记录图像的界面有点笨拙
*   您不能记录和可视化其他数据格式，如音频/视频或自定义 html

## 最后的想法

这篇文章中有几件事我们没有涉及到。值得一提的两个有趣特性是:

*   [公平指标](https://web.archive.org/web/20221207172226/https://www.tensorflow.org/tensorboard/fairness-indicators)仪表板(目前处于测试阶段)。它允许计算二进制和多类分类器的公平性度量。
*   [What-If 工具](https://web.archive.org/web/20221207172226/https://www.tensorflow.org/tensorboard/what_if_tool) (WIT)使你能够探索和研究经过训练的机器学习模型。这是使用不需要任何代码的可视化界面来完成的。

希望你在这里学到的一切能帮助你监控和调试你的训练，并最终建立更好的模型！