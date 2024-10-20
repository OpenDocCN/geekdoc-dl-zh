# 可视化机器学习模型:指南和工具

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/visualizing-machine-learning-models>

为什么我们需要可视化机器学习模型？

> 如果你拒绝将决策权交给你并不完全了解其流程的人，那为什么还要雇人来工作呢？没有人知道人类的大脑(拥有一千亿个神经元！)做决定。”–[*卡西科兹尔科夫*](https://web.archive.org/web/20221201170702/https://medium.com/hackernoon/explainable-ai-wont-deliver-here-s-why-6738f54216be)

这句话被一些人用来批评最近对可解释人工智能的推动。起初，这听起来像是一个有效的观点，对吗？但是它没有考虑到我们不想复制人类的大脑。我们想建造更好的东西。

机器学习模型正在用万亿字节的数据进行训练，目标是在提高效率的同时做出正确的决策，这一点人类做得相当好。

我们赋予 ML 模型的责任意味着我们需要使它们尽可能透明，否则，我们无法信任它们。

为此，我们需要可视化 ML 模型。为了理解这一点，让我们进入可视化的 5 W:为什么，谁，什么，何时，何地。

## 机器学习中模型可视化的 5 W

### 1.为什么我们要可视化模型？

虽然我们已经在概述中对此进行了一些讨论，但让我们试着深入细节。

**可解释性**

我们需要了解模型的决策过程。在神经网络的情况下，这个问题的程度变得特别明显。

真实世界的神经网络模型具有数百万个参数和极端的内部复杂性，因为它们在训练期间使用许多非线性变换。可视化这种复杂的模型将有助于我们建立对自动驾驶汽车、帮助医生诊断的医学成像模型或卫星图像模型的信任，这些模型在救援规划或安全工作中可能至关重要。

**调试&改进**

构建机器学习模型是一个充满实验的迭代过程。寻找超参数的最佳组合可能相当具有挑战性。可视化可以加速这个过程。

反过来，这可以加速整个开发过程，即使模型在开发过程中遇到一些问题。

**对比&选择**

从一组表现良好的模型中选择最佳模型的行为可以简单地简化为可视化模型中提供最高精度或最低损失的部分，同时确保模型不会过度拟合。

框架可以被设计成在单个模型随时间训练时比较单个模型的不同快照，即，比较 n1 个时期之后的模型和 n2 个时期的训练时间之后的相同模型。

**教学理念**

也许教学是可视化最有用的地方，用于教育新手关于机器学习的基本概念。

可以设计交互式平台，用户可以在其中摆弄多个数据集并切换参数，以观察对模型中间状态和输出的影响。这将有助于对模型如何工作建立直觉。

### 2.谁应该使用可视化？

**数据科学家/机器学习工程师**

主要专注于开发、试验和部署模型的人将从可视化中受益最多。

许多从业者已经使用的一些著名工具包括 TensorBoard、DeepEyes 或 Blocks。所有这些工具都为用户提供了对超参数调整、修剪不必要的层等事情的扩展控制，从而允许他们的模型实现更好的性能。

**模型用户**

可视化可能对其他涉众有好处，可能有一些技术背景，但主要是通过 API 处理模型服务的消费。

例如 Activis，这是一个由脸书开发的可视化分析系统，供他们自己的工程师探索内部部署的神经网络。

这种可视化工具对于那些只希望使用预训练模型来预测自己任务的人来说非常有用。

**新手用户**

在“为什么”部分，我提到了可视化如何帮助新学生学习什么是机器学习——这一点在这里也是正确的。

这个群体还可以进一步扩大，包括好奇的消费者，他们由于害怕隐私侵犯而不愿使用 ML 驱动的应用程序。

一些基于 web 的 JavaScript 框架，如 ConvNetJS & TensorFlow.js，使开发人员能够为模型创建高度交互式的可探索解释。

### 3.我们能想象什么？

**模型架构**

您可以想象的第一件也是最主要的事情是模型架构。它告诉我们有多少层，它们的位置顺序，等等。

这一部分还包括计算图形，它定义了一个模型在历元迭代后如何训练、测试、保存到磁盘和检查点。

所有这些都可以帮助开发人员更好地理解他们的模型内部发生了什么。

**学习参数**

在训练的反向传播阶段调整的参数属于这一类。

可视化权重和偏差可能有助于理解模型学到了什么。同样，在卷积神经网络中，我们可以看看学习过的过滤器，看看模型学习了什么样的图像特征。

**模型指标**

每个时期计算的诸如损失、准确性和其他误差度量的汇总统计可以表示为模型训练过程中的时间序列。

通过一组数字来表示模型可能看起来有点抽象，但是，它们有助于在训练时跟踪模型的进展。

这些指标不仅描述了单个模型的性能，而且对于同时比较多个模型也是至关重要的。

### 4.什么时候可视化最相关？

**训练期间**

训练时使用可视化是监控和跟踪模型性能的好方法。有很多工具正好可以做到这一点(neptune.ai、weights 和 biases 等)，我们稍后将讨论它们。

例如， [Deep View](https://web.archive.org/web/20221201170702/https://arxiv.org/pdf/1909.09154.pdf) 使用其自身的监测指标(如辨别力和密度指标)来可视化模型，这有助于通过在训练阶段早期简单地观察神经元密度来检测过度拟合。

另一个工具 [Deep Eyes](https://web.archive.org/web/20221201170702/https://ieeexplore.ieee.org/document/8019872) ，可以识别稳定和不稳定的层和神经元，因此用户可以修剪他们的模型以加快训练速度。

**训练后**

有一些技术，如属性可视化，用于重新生成一个突出显示重要区域的图像，以及特征可视化，用于生成一个全新的图像，该图像被认为是同一类别的代表。它们通常在训练模型之后在计算机视觉领域中执行。

一些工具，如[嵌入投影仪](https://web.archive.org/web/20221201170702/https://projector.tensorflow.org/)，专门用于可视化由训练过的神经网络产生的 2D 和 3D 嵌入。

同样，如前所述，ActiVis、RNNVis、LSTMVis 等工具也是在培训后使用的，用于可视化甚至比较不同的模型。

### 5.可视化应用在哪里？

**应用领域&型号**

可视化已经在自动驾驶、城市规划、医学成像等领域大量使用，以增加用户对模型的信任。

视觉分析系统正在被开发，以更多地了解更难的网络类型，如 GANs，这种网络仅出现了几年，但在数据生成方面取得了显著的成果。

例子包括 DGMTracker 和 GANViz，它们专注于理解 GANs 的训练动态，以帮助模型开发者更好地训练这些复杂的模型。

**研究&发展**

将可视化与研究相结合，创造了模型可解释性和民主化的工具和框架。这一快速发展领域的另一个结果是，新的作品立即被公开和开源，而不必等待它在某个会议上“正式”发表。

例如，用于实现神经网络的最流行的库是开源的，并且对改进代码库的所有方面都有一致的贡献。

到目前为止，我们已经讨论了进行可视化的所有理论方面，现在让我们来看看最重要的一个方面。

## 我们如何可视化模型？

当我们谈论可视化模型时，我们实际上是在谈论绘制一幅允许模型学习和得出推论的关键组件的图片。如果我们想象一下，我们就能很好地看到里面:

### 1.模型架构

模型的设计给出了一个很好的关于数据如何在模型内部流动的想法。将它可视化有助于跟踪在哪个阶段应用了什么操作。

一种流行的方法，特别是在神经网络中，是用一个**节点链接图**，其中神经元被显示为节点，边权重被显示为链接。由于 Tensorboard 越来越受欢迎，这种方法也正在成为标准。

除此之外，如果你想窥视内部，某些机器学习算法有内置的规定。我们将在下一节中看看这方面的例子。

### 2.模特培训

监控和观察一个时期接一个时期计算的多个度量(如损失和准确性)有助于在训练阶段跟踪模型进展。

这可以通过将指标视为时间序列并绘制成线图来实现，这一步不需要外部帮助。

另一种方法是使用 Tensorboard 等专门为此目的设计的复杂工具。使用框架的好处是它们非常灵活，交互性强，并且节省你大量的时间。

### 3.模型推理

推理是从训练好的模型中得出结论的过程。可视化结果有助于解释和追溯模型如何生成其估计值。有几种方法可以做到这一点:

*   可视化实例级观察，对单个数据实例在整个网络中的转换过程及其最终输出进行深入分析和审查。
*   借助混淆矩阵或热图，这可以进一步扩展到识别和分析错误分类的实例。这将允许我们了解特定实例何时会失败以及它是如何失败的。
*   这个阶段的可视化是进行交互式实验的好方法——用输入数据或超参数进行实验，看看它如何影响结果。Tensorflow 游乐场就是一个很好的例子。

到目前为止，我们已经研究了进入可视化世界所需的所有先决条件。现在，是时候将“如何做”部分扩展到实用性，并检查一些工具来完成这项工作。

### 模型架构可视化

由于决策树的树状结构，它是容易解释的模型。您可以简单地检查分支上的条件，并在模型预测出来时跟踪流程。

有几种方法可以可视化决策树。先说 sklearn 自己提供的。

进行所需的进口。

```py
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris

```

如你所见，在这个例子中，我们将使用著名的虹膜数据集。

下一步是定义树并使之适合数据。

```py
iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(x, y)

```

现在让我们画出拟合的树。

```py
plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, fontsize=10)
plt.show()

```

这是它从另一端出来的方式。

*   我们的数据集中有 4 个特征，萼片长度、萼片宽度、花瓣长度、花瓣宽度，顺序相同。根节点根据花瓣长度分割整个种群。
*   这导致具有分类样本的叶节点，而剩余的在花瓣宽度上再次分裂，因为相关联的基尼杂质指数仍然很高。
*   这个循环继续下去，直到我们获得具有低 Gini 杂质指数的同质节点，或者达到 MAX_DEPTH。
*   总而言之，我们对分类决策树模型的架构有了一个相当不错的想法。

另一种可视化决策树的方法是使用 dTreeViz 库。它不仅适用于 scikit-learn 树，还支持 XGBoost、Spark MLlib 和 LightGBM 树。

我们来看看它和 sklearn 的功能有什么不同。

首先，库需要通过 pip 或 conda 安装，你可以在这里找到说明[。](https://web.archive.org/web/20221201170702/https://github.com/parrt/dtreeviz#install)

安装完成后，就可以进行所需的导入了。

```py
from sklearn import tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from dtreeviz.trees import *

```

定义、拟合和绘制树。

```py
classifier = tree.DecisionTreeClassifier(max_depth=4)
iris = load_iris()
classifier.fit(iris.data, iris.target)
viz = dtreeviz(classifier, iris.data, iris.target,
target_name='variety',
feature_names= iris.feature_names,
class_names=["setosa", "versicolor", "virginica"])

viz.view()

```

这就是我们得到的。

*   所获得的图传达了与 sklearn 非常相似的含义，但是对于每个决策节点，它更具有直方图的描述性。
*   您可以通过设置参数 fancy=False 来关闭这些图。
*   类似地，在这个库的帮助下，您还可以可视化回归树、特征-目标空间热图和决策边界。

如果您正在处理一个需要可视化的神经网络模型，这可能是一种方法。让我们来看看如何利用它。

与 dVizTree 类似，这个库也依赖于需要安装的 graphviz。你可以在这里找到安装说明[。](https://web.archive.org/web/20221201170702/https://github.com/RedaOps/ann-visualizer#Installation)

进行所需的进口。

```py
import keras
from keras.models import Sequential
from keras.layers import Dense
from ann.visualizer.visualize import ann_viz

```

现在让我们定义我们的神经网络。

```py
network = Sequential()
network.add(Dense(units=6, activation='relu',
kernel_initializer='uniform', input_dim=7))
network.add(Dense(units=4, activation='relu',
kernel_initializer='uniform'))
network.add(Dense(units=1, activation='sigmoid",
kernel_initializer='uniform')) 
```

绘制网络。

```py
ann_viz(network, view=True, title=’Example ANN’)

```

这就是输出。

*   这很好地概述了我们定义的神经网络模型的架构。
*   我们可以用代码计算每一层中神经元的数量，并看到它正如我们所希望的那样出现。
*   这个库唯一的缺点是它只能和 Keras 一起工作。

我们刚刚看到了一个如何可视化人工神经网络架构的例子，但这并不是这个库所能做的全部。我们也可以使用这个库来可视化一个卷积神经网络。让我们看看怎么做。

首先，像往常一样，我们来定义一下 CNN。

```py
def build_cnn_model():
model=keras.models.Sequential()

model.add(Conv2D(32, (3, 3), padding="same",
input_shape=(32, 32, 3), activation="relu"))

model.add(Conv2D(64, (3, 3), padding="same",
input_shape=(32, 32, 3),activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

return model

```

出于可视化的目的，我们保持网络的规模较小。现在让我们画出来，看看我们会得到什么。

*   这确实为我们的 CNN 的设计描绘了一幅很好的图画，所有的层次都描述得很好。
*   您只需要修改网络代码来获得新的可视化。

正如其创造者所描述的那样，Netron 是深度学习和机器学习模型的查看器工具，可以为模型的架构生成漂亮的描述性可视化。

它是一个跨平台的工具，可以在 Mac、Linux 和 Windows 上工作，支持多种框架和格式，如 Keras、TensorFlow、Pytorch、Caffe 等。我们如何利用这个工具？

作为一个独立于操作系统的工具，你可以按照[这些](https://web.archive.org/web/20221201170702/https://github.com/lutzroeder/netron#install)的说明将它安装在你的机器上，或者简单地使用他们的 web 应用程序，我们将在这里使用。

让我们想象一下我们为最后一个工具定义的 CNN。我们需要做的就是保存模型，并以. h5 格式或任何其他支持的格式上传保存的文件。这是我们得到的结果:

*   起初，它可能看起来类似于我们用 ANN visualizer 得到的结果，但两者之间有一个很大的区别——Netron 的交互性更强。
*   我们可以根据自己的方便将图表的方向更改为水平或垂直。
*   不仅如此，所有带颜色的节点都是可扩展的，单击它们可以查看各自节点的属性并获得更好的理解。例如，当我们单击 max_pooling2d 时，我们会得到这样的结果:

*   我们可以看到许多属性，如数据类型、步幅大小、可训练性等。可以由此推断出被点击的节点，使它比我们以前的工具好一点。

这个工具主要用于参数化地说明神经网络(NN ),并将这些图形导出到可缩放矢量图形(SVG ),因此得名 NN-SVG。

该工具可以生成三种类型的图形:

*   经典的全连接神经网络(FCNN)图，
*   卷积神经网络(CNN)数字，以及
*   深度神经网络遵循 AlexNet 论文中介绍的风格。

这个工具是托管的，所以不需要任何安装。以下是借助该工具创建的简单神经网络架构的示例:

*   我们有很多选择，比如:

1.  边缘宽度与边缘重量成比例，
2.  边缘不透明度与边缘权重成比例，
3.  边缘颜色与边缘重量成比例，
4.  层间距，
5.  操纵建筑和重量。

*   所有这些选项都可以让您非常快速地创建直观的插图。

如上所述，我们还可以使用该工具创建卷积神经网络的设计，如下例所示:

*   我们在 CNN 也有一系列的选择，就像我们在神经网络上做的一样。
*   你可以简单地在那里操作架构来得到一个新的输出，这个工具非常具有交互性，是一个非常好的选择。
*   这个工具在为研究和出版物创建网络图方面非常流行。

没有 TensorFlow 的开源可视化工具包 TensorBoard，任何模型可视化教程都是不完整的。我把它留到了最后，因为它很大。在这里，我们将只讨论它在模型架构中的使用，我们将在下一节中继续讨论它。

因此，首先，TensorBoard 安装可以通过这两个命令中的任何一个来完成。

```py
pip install tensorboard

conda install -c conda-forge tensorboard

```

现在让我们通过在单元格中运行这个命令来将 TensorBoard 加载到我们的笔记本中。

```py
%load_ext tensorboard

```

加载后，我们必须创建一个日志目录，TensorBoard 将在其中存储所有日志并从中读取，以显示各种可视化效果，之后 TensorBoard 必须重新加载更改。

```py
log_folder=’logs’
%reload_ext tensorboard

```

现在让我们进行所需的导入并定义我们的模型。在这个练习中，我们将使我们的模型适合 MNIST 数据集。

```py
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X train, X_test = X_train / 255.0, X_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='sgd',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

```

现在我们需要创建一个 TensorBoard 回调，它负责记录所有的事件，然后指定我们创建的日志目录。

```py
callbacks = [TensorBoard(log_dir=log_folder, histogram_freq=1,
 write_graph=True, write_images=True,
 update_freq='epoch', profile_batch=2)]

```

最后，我们使模型适合我们的数据，并传入回调，以便以后可以可视化所有内容。

```py
model.fit(X_train, y_train, epochs=5,
    validation_split=0.15, callbacks=callbacks)

```

现在，让我们通过运行以下命令将 TensorBoard 窗口直接加载到 jupyter 笔记本中:

```py
%tensorboard --logdir={log_folder}

```

接下来，如果导航到 graphs 选项卡，我们将看到以下输出:

这就是我们如何用 TensorBoard 看到模型的架构。

我们已经探索了可视化模型架构的工具和框架，现在让我们进入下一部分——培训可视化。

### 模型训练可视化

从我们停止的地方继续，在上一节中，我们已经将我们的神经网络拟合到 MNIST 数据集，并检查了“图形”选项卡。但是，事实证明 TensorBoard 中还有许多其他选项卡可供探索。

让我们从“标量”开始。

如下图所示，该选项卡处理一个又一个历元的损失和精度计算图。

下一个选项卡是“图像”。

该选项卡显示权重和偏差。每个图像都有一个滑块，我们可以调整它来显示不同时期的参数。

接下来是“分发”选项卡。

它显示了每个历元上某个密集层的权重和偏差的分布。

“直方图”选项卡的功能与“分布”选项卡类似，只是借助了直方图。

最有趣的选项卡是“投影仪”。它可以可视化任何类型的向量表示，无论是单词嵌入还是图像的 numpy 数组表示。

默认情况下，它使用主成分分析(PCA)将这种可视化绘制到 3D 空间中，但也有其他降维方法的选项，如 UMAP 或 T-SNE。事实上，您可以定义自己的自定义方法。

用 PCA 看起来是这样的。

我们已经讨论了所有的选项卡，但是我们仍然可以使用 TensorBoard 做很多事情，比如可视化训练数据、绘制混淆矩阵或超参数调整，但是它们超出了本文的范围。

让我们看看另一个工具。

Neptune 是一个[元数据存储库](/web/20221201170702/https://neptune.ai/product#what-is-metadata-store)，可以免费用于个人项目，也可以作为 API 服务使用。

虽然编码部分可以在任何地方完成，但只要代码连接到 Neptune，持续的跟踪和日志记录就会在 [Neptune 的 UI](https://web.archive.org/web/20221201170702/https://docs.neptune.ai/you-should-know/core-concepts#web-interface-neptune-ui) 中持续进行，从而简化项目管理。

让我们从安装所需的东西开始:

```py
!pip install -q neptune-client
!pip install -q neptune-contrib

```

我们将使用与上述示例相似的模型和 MNIST 数据集，因此我们不会重复这一部分。

为了在每个批次和时期之后记录度量，让我们创建一个 NeptuneLogger 回调。这部分类似于我们在前面的例子中创建的 TensorBoard 回调。

```py
from tensorflow.keras.callbacks import Callback
class NeptuneLogger(Callback):
    def on_batch_end(self, batch, logs={}):
        for log_name, log_value in logs. items():
            run["batch/{}".format(log_name)].log(log_value)

    def on epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            run["epoch/{}".format(log_name)].log(log_value)

```

为了将我们的代码连接到 Neptune 应用程序，我们需要一个 API 令牌。要获得 API 令牌，您需要[向 Neptune 注册](https://web.archive.org/web/20221201170702/https://docs.neptune.ai/getting-started/installation)并创建一个项目。那个项目的名字将违背参数*项目*和对应的 API 令牌违背 *api_token。*

现在让我们初始化 API。

```py
run = neptune.init(project=YOUR_PROJECT_NAME,  api_token=YOUR_API_TOKEN)

```

现在让我们来处理我们想要记录的任何内容。

```py
EPOCHS = 5
BATCH_SIZE = 32

run["parameters/epochs"] = EPOCHS
run["parameters/batch_size"] = BATCH_SIZE

run[ "sys/name"] = "metrics"
run[ "sys/tags"].add("demo")
```

太好了！现在剩下要做的就是将我们的 NeptuneLogger 作为 keras 回调函数传递。

```py
history = model.fit(x=x_train, y=y_train,
      epochs=EPOCHS, batch_size=BATCH_SIZE,
      validation_data=(x_test, y_test),
      callbacks=[NeptuneLogger()])

```

一旦执行完最后一个代码单元，我们就可以进入 Neptune 应用程序的 UI 来可视化我们记录的任何内容。

针对批次绘制的训练准确度/损失。

训练准确度/损失相对于时期绘制。

验证准确度/损失对时期作图。

除了损失和准确性之类的简单指标，您还可以轻松地绘制其他东西，如混淆矩阵或 AUC-ROC 曲线([参见此处的示例](https://web.archive.org/web/20221201170702/https://app.neptune.ai/common/project-cv/e/PROJCV-103/all?path=model&attribute=visualization))。

虽然这只是一个有限日志记录的演示，但是您可以想象当处理一个涉及一致的再训练和更新的项目时，这个工具使模型的不同方面可视化是多么容易。

如果你正在自己运行一些实验，并且正在寻找一个可视化工具，TensorBoard 是一个不错的选择。海王星更适合那些寻找复杂工具的研究人员，这将使他们能够更深入地研究实验过程。它还提供团队协作功能。

[查看海王星和张量板的详细对比。](/web/20221201170702/https://neptune.ai/vs/tensorboard)

就像 Neptune 一样，这个工具也有助于跟踪、监控和可视化 ML 模型和项目。

首先，在他们的[网站](https://web.archive.org/web/20221201170702/https://wandb.ai/)上注册，然后通过以下命令安装&登录:

```py
pip install wandb
wandb login

```

输入 API 密钥后，您应该就一切就绪了。现在，让我们为 Keras 模型创建所需的导入。

```py
import wandb
from wandb.keras import WandbCallback

```

让我们初始化 wandb 并开始我们的项目。

```py
wandb.init(project=’blog-demo’)

```

现在我们需要做的就是训练我们目前使用的模型，并将 WandbCallback 传递给 log metrics。

```py
model.fit(X_train, y_train, validation_data=(X_test, y_test),
    callbacks=[WandbCallback()], epochs=5)

```

现在我们可以进入 UI，看看记录了什么。

这只是看一看它可能是什么样子。然而，就像 Neptune 一样，这可以扩展到绘制许多不同的东西，以及基于这些记录的指标比较不同的模型。

TensorWatch 是微软研究院提供的数据科学调试和可视化工具。大多数当前可用的工具遵循“所见即所得”(WYSIWYL)的方法，该方法使用许多预定义的可视化来表示结果。

这可能会导致模型不断变化的问题。TensorWatch 通过将一切都视为流来解决这一问题。对其工作原理的简单描述是:

*   在 TensorWatch 流中写入内容时，值会被序列化并发送到 TCP/IP 套接字，其中包括您指定的文件。
*   在 Jupyter 笔记本中，以前记录的值将从文件中加载，然后 TCP-IP 套接字将监听任何其他未来的值。
*   然后，可视化器监听流，并在值到达时呈现它们。

使用 TensorWatch 的唯一缺点是目前只支持 PyTorch 框架。

让我们从安装开始吧。

```py
pip install tensorwatch

```

接下来，我们需要安装一个名为 regim 的 Python 包，它允许我们获取 PyTorch 模型文件，并使用少量代码在指定的数据集上运行训练和测试时期。

使用 regim 包，我们可以使用训练数据集来训练一个跨时期的模型，维护一些指标，并对事件进行回调。在每个时期之后，它可以在测试数据集上运行到目前为止训练好的模型，并在其上维护度量。

它为训练和测试周期维护了单独的观察器，所以我们可以分别看到每个周期的度量。port 参数指定其套接字相对于基线端口的偏移量。

```py
git clone https://github.com/sytelus/regim.git
cd regim
pip install -e .

```

然后，从安装 regim 的文件夹中运行您的培训脚本。

```py
python mnist_main.py

```

通过 regim 包，我们可以使用训练数据集来训练一个跨时期的模型，维护一些指标，并对事件进行回调。在每个时期之后，它可以在测试数据集上运行到目前为止训练好的模型，并在其上维护度量。

它为训练和测试周期维护了单独的观察器，所以我们可以分别看到每个周期的度量。port 参数指定其套接字相对于基线端口的偏移量。

```py
train = tw.WatcherClient(port=0)
test = tw.WatcherClient(port=1)

```

现在，让我们绘制几个指标，如训练损失、训练准确性、测试损失和测试准确性。

```py
loss_stream = train.create_stream(expr='lambda d:
                                 (d.metrics.epochf,
                                 d.metrics.batch_loss)', event_name='batch')
loss_plot = tw.Visualizer(loss_stream, vis_type='line',
                                 xtitle='Epoch', ytitle='Train Loss')

acc_stream = train.create_stream(expr='lambda d:
                                 (d.metrics.epochf, d.metrics.batch_accuracy)', event_name='batch')
acc_plot = tw.Visualizer(acc_stream, vis_type='line',
                                 host=loss_plot, xtitle='Epoch', ytitle='Train Accuracy', yrange=(0,))

test loss_stream = test.create_stream(expr='lambda d:
                                 (d.metrics.epochf, d.metrics.batch_loss)', event_name='batch')
test_loss_plot = tw.Visualizer(test_loss_stream, vis_type='line',
                                 host=loss_plot, xtitle= 'Epoch', ytitle='Test Loss', yrange=(0,))

test_acc_stream = test.create_stream(expr='lambda d:
                                 (d.metrics.epochf,
                                 d.metrics.batch_accuracy)', event_name='batch')
test_acc_plot = tw.Visualizer(test_acc_stream, vis type='line',
                                 host=loss_plot, xtitle='Epoch', ytitle='Test Accuracy',yrange=(0,))

test_acc_plot.show()
```

这是我们得到的结果:

类似地，我们也可以通过以下方式绘制每层的平均重量梯度:

```py
grads_stream = train.create_stream(expr='lambda
                                 d:grads_abs_mean(d.model)',
                                 event_name='batch', throttle=1)

grads_plot = tw.Visualizer(grads_stream, vis_type='line',
                                 title="Weight Gradients",
                                 clear_after_each=True,
                                 xtitle="Layer",
                                 ytitle="Abs Mean Gradient', history_len=20)

grads_plot.show()
```

下面是渐变的样子。在这一点上，很明显，这些只是我们可以用这个工具做的一小部分事情，还有更多要探索的。由于这个工具仍在开发中，一旦它开始支持更广泛的框架，我们肯定会看到更多。

这个工具是一个荣誉称号，因为它在学习和教育非专家用户关于神经网络的内部机制方面有更多的用途。

这是谷歌提供的基于 Tensorflow 的开源工具。您可以在浏览器中模拟小型神经网络，并在使用它们时观察结果。

看起来是这样的:

*   您可以在分类和回归之间切换问题类型，并修改网络和问题的每个可能方面，从激活函数到多个隐藏层，从使用纯数据集到有噪声的数据集。

*   一旦一切都设置好了，只需点击播放按钮就可以开始训练，之后您可以观察形成的决策边界(模型如何分隔不同的类)。

*   模型训练看起来很有趣，并且给你这个黑箱背后的直觉。

我们已经介绍了相当多的可视化训练工具，让我们进入可视化的最后一部分。

### 模型推理可视化

这主要涉及解释模型生成的预测，并收集关于它们最初是如何和为什么达到的想法。

我在这个博客的另一篇文章中广泛讨论了这个问题。它涵盖了相当多的工具，可以成为您的 [MLOps 工具](/web/20221201170702/https://neptune.ai/blog/best-mlops-tools)库的一个很好的补充。

## 包扎

在本文中，我们涉及了很多领域，从寻找为什么我们首先需要可视化模型的答案开始，到获得大量可以帮助我们创建可视化的工具的实践经验。

我希望这篇文章能帮助你更好地理解可视化，并且下次当你遇到可视化就是答案的问题时，你会利用这里讨论的工具和获得的知识。

### 未来方向

这里讨论的工具和框架只是所有工具中最流行的。随着研究的快速发展，你应该时刻留意新的工具。

除此之外，始终积极寻找可视化的新用例，在这些用例中，展示模型的图片或插图可能会解决手头的问题。

目前就这些。感谢阅读！

### 资源