# 使用 Keras 调谐器自动优化超参数

> 原文：<https://blog.paperspace.com/hyperparameter-optimization-with-keras-tuner/>

**超参数**是决定机器学习模型结构并控制其学习过程的配置。它们不应该与模型参数(如偏差)混淆，后者的最佳值是在训练期间确定的。

超参数是可调整的配置，可手动设置和调整以优化模型性能。它们是顶级参数，其值有助于确定模型参数的权重。两种主要类型的超参数是确定模型结构的模型超参数(如层数和层单元)和影响和控制学习过程的算法超参数(如优化算法和学习速率)。

用于训练神经网络的一些标准超参数包括:

1.隐藏层数

2.隐藏层的单位数

3.退出率——通过在训练期间随机退出节点，可以使用单一模型来模拟大量不同的网络架构

4.激活函数(Relu，Sigmoid，Tanh) -在给定一个或一组输入的情况下，定义该节点的输出

5.优化算法(随机梯度下降、Adam Optimizer、RMSprop、e.t.c) -用于更新模型参数和最小化损失函数值的工具，根据训练集进行评估。

6.损失函数——衡量你的模型在预测预期结果方面有多好

7.学习率——控制每次更新模型权重时，响应估计误差而改变模型的程度

8.训练迭代次数(epochs) -学习算法在整个训练数据集中工作的次数。

9.批量大小——这个梯度下降的超参数控制着在模型的内部参数更新之前训练样本的数量。

在建立机器学习模型时，会设置超参数来指导训练过程。根据初始训练后模型的性能，这些值被反复调整以改进模型，直到选择出产生最佳结果的值的组合。调整超参数以获得优化机器学习模型性能的正确值集的过程被称为超参数调整。

在深度学习中，调整超参数可能具有挑战性。这主要是由于需要正确设置的不同配置、重新调整这些值以提高性能的几次尝试以及为超参数设置次优值所产生的不良结果。在实践中，这些值通常是基于某些推理来设置和微调的，例如特定问题的一般原则(例如，使用 softmax 激活函数进行多类分类)、构建模型的先前经验(例如，将隐藏层的单元逐渐减少到原来的 2 倍)、领域知识和输入数据的大小(为较小的数据集构建更简单的网络)。

即使有了这种认识，仍然很难为这些超参数得出完美的值。从业者通常使用试错法来确定最佳超参数。这是通过基于他们对问题的理解来初始化值，然后在为模型选择具有最佳性能的最终值之前，根据模型的性能在几次训练试验中本能地调整值来完成的。

以这种方式手动微调超参数对于管理计算资源来说通常是费力、耗时、次优和低效的。另一种方法是利用可扩展的超参数搜索算法，如贝叶斯优化、随机搜索和超波段。Keras Tuner 是一个可扩展的 Keras 框架，它提供了这些内置的算法，用于深度学习模型的超参数优化。它还提供了一种优化 Scikit-Learn 模型的算法。

在本文中，我们将学习如何使用 Keras Tuner 的各种功能来自动搜索最佳超参数。任务是使用 Keras 调谐器获得最佳超参数，以构建一个模型，对 CIFAR-10 数据集的图像进行准确分类。

## 1.设置。

使用 Keras Tuner 需要安装 Tensorflow 和 Keras Tuner 包，并导入构建模型所需的库。
KerasTuner 需要 Python 3.6+和 TensorFlow 2.0+。这些预装在梯度机器上。

```py
# install required packages
pip install tensorflow
pip install keras_tuner
```

```py
# import required packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, BatchNormalization
from tensorflow.keras.layers import ReLU, MaxPool2D, AvgPool2D, GlobalAvgPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import keras_tuner as kt
from sklearn.model_selection import train_test_split
```

## 2.加载并准备数据集。

我们将加载包含 10 个对象类的 50，000 个训练和 10，000 个测试图像的 CIFAR-10 数据集。你可以在这里阅读更多关于数据集[的内容。我们还归一化图像像素值以具有相似的数据分布并简化训练。](https://www.cs.toronto.edu/~kriz/cifar.html)

预处理数据集版本被预加载到 Keras 数据集模块中，以便于访问和使用。

### 2.1 加载数据集并归一化图像像素值。

```py
# load the CIFAR-10 dataset from keras
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize the image pixel values
img_train = x_train.astype('float32') / 255.0
img_test = x_test.astype('float32') / 255.0

# split the train data into train and validation sets
x_train, y_train, x_val, y_val = train_test_split(x_train, y_train,                                                             test_size=0.25)
```

## 3.打造超模。

现在我们已经完成了设置并准备好了输入数据，我们可以为超调构建模型了。这是通过使用 Keras Tuner 定义一个搜索模型(称为超级模型)来完成的，然后将该模型传递给一个调谐器进行超调。

超模型要么通过创建一个定制的模型构建器函数，利用内置模型来定义，要么为高级用例子类化 Tuner 类。
我们将使用前两种方法来创建搜索模型，以自动调整我们的超参数。

### 3.a .使用定制模型。

为了使用定制模型，我们将通过定义我们需要的层来定义模型构建功能，定制用于找到最佳参数的搜索空间，并在我们不调整超参数时为它们定义默认值。

#### 3.a.1 定义建模功能。

该函数采用一个参数(hp ),该参数实例化 Keras Tuner 的*超参数*对象，并用于定义超参数值的搜索空间。我们还将编译并返回超模型以供使用。我们将使用 Keras 功能模型模式来构建我们的模型。

```py
# function to build an hypermodel
# takes an argument from which to sample hyperparameters
def build_model(hp):
​
  inputs = Input(shape = (32, 32, 3)) #input layer
  x = inputs
​
  # iterate a number of conv blocks from min_value to max_value
  # tune the number of filters
  # choose an optimal value from min_value to max_value
  for i in range(hp.Int('conv_blocks',min_value = 3, max_value = 5, default=3)): # Int specifies the dtype of the values
    filters = hp.Int('filters_' + str(i),min_value = 32,max_value = 256, step=32) 
​
    for _ in range(2):
      # define the conv, BatchNorm and activation layers for each block
      x = Convolution2D(filters, kernel_size=(3, 3), padding= 'same')(x)
      x = BatchNormalization()(x)
      x = ReLU()(x)
​
    # choose an optimal pooling type
    if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max': # hp.Choice chooses from a list of values
        x = MaxPool2D()(x)
    else:
        x = AvgPool2D()(x)
​
  x = GlobalAvgPool2D()(x) # apply GlobalAvG Pooling
​
  # Tune the number of units in the  Dense layer
  # Choose an optimal value between min_value to max_value
  x = Dense(hp.Int('Dense units',min_value = 30, max_value = 100, step=10, default=50), activation='relu')(x)
  outputs = Dense(10, activation= 'softmax')(x) # output layer 

  # define the model
  model = Model(inputs, outputs)
​
  # Tune the learning rate for the optimizer
  # Choose an optimal value frommin_value to max_value
  model.compile(optimizer= Adam(hp.Float('learning_rate', min_value = 1e-4, max_value =1e-2, sampling='log')), 
                loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])
  return model
```

###### **理解代码。**

**第 3 行:**我们定义了一个模型构建函数( **build_model** )并传递了一个参数 *(hp* )，该参数实例化了 Keras Tuner 包的超参数对象，用于定义超参数值的搜索空间。

第 5-6 行:我们定义我们的输入层，并将其传递给一个变量(x)

**第 11 行:**我们为我们的模型定义卷积块数量的搜索空间。我们使用 hp.Int 函数*来创建一个整数超参数搜索空间。这创建了从*最小值* + 1 到*最大值*的搜索空间。这将在 4 和 5 个卷积块的空间中搜索使精度最大化的最佳值。*

**第 12 行:**我们为块中每个卷积层的滤波器数量定义一个搜索空间。32 的*步骤*将连续卷积层的滤波器单元增加 32。

第 14-24 行:我们为每个块定义一组三层。每个子层对输入应用卷积、批量归一化和 ReLU 激活。*惠普。池层的 Choice* 函数随机选择一个提供的池应用于输入。然后，我们将预定义的滤波器搜索空间传递给卷积层。

**第 26 行:**我们应用全局平均池和密集层，搜索空间从*最小值*到*最大值*，步长*为 10。我们还通过激活 *softmax* 来定义输出层。*

第 34-40 行:最后，我们使用输入和输出层定义模型，编译模型并返回构建好的超模型。

为了编译模型，我们用 *hp 定义了一个学习率搜索空间。Float* 函数创建一个从 0.0001 到 0.002 的搜索空间，用于选择最佳学习率。

#### 3.a.2 初始化搜索算法(调谐器)。

在构建了*超模型*之后，我们现在可以初始化我们的搜索算法了。我们将不得不从内置的搜索算法中进行选择，如*贝叶斯优化、超波段和随机搜索、*经典机器学习模型。

在我们的例子中，我们将使用*超波段*搜索算法。tuner 函数接受参数，如*超级模型*、用于评估模型的*目标*、用于训练的 *max_epochs* 、每个模型的 *hyperband_iterations* 的数量、用于保存训练日志(可以使用 Tensorboard 可视化)的*目录*和 *project_name* 。

```py
# initialize tuner to run the model.
# using the Hyperband search algorithm
tuner = kt.Hyperband(
    hypermodel = build_model,
    objective='val_accuracy',
    max_epochs=30,
    hyperband_iterations=2,
    directory="Keras_tuner_dir",
    project_name="Keras_tuner_Demo")
```

### 3.b .使用内置模型。

Keras Tuner 目前提供了两个可调的内置模型，HyperResnet 和 HyperXception 模型，它们分别搜索 Resnet 和 Xception 架构的不同组合。使用内置模型定义调谐器类似于使用模型构建功能。

```py
# Initialize a random search tuner
# using the Resnet architecture
# and the Random Search algorithm
tuner = kt.tuners.RandomSearch(
  kt.applications.HyperResNet(input_shape=(32, 32, 3), classes=10),
  objective='val_accuracy',
  max_trials=30)
```

### 4.运行最佳超参数搜索。

然后，我们可以使用我们的调谐器在定义的搜索空间内搜索模型的最佳超参数。该方法类似于使用 Keras 拟合模型。

```py
# Run the search
tuner.search(x_train, y_train,
             validation_data= (x_test,y_test), 
             epochs=30,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
```

### 5.获得并显示最佳超参数和模型。

可以使用调谐器实例的 *get_best_hyperparameters* 方法获得定义的搜索空间内的模型的最佳超参数，并使用 *get_best_models* 方法获得最佳模型。

```py
# Get the optimal hyperparameters
best_hps= tuner.get_best_hyperparameters(1)[0]

# get the best model
best_model = tuner.get_best_models(1)[0]
```

我们还可以查看最佳超参数。在我们的例子中，我们可以这样实现:

```py
nblocks = best_hps.get('conv_blocks')
print(f'Number of conv blocks: {nblocks}')
for hyparam in [f'filters_{i}' for i in range(nblocks)] + [f'pooling_{i}' for i in range(nblocks)] + ['Dense units'] + ['learning_rate']:
    print(f'{hyparam}: {best_hps.get(hyparam)}')
```

这将显示卷积块的数量、卷积和密集层的过滤器和单位的最佳值、池层的选择以及学习率。

我们还可以使用适当的 Keras 函数查看优化模型的概要和结构。

```py
# display model structure
plot_model(best_model, 'best_model.png', show_shapes=True)

# show model summary
best_model.summary()
```

## 6.训练模型。

最后，在调用 *fit* 函数来训练模型之前，我们将使用最佳超参数建立模型。

```py
# Build the model with the optimal hyperparameters
# train the model.
model = tuner.hypermodel.build(best_hps)
model.fit(x_train, y_train, 
          validation_data= (x_val,y_val), 
          epochs= 25,
           callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
```

在这里，我对模型进行了 50 个时期的训练，并添加了一个 *EarlyStopping* 回调，以便在模型不再改进时停止训练。

## 6.评估模型。

我们可以在测试集上评估模型。我们将使用模型的*损失*和准确度*得分*来评估模型。您可以尝试其他适用的指标。

```py
# evaluate the result
eval_result = model.evaluate(x_test, y_test)
print(f"test loss: {eval_result[0]}, test accuracy: {eval_result[1]}")
```

## 总结。

超参数是机器学习模型性能的关键决定因素，用试错法来调整它们是低效的。Keras Tuner 应用搜索算法在定义的搜索空间中自动找到最佳超参数。

在本文中，我们利用 Keras 调谐器来确定多类分类任务的最佳超参数。我们能够使用自定义模型和内置模型在超模中定义搜索空间，然后利用提供的搜索算法自动搜索几个值和组合，为我们的模型找到超参数的最佳组合。

您可以查看 [Keras Tuner guide](https://keras.io/guides/keras_tuner/) ,了解如何在 Tensorboard 上可视化调优过程、分发超调过程、定制搜索空间以及为高级用例子类化 Tuner 类。