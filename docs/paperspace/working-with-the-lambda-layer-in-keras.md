# 在 Keras 中使用 Lambda 图层

> 原文：<https://blog.paperspace.com/working-with-the-lambda-layer-in-keras/>

Keras 是一个流行且易于使用的库，用于构建深度学习模型。它支持所有已知类型的层:输入层、密集层、卷积层、转置卷积层、整形层、归一化层、下降层、展平层和激活层。每一层都对数据执行特定的操作。

也就是说，您可能希望对未应用于任何现有图层的数据执行操作，然后这些现有图层类型将无法满足您的任务。举一个简单的例子，假设您需要一个层来执行在模型架构的给定点添加固定数字的操作。因为没有现有的层可以做到这一点，所以您可以自己构建一个。

在本教程中，我们将讨论在 Keras 中使用`Lambda`层。这允许您指定要作为函数应用的操作。我们还将看到在构建具有 lambda 层的模型时如何调试 Keras 加载特性。

本教程涵盖的部分如下:

*   使用`Functional API`构建 Keras 模型
*   添加一个`Lambda`层
*   将多个张量传递给λ层
*   保存和加载带有 lambda 层的模型
*   加载带有 lambda 层的模型时解决系统错误

## **使用`Functional API`** 构建 Keras 模型

有三种不同的 API 可用于在 Keras 中构建模型:

1.  顺序 API
2.  功能 API
3.  模型子类化 API

你可以在[这篇文章](https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing)中找到关于这些的更多信息，但是在本教程中，我们将重点关注使用 Keras `Functional API`来构建一个定制模型。由于我们想专注于我们的架构，我们将只使用一个简单的问题示例，并建立一个模型来识别 MNIST 数据集中的图像。

要在 Keras 中构建模型，您需要将层堆叠在另一层之上。这些层在`keras.layers`模块中可用(在下面导入)。模块名由`tensorflow`前置，因为我们使用 TensorFlow 作为 Keras 的后端。

```py
import tensorflow.keras.layers
```

要创建的第一层是`Input`层。这是使用`tensorflow.keras.layers.Input()`类创建的。传递给这个类的构造函数的必要参数之一是`shape`参数，它指定了将用于训练的数据中每个样本的形状。在本教程中，我们将只使用密集层，因此输入应该是 1-D 矢量。因此，`shape`参数被赋予一个只有一个值的元组(如下所示)。值为 784，因为 MNIST 数据集中每个影像的大小为 28 x 28 = 784。可选的`name`参数指定该层的名称。

```py
input_layer = tensorflow.keras.layers.Input(shape=(784), name="input_layer")
```

下一层是根据下面的代码使用`Dense`类创建的密集层。它接受一个名为`units`的参数来指定该层中神经元的数量。请注意该图层是如何通过在括号中指定该图层的名称来连接到输入图层的。这是因为函数式 API 中的层实例可在张量上调用，并且也返回张量。

```py
dense_layer_1 = tensorflow.keras.layers.Dense(units=500, name="dense_layer_1")(input_layer)
```

在密集层之后，根据下一行使用`ReLU`类创建一个激活层。

```py
activ_layer_1 = tensorflow.keras.layers.ReLU(name="activ_layer_1")(dense_layer_1)
```

根据下面的代码行，添加了另外两个 dense-ReLu 层。

```py
dense_layer_2 = tensorflow.keras.layers.Dense(units=250, name="dense_layer_2")(activ_layer_1)
activ_layer_2 = tensorflow.keras.layers.ReLU(name="relu_layer_2")(dense_layer_2)

dense_layer_3 = tensorflow.keras.layers.Dense(units=20, name="dense_layer_3")(activ_layer_2)
activ_layer_3 = tensorflow.keras.layers.ReLU(name="relu_layer_3")(dense_layer_3)
```

下一行根据 MNIST 数据集中的类数量将最后一个图层添加到网络架构中。因为 MNIST 数据集包括 10 个类(每个数字对应一个类)，所以此图层中使用的单位数为 10。

```py
dense_layer_4 = tensorflow.keras.layers.Dense(units=10, name="dense_layer_4")(activ_layer_3)
```

为了返回每个类的分数，根据下一行在前一密集层之后添加一个`softmax`层。

```py
output_layer = tensorflow.keras.layers.Softmax(name="output_layer")(dense_layer_4)
```

我们现在已经连接了层，但是模型还没有创建。为了构建一个模型，我们现在必须使用`Model`类，如下所示。它接受的前两个参数代表输入和输出层。

```py
model = tensorflow.keras.models.Model(input_layer, output_layer, name="model")
```

在加载数据集和训练模型之前，我们必须使用`compile()`方法编译模型。

```py
model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.0005), loss="categorical_crossentropy")
```

使用`model.summary()`我们可以看到模型架构的概述。输入层接受一个形状张量(None，784 ),这意味着每个样本必须被整形为一个 784 元素的向量。输出`Softmax`图层返回 10 个数字，每个数字都是该类 MNIST 数据集的分数。

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_layer (InputLayer)     [(None, 784)]             0         
_________________________________________________________________
dense_layer_1 (Dense)        (None, 500)               392500    
_________________________________________________________________
relu_layer_1 (ReLU)          (None, 500)               0         
_________________________________________________________________
dense_layer_2 (Dense)        (None, 250)               125250    
_________________________________________________________________
relu_layer_2 (ReLU)          (None, 250)               0         
_________________________________________________________________
dense_layer_3 (Dense)        (None, 20)                12550     
_________________________________________________________________
relu_layer_3 (ReLU)          (None, 20)                0         
_________________________________________________________________
dense_layer_4 (Dense)        (None, 10)                510       
_________________________________________________________________
output_layer (Softmax)       (None, 10)                0         
=================================================================
Total params: 530,810
Trainable params: 530,810
Non-trainable params: 0
_________________________________________________________________
```

现在我们已经构建并编译了模型，让我们看看数据集是如何准备的。首先，我们将从`keras.datasets`模块加载 MNIST，将它们的数据类型更改为`float64`，因为这使得训练网络比将其值留在 0-255 范围内更容易，最后重新调整，使每个样本都是 784 个元素的向量。

```py
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

x_train = x_train.astype(numpy.float64) / 255.0
x_test = x_test.astype(numpy.float64) / 255.0

x_train = x_train.reshape((x_train.shape[0], numpy.prod(x_train.shape[1:])))
x_test = x_test.reshape((x_test.shape[0], numpy.prod(x_test.shape[1:])))
```

因为在`compile()`方法中使用的损失函数是`categorical_crossentropy`，样本的标签应该根据下一个代码进行热编码。

```py
y_test = tensorflow.keras.utils.to_categorical(y_test)
y_train = tensorflow.keras.utils.to_categorical(y_train)
```

最后，模型训练开始使用`fit()`方法。

```py
model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_test, y_test))
```

至此，我们已经使用已经存在的层类型创建了模型架构。下一节讨论使用`Lambda`层构建定制操作。

## **使用λ层**

假设在名为`dense_layer_3`的稠密层之后，我们想要对张量进行某种操作，比如给每个元素加值 2。我们如何做到这一点？现有的层都没有这样做，所以我们必须自己建立一个新的层。幸运的是，`Lambda`层的存在正是为了这个目的。大家讨论一下怎么用吧。

首先构建将执行所需操作的函数。在这种情况下，名为`custom_layer`的函数创建如下。它只接受输入张量并返回另一个张量作为输出。如果不止一个张量被传递给函数，那么它们将作为一个列表被传递。

在这个例子中，只有一个张量作为输入，输入张量中的每个元素加 2。

```py
def custom_layer(tensor):
    return tensor + 2
```

在构建了定义操作的函数之后，接下来我们需要使用下一行中定义的`Lambda`类创建 lambda 层。在这种情况下，只有一个张量被提供给`custom_layer`函数，因为 lambda 层可以在名为`dense_layer_3`的稠密层返回的单个张量上调用。

```py
lambda_layer = tensorflow.keras.layers.Lambda(custom_layer, name="lambda_layer")(dense_layer_3)
```

下面是使用 lambda 层后构建完整网络的代码。

```py
input_layer = tensorflow.keras.layers.Input(shape=(784), name="input_layer")

dense_layer_1 = tensorflow.keras.layers.Dense(units=500, name="dense_layer_1")(input_layer)
activ_layer_1 = tensorflow.keras.layers.ReLU(name="relu_layer_1")(dense_layer_1)

dense_layer_2 = tensorflow.keras.layers.Dense(units=250, name="dense_layer_2")(activ_layer_1)
activ_layer_2 = tensorflow.keras.layers.ReLU(name="relu_layer_2")(dense_layer_2)

dense_layer_3 = tensorflow.keras.layers.Dense(units=20, name="dense_layer_3")(activ_layer_2)

def custom_layer(tensor):
    return tensor + 2

lambda_layer = tensorflow.keras.layers.Lambda(custom_layer, name="lambda_layer")(dense_layer_3)

activ_layer_3 = tensorflow.keras.layers.ReLU(name="relu_layer_3")(lambda_layer)

dense_layer_4 = tensorflow.keras.layers.Dense(units=10, name="dense_layer_4")(activ_layer_3)
output_layer = tensorflow.keras.layers.Softmax(name="output_layer")(dense_layer_4)

model = tensorflow.keras.models.Model(input_layer, output_layer, name="model")
```

为了查看馈送到 lambda 层之前和之后的张量，我们将创建两个新模型，除了上一个模型。我们将这些称为`before_lambda_model`和`after_lambda_model`。两种模型都使用输入层作为输入，但输出层不同。`before_lambda_model`模型返回`dense_layer_3`的输出，T3 是正好存在于 lambda 层之前的层。`after_lambda_model`模型的输出是来自名为`lambda_layer`的λ层的输出。这样做，我们可以看到应用 lambda 层之前的输入和之后的输出。

```py
before_lambda_model = tensorflow.keras.models.Model(input_layer, dense_layer_3, name="before_lambda_model")
after_lambda_model = tensorflow.keras.models.Model(input_layer, lambda_layer, name="after_lambda_model")
```

下面列出了构建和训练整个网络的完整代码。

```py
import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.datasets
import tensorflow.keras.utils
import tensorflow.keras.backend
import numpy

input_layer = tensorflow.keras.layers.Input(shape=(784), name="input_layer")

dense_layer_1 = tensorflow.keras.layers.Dense(units=500, name="dense_layer_1")(input_layer)
activ_layer_1 = tensorflow.keras.layers.ReLU(name="relu_layer_1")(dense_layer_1)

dense_layer_2 = tensorflow.keras.layers.Dense(units=250, name="dense_layer_2")(activ_layer_1)
activ_layer_2 = tensorflow.keras.layers.ReLU(name="relu_layer_2")(dense_layer_2)

dense_layer_3 = tensorflow.keras.layers.Dense(units=20, name="dense_layer_3")(activ_layer_2)

before_lambda_model = tensorflow.keras.models.Model(input_layer, dense_layer_3, name="before_lambda_model")

def custom_layer(tensor):
    return tensor + 2

lambda_layer = tensorflow.keras.layers.Lambda(custom_layer, name="lambda_layer")(dense_layer_3)
after_lambda_model = tensorflow.keras.models.Model(input_layer, lambda_layer, name="after_lambda_model")

activ_layer_3 = tensorflow.keras.layers.ReLU(name="relu_layer_3")(lambda_layer)

dense_layer_4 = tensorflow.keras.layers.Dense(units=10, name="dense_layer_4")(activ_layer_3)
output_layer = tensorflow.keras.layers.Softmax(name="output_layer")(dense_layer_4)

model = tensorflow.keras.models.Model(input_layer, output_layer, name="model")

model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.0005), loss="categorical_crossentropy")
model.summary()

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

x_train = x_train.astype(numpy.float64) / 255.0
x_test = x_test.astype(numpy.float64) / 255.0

x_train = x_train.reshape((x_train.shape[0], numpy.prod(x_train.shape[1:])))
x_test = x_test.reshape((x_test.shape[0], numpy.prod(x_test.shape[1:])))

y_test = tensorflow.keras.utils.to_categorical(y_test)
y_train = tensorflow.keras.utils.to_categorical(y_train)

model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_test, y_test))
```

请注意，您不必编译或训练这两个新创建的模型，因为它们的层实际上是从存在于`model`变量中的主模型中重用的。在这个模型被训练之后，我们可以使用`predict()`方法返回`before_lambda_model`和`after_lambda_model`模型的输出，看看 lambda 层的结果如何。

```py
p = model.predict(x_train)

m1 = before_lambda_model.predict(x_train)
m2 = after_lambda_model.predict(x_train)
```

下一段代码只打印前两个样本的输出。可以看到，从`m2`数组返回的每个元素实际上都是`m1`加 2 后的结果。这正是我们在自定义 lambda 层中应用的操作。

```py
print(m1[0, :])
print(m2[0, :])

[ 14.420735    8.872794   25.369402    1.4622561   5.672293    2.5202641
 -14.753801   -3.8822086  -1.0581762  -6.4336205  13.342142   -3.0627508
  -5.694006   -6.557313   -1.6567478  -3.8457105  11.891999   20.581928
   2.669979   -8.092522 ]
[ 16.420734    10.872794    27.369402     3.462256     7.672293
   4.520264   -12.753801    -1.8822086    0.94182384  -4.4336205
  15.342142    -1.0627508   -3.694006    -4.557313     0.34325218
  -1.8457105   13.891999    22.581928     4.669979    -6.0925217 ]
```

在本节中，lambda 层用于对单个输入张量进行运算。在下一节中，我们将看到如何将两个输入张量传递给这一层。

## **将一个以上的张量传递给λ层**

假设我们想要做一个依赖于名为`dense_layer_3`和`relu_layer_3`的两层的操作。在这种情况下，我们必须调用 lambda 层，同时传递两个张量。这可以简单地通过创建一个包含所有这些张量的列表来完成，如下一行所示。

```py
lambda_layer = tensorflow.keras.layers.Lambda(custom_layer, name="lambda_layer")([dense_layer_3, activ_layer_3])
```

这个列表被传递给`custom_layer()`函数，我们可以简单地根据下一个代码获取各个层。它只是把这两层加在一起。Keras 中实际上有一个名为`Add`的层，可以用来添加两层或更多层，但是我们只是展示如果有另一个 Keras 不支持的操作，你可以自己怎么做。

```py
def custom_layer(tensor):
    tensor1 = tensor[0]
    tensor2 = tensor[1]
    return tensor1 + tensor2
```

接下来的代码构建了三个模型:两个用于捕获传递给 lambda 层的来自`dense_layer_3`和`activ_layer_3`的输出，另一个用于捕获 lambda 层本身的输出。

```py
before_lambda_model1 = tensorflow.keras.models.Model(input_layer, dense_layer_3, name="before_lambda_model1")
before_lambda_model2 = tensorflow.keras.models.Model(input_layer, activ_layer_3, name="before_lambda_model2")

lambda_layer = tensorflow.keras.layers.Lambda(custom_layer, name="lambda_layer")([dense_layer_3, activ_layer_3])
after_lambda_model = tensorflow.keras.models.Model(input_layer, lambda_layer, name="after_lambda_model")
```

为了查看来自`dense_layer_3`、`activ_layer_3`和`lambda_layer`层的输出，下一段代码预测它们的输出并打印出来。

```py
m1 = before_lambda_model1.predict(x_train)
m2 = before_lambda_model2.predict(x_train)
m3 = after_lambda_model.predict(x_train)

print(m1[0, :])
print(m2[0, :])
print(m3[0, :])

[ 1.773366   -3.4378722   0.22042789 11.220362    3.4020965  14.487111
  4.239182   -6.8589864  -6.428128   -5.477719   -8.799093    7.264849
 17.503246   -6.809489   -6.846208   16.094025   24.483786   -7.084775
 17.341183   20.311539  ]
[ 1.773366    0\.          0.22042789 11.220362    3.4020965  14.487111
  4.239182    0\.          0\.          0\.          0\.          7.264849
 17.503246    0\.          0\.         16.094025   24.483786    0.
 17.341183   20.311539  ]
[ 3.546732   -3.4378722   0.44085577 22.440723    6.804193   28.974222
  8.478364   -6.8589864  -6.428128   -5.477719   -8.799093   14.529698
 35.006493   -6.809489   -6.846208   32.18805    48.96757    -7.084775
 34.682365   40.623077  ]
```

使用 lambda 层现在很清楚了。下一节将讨论如何保存和加载使用 lambda 层的模型。

## **保存并加载带有 Lambda 图层的模型**

为了保存一个模型(不管它是否使用 lambda 层),使用了`save()`方法。假设我们只对保存主模型感兴趣，下面是保存它的代码行。

```py
model.save("model.h5")
```

我们还可以使用`load_model()`方法加载保存的模型，如下一行所示。

```py
loaded_model = tensorflow.keras.models.load_model("model.h5")
```

希望模型能够成功加载。不幸的是，Keras 中的一些问题可能会导致在加载带有 lambda 层的模型时出现`SystemError: unknown opcode`。这可能是由于使用 Python 版本构建模型并在另一个版本中使用它。我们将在下一节讨论解决方案。

## **加载带有 Lambda 层的模型时解决系统错误**

为了解决这个问题，我们不打算以上面讨论的方式保存模型。相反，我们将使用`save_weights()`方法保存模型权重。

现在我们只保留了重量。模型架构呢？将使用代码重新创建模型架构。为什么不将模型架构保存为 JSON 文件，然后再次加载呢？原因是加载架构后错误仍然存在。

总之，经过训练的模型权重将被保存，模型架构将使用代码被复制，并且最终权重将被加载到该架构中。

可以使用下一行保存模型的权重。

```py
model.save_weights('model_weights.h5')
```

下面是复制模型架构的代码。`model`将不会被训练，但保存的权重将再次分配给它。

```py
input_layer = tensorflow.keras.layers.Input(shape=(784), name="input_layer")

dense_layer_1 = tensorflow.keras.layers.Dense(units=500, name="dense_layer_1")(input_layer)
activ_layer_1 = tensorflow.keras.layers.ReLU(name="relu_layer_1")(dense_layer_1)

dense_layer_2 = tensorflow.keras.layers.Dense(units=250, name="dense_layer_2")(activ_layer_1)
activ_layer_2 = tensorflow.keras.layers.ReLU(name="relu_layer_2")(dense_layer_2)

dense_layer_3 = tensorflow.keras.layers.Dense(units=20, name="dense_layer_3")(activ_layer_2)
activ_layer_3 = tensorflow.keras.layers.ReLU(name="relu_layer_3")(dense_layer_3)

def custom_layer(tensor):
    tensor1 = tensor[0]
    tensor2 = tensor[1]

    epsilon = tensorflow.keras.backend.random_normal(shape=tensorflow.keras.backend.shape(tensor1), mean=0.0, stddev=1.0)
    random_sample = tensor1 + tensorflow.keras.backend.exp(tensor2/2) * epsilon
    return random_sample

lambda_layer = tensorflow.keras.layers.Lambda(custom_layer, name="lambda_layer")([dense_layer_3, activ_layer_3])

dense_layer_4 = tensorflow.keras.layers.Dense(units=10, name="dense_layer_4")(lambda_layer)
after_lambda_model = tensorflow.keras.models.Model(input_layer, dense_layer_4, name="after_lambda_model")

output_layer = tensorflow.keras.layers.Softmax(name="output_layer")(dense_layer_4)

model = tensorflow.keras.models.Model(input_layer, output_layer, name="model")

model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.0005), loss="categorical_crossentropy")
```

下面是如何使用`load_weights()`方法加载保存的权重，并将其分配给复制的架构。

```py
model.load_weights('model_weights.h5')
```

## **结论**

本教程讨论了使用`Lambda`层创建自定义层，该层执行 Keras 中预定义层不支持的操作。`Lambda`类的构造函数接受一个指定层如何工作的函数，该函数接受调用层的张量。在函数内部，您可以执行任何想要的操作，然后返回修改后的张量。

尽管 Keras 在加载使用 lambda 层的模型时存在问题，但我们也看到了如何通过保存训练好的模型权重、使用代码重新生成模型架构，并将权重加载到该架构中来简单地解决这个问题。