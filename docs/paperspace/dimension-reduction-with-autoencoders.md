# 降维-自动编码器

> 原文：<https://blog.paperspace.com/dimension-reduction-with-autoencoders/>

###### 本教程是关于降维的 7 部分系列的一部分:

1.  [理解主成分分析(PCA)的降维](https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/)
2.  [利用独立成分分析(ICA)深入降低维度](https://blog.paperspace.com/dimension-reduction-with-independent-components-analysis/)
3.  [多维标度(MDS)](https://blog.paperspace.com/dimension-reduction-with-multi-dimension-scaling/)
4.  【T0 向】T1
5.  t-SNE
6.  [IsoMap](https://blog.paperspace.com/dimension-reduction-with-isomap)
7.  **[Autoencoders](https://blog.paperspace.com/dimension-reduction-with-autoencoders)**

(这篇文章假设你有神经网络的工作知识。有代码的笔记本可在 [github repo](https://github.com/asdspal/dimRed) 获得

自动编码器可以定义为神经网络，其主要目的是学习数据集中的基础流形或特征空间。自动编码器试图在输出端重建输入。与其他非线性降维方法不同，自动编码器不努力保持单一属性，如距离(MDS)、拓扑(LLE)。自动编码器通常由两部分组成:将输入转换为隐藏代码的编码器和从隐藏代码重构输入的解码器。自动编码器的一个简单例子就是下图所示的神经网络。

![autoencoder](img/166be1b1b8c61a73787a33467d29845d.png)

有人可能会问“如果输出和输入一样，自动编码器有什么用？如果最终结果与输入相同，特征学习或降维是如何发生的？”。
自动编码器背后的假设是，转换`input --> hidden --> input`将帮助我们了解数据集的重要属性。我们要学习的性质反过来又取决于对网络的限制。

**自动编码器的类型**
让我们来讨论几种流行的自动编码器。

1.  正则化自动编码器:这些类型的自动编码器在其损失函数中使用各种正则化项来实现所需的属性。
    隐藏代码的大小可以大于输入大小。1.1 稀疏自动编码器-稀疏自动编码器增加了隐藏层稀疏性的惩罚。正则化迫使隐藏层仅激活每个数据样本的一些隐藏单元。通过激活，我们意味着如果第 j 个隐藏单元的值接近 1，则它被激活，否则被去激活。从去激活的节点到下一层的输出为零。这种限制迫使网络仅压缩和存储数据的重要特征。稀疏自动编码器的损失函数可以表示为
    L(W，b) = J(W，b) +正则化项
    ![sparse_encoder](img/a8b78f5f52a3910fcd2926c5fd0bd293.png)
    中间层表示隐藏层。绿色和红色节点分别表示停用和激活的节点。

1.2 去噪自动编码器:在去噪自动编码器中，随机噪声被故意添加到输入中，并且网络被迫重建未掺杂的输入。解码器功能学会抵抗输入中的微小变化。这种预训练产生了在一定程度上不受输入中噪声影响的健壮的神经网络。
![denoising_autoencoder](img/d3f676c8f023471cea9d42fed22ded6a.png)
标准正态函数被用作噪声函数来产生被破坏的输入。

1.3 收缩自动编码器:收缩自动编码器不是向输入添加噪声，而是在特征提取函数的导数的大值上添加惩罚。当输入的变化不显著时，小的特征提取函数(f(x))导数值导致可忽略的特征变化。在收缩编码器中，特征提取功能是鲁棒的，而在去噪编码器中，解码器功能是鲁棒的。
2。变分自动编码器:变分自动编码器基于非线性潜变量模型。在潜在变量模型中，我们假设可观察的 *x* 是从隐藏变量 y 中产生的。这些隐藏变量 *y* 包含了关于数据的重要属性。这些自动编码器由两个神经网络组成，第一个用于学习潜在变量分布，第二个用于从潜在变量分布获得的随机样本中产生可观测值。除了最小化重建损失之外，这些自动编码器还最小化潜在变量的假设分布和由编码器产生的分布之间的差异。它们在生成图像方面非常受欢迎。
![variational-encoder](img/9b35977f1e2398a6f4b39047129cd34a.png)
潜在变量分布的一个好选择是高斯分布。如上图所示，编码器输出假设高斯的参数。接下来，从高斯分布中提取随机样本，解码器从随机样本中重构输入。
3。欠完整自动编码器:在欠完整自动编码器中，隐藏层的大小小于输入层。通过减小隐藏层的大小，我们迫使网络学习数据集的重要特征。一旦训练阶段结束，解码器部分被丢弃，编码器被用于将数据样本变换到特征子空间。如果解码器变换是线性的并且损失函数是 MSE(均方误差),则特征子空间与 PCA 的特征子空间相同。对于一个学习有用东西的网络来说，隐藏代码的大小不应该接近或大于网络的输入大小。还有，一个高容量(深度和高度非线性)的网络，可能学不到什么有用的东西。降维方法基于这样的假设，即数据的维度被人为地膨胀，而其固有维度要低得多。随着我们在自动编码器中增加层数，隐藏层的尺寸将不得不减小。如果隐藏层的大小变得小于数据的固有维度，将会导致信息丢失。解码器可以学习将隐藏层映射到特定的输入，因为层数很大并且是高度非线性的。
多层编码器和解码器的图像。下面显示了一个简单的自动编码器。


欠完整自动编码器的损失函数由下式给出:

```py
L(x，g(f(x)))=(x-g(f(x)))²
```

因为这篇文章是关于使用自动编码器降维的，我们将在 pyspark 上实现欠完整自动编码器。【spark 的开源深度学习库很少。例如英特尔的 [bigdl](https://github.com/intel-analytics/BigDL) ，雅虎的 [tensorflowonspark](https://github.com/yahoo/TensorFlowOnSpark) ，databricks 的 [spark 深度学习](https://github.com/databricks/spark-deep-learning)。
我们将使用英特尔的 bigdl。

步骤 1 安装 bigdl
如果你已经安装了 spark run `pip install --user bigdl --no-deps`否则运行`pip install --user bigdl`。在后一种情况下，pip 将安装 pyspark 和 bigdl。

第二步。必要的进口

```py
%matplotlib inline
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# some imports from bigdl
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.dataset.transformer import *
from pyspark import SparkContext
sc=(SparkContext.getOrCreate(
              conf=create_spark_conf().
              setMaster("local[4]")>
              set("spark.driver.memory","2g")))

# function to initialize the bigdl library
init_engine() 
```

第三步。加载并准备数据

```py
# bigdl provides a nice function for 
# downloading and reading mnist dataset

from bigdl.dataset import mnist
mnist_path = "mnist"
images_train, labels_train = mnist.read_data_sets(mnist_path, "train")

# mean and stddev of the pixel values

mean = np.mean(images_train)
std = np.std(images_train)

# parallelize, center and scale the images_train
rdd_images =  (sc.parallelize(images_train).
                            map(lambda features: (features - mean)/std))

print("total number of images ",rdd_images.count()) 
```

步骤 3 为模型创建函数

```py
# Parameters for training

BATCH_SIZE = 100
NUM_EPOCHS = 2

# Network Parameters
SIZE_HIDDEN = 32
# shape of the input data
SIZE_INPUT = 784 
# function for creating an autoencoder

def get_autoencoder(hidden_size, input_size):

    # Initialize a sequential type container
    module = Sequential()

    # create encoder layers
    module.add(Linear(input_size, hidden_size))
    module.add(ReLU())

    # create decoder layers
    module.add(Linear(hidden_size, input_size))
    module.add(Sigmoid())

    return(module) 
```

步骤 4 建立深度学习图表

```py
undercomplete_ae = get_autoencoder( SIZE_HIDDEN, SIZE_INPUT)

# transform dataset to rdd(Sample) from rdd(ndarray).
# Sample represents a record in the dataset. A sample 
# consists of two tensors a features tensor and a label tensor. 
# In our autoencoder features and label will be same
train_data = (rdd_images.map(lambda x:
                    Sample.from_ndarray(x.reshape(28*28),
                    x.reshape(28*28))))

# Create an Optimizer
optimizer = Optimizer(
    model = undercomplete_ae,
    training_rdd = train_data,
    criterion = MSECriterion(),
    optim_method = Adam(),
    end_trigger = MaxEpoch(NUM_EPOCHS),
    batch_size = BATCH_SIZE)

# write summary 
app_name='undercomplete_autoencoder-'+dt.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary = TrainSummary(log_dir='/tmp/bigdl_summary',
                                 app_name=app_name)

optimizer.set_train_summary(train_summary)

print("logs to saved to ",app_name) 
```

第五步训练模型

```py
# run training process
trained_UAE = optimizer.optimize() 
```

第六步根据测试数据模拟性能

```py
# let's check our model performance on the test data

(images, labels) = mnist.read_data_sets(mnist_path, "test")
rdd_test =  (sc.parallelize(images).
                    map(lambda features: ((features - 
                    mean)/std).reshape(28*28)).map(
                    lambda features: Sample.
                    from_ndarray(features, features)))

examples = trained_UAE.predict(rdd_test).take(10)
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(images[i], (28, 28)))
    a[1][i].imshow(np.reshape(examples[i], (28, 28))) 
```

![](img/3816e7a8f5c722b85e38434242c45e03.png)
正如我们从图像中看到的，重建非常接近原始输入。
**结论**:通过这篇文章，我们讨论了自动编码器如何用于降维。一开始，我们讨论了不同类型的自动编码器及其用途。后来，我们使用 intel 的 bigdl 和 pyspark 实现了一个欠完整的自动编码器。更多关于 bigdl 的教程，请访问 [bigdl 教程](https://github.com/intel-analytics/BigDL-Tutorials)

这篇文章结束了我们关于降维的系列文章。