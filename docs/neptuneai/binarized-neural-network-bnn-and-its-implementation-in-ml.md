# 二值化神经网络(BNN)及其在机器学习中的实现

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/binarized-neural-network-bnn-and-its-implementation-in-ml>

二值化神经网络(BNN)来自 Courbariaux、Hubara、Soudry、El-Yaniv 和 beng io 2016 年的一篇[论文](https://web.archive.org/web/20230214183854/https://arxiv.org/pdf/1602.02830.pdf)。它引入了一种新的方法来训练神经网络，其中权重和激活在训练时被二值化，然后用于计算梯度。

通过这种方式，可以减少内存大小，并且位运算可以提高功效。GPU 消耗巨大的功率，使得神经网络很难在低功率设备上训练。BNNs 可以将功耗降低 32 倍以上。

该论文表明，可以使用二进制矩阵乘法来减少训练时间，这使得在 MNIST 上训练 BNN 的速度提高了 7 倍，达到了接近最先进的结果。

在本文中，我们将看到二进制神经网络是如何工作的。我们将深入研究该算法，并查看实现 BNNs 的库。

## 二进制神经网络如何工作

在我们深入挖掘之前，让我们看看 BNNs 是如何工作的。

在论文中，他们使用两个函数来二进制化 x(权重/激活)的值——确定性的和随机的。

![](img/7dc241dc375bb8ef34ee805d2178b41a.png)

Eq. 1

情商。1 是确定性函数，其中 signum 函数用于实值变量。

随机函数使用硬 sigmoid:

![BNN equation](img/ae23025fc1a930a84c0cfa68391f5483.png)

Eq. 2

方程式中的 xb。1 和 Eq。2 是实值变量(权重/激活)的二进制值。1 很简单。

在 Eq 中。2

![BNN equation](img/118aa29d58a105d0309591b8e1be4f8d.png)

Eq. 3

确定性函数在大多数情况下使用，除了少数实验中随机函数用于激活。

除了二进制化权重和激活之外，BNNs 还有两个更重要的方面:

*   为了让优化器工作，您需要实值权重，所以它们在实值变量中累积。即使我们使用二进制权重/激活，我们也使用实值权重进行优化。
*   当我们使用确定性或随机函数进行二值化时，会出现另一个问题。当我们反向传播时，这些函数的导数为零，这使得整个梯度为零。所以我们可以使用饱和 STE(直通估计量)，这是之前由 Hinton 提出并由 Bengio 研究的。在饱和 STE 中，signum 的导数用 1 [{x < =1}] 代替，简单来说就是当 x < =1 时，用恒等式(1)代替导数零。所以，当 x 太大时，它抵消了梯度，因为导数为零。

![Binary weight filter](img/e6881d5ae66e8e2fc175da64e518caf9.png)

*Fig.* 1 – A *binary weight filter from the 1st convolutional layer of BNN | [Source](https://web.archive.org/web/20230214183854/https://arxiv.org/pdf/1602.02830.pdf)*

## 基于移位的批处理规范化和基于移位的 AdaMax 优化

除了常规批处理规范化和 Adamax 优化之外，还有一种替代方法。BatchNorm 和 Adam optimizer 都包含大量乘法运算。为了加快这个过程，它们被基于移位的方法所取代。这些方法使用位运算来节省时间。BNN 的论文声称，当用基于班次的批处理规范化和基于班次的 Adam 优化程序替换批处理规范化和 Adam 优化程序时，没有观察到准确度损失。

## 加速训练

BNN 论文中介绍的方法可以加速 BNNs 的 GPU 实现。与使用 cuBLAS 相比，它可以提高时间效率。

cuBLAS 是一个 CUDA 工具包库，提供 GPU 加速的基本线性代数子程序(BLAS)。

一种称为 SWAR 的方法，用于在寄存器内执行并行操作，用于加速计算。它将 32 位二进制变量连接到 32 位寄存器。

SWAR 可以在 Nvidia GPU 上仅用 6 个时钟周期评估这 32 个连接，因此理论上速度提高了 32/6 = 5.3 倍。值+1 和-1 对于执行此操作非常重要，因此我们需要将变量二进制化为这两个值。

让我们来看看一些性能统计数据:

![BNN performance stats](img/59e6bc8ebe460e9c2622cfb3864dee8a.png)

*Fig. 2 – Comparison between Baseline kernel, cuBLAS and the XNOR kernel for time and accuracy. | [Source](https://web.archive.org/web/20230214183854/https://arxiv.org/pdf/1602.02830.pdf)*

正如我们在图 2 中看到的，所有三种方法的精确度；使用 cuBLAS 库和 paper 的 XNOR 核的未优化基线核在图的第三部分是相同的。在第一部分中，将矩阵乘法时间与 8192 x 8192 x 8192 矩阵进行比较。在第二部分中，在多层感知器上推断 MNIST 的全部测试数据。我们可以清楚地看到，XNOR 内核的性能更好。在矩阵乘法的情况下，XNOR 比基线内核快 23 倍，比 cuBLAS 内核快 3.4 倍。

我们可以看到，在运行 MNIST 测试数据时，cuBLAS 和 XNOR 内核之间的差异较小。这是因为在第一层，值不是二进制的，所以基线内核用于计算，从而导致一点延迟。但这不是什么大问题，因为输入图像通常只有 3 个通道，这意味着计算量更少。

## 密码

我们来看一些实现了 BNNs 的 Github repos。

BNNs 的前两个实现包含在原始论文中，虽然一个是用 lua(torch)实现的，另一个是用 Python 实现的，但是是用 theano 实现的。

### 提诺:

[https://github . com/matthieurbarillas/binary net](https://web.archive.org/web/20230214183854/https://github.com/MatthieuCourbariaux/BinaryNet)

### 火炬:

[https://github.com/itayhubara/BinaryNet](https://web.archive.org/web/20230214183854/https://github.com/itayhubara/BinaryNet)

### PyTorch:

BNN 论文的作者之一提供了一个 pytorch 实现，包括 alexnet 二进制、resnet 二进制和 vgg 二进制等架构，具有不同的层数(resent18、resnet34、resnet50 等。)

[https://github.com/itayhubara/BinaryNet.pytorch](https://web.archive.org/web/20230214183854/https://github.com/itayhubara/BinaryNet.pytorch)

没有文档，但是代码很直观。在子目录“模型”中，实现了三个二值化网络:vgg、resnet 和 alexnet。

使用文件“data.py”向 BNN 网络发送自定义数据集。‘preprocess . py’里也有很多变换选项。

### Keras/TensorFlow:

到目前为止，我见过的最好的包之一是 [Larq](https://web.archive.org/web/20230214183854/https://larq.dev/) ，这是一个开源包，其中构建和训练二进制神经网络非常容易。

在前面讨论的包中，有可以使用的预先实现的网络。但是有了 Larq，你可以用一种非常简单的方式创建新的网络。这就像 Keras API，例如，如果你想添加一个二值化的 conv 层，而不是' tf.keras.layers.Conv2D '，你可以使用' larq.layers.Conv2D '。

关于这个包最好的一点是[文档](https://web.archive.org/web/20230214183854/https://docs.larq.dev/larq/)非常好，社区正在积极开发它，所以支持也很好。

尽管它有很棒的文档，但让我们看看文档中的一个例子，这样您就可以大致了解该库的易用性。

```py
import tensorflow as tf
import larq as lq

kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip",
              use_bias=False)

model = tf.keras.models.Sequential([
    lq.layers.QuantConv2D(128, 3,
                          kernel_quantizer="ste_sign",
                          kernel_constraint="weight_clip",
                          use_bias=False,
                          input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

    lq.layers.QuantConv2D(128, 3, padding="same", **kwargs),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

    lq.layers.QuantConv2D(256, 3, padding="same", **kwargs),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

    lq.layers.QuantConv2D(256, 3, padding="same", **kwargs),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

    lq.layers.QuantConv2D(512, 3, padding="same", **kwargs),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

    lq.layers.QuantConv2D(512, 3, padding="same", **kwargs),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
    tf.keras.layers.Flatten(),

    lq.layers.QuantDense(1024, **kwargs),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

    lq.layers.QuantDense(1024, **kwargs),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),

    lq.layers.QuantDense(10, **kwargs),
    tf.keras.layers.BatchNormalization(momentum=0.999, scale=False),
    tf.keras.layers.Activation("softmax")
])

```

**注意，如前所述，我们没有将希格诺和 STE 用于输入层。**我们来看看最终的架构。

```py
model.summary()

```

```py
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param 
=================================================================
quant_conv2d (QuantConv2D)   (None, 30, 30, 128)       3456
_________________________________________________________________
batch_normalization (BatchNo (None, 30, 30, 128)       384
_________________________________________________________________
quant_conv2d_1 (QuantConv2D) (None, 30, 30, 128)       147456
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 128)       0
_________________________________________________________________
batch_normalization_1 (Batch (None, 15, 15, 128)       384
_________________________________________________________________
quant_conv2d_2 (QuantConv2D) (None, 15, 15, 256)       294912
_________________________________________________________________
batch_normalization_2 (Batch (None, 15, 15, 256)       768
_________________________________________________________________
quant_conv2d_3 (QuantConv2D) (None, 15, 15, 256)       589824
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 256)         0
_________________________________________________________________
batch_normalization_3 (Batch (None, 7, 7, 256)         768
_________________________________________________________________
quant_conv2d_4 (QuantConv2D) (None, 7, 7, 512)         1179648
_________________________________________________________________
batch_normalization_4 (Batch (None, 7, 7, 512)         1536
_________________________________________________________________
quant_conv2d_5 (QuantConv2D) (None, 7, 7, 512)         2359296
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 3, 3, 512)         0
_________________________________________________________________
batch_normalization_5 (Batch (None, 3, 3, 512)         1536
_________________________________________________________________
flatten (Flatten)            (None, 4608)              0
_________________________________________________________________
quant_dense (QuantDense)     (None, 1024)              4718592
_________________________________________________________________
batch_normalization_6 (Batch (None, 1024)              3072
_________________________________________________________________
quant_dense_1 (QuantDense)   (None, 1024)              1048576
_________________________________________________________________
batch_normalization_7 (Batch (None, 1024)              3072
_________________________________________________________________
quant_dense_2 (QuantDense)   (None, 10)                10240
_________________________________________________________________
batch_normalization_8 (Batch (None, 10)                30
_________________________________________________________________
activation (Activation)      (None, 10)                0
=================================================================
Total params: 10,363,550
Trainable params: 10,355,850
Non-trainable params: 7,700

```

现在你可以像训练一个在 keras 中实现的普通神经网络一样训练它。

![](img/fe34857d1d7e68b4c676b5b18f16e936.png)？海王星与 [PyTorch](https://web.archive.org/web/20230214183854/https://docs.neptune.ai/essentials/integrations/deep-learning-frameworks/pytorch) 
![](img/fe34857d1d7e68b4c676b5b18f16e936.png)的融合？海王星与 [TensorFlow/Keras 的整合](https://web.archive.org/web/20230214183854/https://docs.neptune.ai/essentials/integrations/deep-learning-frameworks/tensorflow-keras)

## 应用程序

bnn 具有功率效率，因此可用于低功率器件。这是 BNNs 最大的优势之一。您可以使用 LCE(Larq 计算引擎)和 Tensorflow Lite Java 在 Android 上训练和推断神经网络，消耗更少的功率。

![BNN Tensorflow](img/df717ee0549bdc1e58be4867276839e0.png)

*Fig. 3 – Example of Image Classification using LCE Lite | [Source](https://web.archive.org/web/20230214183854/https://docs.larq.dev/compute-engine/quickstart_android/)*

你可以点击下面的[链接](https://web.archive.org/web/20230214183854/https://docs.larq.dev/compute-engine/quickstart_android/)阅读更多关于在 Android 设备上使用 BNNs 的信息。

## 结论

深度网络需要耗电的 GPU，很难在低功耗设备上训练它们。因此，二元神经网络的概念似乎很有前途。

它们消耗更少的功率而没有任何精度损失，并且可以在移动设备中用于训练 dnn。好像挺有用的！

感谢阅读。

### 参考

如果你想深入了解 BNNs，这里有一些参考资料: