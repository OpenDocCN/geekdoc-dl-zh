# PyTorch 损失函数:最终指南

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/pytorch-loss-functions>

你的神经网络可以完成很多不同的任务。无论是对数据进行分类，如将动物图片分为猫和狗，回归任务，如预测月收入，还是其他任何事情。每个任务都有不同的输出，需要不同类型的损失函数。

您配置损失函数的方式可以决定算法的性能。通过[正确配置损失函数](https://web.archive.org/web/20230304142301/https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)，你可以确保你的模型按照你想要的方式工作。

幸运的是，我们可以使用损失函数来充分利用机器学习任务。

在本文中，我们将讨论 PyTorch 中流行的损失函数，以及如何构建自定义损失函数。一旦你读完了，你应该知道为你的项目选择哪一个。

检查如何通过 Neptune + PyTorch 集成来监控您的 PyTorch 模型训练并跟踪所有模型构建元数据。

## 损失函数有哪些？

在我们进入 PyTorch 细节之前，让我们回忆一下损失函数是什么。

损失函数用于测量预测输出和提供的目标值之间的误差。损失函数告诉我们算法模型离实现预期结果有多远。“损失”一词意味着模型因未能产生预期结果而受到的惩罚。

例如，损失函数(姑且称之为 **J** )可以采用以下两个参数:

*   预测产量( **y_pred**
*   目标值( **y** )

![neural network loss
](img/88cd419a31ac33228ee0d8e7eebe6a08.png)

*Illustration of a neural network loss*

此函数将通过比较模型的预测输出和预期输出来确定模型的性能。如果 **y_pred** 和 **y** 之间的偏差很大，损失值会很高。

如果偏差很小或者值几乎相同，它将输出一个非常低的损耗值。因此，当模型在所提供的数据集上进行训练时，您需要使用一个损失函数来适当地惩罚模型。

损失函数根据算法试图解决的问题陈述而变化。

## 如何添加 PyTorch 损失函数？

PyTorch 的 **torch.nn** 模块有多个标准损失函数，你可以在你的项目中使用。

要添加它们，您需要首先导入库:

```py
import torch
import torch.nn as nn
```

接下来，定义您想要使用的损失类型。以下是定义平均绝对误差损失函数的方法:

```py
loss = nn.L1Loss()
```

添加函数后，您可以使用它来完成您的特定任务。

## PyTorch 中有哪些损失函数？

广义来说，PyTorch 中的损失[函数分为两大类:](https://web.archive.org/web/20230304142301/https://cs230.stanford.edu/blog/pytorch/)[回归损失和分类损失。](https://web.archive.org/web/20230304142301/https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0)

**回归损失函数**在模型预测连续值时使用，如人的年龄。

**分类损失函数**在模型预测离散值时使用，例如电子邮件是否为垃圾邮件。

**排名损失函数**在模型预测输入之间的相对距离时使用，例如根据电子商务搜索页面上的相关性对产品进行排名。

现在我们将探索 PyTorch 中不同类型的损失函数，以及如何使用它们:

### 1.PyTorch 平均绝对误差(L1 损失函数)

```py
torch.nn.L1Loss

```

[平均绝对误差](https://web.archive.org/web/20230304142301/https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss) (MAE)，也称为 L1 损失，计算实际值和预测值之间的绝对差的**和的平均值。**

它检查一组预测值的误差大小，而不关心它们的正负方向。如果不使用误差的绝对值，那么负值会抵消正值。

Pytorch L1 损失表示为:

**x** 代表实际值， **y** 代表预测值。

什么时候可以使用？

*   回归问题，特别是当目标变量的分布有异常值时，如与平均值相差很大的小值或大值。**它被认为对异常值更稳健。**

**例子**

```py
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)

mae_loss = nn.L1Loss()
output = mae_loss(input, target)
output.backward()

print('input: ', input)
print('target: ', target)
print('output: ', output)
```

```py
###################### OUTPUT ######################

input:  tensor([[ 0.2423,  2.0117, -0.0648, -0.0672, -0.1567],
        [-0.2198, -1.4090,  1.3972, -0.7907, -1.0242],
        [ 0.6674, -0.2657, -0.9298,  1.0873,  1.6587]], requires_grad=True)
target:  tensor([[-0.7271, -0.6048,  1.7069, -1.5939,  0.1023],
        [-0.7733, -0.7241,  0.3062,  0.9830,  0.4515],
        [-0.4787,  1.3675, -0.7110,  2.0257, -0.9578]])
output:  tensor(1.2850, grad_fn=<L1LossBackward>)
```

### 2.PyTorch 均方误差损失函数

```py
torch.nn.MSELoss

```

[均方误差(MSE)](https://web.archive.org/web/20230304142301/https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) ，也称为 L2 损失，计算实际值和预测值之间的平方差的**平均值。**

Pytorch MSE Loss 总是输出正的结果，不管实际值和预测值的符号如何。为了提高模型的准确性，您应该尝试减少 L2 损失-一个完美的值是 0.0。

平方意味着较大的错误比较小的错误产生更大的误差。如果分类器偏离 100，则误差为 10，000。如果相差 0.1，误差为 0.01。这个**惩罚犯大错**的模特，鼓励小错。

Pytorch L2 损失表示为:

**x** 代表实际值， **y** 代表预测值。

什么时候可以使用？

*   MSE 是大多数 Pytorch 回归问题的默认损失函数。

**例子**

```py
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
mse_loss = nn.MSELoss()
output = mse_loss(input, target)
output.backward()

print('input: ', input)
print('target: ', target)
print('output: ', output)

```

```py
###################### OUTPUT ######################

input:  tensor([[ 0.3177,  1.1312, -0.8966, -0.0772,  2.2488],
        [ 0.2391,  0.1840, -1.2232,  0.2017,  0.9083],
        [-0.0057, -3.0228,  0.0529,  0.4084, -0.0084]], requires_grad=True)
target:  tensor([[ 0.2767,  0.0823,  1.0074,  0.6112, -0.1848],
        [ 2.6384, -1.4199,  1.2608,  1.8084,  0.6511],
        [ 0.2333, -0.9921,  1.5340,  0.3703, -0.5324]])
output:  tensor(2.3280, grad_fn=<MseLossBackward>)
```

### 3.PyTorch 负对数似然损失函数

```py
torch.nn.NLLLoss

```

[负对数似然损失函数](https://web.archive.org/web/20230304142301/https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) (NLL)仅适用于将 softmax 函数作为输出激活层的模型。 [Softmax](https://web.archive.org/web/20230304142301/https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/) 指的是计算层中每个单元的归一化指数函数的激活函数。

Softmax 函数表示为:

该函数获取一个大小为 **N** 、**的输入向量，然后修改这些值，使每个值都在 0 和 1 之间。此外，它对输出进行归一化，使得向量的 **N** 值之和等于 1。**

NLL 使用否定的含义，因为概率(或可能性)在 0 和 1 之间变化，并且这个范围内的值的对数是负的。最终，损失值变为正值。

在 NLL 中，最小化损失函数有助于我们获得更好的输出。负对数似然是从近似最大似然估计(MLE)中检索的。这意味着我们试图最大化模型的对数似然，结果，[最小化 NLL](https://web.archive.org/web/20230304142301/https://quantivity.wordpress.com/2011/05/23/why-minimize-negative-log-likelihood/) 。

在 NLL 中，模型因以较小的概率做出正确预测而受到惩罚，因以较高的概率做出预测而受到鼓励。对数做惩罚。

NLL 不仅关心预测是否正确，还关心模型是否确定预测得分高。

Pytorch NLL 损失表示为:

其中 x 是输入，y 是目标，w 是重量，N 是批量。

什么时候可以使用？

*   多类分类问题

**例子**

```py
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)

target = torch.tensor([1, 0, 4])

m = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()
output = nll_loss(m(input), target)
output.backward()

print('input: ', input)
print('target: ', target)
print('output: ', output)
```

```py

input:  tensor([[ 1.6430, -1.1819,  0.8667, -0.5352,  0.2585],
        [ 0.8617, -0.1880, -0.3865,  0.7368, -0.5482],
        [-0.9189, -0.1265,  1.1291,  0.0155, -2.6702]], requires_grad=True)
target:  tensor([1, 0, 4])
output:  tensor(2.9472, grad_fn=<NllLossBackward>)
```

### 4.PyTorch 交叉熵损失函数

```py
torch.nn.CrossEntropyLoss

```

[该损失函数](https://web.archive.org/web/20230304142301/https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)计算一组给定事件或随机变量的两个概率分布之间的差异。

它用于计算一个分数，该分数总结了预测值和实际值之间的平均差异。为了增强模型的准确性，您应该尝试最小化得分—交叉熵得分在 0 到 1 之间，一个完美的值是 0。

其他损失函数，如平方损失，惩罚不正确的预测。 [**交叉熵**](https://web.archive.org/web/20230304142301/https://en.wikipedia.org/wiki/Cross_entropy) **对非常自信和错误的人大加惩罚。**

与负对数似然损失不同，负对数似然损失不会基于预测置信度进行惩罚，交叉熵会惩罚不正确但有把握的预测，以及正确但不太有把握的预测。

交叉熵函数有很多种变体，其中最常见的类型是**二元交叉熵(BCE)** 。BCE 损失主要用于二元分类模型；即只有两个类别的模型。

Pytorch 交叉熵损失表示为:

其中 x 是输入，y 是目标，w 是重量，C 是类别数，N 是小批量维度。

什么时候可以使用？

*   二进制分类任务，这是 Pytorch 中的默认损失函数。
*   创建有信心的模型-预测将是准确的，并且具有更高的概率。

**例子**

```py
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)

cross_entropy_loss = nn.CrossEntropyLoss()
output = cross_entropy_loss(input, target)
output.backward()

print('input: ', input)
print('target: ', target)
print('output: ', output)
```

```py

input:  tensor([[ 0.1639, -1.2095,  0.0496,  1.1746,  0.9474],
        [ 1.0429,  1.3255, -1.2967,  0.2183,  0.3562],
        [-0.1680,  0.2891,  1.9272,  2.2542,  0.1844]], requires_grad=True)
target:  tensor([4, 0, 3])
output:  tensor(1.0393, grad_fn=<NllLossBackward>)
```

### 5.PyTorch 铰链嵌入损失函数

```py
torch.nn.HingeEmbeddingLoss

```

[铰链嵌入损失](https://web.archive.org/web/20230304142301/https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html#torch.nn.HingeEmbeddingLoss)用于计算存在输入张量 **x** 和标签张量 **y** 时的损失。目标值介于{1，-1}之间，这有利于二进制分类任务。

使用铰链损失函数，只要实际类值和预测类值之间的符号存在差异，就可以给出更大的误差。这激励例子有正确的标志。

铰链嵌入损耗表示为:

什么时候可以使用？

*   分类问题，尤其是在确定两个输入是相似还是不相似时。
*   学习非线性嵌入或半监督学习任务。

**例子**

```py
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)

hinge_loss = nn.HingeEmbeddingLoss()
output = hinge_loss(input, target)
output.backward()

print('input: ', input)
print('target: ', target)
print('output: ', output)
```

```py
###################### OUTPUT ######################

input:  tensor([[ 0.1054, -0.4323, -0.0156,  0.8425,  0.1335],
        [ 1.0882, -0.9221,  1.9434,  1.8930, -1.9206],
        [ 1.5480, -1.9243, -0.8666,  0.1467,  1.8022]], requires_grad=True)
target:  tensor([[-1.0748,  0.1622, -0.4852, -0.7273,  0.4342],
        [-1.0646, -0.7334,  1.9260, -0.6870, -1.5155],
        [-0.3828, -0.4476, -0.3003,  0.6489, -2.7488]])
output:  tensor(1.2183, grad_fn=<MeanBackward0>)
```

### 6.PyTorch 边际排序损失函数

```py
torch.nn.MarginRankingLoss

```

[边际排名损失](https://web.archive.org/web/20230304142301/https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html#torch.nn.MarginRankingLoss)计算一个标准来预测输入之间的相对距离。这不同于其他损失函数，如 MSE 或交叉熵，它们学习从给定的输入集直接预测。

利用边际排序损失，只要有输入 **x1** 、 **x2** ，以及标签张量 **y** (包含 1 或-1)，就可以计算损失。

当 **y** == 1 时，第一次输入将被假定为一个较大的值。它的排名会高于第二个输入。如果 **y** == -1，则第二个输入的排名会更高。

Pytorch 利润排名损失表示为:

什么时候可以使用？

**例子**

```py
import torch
import torch.nn as nn

input_one = torch.randn(3, requires_grad=True)
input_two = torch.randn(3, requires_grad=True)
target = torch.randn(3).sign()

ranking_loss = nn.MarginRankingLoss()
output = ranking_loss(input_one, input_two, target)
output.backward()

print('input one: ', input_one)
print('input two: ', input_two)
print('target: ', target)
print('output: ', output)
```

```py

input one:  tensor([1.7669, 0.5297, 1.6898], requires_grad=True)
input two:  tensor([ 0.1008, -0.2517,  0.1402], requires_grad=True)
target:  tensor([-1., -1., -1.])
output:  tensor(1.3324, grad_fn=<MeanBackward0>)
```

### 7.PyTorch 三重边界损失函数

```py
torch.nn.TripletMarginLoss

```

[三重余量损失](https://web.archive.org/web/20230304142301/https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss)计算模型中测量三重损失的标准。通过这个损失函数，可以计算出有输入张量、 **x1** 、 **x2** 、 **x3** 以及大于零的裕量时的损失。

一个三联体由 **a** (主播) **p** (正例) **n** (反例)组成。

Pytorch 三线态余量损失表示为:

什么时候可以使用？

**例子**

```py
import torch
import torch.nn as nn
anchor = torch.randn(100, 128, requires_grad=True)
positive = torch.randn(100, 128, requires_grad=True)
negative = torch.randn(100, 128, requires_grad=True)

triplet_margin_loss = nn.TripletMarginLoss(margin=1.0, p=2)
output = triplet_margin_loss(anchor, positive, negative)
output.backward()

print('anchor: ', anchor)
print('positive: ', positive)
print('negative: ', negative)
print('output: ', output)

```

```py

anchor:  tensor([[ 0.6152, -0.2224,  2.2029,  ..., -0.6894,  0.1641,  1.7254],
        [ 1.3034, -1.0999,  0.1705,  ...,  0.4506, -0.2095, -0.8019],
        [-0.1638, -0.2643,  1.5279,  ..., -0.3873,  0.9648, -0.2975],
        ...,
        [-1.5240,  0.4353,  0.3575,  ...,  0.3086, -0.8936,  1.7542],
        [-1.8443, -2.0940, -0.1264,  ..., -0.6701, -1.7227,  0.6539],
        [-3.3725, -0.4695, -0.2689,  ...,  2.6315, -1.3222, -0.9542]],
       requires_grad=True)
positive:  tensor([[-0.4267, -0.1484, -0.9081,  ...,  0.3615,  0.6648,  0.3271],
        [-0.0404,  1.2644, -1.0385,  ..., -0.1272,  0.8937,  1.9377],
        [-1.2159, -0.7165, -0.0301,  ..., -0.3568, -0.9472,  0.0750],
        ...,
        [ 0.2893,  1.7894, -0.0040,  ...,  2.0052, -3.3667,  0.5894],
        [-1.5308,  0.5288,  0.5351,  ...,  0.8661, -0.9393, -0.5939],
        [ 0.0709, -0.4492, -0.9036,  ...,  0.2101, -0.8306, -0.6935]],
       requires_grad=True)
negative:  tensor([[-1.8089, -1.3162, -1.7045,  ...,  1.7220,  1.6008,  0.5585],
        [-0.4567,  0.3363, -1.2184,  ..., -2.3124,  0.7193,  0.2762],
        [-0.8471,  0.7779,  0.1627,  ..., -0.8704,  1.4201,  1.2366],
        ...,
        [-1.9165,  1.7768, -1.9975,  ..., -0.2091, -0.7073,  2.4570],
        [-1.7506,  0.4662,  0.9482,  ...,  0.0916, -0.2020, -0.5102],
        [-0.7463, -1.9737,  1.3279,  ...,  0.1629, -0.3693, -0.6008]],
       requires_grad=True)
output:  tensor(1.0755, grad_fn=<MeanBackward0>)

```

### 8.PyTorch Kullback-Leibler 散度损失函数

```py
torch.nn.KLDivLoss

```

[kull back-lei bler 散度](https://web.archive.org/web/20230304142301/https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss)，简称 KL 散度，计算两个概率分布之间的差异。

使用此损失函数，您可以计算在预测概率分布用于估计预期目标概率分布的情况下损失的信息量(以位表示)。

它的输出告诉你两个概率分布的**接近度。如果预测的概率分布与真实的概率分布相差甚远，就会导致巨大的损失。如果 KL 散度的值为零，则意味着概率分布是相同的。**

KL 散度的表现就像交叉熵损失一样，在如何处理预测和实际概率方面有一个关键的区别。交叉熵根据预测的置信度惩罚模型，而 KL 散度不会。KL 散度仅评估概率分布预测如何不同于基本事实的分布。

KL 发散损失表示为:

**x** 代表真实标签的概率， **y** 代表预测标签的概率。

什么时候可以使用？

*   逼近复杂函数
*   多类分类任务
*   如果您希望确保预测的分布类似于定型数据的分布

**例子**

```py
import torch
import torch.nn as nn

input = torch.randn(2, 3, requires_grad=True)
target = torch.randn(2, 3)

kl_loss = nn.KLDivLoss(reduction = 'batchmean')
output = kl_loss(input, target)
output.backward()

print('input: ', input)
print('target: ', target)
print('output: ', output)
```

```py
###################### OUTPUT ######################

input:  tensor([[ 1.4676, -1.5014, -1.5201],
        [ 1.8420, -0.8228, -0.3931]], requires_grad=True)
target:  tensor([[ 0.0300, -1.7714,  0.8712],
        [-1.7118,  0.9312, -1.9843]])
output:  tensor(0.8774, grad_fn=<DivBackward0>)
```

## 如何在 PyTorch 中创建自定义损失函数？

PyTorch 允许您创建自己的自定义损失函数，并在项目中实现。

以下是你如何创建自己简单的交叉熵损失函数。

### 将自定义损失函数创建为 python 函数

```py
def myCustomLoss(my_outputs, my_labels):

    my_batch_size = my_outputs.size()[0]

    my_outputs = F.log_softmax(my_outputs, dim=1)

    my_outputs = my_outputs[range(my_batch_size), my_labels]

    return -torch.sum(my_outputs)/number_examples
```

还可以创建其他[高级 PyTorch 自定义损耗函数](https://web.archive.org/web/20230304142301/https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch/comments)。

### 使用类定义创建自定义损失函数

让我们修改计算两个样本之间相似性的 Dice 系数，作为二元分类问题的损失函数:

```py
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice
```

## 如何监控 PyTorch 损失函数？

很明显，在训练模型时，需要关注损失函数值，以跟踪模型的性能。随着损失值不断降低，模型不断变好。我们有很多方法可以做到这一点。让我们来看看它们。

为此，我们将训练一个在 PyTorch 中创建的简单神经网络，该网络将对著名的[虹膜数据集](https://web.archive.org/web/20230304142301/https://www.kaggle.com/datasets/uciml/iris)执行分类。

为获取数据集进行必要的导入。

```py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

```

正在加载数据集。

```py
iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

```

对数据集进行缩放，使均值=0，方差=1，可以快速收敛模型。

```py
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

```

将数据集以 80:20 的比例分成训练和测试。

```py
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

```

为我们的神经网络及其训练做必要的导入。

```py
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

```

定义我们的网络。

```py
class PyTorch_NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PyTorch_NN, self).__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.hidden_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = F.softmax(self.output_layer(x), dim=1)
        return x

```

定义用于获得精确度和训练网络的函数。

```py
def get_accuracy(pred_arr,original_arr):
    pred_arr = pred_arr.detach().numpy()
    original_arr = original_arr.numpy()
    final_pred= []

    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0

    for i in range(len(original_arr)):
        if final_pred[i] == original_arr[i]:
            count+=1
    return count/len(final_pred)*100

def train_network(model, optimizer, criterion, X_train, y_train, X_test, y_test, num_epochs):
    train_loss=[]
    train_accuracy=[]
    test_accuracy=[]

    for epoch in range(num_epochs):

        output_train = model(X_train)

        train_accuracy.append(get_accuracy(output_train, y_train))

        loss = criterion(output_train, y_train)
        train_loss.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            output_test = model(X_test)
            test_accuracy.append(get_accuracy(output_test, y_test))

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {sum(train_accuracy)/len(train_accuracy):.2f}, Test Accuracy: {sum(test_accuracy)/len(test_accuracy):.2f}")

    return train_loss, train_accuracy, test_accuracy

```

创建模型、优化器和损失函数对象。

```py
input_dim  = 4
output_dim = 3
learning_rate = 0.01

model = PyTorch_NN(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

```

### 1.监视笔记本电脑中 PyTorch 的丢失

现在，您一定注意到了 train_network 函数中的打印语句，用于监控损失和准确性。这是做这件事的一种方法。

```py
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

train_loss, train_accuracy, test_accuracy = train_network(model=model, optimizer=optimizer, criterion=criterion, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_epochs=100)

```

我们得到这样的输出。

如果需要，我们也可以使用 Matplotlib 绘制这些值。

```py
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 6), sharex=True)

ax1.plot(train_accuracy)
ax1.set_ylabel("training accuracy")

ax2.plot(train_loss)
ax2.set_ylabel("training loss")

ax3.plot(test_accuracy)
ax3.set_ylabel("test accuracy")

ax3.set_xlabel("epochs")

```

我们会看到这样一个图表，表明损失和准确性之间的相关性。

这个方法不错，而且行之有效。但是我们必须记住，我们的问题陈述和模型越复杂，就需要越复杂的监控技术。

### 2.使用 neptune.ai 监控 PyTorch 损失

监控度量标准的一个更简单的方法是将它们记录在一个类似 Neptune 的服务中，并专注于更重要的任务，比如构建和训练模型。

为此，我们只需要遵循几个小步骤。

*注:最新的代码示例请参考 [Neptune-PyTorch 集成文档](https://web.archive.org/web/20230304142301/https://docs.neptune.ai/integrations/pytorch/)。*

首先，让我们[安装需要的东西](https://web.archive.org/web/20230304142301/https://docs.neptune.ai/setup/installation/)。

```py
pip install neptune-client
```

现在让我们[初始化一次海王星运行](https://web.archive.org/web/20230304142301/https://docs.neptune.ai/logging/new_run/)。

```py
import neptune.new as neptune

run = neptune.init_run()
```

我们还可以分配配置变量，例如:

```py
run["config/model"] = type(model).__name__
run["config/criterion"] = type(criterion).__name__
run["config/optimizer"] = type(optimizer).__name__
```

这是它在用户界面上的样子。

最后，我们可以通过在 train_network 函数中添加几行来记录我们的损失。请注意与“运行”相关的行。

```py
def train_network(model, optimizer, criterion, X_train, y_train, X_test, y_test, num_epochs):
    train_loss=[]
    train_accuracy=[]
    test_accuracy=[]

    for epoch in range(num_epochs):

        output_train = model(X_train)

        acc = get_accuracy(output_train, y_train)
        train_accuracy.append(acc)
        run["training/epoch/accuracy"].log(acc)

        loss = criterion(output_train, y_train)
        run["training/epoch/loss"].log(loss)

        train_loss.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            output_test = model(X_test)
            test_acc = get_accuracy(output_test, y_test)
            test_accuracy.append(test_acc)

            run["test/epoch/accuracy"].log(test_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {sum(train_accuracy)/len(train_accuracy):.2f}, Test Accuracy: {sum(test_accuracy)/len(test_accuracy):.2f}")

    return train_loss, train_accuracy, test_accuracy

```

这是我们在仪表盘上看到的。绝对无缝。

你可以在 Neptune 应用程序中查看这次运行[。不用说，你可以用任何损失函数做到这一点。](https://web.archive.org/web/20230304142301/https://app.neptune.ai/jhabhishek3797/PyTorch-loss-functions/experiments?split=bth&dash=charts&viewId=standard-view)

*注:最新的代码示例请参考 [Neptune-PyTorch 集成文档](https://web.archive.org/web/20230304142301/https://docs.neptune.ai/integrations/pytorch/)。*

## 最后的想法

我们研究了 PyTorch 中最常见的损失函数。您可以选择适合您项目的任何函数，或者创建您自己的自定义函数。

希望这篇文章可以作为你在机器学习任务中使用 PyTorch 损失函数的快速入门指南。

如果你想更深入地了解这个主题或了解其他损失函数，可以访问 [PyTorch 官方文档](https://web.archive.org/web/20230304142301/https://pytorch.org/docs/stable/nn.html#loss-functions)。