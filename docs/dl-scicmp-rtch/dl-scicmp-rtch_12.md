# 9  损失函数

> 原文：[`skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/loss_functions.html`](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/loss_functions.html)

损失函数的概念对于机器学习至关重要。在任何迭代中，当前的损失值表示估计值与目标之间的距离。然后，它被用来更新参数，以减少损失的方向。

在我们的应用示例中，我们已经使用了一个损失函数：均方误差，手动计算如下

```r
library(torch)

loss <- (y_pred - y)$pow(2)$sum()
```

*正如你所期望的，这又是一个不需要这种手动努力的地方。

在我们重新构建运行示例之前的最后一章概念中，我们想要讨论两件事：首先，如何使用`torch`的内置损失函数。其次，选择哪个函数。

## 9.1 `torch`损失函数

在`torch`中，损失函数以`nn_`或`nnf_`开头。

使用`nnf_`，你直接*调用一个函数*。相应地，它的参数（估计值和目标）都是张量。例如，这里有一个`nnf_mse_loss()`，它是我们手动编写的内置函数的类似物：

```r
nnf_mse_loss(torch_ones(2, 2), torch_zeros(2, 2) + 0.1)
```

```r
torch_tensor
0.81
[ CPUFloatType{} ]
```

与`nn_`相反，你创建一个对象：

```r
l <- nn_mse_loss()
```

*这个对象可以对张量进行调用，以产生所需的损失：

```r
l(torch_ones(2, 2),torch_zeros(2, 2) + 0.1)
```

```r
torch_tensor
0.81
[ CPUFloatType{} ]
```

选择对象还是函数主要是一个偏好和上下文的问题。在较大的模型中，你可能会结合几个损失函数，然后，创建损失对象可以导致更模块化和更易于维护的代码。在这本书中，除非有充分的理由这样做，否则我主要会使用第一种方法。

接下来是第二个问题。
  
## 9.2 应该选择哪种损失函数？

在深度学习或机器学习整体中，大多数应用的目标是做一件（或两件）事情：预测一个数值，或估计一个概率。我们运行示例中的回归任务做前者；现实世界的应用可能预测温度、推断员工流失或预测销售。在第二组中，原型任务是*分类*。为了根据其最显著的内容对图像进行分类，我们实际上计算了相应的概率。然后，当“狗”的概率为 0.7，而“猫”的概率为 0.3 时，我们说它是一只狗。

### 9.2.1 最大似然

在分类和回归中，最常用的损失函数都是基于*最大似然*原理构建的。最大似然意味着：我们希望以这种方式选择模型参数，即我们观察到的或可能观察到的*数据*是最有可能的。这个原理不仅“只是”基本的，而且直观上也很吸引人。想象一个简单的例子。

假设我们有值 7.1、22.14 和 11.3，并且我们知道基础过程遵循正态分布。那么，这些数据更有可能是通过均值 14 和标准差 7 的分布生成的，而不是均值 20 和标准差 1 的分布生成的。

### 9.2.2 回归

在回归（隐含地假设目标分布为正态¹）中，为了最大化似然性，我们只需继续使用均方误差——我们一直在计算的那个损失。最大似然估计器具有各种理想的统计特性。然而，在具体应用中，可能有理由使用不同的估计器。

例如，假设一个数据集有异常值，由于某种原因，预测值和目标值被发现有显著偏差。均方误差将赋予这些异常值很高的权重。在这种情况下，可能的替代方案是平均绝对误差（`nnf_l1_loss()`）和光滑 L1 损失（`nn_smooth_l1_loss()`）。后者是一种混合类型，默认情况下计算绝对（L1）误差，但当绝对误差非常小的时候，会切换到平方（L2）误差。

### 9.2.3 分类

在分类中，我们是在比较两个 *分布*。估计值是一个概率，目标也可以被视为一个。从这个角度来看，最大似然估计等同于最小化 Kullback-Leibler 散度（KL 散度）。

KL 散度是衡量两个分布差异的度量。它取决于两个因素：数据的似然性，由某些数据生成过程确定，以及模型下数据的似然性。然而，在机器学习场景中，我们只关心后者。在这种情况下，要最小化的标准简化为两个分布之间的 *交叉熵*。交叉熵损失正是分类任务中常用的一种。

在 `torch` 中，有几种损失函数的变体可以计算交叉熵。关于这个主题，有一个快速参考表会很方便；所以这里有一个快速查找表（表 9.1 简化了相当长的函数名；见 表 9.2 以获取映射）：

表 9.1：根据它们处理的数据类型（二元 vs. 多类）和预期输入（原始分数、概率或对数概率）的损失函数。

|  | **数据** |  | **输入** |  |  |
| --- | --- | --- | --- | --- | --- |
|  | 二元 | 多类 | 原始分数 | 概率 | 对数概率 |
| *BCeL* | 是 |  | 是 |  |  |
| *Ce* |  | 是 | 是 |  |  |
| *BCe* | 是 |  |  | 是 |  |
| *Nll* |  | 是 |  |  | 是 |

表 9.2：用于引用 `torch` 损失函数的缩写。

| *BCeL* | `nnf_binary_cross_entropy_with_logits()` |
| --- | --- |
| *Ce* | `nnf_cross_entropy()` |
| *BCe* | `nnf_binary_cross_entropy()` |
| *Nll* | `nnf_nll_loss()` |

要选择适用于您的用例的函数，有两个因素需要考虑。

首先，是否只有两种可能的类别（“狗 vs. 猫”，“有人在场 / 没有人在场”，等等），还是有几个？

其次，估计值的类型是什么？它们是原始分数（理论上，任何介于正负无穷之间的值）吗？是概率（介于 0 和 1 之间的值）？或者（最后）是 log 概率，即应用了对数的概率？（在最后一种情况下，所有值都应该是负数或等于零。）

#### 9.2.3.1 二元数据

从二元数据开始，我们的示例分类向量是一系列零和一。在考虑概率时，最直观的想法是将一视为存在，零视为所讨论的类别之一（例如猫或非猫）不存在。

```r
target <- torch_tensor(c(1, 0, 0, 1, 1))
```

*原始分数可以是任何东西。例如：

```r
unnormalized_estimate <-
 torch_tensor(c(3, 2.7, -1.2, 7.7, 1.9))
```

*要将这些转换为概率，我们只需要将它们传递给`nnf_sigmoid()`。`nnf_sigmoid()`将其参数压缩到零和一之间的值：

```r
probability_estimate <- nnf_sigmoid(unnormalized_estimate)
probability_estimate
```

```r
torch_tensor
 0.9526
 0.9370
 0.2315
 0.9995
 0.8699
[ CPUFloatType{5} ]
```

从上面的表格中，我们可以看到，给定`unnormalized_estimate`和`probability_estimate`，我们可以将它们两者都作为损失函数的输入——但我们必须选择合适的一个。只要我们做到了这一点，两种情况下的输出都必须是相同的。

让我们看看（首先是原始分数）：

```r
nnf_binary_cross_entropy_with_logits(
 unnormalized_estimate, target
)
```

```r
torch_tensor
0.643351
[ CPUFloatType{} ]
```

现在，概率：

```r
nnf_binary_cross_entropy(probability_estimate, target)
```

```r
torch_tensor
0.643351
[ CPUFloatType{} ]
```

这正如预期的那样。这在实践中意味着什么？这意味着当我们为二元分类构建模型，并且最终层计算未归一化的分数时，我们不需要附加 sigmoid 层来获得概率。我们只需在训练网络时调用`nnf_binary_cross_entropy_with_logits()`即可。实际上，这样做是首选方式，这也得益于数值稳定性。
  
#### 9.2.3.2 多类数据

接下来是关于多类数据，现在最直观的框架是在（几个）*类别*的术语下，而不是单个类别的存在或不存在。将类别视为类别索引（可能是索引某个查找表）。从技术上来说，类别从 1 开始：

```r
target <- torch_tensor(c(2, 1, 3, 1, 3), dtype = torch_long())
```

*在多类场景中，原始分数是一个二维张量。每一行包含一个观察值的分数，每一列对应一个类别。以下是原始估计可能的样子：

```r
unnormalized_estimate <- torch_tensor(
 rbind(c(1.2, 7.7, -1),
 c(1.2, -2.1, -1),
 c(0.2, -0.7, 2.5),
 c(0, -0.3, -1),
 c(1.2, 0.1, 3.2)
 )
)
```

*根据上述表格，给定这个估计，我们应该调用`nnf_cross_entropy()`（我们将在下面的结果比较中这样做）。

所以这是第一个选项，它的工作方式与二元数据完全一样。对于第二个，有一个额外的步骤。

首先，我们再次使用`nnf_softmax()`将原始分数转换为概率。对于大多数实际用途，`nnf_softmax()`可以被视为`nnf_sigmoid()`的多类等效。严格来说，它们的效果并不相同。简而言之，`nnf_sigmoid()`对低分和高分值同等对待，而`nnf_softmax()`会加剧最高分与其他分数之间的距离（“赢家通吃”）。

```r
probability_estimate <- nnf_softmax(unnormalized_estimate,
 dim = 2
)
probability_estimate
```

```r
torch_tensor
 0.0015  0.9983  0.0002
 0.8713  0.0321  0.0965
 0.0879  0.0357  0.8764
 0.4742  0.3513  0.1745
 0.1147  0.0382  0.8472
[ CPUFloatType{5,3} ]
```

第二步，在二进制情况下不需要的这一步，是将概率转换为对数概率。在我们的例子中，这可以通过对刚刚计算出的`probability_estimate`调用`torch_log()`来实现。或者，这两个步骤可以一起由`nnf_log_softmax()`处理：

```r
logprob_estimate <- nnf_log_softmax(unnormalized_estimate,
 dim = 2
)
logprob_estimate
```

```r
torch_tensor
-6.5017 -0.0017 -8.7017
-0.1377 -3.4377 -2.3377
-2.4319 -3.3319 -0.1319
-0.7461 -1.0461 -1.7461
-2.1658 -3.2658 -0.1658
[ CPUFloatType{5,3} ]
```

现在我们有了两种可能形式的估计，我们可以再次比较适用损失函数的结果。首先，`nnf_cross_entropy()`在原始分数上：

```r
nnf_cross_entropy(unnormalized_estimate, target)
```

```r
torch_tensor
0.23665
[ CPUFloatType{} ]
```

其次，`nnf_nll_loss()`在对数概率上：

```r
nnf_nll_loss(logprob_estimate, target)
```

```r
torch_tensor
0.23665
[ CPUFloatType{} ]
```

在应用方面，对于二进制情况所说的也适用于这里：在多类分类网络中，在最后不需要有 softmax 层。

在结束这一章之前，让我们回答一个可能出现在脑海中的问题。二进制分类不是多类设置的一个子类型吗？在这种情况下，我们不应该，无论选择哪种方法，都会得到相同的结果吗？
  
#### 9.2.3.3 检查：二进制数据，多类方法

让我们看看。我们再次使用上面提到的二进制分类场景。这里它是这样的：

```r
target <- torch_tensor(c(1, 0, 0, 1, 1))

unnormalized_estimate <- 
 torch_tensor(c(3, 2.7, -1.2, 7.7, 1.9))

probability_estimate <- nnf_sigmoid(unnormalized_estimate)

nnf_binary_cross_entropy(probability_estimate, target)
```

```r
torch_tensor
0.64335
[ CPUFloatType{} ]
```

我们希望以多类方式做事能得到相同的价值。我们已经有概率（即`probability_estimate`）；我们只需要将它们放入`nnf_nll_loss()`期望的“按类别观察”格式：

```r
# logits
multiclass_probability <- torch_tensor(rbind(
 c(1 - 0.9526, 0.9526),
 c(1 - 0.9370, 0.9370),
 c(1 - 0.2315, 0.2315),
 c(1 - 0.9995, 0.9995),
 c(1 - 0.8699, 0.8699)
))
```

*现在，我们仍然想要应用对数函数。还有一件事需要注意：在二进制设置中，类别被编码为概率（要么是 0 要么是 1）；现在，我们正在处理索引。这意味着我们需要将`target`张量中的值加 1：

```r
target <- target + 1
```

*最后，我们可以调用`nnf_nll_loss()`：

```r
nnf_nll_loss(
 torch_log(multiclass_probability),
 target$to(dtype = torch_long())
)
```

```r
torch_tensor
0.643275
[ CPUFloatType{} ]
```

我们做到了。结果确实是一样的。
  
 * *

1.  对于那些假设看起来不太可能的情况，提供了分布适当的损失函数（例如，泊松负对数似然，作为`nnf_poisson_nll_loss()`可用。↩︎

