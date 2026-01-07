# 6  从零开始构建神经网络

> 原文：[`skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/network_1.html`](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/network_1.html)

在本章中，我们将解决一个回归任务。但是等等——不是 `lm` 方式。我们将构建一个真正的神经网络，仅使用张量（不言而喻，是 `autograd` 启用的）。当然，这并不是你将来使用 `torch` 的方式；但这并不意味着这是一个无用的努力。相反。在了解了原始机制之后，你将能够更加欣赏 `torch` 为你节省的辛勤工作。更重要的是，理解基础知识将是对深度学习被意外地视为某种“魔法”的诱惑的有效解毒剂。这只是一些矩阵运算；一个人必须学会如何编排它们。

让我们从构建一个能够执行回归的网络所需的东西开始。

## 6.1 理念

简而言之，网络是从输入到输出的 *函数*。因此，一个合适的函数就是我们正在寻找的。

为了找到它，让我们首先将回归视为 *线性* 回归。线性回归所做的就是乘法和加法。对于每个自变量，都有一个 *系数* 乘以它。除此之外，还有一个所谓的 *偏置* 项在最后被加上。（在二维中，回归系数和偏置对应于回归线的斜率和 x 轴截距。）

想想看，乘法和加法是我们可以用张量做的事情——甚至可以说它们就是为了这个目的而设计的。让我们举一个例子，其中输入数据由一百个观测值组成，每个观测值有三个特征。例如：

```r
library(torch)

x <- torch_randn(100, 3)
x$size()
```

```r
[1] 100   3
```

为了存储应该乘以 `x` 的每个特征的系数，我们需要一个长度为 3 的列向量，这是特征的数目。或者，为了准备我们很快将要做的修改，这可以是一个列长为三的矩阵，即一个有三行矩阵。它应该有多少列？让我们假设我们想要预测一个单个输出特征。在这种情况下，矩阵的大小应该是 3 x 1。

这里有一个合适的候选者，随机初始化。注意张量是如何通过 `requires_grad = TRUE` 创建的，因为它代表了一个我们希望网络能够 *学习* 的参数。

```r
w <- torch_randn(3, 1, requires_grad = TRUE)
```

*偏置张量的大小必须是 1 x 1：

```r
b <- torch_zeros(1, 1, requires_grad = TRUE)
```

*现在，我们可以通过将数据与权重矩阵 `w` 相乘并加上偏置 `b` 来得到一个“预测”：

```r
y <- x$matmul(w) + b
print(y, n = 10)
```

```r
torch_tensor
-2.1600
-3.3244
 0.6046
 0.4472
-0.4971
-0.0530
 5.1259
-1.1595
-0.5960
-1.4584
... [the output was truncated (use n=-1 to disable)]
[ CPUFloatType{100,1} ][ grad_fn = <AddBackward0> ]
```

用数学符号表示，我们在这里实现的是以下函数：

$$ f(\mathbf{X}) = \mathbf{X}\mathbf{W} + \mathbf{b} $$

这与神经网络有什么关系？
  
## 6.2 层

回到神经网络术语，我们在这里原型化了具有单个层的网络的行为：输出层。然而，单层网络并不是你感兴趣构建的类型——为什么你要这样做，当你可以简单地做线性回归呢？事实上，神经网络的一个定义特征是它们能够无限（理论上）地链接多个层。在这些层中，除了输出层之外的所有层都可以被称为“隐藏层”，尽管从使用像 `torch` 这样的深度学习框架的人的角度来看，它们其实并不那么“隐藏”。

假设我们想让我们的网络有一个隐藏层。它的大小，即它拥有的单元数量，将是确定网络能力的一个重要因素。这个数字反映在我们创建的权重矩阵中：一个有八个单元的层需要一个有八个列的权重矩阵。

```r
w1 <- torch_randn(3, 8, requires_grad = TRUE)
```

*每个单元也有自己的偏置值。

```r
b1 <- torch_zeros(1, 8, requires_grad = TRUE)
```

就像我们之前看到的，隐藏层会将接收到的输入乘以权重并加上偏置。也就是说，它应用了上面显示的函数 $f$。然后，再应用另一个函数。这个函数接收来自隐藏层的输入并产生最终的输出。简而言之，这里发生的是函数组合：调用第二个函数 $g$，整体变换是 $g(f(\mathbf{X}))$，或者 $g \circ f$。

为了让 $g$ 产生与上面单层架构类似的单个输出，其权重矩阵必须将八个列的隐藏层映射到单个列。也就是说，`w2` 看起来是这样的：

```r
w2 <- torch_randn(8, 1, requires_grad = TRUE)
```

*偏置 `b2` 是一个单一值，就像 `b1`：

```r
b2 <- torch_randn(1, 1, requires_grad = TRUE)
```

*当然，没有必要只停留在单个隐藏层，一旦我们构建了完整的设备，请随时尝试代码。但首先，我们需要添加一些其他类型的组件。首先，在我们的最新架构中，我们正在链式或组合函数——这是好的。但所有这些函数都在做加法和乘法，这意味着它们是线性的。然而，神经网络的强大之处通常与非线性相关。为什么？
  
## 6.3 激活函数

想象一下，如果我们有一个有三层的网络，并且每一层所做的只是将其输入乘以其权重矩阵。（有偏置项实际上并没有改变什么。但它使例子更复杂，所以我们“抽象化”它。）

这给我们带来了一系列矩阵乘法：$f(\mathbf{X}) = ((\mathbf{X} \mathbf{W}_1)\mathbf{W}_2)\mathbf{W}_3$。现在，这可以被重新排列，使得所有权重矩阵在应用到 $\mathbf{X}$ 之前都乘在一起：$f(\mathbf{X}) = \mathbf{X} (\mathbf{W}_1\mathbf{W}_2\mathbf{W}_3)$。因此，这个三层网络可以被简化为一个单层网络，其中 $f(\mathbf{X}) = \mathbf{X} \mathbf{W}_4$。现在，我们已经失去了与深度神经网络相关的所有优势。

这就是激活函数，有时也称为“非线性”，发挥作用的地方。它们引入了无法通过矩阵乘法建模的非线性操作。从历史上看，典型的激活函数是`sigmoid`，它今天仍然非常重要。它的构成作用是将输入挤压在零和一之间，得到一个可以解释为概率的值。但在回归中，这通常不是我们想要的，对于大多数隐藏层也是如此。

相反，网络中最常用的激活函数是所谓的*ReLU*，或称为修正线性单元。这是一个相当直接的长名称：所有负值都被设置为零。在`torch`中，可以使用`relu()`函数实现这一点：

```r
t <- torch_tensor(c(-2, 1, 5, -7))
t$relu()
```

```r
torch_tensor
 0
 1
 5
 0
[ CPUFloatType{4} ]
```

为什么这会是非线性呢？线性函数的一个标准是，当你有两个输入时，无论是先加和它们然后应用变换，还是先独立地对两个输入应用变换然后相加，结果都应该是相同的。但是，对于 ReLU 来说，这并不成立：

```r
t1 <- torch_tensor(c(1, 2, 3))
t2 <- torch_tensor(c(1, -2, 3))

t1$add(t2)$relu()
```

```r
torch_tensor
 2
 0
 6
[ CPUFloatType{3} ]
```

```r
t1_clamped <- t1$relu()
t2_clamped <- t2$relu()

t1_clamped$add(t2_clamped)
```

```r
torch_tensor
 2
 2
 6
[ CPUFloatType{3} ]
```

结果并不相同。

总结到目前为止，我们讨论了如何编写层和激活函数。在我们能够构建完整的网络之前，还有一个概念需要讨论。这就是损失函数。
  
## 6.4 损失函数

概括地说，损失是我们离目标有多远的度量。当我们最小化一个函数，就像我们在上一章中所做的那样，这就是当前函数值和它可能取的最小值之间的差异。在神经网络中，我们可以自由选择一个合适的损失函数，只要它符合我们的任务。对于回归类型的任务，这通常会是均方误差（MSE），尽管不一定是这样。例如，可能会有使用平均绝对误差的理由。

在`torch`中，计算均方误差是一行代码：

```r
y <- torch_randn(5)
y_pred <- y + 0.01

loss <- (y_pred - y)$pow(2)$mean()

loss
```

```r
torch_tensor
9.99999e-05
[ CPUFloatType{} ]
```

一旦我们有了损失，我们就能更新权重，减去其梯度的部分。我们已经在上一章中看到了如何做这件事，并且很快还会再次看到。

现在我们将讨论的各个部分放在一起。*  *## 6.5 实现

我们将其分为三个部分。这样，当我们后来重构单个组件以利用更高级的`torch`功能时，将更容易看到封装和模块化正在发生的区域。

### 6.5.1 生成随机数据

我们的示例数据由一百个观测值组成。输入`x`有三个特征；目标`y`只有一个。`y`是由`x`生成的，但添加了一些噪声。

```r
library(torch)

# input dimensionality (number of input features)
d_in <- 3
# number of observations in training set
n <- 100

x <- torch_randn(n, d_in)
coefs <- c(0.2, -1.3, -0.5)
y <- x$matmul(coefs)$unsqueeze(2) + torch_randn(n, 1)
```

*接下来是网络。*  *### 6.5.2 构建网络

网络有两个层：一个隐藏层和一个输出层。这意味着我们需要两个权重矩阵和两个偏置张量。没有特殊的原因，这里的隐藏层有 32 个单元：

```r
# dimensionality of hidden layer
d_hidden <- 32
# output dimensionality (number of predicted features)
d_out <- 1

# weights connecting input to hidden layer
w1 <- torch_randn(d_in, d_hidden, requires_grad = TRUE)
# weights connecting hidden to output layer
w2 <- torch_randn(d_hidden, d_out, requires_grad = TRUE)

# hidden layer bias
b1 <- torch_zeros(1, d_hidden, requires_grad = TRUE)
# output layer bias
b2 <- torch_zeros(1, d_out, requires_grad = TRUE)
```

*使用当前值——随机初始化的结果——这些权重和偏置不会有多大用处。是时候训练网络了。*  *### 6.5.3 训练网络

训练网络意味着将输入通过其层传递，计算损失，并调整参数（权重和偏置）以改善预测。我们不断重复这些活动，直到性能看起来足够好（在现实应用中，这需要非常仔细地定义）。技术上，这些步骤的每次重复应用都称为一个*epoch*。

就像函数最小化一样，决定合适的学习率（减去梯度的分数）需要一些实验。

观察下面的训练循环，你会看到，从逻辑上讲，它由四个部分组成：

+   进行正向传播，得到网络的预测（如果你不喜欢一行代码，可以自由地将其拆分）；

+   计算损失（这同样是一条命令行指令——我们只是添加了一些日志记录）；

+   让*autograd*计算损失相对于参数的梯度；以及

+   相应地更新参数（再次注意，将整个操作包裹在`with_no_grad()`中，并在每次迭代中将`grad`字段归零）。

```r
learning_rate <- 1e-4

### training loop ----------------------------------------

for (t in 1:200) {

 ### -------- Forward pass --------

 y_pred <- x$mm(w1)$add(b1)$relu()$mm(w2)$add(b2)

 ### -------- Compute loss -------- 
 loss <- (y_pred - y)$pow(2)$mean()
 if (t %% 10 == 0)
 cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")

 ### -------- Backpropagation --------

 # compute gradient of loss w.r.t. all tensors with
 # requires_grad = TRUE
 loss$backward()

 ### -------- Update weights -------- 

 # Wrap in with_no_grad() because this is a part we don't 
 # want to record for automatic gradient computation
 with_no_grad({
 w1 <- w1$sub_(learning_rate * w1$grad)
 w2 <- w2$sub_(learning_rate * w2$grad)
 b1 <- b1$sub_(learning_rate * b1$grad)
 b2 <- b2$sub_(learning_rate * b2$grad) 

 # Zero gradients after every pass, as they'd
 # accumulate otherwise
 w1$grad$zero_()
 w2$grad$zero_()
 b1$grad$zero_()
 b2$grad$zero_() 
 })

}
```

```r
Epoch: 10 Loss: 24.92771
Epoch: 20 Loss: 23.56143
Epoch: 30 Loss: 22.3069
Epoch: 40 Loss: 21.14102
Epoch: 50 Loss: 20.05027
Epoch: 60 Loss: 19.02925
Epoch: 70 Loss: 18.07328
Epoch: 80 Loss: 17.16819
Epoch: 90 Loss: 16.31367
Epoch: 100 Loss: 15.51261
Epoch: 110 Loss: 14.76012
Epoch: 120 Loss: 14.05348
Epoch: 130 Loss: 13.38944
Epoch: 140 Loss: 12.77219
Epoch: 150 Loss: 12.19302
Epoch: 160 Loss: 11.64823
Epoch: 170 Loss: 11.13535
Epoch: 180 Loss: 10.65219
Epoch: 190 Loss: 10.19666
Epoch: 200 Loss: 9.766989
```

损失最初下降得很快，然后，不再那么快了。但这个例子并不是为了展示出色的性能；其目的是展示构建一个“真正的”神经网络需要多少行代码。

现在，层、损失、参数更新——所有这些都还相当“原始”：实际上，它们只是*张量*。对于这样一个小的网络来说，这没问题，但对于更复杂的设计来说，很快就会变得繁琐。因此，接下来的两章将展示如何将权重和偏置抽象成神经网络*模块*，用内置的损失函数替换自制的损失函数，并摆脱冗长的参数更新流程。

