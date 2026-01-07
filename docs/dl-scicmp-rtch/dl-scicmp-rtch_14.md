# 11  神经网络模块化

> 原文：[`skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/network_2.html`](https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/network_2.html)

让我们回顾一下我们在前几章中构建的网络。它的目的是回归，但其方法不是线性的。相反，一个激活函数（ReLU，代表“修正线性单元”）引入了非线性，位于单个隐藏层和输出层之间。在这个原始实现中，“层”只是张量：权重和偏差。你不会对它们将被替换为*模块*感到惊讶。

训练过程将如何变化？从概念上讲，我们可以区分四个阶段：前向传播、损失计算、梯度反向传播和权重更新。让我们思考我们的新工具将如何融入其中：

+   前向传播，不是在张量上调用函数，而是调用模型。

+   在计算损失时，我们现在使用 `torch` 的 `nnf_mse_loss()`。

+   梯度反向传播实际上是不变的唯一操作。

+   权重更新由优化器负责处理。

一旦我们做出这些更改，代码将更加模块化，并且可读性大大提高。

## 11.1 数据

作为先决条件，我们生成数据，与上次相同。

```r
# input dimensionality (number of input features)
d_in <- 3
# number of observations in training set
n <- 100

x <- torch_randn(n, d_in)
coefs <- c(0.2, -1.3, -0.5)
y <- x$matmul(coefs)$unsqueeze(2) + torch_randn(n, 1)
```

*## 11.2 网络

通过两个通过 ReLU 激活函数连接的线性层，最简单的选择是一个顺序模块，这与我们在模块介绍中看到的是非常相似的：

```r
# dimensionality of hidden layer
d_hidden <- 32
# output dimensionality (number of predicted features)
d_out <- 1

net <- nn_sequential(
 nn_linear(d_in, d_hidden),
 nn_relu(),
 nn_linear(d_hidden, d_out)
)
```

*## 11.3 训练

这里是更新的训练过程。我们使用 Adam 优化器，这是一个流行的选择。

```r
opt <- optim_adam(net$parameters)

### training loop --------------------------------------

for (t in 1:200) {

 ### -------- Forward pass --------
 y_pred <- net(x)

 ### -------- Compute loss -------- 
 loss <- nnf_mse_loss(y_pred, y)
 if (t %% 10 == 0)
 cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")

 ### -------- Backpropagation --------
 opt$zero_grad()
 loss$backward()

 ### -------- Update weights -------- 
 opt$step()

}
```

```r
Epoch:  10    Loss:  2.549933 
Epoch:  20    Loss:  2.422556 
Epoch:  30    Loss:  2.298053 
Epoch:  40    Loss:  2.173909 
Epoch:  50    Loss:  2.0489 
Epoch:  60    Loss:  1.924003 
Epoch:  70    Loss:  1.800404 
Epoch:  80    Loss:  1.678221 
Epoch:  90    Loss:  1.56143 
Epoch:  100    Loss:  1.453637 
Epoch:  110    Loss:  1.355832 
Epoch:  120    Loss:  1.269234 
Epoch:  130    Loss:  1.195116 
Epoch:  140    Loss:  1.134008 
Epoch:  150    Loss:  1.085828 
Epoch:  160    Loss:  1.048921 
Epoch:  170    Loss:  1.021384 
Epoch:  180    Loss:  1.0011 
Epoch:  190    Loss:  0.9857832 
Epoch:  200    Loss:  0.973796 
```

除了缩短和简化代码外，我们的更改在性能上也有很大的提升。*  *## 11.4 未来展望

你现在对 `torch` 的工作原理以及如何在不同设置中使用它来最小化成本函数有了很多了解：例如，用于训练神经网络。但针对实际应用，`torch` 还有很多其他功能。本书的下一部分，也是内容最丰富的一部分，将专注于深度学习。

