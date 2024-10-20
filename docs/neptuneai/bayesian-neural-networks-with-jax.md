# 贝叶斯神经网络——用 JAX 框架实现、训练和推理

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/bayesian-neural-networks-with-jax>

[贝叶斯神经网络(BNN)不同于人工神经网络(NN)](https://web.archive.org/web/20221206064740/https://www.quora.com/What-is-the-difference-between-a-Bayesian-network-and-an-artificial-neural-network) 。主要区别——BNNs 可以回答“我不确定”。这很有趣，但为什么你会希望神经网络告诉你它不知道你问题的答案？

为了向您展示网络说“我不确定”的重要性，我们需要考虑处理**非分布数据**。在人工智能安全中，[非分布检测](https://web.archive.org/web/20221206064740/https://medium.com/analytics-vidhya/out-of-distribution-detection-in-deep-neural-networks-450da9ed7044)是当有人试图用并非来自数据集的例子愚弄网络时，网络如何感知的。

我们将探索 BNNs 背后的理论，然后用 BNNs 实现、训练和运行数字识别任务的推理。这很棘手，但是我会告诉你你需要做什么来让 BNNs 开始学习。我们将把它编码在新的、热门的 JAX 框架中(如果你不知道，我们将做一个快速介绍)。

在文章的最后，我们将向我们的神经网络输入字母而不是数字，看看它会做什么。我们开始吧！

## 人工神经网络的贝叶斯观点

在我们开始之前，请注意这是一个复杂的话题。如果你觉得这个理论很难理解，那就直接跳到本文的编码部分。稍后，您还可以查看本文末尾链接的其他深入指南。

在非贝叶斯人工神经网络模型中(上图左侧)，我们训练网络参数的点估计。

在贝叶斯人工神经网络(上图右侧)中，我们用分布来表示我们对训练参数的信念，而不是点估计。代替变量，我们有随机变量，我们想从数据中推断。

## 什么是贝叶斯神经网络？

贝叶斯神经网络组件列表:

*   数据集 ***D*** 带有预测器 ***X*** (例如图像)和标签*(例如类)。*
*   *可能性 ***P(D|θ)*** 或 ***P(Y |X，θ)*** 用由 ***θ*** 参数化的神经网络(NN)计算的逻辑上的分类 softmax 分布表示，例如 softmax 多层感知器。

    *   注意:到目前为止，它与非贝叶斯神经网络没有区别。
    *   如果我们“正常地”训练它 SGD 使用交叉熵损失——那么我们可以说，我们得到了参数*的最大似然点估计。参见“深度学习”，第 5.5 章:最大似然估计([“深度学习”。自适应计算和机器学习。”麻省理工学院出版社，2016](https://web.archive.org/web/20221206064740/http://www.deeplearningbook.org/) )*
    *   *然而，使用贝叶斯神经网络，参数来自它们的分布。进一步阅读！** 
*   **在神经网络参数之前，***【P(θ)】用正态分布来表示。***

    *   *它编码了我们对参数值可能是什么的先验知识(或者说缺乏知识)。*
    *   *然而，我们怀疑这些是零附近的一些小值。*
    *   *这一假设来自我们的先验知识，即当我们将 dnn 的参数保持在 0 附近时，它们往往工作得很好。*** 
*   ***在看到数据之后，我们的 NN 参数的后验***P(θ| D)***—人们可以说是“在训练之后”。

    *   这是训练参数的分布。
    *   我们将使用贝叶斯定理来计算它…
    *   …或者至少我们会尝试这样做。*** 

 **### 贝叶斯定理

从理论上讲，贝叶斯定理是我们应该用来根据先验和似然性计算神经网络参数的后验概率的工具。但是，有一个条件。

这个积分很难计算。只有在一些需要使用共轭先验的特殊情况下才容易处理。在“深度学习”，第 5.6 章:贝叶斯统计([“深度学习”中了解更多信息。自适应计算和机器学习。”麻省理工出版社，2016](https://web.archive.org/web/20221206064740/http://www.deeplearningbook.org/) 。在《走向数据科学》网站上还有一篇关于[共轭先验的精彩文章。](https://web.archive.org/web/20221206064740/https://towardsdatascience.com/conjugate-prior-explained-75957dc80bfb)

在我们的例子中，它很难处理，因为这个积分没有解析解。我们使用一个复杂的非线性函数，名为“人工神经网络”。这在计算上也很难处理，因为在分母中有指数数量的可能参数赋值需要评估和求和。

想象一个二元神经网络，它对 N 个参数分配了**2^N个参数。对于 N=272，就是**2^(272)，已经比可见宇宙中的原子数量还要多。让我们同意，272 个参数并不多，要知道现代 CNN-s 有数百万个参数。****

### 变分推理为救援！

不会算？然后近似！

我们用一个分布 Q 来近似后验概率，称为变分分布，最小化它们之间的 KL 散度***[KL](Q(θ)**| |**P(θ| D))***。我们将找到与后验概率最接近的概率分布，它由一小组参数表示，如多元高斯分布的均值和方差，并且我们知道如何从中采样。

此外，我们必须能够通过它进行反向传播，并每次对分布的参数(即均值和方差)进行一点点修改，以查看最终的分布是否更接近我们想要计算的后验分布。

如果后验概率正是我们想要计算的，我们如何知道最终的分布是否更接近后验概率？就是这个想法！

从分布之间的 KL 散度，***D[KL](Q(θ)**| |**P(θ| D))***，可以得到证据下界(ELBO)。

这就是所谓的变分推理。它把推理问题变成了优化问题。通过优化右侧，我们优化了从我们的变分分布 NN 参数***【θ∾Q()***中采样的经典最大似然分类损失(例如交叉熵损失)，减去正则化损失，对于高斯分布，正则化损失采用封闭形式，这意味着它是一个众所周知的方程，您将在一分钟内看到。

通过优化它，我们最大化证据——我们的数据集为真的概率——并最小化我们的变分分布、***【Q(θ)***和后验、 ***P(θ|D)*** 之间的差异。后路正是我们想要的，是我们的目标！

还有一个注意:它被称为证据下限，因为 KL 散度将总是正的。因此，右边是左边证据的下限。详见本教程:[多尔施，卡尔。变型自动编码器教程。](https://web.archive.org/web/20221206064740/http://arxiv.org/abs/1606.05908)

现在，正如所承诺的，我们有了 NN 参数上的分布，***【Q(θ)***，我们知道如何使用 ELBO 学习它。让我们跳到代码中来看看它的实践吧！

## 什么是 JAX？

正如我之前提到的，我们将使用 JAX。

> “JAX[亲笔签名](https://web.archive.org/web/20221206064740/https://github.com/hips/autograd)和 [XLA](https://web.archive.org/web/20221206064740/https://www.tensorflow.org/xla) ，聚在一起进行高性能数值计算和机器学习研究。它提供了 Python+NumPy 程序的可组合转换:区分、矢量化、并行化、实时编译到 GPU/TPU，等等。”~ [JAX 文档](https://web.archive.org/web/20221206064740/https://jax.readthedocs.io/en/latest/)。

您可以查看 JAX 文档，但是您可能不需要它来理解下面的代码。正如作者所说，这就像机器学习和深度学习研究的 NumPy。但是，我建议至少读一节，关于随机数的那一节。这可能不直观，因为通常在 NumPy 中你不必考虑伪随机数发生器的状态，但是在 JAX 中，你可以显式地将它传递给随机值采样函数。

## 基于 JAX 的贝叶斯神经网络用于数字分类

你可以[在这里](https://web.archive.org/web/20221206064740/https://gitlab.com/awarelab/spin-up-with-variational-bayes)找到代码。README 告诉您如何运行它。我鼓励你现在就去做，然后读完这篇文章。该回购包括:

1.  MNIST 的 mlp 分类器(在 JAX 和俳句中)。
2.  MNIST 上的伯努利 vae 生成模型。
3.  Bayes . py–MNIST 上的变分 Bayes NN 分类器。

今天，我们将做最后一个，变分贝叶斯神经网络分类器。我们将讨论代码中最重要的部分。

### 关于 HumbleSL(HSL 包)的说明

HumbleSL 是我写的直接监督学习(SL) Python 库。它提供了进行深度 SL 所需的所有样板代码:

*   一个网络定义工厂，
*   度量和损失，
*   一种数据加载器，
*   火车环线，
*   等等。

它得到了 JAX 图书馆和 T2 俳句框架的支持。它使用 [TensorFlow 数据集](https://web.archive.org/web/20221206064740/https://www.tensorflow.org/datasets)进行数据加载和预处理。

### 培养

#### 下载 MNIST 数据集

第 56-61 行下载训练和测试数据集。

```py
train_dataset = hsl.load_dataset(
     'mnist:3.*.*', 'train', is_training=True, batch_size=FLAGS.batch_size)
 train_eval_dataset = hsl.load_dataset(
     'mnist:3.*.*', 'train', is_training=False, batch_size=10000)
 test_eval_dataset = hsl.load_dataset(
     'mnist:3.*.*', 'test', is_training=False, batch_size=10000)
```

train_dataset 用于训练。train_eval_dataset 用于对训练数据集进行性能评估。test_eval_dataset 用于对测试数据集进行性能评估，你猜对了。

数据集是迭代器，您可以通过以下方式访问图像(和标签)的连续批次:

```py
batch_image, batch_label = next(train_dataset)

```

#### 创建多层感知器(MLP)模型

第 71-74 行创建了 MLP 模型。

```py
net = hk.without_apply_rng(hk.transform(
     hsl.mlp_fn,
     apply_rng=True  
 ))
```

如果你对这个片段到底做了什么感兴趣，请查看[俳句基础](https://web.archive.org/web/20221206064740/https://dm-haiku.readthedocs.io/en/latest/api.html)。所有你需要知道的是，它创建了“标准”的 MLP 与 64 个单位的两个隐藏层。

它在输入端接受一个 28×28 的图像，并返回对应于每个可能的类(数字)的 10 个值。net 对象有两个函数:init 和 apply。

*   params = net.init(next(rng)，batch_image)获取下一个随机生成器状态和图像批次，并返回初始模型参数。它需要随机发生器状态来采样参数。
*   logits = net.apply(params，batch_image)获取模型参数和图像批次，然后返回批次输出(10 个数字的批次)。

你可以把网络想象成一个典型的裸架构。你需要提供一些参数来预测它。

#### 初始化贝叶斯神经网络参数

第 79-85 行获取 MLP 模型参数，并使用它来初始化贝叶斯神经网络参数。

```py
prior = dict(

     mu=params,

     logvar=jax.tree_map(lambda x: -7 * jnp.ones_like(x), params),
 )
```

我们用平均场逼近后验概率。这意味着我们用以均值和方差(或对数方差)为参数的高斯分布来表示我们的变分分布，因为它可以采用任何值，而不仅仅是正值，这简化了训练。我们这样做是因为从高斯分布中取样很容易。

记住这里，后验概率是经过训练的 MLP 参数的分布。我们没有一套 MLP 参数来训练。我们训练近似后验的变分分布，并从中采样 MLP 参数。在代码中，对于变量名，我可能会交替使用 aprx_posterior、posterior 和 prior 来表示变分分布，我承认这不是 100%正确，但在实践中，它们是一回事，或者我想强调训练参数的阶段(即 prior 是未训练的后验)。

#### 初始化优化程序

第 89-90 行定义并初始化 ADAM 优化器。

```py
opt = optix.adam(FLAGS.lr)
 opt_state = opt.init(prior)
```

就这么简单。您传递学习率 FLAGS.lr 和初始参数 prior。当然，优化器用于将梯度应用到参数上。与标准深度学习中的相同。

#### 定义目标

第 92-110 行定义了 ELBO 目标。

```py
def elbo(aprx_posterior, batch, rng):
     """Computes the Evidence Lower Bound."""
     batch_image, batch_label = batch

     params = sample_params(aprx_posterior, rng)

     logits = net.apply(params, batch_image)

     log_likelihood = -hsl.softmax_cross_entropy_with_logits(
         logits, batch_label)

     kl_divergence = jax.tree_util.tree_reduce(
         lambda a, b: a + b,
         jax.tree_multimap(hsl.gaussian_kl,
                           aprx_posterior['mu'],
                           aprx_posterior['logvar']),
    )
     elbo_ = log_likelihood - FLAGS.beta * kl_divergence
     return elbo_, log_likelihood, kl_divergence
```

它获取一批图像(和标签)，对 MLP 参数进行采样，并对它们进行预测。然后，它计算 logits 和标签之间的交叉熵(分类损失)，并计算变分分布和正态分布之间的 KL 散度(正则化损失)。

hsl.gaussian_kl 以封闭形式计算后者。由 flagsβ加权的两者的组合产生 ELBO。这与上面 ELBO 的数学表达式相匹配。损失是负面的 ELBO:

```py
def loss(params, batch, rng):
     """Computes the Evidence Lower Bound loss."""
     return -elbo(params, batch, rng)[0]
```

我们需要取反，因为 JAX 优化器只能做梯度下降。然而，我们需要最大化 ELBO，而不是最小化它。

#### 训练循环

第 116-126 行定义了 SGD 更新步骤。这是我们进行培训所需的最后一块。

```py
@jax.jit
 def sgd_update(params, opt_state, batch, rng):
     """Learning rule (stochastic gradient descent)."""

     grads = jax.grad(loss)(params, batch, rng)

     updates, opt_state = opt.update(grads, opt_state)

     posterior = optix.apply_updates(params, updates)
     return posterior, opt_state
```

此函数执行 SGD 更新的一个步骤。首先，它评估损失函数对于当前参数和该批数据的梯度。然后，计算更新并将其应用于参数。这个函数在一次更新后返回新的变分分布参数和优化器状态。之所以需要后者，是因为 ADAM 优化器存储并更新其自适应矩估计所需的状态。

现在，您只需在循环中运行这个函数，训练就会继续进行。我有一个助手函数:hsl.loop，它还负责检查点和定期评估训练和测试性能。

### 估价

第 128-140 行计算诊断。

```py
def calculate_metrics(params, data):
     """Calculates metrics."""
     images, labels = data
     probs = predict(net, params, images, next(rng), FLAGS.num_samples)[0]
     elbo_, log_likelihood, kl_divergence = elbo(params, data, next(rng))
     mean_aprx_evidence = jnp.exp(elbo_ / FLAGS.num_classes)
     return {
         'accuracy': hsl.accuracy(probs, labels),
         'elbo': elbo_,
         'log_likelihood': log_likelihood,
         'kl_divergence': kl_divergence,
         'mean_approximate_evidence': mean_aprx_evidence,
    }
```

它从对提供的参数和数据运行预测开始。这不同于在 ELBO 物镜中简单地采样一组参数。下一小节将对此进行描述。这些预测与地面实况标注一起用于计算精度 hsl.accuracy 辅助函数。

接下来，我们计算 ELBO、分类损失(log_likelihood)和正则化损失(kl_divergence)。ELBO 用于计算近似证据，这直接来自 ELBO 的公式——它是证据下界，不是吗？这是在当前参数下数据的近似概率，即图像具有相应的标签。越高越好，因为这意味着我们的模型很好地拟合了数据-它为来自数据集的标签提供了高概率。

所有这些指标都放在一个字典中，并返回给调用者。在我们的例子中，hsl.loop 辅助函数将不时地对来自训练和测试数据集的数据以及当前参数调用它。

### 预言；预测；预告

第 41-49 行运行预测。

```py
def predict(net, prior, batch_image, rng, num_samples):
     probs = []
     for i in range(num_samples):
         params_rng, rng = jax.random.split(rng)
         params = sample_params(prior, params_rng)
         logits = net.apply(params, batch_image)
         probs.append(jax.nn.softmax(logits))
     stack_probs = jnp.stack(probs)
     return jnp.mean(stack_probs, axis=0), jnp.std(stack_probs, axis=0)
```

这只是在样本数量 _ 样本参数集上运行预测。然后，对预测进行平均，并计算这些预测的标准偏差作为不确定性的度量。

说到不确定性，现在我们已经有了所有的部分，让我们来玩一下贝叶斯神经网络。

## 玩贝叶斯神经网络

您运行代码并看到以下内容:

```py
      0 | test/accuracy                       0.122
      0 | test/elbo                         -94.269
      0 | test/kl_divergence                 26.404
      0 | test/log_likelihood               -67.865
      0 | test/mean_approximate_evidence     0.000
      0 | train/accuracy                     0.095
      0 | train/elbo                       -176.826
      0 | train/kl_divergence               26.404
      0 | train/log_likelihood             -150.422
      0 | train/mean_approximate_evidence     0.000

```

这些是训练前的诊断，看起来没问题:

*   精确度约为 10%,对于随机初始化的神经网络来说是非常好的。这是随机猜测标签的准确性。
*   ELBO 非常低，这在开始是没问题的，因为我们的变分分布远离真实的后验概率。
*   变分分布和正态分布之间的 KL 散度为正。它一定是正的，因为我们是以封闭形式计算的，而 KL，因为它是距离的度量，不能取负值。
*   MLP 模型返回的真实标签的对数似然或对数概率非常低。这意味着模型将低概率分配给真正的标签。如果我们还没有训练它，这是预料之中的。
*   平均近似证据为 0。同样，我们还没有训练模型，所以它根本没有对数据集建模。

让我们运行 10k 步，再次查看诊断结果:

```py
  10000 | test/accuracy                       0.104
  10000 | test/elbo                         -5.796
  10000 | test/kl_divergence                 2.516
  10000 | test/log_likelihood               -3.280
  10000 | test/mean_approximate_evidence     0.560
  10000 | train/accuracy                     0.093
  10000 | train/elbo                         -5.610
  10000 | train/kl_divergence                 2.516
  10000 | train/log_likelihood               -3.095
  10000 | train/mean_approximate_evidence     0.571

```

这不好。真实标签的概率上升，log_likelihood 和

均值近似证据上升，变分分布更接近正态分布，kl 散度下降。

然而，采用返回概率的 argmax 来推断标签并将其与地面真实标签进行比较的准确度仍然比随机分类器好大约 10%。这不是代码中的错误。诊断是正确的，我们需要两个技巧来训练它。继续读！

### 训练贝叶斯神经网络的技巧

#### 低β值

β参数对分类损失和正则化损失进行加权。beta 越高，正则化越强。太强的正则化会对模型有太多的约束，它将不能对任何知识进行编码。在上面的例子中，它被设置为 flagsβ= 1。

这使得 kl_divergance(正则化损失)大幅下降。但是，太强了！更好的值大约是 FLAGS.beta = 0.001，这是我提供给你的代码中的默认值。

#### 低初始方差

另一件事是变分分布的初始方差。太大了，网络很难开始训练和编码任何有用的知识。这是因为采样参数变化很大。在上面的例子中，它被设置为大约 0.37。在代码中，默认情况下，它被设置为~0.001，这是一个更好的值。

#### 固定示例

现在我们已经将超参数更改为正确的参数，让我们看看 10k 步后的诊断结果:

```py
  10000 | test/accuracy                       0.979
  10000 | test/elbo                         -0.421
  10000 | test/kl_divergence               318.357
  10000 | test/log_likelihood               -0.103
  10000 | test/mean_approximate_evidence     0.959
  10000 | train/accuracy                     0.995
  10000 | train/elbo                         -0.341
  10000 | train/kl_divergence               318.357
  10000 | train/log_likelihood               -0.022
  10000 | train/mean_approximate_evidence     0.966

```

测试准确率为 98%，我们可以同意它现在工作！注意正则化损失(kl_divergence)有多大。

是的，它离正态分布有那么远，但它需要如此。尽管如此，有了这个小测试，它仍然可以防止过度拟合。平均近似证据也非常高，这意味着我们的模型很好地预测了数据。注意 ELBO 也非常接近零(这是它的最大值)。

### 查找不符合分布的示例

我把训练好的模型放在上面的数字“3”和字母“B”上运行。以下是输出结果:

|  | 预言；预测；预告 | 可能性 | Std。戴夫。(不确定性) |
| --- | --- | --- | --- |
|  |  |  | 0 |
|  |  |  |  |

如你所见，它将数字“3”归类为 3 没有任何问题。然而，当我们给它喂食它在训练中没有看到的东西时，一件有趣的事情发生了。该模型将字母“B”分类为 8。如果我们处理的是正常的神经网络，那就是了。

幸运的是，我们训练的贝叶斯神经网络也可以告诉我们它有多确定。

我们看到，在数字“3”的情况下，它是可信的——STD。戴夫。概率为 0 左右。对于字母“B ”,它返回的概率在任一方向上可以变化 0，45 %!

这就像我们的模型告诉我们“如果我不得不猜，那么这是 8，但它可能是任何东西——我以前没见过这个。”

这样，贝叶斯神经网络既可以对图像进行分类，也可以说“我不知道”。我们可以查出性病。戴夫。阈值，在该阈值之后，我们拒绝来自例如我们用来评估我们的模型的测试数据集的分类。

我简单地在整个测试数据集上运行模型，并观察 std。戴夫。它的预测值。然后，我取第 99 个百分位数(99%的其他值较低的值)，在本例中为 0.37。因此，我决定应该拒绝 1%测试图像的分类。

我这样做是因为我知道 MNIST 数据集中有一些疯狂的图像，连我都无法正确分类。回到我们的例子，显然 0.45 > 0.37，所以我们应该拒绝字母“B”的分类。

## 结论

就是这样！现在你可以训练一个不会让你愚弄它的神经网络。不确定性估计是人工智能安全中的一个大主题。我留给你们更多的阅读材料:**