# 平均值和标准偏差

> 原文：<https://winder.ai/mean-and-standard-deviation/>

# 平均值和标准偏差

欢迎光临！本车间来自 [Winder.ai](https://Winder.ai) 。注册以获得更多免费的研讨会、培训和视频。

这个研讨会是关于数据的两个基本度量。我希望你们开始思考如何最好地描述或总结数据。我们怎样才能最好地获取一组数据，并用尽可能少的变量来描述这些数据？这些被称为*汇总统计*，因为它们汇总了统计数据。换句话说，这是你的第一个模型！

```py
import numpy as np 
```

## 平均

*均值*，也称为平均值，是对数据趋势的衡量。例如，如果你被提供一些数据，那么你可以说，平均起来，最有可能用平均值来表示。

平均值计算如下:

$ $ \ mu = \frac{\sum_{i=0}^{n-1}{ x _ I } } { n } $ $

所有观察值的总和除以观察值的数量。

```py
x = [6, 4, 6, 9, 4, 4, 9, 7, 3, 6]; 
```

```py
N = len(x)
x_sum = 0
for i in range(N):
    x_sum = x_sum + x[i]
mu = x_sum / N
print("μ =", mu) 
```

```py
μ = 5.8 
```

当然，我们应该使用库来减少我们必须编写的代码量。对于像这样的低级任务，最常见的库叫做 Numpy。

我们可以将上述内容改写为:

```py
N = len(x)
x_sum = np.sum(x)
mu = x_sum / N
print("μ =", mu) 
```

```py
μ = 5.8 
```

我们可以更进一步，只使用 Numpy 的均值实现:

```py
print("μ =", np.mean(x)) 
```

```py
μ = 5.8 
```

## 标准偏差

为了描述我们的数据，平均值本身并不能提供足够的信息。它告诉我们平均应该观察什么值。但是该值可以是该值的+/- 1 或+/- 100。(+/-是“加或减”的简写，即“可能大于或小于该值”)。

为了提供这一信息，我们需要一个围绕平均值的“扩散”的量度。最常见的“价差”度量是标准偏差。

点击:[WinderResearch.com 了解更多关于标准差的信息——我们为什么要使用标准差，这样做对吗？](https://winder.ai/why-do-we-use-standard-deviation/)。

总体的标准差是:

$ $ \ sigma = \ sqrt { \frac{\sum_{i=0}^{n-1}{(x _ I-\穆)^2 }} {N} }$$

```py
x = [6, 4, 6, 9, 4, 4, 9, 7, 3, 6]; 
```

```py
N = len(x)
mu = np.mean(x)
print("μ =", mu) 
```

```py
μ = 5.8 
```

```py
print("Deviations from the mean:", x - mu)
print("Squared deviations from the mean:", (x - mu)**2)
print("Sum of squared deviations from the mean:", ((x - mu)**2).sum() )
print("Mean of squared deviations from the mean:", ((x - mu)**2).sum() / N ) 
```

```py
Deviations from the mean: [ 0.2 -1.8  0.2  3.2 -1.8 -1.8  3.2  1.2 -2.8  0.2]
Squared deviations from the mean: [  0.04   3.24   0.04  10.24   3.24   3.24  10.24   1.44   7.84   0.04]
Sum of squared deviations from the mean: 39.6
Mean of squared deviations from the mean: 3.96 
```

```py
print("σ =", np.sqrt(((x - mu)**2).sum() / N )) 
```

```py
σ = 1.98997487421 
```

再说一遍，我们不需要编写所有代码。Numpy 当量为:

```py
print("σ =", np.std(x)) 
```

```py
σ = 1.98997487421 
```

## 有什么条件？

你知道他们会是抢手货，对吧？；-)

我在开始的时候没有提到它，但是集中趋势和传播的两个先前的测量是特定于一个非常特殊的数据组合的。

如果观察值是以特殊方式分布的，那么这些指标完美地模拟了底层数据。如果不是，那么这些指标是无效的。

你可能会说“啊？”一些新单词，我们来复习一下。