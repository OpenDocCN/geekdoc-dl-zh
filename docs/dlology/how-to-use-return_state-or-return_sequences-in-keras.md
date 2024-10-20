# 如何在 Keras 中使用 return_state 或 return_sequences

> 原文：<https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/>

###### 发帖人:[程维](/blog/author/Chengwei/) 4 年 8 个月前

([评论](/blog/how-to-use-return_state-or-return_sequences-in-keras/#disqus_thread))

![moon_cycle](img/b22d5bc667805d5b965797a61375221e.png)

你可能注意到了，在几个 Keras 轮回层中，有两个参数、、、`return_state`、、、、、、、`return_sequences`、。在这篇文章中，我将向你展示它们的含义以及在现实生活中何时使用它们。

为了理解他们的意思，我们首先需要打开一个循环层，比如最常用的 LSTM 和格鲁

## 简而言之 RNN

Keras 中实现的递归层的最原始版本, [SimpleRNN](https://keras.io/layers/recurrent/#simplernn) ,它受到消失梯度问题的困扰，使其难以捕捉长程相关性。或者，LSTM 和 GRU 都装有独特的“门”，以避免长期信息“消失”掉。

![rnn](img/fa5f753d7ea9d6da433588e0ece02125.png)

在上图中，我们可以看到给定一个 RNN 层的输入序列，与每个时间步长相关的每个 RNN 单元将生成称为隐藏状态的输出，即^(t) 。

取决于你使用哪一个 RNN，它在如何计算 a^(t) 上有所不同。

![gru-lstm](img/97b20a21cb03c2cf925e5e082697cf2f.png)

c^(t) 以上公式中的每个 RNN 单元称为单元状态。对于 GRU，给定时间步长的单元状态等于其输出隐藏状态。对于 LSTM 来说，输出隐藏状态a^(t>是由“门控”单元状态 c^(t) 由输出门γ[o]产生的，所以 a^(t) 和 c^(t) 是不一样的。不要担心其余的公式。对 RNN 的基本了解应该足够了。)

## 返回序列

返回序列是指返回隐藏状态^(<>)。默认情况下，Keras RNN 图层中的`return_sequences`被设置为 False，这意味着 RNN 图层将只返回最后一个隐藏状态输出 a^(T) 。最后一个隐藏状态输出捕获输入序列的抽象表示。在某些情况下，这就是我们所需要的，例如分类或回归模型，其中 RNN 后面是密集层，以生成用于新闻主题分类的逻辑或用于[情感分析](https://www.dlology.com/blog/tutorial-chinese-sentiment-analysis-with-hotel-review-data/)的分数，或者在生成模型中生成[下一个可能字符的 softmax 概率](https://www.dlology.com/blog/how-to-generate-realistic-yelp-restaurant-reviews-with-keras/)。

在其他情况下，我们需要完整的序列作为输出。设定 `return_sequences` 为真有必要。

让我们定义一个只包含一个 LSTM 层的 Keras 模型。使用常量初始值设定项，以便输出结果可重复用于演示目的。

输出:

```py
[[[0.00767819]
  [0.01597687]
  [0.02480672]]] (1, 3, 1)
```

我们可以看到 LSTM 图层的输出数组形状为(1，3，1)，代表(#样本数、#时间步长数、#LSTM 单位数)。与 return_sequences 设置为 False 时相比，形状将为 (#Samples，#LSTM 单位)，这仅返回最后一个时间步长的隐藏状态。

输出:

```py
[[0.02480672]] (1, 1)
```

在两种主要情况下，您可以应用`return_sequences`到返回完整序列。

1.  堆叠 RNN，前一个或多个 RNN 层应将 `return_sequences` 设置为真，以便随后的一个或多个 RNN 层可以具有作为输入的完整序列。
2.  我们想为每个时间步生成分类。
    1.  如语音识别或更简单的形式- [触发词检测](https://www.dlology.com/blog/how-to-do-real-time-trigger-word-detection-with-keras/)，其中我们为每个时间步长生成一个介于 0~1 之间的值，表示触发词是否存在。
    2.  用 CTC 进行 OCR(光学字符识别)序列建模。

## 返回状态

返回序列指返回单元格状态 c^(t) 。对于 GRU，正如我们在“RNN 简括”一节中讨论的，a^(t)= c^(t)，所以你可以不用这个参数。但是对 LSTM 来说，隐藏状态和细胞状态是不一样的。

在 Keras 中，通过设置 `return_state` 为真，我们可以输出 RNN 除隐藏状态之外的最后一个单元格状态。

输出:

```py
[array([[0.02480672]], dtype=float32), array([[0.02480672]], dtype=float32), array([[0.04864851]], dtype=float32)]
(1, 1)
(1, 1)
(1, 1)
```

LSTM 层的输出有三个分量，它们是( a^(T) ， a^(T) ，c^(T) )，“T”代表最后一个时间步，每个都有形状(#个样本，#个 LSTM 单位)。

您想要设置`return_state`的主要原因是RNN 可能需要在共享权重时用先前的时间步长初始化其单元状态，例如在[编码器-解码器模型](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)中。下面显示了编码器-解码器模型的一段代码。

你已经注意到对于上面的编解码器型号来说 `return_sequences` 和 `return_state` 都被设置为真。在那种情况下，LSTM 的输出将有三个分量，(一一^(一< 1)...T >) ，a^(t)，c^(t))。如果我们从前面的例子中做同样的事情，我们可以更好地理解它的区别。

输出

```py
[array([[[0.00767819],
        [0.01597687],
        [0.02480672]]], dtype=float32), array([[0.02480672]], dtype=float32), array([[0.04864851]], dtype=float32)]
(1, 3, 1)
(1, 1)
(1, 1)
```

值得一提的是，如果我们用 GRU 代替 LSTM，输出将只有两个分量。 ( 一^(1...T >) ，c^(t))自从在 GRU 出现了一个^(t)= c^(t)。

## 结论

为了理解如何使用`return_sequences` 和 `return_state` ，我们先简单介绍两个常用的递归层，LSTM 和 GRU，以及它们的单元格状态和隐藏状态是如何导出的。接下来，我们深入研究了应用两个参数中的每一个的一些案例，以及当你可以考虑在你的下一个模型中使用它们时的技巧。

你可以在 [my GitHub repo](https://github.com/Tony607/Keras-RNN-return) 上找到这篇文章的源代码。

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/&text=How%20to%20use%20return_state%20or%20return_sequences%20in%20Keras) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/)

*   [←在 Google Colab 中运行 TensorBoard 的快速指南](/blog/quick-guide-to-run-tensorboard-in-google-colab/)
*   [如何训练 Keras 模型识别可变长度的文本→](/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/)