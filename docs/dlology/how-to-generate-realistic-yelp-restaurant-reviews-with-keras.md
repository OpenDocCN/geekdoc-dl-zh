# 如何用 Keras 生成真实的 yelp 餐馆评论

> 原文：<https://www.dlology.com/blog/how-to-generate-realistic-yelp-restaurant-reviews-with-keras/>

###### 发帖人:[程维](/blog/author/Chengwei/) 4 年 10 个月前

([评论](/blog/how-to-generate-realistic-yelp-restaurant-reviews-with-keras/#disqus_thread))

![restaurant reviews](img/3224708797d93c0b8be67578d480d1ee.png)

TL；速度三角形定位法(dead reckoning)

看完这篇文章。你将能够建立一个模型来生成像这样的 5 星 Yelp 评论。

*【生成评审文本样本(未修改)】*

```py
<SOR>I had the steak, mussels with a side of chicken parmesan. All were very good. We will be back.<EOR>
<SOR>The food, service, atmosphere, and service are excellent. I would recommend it to all my friends<EOR>
<SOR>Good atmosphere, amazing food and great service.Service is also pretty good. Give them a try!<EOR>
```

我将向您展示如何，

*   获取并准备培训数据。
*   构建字符级语言模型。
*   训练模型时的提示。
*   生成随机评论。

即使在 GPU 上，训练模型也很容易需要几天时间。幸运的是，预先训练的模型重量是可用的。所以我们可以直接跳到有趣的部分来生成评论。

## 准备好数据

Yelp 数据集以 JSON 格式免费提供。

下载并解压后，你会在**数据集**文件夹中找到我们需要的 2 个文件，

*   评论. json
*   商务. json

那两个文件相当大，尤其是 **review.json** 文件(3.7 GB)。

**review.json** 文件的每一行都是对 json 字符串的回顾。这两个文件没有 JSON 开始和结束方括号“[ ]”。所以 JSON 文件的内容作为一个整体不是一个有效的 JSON 字符串。另外，可能很难将整个 **review.json** 文件内容放入内存。所以，让我们先用助手脚本一行一行地把它们转换成 CSV 格式。

```py
python json_converter.py ./dataset/review.json
python json_converter.py ./dataset/business.json
```

之后，您会在**数据集**文件夹中找到这两个文件，

*   点评. csv
*   business.csv

这两个是有效的 CSV 文件，我们可以通过熊猫图书馆打开。

这是我们将要做的。我们只从类别中带有“**餐厅**标签的商家中提取 **5 星**评论文本。

接下来，让我们删除评论中的换行符和任何重复的评论。

向模型展示评审的起点和终点。我们需要在复习课文中添加特殊标记。

因此，最终准备好的评论中的一行将如您所料。

```py
"<SOR>Hummus is amazing and fresh! Loved the falafels. I will definitely be back. Great owner, friendly staff<EOR>"
```

## 建立模型

我们在这里构建的模型是一个**字符级语言模型**，这意味着最小的可区分符号是一个字符。您还可能遇到单词级模型，其中输入是单词标记。

**字符级语言模型**有一些优点和缺点。

**亲:**

*   不用担心未知词汇。
*   能够学习大量词汇。

**反对:**

*   以非常长的序列结束。在捕捉句子前半部分如何影响句子后半部分之间的**长期依赖关系**方面，不如单词级语言模型。
*   和角色级别模型也只是训练起来更加**计算昂贵**。

该模型与官方的[**lstm _ text _ generation . py**演示代码](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py)非常相似，除了我们正在堆叠 RNN 单元，允许在输入和输出层之间的整个隐藏状态中存储更多信息。它能产生更真实的 Yelp 评论。

在展示该模型的代码之前，让我们更深入地了解一下堆叠 RNN 是如何工作的。

你可能在标准神经网络中见过。(即喀拉斯的**致密**层)

第一层取输入 **x** 计算激活值**a^(【1】)**，堆栈下一层计算下一个激活值**a^(【2】)**。

![stack dense](img/f0525d995ce5939a9a1a21ec80f1f5ff.png)

堆叠 RNN 有点像标准的神经网络和“及时展开”。

对于符号**a^(【l】t)**表示激活分配给**层 l，**和 **t** 表示**时间步长 t** 。

![stack rnn](img/092d93066168a901cee7dc41282ca235.png)

让我们来看一看激活值是如何计算的。

要计算**a[2]^3**，有两个输入，**a[2]^2**和**a[1]^3**

g 为激活函数，w[a]^2，b[a]^2 为第二层参数。

![activation a23](img/76d64fd515f5adcc0fbe4a1446589518.png)

我们可以看到，要堆栈 RNNs，前面的 RNN 需要将所有时间步长 a^(t) 返回给后面的 RNN。

默认情况下，Keras 中的 **RNN** 层如 **LSTM** 只返回最后一个时间步长激活值 a^(T) 。为了返回所有时间步长的激活值，我们将`return_sequences`参数 设置为 `True` 。

这就是我们如何在 Keras 中建立模型。每个输入样本是 60 个字符的一键表示，总共有 95 个可能的字符。

每个输出是每个字符的 95 个预测概率的列表。

这是一个图形化的模型结构，可以帮助你把它可视化。

![model structure](img/03af081d953bd4d96e4045886bd51e78.png)

## 训练模型

训练模型的想法很简单，我们用输入/输出对来训练它。每个输入是 60 个字符，对应的输出是紧随其后的字符。

在数据准备步骤中，我们创建了一个干净的五星评论文本列表。总共 1，214，016 行评论。为了简化训练，我们只对长度等于或小于 250 个字符的评论进行训练，最终得到 418，955 行评论。

然后我们打乱评论的顺序，这样我们就不会连续对同一家餐馆的 100 条评论进行训练。

我们以一个长文本串的形式阅读所有评论。然后创建一个 python 字典(即哈希表)，将每个字符映射到从 0 到 94 的索引(总共 95 个唯一字符)。

文本语料库共有 72，662，807 个字符。很难把它作为一个整体来处理。因此，让我们把它分成每个 90k 字符的块。

对于语料库的每一块，我们将生成成对的输入和输出。通过将指针从块的开头移动到结尾，如果 step 设置为 1，则每次移动一个字符。

在 GPU (GTX1070)上为一个时期训练一个组块需要 219 秒，因此训练完整的语料库将需要大约 2 天。

```py
72662807 / 90000 * 219 /60 / 60/ 24 = 2.0 days
```

两个 Keras 回调派上了用场， **ModelCheckpoint** 和 **ReduceLROnPlateau** 。

模型检查点帮助我们保存每次改进的权重。

**ReduceLROnPlateau** 回调在**损失**指标停止下降时自动降低学习率。它的主要好处是我们不需要手动调整学习速度。它的主要弱点是学习率总是在降低和衰减。

为模型定型 20 个时期的代码如下所示。

正如你所猜测的，这需要一个月左右的时间。但是对我来说，大约两个小时的训练已经产生了一些有希望的结果。请随意尝试一下。

## 生成 5 星评价

不管你是直接跳到这一节，还是已经读过前面的到。有趣的部分来了！

使用预先训练的模型权重或您自己训练的权重，我们可以生成一些有趣的 yelp 评论。

这里的想法是，我们用最初的 60 个字符“播种”模型，并要求模型预测下一个字符。

![generate sample](img/2b134965b050f36daed0130c190d17e2.png)

“抽样指数”过程将通过给定的预测产生一些随机性来给最终结果 增加一些变化。

如果温度很低，它总是选择预测概率最高的指数。

生成 300 个字符，代码如下

## 总结和进一步阅读

在这篇文章中，你从头到尾了解了如何构建和训练一个字符级的文本生成模型。源代码可以在我的 [GitHub repo](https://github.com/Tony607/Yelp_review_generation) 上找到，也可以在训练前的模型上玩。

这里显示的模型是以多对一的方式训练的。还有另一种多对多方式的可选实现。考虑输入序列为长度为 7 的字符**cak**，期望输出为**hecake**。可以在这里查看，[char _ rnn _ karpathy _ keras](https://github.com/mineshmathew/char_rnn_karpathy_keras)。

作为建立单词级模型的参考，请查看我的另一个博客:[简单的股票情绪分析，包含 Keras 中的新闻数据](https://www.dlology.com/blog/simple-stock-sentiment-analysis-with-news-data-in-keras/)。

*   标签:
*   [keras](/blog/tag/keras/) ,
*   [深度学习](/blog/tag/deep-learning/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-generate-realistic-yelp-restaurant-reviews-with-keras/&text=How%20to%20generate%20realistic%20yelp%20restaurant%20reviews%20with%20Keras) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-generate-realistic-yelp-restaurant-reviews-with-keras/)

*   [←如何在 Keras 中缺失标签的情况下进行多任务学习](/blog/how-to-multi-task-learning-with-missing-labels-in-keras/)
*   [如何使用 Keras 进行实时触发词检测→](/blog/how-to-do-real-time-trigger-word-detection-with-keras/)