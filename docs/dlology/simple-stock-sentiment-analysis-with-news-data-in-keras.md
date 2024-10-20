# 利用 Keras 中的新闻数据进行简单的股票情绪分析

> 原文：<https://www.dlology.com/blog/simple-stock-sentiment-analysis-with-news-data-in-keras/>

###### 发帖人:[程维](/blog/author/Chengwei/) 4 年 7 个月前

([评论](/blog/simple-stock-sentiment-analysis-with-news-data-in-keras/#disqus_thread))

![stock](img/dbe392f46119d2edb966a79abba4486a.png)

你想知道每天的新闻会对股票市场产生什么影响吗？在本教程中，我们将探索并构建一个模型，该模型从 Reddit 用户那里读取排名前 [25 位的投票世界新闻，并预测道琼斯指数在某一天是上涨还是下跌。](https://www.reddit.com/r/worldnews/?hl=)

读完这篇文章，你会了解到，

*   深度学习序列模型如何预处理文本数据？
*   如何使用预先训练的手套嵌入向量初始化 Keras 嵌入层。
*   建立一个 GRU 模型，可以处理单词序列，并能够考虑词序。

现在让我们开始，读到最后，因为将有一个秘密奖金。

### 文本数据预处理

![reddit-news](img/e1fc019802c790869f6b8cb4b1eae08e.png)

对于输入文本，我们将把每天的所有 25 条新闻连接成一个长字符串。

之后，我们将把所有的句子转换成小写字母，去掉数字和标点这样的字符，因为这些字符不能用手套嵌入来表示。

下一步是将所有的训练句子转换成索引列表，然后用零填充所有这些列表，使它们的长度相同。

在决定最大序列长度之前，可视化所有输入样本的长度分布是有帮助的。

![sentences-length](img/e77098ff792b1828f86a0610a045c948.png)

请记住，我们选择的最大长度越长，训练模型所需的时间就越长，因此，我们没有选择数据集中大约 700 的最长序列长度，而是选择 500 作为折衷，以覆盖所有样本的大部分文本，同时保持相对较短的训练时间。

### 嵌入层

在 Keras 中，嵌入矩阵被表示为“层”,并将正整数(对应于单词的索引)映射到固定大小的密集向量(嵌入向量)中。可以用预训练的嵌入来训练或初始化它。在这一部分，你将学习如何在 Keras 中创建一个嵌入层，用 GloVe 50 维向量初始化它。因为我们的训练集很小，所以我们不会更新单词 embeddings，而是将它们的值保持不变。我将向您展示 Keras 如何允许您设置嵌入是否可训练。

`Embedding()`层以一个大小为(批量大小，最大输入长度)的整数矩阵作为输入，这对应于转换成索引(整数)列表的句子，如下图所示。

![embedding](img/b5fd98097257210b42538966d742ea74.png)

下面的函数处理将句子字符串转换成索引数组的第一步。单词到索引的映射取自 GloVe 嵌入文件，因此我们可以在以后无缝地将索引转换为单词向量。

之后，我们可以像这样实现预训练的嵌入层。

*   将嵌入矩阵初始化为具有正确形状的 numpy 个零数组。(vocab_len，字向量的维度
*   用所有单词嵌入填充嵌入矩阵。
*   通过将 `trainable` 设置为假，定义 Keras 嵌入层并使其不可训练。
*   将嵌入层的权重设置为嵌入矩阵。

让我们通过询问单词“cat”的矢量表示来快速检查一下嵌入层。

结果是一个 50 维的数组。您可以进一步探索单词向量并使用余弦相似度来测量相似度，或者解决单词类比问题，例如男人对女人就像国王对 __。

### 构建和评估模型

该模型的任务是获取新闻字符串序列，并对道琼斯收盘值与前一收盘值相比是上涨还是下跌进行二元分类。如果值上升或保持不变，则输出“1”，如果值下降，则输出“0”。

我们正在构建一个简单的模型，在预训练的嵌入层之后包含两个堆叠的 GRU 层。密集层通过 softmax 激活生成最终输出。GRU 是一种处理和考虑序列顺序的递归网络，它在功能和性能方面类似于 LSTM，但训练起来计算成本更低。

接下来，我们可以训练评估模型。

生成 ROC 或我们的二元分类器来直观地评估其性能也是有帮助的。

![ROC](img/eaffca5b0358ddd5ab6ba7a2ff852955.png)

我们的模型比市场趋势的随机猜测要好 2.8%左右。

关于 ROC 和 AUC 的更多信息，可以阅读我的另一篇博客- [关于如何为 Keras 分类器](https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/)生成 ROC 图的简单指南。

### 结论和进一步的思考

在这篇文章中，我们介绍了一种快速简单的方法来建立一个 Keras 模型，嵌入层用预先训练的手套嵌入进行初始化。读完这篇文章后你可以尝试的东西，

*   使嵌入层权重可训练，从一开始就训练模型，然后比较结果。
*   增加最大序列长度，看看这会如何影响模型性能和训练时间。
*   纳入其他投入，以形成一个多投入 Keras 模型，因为其他因素可能与股票指数波动相关。比如有 [MACD](https://www.investopedia.com/terms/m/macd.asp) (均线收敛/发散振荡器)动量指标供你考虑。为了有多输入，你可以使用 [Keras 功能 API](https://keras.io/getting-started/functional-api-guide/) 。

有什么改进模型的想法吗？评论并分享你的想法。

你可以在 [my Github repo](https://github.com/Tony607/SentimentStock) 中找到完整的源代码和训练数据。

### 投资者的红利

![stock_ticket](img/07cb70fb9de65cfaa4d124120e732567.png)

如果你像我几年前一样是整个投资界的新手，你可能想知道从哪里开始，最好是零佣金的免费投资。通过学习如何免费交易股票，你不仅可以省钱，而且你的投资可能会以更快的速度复利。最好的投资应用之一 Robinhood 就是这么做的，不管你是只买一股还是 100 股，都没有佣金。它是从零开始建造的，通过去除脂肪并把节省下来的钱交给顾客，尽可能地提高效率。加入 Robinhood，我们两个都会免费得到一只苹果，福特，或者 Sprint 这样的股票。确保你使用我的[共享链接](https://share.robinhood.com/chengwz1)。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [情感分析](/blog/tag/sentiment-analysis/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/simple-stock-sentiment-analysis-with-news-data-in-keras/&text=Simple%20Stock%20Sentiment%20Analysis%20with%20news%20data%20in%20Keras) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/simple-stock-sentiment-analysis-with-news-data-in-keras/)

*   [←如何训练 Keras 模型识别长度可变的文本](/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/)
*   [如何使用 Keras 进行无监督聚类→](/blog/how-to-do-unsupervised-clustering-with-keras/)