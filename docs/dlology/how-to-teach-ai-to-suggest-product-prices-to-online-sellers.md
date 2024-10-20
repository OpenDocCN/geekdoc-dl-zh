# 如何教人工智能向在线卖家建议产品价格

> 原文：<https://www.dlology.com/blog/how-to-teach-ai-to-suggest-product-prices-to-online-sellers/>

###### 发帖人:[程维](/blog/author/Chengwei/) 5 年前

([评论](/blog/how-to-teach-ai-to-suggest-product-prices-to-online-sellers/#disqus_thread))

![price prediction](img/9f627de1cf7cb935c2ade0dc31719664.png)

给定产品的品牌、名称、类别和一两句简短的描述，然后我们就可以预测它的价格。有这么简单吗？

在这场比赛中，我们做着完全相同的事情。全世界的开发者都在为“Mercari Prize:价格建议挑战”而战，奖金总额为 10 万美元(第一名:6 万美元，第二名:3 万美元，第三名:1 万美元)。

在这篇文章中，我将带你构建一个简单的模型来解决深度学习库中的挑战。

如果你是 Kaggle 的新手，为了下载数据集，你需要注册一个账户，完全没有痛苦。拥有帐户后，进入 Mercari 价格建议挑战赛的[“数据”选项卡](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data)。

将这三个文件下载到您的本地计算机，解压并保存到一个名为“ **input** 的文件夹中，在根文件夹下创建一个名为“ **scripts** 的文件夹，我们将在这里开始编码。

现在你应该有类似这样的目录结构。

。/Pricing_Challenge
| - **输入**| |-test . tsv
| |-sample _ submission . CSV
| `-train . tsv
`-**脚本**

# 准备数据

首先，让我们花些时间来理解手头的数据集

**train.tsv，test.tsv**

这些文件由产品列表组成。这些文件用制表符分隔。

*   `train_id` 或  `test_id` -清单的 id
*   `name` -清单的标题。请注意，我们已经清理了数据，删除了看起来像价格(例如$20)的文本，以避免泄漏。这些被删除的价格被表示为  `[rm]`
*   `item_condition_id` -卖方提供的物品的状况
*   `category_name` -清单的类别
*   `brand_name`
*   `price` -物品的售价。这是你要预测的目标变量。单位是美元。这个栏目不存在于  `test.tsv` 既然是你会预测的。
*   `shipping` - 1 表示运费由卖家支付，0 表示由买家支付
*   `item_description` -物品的完整描述。请注意，我们已经清理了数据，删除了看起来像价格(例如$20)的文本，以避免泄漏。这些被删除的价格被表示为  `[rm]`

对于价格列，输入取值范围相差很大的神经网络值是有问题的。网络也许能够自动适应这种异构数据，但这肯定会使学习更加困难。

处理这类数据的一个普遍的最佳实践是进行特征标准化:对于输入数据(输入数据矩阵中的一列)中的每个特征，我们对其应用 `log(x+1)`。

让我们看看新的“目标”列的分布。

![target](img/a7e2f32e4e479e4be8e7b5e7323d3efd.png)

## 文本预处理

### 替换收缩

我们将替换如下所示的缩写对，目的是统一词汇，以便更容易训练模型。

```py
"what's" → "what is",
"who'll" → "who will",
"wouldn't" → "would not",
"you'd" → "you would",
"you're" → "you are"
```

在我们这样做之前，让我们计算一下在“item_description”列中有多少行包含任何缩写。

下面列出了 5 个最重要的，这并不奇怪。

```py
can't - 6136
won't - 4867
that's - 4806
it's - 26444
don't - 32645
doesn't - 8520
```

下面是在训练和测试数据集中删除“item_description”和“name”列的缩写的代码。

### Handel 缺失值

为了成功地管理数据，理解缺失值的概念非常重要。如果研究人员没有正确处理缺失值，那么他/她可能会对数据做出不准确的推断。

首先看一下缺少多少数据，以及哪一列。

在输出中，我们知道有 42%的 brand_name 丢失，“category_name”和“item_description”列也丢失了不到 1%的数据。

```py
train_id             0.000000
name                 0.000000
item_condition_id    0.000000
category_name        0.004268
brand_name           0.426757
price                0.000000
shipping             0.000000
item_description     0.000003
dtype: float64
```

假设缺失值的情况非常少；然后，专家研究员可能会从分析中丢弃或忽略这些值。用统计学的语言来说，如果案例的数量少于样本的 5%,那么研究者可以放弃它们。在我们的例子中，我们可以删除带有、 **category_name** 或 **item_description** 列缺失值的行。

但是为了简单起见，让我们用字符串“ **missing** ”替换所有缺失的文本值。

### 创建分类列

有两个文本列具有特殊含义，

*   类别名称
*   品牌名称

不同的产品可能有相同的类别名称或品牌名称，因此从它们创建分类列会很有帮助。

为此，我们将使用 sklearn 的 LabelEncoder。转换之后，我们将有两个新的整数类型的列“category”和“brand”。

### 将文本标记为序列

对于词汇表中的每个唯一单词，我们将把它转换成一个整数来表示。所以一个句子会变成一个整数列表。

首先，我们需要从将要标记的文本列中收集词汇列表。即那 3 列，

*   类别名称
*   项目 _ 描述
*   名字

我们将使用 Keras 的文本处理**标记器**类。

`fit_on_texts()`方法将训练标记器生成词汇单词索引映射。而` texts_to_sequences()`方法实际上将从文本中生成序列。

### 填充序列

上一步生成的单词序列长度不同。因为我们网络中的第一层对于那些序列来说是 t`Embedding`层。每个嵌入层将形状为 `(samples, sequence_length) `的整数的 2D 张量作为输入

一批中的所有序列必须具有相同的长度，因为我们需要将它们打包成一个张量。因此，比其他序列短的序列应该用零填充，而长的序列应该被截断。

首先，我们需要为`sequence_length`我们的每个序列列选择。如果太长的话，模型训练就要花很长时间。如果太短，我们就有截断重要信息的风险。在我们做这个决定之前，最好能看到序列长度的分布。

这一行代码将绘制直方图中“seq_item_description”列的序列长度分布。

![sequence lengths histogram](img/fe1dc0ef4c6e9d0f2673eaec0261fca7.png)

让我们选择 60 作为最大序列长度，因为它覆盖了大多数序列。

在下面的代码中，我们使用 Keras 的序列处理`pad_sequences()`方法为每一列填充长度相同的序列。

# 建立模型

对于这些输入，这将是一个多输入模型

*   **名称**':转换成**序列的文本**
*   **item_desc** ': 文字转换成 **序列**
*   **brand** ':转换成**整数的文本**
*   **类别** ': 文本转换成 **整数**
*   **类别 _ 名称** ': 文本转换成 **序列**
*   **item_condition** ': **整数**
*   **出货** ': **整数** 1 或 0

除“shipping”之外的所有输入将首先进入嵌入层

对于那些序列输入，我们需要将它们提供给`Embedding`层。`Embedding`层将整数索引(代表特定单词)转化为密集向量。它接受整数作为输入，在内部字典中查找这些整数，然后返回相关的向量。这实际上是一种字典查找。

嵌入的序列然后将馈送到`GRU`层，像其他类型的递归网络一样，它擅长学习数据序列中的模式。

非顺序数据嵌入层将被`Flatten`层展平为二维。

包括“船运”在内的所有层将被连接成一个大的二维张量。

接着是几个密集层，最终输出密集层采用“**线性**激活回归到任意价格值，与为`activation`参数指定**无**相同。

我喜欢可视化，所以我也绘制模型结构。

![model plot](img/05a953ea7f73fcdafcf66b41825b0b13.png)

如果你好奇的话，这两行代码就可以完成。

需要安装 [Graphviz 可执行文件](https://www.graphviz.org/download/)。pip 在尝试绘图之前安装了 **graphviz** 和 **pydot** 软件包。

训练模型很简单，我们来训练 2 个历元，`X_train`是我们之前创建的字典，将输入名称映射到 Numpy 数组。

# 评估模型

Kaggle 挑战页面选择了“均方根对数误差”作为损失函数。

下面的代码将采用我们训练好的模型，并在给定验证数据的情况下计算损失值。

# 生成提交文件

如果你计划为测试数据集生成实际价格，并在 Kaggle 上试试运气。这段代码将反转我们之前讨论的特性规范化过程，并将价格写入一个 CSV 文件。

# 摘要

我们演练了如何在给定多个输入特征的情况下预测价格。如何对文本数据进行预处理，处理缺失数据，最后建立、训练和评估模型。

完整的源代码贴在我的 GitHub 上。

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-teach-ai-to-suggest-product-prices-to-online-sellers/&text=How%20to%20teach%20AI%20to%20suggest%20product%20prices%20to%20online%20sellers) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-teach-ai-to-suggest-product-prices-to-online-sellers/)

*   [←在你的浏览器上运行的十大深度学习体验](/blog/top-10-deep-learning-experiences-run-on-your-browser/)
*   [使用 Keras 从网络摄像头视频中轻松实时预测性别年龄→](/blog/easy-real-time-gender-age-prediction-from-webcam-video-with-keras/)