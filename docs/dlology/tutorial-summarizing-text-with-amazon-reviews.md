# 如何用 Tensorflow 总结亚马逊评论

> 原文：<https://www.dlology.com/blog/tutorial-summarizing-text-with-amazon-reviews/>

###### 发布者:[程维](/blog/author/Chengwei/) 5 年 2 个月前

([评论](/blog/tutorial-summarizing-text-with-amazon-reviews/#disqus_thread))

这个项目的目标是建立一个模型，可以为亚马逊上销售的美食评论创建相关的摘要。该数据集包含超过 50 万条评论，托管在 [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews) 上。

这里有两个例子来展示数据的样子

```py
Review # 1
Good Quality Dog Food
I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most.

Review # 2
Not as Advertised
Product arrived labeled as Jumbo Salted Peanuts...the peanuts were actually small sized unsalted. Not sure if this was an error or if the vendor intended to represent the product as "Jumbo".
```

为了构建我们的模型，我们将使用两层双向 RNN，对输入数据使用 LSTM，对目标数据使用 LSTM。

本工程的标段有:
1。检查数据
2。准备数据
3。建立模型
4。训练模型
5。做我们自己的总结

受带有亚马逊评论的 post [文本摘要](https://medium.com/towards-data-science/text-summarization-with-amazon-reviews-41801c2210b)的启发，通过一些改进和更新来与最新的 TensorFlow 版本 1.3 配合使用，这些改进获得了更好的准确性。

## 改进摘要

### 1.更好地修饰句子

原始代码通过 **text.split()** 对单词进行标记，这不是万无一失的，

例如单词后跟标点符号“你在开玩笑吗？我觉得你是。”会被错误地标记为

【‘是’，‘你’，‘开玩笑？“我”、“想”、“你”、“是”]

我们用这条线代替

这将正确地生成单词列表

['是'，'你'，'开玩笑'，'我'，'想'，'你'，'是']

### 2.增加数据 p 修复 f 过滤和分类速度

最初的作者使用了两个 for 循环来排序和过滤数据，这里我们使用 Python 内置的排序和过滤函数来做同样的事情，但是速度更快。

**过滤** 为长度限制，  `<UNK>`为数量限制

**排序** 摘要和文本按  **文本** 中元素的长度从最短到最长排序

### 3。连接编码器中的 RNN 层

原始代码缺少下面这一行，这就是我们如何通过将当前层的输出馈送到下一层的输入来连接层。因此，原始代码仅表现为编码器中的单个双向 RNN 层。

### 4.解码层使用 **MultiRNNCell**

原作者使用 for 循环将 num_layers 的 LSTMCell 连接在一起，这里我们使用由多个简单单元格( **BasicLSTMCell** )顺序组成的 **MultiRNNCell** 到来简化代码。

## 培训结果

经过 2 个小时的 GPU 训练，亏损降到 1 以下，固定在 0.707。

下面是用训练好的模型生成的一些摘要。

```py
- Review:
 The coffee tasted great and was at such a good price! I highly recommend this to everyone!
- Summary:
 great great coffee

- Review:
 love individual oatmeal cups found years ago sam quit selling sound big lots quit selling found target expensive buy individually trilled get entire case time go anywhere need water microwave spoon know quaker flavor packets
- Summary:
 great taste
```

在我的 GitHub 上查看完整的源代码。

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/tutorial-summarizing-text-with-amazon-reviews/&text=How%20to%20Summarize%20Amazon%20Reviews%20with%20Tensorflow) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/tutorial-summarizing-text-with-amazon-reviews/)

*   [←如何使用 Keras 对患者查询进行分类(1 分钟培训)](/blog/tutorial-medical-triage-with-patient-query/)
*   [改进马达声学分类器的一个简单技巧→](/blog/one-simple-trick-to-improve-the-motor-acoustic-classifier/)