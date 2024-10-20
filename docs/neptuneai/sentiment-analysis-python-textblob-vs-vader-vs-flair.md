# Python 中的情感分析:TextBlob vs Vader 情感 vs 天赋 vs 从头构建

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/sentiment-analysis-python-textblob-vs-vader-vs-flair>

情感分析是最广为人知的自然语言处理(NLP)任务之一。这篇文章的目的是让读者对情感分析有一个非常清晰的理解，以及在 NLP 中实现情感分析的不同方法。所以让我们开始吧。

在过去的五年里，自然语言处理领域有了很大的发展，像 Spacy、TextBlob 等开源包。为 NLP 提供随时可用的功能，如情感分析。有这么多免费的软件包，让你不知道该为你的应用程序使用哪一个。

在本文中，我将讨论最流行的 NLP 情感分析包:

最后，我还将在一个公共数据集上比较它们各自的性能。

什么是情感分析？

## 情感分析是确定自然语言中给定表达的情感值的任务。

它本质上是一个多类文本分类文本，其中给定的输入文本被分类为积极、中性或消极情绪。类别的数量可以根据训练数据集的性质而变化。

比如有时候会公式化为 1 为正面情感，0 为负面情感标签的二元分类问题。

情感分析的应用

## 情感分析在很多领域都有应用，包括分析用户评论、推特情感等。让我们来看看其中的一些:

**电影评论:**分析在线电影评论，以获得观众对电影的见解。

*   **新闻舆情分析:**针对特定机构分析新闻舆情，获取洞察。
*   **社交媒体情绪分析:**分析脸书帖子、twitter 推文等的情绪。
*   **在线美食评论:**从用户反馈中分析美食评论的情感。
*   python 中的情感分析

python 中有许多可用的包，它们使用不同的方法来进行情感分析。在下一节中，我们将介绍一些最流行的方法和软件包。

## 基于规则的情感分析

基于规则的情感分析是计算文本情感的基本方法之一。这种方法只需要很少的前期工作，而且思想非常简单，不需要使用任何机器学习来理解文本情感。例如，我们可以通过计算用户在他/她的推文中使用“悲伤”这个词的次数来计算出一个句子的情感。

现在，让我们看看一些使用这种方法的 python 包。

## **文本块**

这是一个简单的 python 库，提供了对不同 NLP 任务的 API 访问，如情感分析、拼写纠正等。

[Textblob](https://web.archive.org/web/20230214174832/https://textblob.readthedocs.io/en/dev/) 情感分析器返回给定输入句子的两个属性:

**极性**是介于[-1，1]之间的浮点数，-1 表示消极情绪，+1 表示积极情绪。

### **主观性**也是一个位于[0，1]范围内的浮点数。主观句一般指个人观点、情感或判断。

让我们看看如何使用 Textblob:

Textblob 将忽略它不认识的单词，它将考虑它可以分配极性的单词和短语，并进行平均以获得最终分数。

*   **VADER 情绪**

用于情感推理的效价感知词典(VADER) 是另一个流行的基于规则的情感分析器。

```py
from textblob import TextBlob

testimonial = TextBlob("The food was great!")
print(testimonial.sentiment)
```

```py
 Sentiment(polarity=1.0, subjectivity=0.75)

```

它使用一系列词汇特征(例如单词)来计算文本情感，这些词汇特征根据它们的语义取向被标记为正面或负面。

Vader 情感返回给定输入句子被

### 积极、消极和中立。

例如:

“食物棒极了！”
阳性:99%
阴性:1%
中性:0%

这三个概率加起来是 100%。

让我们看看如何使用 VADER:

Vader 针对社交媒体数据进行了优化，当与来自 twitter、facebook 等的数据一起使用时，可以产生良好的结果。

基于规则的情感分析方法的主要缺点是，该方法只关心单个单词，而完全忽略了使用它的上下文。

例如，*“the party wave”*在被任何基于令牌的算法考虑时都将是否定的。

嵌入型模型

```py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
sentence = "The food was great!"
vs = analyzer.polarity_scores(sentence)
print("{:-<65} {}".format(sentence, str(vs)))
```

```py
{'compound': 0.6588, 'neg': 0.0, 'neu': 0.406, 'pos': 0.594}

```

文本嵌入是 NLP 中单词表示的一种形式，其中同义词相似的单词使用相似的向量来表示，当在 n 维空间中表示时，这些向量将彼此接近。

基于嵌入的 python 包使用这种形式的文本表示来预测文本情感。这导致 NLP 中更好的文本表示，并产生更好的模型性能。

其中一个就是天赋。

**天赋**

## Flair 是一个简单易用的 NLP 框架。

它提供了各种功能，例如:

预先训练的情感分析模型，

文本嵌入，

NER，

### 还有更多。

让我们看看如何使用 flair 非常简单有效地进行情感分析。

Flair 预训练情感分析模型在 IMDB 数据集上进行训练。要加载并使用它进行预测，只需:

*   如果您喜欢为您的领域定制一个情感分析器，可以使用 flair 使用您的数据集训练一个分类器。
*   使用 flair 预训练模型进行情感分析的缺点是，它是在 IMDB 数据上训练的，并且该模型可能不会很好地概括来自 twitter 等其他领域的数据。

*   从头开始构建情感分析模型

在本节中，您将学习何时以及如何使用 TensorFlow 从头开始构建情感分析模型。所以，我们来检查一下怎么做。

```py
from flair.models import TextClassifier
from flair.data import Sentence

classifier = TextClassifier.load('en-sentiment')
sentence = Sentence('The food was great!')
classifier.predict(sentence)

print('Sentence above is: ', sentence.labels)

```

```py
[POSITIVE (0.9961)

```

**为什么要定制型号？**

我们先来了解一下你什么时候会需要一个定制的**情绪分析**模型。例如，您有一个利基应用程序，如分析航空公司评论的情绪。

通过构建自定义模型，您还可以对输出进行更多的控制。

## **TFhub **

[TensorFlow Hub](https://web.archive.org/web/20230214174832/https://www.tensorflow.org/hub) 是一个训练有素的机器学习模型库，可随时进行微调，并可部署在任何地方。

出于我们的目的，我们将使用通用语句编码器，它将文本编码为高维向量。您还可以使用任何您喜欢的文本表示模型，如 GloVe、fasttext、word2vec 等。

### **型号**

因为我们使用一个通用的句子编码器来矢量化我们的输入文本，所以在模型中不需要嵌入层。如果你计划使用任何其他的嵌入模型，比如 GloVe，请随意关注我之前的[帖子](https://web.archive.org/web/20230214174832/https://neptune.ai/blog/document-classification-small-datasets)来获得一步一步的指导。在这里，我将为我们的目的构建一个简单的模型。

**数据集**

对于我们的例子，我将使用来自 Kaggle 的 twitter 情感分析数据集。这个数据集包含 140 万条带标签的推文。

### 你可以从[这里](https://web.archive.org/web/20230214174832/https://www.kaggle.com/kazanova/sentiment140)下载数据集。

要在 Colab 中运行示例，只需在笔记本提示时上传 Kaggle API 密钥，它就会自动为您下载数据集。

要在 Colab 中运行示例，只需在笔记本提示时上传 Kaggle API 密钥，它就会自动为您下载数据集。

**示例:使用 Python 进行 Twitter 情感分析**

这是 Colab 笔记本的链接。

### [**举例:用 Python 进行 Twitter 情感分析。**](https://web.archive.org/web/20230214174832/https://colab.research.google.com/drive/1tUr5t0ZJ-I4Ni40dkbjku92HAU5SyR_2?usp=sharing)

在同一个笔记本里，**我实现了我们上面讨论的所有算法。**

比较结果

![](img/0690b683e16f056b3a108bcd4221c261.png)

现在，让我们比较笔记本上的结果。

你可以看到我们的定制模型没有任何[超参数调整](/web/20230214174832/https://neptune.ai/blog/hyperparameter-tuning-in-python-a-complete-guide-2020)产生最好的结果。

我只在 Twitter 数据上训练了使用模型，其他的都是现成的。

你可以看到上面的包没有一个能很好地概括 twitter 数据，我一直在做一个很酷的开源项目来开发一个专门针对 twitter 数据的包，这个项目正在积极开发中。

请随意查看我在 GitHub 上的[项目。](https://web.archive.org/web/20230214174832/https://github.com/shahules786/Twitter-Sentiment)

最后的想法

在本文中，我讨论了情感分析以及用 python 实现它的不同方法。

## 我还在一个公共数据集上比较了它们的性能。

希望你会发现它们在你的一些项目中有用。

You can see that our custom model without any [hyperparameter tuning](/web/20230214174832/https://neptune.ai/blog/hyperparameter-tuning-in-python-a-complete-guide-2020) yields the best results. 

![](img/0690b683e16f056b3a108bcd4221c261.png)

I have only trained the Use model on the Twitter data, the other ones come out-of-the-box.

You can see that none of the above packages are generalizing well on twitter data, I have been working on a cool open source project to develop a package especially for twitter data and this is under active contribution. 

Feel free to check out my [project on GitHub.](https://web.archive.org/web/20230214174832/https://github.com/shahules786/Twitter-Sentiment)

## Final thoughts

In this article, I discussed sentiment analysis and different approaches to implement it in python. 

I also compared their performance on a common dataset. 

Hopefully, you will find them useful in some of your projects.