# 自然语言处理中的标记化:类型、挑战、例子、工具

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/tokenization-in-nlp>

在任何 NLP 项目中，您需要做的第一件事是[文本预处理](https://web.archive.org/web/20221108111346/https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html#:~:text=What%20is%20text%20preprocessing%3F,an%20example%20of%20a%20Task.)。预处理输入文本仅仅意味着将数据转换成可预测和可分析的形式。这是构建令人惊叹的 NLP 应用程序的关键一步。

预处理文本有不同的方法:

*   停止单词删除，
*   符号化，
*   堵塞。

其中，最重要的一步是标记化。它是将文本数据流分解成单词、术语、句子、符号或其他一些有意义的元素(称为标记)的过程。有很多开源工具可以用来执行令牌化过程。

在本文中，我们将深入探讨标记化的重要性和不同类型的标记化，探索一些实现标记化的工具，并讨论面临的挑战。

## 为什么我们需要标记化？

[标记化](https://web.archive.org/web/20221108111346/https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/)是任何 NLP 流水线的第一步。它对你管道的其他部分有着重要的影响。记号赋予器将非结构化数据和自然语言文本分解成可以被视为离散元素的信息块。文档中出现的标记可以直接用作表示该文档的向量。

这就立刻把一个非结构化的字符串(文本文档)变成了适合机器学习的数值数据结构。它们也可以被计算机直接用来触发有用的动作和响应。或者它们可以在机器学习管道中用作触发更复杂决策或行为的功能。

标记化可以分隔句子、单词、字符或子词。当我们将文本拆分成句子时，我们称之为句子标记化。对于单词，我们称之为单词标记化。

**句子标记化的例子**

![](img/ba13b4cdb5236a426eda9ea40877fe86.png)

**单词标记化的例子**

![](img/a7b57f3dbe94d9029846492966ac0528.png)

虽然 Python 中的标记化可能很简单，但我们知道它是开发良好模型和帮助我们理解文本语料库的基础。本节将列出一些可用于标记文本内容的工具，如 NLTK、TextBlob、spacy、Gensim 和 Keras。

### 空白标记化

标记文本最简单的方法是在字符串中使用空格作为单词的“分隔符”。这可以通过 **Python 的 split 函数**来实现，该函数可用于所有 string 对象实例以及 string 内置类本身。您可以根据需要任意更改分隔符。

![](img/f590bbf7295cf7b58d586f8e3a1bd30c.png)

正如您所注意到的，这个内置的 Python 方法在标记一个简单的句子方面已经做得很好了。它的“错误”在最后一个词上，在那里它包括了带有符号**“1995”的句尾标点符号。**我们需要将标记与句子中相邻的标点符号和其他重要标记分开。

在下面的例子中，我们将使用逗号作为分隔符来执行句子标记化。

![](img/4fbc27d63878f10da647c51174777166.png)

### NLTK 单词标记化

[NLTK](https://web.archive.org/web/20221108111346/http://www.nltk.org/) (自然语言工具包)是一个用于自然语言处理的开源 Python 库。它为 50 多个语料库和词汇资源(如 WordNet)提供了易于使用的界面，以及一组用于分类、标记化、词干提取和标记的文本处理库。

您可以使用 NLTK 的 tokenize 模块轻松地对文本中的句子和单词进行标记。

首先，我们将从 NLTK 库中导入相关的函数:

![](img/aa433540bec96718a42541f167eec42e.png)

*   **单词和句子分词器**

![](img/0e9436141f3998636504ade66376aa97.png)

*注意:sent_tokenize 使用来自 token izers/punkt/English . pickle 的预训练模型*

*   **基于标点符号的分词器**

这个分词器根据空格和标点符号将句子拆分成单词。

![](img/241549b607d00ae7c25ce1a8d16815f4.png)

我们可以注意到考虑“Amal。m " word _ token ize 中的一个单词，并在 wordpunct_tokenize 中将其拆分。

这个分词器包含了各种英语单词分词的通用规则。它分隔像(？！。；，)从相邻的标记中分离出来，并将十进制数保留为单个标记。此外，它还包含英语缩写的规则。

例如，“不”被标记为[“做”，“不”]。您可以在这个[链接](https://web.archive.org/web/20221108111346/http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.treebank)中找到 Treebank 标记器的所有规则。

![](img/e9fd6f657ce51e13239dd8441cd54b13.png)

当我们想要在像 tweets 这样的文本数据中应用标记化时，上面提到的标记化器无法产生实用的标记。通过这个问题，NLTK 有了一个专门针对 tweets 的基于规则的标记器。如果我们需要像情感分析这样的任务，我们可以将表情符号分成不同的单词。

![](img/8337f4a9df10da83345b4e20f7db46a8.png)

NLTK 的多词表达式标记器(MWETokenizer)提供了一个函数 add_mwe()，允许用户在对文本使用标记器之前输入多个词表达式。更简单地说，它可以将多词表达式合并成单个令牌。

![](img/4eadfb0e0095d570808393394fca9f70.png)

### TextBlob 单词标记化

[TextBlob](https://web.archive.org/web/20221108111346/https://textblob.readthedocs.io/en/dev/) 是一个用于处理文本数据的 Python 库。它提供了一个一致的 API，用于处理常见的自然语言处理(NLP)任务，如词性标注、名词短语提取、情感分析、分类、翻译等。

让我们从安装 TextBlob 和 NLTK 语料库开始:

```py
$pip install -U textblob 
$python3 -m textblob.download_corpora

```

在下面的代码中，我们使用 TextBlob 库执行单词标记化:

![](img/9a6958c6bff5de43c9f25c9b69071acc.png)

我们可以注意到 TextBlob 标记器删除了标点符号。此外，它还有英语缩写的规则。

### 空间记号化器

SpaCy 是一个开源的 Python 库，可以解析和理解大量文本。提供适合特定语言(英语、法语、德语等)的型号。)，它以最高效的常用算法实现来处理 NLP 任务。

spaCy tokenizer 提供了指定特殊标记的灵活性，这些标记不需要分段，或者需要使用每种语言的特殊规则进行分段，例如，句子末尾的标点符号应该分开，而“U.K .”应该保留为一个标记。

在使用 spaCy 之前，您需要安装它，下载英语语言的数据和模型。

```py
$ pip install spacy
$ python3 -m spacy download en_core_web_sm

```

![](img/a142894e2f12fbf7b1ecfa9aa146bd76.png)

### Gensim 单词标记器

Gensim 是一个 Python 库，用于大型语料库的主题建模、文档索引和相似性检索。目标受众是自然语言处理(NLP)和信息检索(IR)社区。它为标记化提供了实用函数。

![](img/98c0323b6b4540ca20d85d91e0ca2294.png)

### 使用 Keras 的标记化

[Keras](https://web.archive.org/web/20221108111346/https://keras.io/) 开源库是最可靠的深度学习框架之一。为了执行标记化，我们使用 Keras.preprocessing.text 类中的 text_to_word_sequence 方法。Keras 最大的优点是在标记之前将字母表转换成小写字母，这样可以节省大量时间。

![](img/f0edb1a21547be10418fb8a03358de23.png)

***注意:**你可以在这里找到所有的代码示例[。](https://web.archive.org/web/20221108111346/https://github.com/AmalM7/NLP-Stuff/blob/main/Tokenization_in_NLP.ipynb)*

### 可能有用

检查如何[跟踪您的 TensorFlow / Keras 模型训练元数据](https://web.archive.org/web/20221108111346/https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras)(指标、参数、硬件消耗等)。

## 挑战和局限

让我们讨论一下标记化任务的挑战和局限性。

通常，该任务用于用英语或法语编写的文本语料库，其中这些语言通过使用空格或标点符号来分隔单词，以定义句子的边界。不幸的是，这种方法不适用于其他语言，如汉语、日语、朝鲜语、印地语、乌尔都语、泰米尔语等。这个问题产生了开发一个结合所有语言的通用标记化工具的需求。

另一个限制是阿拉伯文本的标记化，因为阿拉伯语作为一种语言具有复杂的形态。例如，一个阿拉伯单词可能包含多达六个不同的标记，如单词“عقد”(eaqad)。

![Tokenization challenges](img/dba44d4a54a77112a38a2913e5fff6c3.png)

*One Arabic word gives the meanings of 6 different words in the English language. | [Source](https://web.archive.org/web/20221108111346/https://www.pinterest.com/pin/484699978625704765/visual-search/?x=17&y=14&w=530&h=530&cropSource=6) *

在自然语言处理方面有很多研究正在进行。你需要选择一个挑战或问题，并开始寻找解决方案。

## **结论**

通过这篇文章，我们已经了解了来自各种库和工具的不同记号赋予器。

我们看到了这项任务在任何 NLP 任务或项目中的重要性，并且我们还使用 Python 和 Neptune 实现了跟踪。您可能会觉得这是一个简单的主题，但是一旦深入到每个记号赋予器模型的细节，您会注意到它实际上非常复杂。

从上面的例子开始练习，并在任何文本数据集上尝试它们。你练习得越多，你就越能理解标记化是如何工作的。

如果你陪我到最后——谢谢你的阅读！

### 阿迈勒·门兹利

创新、足智多谋、自我激励的数据科学家。我热衷于用数据解决难题，我相信它是我们今天最强大的工具，来回答宇宙中最模糊的问题。此外，我喜欢教学、指导和写技术博客。

* * *

**阅读下一篇**

## 如何构建和管理自然语言处理(NLP)项目

Dhruvil Karani |发布于 2020 年 10 月 12 日

如果说我在 ML 行业工作中学到了什么的话，那就是:**机器学习项目很乱。**

这并不是说人们不想把事情组织起来，只是在项目过程中有很多事情很难组织和管理。

你可以从头开始，但有些事情会阻碍你。

一些典型的原因是:

*   笔记本中的快速数据探索，
*   取自 github 上的研究报告的模型代码，
*   当一切都已设置好时，添加新的数据集，
*   发现了数据质量问题并且需要重新标记数据，
*   团队中的某个人“只是快速地尝试了一些东西”,并且在没有告诉任何人的情况下改变了训练参数(通过 argparse 传递),
*   从高层推动将原型转化为产品“仅此一次”。

多年来，作为一名机器学习工程师，我学到了一堆**东西，它们可以帮助你保持在事物的顶端，并检查你的 NLP 项目**(就像你真的可以检查 ML 项目一样:)。

在这篇文章中，我将分享我在从事各种数据科学项目时学到的关键指针、指南、技巧和诀窍。许多东西在任何 ML 项目中都是有价值的，但有些是 NLP 特有的。

[Continue reading ->](/web/20221108111346/https://neptune.ai/blog/how-to-structure-and-manage-nlp-projects-templates)

* * *