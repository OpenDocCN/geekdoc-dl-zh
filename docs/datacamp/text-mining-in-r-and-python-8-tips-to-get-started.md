# R 和 Python 中的文本挖掘:8 个入门技巧

> 原文：<https://web.archive.org/web/20230101103415/https://www.datacamp.com/blog/text-mining-in-r-and-python-8-tips-to-get-started>

你想开始学习文本挖掘吗，但是你开始的大部分教程很快就变得相当复杂了？或者你找不到合适的数据集来处理？

DataCamp 的最新文章将带你浏览 8 个提示和技巧，帮助你开始文本挖掘并保持下去。

## 1.对文字感到好奇

在数据科学中，几乎任何事情的第一步都是好奇。文本挖掘也不例外。

你应该对 StackOverflow 的数据科学家大卫·罗宾逊这样的文本感到好奇，几周前，他在自己的博客中写道，“我看到了一个简单的假设[……]需要用数据来研究”。

(对于那些想知道假设是什么的人，假设是这样的:

> 每一条不夸张的推文都来自 iPhone(他的工作人员)。
> 
> 每一条双曲线的推文都来自安卓(来自他)。pic.twitter.com/GWr6D8h5ed
> 
> — Todd Vaziri (@tvaziri) [August 6, 2016](https://web.archive.org/web/20220525174915/https://twitter.com/tvaziri/status/762005541388378112)

)

或者，如果你不是真的想验证假设，你应该对你看到的酷词云感到好奇，意识到你想为自己复制它。

你还需要相信文本挖掘有多酷吗？

从最近受到媒体广泛关注的众多文本挖掘用例中的一个获得灵感，如[南方公园对话](https://web.archive.org/web/20220525174915/http://kaylinwalker.com/text-mining-south-park/)、……

## 2.获得你需要的技能和知识

当你变得好奇时，是时候加快你的游戏并开始发展你关于文本挖掘的知识和技能了。你可以通过完成一些教程和课程轻松做到这一点。

在这些课程中，你应该注意的是，它们至少向你介绍了数据科学工作流程中的一些步骤，如数据准备或预处理、数据探索、数据分析等

DataCamp 为那些希望开始文本挖掘的人提供了一些材料:最近，Ted Kwartler 写了一篇关于从 Google Trends 和 Yahoo 的股票服务中挖掘数据的客座教程。这本简单易懂的 R 教程让您通过实践来学习文本挖掘，对于任何文本挖掘初学者来说都是一个很好的开始。

此外，Ted Kwartler 还是 DataCamp 的 R 课程[“文本挖掘:词汇袋”](https://web.archive.org/web/20220525174915/https://www.datacamp.com/courses/intro-to-text-mining-bag-of-words/)的讲师，该课程将向您介绍各种用于分析和可视化数据的基本主题，并让您在现实世界的案例研究中实践您所获得的文本挖掘技能。

另一方面，你也有一些不一定局限于 r 的其他材料。对于 Python，你可以查看这些教程和/或课程:对于 Python 中的文本分析的介绍，你可以去[本教程](https://web.archive.org/web/20220525174915/http://nealcaren.web.unc.edu/an-introduction-to-text-analysis-with-python-part-1/)。或者你也可以通过这个[入门 Kaggle 教程](https://web.archive.org/web/20220525174915/https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words)。

然而，你对其他资源更感兴趣吗？去 DataCamp 的[学习数据科学 Python 资源& R](https://web.archive.org/web/20220525174915/https://www.datacamp.com/community/tutorials/learn-data-science-resources-for-python-r/) 教程！

## 3.单词，单词，单词-找到你的数据

一旦掌握了分析和可视化数据所需的基本概念和主题，就该去寻找数据了！

相信我们，当我们告诉你有很多方法可以获取你的数据。除了提及 Google Trends 和 Yahoo，您还可以从以下网站获取数据:

*   推特！R 和 Python 都提供了允许您连接到 Twitter API 并检索 tweets 的包或库。在下一节中，您将了解到更多这方面的内容。
*   互联网档案馆(Internet Archive)是一个非营利性图书馆，拥有数百万免费书籍、电影、软件、音乐、网站等。
*   古腾堡计划提供超过 55，000 本免费电子书。它们中的大多数都是公认的文学作品，因此，如果你想对莎士比亚、简·奥斯汀、埃德加·爱伦·坡等作家的作品进行分析，它们将是一个很好的来源。
*   对于文本挖掘的学术方法，您可以使用 [JSTOR 的数据内容进行研究](https://web.archive.org/web/20220525174915/http://about.jstor.org/service/data-for-research)。这是一个免费的自助式工具，允许计算机科学家、数字人文主义者和其他研究人员选择 JSTOR 上的内容并与之交互。
*   如果你想对连续剧或电影进行文本挖掘，就像上面给出的例子一样，你可以考虑下载字幕。一个简单的谷歌搜索绝对可以为你提供你所需要的，来形成你自己的语料库，从而开始文本挖掘。
*   你也可以从语料库中获取数据。两个著名的语料库是:
    *   [路透社文本语料库](https://web.archive.org/web/20220525174915/http://www.daviddlewis.com/resources/testcollections/reuters21578/)。有些人会认为这不是最多样化的语料库，但如果你刚刚开始学习文本挖掘，这是非常好的。
    *   棕色文集包含了 500 个来源的文本，这些文本按体裁分类。

如你所见，可能性是无限的。任何包含文本的内容都可以成为文本挖掘案例研究的主题。

## 4.为工作寻找合适的工具

现在您已经找到了数据的来源，您可能希望使用正确的工具将它们归您所有，并对其进行分析。

您将学习的教程和课程将为您提供一些入门工具。

但是，根据你所学的课程或教程，你可能会错过一些。为了完整起见，这里列出了 R 中用于文本挖掘的一些包:

*   毫无疑问，R 中最常用的文本挖掘包之一是 [`tm`包](https://web.archive.org/web/20220525174915/https://www.rdocumentation.org/packages/tm/versions/0.6-2)。这个包通常与更具体的包一起使用，例如 Twitter 包，您可以使用它从 twitteR 网站提取 tweets 和 followers。
*   要用 R 做网页抓取，你应该使用 [`rvest`库](https://web.archive.org/web/20220525174915/https://www.rdocumentation.org/packages/rvest/versions/0.3.2)。关于使用`rvest`的简短教程，请点击这里的[。](https://web.archive.org/web/20220525174915/https://www.datacamp.com/community/tutorials/scraping-javascript-generated-data-with-r/)

对于 Python，您可以依赖这些库:

*   自然语言工具包，包含在 [`nltk`包](https://web.archive.org/web/20220525174915/http://www.nltk.org/)中。这个软件包非常有用，因为你可以轻松访问 50 多个语料库和词汇资源。你可以在这一页的[上看到这些的列表。](https://web.archive.org/web/20220525174915/http://www.nltk.org/nltk_data/)
*   如果你想挖掘 Twitter 数据，你有很多选择。其中一个用的比较多的包是 [`Tweepy`包](https://web.archive.org/web/20220525174915/http://www.tweepy.org/)。
*   对于网页抓取来说， [`scrapy`包](https://web.archive.org/web/20220525174915/https://scrapy.org/)会派上用场，从网站中提取你需要的数据。也可以考虑使用 [`urllib2`](https://web.archive.org/web/20220525174915/https://docs.python.org/2/library/urllib2.html) ，一个打开网址的包。然而，有时候,`requests`软件包更值得推荐，甚至可能使用起来更方便。有人说它“更人性化”或更具声明性，因为有些事情，比如设置用户代理和请求页面，只有一行代码。你有时也会看到一些人提到`urllib`包，但这似乎并不太受欢迎:大多数开发人员只提到一两个他们觉得特别有用的功能。

## 5.准备是成功的一半——预处理你的数据

当我告诉你数据科学家花 80%的时间清理他们的数据时，你可能不会感到惊讶。

文本挖掘在这方面也不例外。

文本数据可能是脏的，因此您应该确保花费足够的时间来清理它。

如果您不确定预处理数据的含义，一些标准的预处理步骤包括:

*   提取文本和结构，以便获得想要处理的文本格式，
*   删除停用词，如“that”或“and”，
*   词干，用来提取单词的词根。这可以借助字典或语言规则或算法(如波特算法)来完成。

这些步骤看起来很难，但是预处理数据并不需要这样。

在很大程度上，上一节提到的库和包已经帮了你很多。例如，R 中的`tm`库允许你用它的内置函数做一些预处理:你可以做词干处理和删除停用词，消除空白并将单词转换成小写。类似地，Python 中的 [`nltk`包](https://web.archive.org/web/20220525174915/http://www.nltk.org/)允许您进行大量预处理，因为有内置函数。

但是，您仍然可以更进一步，基于正则表达式做一些预处理来描述您感兴趣的字符模式。这样，您也将加快数据清理的过程。

对于 Python，你可以利用 [`re`库](https://web.archive.org/web/20220525174915/https://docs.python.org/3/library/re.html)，对于 R，有一堆函数可以帮你解决，比如`grep()`、`grepl()`、`regexpr()`、`gregexpr()`、`sub()`、`gsub()`、`strsplit()`。

如果想了解更多 R 中的这些函数和正则表达式，可以随时查看[本页](https://web.archive.org/web/20220525174915/https://www.rdocumentation.org/packages/base/versions/3.3.1/topics/regex)。

## 6.数据科学家的奇遇-探索您的数据

到目前为止，您将会兴奋地开始您的分析。然而，在开始分析之前查看一下数据总是一个好主意。

在上面提到的基础包或库的帮助下，快速开始探索数据的一些想法:

*   创建一个文档术语矩阵:这个矩阵中的元素表示一个术语(一个单词或一个 n 元语法)在语料库的一个文档中的出现。
*   在创建了文档术语矩阵之后，您可以使用直方图来可视化语料库中单词的频率。
*   您可能还想知道语料库中两个或更多术语之间的相关性。
*   为了可视化你的语料库，你也可以制作一个单词云。在 R 中，你可以使用 [wordcloud](https://web.archive.org/web/20220525174915/https://www.rdocumentation.org/packages/wordcloud/versions/2.5) 库。如果你想在 Python 中做同样的事情，也有一个同名的 Python 包。

在深入分析之前探索数据的好处是，您已经知道要处理什么了。如果您在文档术语矩阵或直方图中看到有许多稀疏单词，您可以决定将它们从语料库中删除。

## 7.提升你的文本挖掘技能

当您使用上一步中提到的工具对数据进行了预处理和基本的文本分析后，您还可以考虑使用您的数据集来拓展您的文本挖掘技能。

因为还有更多。

说到文本挖掘，你只看到了冰山一角。

首先，你应该考虑探索文本挖掘和自然语言处理(NLP)之间的区别。更多 R 语言的 NLP 库可以在[本页](https://web.archive.org/web/20220525174915/https://www.rdocumentation.org/taskviews#NaturalLanguageProcessing)找到。有了 NLP，你会发现命名实体识别、词性标注和解析器、情感分析……对于 Python，你可以利用 [`nltk`包](https://web.archive.org/web/20220525174915/http://www.nltk.org/)。你可以通过`nltk`包[在这里](https://web.archive.org/web/20220525174915/http://streamhacker.com/2010/05/10/text-classification-sentiment-analysis-naive-bayes-classifier/)找到一个关于情绪分析的完整教程。

除了这些软件包之外，您还可以查看更多工具，以开始深入学习和统计主题检测建模(如潜在狄利克雷分配或 LDA)等主题。下面列出了一些可用于处理这些主题的软件包:

*   Python 包:Python 包`gensim`实现了 word2vec 等等，还有[手套](https://web.archive.org/web/20220525174915/http://nlp.stanford.edu/projects/glove/)。此外，如果你想进一步发现深度学习，那么 [`theano`](https://web.archive.org/web/20220525174915/http://deeplearning.net/software/theano/) 可能也应该在你的列表上。最后，如果你想实现 LDA，使用 gensim 。

*   r 包:对于矢量化和单词嵌入的方法，使用 [text2vec](https://web.archive.org/web/20220525174915/https://www.rdocumentation.org/packages/text2vec/versions/0.3.0) 。然而，如果您对情感分析更感兴趣，那么结合使用`syuzhet`库和`tm`库可能是一个不错的选择。最后，R 的 [topicmodels](https://web.archive.org/web/20220525174915/https://www.rdocumentation.org/packages/topicmodels/versions/0.2-4) 库是统计主题检测建模的理想选择。

这些包并不是现存的全部。

由于文本挖掘是一个非常热门的话题，在过去的几年里，在研究方面有很多发现，你可以预计，随着多媒体挖掘、多语言文本挖掘等技术的出现，它在未来几年将继续发挥重要作用

## 8.不仅仅是文字——可视化您的结果

不要忘记传达你的分析结果！这可能是你能做的最棒的事情之一，因为视觉表现吸引人。

你的想象就是你的故事。

因此，不要退缩，不要把你在分析中发现的相关性或主题形象化。

对于 Python 和 R 来说，你都有特定的包来帮助你做到这一点。因此，您应该完成您必须与这些特定数据可视化库一起使用的包列表，以呈现您的结果:

对于 Python，你可以考虑使用 [`NetworkX`](https://web.archive.org/web/20220525174915/https://networkx.github.io/) 包来可视化复杂网络。然而， [`matplotlib`包](https://web.archive.org/web/20220525174915/https://pypi.python.org/pypi/matplotlib/1.5.3)也可以方便地用于其他类型的可视化。还有 [`plotly`软件包](https://web.archive.org/web/20220525174915/https://pypi.python.org/pypi/plotly)，它允许你在线制作交互式的、出版物质量的图表，是可视化展示你的结果的首选软件包之一。

一个**提示**给所有数据可视化的狂热爱好者:尝试链接 Python 和 D3，这是一个用于动态数据操作和可视化的 JavaScript 库，它允许您的观众成为数据可视化过程的积极参与者。

对于 R 来说，除了您已经知道的库之外，例如 [`ggplot2`](https://web.archive.org/web/20220525174915/https://www.rdocumentation.org/packages/ggplot2/versions/2.1.0) ，这总是一个很好的使用方法，您还可以使用 [`igraph`](https://web.archive.org/web/20220525174915/https://www.rdocumentation.org/packages/igraph/versions/1.0.1) 库来分析关注或被关注和转发的关系。你还想要更多吗？可以考虑使用 [`plotly`](https://web.archive.org/web/20220525174915/https://www.rdocumentation.org/packages/plotly/versions/4.5.2) 和 [`networkD3`](https://web.archive.org/web/20220525174915/https://www.rdocumentation.org/packages/networkD3/versions/0.2.13) 来链接 R 和 JavaScript，或者使用 [`LDAvis`](https://web.archive.org/web/20220525174915/https://www.rdocumentation.org/packages/LDAvis/versions/0.3.2) 库来交互式地可视化主题模型。

## 从 DataCamp 开始您的文本挖掘之旅

你想在 R 中开始文本挖掘吗？前往[“文本挖掘:单词袋”](https://web.archive.org/web/20220525174915/https://www.datacamp.com/courses/intro-to-text-mining-bag-of-words/)课程，通过实践学习文本挖掘！本课程是快速开始文本挖掘案例研究的最佳资源，可以向您介绍用于分析和可视化数据的各种基本主题，并在真实的案例研究中练习您所掌握的文本挖掘技能。你还没有时间开始上课吗？试试 Ted Kwartler 的交互式 R 教程。这是快速开始文本挖掘案例研究的最佳资源。

简而言之，使用 DataCamp 开始您的文本挖掘之旅吧！