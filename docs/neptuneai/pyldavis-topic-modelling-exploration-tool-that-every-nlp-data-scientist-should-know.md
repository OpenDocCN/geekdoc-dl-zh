# pyLDAvis:每个 NLP 数据科学家都应该知道的主题建模探索工具

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know>

你有没有想过根据新闻、论文或推文的主题对它们进行分类？知道如何做到这一点可以帮助你过滤掉不相关的文档，并且通过只阅读你感兴趣的内容来节省时间。

这就是文本分类的目的——允许您训练您的模型识别主题。这种技术允许您使用数据标签来训练您的模型，并且它是监督学习。

在现实生活中，您可能没有用于文本分类的数据标签。你可以检查每一个文档来标记它们，或者雇佣其他人来做这件事，但是这需要很多时间和金钱，尤其是当你有超过 1000 个数据点的时候。

没有训练数据，你能找到你的文档的主题吗？可以，可以用主题建模来做。

## 什么是主题建模？

使用[主题建模](https://web.archive.org/web/20221110095745/https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0)，可以对一组文档进行单词聚类。这是无监督的学习，因为它自动将单词分组，而没有预定义的标签列表。

如果你输入模型数据，它会给你不同的单词集合，每一个单词集合都描述了主题。

```py
(0, '0.024*"ban" + 0.017*"order" + 0.015*"refugee" + 0.015*"law" + 0.013*"trump" '
 '+ 0.011*"kill" + 0.011*"country" + 0.010*"attack" + 0.009*"state" + '
 '0.009*"immigration"')
(1, '0.020*"student" + 0.020*"work" + 0.019*"great" + 0.017*"learn" + '
  '0.017*"school" + 0.015*"talk" + 0.014*"support" + 0.012*"community" + '
  '0.010*"share" + 0.009*"event")
```

当你看到第一组单词的时候，你会猜测这个话题是军事和政治。看第二组单词，你可能会猜测话题是公共事件或学校。

这个挺有用的。你的文本被自动分类，不需要给它们贴标签！

## 用 pyLDAvis 可视化主题建模

主题建模是有用的，但是仅仅看上面的单词和数字的组合是很难理解的。

理解数据最有效的方法之一是通过可视化。有没有一种方法可以将 LDA 的结果可视化？是的，我们可以用[皮戴维斯](https://web.archive.org/web/20221110095745/https://github.com/bmabey/pyLDAvis)。

PyLDAvis 允许我们解释如下主题模型中的主题:

很酷，不是吗？现在我们将学习如何使用主题建模和 pyLDAvis 对推文进行分类并可视化结果。我们将分析包含 6000 条推文的真实 Twitter 数据集 [**。**](https://web.archive.org/web/20221110095745/https://datapane.com/u/khuyentran1401/reports/tweets/)

看看能找到什么话题。

## 如何开始使用 pyLDAvis 以及如何使用它

将 pyLDAvis 安装在:

```py
pip install pyldavis
```

处理数据的脚本可以在[这里](https://web.archive.org/web/20221110095745/https://ui.neptune.ai/khuyentran1401/sandbox/n/ac75203d-2de0-4bbf-a559-ec7763d919d8/e7960ea4-c002-4ea6-949f-8964c1e33491)找到。下载处理后的[数据](https://web.archive.org/web/20221110095745/https://datapane.com/u/khuyentran1401/reports/processed_tweets/)。

接下来，让我们导入相关的库:

```py
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel

from pprint import pprint

import spacy

import pickle
import re 
import pyLDAvis
import pyLDAvis.gensim

import matplotlib.pyplot as plt 
import pandas as pd
```

如果您想访问上面的数据并阅读本文，请下载数据并将数据放在当前目录中，然后运行:

```py
tweets = pd.read_csv('dp-export-8940.csv') 
tweets = tweets.Tweets.values.tolist()

tweets = [t.split(',') for t in tweets]
```

## 如何使用 LDA 模型

主题建模包括计算单词和对相似的单词模式进行分组，以描述数据中的主题。如果模型知道词频，以及哪些词经常出现在同一文档中，它将发现可以将不同的词组合在一起的模式。

我们首先将一个单词集合转换成一个单词包，单词包是一个元组列表(word_id，word_frequency)。**gensim . corpora . dictionary**是一个很好的工具:

```py
id2word = Dictionary(tweets)

corpus = [id2word.doc2bow(text) for text in tweets]
print(corpus[:1])

[[(0, 1), (1, 1), (2, 1), (3, 3), (4, 1), (5, 2), (6, 2), (7, 1), (8, 1), (9, 1), (10, 1), (11, 2), (12, 2), (13, 1), (14, 1), (15, 1), (16, 2), (17, 1), (18, 1), (19, 1), (20, 2), (21, 1), (22, 1), (23, 1), (24, 1), (25, 2), (26, 1), (27, 1), (28, 1), (29, 1), (30, 1), (31, 1), (32, 1), ... , (347, 1), (348, 1), (349, 2), (350, 1), (351, 1), (352, 1), (353, 1), (354, 1), (355, 1), (356, 1), (357, 1), (358, 1), (359, 1), (360, 1), (361, 1), (362, 2), (363, 1), (364, 4), (365, 1), (366, 1), (367, 3), (368, 1), (369, 8), (370, 1), (371, 1), (372, 1), (373, 4)]]
```

这些元组是什么意思？让我们将它们转换成人类可读的格式来理解:

```py
[[(id2word[i], freq) for i, freq in doc] for doc in corpus[:1]]

[[("'d", 1),
  ('-', 1),
  ('absolutely', 1),
  ('aca', 3),
  ('act', 1),
  ('action', 2),
  ('add', 2),
  ('administrative', 1),
  ('affordable', 1),
  ('allow', 1),
  ('amazing', 1),
...
  ('way', 4),
  ('week', 1),
  ('well', 1),
  ('will', 3),
  ('wonder', 1),
  ('work', 8),
  ('world', 1),
  ('writing', 1),
  ('wrong', 1),
  ('year', 4)]]
```

现在让我们构建一个 LDA 主题模型。为此，我们将使用[gensim . models . LDA model . LDA model](https://web.archive.org/web/20221110095745/https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel):

```py
lda_model = LdaModel(corpus=corpus,
                   id2word=id2word,
                   num_topics=10, 
                   random_state=0,
                   chunksize=100,
                   alpha='auto',
                   per_word_topics=True)

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

```

这里好像有一些**模式**。第一个话题可能是政治，第二个话题可能是体育，但模式不清楚。

```py
[(0,
 '0.017*"go" + 0.013*"think" + 0.013*"know" + 0.010*"time" + 0.010*"people" + '
 '0.008*"good" + 0.008*"thing" + 0.007*"feel" + 0.007*"need" + 0.007*"get"'),
(1,
 '0.020*"game" + 0.019*"play" + 0.019*"good" + 0.013*"win" + 0.012*"go" + '
 '0.010*"look" + 0.010*"great" + 0.010*"team" + 0.010*"time" + 0.009*"year"'),
(2,
 '0.029*"video" + 0.026*"new" + 0.021*"like" + 0.020*"day" + 0.019*"today" + '
 '0.015*"check" + 0.014*"photo" + 0.009*"post" + 0.009*"morning" + '
 '0.009*"year"'),
(3,
 '0.186*"more" + 0.058*"today" + 0.021*"pisce" + 0.016*"capricorn" + '
 '0.015*"cancer" + 0.015*"aquarius" + 0.013*"arie" + 0.008*"feel" + '
 '0.008*"gemini" + 0.006*"idea"'),
(4,
 '0.017*"great" + 0.011*"new" + 0.010*"thank" + 0.010*"work" + 0.008*"good" + '
 '0.008*"look" + 0.007*"how" + 0.006*"learn" + 0.005*"need" + 0.005*"year"'),
(5,
 '0.028*"thank" + 0.026*"love" + 0.017*"good" + 0.013*"day" + 0.010*"year" + '
 '0.010*"look" + 0.010*"happy" + 0.010*"great" + 0.010*"time" + 0.009*"go"')]

```

让我们使用 pyLDAvis 来可视化这些主题:

点击[此处](https://web.archive.org/web/20221110095745/https://ui.neptune.ai/khuyentran1401/sandbox/n/Topic-modeling-b25db361-8995-42ee-bd50-6f03fa8d5847/65d94d91-43e9-4c21-bee6-8a9bf7584087#ldavis_el55571398076541038884947444595)亲自与可视化互动。

*   每个气泡代表一个主题。气泡越大，语料库中关于该主题的推文数量的百分比越高。
*   蓝色条代表语料库中每个单词的总频率。如果没有选择主题，将显示最常用单词的蓝色条。
*   红色条给出了给定主题产生给定术语的估计次数。从下面的图片中可以看到，单词“go”大约有 22，000 个，在主题 1 中使用了大约 10，000 次。红色条最长的单词是属于该主题的推文使用最多的单词。

*   气泡之间的距离越远，它们之间的差异就越大。例如，很难区分主题 1 和主题 2。它们看起来都是关于社会生活的，但是区分话题 1 和话题 3 要容易得多。我们可以看出话题 3 是关于政治的。

一个好的主题模型会有分散在整个图表中的大而不重叠的气泡。从图中我们可以看到，气泡聚集在一个地方。我们能做得比这更好吗？

是的，因为幸运的是，有一个更好的主题建模模型叫做 LDA Mallet。

## 如何使用 LDA Mallet 模型

如果一个主题中的单词相似，我们的模型会更好，所以我们将使用主题连贯性来评估我们的模型。主题一致性通过测量主题中高分单词之间的语义相似度来评估单个主题。**好的模型会产生话题连贯性分数高的话题。**

```py
coherence_model_lda = CoherenceModel(model=lda_model, texts=tweets, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\\nCoherence Score: ', coherence_lda)

Coherence Score:  0.3536443343685833

```

这是我们的基线。我们刚刚使用了 Gensim 的内置版本的 [LDA 算法](https://web.archive.org/web/20221110095745/https://diging.github.io/tethne/api/tutorial.mallet.html)，但是有一个 LDA 模型提供了更好的主题质量，称为 [**LDA Mallet 模型**](https://web.archive.org/web/20221110095745/https://medium.com/swlh/topic-modeling-lda-mallet-implementation-in-python-part-2-602ffb38d396) 。

让我们看看是否可以用 LDA Mallet 做得更好。

```py
mallet_path = 'patt/to/mallet-2.0.8/bin/mallet' 
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)

pprint(ldamallet.show_topics(formatted=False))

coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=tweets, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\\nCoherence Score: ', coherence_ldamallet)

Coherence Score:  0.38780981858635866
```

**连贯性评分更好！**如果我们增加或减少题目数量，分数会更好吗？让我们通过微调模型来找出答案。[本教程](https://web.archive.org/web/20221110095745/https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#16buildingldamalletmodel)很好地解释了如何调整 LDA 模型。下面是文章中的源代码:

```py
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=tweets, start=2, limit=40, step=4)

limit=40; start=2; step=4;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
```

![coherence score](img/9ab10189faf582e47b37c7dbf2caac14.png)

看起来**的连贯性分数随着话题**数量的增加而增加。我们将使用具有最高一致性分数的模型:

```py
best_result_index = coherence_values.index(max(coherence_values))
optimal_model = model_list[best_result_index]

model_topics = optimal_model.show_topics(formatted=False)
print(f'''The {x[best_result_index]} topics gives the highest coherence score \\
of {coherence_values[best_result_index]}''')
```

这 34 个话题的连贯得分最高，为 0.3912。

厉害！我们得到更好的连贯性分数。让我们看看如何使用 pyLDAVis 对单词进行聚类。

为了使用 pyLDAVis 可视化我们的模型，我们需要将 LDA Mallet 模型转换成 LDA 模型。

```py
def convertldaGenToldaMallet(mallet_model):
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, eta=0,
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim

optimal_model = convertldaGenToldaMallet(optimal_model)
```

**您可以在这里** **访问调好的型号** [**。然后用 pyLDAvis 进行可视化:**](https://web.archive.org/web/20221110095745/https://github.com/khuyentran1401/Data-science/blob/master/data_science_tools/pyLDAvis/pyLDAvis.ipynb)

```py
pyLDAvis.enable_notebook()
p = pyLDAvis.gensim.prepare(optimal_model, corpus, id2word)
p
```

点击[这里](https://web.archive.org/web/20221110095745/https://ui.neptune.ai/khuyentran1401/sandbox/n/pyLDAvis-284ce842-9956-49c1-9024-4a6a0071fc0b/0ca19636-7716-4a59-887a-aa3ebc087ebc)来想象你自己。现在区分不同的话题更容易了。

*   第一个泡沫似乎是关于个人关系的
*   第二个泡沫似乎与政治有关
*   第五个泡沫似乎是关于积极的社会事件
*   第六个泡泡似乎是关于足球的
*   第七个泡沫似乎是关于家庭的
*   第 27 个泡沫似乎是关于体育的

还有很多。对于这些泡沫的话题，大家有不同的猜测吗？

## 结论

感谢阅读。希望您已经了解了主题建模是什么，以及如何用 pyLDAvis 可视化您的模型的结果。

虽然主题建模不如文本分类准确，但如果你没有足够的时间和资源给你的数据贴标签，这是值得的。为什么不先尝试一个更简单的解决方案，然后再想出更复杂、更耗时的方法呢？

可以在这里玩转本文[中的代码。我鼓励您将这些代码应用到您自己的数据中，看看您会得到什么。](https://web.archive.org/web/20221110095745/https://github.com/khuyentran1401/Data-science/tree/master/data_science_tools/pyLDAvis)

### 坤延川

Python 评估开发人员和数据科学家，喜欢尝试新的数据科学方法并为开源做出贡献。她目前每天都在 Data Science Simplified 上通过文章和日常数据科学技巧分享一点点美好的事物。

* * *

**阅读下一篇**

## 自然语言处理的探索性数据分析:Python 工具完全指南

11 分钟阅读|作者 Shahul ES |年 7 月 14 日更新

探索性数据分析是任何机器学习工作流中最重要的部分之一，自然语言处理也不例外。但是**你应该选择哪些工具**来高效地探索和可视化文本数据呢？

在这篇文章中，我们将**讨论和实现几乎所有的主要技术**，你可以用它们来理解你的文本数据，并给你一个完成工作的 Python 工具的完整之旅。

## 开始之前:数据集和依赖项

在本文中，我们将使用来自 Kaggle 的百万新闻标题数据集。如果您想一步一步地进行分析，您可能需要安装以下库:

```py
pip install \
   pandas matplotlib numpy \
   nltk seaborn sklearn gensim pyldavis \
   wordcloud textblob spacy textstat
```

现在，我们可以看看数据。

```py
news= pd.read_csv('data/abcnews-date-text.csv',nrows=10000)
news.head(3)
```

![jupyter output](img/ac19fdb9de7b54f1ed6616cfce0a5fb5.png)

数据集只包含两列，发布日期和新闻标题。

为了简单起见，我将探索这个数据集中的前 **10000 行**。因为标题是按*发布日期*排序的，所以实际上从 2003 年 2 月 19 日*到 2003 年 4 月 7 日*有**两个月。**

好了，我想我们已经准备好开始我们的数据探索了！

[Continue reading ->](/web/20221110095745/https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools)

* * *