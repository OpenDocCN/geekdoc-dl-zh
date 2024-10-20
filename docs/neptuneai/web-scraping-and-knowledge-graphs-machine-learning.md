# 具有机器学习的知识图[指南]

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/web-scraping-and-knowledge-graphs-machine-learning>

你需要在网上获取一些信息。比如关于尤塞恩博尔特的几段话。你可以从维基百科复制并粘贴信息，这不会有太多的工作。

但是，如果您需要获得关于尤塞恩·博尔特参加的所有比赛的信息，以及关于他和他的竞争对手的所有相关统计数据，该怎么办呢？如果你想对所有运动都这样做，而不仅仅是跑步，会怎么样？

机器学习工程师通常需要建立如上例的复杂数据集来训练他们的模型。网络搜集是收集必要数据的一种非常有用的方法，但也带来了一些挑战。

在这篇文章中，我将解释如何收集公开可用的数据，并从收集的数据中构建知识图，以及来自[自然语言处理](https://web.archive.org/web/20221206032313/https://www.datacamp.com/community/tutorials/web-scraping-python-nlp) (NLP)的一些关键概念。

## 什么是网页抓取？

Web 抓取(或 web 采集)是用于数据提取的数据抓取。**一词通常指用机器人或网络爬虫**收集数据。这是一种复制形式，从 web 上收集并复制特定数据，通常复制到本地数据库或电子表格中，以供以后使用或分析。

你可以用在线服务、API 来做网络抓取，或者你可以写你自己的代码来做。

网络抓取有两个关键要素:

*   爬虫:爬虫是一种算法，通过浏览互联网上的链接来浏览网页以搜索特定的数据。
*   **抓取器**:抓取器从网站中提取数据。刮刀的**设计可以有很大变化**。这取决于**项目的复杂程度和范围**。最终，它必须快速准确地提取数据。

现成库的一个很好的例子是 Wikipedia scraper 库。它为你做了很多繁重的工作。您向 URL 提供所需的数据，它从这些站点加载所有的 HTML。scraper 从这个 HTML 代码中获取您需要的数据，并以您选择的格式输出数据。这可以是 excel 电子表格或 CSV，或者类似于 [JSON](https://web.archive.org/web/20221206032313/https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Objects/JSON) 的格式。

## 知识图表

网络上可用的内容数量已经令人难以置信，而且还在以越来越快的速度增长。数十亿个网站与万维网相连， [**搜索引擎**](/web/20221206032313/https://neptune.ai/blog/building-search-engine-with-pre-trained-transformers-guide) 可以通过这些链接，以极高的精度和速度提供有用的信息。这部分归功于知识图表。

不同的组织有不同的知识图表。例如， [**谷歌知识图**](https://web.archive.org/web/20221206032313/https://en.wikipedia.org/wiki/Google_Knowledge_Graph) 是一个知识库**，谷歌及其服务利用从各种来源收集的信息来增强搜索引擎结果**。为了更好的用户体验，以及存储和检索有用的信息，脸书或亚马逊的产品也使用了类似的技术。

知识图没有正式的定义。广义地说，KG 是一种添加了约束的语义网络。它的范围、结构和特征，甚至它的用途在开发过程中还没有完全实现。

将知识图和机器学习(ML)结合在一起可以系统地提高系统的准确性，并扩展机器学习能力的范围。**得益于知识图，从机器学习模型推断出的结果将具有更好的可解释性和可信度**。

将知识图和 ML 结合在一起创造了一些有趣的机会。在我们可能没有足够数据的情况下，可以使用 kg 来扩充训练数据。最大似然模型的主要挑战之一是解释最大似然系统做出的预测。通过将解释映射到图中的适当节点并总结决策过程，知识图可以帮助克服这个问题。

另一种看待它的方式是，知识图存储从信息提取任务中产生的数据。KG 的许多实现利用了一个叫做**三元组**的概念——一组三个项目(**一个主语、一个谓语和一个宾语**)，我们可以用它们来存储关于某件事情的信息。

还有另一种解释:知识图是一种数据科学工具，处理相互关联的实体 **(** 组织、人员、事件、地点 **)** 。**实体是通过边连接的节点**。kg 具有实体对，可以遍历这些实体对来发现非结构化数据中有意义的连接。

节点 A 和节点 B 是两个不同的实体。这些节点由表示它们之间关系的边连接起来。这是我们能制造的最小公斤——也被称为三公斤。知识图表有各种形状和大小。

## 网络抓取、计算语言学、NLP 算法和图论(带 Python 代码)

唷，这是一个冗长的标题。无论如何，要从文本构建知识图，帮助我们的机器理解自然语言很重要。我们使用 NLP 技术来实现这一点，例如**句子分割、依存解析、词性(POS)标注和实体识别**。

构建 KG 的第一步是收集您的资源——让我们在网上搜索一些信息。维基百科将是我们的来源(经常检查数据来源，网上很多信息都是假的)。

对于这个博客，我们将使用[维基百科 API](https://web.archive.org/web/20221206032313/https://github.com/martin-majlis/Wikipedia-API/) ，一个直接的 Python 包装器。 [Neptune](/web/20221206032313/https://neptune.ai/) 在一个地方管理模型构建元数据。记录、存储、显示、组织、比较和查询您所有的 MLOps 元数据。

[实验跟踪](/web/20221206032313/https://neptune.ai/experiment-tracking)以及为运行大量实验的研究和生产团队构建的模型注册表。

### 安装和设置

#### 安装依赖项并抓取数据

```py
!pip install wikipedia-api neptune-client neptune-notebooks pandas spacy networkx scipy
```

按照以下链接在您的笔记本上安装和设置 Neptune:rn-[Neptune](https://web.archive.org/web/20221206032313/https://docs.neptune.ai/getting-started/installation#install-neptune-client-for-python)
–[Neptune Jupyter 扩展指南](https://web.archive.org/web/20221206032313/https://docs.neptune.ai/integrations-and-supported-tools/ide-and-notebooks/jupyter-lab-and-jupyter-notebook/install-neptune-notebooks-jupyter-extension-3-min#about-neptune-notebooks)

下面的函数在维基百科中搜索给定的主题，并从目标页面及其内部链接中提取信息。

```py
import wikipediaapi  
import pandas as pd
import concurrent.futures
from tqdm import tqdm
```

below 函数允许您根据作为函数输入提供的主题获取文章。

```py
def scrape_wikipedia(name_topic, verbose=True):
   def link_to_wikipedia(link):
       try:
           page = api_wikipedia.page(link)
           if page.exists():
               return {'page': link, 'text': page.text, 'link': page.fullurl, 'categories': list(page.categories.keys())}
       except:
           return None

   api_wikipedia = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
   name_of_page = api_wikipedia.page(name_topic)
   if not name_of_page.exists():
       print('Page {} is not present'.format(name_of_page))
       return

   links_to_page = list(name_of_page.links.keys())
   procceed = tqdm(desc='Scraped links', unit='', total=len(links_to_page)) if verbose else None
   origin = [{'page': name_topic, 'text': name_of_page.text, 'link': name_of_page.fullurl, 'categories': list(name_of_page.categories.keys())}]

   with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
       links_future = {executor.submit(link_to_wikipedia, link): link for link in links_to_page}
       for future in concurrent.futures.as_completed(links_future):
           info = future.result()
           origin.append(info) if info else None
           procceed.update(1) if verbose else None
   procceed.close() if verbose else None

   namespaces = ('Wikipedia', 'Special', 'Talk', 'LyricWiki', 'File', 'MediaWiki',
                 'Template', 'Help', 'User', 'Category talk', 'Portal talk')
   origin = pds.DataFrame(origin)
   origin = origin[(len(origin['text']) > 20)
                     & ~(origin['page'].str.startswith(namespaces, na=True))]
   origin['categories'] = origin.categories.apply(lambda a: [b[9:] for b in a])

   origin['topic'] = name_topic
   print('Scraped pages', len(origin))

   return origin
```

让我们测试主题为“新冠肺炎”的函数。

```py
wiki_data = wiki_scrape('COVID 19')

```

```py
o/p: Links Scraped: 100%|██████████| 1965/1965 [04:30<00:00,  7.25/s]ages scraped: 1749

```

将数据保存到 csv:

```py
data_wikipedia.to_csv('scraped_data.csv')
```

导入库:

```py
import spacy
import pandas as pd
import requests
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

from spacy.tokens import Span
from spacy.matcher import Matcher

import matplotlib.pyplot as plot
from tqdm import tqdm
import networkx as ntx
import neptune.new as neptune

%matplotlib inline
```

```py
run = neptune.init(api_token="your API key",
                   project="aravindcr/KnowledgeGraphs")
```

上传数据到海王星:

```py
run["data"].upload("scraped_data.csv")
```

在这里下载数据[。在](https://web.archive.org/web/20221206032313/https://github.com/AravindR7/Web_Scraping_Knowledge_Graphs/blob/main/scraped_data.zip)[号海王](https://web.archive.org/web/20221206032313/https://app.neptune.ai/aravindcr/KnowledgeGraphs/e/KNOW-9/all?path=&attribute=data)号上也有:

```py
data = pd.read_csv('scraped_data.csv')

```

查看第 10 行的数据:

```py
data['text'][10]

```

输出:

```py
The AbC-19 rapid antibody test is an immunological test for COVID-19 exposure
developed by the UK Rapid Test Consortium and manufactured by Abingdon
Health. It uses a lateral flow test to determine whether a person has IgG
antibodies to the SARS-CoV-2 virus that causes COVID-19. The test uses a single
drop of blood obtained from a finger prick and yields results in 20 minutes.
```

### 句子分割

构建知识图的第一步是将文本文档或文章分割成句子。然后我们把例子限制在只有一个主语和一个宾语的简单句上。

```py
docu = nlp('''The AbC-19 rapid antibody test is an immunological test for COVID-19 exposure developed by
the UK Rapid Test Consortium and manufactured by Abingdon Health. It uses a lateral flow test to determine
whether a person has IgG antibodies to the SARS-CoV-2 virus that causes COVID-19\. The test uses a single
drop of blood obtained from a finger prick and yields results in 20 minutes.nnSee alsonCOVID-19 rapid
antigen test''')

for tokn in docu:
   print(tokn.text, "---", tokn.dep_)
```

下载如下所示的预训练空间模型:

```py
python -m spacy download en
```

SpaCy 管道分配单词向量、上下文特定的标记向量、词性标记、依存解析和命名实体。通过扩展 SpaCy 的注释管道，您可以解析共同引用(在下面的代码中解释)。

知识图可以从词性和依存句法分析中自动构建。使用 NLP 库空间，从语法模式中提取实体对是快速的，并且可扩展到大量文本。

下面的函数**将实体对定义为由根动词**连接的具有主客体依赖关系的实体/名词块。可以使用其他近似方法来产生不同类型的连接。这种联系可以称为主谓宾三元组。

主要意思是浏览一个句子，提取主语和宾语，以及它们何时出现。下面的函数包含了上面提到的一些步骤。

### 实体提取

您可以借助词性( **POS** )标签从句子中提取单个单词实体。名词和专有名词将是实体。

然而，当一个实体包含多个单词时，仅靠 POS 标签是不够的。我们需要分析句子的依存关系树。构建知识图，最重要的是它们之间的节点和边。

这些节点将是出现在维基百科句子中的实体。边是连接这些实体的关系。我们将以无人监督的方式提取这些元素，也就是说，我们将使用句子的语法。

这个想法是通过一个句子，当主语和宾语被重构时，提取它们。

```py
def extract_entities(sents):

   enti_one = ""
   enti_two = ""

   dep_prev_token = "" 

   txt_prev_token = "" 

   prefix = ""
   modifier = ""

   for tokn in nlp(sents):

       if tokn.dep_ != "punct":

           if tokn.dep_ == "compound":
               prefix = tokn.text

               if dep_prev_token == "compound":
                   prefix = txt_prev_token + " "+ tokn.text

           if tokn.dep_.endswith("mod") == True:
               modifier = tokn.text

               if dep_prev_token == "compound":
                   modifier = txt_prev_token + " "+ tokn.text

           if tokn.dep_.find("subj") == True:
               enti_one = modifier +" "+ prefix + " "+ tokn.text
               prefix = ""
               modifier = ""
               dep_prev_token = ""
               txt_prev_token = ""

           if tokn.dep_.find("obj") == True:
               enti_two = modifier +" "+ prefix +" "+ tokn.text

           dep_prev_token = tokn.dep_
           txt_prev_token = tokn.text

   return [enti_one.strip(), enti_two.strip()]
```

```py
extract_entities("The AbC-19 rapid antibody test is an immunological test for COVID-19 exposure developed by the UK Rapid Test")
```

```py
['AbC-19 rapid antibody test', 'COVID-19 UK Rapid Test']
```

现在让我们使用函数来提取 800 个句子的实体对。

```py
pairs_of_entities = []
for i in tqdm(data['text'][:800]):
   pairs_of_entities.append(extract_entities(i))

```

句子中的主语和宾语对:

```py
pairs_of_entities[36:42]

```

输出:

```py
[['where aluminium powder', 'such explosives manufacturing'],
 ['310  people', 'Cancer Research UK'],
 ['Structural External links', '2 PDBe KB'],
 ['which', '1 Medical Subject Headings'],
 ['Structural External links', '2 PDBe KB'],
 ['users', 'permanently  taste']]

```

### 关系抽取

有了实体提取，一半的工作就完成了。为了构建知识图，我们需要连接节点(实体)。这些边是节点对之间的关系。下面的函数能够从这些句子中捕获这样的谓词。我用的是 spaCy 的基于规则的匹配。函数中定义的模式试图找到**词根**或句子中的主要动词。

```py
def obtain_relation(sent):

   doc = nlp(sent)

   matcher = Matcher(nlp.vocab)

   pattern = [{'DEP':'ROOT'},
           {'DEP':'prep','OP':"?"},
           {'DEP':'agent','OP':"?"},
           {'POS':'ADJ','OP':"?"}]

   matcher.add("matching_1", None, pattern)

   matcher = matcher(doc)
   h = len(matcher) - 1

   span = doc[matcher[h][1]:matcher[h][2]]

   return (span.text
```

上面写的模式试图在句子中找到词根。一旦它被识别出来，它就会检查它后面是否跟有介词或代理词。如果答案是肯定的，那么它将被添加到词根中。

```py
relations = [obtain_relation(j) for j in tqdm(data['text'][:800])]
```

提取的最常见关系:

```py
pd.Series(relations).value_counts()[:50]

```

## 让我们建立一个知识图表

现在我们终于可以从提取的实体中创建一个知识图了

让我们使用 [***networkX***](https://web.archive.org/web/20221206032313/https://networkx.org/documentation/stable/tutorial.html) 库来绘制网络。我们将创建一个节点大小与度中心性成比例的有向多图网络。换句话说，任何连接的节点对之间的关系都不是双向的。它们只是从一个节点到另一个节点。

```py
source = [j[0] for j in pairs_of_entities]

target = [k[1] for k in pairs_of_entities]

data_kgf = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

```

*   我们使用 networkx 库从数据帧创建一个网络。
*   这里，节点将被表示为实体，而边表示节点之间的关系

```py
graph = ntx.from_pandas_edgelist(data_kgf, "source", "target",
                         edge_attr=True, create_using=ntx.MultiDiGraph())
```

```py
plot.figure(figsize=(14, 14))
posn = ntx.spring_layout(graph)
ntx.draw(graph, with_labels=True, node_color='green', edge_cmap=plot.cm.Blues, pos = posn)
plot.show()
```

*   从上面的图表中，不清楚在图表中捕捉到了什么关系
*   让我们用一些关系来形象化图表。我在这里选择:

```py
graph = ntx.from_pandas_edgelist(data_kgf[data_kgf['edge']=="Information from"], "source", "target",
                         edge_attr=True, create_using=ntx.MultiDiGraph())

plot.figure(figsize=(14,14))
pos = ntx.spring_layout(graph, k = 0.5) 
ntx.draw(graph, with_labels=True, node_color='green', node_size=1400, edge_cmap=plot.cm.Blues, pos = posn)
plot.show()

```

*   另一个用关系名“链接”过滤的图可以在这里找到[。](https://web.archive.org/web/20221206032313/https://app.neptune.ai/aravindcr/KnowledgeGraphs/e/KNOW-9/all?path=graphs&attribute=filtered_relations2)

### 记录元数据

我已经把上面的 networkx 图登录到[海王星](https://web.archive.org/web/20221206032313/https://docs.neptune.ai/you-should-know/logging-metadata)了。你可以找到那个特定的[路径](https://web.archive.org/web/20221206032313/https://app.neptune.ai/aravindcr/KnowledgeGraphs/e/KNOW-9/all?path=graphs)。根据获得的输出，将您的图像记录到不同的路径。

```py
run['graphs/all_in_graph'].upload('graph.png')
run['graphs/filtered_relations'].upload('info.png')
run['graphs/filtered_relations2'].upload('links.png')
```

所有的图表都可以在这里找到[。](https://web.archive.org/web/20221206032313/https://app.neptune.ai/aravindcr/KnowledgeGraphs/e/KNOW-9/all?path=graphs)

## 共指消解

要获得更精确的图形，还可以使用共指解析。

共指消解是内视感知的 NLP 等价物，用于信息检索系统、对话代理和虚拟助手，如 Alexa。这是一项对文本中提及相同潜在实体的内容进行聚类的任务。

***【我】【我的】*** **，和** ***【她】*** **属于同一个集群，和** ***【乔】*** **和** ***【他】*** **属于同一个集群。**

解决共同引用的算法通常寻找与引用表达式兼容的最近的前面提及。也可以训练神经网络，而不是使用基于规则的依存解析树，神经网络将单词嵌入和提及之间的距离作为特征来考虑。

通过规范化文本、删除冗余和指定实体代词，这显著改进了实体对提取。

如果您的用例是特定于领域的，那么训练一个[定制实体识别模型](https://web.archive.org/web/20221206032313/https://spacy.io/usage/training#ner)是值得的。

知识图表可以自动构建并探索，以揭示关于该领域的新见解。

笔记本上传到[海王星](https://web.archive.org/web/20221206032313/https://app.neptune.ai/aravindcr/KnowledgeGraphs/n/00f8da9b-76d9-4740-a06f-30b8356eea32/35fe83d7-323b-4f6e-bfe3-eb994d9f8ca2)。

[GitHub](https://web.archive.org/web/20221206032313/https://github.com/AravindR7/Web_Scraping_Knowledge_Graphs/blob/main/Web_Scraping_Knowledge_Graphs_ML.ipynb) 上的笔记本。

## 大规模知识图表

为了有效地将 1749 页的整个语料库用于我们的主题，使用在 ***wiki_scrape*** 函数中创建的列来为每个节点添加属性。然后你可以跟踪每个节点的页面和类别。您可以使用多重和并行处理来减少执行时间。

kg 的一些使用案例包括:

## 未来的挑战

### 实体歧义消除和身份管理

在其最简单的形式中，挑战是为一个实体的发言或提及分配一个唯一的规范化身份和类型。

自动提取的很多实体都有非常相似的表面形态，比如同名或相近的人，或者同名或相近的电影、歌曲、书籍。两个名称相似的产品可能指的是不同的列表。没有正确的链接和歧义消除，实体将与错误的事实不正确地关联，并导致下游不正确的推理。

### 类型成员和解析

今天，大多数知识图系统允许每个实体有多种类型，不同的环境有不同的类型。古巴可以是一个国家，也可以指古巴政府。在某些情况下，知识图系统将类型分配推迟到运行时。每个实体描述其属性，应用程序根据用户任务使用特定的类型和属性集合。

### 管理不断变化的知识

一个有效的实体链接系统需要基于其不断变化的输入数据有机地增长。例如，公司可能会合并或拆分，新的科学发现可能会将单个现有实体拆分为多个实体。

当一家公司收购另一家公司时，收购公司是否会改变身份？身份是随着姓名权的获得而产生的吗？例如，在医疗保健行业构建的 kg 的情况下，患者数据会随着时间的推移而改变。

### 从多个结构化和非结构化来源中提取知识

结构化知识(包括实体、它们的类型、属性和关系)的提取仍然是一个全面的挑战。大规模增长的图表需要手动方法以及从开放领域的非结构化数据中进行无监督和半监督的知识提取。

### 管理大规模运营

管理规模是直接影响与性能和工作负载相关的几项运营的根本挑战。它还会影响其他操作，例如管理大规模知识图的快速增量更新，从而间接证明自己。

*注:有关不同科技巨头如何在其产品和相关挑战中实施[行业规模知识图的更多细节，请查看本文](https://web.archive.org/web/20221206032313/https://queue.acm.org/detail.cfm?id=3332266)。*

## 自然语言处理

自然语言处理(Natural Language Processing)是计算机科学的一个分支，旨在让计算机能够处理和理解人类语言。从技术上来说，NLP 的主要**任务将是给计算机编程，以分析和处理大量的自然语言数据**。

各种学科都学习语言。每个学科都有自己的问题和解决问题的方法。

### 语言中的歧义

**NLP 中使用的歧义**可以指的是以一种以上的方式被理解的**能力**。自然语言是模糊的。NLP 有以下模糊之处:

*   **词汇歧义**是单个单词的歧义。例如，单词 **well** 可以是副词、名词或动词。
*   **句法歧义**是在一个句子或单词序列中存在两种或两种以上可能的意思。例如“鸡肉可以食用了”。这句话要么表示鸡已经熟了，现在可以吃了，要么表示鸡已经可以喂了。
*   **回指歧义**是指在文本中向后指称(或另一个语境中的实体)。一个短语或单词指的是前面提到的东西，但是有不止一种可能性。例如，“玛格丽特邀请苏珊来访，她给了她一顿美餐。”(她=玛格丽特；她=苏珊)。“玛格丽特邀请苏珊来访，但她告诉她她必须去工作”(她=苏珊；她=玛格丽特。)
*   **语用歧义**可以定义为有多种解释的词语。当一个句子中的词语意义不明确时，就会产生语用歧义；它以不同的意义结束。

### 自然语言处理中的文本相似性度量

文本相似性用于确定两个文本文档在上下文或含义方面有多相似。有各种相似性度量，例如:

*   **余弦相似度，**
*   **欧几里德距离，**
*   **Jaccard 相似度。**

所有这些指标都有自己的规格来衡量两个查询之间的相似性。

#### 余弦相似性

[余弦相似度](https://web.archive.org/web/20221206032313/https://en.wikipedia.org/wiki/Cosine_similarity)是在 NLP 中测量两个文档之间的文本相似度的度量，不管它们的大小如何。**一个词用向量的形式来表示。文本文档用 n 维向量空间表示。**

余弦相似性度量两个 n 维向量在多维空间中投影的夹角余弦。两个文档的余弦相似度将从 **0 到 1** 的范围**变化**。如果余弦相似性得分是 1，这意味着 2 个向量具有相同的方向。越接近 0 的值表示 2 个文档的相似度越低。

两个非零向量的余弦相似度的数学方程为:

余弦相似度是比 [**欧几里德距离**](https://web.archive.org/web/20221206032313/https://en.wikipedia.org/wiki/Euclidean_distance#:~:text=In%20mathematics%2C%20the%20Euclidean%20distance,being%20called%20the%20Pythagorean%20distance.) 更好的度量，因为如果两个文本文档相距欧几里德距离很远，它们在上下文方面仍然有可能彼此接近。

#### 雅克卡相似性

Jaccard 相似性也称为 Jaccard 指数和 Union 上的交集。

[Jaccard 相似度](https://web.archive.org/web/20221206032313/https://en.wikipedia.org/wiki/Jaccard_index)用于确定两个文本文档之间的相似度，即**所有单词中存在多少个常用单词。**

Jaccard 相似性被定义为两个文档的交集除以两个文档的并集，这两个文档是指总单词数中的共同单词数。

Jaccard 相似性的数学表示为:

Jaccard 相似性得分在 **0 到 1** 的**范围**内。如果两个文档相同，Jaccard 相似度为 1。如果两个文档之间没有共同的单词，则 Jaccard 相似性得分为零。

#### Python 代码查找 Jaccard 相似性

```py
def jaccard_similarity(doc1, doc2):

 words_doc1 = set(doc1.lower().split())
 words_doc2 = set(doc2.lower().split())

 intersection = words_doc1.intersection(words_doc2)

 union = words_doc1.union(words_doc2)

 return float(len(intersection)) / len(union)
```

```py
docu_1 = "Work from home is the new normal in digital world"
docu_2 = "Work from home is normal"

jaccard_similarity(docu_1, docu_2)
```

**输出:0.5**

**doc_1** 和 **doc_2** 之间的 Jaccard 相似度为 **0.5**

**以上三种方法都有一个相同的假设:**文档(或句子)如果有常用词就是相似的。这个想法很直白。它适合一些基本情况，如比较前两个句子。

然而，通过比较第一个和第三个句子(例如，尝试使用传达相同意思的不同句子，并使用上面的 Python 函数来比较相似性)，分数可以相对较低，即使两者描述了相同的新闻。

另一个**限制是上述方法不处理同义词**。例如,“buy”和“purchase”应该有相同的意思(在某些情况下),但是上述方法会对这两个词进行不同的处理。

那么解决方法是什么呢？你可以使用单词嵌入(Word2vec，GloVe，FastText)。

对于 NLP 的一些基本概念和用例，我将附上一些我在 Medium 上写的文章，以及一篇在 [Neptune 的博客](/web/20221206032313/https://neptune.ai/blog)上写的文章，以供参考:

## 结论

我希望你在这里学到了一些新的东西，这篇文章帮助你理解了 web 抓取、知识图和一些有用的 NLP 概念。

感谢阅读，继续学习！

**参考文献:**