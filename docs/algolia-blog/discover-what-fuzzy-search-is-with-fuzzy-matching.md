# 模糊搜索不仅仅是纠正打字错误

> 原文：<https://www.algolia.com/blog/engineering/discover-what-fuzzy-search-is-with-fuzzy-matching/>

术语 *模糊搜索* 有几个意思，都开启了 *近似匹配* 的思路。对该术语最常见的理解涉及 *模糊字符串匹配* ，其中搜索引擎匹配与 *不匹配的单词，确切地说是* 。典型的例子是拼写错误，拼写错误的查询会出现拼写正确的结果。但是，除了纠正错别字之外，还有更多模糊性 ，你将会看到。例如，它还可以纠正表述不当的查询，识别口语词汇，扩展前缀，或者在查询和被搜索的内容之间建立松散的类别关系。

本文中描述的模糊匹配使搜索引擎能够应用一些(稍微)不精确的匹配技术来确保它返回所有相关结果。从这个意义上说，关联不是一门精确的科学。

## [](#search-relevance-in-general)搜索相关性，一般来说

搜索的起点是 *文本匹配* ，将查询的 *精确的* 字符、字母、单词匹配到数据集中的记录。假设你搜索一艘“船”，搜索引擎会返回所有包含“船”这个词的记录。不是“独木舟”或“游船”。

文本匹配大有帮助。有了好的数据，查询“boat”将返回每个相关的记录，只要它包含单词“boat”，例如，在标题、描述和类别中。通过更智能的数据，当你搜索一艘“船”时，搜索引擎可以返回任何漂浮的东西，以及划船课程或邮轮假期。

但是，如果单词“boat”没有出现在记录中，即使数据包含与 boat 相关的信息，该怎么办呢？这就是同义词出现的地方。但如果有人想找到“独木舟”并合理地拼写成“卡努”呢？如果不允许输入错误，搜索引擎将不会返回任何结果。

模糊匹配是为了增强相关性，使相关性和整体搜索体验更加直观。当做得好的时候，模糊匹配无缝地集成了项，而 *没有* 项，这样做将会严重丧失机会。像做拼图一样，模糊搜索收集 *相似的* 棋子，帮助你定位 *确切的* 棋子。此外，因为人们在搜索 时 *定义了* 他们需要什么 *，模糊搜索基于相似性调出 *选项* ，就像在点最好的饭菜之前先尝一尝。*

Algolia(允许输入错误的模糊搜索)、Elastic(模糊搜索)、Coveo(模糊搜索)和 Constructor.io(模糊搜索规则)等搜索工具提供模糊匹配。他们都以不同的方式处理这个问题。我们将讨论 Algolia 的各种技术。

## [](#definitions-of-fuzzy-search-fuzzy-matching)模糊搜索的定义&模糊匹配

### [](#what-is-fuzzy-search)什么是模糊 ***搜索*** ？

*近似的* 字符串匹配与 *精确的* 字符串匹配相对。模糊搜索匹配两个或更多的词，即使有错别字或拼写错误。模糊搜索解决了笨拙的手指、赶时间和粗心的打字者、移动用户以及世界上每种语言的拼写的复杂性。模糊搜索还可以在匹配用户生成的数据中发挥作用，这通常是不可靠的，因为用户会对相同的单词拼写错误并使用替代或本地化的拼写。这也可以包括基于 [音标和](https://en.wikipedia.org/wiki/Phonetic_algorithm) 音的匹配。

其他词进行模糊搜索: [模糊或近似字符串匹配](https://en.wikipedia.org/wiki/Approximate_string_matching) 。

例子:

*   输入" helo "，查找带有" hello "的记录
*   输入“帮助”，查找有帮助和你好的记录

参见下面的更多例子&如何做到这一点。

### [](#what-is-fuzzy-matching)什么是模糊 ***匹配*** ？

扩展了搜索的模糊性，包括根据 *相似性* 查找信息。[](https://dataladder.com/fuzzy-matching-101/)模糊匹配是一个宽泛的术语，所以我们将只谈论基于语言的相似性，如同义词、语法(复数)、动词变化、名词词尾等。)、基于字典的技术以及其他启发式或 NLP 方法。下面，我们还将包括搜索引擎通常使用的相似性，如部分单词、短语或查询匹配，以及过滤器的使用。

举例:

*   键入“裤子”，找到裤子、长裤、休闲裤( **同义词** )
*   键入“be”，查找甲壳虫、、**前缀搜索**、 )

查看更多示例&这是如何做到的，如下。

### [](#other-usages-of-%e2%80%9cfuzzy%e2%80%9d)“模糊”的其他用法

*   [**模糊集**](https://nitsri.ac.in/Department/Computer%20Science%20&%20Engineering/FuzzyLogic.pdf) 根据 *共有特征对词语(和物体)进行分组。* 例如，一个美式足球、橄榄球和一个鸡蛋的形状使这些物品或多或少有些相似(长方形)，但足球和橄榄球更接近，因为它们也具有弹跳和运动的特性(尽管有勺子里的鸡蛋游戏)。
*   [**模糊逻辑**](https://www.geeksforgeeks.org/fuzzy-logic-introduction/?ref=lbp) 基于 *相对接近度* 构建逻辑关系，而不是像真和假这样的二进制关系(两个对象匹配的基础是两者都“高”，而不是具有完全相同的高度)。

注意，我们将排除任何基于语义、机器学习或统计技术(如 TF-IDF)的模糊匹配。

## [](#fuzzy-search-using-typo-tolerance)使用错别字公差进行模糊搜索

[错别字容忍](https://www.algolia.com/doc/guides/managing-results/optimize-search-results/typo-tolerance/) 允许用户在打字时出错，但仍能找到他们要找的记录。这是通过匹配拼写相近的单词来完成的。

### [](#what-exactly-is-a-typo)到底什么是错别字？

*   一个单词中少了一个字母，“strm”→“storm”
*   一封无关的信，“stoorm”→“storm”
*   倒字母:“风暴”→“风暴”
*   替换字母:“storl”→“storm”

### [](#spelling-distance-damerau%e2%80%93levenshtein-distance-algorithm)拼写距离(*damer au–Levenshtein 距离*)算法

Algolia 的错别字容差算法基于距离，遵循*damer au–Levenshtein 距离*算法。距离是指键入的单词与其在索引中的精确匹配之间的拼写差异。距离有一个精确的含义:它是将一个单词转换成另一个单词所需的最少操作次数(字符添加、删除、替换或换位)。完美匹配是距离= 0。当存在完全匹配，或者距离很小(一个或两个字母输入错误)时，则进行匹配并将记录添加到结果中。

例如，如果引擎收到类似“strm”的单词，这可能意味着“storm”或“strum”(距离= 1 /一个字母不正确)，或者“star”或“warm”(距离= 2 /两个字母不正确)。

距离建立了一个耐受阈值。阈值为 2:当一个单词有 3 个或更多的错误时，它是“不可容忍的”(不包括在结果中)。

### [](#more-examples-of-fuzzy-search-based-on-typo-tolerance)更多基于错别字容忍度的模糊搜索示例

下面是几个例子，说明 Algolia 如何计算转换一个单词所需的操作，以找到带有“Michael”的记录:

| 迈克尔 | 0 个错别字 |
| 米凯尔 | 1 个错别字(替换:h → k) |
| 米凯尔 | 1 个错别字(删除:h) |
| 米克海尔 | 1 个错别字(加法:k) |
| 迈克尔 | 1 个错别字(换位:一个⇄ e) |
| 米克尔 | 2 个错别字(替换:h → k，添加:l) |
| 提查尔 | 2 个错别字(替换:m → T，第一个字母) |
| tickael | 3 个错别字(替换:m → T，首字母，替换 h → k) |

### [](#ranking-and-typo-tolerance)排名和错别字公差

就相关性而言:所有基于拼写差异的模糊匹配都被认为不如精确匹配相关。因此，距离为 0 的记录(即完全匹配的记录)的排名(排序)高于有拼写错误的记录。同样，有 1 个错别字的记录比有 2 个错别字的记录排名更高。

## [](#fuzzy-matching-with-synonyms)用同义词进行模糊匹配

模糊匹配应该使用户能够边说边写，并允许他们使用自己的词汇。实现这一点的一种方法是使用同义词。

[同义词](https://www.algolia.com/doc/guides/managing-results/optimize-search-results/adding-synonyms/) 告诉引擎哪些单词和表达式被认为是相等的——例如，pants =长裤。因此，搜索“裤子”将返回“裤子” *和* “裤子”，搜索“裤子”也将返回“裤子”和“裤子”。

同义词可以是字典驱动的。你可以使用同义词词典。然而，基于词库的搜索算法是不可用的，因为它会返回太多相似的结果。每个单词在同义词库中都有多个同义词，这对于作者很有用，但对于查询一组非常特定的数据却不太有用。例如，“slacks”可能有几个意思(懒惰、散漫、不活跃)，每一个都有几个同义词。但是在一家服装店的索引中，与“休闲裤”相关的同义词只有“裤子”和“长裤”。

最后，搜索引擎可以使用机器学习来检测一段时间内常见的同义词，从而建立起一个相关同义词 [的动态列表](https://www.algolia.com/doc/guides/algolia-ai/dynamic-synonym-suggestions/) 。

## [](#other-methods-for-fuzzy-matching)其他用于模糊匹配的方法

### [](#partial-word-matching-with-prefix-search)部分词匹配带前缀搜索

[前缀匹配](https://www.algolia.com/doc/guides/managing-results/optimize-search-results/empty-or-insufficient-results/#adjusting-prefix-search) 是 Algolia 的 as-you-type 搜索体验的核心。它使引擎能够基于部分单词开始匹配记录。

例如，只要用户键入“a”、“ap”、“apr”，就会返回包含“杏子”的记录。在显示结果之前，引擎不需要等待全词匹配。

### [](#flexible-string-matching-by-searching-in-middle-of-the-word-%e2%80%9ccontains%e2%80%9d)通过在单词中间搜索进行灵活的字符串匹配("包含")

可以设置一个查询，根据记录属性的 [字位](https://www.algolia.com/doc/guides/managing-results/must-do/searchable-attributes/how-to/configuring-searchable-attributes-the-right-way/#understanding-word-position) 对记录进行匹配和排序。换句话说，匹配可以从属性的中间开始。

*   查询“ **徒步** ”匹配记录:“用于 **徒步** 和旅游的背包”。

### [](#partial-phrase-matching-with-optional-words)部分短语搭配可选词

通常，为了使记录匹配，它必须匹配查询中的所有单词。通过创建一个 [可选单词](https://www.algolia.com/doc/guides/managing-results/optimize-search-results/empty-or-insufficient-results/#creating-a-list-of-optional-words) 的列表，可以匹配只匹配部分单词的记录。

*   查询“ **背包徒步** ”匹配只卖背包的商店中的记录。商店可能想从查询中删除单词“backpacks ”(因此，使其成为可选单词),因为他们不会在产品名称中使用“backpack ”,但如果有人碰巧在查询中包含它，他们也不想错过好结果。

### [](#removing-words-if-no-results)如果没有结果就把话去掉

如果完整的短语没有返回结果，搜索引擎可以启动 [移除单词](https://www.algolia.com/doc/guides/managing-results/optimize-search-results/empty-or-insufficient-results/#adjusting-prefix-search) 。

*   查询“ **徒步背包重衣** ”，没有记录匹配(字数太多)，所以引擎去掉“重衣”，找到徒步用的背包。

### [](#fuzzy-grouping-based-on-robust-filtering-faceting)基于鲁棒滤波的模糊分组&刻面

[过滤](https://www.algolia.com/doc/guides/managing-results/refine-results/filtering/) 是所有搜索引擎的基础，也是过滤的前端面[](https://www.algolia.com/doc/guides/managing-results/refine-results/faceting/#difference-between-filters-and-facets)。向数据中添加特征(如品牌、流派)对于任何搜索、浏览和发现体验都至关重要。基于上述模糊分组，使用用户生成的类别和您自己对数据的充分理解来标记记录会创建更丰富的结果组合。

此外,“与”和“或”的过滤语法扩展了内容的分组，允许引擎组合具有多个相似性和差异性的记录。

### [](#optional-filters-filter-scoring)可选滤镜&滤镜打分

[可选筛选](https://www.algolia.com/doc/guides/managing-results/rules/merchandising-and-promoting/how-to/how-to-promote-with-optional-filters/#optional-filters) 的行为类似于普通筛选，这意味着它返回与查询和筛选都匹配的记录。但是，它也会返回与筛选器不匹配的记录。这种影响体现在排序上:匹配过滤器的记录比不匹配过滤器的记录排序更高。如果使用负筛选，则与筛选匹配的项目的排名低于其他记录。

*F* *过滤评分* 允许您引入更多记录，但控制排名，使模糊匹配的排名低于精确匹配。

## [](#so-how-do-you-contain-the-multitude-of-all-these-fuzzy-searches-and-matching)如此..你如何控制所有这些模糊的搜索和匹配

添加模糊逻辑虽然强大，但也有一些风险。它可能会返回 *太多的结果* 供用户筛选，或者它可能会返回 *意外的结果* 可能会破坏用户心目中结果的相关性。

限制信息膨胀的影响是搜索引擎的工作。例如，通过给予精确匹配更高的优先级，确保精确匹配显示在结果的顶部，在模糊匹配之上:

*   在拼写错误的情况下，精确匹配被认为比不精确匹配更相关(即使对用户来说拼写错误很明显)。
*   对于同义词，精确匹配比同义词匹配更相关(即使查询中的单词有多种含义，同义词将是更好的选择)。

本质上，引擎通过在精确匹配后显示模糊匹配来对冲模糊算法的赌注。

也有许多 UI 模式来管理这一点。例如，谷歌和其他搜索引擎会在搜索栏下方添加“建议拼写”。或者他们提供了一个 [自动完成](https://www.algolia.com/blog/ux/autocomplete-predictive-search-a-key-to-online-conversion/) 或 [查询-建议](https://www.algolia.com/products/search-and-discovery/search-autocomplete/) 功能，当用户键入时，他们可以从下拉列表中选择建议的查询(和类别)。

但正确的解决方案是表面之下的解决方案，对用户完全透明，搜索引擎返回的结果让用户产生一种直观的感觉，即显示的项目是最好和最相关的结果——即使不完全匹配查询的文本。