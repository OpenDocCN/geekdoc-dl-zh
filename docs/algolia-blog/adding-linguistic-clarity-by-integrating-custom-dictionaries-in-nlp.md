# 使用自定义词典增加语言的清晰度& NLP

> 原文：<https://www.algolia.com/blog/engineering/adding-linguistic-clarity-by-integrating-custom-dictionaries-in-nlp/>

语言是一种有趣的东西。比如我们理所当然的认为灰姑娘穿了玻璃拖鞋。只有在童话里，人们才能穿着玻璃鞋走路。但也许这是隐喻——作为灰姑娘的脆弱。或者也许这只是一个简单的[](https://writinginmargins.weebly.com/home/cinderellas-shoes-glass-or-fur)的误译，原拉丁词“vair”(松鼠毛皮)为法语“verre”(玻璃)。

语言也很难确定。尤其是迷失在翻译中的时候。但没必要绝望——有时候失去的东西可以有 [令人惊喜和感动的结果](https://www.newyorker.com/books/page-turner/lost-in-translation-what-the-first-line-of-the-stranger-should-be) 。

但是我们并不总是想要惊喜。比如当我们直接提问或搜索与我们的查询相匹配的特定项目时。在那一点上，我们渴望变得清澈透明。

这就是字典的用武之地。字典让我们变得清晰，通过在更大的短语的上下文中加强每个单词的清晰度。我们使用字典来加强我们的自然语言处理(NLP)。以下是方法。

## [](#stop-words-and-plurals-and-compounds-and-segments-%e2%80%93%c2%a0-these-are-a-few-of-my-favorite-things)停用词和复数，以及复合词和分段——这些是我最喜欢的东西

许多用户仍然输入类似“最好的搜索引擎是什么？”而不是更短的“最佳搜索引擎”。他们说话时打字的方式很自然。但是其他人喜欢使用更短、不完整的短语来返回相同的结果。随着搜索技术的进步，即使是无意义的查询，如“引擎最佳搜索”，也会返回很好的结果。

尽管如此，完整的短语仍然很流行——对于声音来说更是如此。语音搜索的成功取决于允许人们自然地说话。而 [**停止词**](https://www.algolia.com/doc/guides/solutions/ecommerce/business-users/initial-configuration/language-settings/#configuring-stop-words) 就是关键。停用词将一个自然短语简化为其最基本的要素: **关键词** 。通过从上述查询中丢弃诸如“什么”、“是”和“该”之类的词，并且只留下关键字“最佳”、“搜索”和“引擎”，搜索引擎可以以更可靠和相关的方式将查询与底层数据进行匹配。

诚然，所有的单词都很重要——“什么”和“为什么”确实是有意义的区别——但如果搜索算法依赖于 *文本* 匹配(与意义匹配或 *语义* 相反)，它唯一的工作就是比较字符和单词。因此，通过删除停用词，可以删除匹配单词“the”的误报。

我们可以这样说 [**归一化**](https://www.algolia.com/doc/guides/managing-results/optimize-search-results/handling-natural-languages-nlp/in-depth/normalization/#what-is-normalization) (如去除重音) [**复数**](https://www.algolia.com/doc/guides/managing-results/optimize-search-results/handling-natural-languages-nlp/in-depth/language-specific-configurations/#ignoring-plurals-and-other-alternative-forms) 。任何专注于文本而非文本含义的搜索算法都应该忽略文本的变化(如复数)，以实现更相关、更明确的单词匹配。

最后，文本匹配也需要将单词分成有用的部分。“船屋”不是房子或船，而是专门用作房子的船。为了帮助达到这种精度水平，文本搜索算法需要通过使用类似于 [**分割**](https://www.algolia.com/doc/guides/managing-results/optimize-search-results/handling-natural-languages-nlp/in-depth/language-specific-configurations/#words-segmentation) 和 [**分解**](https://www.algolia.com/doc/guides/managing-results/optimize-search-results/handling-natural-languages-nlp/in-depth/language-specific-configurations/#splitting-compound-words) 的技术来分解单词的组成部分(原子)。

切分或分解的目标不是理解单词的意思，而是找出一个复杂的单词可以分解成什么。我们试图找到这个词的“原子”。我们在英语中不使用它，因为大多数单词已经被分解了，这是语言的基因。法语也一样。但德语比如*hunde hutte*，意为 狗窝 ，是由“Hund”(狗)和“hütte”(狗窝/房子)组成的。在英语中，这两个词之间已经存在的空间就是我们不需要分解的原因。分段本质上是同样的事情，但是对于根本没有空格的语言(例如，大多数亚洲语言)。

这就是字典的用武之地。

## [](#using-dictionaries-for-text-based-matching)使用字典进行基于文本的匹配

自然语言处理的一种方法是使用字典，例如停用词字典、复数字典和复合词字典。例如，您可以解析从[wikitionary](https://www.wiktionary.org/)下载的停用词列表，不仅是英语，还有许多其他语言。

下面是我们使用的过程:

*   下载完整的维基词典——单词、定义等等
*   提取文字
*   将它们存储在文本文件中
*   将它们编译成二进制格式
*   优化代码以提高性能

我们对每一种语言都这样做，而且它对大多数用例都相当有效。但是当它不起作用时，它就破坏了相关性——这是搜索引擎的一个关键障碍。下面是我们遇到的一些问题

### [](#i%e2%80%99m-%e2%80%9cdown%e2%80%9d-with-relevance)**我“倒”着了**

“羽绒”是一个合理的停用词，除非你在搜索“羽绒服”。销售“皮革”、“麂皮”和“羽绒”夹克的公司不能从查询中删除“羽绒”。

### [](#ambiguity-cloaked-in-%e2%80%9cfur%e2%80%9d-not-squirrel-fur)**暧昧披着“皮毛”(不是松鼠的皮毛)**

使用重音的语言，如法语和西班牙语，在使用重音消除进行规范化时表现良好。例如，从“瞧”到“瞧”不会失去任何意义。事实上，在法语中，去掉重音很少会造成歧义。德国人就没这么幸运了。比如重音的“a”，归一化为“a”后，会改变一些词的意思。

一个有趣的例子是德语单词*whlen*，在英语中是“选择”的意思。如果你去掉口音，大多数人不会反对——除了这个讲德语的瑞士小镇的 1500 名居民。在众多与“瓦伦”匹配的结果中，可能很难找到“瓦伦”这个城镇——因此，损害了该地区的旅游业。

解决的办法是做一个特殊的 [**自定义规范化**](https://www.algolia.com/doc/api-reference/api-parameters/customNormalization/) 为德语。在这种情况下，将“δ”归一化为“ae”。这里有一个完整的列表:

```
ä → ae
ö → oe
ü → ue
Ä → Ae
Ö → Oe
Ü → Ue
ß → ss (or SZ for capital)

```

但是这导致了第二个问题，这说明了搜索引擎在处理语言时所经历的困难。(记住，语言是搞笑的……)所以我们把“für”归一化为“fuer”，但是现在我们失去了停用词“fur”，因为现在归一化的“fuer”不是停用词。

这就是 ***自定义*词典**的用武之地。

## [](#the-solution-%e2%80%93-giving-customers-control-with-custom-dictionaries)解决方案——通过自定义词典给予客户控制权

我们意识到每个类别一个字典是不够的，我们需要为每个客户提供一个额外的字典，他们可以用它来覆盖 Wiktionary 的默认设置或添加他们自己的单词。 所以现在我们每个类别都有两本词典(停用词、复数等。):每种语言一个，我们随软件一起发布，每个客户一个自定义词典，他们可以添加单词。 添加自定义词典——也就是说，允许每个客户覆盖他们自己的单词并将其添加到我们的词典中——需要对我们处理标准词典的方式进行一些重构:每个词典检索功能都有不同的接口，每个词典数据集都有不同的格式。所以第一步是标准化我们的代码和数据。

## [](#normalizing-our-codebase-and-standardizing-our-dictionaries-interface)规范我们的代码库和标准化我们的字典接口

我们检查了作为基础产品的一部分发送给客户的当前词典。我们想抽象出每本词典的相似之处。因为他们都有相同的数据和目标，所以我们能够做以下事情:

*   创建相似数据=单词列表
*   编码相同的目标=检索单词的能力

将这些字典放入一个单一的界面，主要任务包括(按此顺序):

1.  将所有字典数据集转换为具有相同的数据结构:[*trie*](https://en.wikipedia.org/wiki/Trie)
2.  将现有词典迁移至新格式

最后，我们的复数字典文件的新格式是:

```
[2-letter country code]=[word1,word2,..]

```

为了和我们的介绍保持一致，这里有一个关于复数的好例子:

```
en=feet,feets,foot,foots
en=slipper,slippers
en=squirrel,squirrels
en=fur,furs
en=Cinderella,Cinderellas
en=Cinderfella,Cinderfellas

```

这是第一部分:统一数据的接口和结构。

这样，我们实现了以下目标:

*   一个适用于所有字典的更简单的字典接口
*   共同化的工具和测试
*   更容易维护

## [](#plugging-in-customer-created-custom-dictionaries)插在客户自创、自定义的字典里

现在，我们为每个词典提供了一个单一的接口，我们能够为每个 NLP 技术集成客户定义的单词，例如，[客户特定的停用词](https://www.algolia.com/doc/guides/managing-results/optimize-search-results/handling-natural-languages-nlp/how-to/customize-stop-words/#creating-your-own-custom-stop-word-dictionary)(参见上面的“down”示例)，客户特定的规范化(参见上面的“für”示例)，等等。

这些自定义词典被添加到静态词典之上的索引中。我们对字典查找进行了优先级排序:查询首先查询定制字典，然后查询静态字典。如果找到了这个单词，那么引擎就不需要查看静态字典。

就是这样:我们的顾客现在可以穿上一只拖鞋，帮助他们自己的顾客找到另一只拖鞋——毛皮或玻璃的。