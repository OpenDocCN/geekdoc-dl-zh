# 分面搜索，每个 JSON 属性都很重要

> 原文：<https://www.algolia.com/blog/engineering/facets-and-faceted-search-making-every-attribute-count/>

什么是刻面？什么是分面搜索？什么样的数据最能代表方面？

快速定义:一个 *[面](https://www.algolia.com/doc/guides/managing-results/refine-results/faceting/)* 是一个屏幕上的 *[过滤器](https://www.algolia.com/doc/guides/managing-results/refine-results/filtering/)*允许终端用户缩小他们的搜索范围，给他们更多的控制他们的搜索结果的相关性。电子商务产品的典型方面是类似“品牌”或“价格”的属性，方面的 *值* 是各个品牌和价格 es。通过单击一个方面值，用户可以包括和排除整个产品类别。例如，通过在“品牌”类别中选择“苹果”，用户可以排除除苹果产品之外的所有产品。

> ***这就是我们所说的*** **分面搜索体验** ***，其中类别和常用术语就像搜索栏中的文本一样驱动着搜索。每一个伟大的在线搜索体验都提供了一个多方面的搜索。***

## [](#the-data-behind-faceted-search%c2%a0)刻面搜索背后的数据

在本文中，我们从头开始，从数据到 UI，来看方面。具体来说，我们描述了如何使用 JSON 来创建一个出色的多面搜索体验。**精心设计的分面数据策略可确保有效过滤和高速性能。**它还使得前端开发人员能够轻松编码，这在快节奏的编码环境中是一个关键的考虑因素。

### [](#facets-clean-up-your-data%c2%a0)刻面清理你的数据

好的搜索总是从 [开始清理数据](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/in-depth/prepare-data-in-depth/) 。最好的搜索索引是结构良好的，写得很好的，不包含任何误导或无关的内容。An[*index*](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/#algolia-index)是一组可搜索的数据，索引就是创建索引的过程。这些是搜索技术中描述搜索引擎如何构建数据的标准术语。尽管每个引擎的数据结构不同，但最终结果总是一个索引。我们使用数据和索引这些词，而不是数据集、数据库或任何其他类似的术语。

每个搜索索引都包括方面，因为它们组织数据并确保简单性和完整性。请考虑以下情况:第一个示例不包含任何方面；下一个包含方面。

#### **没有刻面**

标题:星球大战:第五集——帝国反击战

描述:70 年代的流行科幻电影，一部美国太空歌剧，卢克·天行者与汉·索洛、莱娅公主和丘巴卡一起，对抗达斯·维德和义军同盟，拯救银河帝国。这部电影由乔治·卢卡斯创作，卢卡斯影业制作，现在由华特·迪士尼电影公司拥有和发行。演员阵容包括马克·哈米尔、哈里森·福特、凯丽·费雪、比利·迪·威廉姆斯、安东尼·丹尼尔斯、大卫·鲍罗斯、肯尼·贝克、彼得·梅靥和弗兰克·奥兹。这部电影以 4.4 亿美元成为 1980 年票房最高的电影。

#### **带刻面**

标题:星球大战:第五集——帝国反击战

**描述**:一部美国太空歌剧。卢克·天行者与汉·索洛、莱娅公主和丘巴卡一起，抗击达斯·维德和义军同盟，拯救了银河帝国。

**类型**:科幻、太空歌剧

**时代**:70 后，80 后

故事:乔治·卢卡斯

传奇:星球大战

**工作室**:卢卡斯影业，华特·迪士尼工作室

**制作事实**:该片由卢卡斯影业制作，现归华特·迪士尼电影制片厂所有。

演员:马克·哈米尔、哈里森·福特、凯丽·费雪、比利·迪·威廉姆斯、安东尼·丹尼尔斯、大卫·鲍罗斯、肯尼·贝克、彼得·梅靥、弗兰克·奥兹。

角色:卢克·天行者、汉·索洛、莱娅公主、丘巴卡、达斯·维德、约迪、C-3PO、R2-D2

如你所见，我们大大缩短了“描述”,直接切入主题。我们把它的内容分成了几个不同的[](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/#attributes-for-searching)，像“制作事实”、“流派”、“演员”和“传奇”。其中一些属性将显示在搜索结果中。然而，大部分“描述”被分解成对搜索和分面有用的信息块。这一分析过程是为最终用户创建卓越的可搜索索引和多面搜索体验的关键一步。

> **以这种方式将你的数据分解成小块的信息，确保每个属性都有价值。这是使用刻面**的一个好处

基本原理是:一个属性中的大量文本不如多个属性中的较小文本块适合搜索，其中许多可以作为刻面。有了这些面块，[用户可以过滤他们的结果](https://www.nngroup.com/articles/filters-vs-facets/) 。一些例子:

*   包括乔治·卢卡斯制作和编剧的所有电影。
*   排除除科幻片、次类型太空歌剧以外的所有电影。
*   看完所有《星球大战》电影，感谢“传奇”的一面。

大多数公司已经有了结构化数据，所以在构建可搜索索引时，只需要转移这种结构。另一方面，一些公司有多个数据源，所以他们必须在合并这些数据源的同时设计新的结构。

### [](#faceted-search-%e2%80%94-facets-are-not-only-for-filtering)刻面搜索—刻面不仅用于过滤

用小块信息建立你的数据不仅能让你的搜索引擎过滤记录。它还将注意力集中在信息上，使得 *未过滤的* 搜索更加精确。这只有在您将 facet 属性定义为[*searchable*](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/how-to/setting-searchable-attributes/)的情况下才有可能，也就是说，您告诉您的搜索引擎在查看“标题”和“描述”之前先查看 facet 的“年份”和“类型”属性。现在，用户可以键入“70 年代的科幻电影”并找到 星球大战 *，而无需使用过滤器* 。

## [](#let%e2%80%99s-get-technical-%e2%80%94-engine-level-faceting-json)让我们来看看技术——发动机级刻面& JSON

在索引中构建方面时，有很多技术上的考虑:

*   表示输入到索引中的方面。
*   将方面表示为搜索请求的输入。
*   响应于搜索请求接收方面。
*   用 AND、OR、NOT 组合刻面。
*   引擎如何表示刻面。

在本文中，我们只讨论第一项——如何将数据分解并表示为面，从而为面过滤和面搜索奠定基础。虽然我们触及了其他方面，但我们将把更详细的讨论留到以后的文章中。

### [](#representing-facets-as-input-into-your-search-index)表示输入到你的搜索索引中的面

最佳刻面数据格式允许:

*   一组完全可定制的方面，由您的应用/业务需求决定。
*   一个容易理解的数据结构。
*   灵活性，这样索引中的每条记录都可以有不同的属性集。
*   允许多个方面值和类别层次结构的可伸缩结构。

总之， [JSON](https://www.json.org/json-en.html) 。

### [](#what-is-json-vs-relational-databases)什么是 JSON vs .关系数据库

这里有一段非常非常简短的历史: [关系数据库](https://en.wikipedia.org/wiki/Relational_database) 建立了一个标准，使用了外键、原子数据和其他类似的对以前数据技术的改进。关系数据库允许人们通过将数据库中每一项的特征分布在一个小型的、定义明确的链接表系统中，来使他们的数据正常化。它创造了同质性，具有可靠且易于理解的一致结构。然而，这种同质性并不总能满足具有不同数据需求的每个应用程序的需求。因此，解决方案是创建具有不同结构的多个数据库，有时具有相同的底层信息—这是一个需要额外资源的难以维护的解决方案。

[实体-属性-值](https://en.wikipedia.org/wiki/Entity%E2%80%93attribute%E2%80%93value_model) (EAV)模型为关系模型增加了灵活性:现在，在同一个数据库中，每一项都可以用不同的方式定义。EAV 完全是关于异构性的——允许单个数据库以有效的方式包含各种各样的信息，其中每个项目都可以有自己独特的完整的描述性标签集，称为 *键/值。EAV 通过使用多种模式来保持数据不重复的原则，并且经常将其唯一值局限在一个表中。*

JSON 既包含了 EAV 原则，又脱离了 EAV 的关系根源 ，最引人注目的是取消了表和行的模式，放宽了数据不重复的原则。JSON 不断重复信息，这就是它如此易于使用、如此强大和灵活的原因。不再有约束——每个记录项都存在于真空中，包含其自己独特的属性集。也没有实体必须遵循的预定义模式。现在，只有 JSON 对象(或记录)、键和值。

### [](#why-is-json-so-good-for-building-a-searchable-index%c2%a0)为什么 JSON 对于构建可搜索的索引如此之好？

搜索引擎需要原子地 对待其索引 *中的每一条记录，而不依赖于与其他记录的关系。它需要*de-*规范化它的数据，而不是像关系数据库那样规范化它。*

此外，引擎需要知道记录包含什么，因此能够搜索任何索引，无论是商业产品、电影还是专业服务。它只需要返回任何匹配搜索查询的记录。大多数搜索引擎不依赖语义；而是依靠 *文本匹配* ，也就是说一个搜索引擎只需要将一个物品属性中的内容与搜索栏中的文本进行匹配即可。匹配可以是完全的或部分的。JSON 以其可读性将数据完全集中在文本内容上。

### [](#json-keyvalues-are-easy-to-read-easy-to-search)JSON 键/值易于阅读，易于搜索。

```
[
  {
    "objectID": 42,
    "title": "Star Wars: Episode V - The Empire Strikes Back",
    "type": “movie”,
    "genre": ["science fiction", "space opera"],
    "series": "star wars"
  },
  {
    "objectID": 43,
    "title": "LEGO: Star Wars Yoda",
    "type": “toy”,
    "brand": “Lego”,
    "series": "star wars"
  }
]
```

这里发生的事情很清楚。我们有两张唱片。一个是电影，另一个是玩具。它们有不同的属性:电影有“类型”，玩具有“品牌”。但是，它们都有“系列”，可以有选择地用来链接“星球大战”上的两个记录。

一些搜索引擎直接使用 JSON 格式作为搜索的基础，不需要转换。然而，大多数将这些数据转换成他们自己专有的索引格式。无论哪种情况，过程都是一样的:遍历每个记录和键，然后保存(或搜索)值。最好的算法除了关键字和值之外什么都不期望——尽管大多数引擎要求最少的必需属性。例如，“对象 ID”——每个记录的唯一标识符，用于在索引时查找和更新记录。但不要被误导。“objectID”的功能不像关系数据库中的主键。在搜索中，主键/外键关系几乎没有用处。有趣的是，搜索依赖于方面而不是键来链接记录。你可以在上面的例子中看到:玩具和电影通过“系列”方面联系在一起。

### [](#rest-apis-and-the-joys-of-generating-and-visualizing-json)REST API 以及生成和可视化 JSON 的乐趣

和 JSON 一样，REST API 通过提供 SOAP API 的替代品简化了标准。它这样做的原因相似:用更少的形式增加更多的灵活性。SOAP 基于一个具有严格规范和规则的协议。REST API 消除了这一点。它提供了一个端点，并接受不同的数据格式，比如 CSV、XML 和 JSON。由端点和 REST API 开发人员与用户交流格式化数据中要发送的内容。最好的 API 不需要太多具体的数据。他们关注完成工作所需的信息。

例如，下面是如何搜索:

```
curl -X POST \
     -H "API-Key: 123” \
     -H "Application-Id: app_movies” \
     --data-binary '{ "params": "query=star wars” }' \
     "https://some-movie-api.com/indexes/movies/query" 
```

“数据-二进制”字段包含全部信息:使用这些参数执行搜索。在这种情况下——毫不奇怪:这是对“星球大战”的搜索。

### [](#from-curl-to-api-clients)从 Curl 到 API 客户端

下面是如何 *使用 API 客户端在名为“app-movies”的索引中创建* 记录。 最好的搜索引擎都是把 Curl 繁琐的语法包装成不同编程语言的 API。例如，下面的代码使用了一个 JavaScript API 客户端:

```
index.search(“star wars”);
index.save('{
"ojectID": 42, "title":"Start Wars: Episode V - The Empire Strikes Back","Description":"An American space opera, Luke Skywalker, along with Han Solo, Princess Leia, and Chewbacca, fight Darth Vader and the Rebel Alliance to save the Galactic Empire.","Genre": ["science fiction, space opera"],"Era": ["70s, 80s"]","story":"George Lucas","Saga: Star Wars","Studio":["Lucasfilm, Walt Disney Studios"],"Production facts":"The film is produced by Lucasfilm, now owned by Walt Disney Studios films.","Actors":["Mark Hamill, Harrison Ford, Carrie Fisher, Billy Dee Williams, Anthony Daniels, David Prowse, Kenny Baker, Peter Mayhew, Frank Oz"],"Characters":["Luke Skywalker, Han Solo, Princess Leia, Chewbacca, Darth Vader, Yodi, C-3PO, R2-D2]"
}' ); 
```

无论您使用哪种格式，Curl 还是 JavaScript，您都可以看到在本文开头封装完整的“星球大战:第五集——帝国反击战”示例是多么容易。并以一种人类- *和* 的机器可读格式发送。

## [](#parting-words-and-a-few-more-details-on-faceted-search)临别赠言及更多关于刻面搜索的细节

重要的一点是，刻面不是可选的。他们在搜索体验的每一个方面都发挥着作用。在功能上，方面用于搜索数据、过滤记录和对结果进行排序。从技术上来说，他们组织你的数据并确保其质量和准确性，这是创造优质搜索体验的核心。

另一个要点是，如何在索引中构建方面决定了它们的功能有效性。现代搜索需要灵活性，以适应每个在线业务的快速变化。它还需要多样性来管理大量不断变化的用户口味和需求。JSON 实现了这种灵活性和多样性。开发人员需要一种像 JSON 格式这样的无模式数据结构，以尽可能好的方式表示方面和过滤的多样性。例如，嵌套属性。我们将在本系列的第 2 部分刻面搜索中看到这一点。

* * *

这是我们的 Facets & Data 系列的第一篇文章。本系列的重点是技术性的，概述了方面搜索的逻辑和方面数据模型。

*   第一篇文章定义了什么是分面，并解释了分面在结构化数据中的重要作用。它还说明了 JSON 是表示包括方面在内的索引数据的最灵活的方式。
*   在本文(第二篇)中，我们介绍了刻面最常见的数据结构:简单的刻面值、嵌套刻面、层次类别以及用户和人工智能标记，所有这些都用于刻面搜索的不同方面。
*   第三篇文章——[用动态分面](https://www.algolia.com/blog/engineering/implementing-faceted-search-with-dynamic-faceting-with-code/)实现分面搜索——继续以数据为中心焦点，但是我们现在讨论访问数据背后的*过程*。我们看一下查询搜索过程，从查询到执行再到响应，并展示如何动态生成方面。