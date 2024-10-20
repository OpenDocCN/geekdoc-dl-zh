# 通过爬虫或 API 改善搜索索引的 12 种方法

> 原文：<https://www.algolia.com/blog/engineering/12-ways-to-improve-your-search-index/>

开始新业务时，搜索索引通常是我们与客户讨论的第一个话题。无论是大型企业级网站还是小型电子商务商店，向网站添加搜索的第一步是通过网站爬虫或 API 索引您的内容。站点的架构、模式和内容都会影响索引。

在这篇文章中，我们将涉及许多我们与客户讨论的话题，并分享一些改进搜索索引的实用技巧。

*注意:这不是一篇关于 SEO 的文章。虽然* [*站内搜索优化和 SEO*](https://www.algolia.com/blog/ux/google-core-web-vitals-seo-rankings-search-bar-optimization/) *是相关的——你为站内搜索优化搜索索引的工作也有助于谷歌搜索或必应的可见性——它们满足不同的需求。搜索引擎优化是面向互联网的可见性，而现场搜索解决用户体验。然而，XML 站点地图、内部链接、元标签等。，你为一个人创造会帮助另一个人！*

## [](#what-is-a-search-index)**什么是搜索索引？**

一个 [搜索索引](https://www.algolia.com/blog/product/what-is-a-search-index-and-how-does-it-work/) 帮助用户在一个网站上快速找到信息。它旨在将搜索查询映射到网页、文档或其他站点内容。这类似于一本书的索引。它允许用户使用关键字快速找到有用的信息，但与书籍相比有许多技术优势，例如帮助访问者更快地找到他们想要的东西。搜索索引既可以通过网络爬虫创建，也可以通过 API 访问创建，但两者在不同的情况下都有各自的优势。

## [](#what-is-full-text-search)**什么是全文搜索？**

全文搜索需要索引你网站上的每一个词，以使搜索引擎在许多记录中导航变得容易。传统上，全文搜索引擎使用“倒排索引”，实质上是文档中所有关键字以及这些关键字位置的映射。

![full text search](img/a01e1d28558efa799df1e9636a88d0f9.png)

在上面的例子中，关键词“便携”和“声音”不在索引中，但是一个人工智能支持的搜索引擎理解上下文来提供很好的结果。

人工智能支持的搜索引擎现在可以超越关键词来理解上下文，以提供更丰富的结果。以查询“便携音”为例。如果基于关键字的搜索引擎在索引中具有术语“便携式”和“扬声器”，则结果页面可能包括正确的项目。通过机器学习，即使关键词不在网站上，你也可以通过检测上下文和单词之间的相似性来获得好的结果。例如，机器可以学习“便携”一词与“手持”、“移动”、“电话”相似，都是在含义上接近的*，但不一定是同义的 。*

## [](#search-crawlers-and-apis)**搜索爬虫和 API**

构建搜索引擎索引有两种主要方式——搜索爬虫或通过 API 直接从数据库中提取数据。每种方法对不同的情况都有好处。

举例来说，对于大多数静态网站，一个 [爬虫](https://www.algolia.com/products/search-and-discovery/crawler/) 就可以了。又快又全面。 [API 驱动的索引](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/) 是拥有动态或不断变化的数据的网站的理想选择。API 有自己的优势，比如快速添加新数据源的能力。

## [](#what-is-fast-indexing)**什么是快速索引？**

当您添加新内容或更改现有内容时，您希望结果可以实时搜索。 [快速索引](https://www.algolia.com/blog/product/an-exploration-of-search-and-indexing-real-time-indexing-scenarios/) 是零售商和品牌销售新产品或发起活动的必备。有时，当我们的客户在快速索引方面遇到问题时，通常是由于这样的问题:

*   由于复杂的 API 架构问题，内容索引速度不够快
*   内容在索引中，但不在结果中显示
*   PDF 和 DOC 文件无法索引

大多数问题都可以相对较快地解决。首先要做的是检查爬虫如何看待你的网站文档，或者你的数据管道是否阻塞。使用 sitemap.xml 文件帮助爬虫总是一个好的实践，可以帮助你的内容快速地被索引。如果你通过 API 索引你的站点，很可能有一个需要解决的集成问题。

为了帮助完成这一切，并简化索引过程，我们提供了多种编程语言的 API 客户端、帮助您可视化索引和爬行过程的仪表板，以及以各种方便的方式与 API 交互的 CLI 工具。

## [](#12-ways-to-optimize-and-enrich-your-search-index)**12 种方法优化和丰富你的搜索索引**

无论你是使用搜索爬虫还是通过 API 连接你的站点，都有很多方法来配置和改进搜索索引。下面的实际建议直接来自我们经常与通过爬虫或 API 构建索引的客户的对话。其中一些方法更适合基于爬虫的索引，另一些与 API 索引相关，还有一些两者都相关。

这里有 12 种方法可以优化你的搜索索引

### [](#1-open-graph-metadata)**1。打开图形元数据**

脸书在 2010 年发布了他们的 [开放图协议](https://en.wikipedia.org/wiki/Facebook_Platform#Open_Graph_protocol) ，从那以后它被搜索引擎广泛使用。搜索结果通常包括图像预览，通常由 Open Graph 提供支持。

通过在你的内容中添加开放图形标签，你可以用如下信息来改进搜索索引:

*   带有内容类型的标题
*   图片和网址
*   添加额外的打开图形数据

除了标题、描述和图片之外，Open Graph 还可以使用大量其他数据来丰富搜索索引，但许多人不知道或不使用它们。更多信息，请访问[https://ogp.me/](https://ogp.me/)

### [](#2-schema-org-formats)**2。Schema.org 格式**

Open Graph 只是丰富 web 和搜索引擎索引数据的几个开放协议之一。有不同种类的模式可以用来标记页面内容。例如，如果你是一个食谱网站，你将有不同的标准来标记内容，比如说，一个活动网站。

Schema.org 为不同类型的网站发布和维护不同的模式词汇表。例如，对于音乐会、讲座或节日等活动，可以通过 HTML(或 JSON-LD)格式的标记添加票务信息，如<a class = " local link " href = "/offers ">offers</a>属性。重复的事件可以被构造为单独的事件对象。

### [](#3-article-publish-and-modified-times)**3。文章发布和修改次数**

文章发布和文章修改日期/时间对于能够按新近度对内容进行排序是非常重要的。开放图形或 schema.org 格式都支持时间戳。

|  | 文章:发布时间–日期时间–文章首次发布的时间。 |
|  | 文章:修改时间–日期时间–文章上次修改的时间。 |

### [](#4-identify-header-and-footer-content)**4。识别页眉和页脚内容**

各种各样的内容，比如你的导航，页脚，以及任何不特定于页面的内容，都应该放在页眉和页脚标签中，这样搜索引擎就会忽略它。通过标记页眉和页脚的内容，您可以让搜索引擎更好地理解页面的内容，从而可以正确地对其进行索引—在这种情况下，是导航数据与正文数据。

### [](#5-augmenting-your-search-index)**5。增加您的搜索索引**

搜索索引可以通过多种方式增加数据，例如:

*   通过 Google Vision API 添加颜色元数据
*   使用第三方数据，如产品评级
*   提取输入数据，用于创建过滤器和面

随着新信息添加到索引中，数据可能会增强。搜索引擎利用这些数据来提供更好的结果，让消费者更容易、更快地找到他们想要的东西。电子商务网站经常定期更新它们的项目，并且丰富的数据可以在更新期间被合并。

### [](#6-business-performance-data)**6。经营业绩数据**

你的索引多于你的内容。非现场数据，如产品等级、利润、库存水平等。，对于搜索索引来说非常有用，有助于结果排名。可能有许多产品与搜索您网站的客户相关，但您的业务数据可以用来增强结果，以确保最好的产品被推到顶部。我们提供 [自定义排名](https://www.algolia.com/doc/guides/managing-results/must-do/custom-ranking/)[助推](https://www.algolia.com/doc/guides/managing-results/must-do/custom-ranking/how-to/boost-or-penalize-some-records/) ，帮助客户利用这类业务数据构建转换飞轮。

### [](#7-merchandising-and-campaign-data)**7。营销和活动数据**

许多零售商进行季度、季节或假日销售。通过向站点索引添加销售和活动数据，您可以调整结果以显示销售项目。

您可以添加特定的销售字段或使用折扣字段来计算何时有销售。在后一种情况下，搜索引擎将知道你的展示价格低于你的常规价格，这有助于对打折商品进行排序，以帮助访问者找到最佳的节省。然后，您还可以使用一种算法(通过我们的 [排名公式](https://www.algolia.com/doc/guides/managing-results/relevance-overview/in-depth/ranking-criteria/) )根据不同商品的销售状况或其他属性对其进行提升。

![merchandising](img/1fca2e8d2fd6c4ccb4b0529b2a1e39e6.png)

*搜索索引应包括可用于构建过滤器和方面的字段和数据*

### [](#8-filters)**8。过滤器**

可以使用您的搜索索引构建搜索过滤器和方面。我们可以自动推断和创建过滤器(例如，使用 [查询分类](https://www.algolia.com/doc/guides/algolia-ai/query-categorization/) )，但是您也可以在需要时设计定制过滤器。决定提供最佳过滤器的关键在于了解您的客户以及他们希望如何分割您的产品。查看我们关于 [过滤器和刻面](https://www.algolia.com/blog/ux/filters-vs-facets-in-site-search/) 的指南，了解更多信息。

### [](#9-content-type)**9。内容类型**

有不同的元标签可以帮助搜索索引按类型理解内容。内容会把访问者带到视频、文档、页面或其他地方吗？使用 HTML 或 JSON-LD 标签将您的内容标识为视频、音频、摘要等。，帮助您的搜索索引按类型对内容进行排序或筛选。

### [](#10-personalization)**10。个性化**

[客户期望](https://www.forbes.com/sites/jimvinoski/2020/01/20/new-research-shows-consumers-already-expect-mass-personalization-time-to-get-ready/?sh=2da3ffb8223e) ，希望搜索结果个性化。如果你为会员提供免费送货，这些信息应该在数据中。如果有按位置的折扣，那么您也会希望在记录中包含地理数据。通过将您的搜索索引连接到这些数据，您可以轻松地对搜索结果进行个性化设置。

### [](#11-integration-with-other-third-party-systems)**11。与其他第三方系统集成**

[大型企业](https://www.algolia.com/blog/product/what-is-enterprise-search-and-how-does-it-benefit-your-organizations-employees-and-consumers/)通常拥有复杂的基础设施，数据来自不同的系统。需要将 [的数据](https://www.algolia.com/blog/product/unlock-organizational-knowledge-with-searchable-company-wide-data/) 与你的供应链管理或者[PIM](https://en.wikipedia.org/wiki/Product_information_management)整合吗？您将希望您的搜索解决方案支持一个 API 来实现系统间的即时数据索引。

### [](#12-review-your-analytics-and-search-metrics)**十二。查看您的分析和搜索指标**

网站所有者应该计划花一些时间审查他们的 [分析和搜索指标](https://www.algolia.com/products/search-and-discovery/analytics/) ，以确定客户正在查询的关键词。了解客户如何搜索有助于发现丰富索引、添加或调整过滤器以及改进搜索引擎结果的机会。

## [](#its-all-about-indexing-great-content)一切都是为了索引伟大的内容

构建丰富的搜索索引可以极大地提高搜索性能和客户满意度。通过了解可以包含的不同类型的数据，网站所有者可以确保他们为客户提供最佳的搜索体验。

要了解更多关于如何设置您的搜索索引或利用我们的个性化和定制排名功能， [今天就联系我们](https://www.algolia.com/contactus/) ！我们提供了一个 [免费试用版](https://www.algolia.com/users/sign_up) 和 [演示版](https://www.algolia.com/demorequest/) ，以便您可以探索我们的解决方案所能提供的一切。