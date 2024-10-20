# 如何优化搜索索引以获得最佳性能

> 原文：<https://www.algolia.com/blog/engineering/search-indexing-best-practices-for-top-performance-with-code-samples/>

每个搜索界面都依赖于一个快速的 **后端数据索引过程** ，尽可能及时地更新搜索结果。但是搜索索引只是硬币的一面。另一面是高质量相关搜索引擎的实时速度。

对于所有搜索引擎来说，**的*搜索请求*优先级最高，其次是*索引*** **(非常)接近。**这有几个原因，但最重要的是一个商业论点:每个搜索都是潜在的游戏规则改变者，是一条转化之路。任何缓慢或丢失的搜索请求，或不相关的结果，都是潜在的财务或业务损失。

为了达到最大的速度和相关性，搜索引擎必须:

*   搜索请求优先于索引请求
*   构建索引，使查询以最佳相关性实时(毫秒)执行

因此，更新一个索引需要一点额外的时间。但是，如果你学会遵循一些索引的最佳实践，你会扯平的。

“一切都很好，”全栈和后端开发人员说。“我明白搜索的优先级。但是我想知道更多关于我的数据。我如何将我的数据上传到您的服务器上？它能处理我的用例吗？它接受任何类型的数据吗？它简单、安全、快速吗？”

在最近一篇关于索引 的 [文章中，我们探讨了各种高级用例，并关注于两个搜索索引要素:快速更新和广泛适用性。现在是时候深入研究代码，解释一些速度提升算法和索引最佳实践，确保您在任何搜索用例 *中获得最高的索引速度。*](https://www.algolia.com/blog/product/an-exploration-of-search-and-indexing-real-time-indexing-scenarios/)

这里主要关注两个方面:

*   我们指数的广泛适用性
*   我们的索引 API 的高性能更新您的数据

## [](#the-wide-applicability-of-our-search-indexing)我们搜索索引的广泛适用性

为了理解索引本身，我们需要将它从搜索中分离出来，并概述最流行的索引场景:

### [](#indexing-for-search)索引搜索

一个 [结构良好的索引](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/) 为一个快速且功能齐全的面向客户的搜索界面提供了基础，具有很强的相关性。事实上，索引对于搜索相关性是如此重要，以至于它需要像前端一样精心设计和实现。

### [](#indexing-to-create-a-company-wide-multi-purpose-searchable-data-layer)索引创建公司范围内的多用途可搜索数据层

[多个指标](https://www.algolia.com/doc/guides/sending-and-managing-data/manage-indices-and-apps/manage-indices/concepts/one-or-more-indices/) 可以形成所有后台数据的单一接触点。当以某种方式组合在一起时，您的索引可以创建一个公司范围的可搜索数据层，该数据层位于您的后台和所有内部(员工)或外部(客户、合作伙伴)使用的前端之间。

### [](#indexing-as-a-%e2%80%9cmatchmaker%e2%80%9d-%e2%80%93-the-collaborative-indexing-use-case)索引作为“媒人”——协同索引用例

“媒人”场景是 X 公司建立一个 Algolia 指数，并向外部数据提供商提供。在这个场景中，公司 X 构建了一个协作网站，例如一个 [市场](https://www.algolia.com/industries-and-solutions/marketplaces/) 或流平台，在那里它显示多个供应商、合作伙伴和贡献者的产品/媒体。为了实现这一点，X 公司将其 Algolia 索引公开给这些外部数据提供者，允许他们在理解格式后发送数据。

前两种场景的主要区别如下:

*   单个搜索界面需要至少一个索引，在构建时应考虑到该界面。
*   在 [全公司](https://www.algolia.com/industries-and-solutions/enterprise/) 数据层场景中，就不一样了:你需要概括你的指标的结构。组成这个多用途数据层的数据需要结构化，以便(a)允许来自差异很大的后台应用程序的多种数据输入，以及(b)服务于多种用例及界面，无论是面向用户的还是系统到系统的。

## [](#what-about-indexing-performance)索引性能怎么样？

如果不是在所有情况下都有效，我们的索引就不可能有广泛的适用性，也无法在竞争激烈的数字商业环境中生存。虽然我们提供开箱即用的高索引速度，但这取决于实施最佳索引实践。这就是这篇文章的内容。

简单说说我们所说的“开箱即用的高性能”是什么意思。我们的索引采用了以下技术:

*   使用高级索引技术的搜索引擎
*   高性能 [裸机服务器](https://www.algolia.com/doc/guides/scaling/servers-clusters/) 针对性能进行配置
*   全球可用的基于集群的云基础架构，具有低延迟和服务器冗余(即无服务器停机)
*   一个 API 用重试的方法来保证(合同上) [99.99%的可用性](https://www.algolia.com/policies/sla/)

## [](#best-practices-for-fast-indexing-performance-with-code-snippets)快速索引性能的最佳实践(带代码片段)

最重要的索引实践是运行一个 **批处理算法** ，该算法在一次索引操作中定期、及时地更新多个记录。对于所有的用例都是如此。

为什么我们推荐批处理？因为每个索引请求都有很小的性能代价。索引请求包括对整个索引进行小的“重新索引”,这可能需要 1 秒钟，如果索引非常大，可能需要更长时间。因此，发送数百个索引请求(一次一条记录)会创建一个索引队列，从而降低整个索引过程的速度。为了减轻这种情况，通过发送较少的请求来限制服务器上的索引请求数量是很重要的。

考虑到所有这些，这里有 3 个最重要的索引最佳实践(数据更新的标准费用):

1.  分批更新，而不是一次发送一条记录的更新
2.  增量更新代替完全(重新)索引
3.  部分索引(仅更新已更改的属性)

### [](#1-batch-indexing-instead-of-updating-one-record-at-a-time)1–批量索引而不是一次更新一条记录

一个常见的错误是一次发送一条记录。如果您的后端数据不断变化，那么在每次发生变化时都发送是错误的。如上所述，当您创建一个等待处理的 100 个索引请求的队列时，就会出现瓶颈。

相反，作为最佳做法，使用 [批量标引](https://www.algolia.com/doc/guides/sending-and-managing-data/send-and-update-your-data/how-to/sending-records-in-batches/) 。您将每个更改发送到一个临时缓存，然后定期将该缓存发送到 Algolia，例如，对于较大的索引，每 5 分钟或 30 分钟发送一次。批处理速度不要超过 1 分钟，因为这样会造成瓶颈。

这个程式码范例会建立新的索引。它使用 [Algolia 的 Python API](https://www.algolia.com/doc/api-client/getting-started/install/python/?client=python) 的 [`save_objects`](https://www.algolia.com/doc/api-reference/api-methods/save-objects/) 方法*批量保存* 10000 个记录块。

```
#python

import json
from algoliasearch import algoliasearch

client = algoliasearch.Client('YourApplicationID', 'YourAdminAPIKey')
algolia_index = client.init_index('bubble_gum')

with open('bubble_gum.json') as f:
  records = json.load(f)

  chunk_size = 10000
  for i in range(0, len(records), chunk_size):
    algolia_index.save_objects(records[i:i + chunk_size])

```

看看我们的 API 如何[自动化批处理过程](https://www.algolia.com/doc/guides/sending-and-managing-data/send-and-update-your-data/how-to/sending-records-in-batches/#using-the-api) 。

### [](#2-incremental-updates-instead-of-full-indexing)2–增量更新代替完全索引

根据前面的建议，您不希望在一个批处理中发送太多的记录。为了减小索引请求的大小，您应该执行[增量更新](https://www.algolia.com/doc/guides/sending-and-managing-data/send-and-update-your-data/how-to/incremental-updates/)，其中您只更新新记录 。

此代码增加了一个新的泡泡糖系列。

```
#python

algolia_index.save_objects([
  {"objectID": "myID1", "item": "Classic Bubble Gum", "price": "3.99"},
  {"objectID": "myID2", "item": "Raspberry Bubble Gum", "price": "3.99"},
  {"objectID": "myID3", "item": "Cherry Bubble Gum", "price": "3.99"},
  {"objectID": "myID4", "item": "Blueberry Bubble Gum", "price": "3.99"},
  {"objectID": "myID5", "item": "Mulberry Bubble Gum", "price": "3.99"},
  {"objectID": "myID6", "item": "Lemon Bubble Gum", "price": "3.99"}
])

```

注意:最好每晚或至少每周对所有记录进行一次完整的重新索引。

查看我们完整的[增量更新解决方案](https://www.algolia.com/doc/guides/sending-and-managing-data/send-and-update-your-data/how-to/incremental-updates/#replacing-the-old-record)。

### [](#3-partial-indexing-updating-only-changed-attributes)3–部分索引(仅更新更改的属性)

为了进一步降低索引流量，您需要只发送已经改变的 *属性* ，而不是整个记录。为此，您将使用一个 [`partial indexing`](https://www.algolia.com/doc/api-reference/api-methods/partial-update-objects/) 策略。

这个代码只改变一些泡泡糖的价格，没有其他属性。

```
#python

algolia_index.save_objects([
  {'objectID': 'myID1', 'price': 4.99},
  {'objectID': 'myID3', 'price': 4.99},
  {'objectID': 'myID6', 'price': 2.99}
])

```

查看我们完整的[部分索引解决方案。](https://www.algolia.com/doc/guides/sending-and-managing-data/send-and-update-your-data/in-depth/the-different-synchronization-strategies/#partial-record-updates)

## [](#next-readings)下一个读数

我们关于索引的第一篇文章展示了标准和高级索引用例的高级概述。本文向您介绍了索引最佳实践和标准索引过程的实现细节。我们的下一篇文章将讨论如何在高级用例中优化索引。

我们剩余的文章将为我们讨论的一些高级索引用例提供前端&后端代码，从实时定价开始。

要开始索引，您可以 [免费上传您的数据，](https://www.algolia.com/users/sign_up) 或从我们的搜索专家那里获得一个 [定制演示](https://www.algolia.com/demorequest/) 。