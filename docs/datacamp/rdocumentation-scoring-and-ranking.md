# 记录:评分和排名

> 原文：<https://web.archive.org/web/20230101103415/https://www.datacamp.com/blog/rdocumentation-scoring-and-ranking>

RDocumentation.org 的核心特征之一是它的搜索功能。从一开始，我们就希望有一个超级简单的搜索栏，可以找到您想要的东西，而不需要复杂的表单来询问包名、函数名、版本或其他任何信息。只是一个简单的搜索栏。

在今天的[技术博客](https://web.archive.org/web/20220704235014/https://tech.datacamp.com/)帖子中，我们将重点介绍我们用来为用户提供相关且有意义的结果的技术和技巧。

## 弹性搜索

RDocumentation.org 使用 [Elasticsearch](https://web.archive.org/web/20220704235014/https://www.elastic.co/products/elasticsearch) 来索引和搜索所有的 R 包和主题。

> Elasticsearch 是一个开源、可扩展、分布式的企业级搜索引擎。

Elasticsearch 非常适合查询文档，因为它不使用传统的 SQL 数据，而是将文档存储在类似 JSON 的数据结构中。每个文档只是一组简单数据类型(字符串、数字、列表、日期等)的*键值*对。它的分布式本质意味着弹性搜索可以非常快。

一个 Elasticsearch 集群可以有多个索引，每个索引可以有多个*文档类型*。文档类型只是描述文档的结构应该是什么样子。要了解更多关于弹性搜索类型的信息，你可以访问 Elasticsearch 指南[。](https://web.archive.org/web/20220704235014/https://www.elastic.co/guide/en/elasticsearch/guide/current/mapping.html)

RDocumentation.org 使用三种不同的类型:`package_version`、`topic`和`package`。前 2 个为主；稍后再讨论`package`。

因为 RDocumentation.org 是开源的，你可以在我们的 [github repo](https://web.archive.org/web/20220704235014/https://github.com/datacamp/RDocumentation-elasticsearch/blob/master/mappings_rdoc.json) 中看到 Elasticsearch 的映射。

### 包 _ 版本类型

`package_version`类型就像一个包的`DESCRIPTION`文件的翻译，它以你可以在那里找到的主字段为特色；`package_name`、`version`、`title`、`description`、`release_date`、`license`、`url`、`copyright`、`created_at`、`updated_at`、`latest_version`、`maintainer`和`collaborators`。`maintainer`和`collaborators`是从`DESCRIPTION`文件的`Authors`字段中提取的

### 主题类型

主题文档解析自 r 文档的标准格式`Rd`文件，`topic`类型有以下键:`name`、`title`、`description`、`usage`、`details`、`value`、`references`、`note`、`author`、`seealso`、`examples`、`created_at`、`updated_at`、`sections`、`aliases`和`keywords`。

## 弹性搜索评分

在进行任何评分之前，Elasticseach 首先通过检查文档是否与查询匹配来减少候选集。基本上，查询是一个单词(或一组单词)。基于查询设置，Elasticsearch 在特定类型的特定字段中搜索匹配。

然而，匹配并不一定意味着文档是相关的；同一个词在不同的上下文中可能有不同的意思。基于查询设置，我们可以按类型和字段进行过滤，并包含更多上下文信息。这种上下文信息将提高相关性，这就是得分的地方。

Elasticsearch 在幕后使用了 [Lucene](https://web.archive.org/web/20220704235014/http://lucene.apache.org/core/) ，因此评分是基于 Lucene 的[实用评分函数](https://web.archive.org/web/20220704235014/https://lucene.apache.org/core/4_6_0/core/org/apache/lucene/search/similarities/TFIDFSimilarity.html#formula_coord)，该函数汇集了一些模型，如 [*TF-IDF*](https://web.archive.org/web/20220704235014/https://www.elastic.co/guide/en/elasticsearch/guide/current/scoring-theory.html#tfidf) 、 [*向量空间模型*](https://web.archive.org/web/20220704235014/https://www.elastic.co/guide/en/elasticsearch/guide/current/scoring-theory.html#vector-space-model) 和 [*布尔模型*](https://web.archive.org/web/20220704235014/https://www.elastic.co/guide/en/elasticsearch/guide/current/scoring-theory.html#boolean-model) 对文档进行评分。

如果你想更多地了解如何在 Elasticsearch 中使用该功能，你可以查看 Elasticsearch 指南的这一部分。

提高相关性的一个方法是对某些字段进行增强。例如，在 RDocumentation.org 全搜索中，我们自然会提升包的字段`package_name`和`title`，主题的字段`aliases`和`name`。

### 提升流行文档

提高相关性的另一个有效方法是根据文档的受欢迎程度来提升文档。背后的想法是，如果一个包更受欢迎，用户更有可能搜索这个包。首先显示更受欢迎的包将增加我们显示用户实际寻找的可能性。

#### 用下载量来衡量受欢迎程度

衡量受欢迎程度有多种方法。我们可以使用用户给出的投票或排名等直接指标(如亚马逊产品的评分)，或销售商品数量或观看次数(对于 YouTube 视频)等间接指标。

在 RDocumentation.org，我们选择了后者。更具体地说，我们用下载次数来衡量受欢迎程度。间接测量通常更容易收集，因为它们不需要用户主动输入。

#### 时间框架

使用下载数量时出现的一个问题是，旧包的总下载量自然会比新包多。这并不意味着它们更受欢迎，然而，它们只是存在的时间更长而已。如果一个包在几年前非常流行，但现在已经过时，不再被社区积极使用，该怎么办？

为了解决这个问题，我们只考虑最近一个月的下载次数。这样，旧包的受欢迎程度就不会被人为地提高，过时的包也会很快消失。

#### 直接下载与间接下载

另一个问题来自反向依赖。r 包通常依赖于大量的其他包。有很多反向依赖的包会比其他包下载得更多。然而，这些包通常是更低级的，最终用户不直接使用。我们必须小心不要给他们的下载量太多的权重。

以 [Rcpp](https://web.archive.org/web/20220704235014/https://www.rdocumentation.org/packages/Rcpp/versions/0.12.9) 为例。CRAN 上超过 70%的软件包，综合 R 档案网络，依赖于这个包，这显然使它成为下载最多的 R 包。然而，很少有 R 用户会直接使用这个包并搜索它的文档。

为了解决这个问题，我们需要将*直接下载*(因为用户请求而发生的下载)和*间接下载*(因为依赖包被下载而发生的下载)分开。为了从 CRAN 日志中区分直接和间接下载，我们使用了 Arun Srinivasan 在 [cran.stats 包](https://web.archive.org/web/20220704235014/https://www.rdocumentation.org/packages/cran.stats/versions/0.1/topics/stats_logs)中描述的启发式方法。

我们现在有了一个有意义的流行度指标:*上个月的直接下载数量*。Elasticsearch 提供了注入这些额外信息的简单方法；更多细节，请看[关于 elastic.co](https://web.archive.org/web/20220704235014/https://www.elastic.co/guide/en/elasticsearch/guide/current/boosting-by-popularity.html)的这篇文章。

分数修改如下:

```py
new_score = old_score * log(1 + number of direct downloads in the last month) 
```

我们使用一个`log()`函数来平滑`number of downloads`值，因为每个后续下载的权重较小；0 和 1000 次下载之间的差异应该比 100，000 和 101，000 次下载之间的差异对流行度分数的影响更大。

这种重新评分提高了 documentation 提供的搜索结果的整体相关性，因此，用户可以专注于阅读文档，而不是搜索它。

如果你想了解更多关于 Elasticsearch 查询是如何实现的，你可以看看 GitHub 上的 [RDocumentation 项目。查询本身位于](https://web.archive.org/web/20220704235014/https://github.com/datacamp/RDocumentation-app) [`SearchController`](https://web.archive.org/web/20220704235014/https://github.com/datacamp/RDocumentation-app/blob/master/api/controllers/SearchController.js) 。

如果你想了解更多关于 RDocumentation.org 是如何实施的，请查看我们的回复:

*   [RDocumentation-app](https://web.archive.org/web/20220704235014/https://github.com/datacamp/RDocumentation-app) :运行 rdocumentation.org 的 web 应用。
*   [RDocumentation-elasticsearch](https://web.archive.org/web/20220704235014/https://github.com/datacamp/RDocumentation-elasticsearch):服务 rdocumentation.org 的 elastic search 服务器的配置和 feeders。
*   [RDocumentation](https://web.archive.org/web/20220704235014/https://github.com/datacamp/RDocumentation) : R package，将 rdocumentation.org 集成到您的 R 工作流中
*   [RDocumentation-Lambda-worker](https://web.archive.org/web/20220704235014/https://github.com/datacamp/RDocumentation-lambda-worker):AWS Lambda 管道解析 rdocumentation.org 的包文档

**关于 r 文档**

从 CRAN、BioConductor 和 GitHub——当前 R 文档的三个最常见来源——收集 R 包的帮助文档。然而，RDocumentation 不仅仅是简单地汇总这些信息，而是通过 RDocumentation 包将所有这些文档带到您的手边。RDocumentation 包覆盖了 utils 包中的基本帮助功能，使您可以在 RStudio IDE 中轻松访问 RDocumentation。查找最新和最流行的 R 包，搜索文档并发布社区示例。

[![](img/4548496d09ae690644f0866f62bf7fc3.png)](https://web.archive.org/web/20220704235014/https://www.rdocumentation.org/)