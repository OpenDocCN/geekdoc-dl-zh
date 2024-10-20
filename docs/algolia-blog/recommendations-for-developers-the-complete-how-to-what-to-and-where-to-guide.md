# 开发者推荐:推荐完整指南

> 原文：<https://www.algolia.com/blog/engineering/recommendations-for-developers-the-complete-how-to-what-to-and-where-to-guide/>

推荐的是一款网上买家和媒体消费者必备。他们已经成为在网上寻找最好的数字内容不可或缺的一部分。当我们输入查询、浏览结果、查看商品和购物时，它们就会出现。

事实上，我们正处于这样一个阶段，许多用户如果不点击推荐，甚至无法想象在网上购物、看电影、阅读文章或做任何事情。要么是可供选择的内容太多，要么是推荐的内容太相关，让人无法抗拒。

为了量化这一点， **几乎所有的在线用户活动，如浏览、点击和转换，都涉及搜索或推荐** 。例如，零售 网站上的[](https://resources.algolia.com/top-resources/report-stateofsearch21)购物者有 43%会直接进入搜索栏。此外， *推荐占了一些领先网站*转化率的很大一部分

*   [人们在 YouTube 上观看内容的 70%](https://blog.hootsuite.com/how-the-youtube-algorithm-works/#:~:text=The%20goal%20was%20to%20find,watching%20videos%20the%20algorithm%20recommends.)
*   [](https://evdelo.com/amazons-recommendation-algorithm-drives-35-of-its-sales/)消费者在亚马逊上购买商品的 35%
*   [](https://www.wired.co.uk/article/how-do-netflixs-algorithms-work-machine-learning-helps-to-predict-what-viewers-will-like)80%的人看好网飞

**这对开发者意味着什么？没有机器学习背景的开发人员如何构建数据、算法和用户界面来提供如此有意义和相关的建议(和业务结果)？**

我们来看看吧！

## [](#how-to-display-recommendations-on-your-ui)如何在你的 UI 上显示推荐

我们将查看显示和生成建议的示例代码:

*   The API——如何检索建议
*   用户界面–如何显示推荐
*   数据——如何通过用户事件获取使用分析

### [](#when-and-where-to-display-recommendations)**何时何地显示建议**

您可以在任何页面上显示推荐。比如在 **首页** ，在这里你可以推荐你最热门、最潮流的单品。正如您将在下面看到的，推荐使用各种机器学习模型来检测趋势项目。

但是当你的用户浏览你的网站时，他们会希望在任何时候看到与他们正在做的事情相关的推荐。这里有一个页面列表，在这里推荐是一个很大的资产。它们都与用户正在查看的一个或多个 **特定项目**:

*   产品详情页面
*   搜索结果页面
*   文章页数
*   视频页面
*   播客页面
*   检查页面

最后， **方面和类别:** 你可以对属于某个方面或类别的项目提出建议。这些建议可以出现在上面的页面列表中，但是它们与下面的页面最相关:

### [](#how-to-retrieve-recommendations-%e2%80%93-the-code)**如何检索推荐-代码**

[在任何这些页面上实现建议](https://www.algolia.com/doc/guides/algolia-recommend/how-to/set-up/#show-recommendations-in-your-user-interface) 只需要**一个 API 调用**。下面是一些例子(用 JavaScript 参见我们的 [推荐 API 参考](https://www.algolia.com/doc/api-client/methods/recommend/) 所有语言):

对于**主页**，可以显示[趋势项](https://www.algolia.com/doc/api-reference/api-methods/get-trending-global-items/) :

```
recommendedItems = client.getTrendingGlobalItems([
  {
    indexName: 'your_index_name',
    threshold: 80
  },
]);

```

对于**商品详情页**，可以显示 [相关商品](https://www.algolia.com/doc/api-reference/api-methods/get-related-products/) :

```
recommendedItems = client.getRelatedProducts([
  {
    indexName: 'your_index_name',
    objectID: 'your_item_id',
  },
]);

```

在**结账页面**，可以显示经常一起购买的[](https://www.algolia.com/doc/api-reference/api-methods/get-frequently-bought-together/):

```
recommendedItems = client.getFrequentlyBoughtTogether([
  {
    indexName: 'your_index_name',
    objectID: 'your_item_id',
  },
]);

```

对于**类别页面**，可以显示特定类别 : 内的[趋势项](https://www.algolia.com/doc/api-reference/api-methods/get-trending-items-for-facet/)

```
recommendedItems = client.getTrendingItemsForFacet([
  {
    indexName: 'your_index_name',
    threshold: 80,
    facetName: 'category',
    facetValue: 'sweaters'
  },
]);

```

### [](#how-to-display-recommendations-%e2%80%93-the-ui)**如何显示推荐–UI**

最后，对于 UI，你可以使用我们的[**InstantSearch 前端库**](https://www.algolia.com/developers/?refinementList%5Btype%5D%5B0%5D=Front-end%20libraries&page=1&configure%5BhitsPerPage%5D=15&configure%5BclickAnalytics%5D=true#integrations) 到用下面的 6 行代码显示上面的任何 API 调用(见下面的视频演示):

```
recommendedItems = relatedProducts({
    container: '#relatedProducts',
    recommendClient,
    indexName: 'YourIndexName',
    objectIDs: ['YOUR_ITEM_ID'],
    itemComponent,
});

```

[https://www.youtube.com/embed/XEfFCJ5_uBI?feature=oembed](https://www.youtube.com/embed/XEfFCJ5_uBI?feature=oembed)

视频

详见我们的 [*潮流模特专题聚焦*](https://www.algolia.com/blog/engineering/feature-spotlight-trends-models-in-recommend/) 文章。或者看完整个 [推荐教程](https://www.algolia.com/doc/guides/algolia-recommend/how-to/set-up/#show-recommendations-in-your-user-interface) 这里。

### [](#capturing-usage-analytics-with-user-events)**捕捉用户事件的使用分析**

正如你将看到的，几乎所有的推荐都来源于捕捉用户点击、浏览和转化时的行为。虽然不要求开始使用推荐，**以** **[分析事件](https://www.algolia.com/doc/guides/algolia-recommend/how-to/set-up/#collect-user-events-for-algolia-recommend)** **的形式捕捉相关的使用活动，并将这些数据输入推荐引擎的机器学习模型，将提升你的推荐**。

同样，无论您是从 [前端](https://www.algolia.com/doc/guides/sending-events/implementing/how-to/sending-events-frontend/) 还是从 [后端](https://www.algolia.com/doc/guides/sending-events/implementing/how-to/sending-events-backend/) 发送事件，都只需要一个 API 调用来捕获事件。

下面是从前端发送一个**点击事件**的代码:

```
insights_library('clickedObjectIDsAfterSearch', {
    userToken: 'user-123456',
    eventName: 'Product Clicked',
    index: 'products',
    queryID: 'cba8245617aeace44',
    objectIDs: ['9780545139700'],
    positions: [7],
});

```

下面是在搜索之后发送一个**转换事件**的代码:

```
insights_library('convertedObjectIDsAfterSearch', {
    userToken: 'user-123456',
    index: 'products',
    eventName: 'Product Wishlisted',
    queryID: 'cba8245617aeace44',
    objectIDs: ['9780545139700', '9780439785969']
});

```

入门 [发送分析事件](https://www.algolia.com/doc/guides/sending-events/getting-started/) 此处。

## [](#under-the-hood-%e2%80%93-the-recommendations-data-models-in-a-nutshell)引擎盖下——推荐数据&车型概括地说

### [](#the-data)**数据**

我们在网上搜索的大部分内容都涉及到 *结构化内容。* 也就是说，我们搜索的网站，其数据包含明确定义的项目，比如产品、文章、电影，其中每个项目(产品、电影或文章)都是一条带有属性的记录。搜索引擎将这些数据称为索引，通常表示为带有属性的[无模式记录集合。想想 JSON。](https://www.algolia.com/blog/engineering/facets-data-model-of-json-records/)

搜索引擎的工作是返回其属性与查询中的字符相匹配的每条记录。推荐引擎的工作是什么？首先，**推荐引擎使用与搜索引擎相同的记录(及其属性)**。

还有第二个数据集包含实际的推荐。因为推荐器引擎使用多个学习模型，所以必须以某种方式准备第二个数据集，以允许引擎应用其各种学习模型。如下所述，**推荐使用机器学习模型来学习项目并在项目**之间建立*关系*。

有了这些数据，推荐引擎就可以根据建立项目之间的关系

1.  数据("基于内容的过滤")
2.  用户行为&事件【协同过滤】

### [](#1-relationships-based-on-the-data)**1。基于数据的关系**

引擎使用 ML 模型 *基于内容的过滤* 基于相似性生成项目组，其中相似性来自于比较所有记录的属性内容。属性 *显著* 匹配的记录被认为是相似的，因此被分组在一起。推荐引擎使用这些组来基于相关记录在同一组中的包含性来推荐相关记录。

例如，推荐器引擎可以读取一个索引，并且看到一些项目包含具有类似“运动鞋”“城市居民”“跑步者”的数据的属性；它可以通过查看标题、描述和其他属性(标题优先)中的内容来做到这一点。如果看到足够多的相同数据，它可以将所有符合这三个标准的记录组合在一起。我们姑且称这个群体为“城市跑者——球鞋”。稍后，当推荐器引擎看到用户正在选择“city-runner-运动鞋”组中的项目时，它可以推荐该组中的其他项目。

使用基于内容的过滤的一个好处是，您可以在第一天就开始显示推荐，而无需等待下一个模型所需的用户分析。然而，基于内容的过滤模型也可以与分析相结合(我们称之为“混合引擎”模型)，以产生更强的关系，正如我们现在将要看到的。

### [](#2-relationships-based-on-user-behavior-events)**2。基于用户行为的关系&事件**

引擎也可以使用 ML 模型 *协同过滤* 根据用户对索引中的记录所做的事情来生成分组。当用户点击或查看或购买或以任何其他方式转换时，推荐引擎可以开始将项目分组为“相关”、“经常一起购买”或“趋势”。这些关系不需要看数据；相反，它们完全依赖于用户与数据的交互方式。

基于协作的推荐本质上依赖于先前的用户活动来对具有相似点击和转换集合的项目进行分组。概括地说，这个想法是当许多用户点击或转换不同的项目时，引擎可以开始检测一些项目共有的点击和转换模式。共享相同模式的项目被认为是相关的。这被称为 *基于项目的剖析。* 这里有个例子。

我们可以创建一个类似“观看摇滚音乐纪录片的客户”的个人资料。这是通过捕捉用户的搜索词、点击和转化等活动来实现的。如果用户查询“披头士”并点击他们的“回来”和“顺其自然”纪录片，这表明了一种模式。如果许多用户对其他音乐艺术家采取类似的行动(搜索音乐艺术家并点击他们的纪录片)，推荐引擎将确认有一个音乐纪录片档案，其中可能包含许多项目。从那时起，只要用户表明这种偏好(通过搜索或观看音乐纪录片)，引擎就可以开始推荐最常观看的音乐纪录片。

一般逻辑就是这样。现在让我们来看看 Algolia 到底能提供什么。

## [](#what-recommendation-models-algolia-recommend-offers)有哪些推荐车型 Algolia 推荐优惠

Algolia 在 2021 年发布了第一个版本的 Recommend，有两个型号。从此， [推荐](https://www.algolia.com/products/recommendations/) 就演变成了提供附加推荐的车型。总的来说，我们的模型分为三大类:

*   相关内容(媒体内容、产品)
*   经常一起买(看，读，…)
*   趋势(项目和方面)

### [](#related-content-and-related-products)**【相关内容(及相关产品)**

Algolia 采用一种混合方法，使用协作和基于内容的过滤模型来生成相关内容。

注:您 *可以开始使用* 类似相关内容和相关产品的推荐模型，仅使用基于内容的过滤，不需要协作(基于事件)数据。不过，我们将解释协作过滤(可以随着时间的推移添加)如何提升体验。

从协同过滤开始， [推荐](https://www.algolia.com/products/recommendations/) 分组项目基于 *项目档案* 。它的工作方式如下:每当一个项目被点击时，我们用点击该项目的用户的 ID 存储该项目的对象 ID。当 Recommend 获得足够多的用户点击和转换来建立一个项目的档案(10 个用户至少有 10，000 个事件)时，它将有足够多的用户事件来创建有意义的项目档案。此时，查看简档项目的用户可以接收对属于同一项目简档的其他项目的推荐。

为了加强这些项目关系，我们添加了基于内容的过滤，查找记录属性的相似性，如上所述。然后，我们结合两个模型的结果，保持每个模型的最强关系。

这里重要的一点是，由于相关内容使用基于内容的过滤，你甚至可以在积累了协同过滤所需的 10K 事件之前就开始显示推荐。

### [](#frequently-bought-together-or-watched-read-etc)**经常一起买(或者看，看等。)**

基于协作过滤(仅限转换)，Recommend 根据同时转换(购买、添加到购物车、查看)两个或更多项目的用户数量生成项目配置文件。推荐需要 10 个用户对两个或更多项目进行至少 1000 次转换事件，才能开始生成有意义的推荐。

我们通常会在三种情况下看到“频繁一起购买”:当用户正在查看一件商品，将一件商品添加到购物车中，或者刚刚购买完一件商品。

转换事件是不可知论的:它们可以用来表示项目是否被一起观看或阅读，而不仅仅是购买。所以“勤 *行* 合”是对这种模式的一个比较笼统的理解。

### [](#trends)**趋势**

有了趋势，我们不再需要用户 ID，而只关注一个项目或方面最近的受欢迎程度。最基本的计算是用统计来确定哪些物品具有 *最强的上升趋势* 的转换率。我们提供三种趋势:

*   *   我们查看 *整个目录* 中的商品，并推荐转化率上升趋势最强的商品。

*   **趋势项，按面过滤**

*   *   我们查看一个方面或类别 内的所有项目 *，并推荐转化率上升趋势最强的项目。*

*   *   我们查看各个方面，并推荐其项目转化率上升趋势最强的方面。

## [](#the-anonymity-of-recommendations%c2%a0)匿名推荐

毫无疑问，你已经看到了，我们非常依赖捕获用户 id 来创建个人资料。虽然这引起了合理的担忧，但我们已经建立了我们的机器学习模型，以确保完全匿名。

因此，每当我们提到 *用户 ID* 时，一切都是匿名的，这意味着当我们保存用户 ID 时，除了这个唯一的 ID 之外，我们不保存任何个人特征。我们只对 ID 的唯一性感兴趣，以帮助对类似的项目进行分组。

我们创建的 *简档* 是 *项* 简档，也就是将用户行为集合相似的项分组在一起。因此，我们对分析用户不感兴趣，所以我们不需要存储任何个人信息。我们只利用用户 ID 来加强模型的项目配置文件的生成。

## [](#next-steps)下一步

正如我们已经讨论过的，我们的模型可以在你网站的不同部分显示推荐。无论您希望优化客户旅程的哪个阶段，我们都可以提供帮助！ [报名免费试用推荐](https://www.algolia.com/users/sign_up) 。