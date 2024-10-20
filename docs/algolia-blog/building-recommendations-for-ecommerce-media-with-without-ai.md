# 为电子商务和媒体建立推荐，有或没有 AI 

> 原文：<https://www.algolia.com/blog/ai/building-recommendations-for-ecommerce-media-with-without-ai/>

随着电子商务和数字媒体的普及，大多数在线公司用于推荐的工具并没有发生实质性的变化——直到最近。大约从最早的数字市场开始，早期的推荐工具就基于启发式过滤器，将商品分成不同的类别，并跟踪用户购物车或观察列表中的同现情况。

虽然基于过滤的推荐(如下所述)实现起来很简单，并且不一定依赖于复杂的算法，但它们有其局限性。基于机器学习的现代解决方案 [提供了重大改进](https://medium.com/thelaunchpad/no-machine-learning-in-your-product-start-here-2df776d10a5c) 。

亚马逊处于人工智能推荐系统的最前沿——[推荐占其销售额的 35%](https://www.sailthru.com/marketing-blog/recommendation-algorithms-guide/)。他们的算法被小心翼翼地保护着，训练有素——而且很昂贵，但有可能以少得多的投资获得很好的结果。Netlify 和 Spotify 也是如此。

推荐引擎可以围绕 [多种车型](https://www.sailthru.com/marketing-blog/recommendation-algorithms-guide/) 。根据你的产品和目标消费者，会根据你的需求使用不同的算法。任何在亚马逊上购物的人都会非常熟悉两个最广泛使用的模型。“相关项目”和“那些谁买了这个也买了”推荐都是 [协同过滤](https://fortune.com/2012/07/30/amazons-recommendation-secret/) 算法，实现它们有一系列选项。

在本帖中，我们将介绍几种不同的推荐工具。您可以使用基于标记和过滤的经典方法。或者，我们将展示机器学习(ML)和人工智能(AI)如何提供隐私安全和更自然的推荐。最后，您将看到 [Algolia 推荐](https://www.algolia.com/products/recommendations/) 如何解决这些问题，为开发者提供一个 API 级别的推荐引擎。但是首先，理解用户的背景以及他们会如何看待你的推荐是很重要的。

## [](#make-high-quality-recommendations-part-of-the-search-journey)让优质推荐成为搜索旅程的一部分

潜在客户通常通过两种方式进入网站。有了具体的指导方针，他们可能知道自己到底在寻找什么——一顶高质量的儿童自行车头盔、一件天然纤维的白色上衣、一个棕色皮搁脚凳。或者他们可能正在浏览，为一个重要的生日寻找礼物，或者为家庭办公室寻找更好的解决方案。

无论哪种情况，通过识别用户行为模式，推荐都有可能将潜在买家与他们自己可能没有发现的产品联系起来。下面，我们来看看两种可能的方法——一种简单的基于过滤器的方法和一种基于 ML 的解决方案。

### [](#get-custom-results-with-simple-tags-with-no-ai)用不带 AI 的简单标签获取自定义结果

在几十件白色衬衫中，用户点击了一件束腰亚麻衬衫。在不使用任何 AI 的情况下，利用标签和销售历史就可以进行推荐。

也许那件衬衫被贴上了“休闲”的标签，是你春季编辑活动的一部分。你可以很容易地返回一个简短的清单，列出三四件符合至少两个标准的其他衬衫——休闲风格、春季、亚麻、白色、束腰长度、价位。

准备您的数据到 [提供这种类型的建议](https://www.algolia.com/doc/ui-libraries/recommend/introduction/what-is-recommend/#related-products-and-related-content) 需要一些前期工作和持续验证，以确保您库存中的所有产品都被正确索引。对于每个项目，您必须决定要“匹配”什么属性，以及每个匹配属性的相对重要性。当一个新产品添加到您的目录中时，它需要被“标记”并分配匹配的属性，这可能不容易自动上传。衬衫的颜色可能会自动填充，但“款式”或“季节”可能不会。使用这种方法有可能获得精确、高质量的建议，尤其是如果您的库存变化不太频繁的话。

另一个非人工智能的方法是使用销售历史。一个合理的假设是，有一个共同购买经历的客户可能也有其他共同之处。销售历史可以为购买了所选商品的人购买的其他商品提供推荐。查找这些产品-购买者-产品关系的查询可能会比上面的产品-产品关系使用更多的资源，因此对于大量库存来说可能是低效的。对于拥有许多产品类别的公司，这也可能导致推荐中出现一些不必要的“噪音”(例如，最畅销的商品出现是因为它普遍受欢迎，而不是因为它与其他产品相关)。此外，当一个新产品被添加到您的目录中时，将没有购买历史可供参考。

销售历史方法的另一个缺点是，它将消费者数据直接引入匹配算法。当然，以一种完全安全和私密的方式做到这一点是可能的，但是任何时候在面向公众的代码中使用客户数据，都有暴露私密数据的更大风险。避免数据泄露是一项需要时间和知识的承诺。

根据你的情况，这些负面影响可能会很大。虽然好的推荐不需要人工智能，但你可以避免处理用户数据和强行微调推荐。让我们来看看采用基于 ML 的方法是什么样子的。

### [](#use-ml-based-ai-solutions-for-better-recommendations)使用基于 ML 的 AI 解决方案获得更好的推荐

基于过滤的非人工智能推荐的问题在于，它们假设用户知道他们想要什么，并能准确描述。有时候就是这样。但通常，最有效的推荐是针对用户自己可能没有找到或发现的项目。

基于最大似然的推荐的主要优点是有助于发现问题。儿童自行车头盔的购买者可能还有其他共同的需求——儿童安全设备，潜在的户外活动，健康饮食，或替代通勤，或其他更难预测的兴趣。

人工智能算法没有试图预测所有这些联系，而是注意到购物者行为中可能难以预测或描述的模式。基于 ML 的解决方案提供的推荐具有意外收获和“适合”的元素，这有助于它们脱颖而出。要开始观察这些模式，首先要收集用户事件。然而，正如前面提到的个性化一样，重要的是不要暴露私有数据。

## [](#capture-user-events-for-high-quality-results-with-low-privacy-risk)捕捉用户事件，以低隐私风险获得高质量的结果

基于 ML 的推荐引擎可用的最丰富的数据是用户事件，如点击和转换。历史上，推荐只反映用户的 *当前* 动作——最近点击的项目，当前搜索词。搜索一件白色上衣，你会得到白色上衣。与单个事件相比，跟踪用户动作序列可以更深入地了解特定用户的目标和兴趣。从白色衬衫到浅色连衣裙再到凉鞋的点击流，比最初的搜索词更能洞察购物者的兴趣。

这种用户数据跟踪会给你的应用带来不必要的隐私风险。人工智能解决方案的一个幸运之处是，它们在用户隐私和数据安全方面取得了胜利，并提供了更好的建议。他们根据许多人过去的行为进行学习，然后根据模型进行预测。

基于 ML 的解决方案专门针对特定会话期间客户与您的产品和服务的交互方式进行调整。特定的搜索词、点击和用户当前访问你的网站的转换可以帮助你根据他们当时正在寻找的东西来定制推荐。

[收集事件数据](https://www.algolia.com/doc/guides/sending-events/getting-started/) 支持的不仅仅是推荐。跟踪用户查看、点击和转换事件可以告诉你很多关于用户如何与你的网站和产品交互的信息。您需要对事件数据进行分类和存储，用线性回归和其他基于 ML 的算法训练您的模型。

如果您选择使用 Algolia，其[Insights API](https://www.algolia.com/doc/rest-api/insights/)将成为您收集用户事件数据并将其发送到您的应用程序的中枢。您可以使用这些数据进行 A/B 测试和动态产品重新排名，以响应用户的选择，并提供更有针对性的建议。

下面的代码显示了数据如何发送到 Insights 的简单示例。您前端的一个`onClick`事件处理程序发出一个简单的 POST 请求(这里显示的是您可能模拟 cURL 的事件):

```
curl -X POST \
https://insights.algolia.io/1/events \
-H 'x-algolia-api-key: ${ADMIN_API_KEY}' \
-H 'x-algolia-application-id: ${APPLICATION_ID}' \
-H "Content-Type: application/json" \
-d '{
    "events": [
        {
          "eventType": "click",
          "eventName": "Product Clicked",
          "index": "products",
          "userToken": "user-123456",
          "timestamp": 1529055974,
          "objectIDs": ["9780545139700", "9780439784542"],
          "queryID": "43b15df305339e827f0ac0bdc5ebcaa7",
          "positions": [7, 6]
        },
      ]
    }'

```

A `queryID`将点击事件绑定到特定的 Algolia 搜索，并且仅在搜索的一个小时内有效。基于会话的令牌将点击事件连接到用户。暴露敏感用户数据的风险很小，因为根本不会收集这些数据。与此同时，用户可以获得与他们当时的行为相应的推荐。

## [](#use-collaborative-filtering-to-generate-precise-recommendations)使用协同过滤生成精准推荐

最常见的“频繁一起买”建议先从 [协同过滤](https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0) 算法说起。任何推荐人工智能的基本目标都是找到用户和产品之间的相似之处，并基于这些相似之处提出建议。协同过滤算法被广泛用于数据科学中，根据具有相似偏好的其他用户对产品的评价，预测用户对产品的评价。 [协同过滤](https://towardsdatascience.com/collaborative-filtering-and-embeddings-part-1-63b00b9739ce) 很多解释都是以电影为例。一个简化的场景:

*   我爱古装剧，给过 *皇冠**叫接生婆这样的电视剧很高的收视率。*
*   我真的很反感奇幻太多的电视剧，给了 *夏洛克* 和 *神秘博士很低的收视率。* 姐姐爱死他们了！
*   帕特和山姆也都喜爱古装剧，并且对 *王冠* 评价也很高。
*   帕特也喜欢奇幻类的节目，并给我不喜欢的节目打高分。
*   山姆和我更相似，所以他对奇幻电视节目评价较低。
*   一部被归类为神秘/犯罪的新剧上映了，但它同时包含了历史和幻想的元素。帕特对它评价很高，萨姆评价较低。
*   在此之前，我们没有人给推理/犯罪节目评分。
*   因为我的喜好和山姆的大体匹配，所以不推荐我看这个节目。我妹妹的偏好与帕特的相匹配，她在她的推荐队列中获得了新节目！

在规模上，协同过滤算法比这更复杂，尽管基本原理非常相似。与我们上面讨论的非人工智能解决方案不同，可伸缩性是基于 ML 的推荐系统的一个卖点。数据集越大，使用人工智能算法的情况就越有说服力，因为有更多的机会发现用户行为的模式。亚马逊在很大程度上依赖于协同过滤，这导致产品推荐可以感觉到意外收获和独特的个性化。

协同过滤算法分为 [几类](https://realpython.com/build-recommendation-engine-collaborative-filtering/#user-based-vs-item-based-collaborative-filtering)——基于用户、基于项目、混合和矩阵分解。即使在混合系统中，也有一系列的选择来决定有多少种方法可以组合以及以什么样的优先顺序组合。根据用户数量、项目类型、每个用户的评级密度和项目类别，每种方法最适合不同类型的数据集。

## [](#use-algolia-recommend-to-get-an-ml-based-solution-that%e2%80%99s-always-improving)使用 Algolia 推荐获得一个不断改进的基于 ML 的解决方案

虽然没有适用于所有目的的单一推荐解决方案，但在大多数情况下，一个非常好的选项并不一定很复杂。对于电子商务公司来说，协同过滤方法可能仍然是最好的选择。

Algolia 推荐在一个 API 中交付 [高性能 AI 驱动的](https://www.algolia.com/blog/ai/the-anatomy-of-high-performance-recommender-systems-part-iv/) 推荐。API 外形意味着它实现起来很快，尤其是因为简单的前端 UI 代码片段。这也是确保您的推荐随着时间推移会变得更好的一种方式。那些通过 API 交付的迭代改进会自动实现——无需您重新编码。

如果没有安果指数， [创建账号和 app](https://www.algolia.com/users/sign_up?utm_source=blog&utm_medium=main_blog&utm_campaign=recommendations_articles&utm_id=external_everydeveloper_article) 。实施 Algolia 推荐，独立或与 Algolia 搜索，将是一个简单的步骤，让您的电子商务网站获得最佳结果。你可以 [看一下](https://resources.algolia.com/ui-libraries/webinar-recommendlaunch-dg-retail)[动作](https://resources.algolia.com/discover-algolia/video-optimizewithrecommend-retail) 两个视频，或者从我们的 [深入了解一下](https://www.algolia.com/doc/guides/algolia-recommend/overview/) 。如果您不熟悉 Algolia，这些资源可以帮助您加深对推荐引擎的理解，并让您感受到使用 Algolia 是多么容易。