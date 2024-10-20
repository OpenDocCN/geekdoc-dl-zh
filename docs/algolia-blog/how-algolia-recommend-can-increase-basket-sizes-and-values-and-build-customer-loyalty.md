# 增加您的客户群，通过推荐建立他们的忠诚度

> 原文：<https://www.algolia.com/blog/product/how-algolia-recommend-can-increase-basket-sizes-and-values-and-build-customer-loyalty/>

你已经在发送 Algolia 点击和转换事件了吗？如果是这样，那么你就可以开始向顾客展示推荐的产品了。

我们都曾在网上购物时看到并点击过推荐。只要我们向购物车中添加一件商品，就会提示我们继续购买经常一起购买的商品。

仔细看看。这些建议可能是由 Algolia 推荐提供的。

当在线商店以非侵入性和及时的方式建议相关商品时，消费者会很感激。想想珍妮弗，她正在为她的病狗 [露娜](https://www.algolia.com/blog/ux/how-can-recommending-frequently-bought-together-products-increase-roi/) 找床。她找到一张完美的床，并把它放入购物车。然后，她注意到一个“经常一起购买”的部分，推荐狗套、枕头和其他可以安慰生病的狗的配件。珍妮弗也可能看到其他狗主人为他们心爱的宠物买的物品。

![Recommendations increase basket size](img/2d925d0af1f7bbbb09b938c7e8ddb382.png)

这就是 [Algolia 如何推荐](https://www.algolia.com/products/recommendations/) 增加你的客户的篮子大小和价值，同时改善他们的体验和忠诚度。

但是 Algolia 怎么知道该推荐什么呢？Algolia 知道 Jennifer 是宠物主人还是知道 Luna 生病了？

简单的回答是否定的。Algolia 对 Jennifer 一无所知，只知道她在遵循一种基于相似顾客的相似购买行为的消费模式。许多像 Jennifer 一样的顾客已经搜索、点击和购买了类似的产品..

虽然这听起来很简单，但要混合和匹配客户想要的东西，实际上需要大量数据和精心实施 [人工智能驱动的机器学习](https://www.algolia.com/blog/ai/the-anatomy-of-high-performance-recommender-systems-part-1/) 才能做到这一点。

通过收集用户行为(基于事件的分析)和训练模型来展示特征来推荐作品比如: *F* *频繁一起买* 、 *潮流单品* 、T *软管谁买了这个还买了* 。这些模型经过精心设计，符合客户最初的购买意图，从而保证客户继续购物的可能性更大。可以想象，只有最相关的推荐才能推动财务增长和客户忠诚度。

## [](#with-algolia-recommend-you-can-significantly-improve-your-business-performance)有了阿果推荐你可以显著提高你的经营业绩

在告诉你如何设置它之前，让我们看看正确的推荐如何能够带来更多的追加销售、交叉销售和整体客户忠诚度(询问 [亚马逊](https://www.rejoiner.com/resources/amazon-recommendations-secret-selling-online) )！)

[Gymshark](https://resources.algolia.com/ecommerce/casestudy-gymshark-retail-2?utm_campaign=FY23_%5Bcust%5D&utm_medium=email&utm_source=one_off&utm_content=BlackFriday_Recommend&utm_2nd_camp=B2B_Ecomm)通过在其 Algolia 搜索实现中添加 Algolia 推荐，提高了其收入。Gymshark 的成功指标:

*   在黑色星期五，新用户的订单率增加了 150%，而“加入购物车”率增加了 32%
*   回头客的订单率和“加入购物车”率分别提高了 13%和 10%
*   每用户 1.4 次点击，而之前的解决方案为 1.1 次点击

![](img/369f779e833efbe9675de73ef2d90c47.png)

Gymshark 增加了 Algolia 推荐来应对至关重要的黑色星期五。我们发现，在黑色星期五或圣诞假期等大型购物活动期间训练您的推荐模型，将有助于您及时了解购物者的行为，以迎接下一个大型活动。

## [](#all-you-need-to-do-is-start-training-the-model)你需要做的就是开始训练模型！

下面推荐四个步骤。

1.  捕捉用户活动(事件)(*已经被大多数客户捕捉* )
2.  将这些事件数据发送给 Algolia ( *已被大多数客户发送* )
3.  启用仪表板上的 Algolia 推荐，并让它开始在后台生成推荐
4.  向您的用户显示推荐

### [](#sending-quality-data)**发送质量数据**

尽管大多数客户已经在发送 Algolia [分析](https://www.algolia.com/blog/product/5-reasons-to-add-clicks-to-site-search-analytics-and-code-to-do-it/)(步骤 1 & 2)。但是，仍然有必要回顾一下这两个以数据为中心的步骤，因为它们提供了决定您的建议质量的原始数据。

步骤 1 & 2 完全由客户控制。客户通过捕获 *重要的* 用户活动(事件)来建立他们的分析数据集。这意味着您必须小心捕捉对您的业务“重要”的行动。当然，购买历史几乎总是相关的，但是购物车呢？还是分类页面？刻面？

本质上， **你要跟踪所有你认为很有可能导致转化的用户行为。**

好消息是，Algolia 附带了一个 [分析仪表板](https://www.algolia.com/doc/guides/getting-analytics/search-analytics/understand-reports/#dashboard-metrics) ，你可以用它来测试你的分析数据的质量。

### [](#enable-recommendations-%e2%80%93-and-you%e2%80%99re-good-to-go)**启用推荐——一切就绪**

一旦你发送了重要的用户操作(这里有一个 [大列表](https://www.algolia.com/doc/guides/sending-events/planning/) )，那么你需要做的就是在 Algolia 的仪表盘中启用推荐(步骤 3)，让机器运转起来。当你发送事件时，Algolia 会完成剩下的工作——它会将这些事件输入到它的机器学习模型中，并生成一组可靠的建议。

[https://www.youtube.com/embed/0B5Sd3G4NeI?feature=oembed](https://www.youtube.com/embed/0B5Sd3G4NeI?feature=oembed)

视频

在短时间内——可能是几周，这取决于你网站的活动——你可以开始向你的用户显示这些推荐(步骤 4)。下面是一些 [撬动推荐](https://www.algolia.com/blog/product/6-ways-to-leverage-recommendations-for-ecommerce-retailers/) 的强大方法，以及关于 [如何显示推荐](https://www.algolia.com/blog/product/introducing-algolia-recommend-the-next-best-way-for-developers-to-increase-revenue/) 的概述。

# [](#the-bottom-line)底线

如果你能提供很好的建议，确保你的客户之旅是愉快的，没有压力的，你就能建立客户忠诚度，这将改变你的业务底线。

我们强大的[推荐 API](https://www.algolia.com/products/recommendations/) 和人工智能驱动技术将快速无缝地改善您的用户体验，提高您的网站和移动应用的转化率。

了解领先的电子商务零售商如何利用人工智能推荐引擎的功能来超越和 转变 他们的数字销售目标。[联系我们](https://www.algolia.com/contactus/)，让我们开始行动吧！