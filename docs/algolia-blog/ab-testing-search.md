# 通过 ab 测试优化您的搜索性能

> 原文：<https://www.algolia.com/blog/product/ab-testing-search/>

迄今为止，企业一直在暗中测试搜索相关性，很少甚至没有性能数据或查询后指标来为优化搜索结果提供信息。Algolia 的[分析 API 和点击分析解决方案](https://www.algolia.com/products/search-and-discovery/analytics/)通过为您提供整个搜索生命周期的完整视图开始解决这一挑战——从查询到点击到转化。

今天在我们的企业计划中推出的 A/B 测试功能使您能够测试搜索设置配置变化的性能影响。最好的部分是:您可以完全从仪表板中创建 A/B 测试，而无需一行代码(当然，这也完全可以从 API 中完成)。

## [](#why-ab-testing)为什么要进行 A/B 测试？

搜索和发现是数字体验的核心，相关性是搜索不可忽视的一个方面。对用户体验质量影响最大的莫过于搜索引擎返回结果的相关性。

我们以电子商务为例。

*   **优化转化:**电子商务网站可以使用许多标准来提高相关性，以便用户可以轻松找到他们正在寻找的东西，并更快地转化；例如，在搜索结果中促进产品销售、受欢迎程度、评级。但它会很快变得势不可挡。从哪里开始？如何确保这些新标准不会适得其反？A/B 测试允许连续的、迭代的优化，一个标准接一个标准。
*   **验证业务决策:**定义电子商务业务有各种参数。也许你的运营成本要求你偏爱高利润的产品。但是如何确保这样做不会对销售产生负面影响呢？A/B 测试增加了决策过程的信心。

从文本标记的解析，到您的业务指标对排名公式的影响，再到查询中词的邻近性的重要性，Algolia 提供了数十种功能和设置，允许您微调相关性，以实现出色的搜索结果。

## [](#for-algolia-users-here-is-how)对于阿洛利亚的用户来说，这里是怎么

让我们举一个例子，您有一个当前在生产中使用的索引。姑且称之为 indexA 吧。

该索引的排名策略目前依赖于每个结果的查看次数。你想知道改变排名策略，依靠点赞数而不是浏览量是否会带来更好的搜索转化率。

A/B 测试允许您通过以下步骤做到这一点:

*   创建索引 a 的副本——我们称之为索引 b(事先确保有足够的空间来复制数据)
*   将 indexB 的[自定义排名](https://www.algolia.com/users/sign_in)改为使用点赞数而非点击数
*   如果你还没有做到，实现[点击分析](https://www.algolia.com/doc/guides/getting-analytics/search-analytics/advanced-analytics/)。它将用于测量由索引 B 中的配置更改所导致的性能改进
*   转到仪表板上的 [A/B 测试功能](https://www.algolia.com/doc/guides/ab-testing/what-is-ab-testing/)，在 indexA 和 indexB 之间启用 A/B 测试
*   等到 AB 测试的显著性分数达到 95%才获得足够的流量，就这样了！

根据结果，你会知道你的新排名策略是否会给你的搜索带来更多的点击和转换，所以你可以放心地继续改变！

我们希望这个特性能帮助你更快地迭代你的配置，并实现你的搜索的最佳相关性。我们当然计划指导您一路走来，并帮助您充分利用它！

我们邀请您参加我们即将举办的网络研讨会:“[通过 A/B 测试优化搜索性能](https://go.algolia.com/AB_testing)”和[阅读文档](https://www.algolia.com/doc/guides/ab-testing/what-is-ab-testing/)或[访问产品](https://www.algolia.com/products/search-and-discovery/ab-testing/)页面了解更多信息

不要犹豫，联系并分享你的问题和反馈:[给我们发电子邮件](mailto:support@algolia.com)，[发推特给我们](https://twitter.com/algolia)，评论这个帖子。