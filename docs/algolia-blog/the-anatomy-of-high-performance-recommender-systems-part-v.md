# 推荐和用户操作

> 原文：<https://www.algolia.com/blog/ai/the-anatomy-of-high-performance-recommender-systems-part-v/>

## 建议和用户操作

在本系列的前几篇文章中，您学习了[](https://www.algolia.com/blog/ai/the-anatomy-of-high-performance-recommender-systems-part-1/)，如何使用 [正确的数据源](https://www.algolia.com/blog/ai/the-anatomy-of-high-performance-recommender-systems-part-2/) ，如何 [设计它将从中学习的正确特性](https://www.algolia.com/blog/ai/the-anatomy-of-high-performance-recommender-systems-part-3/) ，以及有哪些 [工业级推荐模型和技术](https://www.algolia.com/blog/ai/the-anatomy-of-high-performance-recommender-systems-part-iv/)

现在你已经有了一个可靠的推荐模型，是时候回到你的用户身边了。为什么用户甚至会在意推荐？在这里，从一些角度来看用户在网上导航时必须做出的越来越复杂的决定是有好处的。

引用网飞的研究:

> *“人类在生活的各个方面都面临着越来越多的选择——当然是在视频、音乐和书籍等媒体方面，但更重要的是，在健康保险计划、治疗和测试、求职、教育和学习、约会和寻找生活伴侣以及其他许多选择非常重要的领域。*
> 
> “我们相信，推荐系统领域将继续发挥关键作用，利用现有的丰富数据使这些选择变得可管理，有效地引导人们选择真正最好的几个选项进行评估，从而做出更好的决策。”

卡洛斯·a·戈麦斯-乌里韦&尼尔·亨特。2015.[网飞推荐系统:算法、商业价值和创新](https://doi.org/10.1145/2843948)

我们已经到达了这样一个点，选择的数量让[瘫痪](https://www.ted.com/talks/malcolm_gladwell_choice_happiness_and_spaghetti_sauce)。以 YouTube 为例:现在有 82 年的视频内容每天上传到他们的目录。作为一个用户，我甚至从哪里开始为我*将会重视的 gem 探索这一堆内容呢？
*

在我们系列的一开始，我们就着手“ *筛选出用户的选择，并根据他们的要求或喜好为他们提供最合适的建议* ”。一旦你手头有了一个高性能的推荐系统，是时候充分利用它可以生成的**预测，在一个战略性的时间和地点向你的用户展示它们，以建议要采取的相关 **行动** 。**

 **## [](#what-you-can-get-from-a-high-performance-recommender-system)你能从高性能推荐系统中得到什么

正如我们在上面网飞的引文中所看到的，对人类来说，能够独自在他们面对的网上选择的海洋中航行已经太晚了。因此，推荐系统的兴起是由商业需要驱动的:如果人们很难做出选择来访问你的内容，并且如果你有足够的数据来帮助他们做出正确的选择，那么你会因为 *而不是* 帮助他们而失去业务。

什么样的推荐可以帮助你的用户充分利用你的服务？在本系列的 [第四部分](https://www.algolia.com/blog/ai/the-anatomy-of-high-performance-recommender-systems-part-iv/) 中，我们看到不同的推荐器模型可以提供不同种类的推荐:

从用户的角度来看，在每种情况下，他们都获得了 **浏览快捷方式** :取代用户随机浏览一段时间，希望最终偶然发现正确的项目，高性能的推荐者可以直接为他们指出旅程的目的地。

不仅如此，用户还获得了 **新的机会** :登陆这个页面的人可能不知道他们可能对一些相关项目感兴趣。这些预测可以让你扩大用户对你能提供的内容的接触，帮助他们充分利用大量的选择。

这使得企业能够为用户提供数量惊人的新内容，同时帮助他们理解这些内容:如果我们回到 YouTube 的例子，82 年来每天都有新内容出现，平均会话长度仍然很长—**13m**[平均](https://www.statista.com/statistics/910910/us-most-popular-us-video-streaming-services-session-duration/) 。因此，YouTube 用户的参与度很高，**月活跃用户 20 亿，其中**1.22 亿** 为 [日活跃用户](https://backlinko.com/youtube-users#monthly-active-users) 。**

 **同样， [抖音在这一点上非常出色](https://www.wsj.com/video/series/inside-tiktoks-highly-secretive-algorithm/investigation-how-tiktok-algorithm-figures-out-your-deepest-desires/) :你开始观看任何种类的内容，你的互动告知系统你的兴趣。如果你表示对烹饪视频感兴趣，随着系统学习迎合你当前的口味，你可能会突然在你的提要中看到很多这样的视频。

## [](#encouraging-actions-showing-how-users-can-profit-from-strategic-recommendations)鼓励行动:展示用户如何从战略建议中获利

在上一节中，我们已经看到了为什么有很多产品的企业对公开工业级推荐以帮助他们的用户有着至关重要的兴趣。然而，这不仅仅是把推荐信硬塞给他们。让我们举一个极端的例子，一个脱销的产品页面显示你的前 500 个相关商品作为备选:即使有一个高性能的系统，用户也不太可能滚动浏览它们。

我们如何从 *获得好的推荐* 到 *展示战略推荐？T52*

把推荐当成一种手段而不是目的:推荐是你让用户参与进来的方式，但是 *行动* 是你引导他们的目标。因此，一份好的推荐信应该是:

*   有限公司
*   上下文
*   省力

### [](#limited)**限定**

只显示一个****相关的** 项。**

 **人们很容易被诱惑去展示更多的选择，认为这只能提供更多的机会。然而，这种直觉很容易被[](https://en.wikipedia.org/wiki/Analysis_paralysis)推翻，而且有一个 [的研究机构](https://faculty.washington.edu/jdb/345/345%20Articles/Iyengar%20%26%20Lepper%20(2000).pdf) 证明，少量的选项(3-6)可以比大量的选项(10-24)转换高达 10 倍。我们在这里是为了帮助用户过滤不相关的选项，而不是为了带回太多信息的综合症，我们已经开始着手解决了！

### [](#contextual)**语境**

一个推荐物品此时此地应该有用。

推荐的价值来源于它在当前时间和地点为当前用户提供了多少服务；这种直接的相关性使得上下文意识成为提供优秀推荐的一个重要方面。

这里有一些例子

*   在 *产品登陆页面上，用户在购物车* 中添加了一件商品后，用户想知道 *中是否有他们需要* 购买的所有商品。这正是时候推荐 **补充，** ***经常一起买的*** **产品一键添加到购物车** 按钮 **。**
*   另一方面，在一个产品登陆页面上，用户正在 *寻找其他选项* 来解决同样的需求；推荐 **替代，** ***相关产品*** **带链接看其详情** 肯定会带来更多价值。
*   *一个视频播放完* 后，强烈推荐相似的视频不太可能令人满意；最好推荐那些经常一起看*的内容，最大限度地提高用户参与度。*
**   *在某个类别的登陆页面上* ，用户正在寻找灵感或者一个新的想法；因此，显示具有该类别最佳推荐的轮播，例如，通过向该类别 中的最畅销者显示 ***相关产品*** **，可能在将首次访问者转化为顾客方面表现良好。***

 *### [](#effort-saving)**省力**

只有在为用户节省时间并帮助他们实现目标时，推荐才有价值。

*   一方面，这意味着你需要 **保守** :推荐变化多端的内容，虽然增加了意外收获，却带来了用户受挫的风险。设身处地为用户着想，以初学者的视角浏览服务，并想知道:“这些推荐能帮助我更好地实现目标吗？”
*   另一方面，这意味着你可以 **创新** :推荐在很多方面都很有用，不仅仅是作为你网站上的旋转木马。将这样的推荐反馈到营销平台和你的内容管理系统等其他工具中，会释放出很多价值。例如:
    *   **邮件活动** **，如 SendGrid** :您可以根据相关建议定制您的邮件。这可以把普通的每周电子邮件变成动态的、高度相关的时事通讯！
    *   **营销自动化** **n** **，例如，Shopify** :您可以让您商店中的应用程序利用来自 Algolia 的预测，将这些用户/商品洞察与您现有的商品和促销内容相结合，从而提高您当前 Shopify 应用程序的实用性。
    *   **CRM 集成，例如 HubSpot** :您可以通过对潜在感兴趣的服务进行分段推荐来扩充客户的资料，以通知您的销售团队在下次致电时提出正确的新功能。
    *   **像 Segment 或 BigQuery** 这样的数据平台:从这些平台你可以使用生成的推荐来 **扩充你自己的数据。**

如果您遵循这些指导原则，您可以相信，您从系统中获得的可靠建议会有效地为您的用户服务。

推荐系统可能只需要数据和算法，但推荐系统本身并不能让用户满意。决定如何利用其 **推荐** ，通过在正确的时间向正确的人建议正确的 **行动** 来最大限度地提高用户的满意度，这是让这种高性能推荐系统为您的用户群服务的关键，并最终使[为您的业务充分利用推荐](https://www.algolia.com/blog/product/introducing-algolia-recommend-the-next-best-way-for-developers-to-increase-revenue/)。

## [](#how-do-we-measure-the-success-of-your-recommendations)我们如何衡量你的推荐是否成功？

在这篇博文中，我们了解了什么是好的推荐，用户可以根据它们采取哪些行动，以及它们如何最好地服务于你的用户，以指导他们在你的服务中的行动。要了解 Algolia 推荐，[从这里开始](https://www.algolia.com/products/recommendations/)或查看我们的[推荐 API 文档](https://www.algolia.com/doc/guides/algolia-recommend/overview/)。

但是我们怎么知道这是正确的行动呢？您如何根据您的业务指标来衡量您的建议的有效性？一个人如何避免为了的短期表现而优化 *的长期商业目标* 【确保这个用户不会在一次又一次深夜狂看会议后取消他们的会员资格？)

**请继续关注本系列的下一部分——结果和评估，深入了解在您的产品中利用推荐系统的这些关键方面！**

*如果你想继续对话，在推特上联系保罗-路易:*[*【https://twitter.com/PaulLouisNech】*](https://twitter.com/PaulLouisNech)*******