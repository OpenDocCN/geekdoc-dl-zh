# 电影和视频推荐是如何工作的？

> 原文：<https://www.algolia.com/blog/ux/how-movie-and-video-recommendations-work-and-why-streaming-services-must-get-it-right/>

你如何在线观看电影和视频？在你电脑上的 YouTube 上？手机应用程序中的网飞？或者更好的说法是，你使用多少流媒体服务来观看你推荐的电影和视频？

像网飞这样的视频流媒体公司——也被称为[](https://www.techopedia.com/definition/29145/over-the-top-application-ott)媒体服务——已经开始主导娱乐行业，不仅改变了消费者的观看习惯，也改变了行业的运营方式。

就像音乐领域的 Spotify 一样，OTT 媒体服务使用了一种现在很常见的方法；一种“胜利公式”它看起来是这样的:免费试用后相对便宜的月订阅费，然后是推荐内容的扩展库和高度个性化的用户体验。

让我们来看看流媒体服务如何为用户生成个性化的视频推荐，为什么获得正确的内容推荐很重要，以及您的企业如何利用这些知识来改善您的客户体验。

## [](#the-mythical-algorithm)神话般的算法

在谈论高质量的 [个性化推荐](https://www.algolia.com/blog/product/what-are-media-content-recommendations-and-why-are-they-important/) 时，人们往往关注一个方面:推荐算法是如何工作的。对许多人来说，算法——一套循序渐进的程序或遵循的规则——呈现出一种近乎神话的形式。对另一些人来说，这是一群硅谷天才无比复杂的大脑产物。不管怎样，这都被视为一家公司成功的秘密。

但是这些算法并没有你想象的那么神秘。算法功能并不局限于拥有最聪明的计算机科学家的精英组织。你不需要和谷歌竞争来雇佣最好的工程师。任何企业都可以通过电影推荐 API 使用这项技术，您可以轻松地将它与您的产品和服务集成在一起。此外，这项技术可以帮助企业[](https://resources.algolia.com/media/webinar-recommendlaunch-dg-retail)通过吸引和留住高度参与、满意的客户来增加收入。

## [](#why-the-customer-experience-matters)为什么客户体验很重要

毫不奇怪，亚马逊是这方面做得最好的公司之一。可以说，杰夫·贝索斯对客户的痴迷和他对亚马逊成为“地球上最以客户为中心的公司”的 [愿景](https://www.marketingweek.com/mark-ritson-jeff-bezos-success-focusing-on-customer/) 导致了对用户体验的重视。如今，推荐是必不可少的，因为客户期望并要求与他们使用的每件产品或服务相关的高度个性化的体验。

它带来的不仅仅是卓越的客户体验。麦肯锡 [的研究](https://www.mckinsey.com/business-functions/growth-marketing-and-sales/our-insights/the-value-of-getting-personalization-right-or-wrong-is-multiplying) 发现，做好个性化客户体验的组织可以从中获得 40%的高收入。

这对于视频和电影流媒体服务尤为重要。有了如此多的选择——想想网飞庞大的电影和电视节目库或 YouTube 推荐——推荐已经成为提供成功客户体验的重要方式。否则，用户会继续浏览，直到被 [的选择淹没](https://thedecisionlab.com/reference-guide/psychology/choice-overload) ，他们决定做些别的事情:做饭、开车兜风、玩游戏。或者查看竞争对手的视频内容。

## [](#the-business-case-for-video-recommendations)商业案例视频推荐

为了说明这一切有多重要，有必要快速概述一下推荐视频的商业案例。电影和视频推荐至关重要，因为它们:

*   **提高用户参与度:** 客户希望他们的服务能够了解他们的偏好并提供无缝的个性化体验。他们希望能够看着他们的手机或打开他们的电视，并立即收到一个符合他们心意的推荐。
*   **提供竞争优势:** 建议是一种竞争优势。如果你的推荐引擎和竞争对手的不一样，客户会很快改变忠诚。
*   **增加订阅数量:** 在竞争激烈的行业，吸引和留住订阅用户是关键。当你能优化你的推荐时，你就有了更满意的客户，他们会长期坚持你的服务。
*   **发展业务:** 没有以上这些要素，你的业务根本就跟不上。没有比这更有说服力的商业案例了。没有好的推荐，几乎不可能发展你的业务。

## [](#what%e2%80%99s-under-the-hood)引擎盖下有什么

那么一个 [推荐引擎](https://www.algolia.com/blog/ai/building-recommendations-for-ecommerce-media-with-without-ai/) 是如何工作的呢？我们可以获得关于数据科学的技术，但这种讨论不会增加太多价值。在这一点上值得知道的只是 Algolia 这样的公司提供电影推荐 API。

但是既然你问了，这里有一个基本的高层次概述。

有 [两种方式](https://towardsdatascience.com/how-to-build-a-movie-recommendation-system-67e321339109) 生成视频推荐:

### [](#content-based)内容基础

这种方法会考虑客户的偏好(喜欢、不喜欢、活动)，并根据他们的电影推荐相似的内容。如果客户一遍又一遍地观看 *【侏罗纪公园】* ，电影推荐系统会推荐相关视频，比如另一部恐龙电影或特许经营。为了有效地完成这项工作，你需要有效地标记和描述你的内容，例如，包括类型、子类型、导演、演员和节目长度。通过这种方式，推荐引擎可以提供您的客户渴望观看的播放列表内容。

### [](#collaborative-filtering)协同过滤

这种技术利用了过去围绕用户喜欢的电影类型的交互，但也考虑了其他用户的偏好。通过在多个用户的偏好之间建立联系，它可以预测其他观众可能喜欢什么。当你大规模这样做的时候，你可以得到一些非常准确和有趣的推荐视频和电影。

## [](#the-benefits-of-a-prebuilt-recommendation-system)预建推荐系统的好处

听起来很棒，但是从头构建一个推荐引擎不是很复杂吗？

不，如果你选择正确的解决方案来照顾你的技术方面，就不会。

Algolia 完善了一个电影推荐 API，易于实施和调整，因此您的企业可以立即开始提供个性化的视频推荐。使用 Algolia [推荐](https://www.algolia.com/products/recommendations/) ，您的开发人员将获得一个简单而强大的 API，来交付您的客户所期望的推荐体验。

要了解电影推荐 API 如何工作以及如何为您的客户改进视频和电影推荐，请立即联系 [美国](https://www.algolia.com/contactus/) 。