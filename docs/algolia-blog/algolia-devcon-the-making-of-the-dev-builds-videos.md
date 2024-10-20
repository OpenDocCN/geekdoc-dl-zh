# Algolia DevCon:开发者如何构建应用程序

> 原文：<https://www.algolia.com/blog/engineering/algolia-devcon-the-making-of-the-dev-builds-videos/>

2022 年 9 月 14 日和 15 日，Algolia 举办了一场名为 DevCon — [的会议，点击](https://www.algolia.com/blog/algolia/its-a-wrap-the-algolia-devcon-2022-recap/) 查看会议摘要并观看演示视频。有很多精彩的演讲，比如 OpenBase 的高级开发人员讲述了他们在构建他们非常受欢迎的服务时学到的五个教训，以及 Algolia 的产品经理 Alexandre Collin 向我们介绍了 InstantSearch 的发展。在我们通过设备观看的任何大型活动中，广告也是节目的一部分！因此，如果你想了解并观看由开发者社区成员制作的很酷的项目——开发构建——我们在 DevCon 的主要演讲之间溜了进去，那么请继续阅读！

## [](#anthony-%e2%80%94-orchestra-search)安东尼—管弦乐队搜索

[https://www.youtube.com/embed/_C9PMLr_XC8?feature=oembed](https://www.youtube.com/embed/_C9PMLr_XC8?feature=oembed)

视频

Anthony 是 RedwoodJS 项目的核心维护者，也是 QuickNode 的开发者拥护者。在学习 web 开发之前，他是一名音乐老师(*tech 里似乎有很多人有音乐背景*)，所以 Anthony 一直有创造性解决问题的头脑。

我敢肯定，我们这些音乐世界之外的人都想知道管弦乐队在这一点或那一点上是如何工作的，所以安东尼开发了一个应用程序来帮助我们搜索管弦乐队中的不同乐器，它们所属的类别，以及它们可以击中的音符。您可以在这里找到[open index](https://github.com/ajcwebdev/algolia/blob/main/data.json)——将 JSON 上传到任何 Algolia 应用程序中，开始试验，甚至在您拥有自己的任何数据之前。

## [](#ashley-%e2%80%94-ecommerce-chatbot)阿什利—电子商务聊天机器人

[https://www.youtube.com/embed/61r_YZn5x84?feature=oembed](https://www.youtube.com/embed/61r_YZn5x84?feature=oembed)

视频

Ashley 是 RenderATL 的开发者拥护者，也是 Mux 的社区工程师。作为一名社会学家和 Pinterest、Tumblr 和 Mailchimp 的校友，她完全有能力建立真正帮助人们并让我们感觉良好的技术。

对于 Algolia DevCon，她建立了一个带有[商务的复古奢侈品寄售店原型。JS](https://commercejs.com/) 、 [React Simple Chatbot](https://github.com/LucasBassetti/react-simple-chatbot) 和 [Algolia](https://algolia.com) 让时尚感触手可及。该店面页面包含一个聊天机器人，作为一个更具交互性的搜索工具，允许用户流畅地描述他们的风格，机器人会推荐一些符合用户特定审美的商品。它的工作原理是从用户发送给聊天机器人的消息中分离出关键词，并将 Algolia 搜索结果动态加载到该窗口中，使客户能够灵活地找到符合其抽象风格的内容，而无需通过品牌名称或其他特定限定符进行搜索。

Ashley 面临的最大挑战之一是同步商业中的产品库存。JS(客户会看到它显示在店面上)和 Algolia 中的搜索索引(聊天机器人正在搜索的内容)。解决这个问题的一个方法是使用 Algolia Commerce。JS 官方集成，这消除了数据管道的复杂性，但她选择了自己与定制商务合作。每当产品数据库改变时触发 Algolia 索引更新的 JS webhook。一个工具组合能力的伟大展示！

## [](#trezy-%e2%80%94-twitch-bot)Trezy—Twitch Bot

[https://www.youtube.com/embed/Sq_9YoM1im4?feature=oembed](https://www.youtube.com/embed/Sq_9YoM1im4?feature=oembed)

视频

Trezy 是一个流行的编码直播流和串行开源贡献者，尤其以他的项目 [fdgt.dev](https://fdgt.dev/) 而闻名，这是一个 API，所以测试和调试 Twitch 应用程序实际上不会掏空我们的钱包。

对于 Algolia DevCon，Trezy 构建了一个 Twitch 机器人来与 DocSearch 进行交互，以便将文档信息提供给流媒体观众，而无需他们离开 Twitch。该应用程序有三个命令，它们协同工作，在支持文档搜索的文档网站中查找内容，然后对结果进行分页。对于聊天版主来说，这是一个有用的工具，因为它可以让他们轻松地解释流中正在发生的事情，以及观众如何在家跟随，即使没有以前的经验。

如果您有兴趣了解 Algolia 如何工作，以及您可以在您的团队中使用它做什么，这就是 DevCon 的其余部分！在 YouTube 上看看[这个播放列表，观看完整长度的演示和演示，如果您有任何具体问题，请随时在 Twitter](https://www.youtube.com/playlist?list=PLuHdbqhRgWHLRlmvQ1OKLdjslSxXrAAjk) 上联系我们[！](https://twitter.com/algolia)

直到明年，

Algolia DevCon 团队