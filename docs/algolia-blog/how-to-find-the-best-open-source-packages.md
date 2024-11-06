# 如何用强大的搜索引擎找到最好的开源包

> 原文：<https://www.algolia.com/blog/engineering/how-to-find-the-best-open-source-packages/>

开源软件包与软件开发密不可分。

开发人员使用软件包进行日期选择和图表、数据库和网络访问、时区和货币转换——几乎任何事情。

和 w e 广泛研究寻找合适的开源包。现在有超过 300 万个开源软件包，每天都有数百个新的冒出来，这使得寻找合适的开源软件包变得很繁琐。

### [](#searching-for-the-best-package-%e2%80%93-the-old-ritual)**寻找最佳套餐——老礼**

假设您正在寻求实现自然语言处理。你从谷歌开始，寻找潜在包裹的线索。你打开像 npm 或 PyPI 这样的注册链接，筛选阅读材料，查看版本历史和下载次数。你打开 GitHub 库的链接，寻找项目维护良好的标志(最后一次提交是什么时候？软件包有足够的维护者和贡献者吗？问题积压看起来像什么？PRs 正在被合并吗？).你也要考虑一个项目有多受欢迎(主要是通过明星数量)。

但是流行度和其他类似的度量标准并不能告诉你事情的全部，因为它们不能传达使用软件包的细微差别，也不能告诉你开发者的体验是什么样的。要执行更定性的查询，您可以询问朋友和同事，并搜索博客帖子、综述、开发人员论坛甚至推文。

### [](#there-must-be-a-better-way)**一定有更好的办法**

在 [Openbase](https://openbase.com/) ，我们认为 *肯定是搜索开源软件包的更好方式。因此，我们建立了一个平台，为开发人员简化流程，在一个单一的多方面搜索界面中为他们提供相同的指标和资源。我们的开源搜索引擎 每次都能帮助开发者找到合适的包，没有麻烦。*

鉴于搜索对我们产品的重要性，我们投入了大量的时间和精力来打造最好的搜索 UI/UX。对我们来说幸运的是，[Algolia](https://www.algolia.com/developers/)，一家领先的网站搜索 SaaS 提供商，为我们提供了一个无限空间、使用和功能的大幅折扣，以及一个持续的合作伙伴关系来帮助我们不断改进我们的产品。

### [](#the-use-case-searching-for-the-right-package-to-accomplish-a-specific-task)**用例:寻找合适的包来完成特定的任务**

作为一名开发人员，你正在寻找一个包来完成一个别人已经完成无数次的特定任务，并且你不想重新发明轮子。

所以你开始寻找…

![openbase algolia open source search engine](img/f6b7930f4df2f58cbed043f618e6fc33.png)

为了帮助缩小您的选择范围，Openbase 团队与开源社区持续合作，为想到的每个任务筛选成千上万个类别。

![categories](img/0511abf9fc4a9454a103fd74107f4278.png)

假设您正在为您的 web 应用程序寻找一个日历组件。

进入 Openbase 时，首先搜索“日历”，然后从搜索结果中选择日历类别。

![search results](img/15cc7c4e17a3717a9e798572ace5fe06.png)

在这个类别中，你可以享受一份精选的最佳日历库列表。虽然类别很有用，但真正的力量来自于根据框架(例如“React”)或其他基于您的特定需求的更高级的过滤器来过滤列表。

**Openbase 允许你比较软件包的星级、下载量、最后提交量和用户情绪。**

通常，你会列出几个看起来很合适的有前途的包。

![page view](img/2a5288cf3f3408aa0f0b107595e383d0.png)

然后，您可以打开每个包的专用页面。我们发现大多数开发者在这个阶段选择 2-4 个包。

在软件包页面，您可以在一个地方获得您需要的关于软件包的所有数据:

*   **人气洞察:** 了解各套餐的相对市场份额和趋势。
*   **保养心得:** 这个包有保养吗？我能依靠它吗？(另请查看“维护”选项卡，了解更多信息)。
*   **包属性:** 是否支持 TypeScript？它可以摇树吗？它有多少依赖关系？
*   **评分和点评:** 是硬还是好用？文档怎么样？是不是自以为是？来自 Openbase 社区的评论有助于您从其他开发人员使用该软件包的经验中学习。

掌握了所有这些知识，您就可以决定适合自己的套餐了！

### [](#appendix-a-look-under-the-hood)**附录:引擎盖下的一瞥**

以下是我们用来构建 Openbase 的一些技术:

在幕后，Openbase 不断地从各种数据源中提取数据，以保持数据的新鲜:

我们会推出新版本、元数据、下载次数等等。

最后，这些包中的许多都伴随着 GitHub repo，所以我们有一个额外的服务 [抓取 GitHub](https://github.com) 来收集元数据，以及关于提交、问题和 PRs 的数据。

我们将继续添加新指标，帮助您选择合适的套餐。