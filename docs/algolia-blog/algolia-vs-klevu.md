# Algolia Vs. Klevu:并排比较|Algolia

> 原文：<https://www.algolia.com/blog/product/algolia-vs-klevu/>

Klevu 和 Algolia 是两个流行的搜索平台。Klevu 主要面向带有 Shopify 和 Magento 插件的电子商务应用。Algolia 是为所有不同类型的用例构建的灵活解决方案。

那么，哪一个最适合你的企业呢？

在这两者之间做出决定时，重要的是要考虑关键因素，如期望的用例、性能、功能和持续维护。在本文中，我们将比较每个工具为您的最终用户、业务用户和开发人员提供的体验，以帮助您找到最佳合作伙伴。

## [](#for-users)为用户

如果你最终想寻找一个能改善用户体验的搜索服务，有几个重要因素需要考虑。以下是 Klevu 和 Algolia 如何提供有助于最佳终端用户体验的因素:

### [](#speed%c2%a0)速度

如果用户不能快速找到他们想要的，他们很可能会沮丧地离开你的网站。

据测量，Klevu 上的搜索查询需要 350 毫秒到 2 秒的处理时间。Klevu 搜索引擎基于 Solr，由 AWS、OVH 和谷歌托管。虽然这为用户提供了灵活性，并可以处理不同的用例，但这种技术上的通用性有一个缺点，即限制了 Klevu 充分优化其搜索算法的速度。这导致了行业平均水平的搜索延迟。Klevu 还在 CDN 中存储和缓存前端资产，因此页面加载相对较快。

Algolia 的搜索查询需要花费 [70 到 100 毫秒](https://www.algolia.com/doc/guides/getting-started/quick-start/tutorials/account-metrics-with-the-dashboard/) 来处理。Algolia 的搜索引擎是在考虑速度的基础上从头开始构建的[](https://highscalability.com/blog/2015/3/9/the-architecture-of-algolias-distributed-search-network.html)。由于最初的 Algolia 产品是为在原生移动设备上运行而构建的，因此它被设计成资源非常少，即使是基本的消费设备也可以运行它。搜索引擎现在运行在云中，但它保留了核心系统和优化，因此提供了极其快速和可扩展的搜索性能。



### [](#typo-tolerance)错别字容忍

用户在搜索时经常出错，但这不应该影响引擎或搜索结果的相关性。

通过为索引内容动态生成同义词，Klevu 自动丰富其目录。这对于快速引导新系统非常有用，因为它不需要太多的自定义同义词管理。然而，他们也使用这些项目(以及模糊匹配)来处理拼写错误，当有搜索失败时就搜索这些项目。这往往不如试图自动纠正拼写错误的关键词的系统表现好。

Algolia 有一个强大的错别字容忍度系统，确保用户可以找到内容，即使某个关键字的拼写与索引内容中的不完全相同。它使用 [距离算法](https://www.algolia.com/doc/guides/managing-results/optimize-search-results/typo-tolerance/) 来计算出现的字符替换、添加、删除和置换的数量，并找到“最接近”的单词距离的阈值是可以配置的，这对于处理不同的语言和用例是很重要的。

### [](#autocomplete)自动完成

自动完成通过减少完成搜索所需的时间来改善用户体验。

Klevu 提供了一个“随键入搜索”的功能，从用户键入的第一个字符开始提供自动完成的建议。这些建议包括热门搜索和最近搜索。虽然 Klevu 的自动完成功能提供了良好的结果和上下文，但其较慢的搜索性能提供了不太理想的用户体验，因为用户可能会看到他们键入的时间和结果填充的时间之间的延迟。

阿哥还提供了 [自动完成](https://blog.algolia.com/search-autocomplete-on-mobile/) 和 [查询建议](https://www.algolia.com/doc/guides/getting-insights-and-analytics/leveraging-analytics-data/query-suggestions/) 。然而，由于 Algolia 的快速搜索性能，用户界面是真正交互式的。这提供了一个动态的环境，鼓励用户尝试搜索并快速找到他们想要的东西，而不会因为延迟和滞后而感到沮丧。

### [](#cross-device-functionality)跨设备功能

无论用户通过何种设备进行搜索，您的搜索都应该可用。

Klevu 支持响应式浏览器界面，可以处理移动设备和所有不同的屏幕尺寸。它特别针对其 Shopify 和 Magento 界面进行了优化。如果你正在使用这些插件中的一个，确保启用 [移动友好布局](https://support.klevu.com/knowledgebase/mobile-friendly-responsive-layout/) 。

Algolia 为各种不同的前端提供客户端软件和 SDK，包括 Android、.NET、Javascript 和 iOS，此外还有一个响应迅速的移动浏览器界面。这意味着你可以在所有不同的平台上为你的用户创建一个快速易用的搜索界面。



### [](#personalized-experience)个性化体验

尽管一个经过良好优化的搜索解决方案肯定会让人感觉个性化，但是使用 [个性化工具](https://blog.algolia.com/build-better-search-personalization-with-control-and-visibility/) 会进一步提升用户体验。

Klevu 允许展示与购物者搜索历史相关的最近浏览和趋势产品。当用户单击搜索栏时，这类似于查询建议。有了这个，用户可以快速跳回最近的搜索或者查看网站上流行的内容。然而，搜索结果本身并不针对特定的最终用户而个性化。

Algolia 能够动态调整搜索结果以满足 [自定义用户需求](https://www.algolia.com/products/personalization/) 。搜索结果可以根据产品浏览量、点击量和转化率等因素进行调整。这些 KPI 可以很容易地调整和配置，以处理不同的用例或业务目标。

## [](#for-businesses)为商家

为了利用网站数据做出明智的商业决策，企业用户需要能够理解和使用搜索工具。他们还需要能够 [分析和利用来自搜索工具的数据](https://blog.algolia.com/internal-site-search-analysis/) 来推动搜索和导航体验的改进。以下是你的业务团队需要考虑的几个基本因素:



### [](#transparency-into-search-analytics)透明度进入搜索分析

拥有强大的分析和报告能力对于做出更明智的战略业务决策至关重要。例如，在搜索中， [自定义排名](https://www.algolia.com/doc/guides/managing-results/must-do/custom-ranking/) 和 [在线销售](https://www.algolia.com/doc/guides/managing-results/rules/merchandising-and-promoting/) 可用于推广季节性产品、新产品线或利润更高的商品。通过分析密切监控您的数据，您可以做出更多由数据驱动的决策，决定推广哪些产品，然后监控这些变化，以确保客户对这些产品做出良好的反应。在选择搜索提供商时，您应该确保他们拥有允许您执行这些类型的分析和监控的所有功能。

Klevu [为搜索到的术语、未找到产品的搜索到的术语、点击的产品和检出的产品提供分析](https://www.klevu.com/features/#analytics) 。这些是通过 Shopify 和 Magento 集成自动收集的。对于使用 API 或其他前端的人来说，指标必须手动推送到后端。

Algolia 自动提供了一个 [数量的数据点](https://www.algolia.com/doc/guides/getting-insights-and-analytics/search-analytics/out-of-the-box-analytics/) ，比如搜索计数、热门过滤属性、热门搜索、不同 IP 和用户计数等等。不需要编码，因为这些指标是通过 API 的使用自动收集的。Algolia 还提供了一个全面的分析仪表板，允许分析师以适当的粒度快速查看和过滤数据。Algolia 还在分析上提供了更多的细化可能性(通过使用标签)，在查询级别提供了更多的分析(例如，点击率、转换率、点击位置)，以及比较分析范围的能力。企业客户也可以通过 API 访问所有这些内容。

### [](#%e2%80%9cwhite-box%e2%80%9d-approach)【白盒】方法

调整和优化搜索是一个迭代和持续的过程，需要针对您的业务和客户。通常，内部搜索的相关性和排名对于商业用户来说是一个黑匣子，他们必须依赖他们的开发者来理解内部排名。因此，重要的是 [搜索规则](https://www.algolia.com/products/rules/) 和配置易于使用和理解。

Klevu 允许各种技术能力的用户修改搜索结果。然而，排序和相关性的逻辑有点不透明。

Algolia 使商业用户能够定制他们的个性化，并选择哪些因素会影响个性化，以确保搜索引擎适合他们非常具体的业务需求。Algolia 的构建使得所有用户都可以通过简单的拖放界面轻松调整搜索规则、相关性、排名等。虽然它的技术开发工具很强大，但 Algolia 认为，对于非技术用户来说，能够快速自己做出更改并跟踪性能是很重要的。

### [](#merchandising%c2%a0)

营销工具对于企业用户根据营销优先级和 KPI 调整内部搜索相关性至关重要。随着消费者品味和技术的不断变化，排名规则和促销活动应受到密切监控和调整。

Klevu 宣传自动分类重新排名。然而，如果你在它的上面做一些销售行为，商业用户将很可能不能控制重新排序，也不能进一步定制搜索引擎。也就是说，Klevu 确实允许满足各种条件的 [提升结果](https://support.klevu.com/knowledgebase/category-navigation-rule-based-merchandising/) 并插入横幅来推广产品。

Algolia 通过无需开发人员参与的可视化用户界面，让企业用户能够轻松调整销售策略。此外，Algolia 的规则允许将高级策略应用于排名策略，以及更多的定制和非常具体的业务方法。



## [](#for-developers)针对开发者

由于开发人员将参与您搜索工具的更新、改进和日常维护，因此您选择的工具能够让开发人员以最少的麻烦完成他们的工作是至关重要的。

### [](#api-first)API-first

由于 Klevu 的设计重点是轻松兼容 Shopify 和 Magento，因此其资源和工具针对这些用户进行了优化。

Algolia 也提供 Shopify 和 Magento 扩展，是一个 API 优先的服务。这确保了开发人员获得与使用第三方插件或其他前端客户端的开发人员相同水平的一致性和质量。通过这样做，企业可以建立在 Algolia 的系统之上，并知道 API 将得到维护、支持和良好的文档记录。

### [](#tooling-and-setup)工装和设置

【Algolia 和 Klevu 都有浏览器客户端和第三方平台插件。

不过，Algolia 也为 Android 提供 SDK，。NET、Javascript 和 iOS，以及一个丰富的、记录良好的 API。所有这些客户端和 SDK 都是通过 [Algolia 的 GitHub](https://github.com/algolia) 开源的，因此所有能力的开发人员都可以轻松地在 Algolia 的基础上进行构建。这允许即插即用和灵活定制更复杂的工作流程。Algolia 还提供了 [健壮的文档](https://www.algolia.com/doc/) 来帮助开发者解决搜索和导航体验各个阶段的挑战。



### [](#indexing)索引

Klevu 通过注入英语、法语和芬兰语目录的同义词来提供自动丰富。它们还为检测查询的属性提供了很好的支持，比如产品的颜色或类型。

Algolia 提供了多种向索引添加结构和标签的功能，以提高搜索的效率和性能。Algolia 的索引时间通常相当快(当然，这取决于您的记录数量)，因为 [基础架构已经针对索引速度](https://blog.algolia.com/inside-the-algolia-engine-part-1-indexing-vs-search/) 进行了优化。它还提供了管理 [不同类型同义词](https://www.algolia.com/doc/guides/managing-results/optimize-search-results/adding-synonyms/) 的工具，包括单向同义词、替代更正和占位符。这些都可以通过 API 和仪表板来管理。



## [](#choosing-the-best-search-partner-for-you)为您选择最佳搜索伙伴

在选择搜索合作伙伴时，你需要确保他们的系统足够灵活，能够满足你的特定用例，并随着时间的推移而扩展。 [请求演示](https://go.algolia.com/deep-dive-demo) 以了解 Algolia 如何为您的企业快速启动搜索体验。