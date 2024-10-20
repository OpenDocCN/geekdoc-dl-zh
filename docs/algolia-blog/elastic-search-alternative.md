# Elasticsearch 替代方案可以叠加吗？

> 原文：<https://www.algolia.com/blog/product/elastic-search-alternative/>

如果你现在正在看这个页面，你已经知道为你的网站配备一个出色的搜索和发现体验的重要性。如果你正在考虑像 Elasticsearch 这样的开源选项，你可能需要一个灵活的解决方案来适应你的特定用例。

虽然 Elasticsearch 是一个 [RESTful](https://www.smashingmagazine.com/2018/01/understanding-using-rest-api/) 搜索和分析引擎，确实涵盖了各种各样的用例，但并不是每个企业都有开发人员的实力和知识来让它在高水平上运行。

另一方面，Algolia 是一个托管的 [搜索即服务](https://blog.algolia.com/what-is-search-as-a-service/) 解决方案，专注于即时、相关和个性化的搜索和发现体验。

所以，问题仍然是:Elasticsearch 或 Algolia 适合你吗？在本帖中，我们将通过 3 个主要问题来引导你完成这个过程:

*   对于你的用户:你想建立什么样的体验？
*   对于您的业务:谁将与您的解决方案互动？
*   对你的开发者来说:你打算在这个过程中投入多少时间和资源？

## [](#for-users)为用户

在网站上实施搜索的最终目标是改善整体用户体验，最终推动转化率、销售额、停留时间和其他 KPI。让我们看看 Elasticsearch 和 Algolia 是如何帮助你为用户打造搜索体验的。

### [](#how-elasticsearch-works-for-users)elastic search 如何为用户服务

使用 Elasticsearch，您将需要从头开始构建您的搜索界面，这允许您对界面进行策划和微调，以便为您的用户提供最佳服务。Elasticsearch 通常最适合于 [文档搜索目的](https://blog.algolia.com/full-text-search-in-your-database-algolia-versus-elasticsearch/) ，但是通过一些专业知识，它可以适合任何范围的用例。

不幸的是，设计和实现最适合用户的功能可能需要大量的工作和技术知识，而大多数开发团队并不具备这些。即使对于有技术能力和时间的团队来说，构建、测试和启动搜索界面也需要几个月的时间。如果你使用 Elasticsearch 来构建面向用户的搜索，请确保所有的商业利益相关者都知道，你还需要一段时间才能为用户提供出色的搜索体验，并收获你辛勤工作的成果。

### [](#how-algolia-works-for-users%c2%a0)Algolia 如何为用户工作

Algolia 的设计从基础到动力体验都推动了转变。Algolia 使用内置功能，如个性化、多语言功能、打字错误容限和同义词支持，为每个用户提供最佳结果。

Algolia 在几个方面改善了用户的搜索体验:

*   **让你的用户更快达到他们的目标。使用 [联合搜索](https://blog.algolia.com/what-is-federated-search/) 可以容忍输入错误并给出随输入搜索的结果，搜索体验更加直观，用户从一开始就参与搜索。**
*   **满足用户需求，不受平台限制。** 通过 [与](https://www.algolia.com/developers/integrations/) 其他关键平台和服务的集成，在移动设备、web 甚至语音助手等各种设备上部署无缝一致的体验。
*   **在用户层面个性化体验。利用用户在各种设备上与你的平台的互动，进一步个性化他们每次回到你的网站的体验。**

在用户所在的地方与他们会面并实现强大的搜索最终会提高转化率、搜索使用率并降低跳出率。

![](img/250a3507216995d181ecb92abc477de8.png)

## [](#for-businesses)为商家

业务团队可能不会陷入搜索解决方案的日常运营中，但他们会关注搜索如何支持业务目标。让我们来看看 Elasticsearch 和 Algolia 如何支持您的业务团队的优先事项。

Elasticsearch 如何为业务团队工作  乍一看，elastic search 看起来很有吸引力，因为初始价格很低。但是，总拥有成本远远高于初始成本。当您考虑到构建界面、维护搜索引擎、添加新功能、支持和排除引擎故障等成本时，构建搜索引擎的终生成本要高得多。

此外，有了 Elasticsearch，搜索仍然是业务团队的黑匣子。业务团队可能不了解为什么项目和产品会有这样的排名。业务团队必须依靠开发人员来解释排名算法、更改算法或添加新的排名因素。虽然在任何搜索实现中，业务-开发人员的关系都需要是牢固的，但在使用开源搜索引擎时，这种关系必须是牢固的 *【尤其是* 】。

要从 Elasticsearch 数据中获得业务洞察力，您需要另一种数据可视化工具。无论是 Kibana(弹性堆栈的一部分)，还是另一种数据可视化工具，请确保将您的分析工具的成本计入构建引擎的总成本。

### [](#how-algolia-works-for-business-teams)Algolia 如何为业务团队工作

Algolia 可以帮助您更好地了解用户的需求，并改进您的搜索界面，从而更好地推动业务成果。

*   **赋予业务团队调整排名因素和相关性策略的权力。Algolia 排名公式可以帮助任何用户快速理解为什么一个结果的排名高于另一个结果。此外，业务用户和技术用户可以根据相关性策略和业务优先级调整和微调排名。**

*   分析你的搜索数据来改进你的网站。用户每次在你的网站上搜索，都会产生 [有价值且可操作的数据](https://blog.algolia.com/internal-site-search-analysis/) 。使用 Algolia 内置的 [分析仪表板](https://www.algolia.com/products/search-and-discovery/analytics/) ，您可以快速了解您的搜索执行情况，包括识别比其他人执行得更好的查询和您网站上错过的机会。网站搜索分析包含在 Algolia 中，无需额外费用。

*   利用机会，用巧妙的商品推销来推销你的产品。 使用 Algolia 的 [内置销售工具、](https://blog.algolia.com/what-is-searchandising/) 让您的非技术用户充分利用他们对您业务的了解，让他们推广最佳内容并完善自己的策略。

**T38**

## [](#for-engineers)为工程师

对于许多公司来说，设计网站搜索的责任将落在开发团队身上。您会希望开发人员的体验尽可能的顺畅，为您的工程师创造最快的整体成长路径。让我们看看这两个选项，看看它们将为您的开发人员创造什么样的体验。

### [](#how-do-developers-work-with-elasticsearch)开发者如何使用 Elasticsearch？

工程师们将不得不做很多繁重的工作，来让一个基于弹性搜索的搜索界面离地运行。幸运的是，Elasticsearch 在线开发者社区非常活跃。这个社区对你的工程师来说是一个很好的资源，甚至可以为特性开发做出贡献。

然而，对于许多团队来说，Elasticsearch 对工程团队的好处可能仅限于此。设计一个可用的、特定于案例的搜索界面是一个耗时的过程，需要很多搜索方面的专业知识，而许多开发团队并不具备这些专业知识。由于在开源平台上构建允许大量的灵活性，一个有搜索经验的开发者可能会以不同的方式处理它；如果主要开发人员离开，组织可能需要“重新学习”如何构建和维护引擎。持续的监控、维护、故障排除和创新工作可能会利用您的团队所不具备的能力。

Elasticsearch 通常最适合拥有大型、高技能开发团队的组织，这些团队以前从事过搜索引擎的开发，并将拥有持续的资源来维护搜索引擎。

### [](#how-do-developers-work-with-algolia)开发者如何与 Algolia 合作？

Algolia 专为开发人员而设计。我们的 [丰富的文档库](https://www.algolia.com/doc/) 包括从启动和运行 Algolia 到为每个平台构建复杂的搜索 UI 的所有技术指南。

Algolia 在几个关键方面支持开发人员:

*   **借助高可用性基础设施提高正常运行时间。 Algolia 构建在安全可靠的基础设施之上，我们提供行业领先的 99.999% SLA(适用于具有 1000 倍折扣政策的精选计划)。 [当前 API 状态](https://status.algolia.com/) 始终可在 Algolia 网站上获得。**

*   使用我们的前端库加快您的实施时间。使用我们的 [即时搜索库](https://www.algolia.com/doc/guides/building-search-ui/getting-started/js/) 快速实现无缝体验，推动用户转换。所有用户都可以通过云即时获得创新和更新，无需额外费用。
*   获取任何平台或用例的灵感。[Algolia 社区](https://community.algolia.com/) 提供了一个来自 Algolia 用户和 Algolia 团队的项目库，帮助您开始使用您已经使用的任何框架、平台或编程语言。
*   **利用内置的安全性。** 受益于内置 [安全性的解决方案](https://www.algolia.com/distributed-secure/security-compliance/) ，这将节省您宝贵的时间，使您的团队能够专注于其他业务需求。

### [](#drive-business-growth-and-impact-with-algolia%c2%a0)推动业务增长并对阿洛利亚产生影响

最终，在 Elasticsearch 和 Algolia 之间作出决定归结为是否 [建立或购买](https://www.algolia.com/blog/product/algolia-vs-open-source-search/) 你的搜索引擎。如果你有一个核心产品或服务需要关注，考虑把搜索留给 Algolia 的搜索专家。

Algolia 会影响业务的方方面面，从最终用户到业务团队，再到开发人员。阅读更多关于 Forrester 如何评估 [Algolia 的总体经济影响](https://www.algolia.com/dg/forrester-tei-report/p/1) 的信息，看看 Algolia 能对你的业务产生的影响。