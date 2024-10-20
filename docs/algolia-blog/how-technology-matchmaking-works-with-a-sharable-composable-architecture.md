# 可共享、可组合架构的技术匹配

> 原文：<https://www.algolia.com/blog/product/how-technology-matchmaking-works-with-a-sharable-composable-architecture/>

对于工程师来说，一个在线市场意味着 *生态系统*——经营一个企业的前端和后台系统。现代生态系统由第三方软件组件**组成，这些组件由各自领域的专家**构建。例如，搜索和人工智能专家构建搜索和发现组件，金融公司提供支付功能，业务专家创建交易和订购流程。

生态系统工程师不必编写一个 可组合架构 的业务逻辑。相反，他们将第三方 API 组件集成到他们现有的基础设施中，并且只专注于管理组件之间的数据和业务逻辑交换。

在我们之前关于 [匹配和白标](https://www.algolia.com/blog/product/white-labeling-and-matchmaking-technology-in-a-growing-digital-marketplace/) 的文章中，我们展示了组件和可组合生态系统本身可以 *共享* ，其中一个企业直接与其他企业共享其基础设施和软件组件，或者向他们展示如何在自己的基础设施上运行相同的组件。本文的重点是“牵线搭桥”，或者说涉及一个市场的部分 *将* 专业编写的 API 组件引入其他市场。

## [](#typical-matchmaking-scenarios)典型的相亲场景

为了说明匹配，我们在第一篇文章中描述了两位企业家如何建立一个成功的市场生态系统。他们的故事从这里开始。他们开始共享他们的接口 *和* 基础设施，这基本上是亚马逊、Etsy 和所有其他市场所做的。随后出现了衍生产品、牵线搭桥和白色标签。

### [](#multiple-uis)多个 ui

他们的一家供应商，一家硬币收集公司，想要一个不同的界面，但不想构建或购买自己的基础设施。marketplace 架构(API 组合的、无头的)使工程师能够将用户界面从后端组件中分离出来。通过这种方式，硬币收藏家能够从他们自己的网站访问生态系统的 API 服务。

### [](#from-ecommerce-to-media)从电商到媒体

另一家厂商想要流媒体音乐。他们创建了一个混合基础设施:一些组件运行在市场的基础设施上；其他组件(用于媒体流)在供应商的基础设施上运行。

### [](#white-labeling)白色标签

其他厂商看到了需要 *复制* 的基础设施，利用自己的基础设施建立自己的网站。那时，最初的 marketplace 工程师看到了一个给他们的 marketplace 生态系统贴上白色标签的机会，这使得外部供应商能够通过以下方式构建他们自己的 market place:

*   购买原始市场生态系统的白标版本，并将其安装在他们自己的基础设施上
*   购买第三方组件，然后根据他们的需求进行集成和配置。

## [](#so-how%e2%80%99s-this-done-an-quick-overview-spoiler-alert)那么这是怎么做到的呢？快速概览[剧透警报]

共享其生态系统的企业(A)提供了一个数据交换 API，允许另一家公司(B)将其数据发送到 A 的索引中。公司 A 的数据交换 API 接收、验证并在没有错误的情况下更新其基础设施上的数据。这为公司 A 提供了一个扩展机会，因为他们引入了这些合作和整合渠道。我们将看到这一切是如何与 Algolia 的搜索和发现 API 一起工作的。

更进一步，A 可以通过白标共享其生态系统，允许 B*将其生态系统软件* 复制到自己的服务器上。在这个场景中，B 管理自己的生态系统，并成为提供第三方 API 组件的公司的直接客户。

让我们进入细节。

## [](#the-software-components-behind-the-ecosystem-and-customer-journey)软件组件背后的生态系统和客户旅程

这里有几个关键的驱动力在起作用:

*   搜索
*   发现
*   订购
*   付款
*   交货

当然还有更多。例如，在后端，有库存管理、客户关系管理、定价工具和其他此类从头到尾运营业务的应用程序。在前端，有用户评论、建议和图像，仅举几例。

市场生态系统提供以下服务来创建客户之旅:

1.  首先，顾客搜索和发现，然后决定他们想要什么
2.  接下来，他们订购、付款并收到一份快递(或流、阅读、下载)

我们不会讨论第二部分。您可以探索管理[订购流程](https://www.pomelopay.com/blog/advantages-online-ordering-system)(购物车、愿望清单、客户账户、CMSs)[支付](https://www.linnworks.com/blog/online-payment-methods)(一般支付、一键式、延期支付)和[交付](https://www.getfareye.com/insights/blog/delivery-management-software)(链接–快速、提货/收集、流、读取)的大量 API 和平台。

在本文中，我们感兴趣的是客户旅程的第一部分，即搜索和发现，在这一阶段，消费者搜索或浏览他们想要购买、浏览、阅读或下载的内容。正如您将看到的，搜索和发现不仅仅是一个搜索栏，它还涉及最终用户与数字市场交互时内容的显示和导航。

## [](#3-ways-to-share-a-search-and-discovery-api)共享搜索和发现 API 的 3 种方式

Algolia 为搜索和发现之旅提供软件组件。如前所述，搜索和发现不仅仅包括搜索栏，还包括菜单导航、搜索、浏览、分面&过滤，以及搜索结果、分类页面和产品视图的相关性和视觉布局。从本质上讲，在客户选择他们想要购买的商品(或观看、阅读、下载)之前，搜索和发现涵盖了前端用户旅程的每个部分。购买后，它还会通过“继续购物”按钮继续。

所有生态系统共享的基础是一个灵活的 [API 优先](https://www.algolia.com/doc/api-client/getting-started/what-is-the-api-client/php/)、功能丰富的产品。首先，A 公司使用 Algolia 这样的搜索和发现 API，创建了一个以搜索为主导的生态系统。公司 A 与公司 1 共享其生态系统..无论是合作伙伴还是衍生产品。

要理解分享是如何工作的，重要的是要理解[什么是](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/)。Algolia 的搜索和发现技术主要依靠一个特别设计的索引和[快速索引过程](https://www.algolia.com/blog/product/an-exploration-of-search-and-indexing-real-time-indexing-scenarios/)。一旦一家公司提取并格式化了它的数据，就会使用一个索引 API 来添加、删除或更新 Algolia 服务器上的数据。因此，数据位于 Algolia 托管的服务器上，这些服务器位于世界各地，以最大限度地减少交易延迟。该公司可以选择最适合其用户群的任何地区。该 API 可以访问任何地区的任何服务器。

**这里有三种方法可以共享搜索和发现 API:**

*   **相同的指数，相同的网站**——公司使用相同的网站和指数。这是第一步，没有分拆，没有白标。亚马逊和 Etsy 就是这么做的。所有供应商将他们的数据发送到同一个共享索引，市场提供网站。
*   **同一指数，不同网站**——公司共享同一指数，但不一定是同一网站。在这种情况下，公司 1..n 将他们的数据直接发送到 A 公司的索引，但是他们建立了自己的网站，并从他们自己的独立网站访问共享的索引。这是一个 UI 匹配场景，其中网站的功能(如搜索、导航、商品销售和其他所有功能)作为他们自己网站的组件介绍给供应商。
*   **相同的指标** ***结构*** **，不同的服务器**——公司从 marketplace 中 100%分离出来，创建自己的 marketplace。但是首先他们获得了索引结构的副本(没有数据),并在他们自己的 Algolia 帐户/服务器上重新创建它。然后他们用自己的数据填充它。这两个指数现在已经分开，两者之间不再有任何联系。这些公司必须建立自己的网站，成为 Algolia 的直接客户。在这里，我们看到了匹配和白标，在前端和后端，供应商购买、集成和配置他们已经了解的组件。

## [](#how-to-share-the-search-index-and-what-makes-up-the-index)如何分享搜索索引，以及*什么*组成索引

索引面临的挑战是如何管理供应商发送给市场索引的数据。这是通过数据交换层完成的，数据交换层接收数据并对其进行验证。为此，市场向其供应商公开了一个数据交换 API。这个 API 接受供应商的数据，对其进行验证，如果数据没有问题，就将其发送给 Algolia。

因此，有两个 API:

*   更新搜索索引的 Algolia API
*   接收和验证输入数据的市场 API

我们不会把 marketplace API 称为 ETL。一个 ETL 通常是**E**extracts，**T**transforms， **L** oads data。但是如这里所定义的，不需要任何 **E** 提取或 **T** 转换，因为市场会要求已经在同一个 JSON 中结构化和格式化的数据被发送到 Algolia 的 API。

因此，marketplace API 的工作是接收-验证-加载数据。验证过程应确保:

*   数据完整，格式正确，不会破坏 Algolia 索引
*   数据不太大或发送不太频繁，从而降低系统速度

### [](#the-contents-of-the-index)索引的内容

市场必须创建一个索引结构来驱动其业务和网站的搜索需求。一个[指标包含](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/in-depth/what-is-in-a-record/) *属性* ，其中有[四个用途](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/#algolia-records) :

*   用于**搜索的属性**(用于匹配用户在搜索栏中键入内容的关键字和文本信息)
*   显示**的属性**(图像或额外的屏幕信息，如价格、作者、品牌)
*   用于**过滤和刻面的属性**(品牌、大小、流派)
*   用于**排序**(价格，日期)**和排名**(销售数量，评分)的属性

此外，一些市场将提供可用于销售和内容管理的属性，如横幅或商品促销。

### [](#the-data-structure)数据结构

只有在市场设定标准的情况下，共享指数才能发挥作用。Algolia 接受一个[无模式 JSON 文件](https://www.algolia.com/developers-tech-blog/code-and-deep-dives/facets-and-faceted-search-making-every-attribute-count)，它足够灵活，允许任何结构或内容。市场利用这种灵活性为其独特的需求创建精确的结构。然而，外部供应商受到更多的限制，因为他们必须使他们的数据符合市场设计的结构。以下是一些指导方针:

*   一些属性需要对所有供应商通用，如项目名称、价格、描述。每个供应商都需要提供这些必需的属性。
*   其他属性将是可选的。虽然每个可选属性都有明确定义的目的(用于搜索、显示、刻面或排名)，但供应商可以随意忽略它。当然，为了在共享网站上最好地展示他们的商品，供应商应该提供大部分(如果不是全部)属性。
*   最先进的市场网站包含动态前端逻辑，因此如果供应商提供某个可选信息，网站可以为该供应商显示不同的内容。

### [](#content-management-merchandising)内容管理，商品销售

*   某些属性可用于促销个别商品。这些属性在销售和内容管理中的实际使用将由市场来管理:工程师为促销设置屏幕占位符，供应商可以通过提供正确的内容来利用这些占位符。

### [](#relevance-index-settings)相关性&指标设置

*   这里不多说了:相同的[索引设置](https://www.algolia.com/doc/framework-integration/rails/index-configuration/index-settings/)会被所有厂商共享。这意味着，所有的配置，如错别字容忍度、排名、可搜索性和方面属性，以及许多其他影响相关性的设置，都将由市场而不是供应商来配置。

### [](#more-about-the-data-exchange-api)关于数据交换 API 的更多信息

市场应用编程接口应该有以下特点:

*   获取将要发送到 Algolia 的相同 JSON 文件。
*   发送响应——成功或错误，并提供详细信息。
*   执行[批处理](https://www.algolia.com/doc/guides/sending-and-managing-data/send-and-update-your-data/how-to/sending-records-in-batches/)，这意味着如果数据非常大，它会将记录分成小批，并以固定的间隔(每批 2 分钟左右)发送。
*   像任何 API 一样，是向后兼容的。

### [](#migrating-the-index-to-a-vendor%e2%80%99s-own-infrastructure)将索引迁移到供应商自己的基础架构

*   这不需要市场或供应商进行结构性改变。供应商只需建立自己的 Algolia 帐户，并将相同的市场设置应用于自己的索引。然后，供应商开始使用相同的 JSON 向自己的索引发送数据。

## [](#how-to-share-the-search-ui)如何分享搜索 UI

关于前端，首先要注意一点:[同一个索引可以支持多个用户界面](https://www.algolia.com/blog/ux/what-is-federated-search/)。这是生态系统共享的关键:不同的供应商创建他们自己的 ui，同时使用相同的索引 API 方法。这是可能的，因为用于搜索的 API 方法可以指向相同的索引和服务器。

### [](#the-search-ui-experience-and-its-look-and-feel)搜索 UI 体验，及其观感

*   搜索的基础是:搜索栏、搜索结果、分面、分页、排序。
*   这些功能从一个用户界面到另一个用户界面看起来非常不同。它们可以放在屏幕上的任何地方。不同网站的用户界面会有很大的不同。但基本面不会改变，基础指数也不会改变。

### [](#the-search-results)搜索结果

*   指标 *设置* 驱动关联。[相关性](https://www.algolia.com/doc/guides/managing-results/relevance-overview/in-depth/defining-relevance/)可以归结为搜索结果的*顺序* 。
*   也就是说，索引中的 *数据* 可以根据其内容和格式对这些设置做出不同的响应。因此，当供应商发送数据时，他们需要仔细遵循索引提供商的内容指南。
*   当供应商将其数据移动到自己的 Algolia 帐户和服务器时，它们最初可能会保持相同的结构和设置。然而，随着时间的推移，他们可能会改变索引配置和数据格式，以适应不断变化的需求，从而保持竞争力。*这也是*复制生态系统和白标的主要原因——对搜索体验的更多控制。

### [](#discovery)发现

## [](#sharing-your-expertise)分享你的专长

任何关于技术的讨论总是与创造技术的人有关。虽然 API 提供财务、业务和用户界面专家功能，但故事是关于设计和开发软件的人，以及最终以 API 的形式公开他们的专业知识的人。这些专家既有技术性的，也有非技术性的。他们的勇气和分享是我们数字经济的动力。

—

Algolia 搜索和发现与无头生态系统紧密结合。要了解它如何改变您的数字战略，请免费注册[](https://www.algolia.com/users/sign_up)，亲自体验。或者今天从我们的搜索专家那里获得一个定制的 [演示](https://www.algolia.com/demorequest/) 。