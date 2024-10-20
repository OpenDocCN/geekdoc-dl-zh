# 发布我们的官方 API 客户端规范和通用测试

> 原文：<https://www.algolia.com/blog/engineering/release-api-client-specifications-common-test-suite/>

Algolia 非常关心开源。我们维护开源库，如 [我们的 web 和移动即时搜索库](https://www.algolia.com/products/instantsearch/) 以构建丰富的用户体验，并与 OpenCollective 社区合作 [帮助资助开源项目](https://blog.algolia.com/supporting-open-source-projects) 。

我们的一个团队负责维护我们的底层开源库，也称为 API 客户端，以及几个 web 框架集成。那些 API 客户端让我们的客户将他们的应用程序连接到 Algolia 服务( [搜索](https://www.algolia.com/products/search/) ， [分析](https://www.algolia.com/products/analytics/) ， [个性化](https://www.algolia.com/products/personalization/) )。更准确地说，这意味着我们必须以 9 种不同的风格维护同一个 API 客户端:C#、Java、Kotlin、Scala、Go、PHP、Ruby、Python 和 JavaScript。

因此，每次需要添加新功能时，我们都必须以最佳方式同步，以便在客户的公共 API 上公开它。同样的事情也适用于 bug:一旦我们在我们的一个客户端上检测到一个潜在的问题，我们也会在所有的客户端中检查它的存在。最后，我们依靠测试来确保功能和性能方面。

今天，我们开源了一个新的 GitHub 库:[algolia/algolia search-client-specs](https://github.com/algolia/algoliasearch-client-specs)来跟踪所有与我们的 API 客户端的公共 API 相关的变化，它们的测试和它们的实现细节。这个存储库还包含我们的 API 客户端的全功能通用测试套件(也称为 CTS)。对于任何对为不受支持的语言实现 Algolia REST 端点的 API 客户端感兴趣的人来说——或者只是为了好玩——这个库是最好的切入点，还有针对 REST 端点的官方 Algolia 文档 。

因为这个存储库对我们的 API 客户端的当前预期状态有一个只读视图，所以问题需要通过[support@algolia.com](mailto:support@algolia.com)，我们的 [话语社区论坛](https://discourse.algolia.com) 或 [堆栈溢出](https://stackoverflow.com/questions/tagged/algolia) 直接报告给我们的团队。