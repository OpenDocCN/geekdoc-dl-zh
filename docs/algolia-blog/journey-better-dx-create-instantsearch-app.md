# 更好的 DX 之旅:创建即时搜索应用程序

> 原文：<https://www.algolia.com/blog/engineering/journey-better-dx-create-instantsearch-app/>

[*即时搜索*](https://community.algolia.com/instantsearch.js/) *by Algolia 是一个前端库家族，在 Algolia APIs 之上创建搜索 ui。*

今天，我们将介绍 [*创建即时搜索应用*](https://github.com/algolia/create-instantsearch-app) : **一个命令行界面(CLI)来从终端** 引导即时搜索应用。你需要的只是[node . js≥8](https://nodejs.org/en/blog/release/v8.0.0/)，不需要安装其他的。

[https://www.youtube.com/embed/Lv0j__EdVqg?feature=oembed](https://www.youtube.com/embed/Lv0j__EdVqg?feature=oembed)

视频

*创建即时搜索应用预览*

## [](#why)**为什么**

开发者体验(DX)一直是 Algolia 的 DNA。我们一直想改善开发者和我们产品之间的互动。在 Algolia 仪表板上花费了大量的努力来方便您的搜索设置，在我们的 API 客户端上花费了大量的努力来与我们的服务器轻松通信。另一方面，我们构建搜索界面的前端解决方案——instant search——虽然功能强大，但开始使用起来相当复杂。

当只有[instant search . js](https://github.com/algolia/instantsearch.js)可用时，开发一个普通的 JavaScript 应用程序是非常简单的。既然我们支持很多环境( [React](https://github.com/algolia/react-instantsearch) ，[React Native](https://github.com/algolia/react-instantsearch)，[Vue](https://github.com/algolia/vue-instantsearch)，[Angular](https://github.com/algolia/angular-instantsearch)，[iOS](https://github.com/algolia/instantsearch-ios)，以及 [)我们已经提供了很好的组件原语来构建搜索 ui，但是我们还能做得更好吗？](https://github.com/algolia/instantsearch-android)

我们从脸书汲取灵感，更具体地说是 [创建 React 应用](https://github.com/facebook/create-react-app)——React 团队正在开展的一个令人敬畏的项目，旨在改善 React 开发人员的体验和入职。

创建即时搜索应用程序让你专注于你的项目，不管你的生态系统如何，你都可以处理所有繁琐和重复的工作:

*   你有新的数据集可以使用吗？运行 CLI 并输入凭据以访问该数据集。
*   你有新的客户需要制作演示吗？ 运行 CLI，输入应用凭证，开始构建体验。
*   您需要 Algolia 团队的帮助吗？ 使用 Create InstantSearch App 生成的 [在线模板](https://github.com/algolia/create-instantsearch-app#previews) 发送给我们！我们将尽最大努力帮助您获得出色的搜索体验。

这个工具是关于实际的搜索实现，而不是构建一个应用程序的过程。它根据给定的信息创建一个工作 UI。在宣布之前，我们已经在内部使用 Create InstantSearch 应用程序几个星期了，它极大地提高了我们的工作效率。

## [](#how-we-use-the-tool-at-algolia)**我们如何在 Algolia 使用工具**

**帮助快速取悦用户**

在 Algolia，解决方案工程师是面向客户的工程师，为客户和合作伙伴构建概念验证。他们经常需要创建即时搜索应用程序，并在其上进行演示和迭代。这个工具帮助他们提高开发速度，展示他们的业务的可能性。

**帮助阿哥利亚帮助你**

臭虫繁殖通常是解决问题的最佳方式。创建即时搜索应用程序允许用户重现 bug:

后者现在是我们帮助客户发现问题的首选方式。我们给他们发送一个在线模板的链接，他们把它分出来并展示 bug。这完全发生在浏览器中。

**帮助你了解即时搜索**

工具永远不应该妨碍新用户。在重写整个 Algolia 文档时，我们决定跳过所有入门指南中的工具解释，提倡使用 Create InstantSearch 应用程序。我们甚至用它来引导我们新的 [互动教程](https://www.algolia.com/doc/onboarding/) 。

## [](#this-is-only-the-beginning)**这只是开始**

创建 InstantSearch 应用程序只是我们迈向更好的前端 DX 之旅的开始。我们有更多的目标要实现，并期待分享更多的改进。同时，试试我们的即时搜索库吧！

> npx create-instant search-app my-app

你可以在 GitHub 上找到文档和源代码 [。如果您有任何反馈，请不要犹豫，在 Github](https://github.com/algolia/create-instantsearch-app) 上让我们知道[，在](https://github.com/algolia/create-instantsearch-app)[上发推文给我们](https://twitter.com/algolia)或者评论这篇文章。