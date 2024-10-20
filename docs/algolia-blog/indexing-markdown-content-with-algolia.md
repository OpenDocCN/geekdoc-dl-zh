# 使用 Algolia 索引减价内容

> 原文：<https://www.algolia.com/blog/engineering/indexing-markdown-content-with-algolia/>

我与 Starschema 全栈工程师 Soma Osvay 合作，写了一个我非常喜欢的项目:开发人员文档。Soma 和我都希望您喜欢这篇文章，其中我们讨论了用例、挑战以及 Soma 提出的实现。

* * *

在 Starschema，我们已经完成了许多项目，作为 markdown documentation 咨询的一部分。当我们支持这些现有的解决方案或者想要开发一个新的项目时，我们经常想要搜索所有的文档。我们目前没有解决方案，因为我们必须手动完成这项工作，这耗费了我们很多工时。

## [](#use-case)用例

我们需要能够搜索与特定主题相关的项目，以检查实现细节，从代码中获得灵感，等等。销售团队还需要一个解决方案来确定我们的团队为某些主题完成了什么样的项目，并能够快速将其传达给潜在客户。对于我们的开发人员来说，拥有类似内部堆栈溢出的东西也很棒，因为我们在 Tableau 中做了很多深入的技术工作。项目经理还需要能够确定哪些员工使用过某些技术，这样他们就可以快速、轻松地得到问题的答案。

简而言之，我们需要能够基于以下属性搜索我们自己的项目:

1.  使用的技术(编码语言、数据库等)
2.  项目关键字(标签)
3.  项目文件本身，如安装说明、技术文件等。
4.  项目发生的时间框架

所有这些属性都存在于内部文档降价文件中——我们只需要一种方法来搜索它们。

## [](#implementation-plan)实施计划

计划使用 NodeJS CLI 作为概念验证，它将:

1.  搜索顶级的 GitHub 公共库并获取 README markdown 文件(这将最终代表我们的内部项目)
2.  将文档文件与项目的基本信息(标题、编程语言、标签等)一起存储到 Algolia 中

CLI 将包含高级日志记录和命令行参数，以便于使用。我们还想把它放在网上，这样人们就可以试用了。

## [](#challenge)挑战

眼下最大的挑战是记录的大小——Algolia 最多只允许我们的记录达到 100KB。然而，大多数降价文档文件要大得多。解决方案是我们需要在索引中将 markdown 文件分成多个部分。然后，我们还需要确保当我们搜索某个东西时，单个项目只出现一次——即使它被分成多个记录。

幸运的是，Algolia 有一个独特的功能，所以我们可以很容易地重复这些结果。

## [](#indexer-implementation)索引器实现

为了尽可能简单地使用索引器，我选择为它创建一个 CLI，如上所述。提供所需的参数后，该工具将自动初始化存储库，删除任何现有记录，并配置相关性设置。

为该工具提供动力的是一个简单的 GitHub API，它获取所需数量的顶级存储库，提取它们的所有元数据，并下载 README 文件。它还会过滤掉缺少所有者或缺少自述文件的存储库，为我们提供最佳结果。最后，它还会将 markdown 内容转换为 HTML，以便在前端更容易地呈现。

为了控制记录大小，该工具会自动将超过 50k 个字符的 READMEs 分割成更多的记录。这样，记录不会变得太大，但是一个记录仍然应该服务于几乎所有的存储库。然后，它会将这些信息一次同步到 Algolia 100 条记录，这样我们就可以满足他们的批处理建议，如文档中的[这里的](https://www.algolia.com/doc/guides/sending-and-managing-data/send-and-update-your-data/how-to/sending-records-in-batches/)所述。

## [](#frontend-implementation)前端实现

作为起点，我利用了 Algolia 发布的 [create-instantsearch-app](https://github.com/algolia/instantsearch/tree/master/packages/create-instantsearch-app) 库，并启动了一个 InstantSearch.js 样板文件。从这里，我可以添加 InstantSearch.js 提供的其他小部件，比如分页和页面大小选择器，它们工作得非常好。

因为我们还通过降价收集了存储库元数据，所以我们还需要定制 hits 组件来包含这些附加信息。通常元数据和库描述一样重要，所以开发人员一眼就能看出它是否是一个流行的库，谁发布了它，标签等等。我还添加了 facets，这样用户就可以通过编程语言、标签或者分叉次数进行过滤。

拼图的最后一块是添加“打开文档”按钮，允许您在不离开应用程序的情况下，在弹出窗口中快速轻松地阅读存储库的降价内容。如果我们点击的记录有多行，它会自动加载额外的记录并将它们连接在一起显示——太棒了！

## [](#conclusion)结论

这个项目是一个有趣的测试，并真正向我展示了 Algolia 对于像我们这样的不同用例是多么灵活。这些现成的小部件在原型制作过程中节省了我大量的时间，并且从最初的几次击键中获得相关的结果给我留下了非常深刻的印象。我还认为，如果我们能够利用 Algolia 推荐的力量，如果我们能够从用户内部点击项目中产生足够多的事件，那将会非常有趣。

你可以点击这里观看 GitHub 测试项目[的现场演示，点击一个按钮，你就可以使用默认的演示凭证来查看我们的索引。对后端索引代码感兴趣？你可以在 GitHub 上找到这里的](https://stackblitz.com/github/algolia-samples/algolia-js-document-indexer/tree/main/frontend?file=src%2Falgolia.js)[！](https://github.com/algolia-samples/algolia-js-document-indexer/tree/main/indexer)

[![Algolia Code Exchange](img/7665551c18b687f25dcadc15cb213b7d.png)](https://www.algolia.com/developers/code-exchange/showcase/indexing-markdown-files-and-metadata-using-node-js "Algolia Code Exchange")

* * *

*我们希望您喜欢来自 Soma 的这篇深入的文章！如果你正在寻找更多这样的内容，我们在 [Algolia 工程博客](https://www.algolia.com/blog/engineering/)上有更多的主题！如果你是 Algolia 的新手，你可以通过注册一个[免费等级账户](https://www.algolia.com/users/sign_up?utm_source=blog&utm_medium=main-blog&utm_campaign=devrel&utm_id=blog-algolia-and-rust)来尝试一下。*