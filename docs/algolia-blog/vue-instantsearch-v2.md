# 让你的生活更加丰富多彩的小贴士😎

> 原文：<https://www.algolia.com/blog/engineering/vue-instantsearch-v2/>

这篇文章是与 [Bram Adams](https://github.com/bramses) 合作完成的。

大新闻！我们自豪地宣布我们的第二个主要版本 [Vue InstantSearch](https://www.algolia.com/doc/guides/building-search-ui/what-is-instantsearch/vue/) 。我们增加了许多有益的优化和新功能。我们长话短说，专注于我们最喜欢的。让我们开始吧。

## [](#new-vue-widgets)新增 Vue 小工具！

我喜欢玩新玩具，你呢？Vue InstantSearch 2 附带了一系列新组件，将使您的搜索变得非常棒。这里有一些你可能会感兴趣的。

### [](#wrap-your-widgets-in-a-container)把你的小工具包在一个容器里

`ais-panel`是一个容器小部件，可以用来为你的站点提供一致的外观和感觉。您可以使用这个小部件在小部件周围一致地显示页眉和页脚。因为它将有一个`ais-Panel--noRefinement`类，所以如果小部件不能使用，以不同的方式隐藏或显示它也变得更加一致。

### [](#conditionally-display-results)有条件地显示结果

你有没有想过，我希望我可以只在某些时候展示这些结果？现在你可以了！使用`ais-state-results`，你可以选择向你的用户显示什么。它可以用来显示一个特定的“没有结果”页面，或者当有人细化一个特定的案例时显示一个特定的横幅。

### [](#showing-breadcrumbs)显示面包屑

我把那只鞋放在哪里了？…哦，对了！使用`ais-breadcrumb`，你可以让用户发现路径，并给他们另一种方式来浏览你的产品。我一直在[概念](https://www.notion.so)中使用我的面包屑！

### [](#performing-inline-configuration)执行内嵌配置

配置的力量。住在 Vue 的舒适！使用`ais-configure`，你可以向 Algolia 提供原始的搜索参数。这将允许您在前端做一些强大的事情，比如限制每个查询返回的命中数，或者只返回不同的结果。

### 显示下拉菜单

一个强大的小部件，`ais-menu-select`给你一个下拉菜单，显示用户可以与之交互的菜单项。这个小部件是高度可定制的，允许你预先渲染结果，让你有机会在结果显示给用户之前对它们进行操作！微小的下拉菜单，巨大的潜力！

### [](#view-your-applied-refinements)查看您应用的细化

`ais-current-refinements`小部件显示用户选择的每个当前细化的药丸。作为开发人员，您可以决定允许在这个小部件中显示哪些属性子集。看到这里的趋势了吗？可定制性已经存在！

### [](#showing-more-results)显示更多结果

这个`ais-infinite-hits`小部件的名字非常贴切！它给你的组件添加一个“显示更多结果”按钮，当按下时，将显示更多的项目。继续滚动，继续滚动…

### 创建数字菜单

这个`ais-numeric-menu`小部件是一个可预先配置的小部件，它创建一个数字对象的单选按钮菜单，可以用作过滤器。要使用这个小部件，只需创建一个包含不同数值的对象列表，如下所示。数学！

```
[
  { label: 'All' },
  { label: '<= 10$', end: 10 },
  { label: '10$ - 100$', start: 10, end: 100 },
  { label: '100$ - 500$', start: 100, end: 500 },
  { label: '>= 500$', start: 500 },
]
```

### [](#toggling-a-refinement)切换一个细化

上蜡，下蜡。也就是说，至少，如果“蜡”是一个 Algolia 属性！您可以使用`ais-toggle-refinement`来显示一个可切换的组件。和以前一样，你也可以用`slot-scope`编辑 CSS 类来实现深度定制！

## [](#new-widget-names)新增 Widget…名称！

名称又能代表什么呢我们已经更新了小部件的名称，使其更接近我们的即时搜索风格。完整的列表可以在[这里](https://www.algolia.com/doc/guides/building-search-ui/upgrade-guides/vue/#renamed-components)找到，但主要的想法是，你过去使用 Algolia InstantSearch 的任何经验都应该无缝地滚动。

## [](#server-side-rendering-ssr)服务器端渲染(SSR)

我们彻底改进了服务器端渲染与 Vue InstantSearch 的工作方式。与前一版本相比，服务器端渲染的功能没有太大变化，但主要区别在于它是如何在幕后实现的。不是等到所有请求都完成(不一定总是正确的)，而是使用静态方法来完成后端请求。阅读文档中关于如何实施 SSR [的更多信息。](https://algolia.com/doc/guides/building-search-ui/going-further/server-side-rendering/vue/)

## [](#url-synchronization)URL 同步

以前，您必须手动读取和翻译“搜索存储”中的每个参数，并使其与 URL 保持同步。这不是一个简单的过程，所以我们用一个专用的 API 来简化它。我们现在映射这些参数，将它们转换成 URL 友好的格式，并提供一种根据需要修改 URL 的方法。在文档中阅读更多关于那个[的内容。](https://www.algolia.com/doc/guides/building-search-ui/going-further/routing-urls/vue/)

## [](#customization-of-widgets)定制的小工具

让我们以一个我们自豪地宣布的特性来结束。我们的小部件现在是完全可扩展和可定制的。在不久的将来，我们会就这个主题发表第二篇博客，但是我们可以给你一些细节。为了定制一个小部件，现在有四个级别来尽可能平滑地定制:

### [](#simple-customization)简单定制

定制小部件最简单的方法是改变简单的文本和图标。为此，您将使用常规的“插槽”。一个例子是`ais-search-box`上的复位图标。

### [](#customizing-the-complete-html)定制完整 HTML

如果您想进一步定制和覆盖组件的整个 HTML 输出，您也可以通过“作用域槽”来实现，每个小部件的根上都有这个槽。一个例子就是使用你自己的组件来代替搜索框。您将获得对当前查询的访问，以及将查询更改为新查询的功能(`refine`)。

### [](#using-the-widget-mixin)使用 widget mixin

如果您发现还需要访问提供给这个小部件的数据的生命周期方法，您可以添加小部件 mixin 并使用 InstantSearch 中提供的连接器。

### [](#matrix-level-control)矩阵级控制

最后，您还可以进入“超矩阵模式”(我喜欢这样称呼它)，并制作自己的连接器来封装直接修改搜索状态的逻辑。

触手可及的完全控制！查看我们的教程[如何定制一个小部件](https://www.algolia.com/doc/guides/building-search-ui/widgets/create-your-own-widgets/vue/)。

## [](#to-conclude)得出结论

从添加小工具到允许完全定制它们的外观、感觉和行为，我们一直在努力确保 Vue InstantSearch 为您提供 Algolia 强大搜索技术的最佳功能！试试 Vue InstantSearch v2 现在[这里](https://codesandbox.io/s/github/algolia/vue-instantsearch/tree/v2/examples/default-theme)，或者阅读[文档](https://www.algolia.com/doc/guides/building-search-ui/what-is-instantsearch/vue/)。