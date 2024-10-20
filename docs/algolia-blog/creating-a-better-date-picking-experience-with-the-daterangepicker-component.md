# 使用 DateRangePicker 组件获得更好的日期选择体验

> 原文：<https://www.algolia.com/blog/engineering/creating-a-better-date-picking-experience-with-the-daterangepicker-component/>

在设计搜索界面时，使用 Algolia 的即时搜索库就足够了。为了便于使用，每个库都有不同风格的 JavaScript。每一个都有制作一个强大的 UI 所需的所有主要部分。

有时候你想超越现有的东西。

在这一点上，您有两个选择:用我们的连接器 API 制作您自己的或者找到一个预构建的定制小部件。

任何认识我的人都不会感到惊讶，当面对想要一个 React 日期选择器时，我希望尽可能少地编写代码来使它工作，但我也希望为圣诞老人提供最好的 UI。当第五次奖金挑战来临时，我需要一些额外的东西。我的第一站？[阿尔戈利亚电码交换](https://www.algolia.com/developers/code-exchange/)！

在对`date`组件进行快速搜索后，我找到了我所需要的:名副其实的[@ algolia/react-instant search-widget-date-range-picker](https://github.com/algolia/react-instantsearch-widget-date-range-picker)！如果你没有使用 React，还有一个 [VanillaJS 版本](https://github.com/algolia/instantsearch-widget-date-range-picker)。

## [](#setting-up-the-project)设置项目

如果您想继续，您需要安装一个新的 React InstantSearch 项目。最简单的方法是运行以下命令:

```
 npx create-instantsearch-app concert-search \
 --app-id latency \
 --api-key 059c79ddd276568e990286944276464a \
 --index-name concert_events_instantsearchjs \
 --template "React InstantSearch"
```

这个怪物命令将建立一个 React InstantSearch 项目，并将其连接到 [Algolia 的 hosted concert 索引](https://github.com/algolia/datasets/tree/master/concerts)。将目录切换到创建的文件夹，运行`yarn start`作为坚实的起点。

至此，您已经获得了一组音乐会的完整搜索体验。

如果您想按日期过滤，您可以设置一个`<RefinementList>`组件，并将其`attribute`属性设置为`date`，但是这些日期是 Unix 时间戳(为了便于比较)。这对用户体验来说并不理想。让我们做得更好。

## [](#installing-and-configuring-the-widget)安装和配置小工具

要启动并运行日期范围选择器，我们需要安装几个依赖项。

```
npm install @algolia/react-instantsearch-widget-date-range-picker @duetds/date-picker

```

这将安装官方的 Algolia React 日期选择器小部件及其依赖项 Duet 日期选择器。

从这里，我们需要将包导入到我们的`src/App.js`文件中，并初始化 Duet 日期选择器以供使用。

//在 create-instantsearch-app 代码中包含的导入之后添加

```
import { DateRangePicker } from '@algolia/react-instantsearch-widget-date-range-picker';  
import { defineCustomElements } from "@duetds/date-picker/dist/loader";

// Defines the custom elements from the date picker for use on the window object  
defineCustomElements(window);
```

既然这些包已经导入，我们就可以在页面上看到它们了。

## [](#adding-the-date-range-picker)添加日期范围选择器

要将选择器添加到页面，我们需要在`<InstantSearch>`组件中选择一个点。该应用程序的基础是一个`search-panel`。默认情况下，这里只有结果，但是我们也可以添加一个过滤器面板。

```
<InstantSearch searchClient={searchClient} indexName="concert_events_instantsearchjs">  
  <div className="search-panel">
    <div className="search-panel__filters"> 
      <DateRangePicker attribute="date" />
    </div>

    <div className="search-panel__results">

      //... Results code

    </div>
  </div>
</InstantSearch>
```

DateRangePicker 接受一个`attribute`属性。该属性接受索引中点击次数的基于日期的属性。

*快速提示*:DateRangePicker 接受一个 Unix 时间戳，从 Epoch 开始以毫秒为单位，而不是以秒为单位。根据您的数据结构，您可能需要在数据中创建一个辅助时间戳。

当您查看呈现的页面时，您现在应该有一个日期选择器。有一些小的 UI 障碍需要清理。

Duet 日期选择器使用了大量 CSS 自定义属性来设计样式。在我们的`src/App.css`文件的开始，我们需要粘贴这些文件并为我们的应用程序进行配置(如果合适的话)。

```
:root {  
  --duet-color-primary: #3c4ee0;  
  --duet-color-text: #333;  
  --duet-color-text-active: #fff;  
  --duet-color-placeholder: #666;  
  --duet-color-button: #f5f5f5;  
  --duet-color-surface: #fff;  
  --duet-color-overlay: rgba(0, 0, 0, 0.8);  
  --duet-color-border: #d6d6e7;  

  --duet-font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica,  
  Arial, sans-serif;  
  --duet-font-normal: 400;  
  --duet-font-bold: 600;  

  --duet-radius: 3px;  
  --duet-z-index: 600;  
}
```

这两个选择器区域仍然靠得有点近，所以让我们也用一点 CSS 来解决这个问题。这可以放在`src/App.css`里，但应该更靠近底部。有许多方法可以获得两个项目之间的空间，快速简单的解决方案是使用 CSS Grid 和`gap`属性。

```
.date-range-picker {  
  display: grid;  
  gap: 1rem;  
}
```

这样，我们就有了一个有效的、用户友好的日期选择器。不再需要处理 Unix 时间戳和转换。

## [](#take-this-further)借此进一步

这在小样本数据集中非常有效，但也可以用于任何 Algolia 即时搜索应用程序。如果你想进一步了解这个例子，可以通过编辑`src/App.js`中的 Hit 组件来创建一个更好的 UI。您还可以显示使用 [CurrentRefinements 组件](https://www.algolia.com/doc/api-reference/widgets/current-refinements/react/)应用的当前过滤器。