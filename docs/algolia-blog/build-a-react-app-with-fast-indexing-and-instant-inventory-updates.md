# 使用快速、即时的库存更新构建 React 搜索应用程序

> 原文：<https://www.algolia.com/blog/engineering/build-a-react-app-with-fast-indexing-and-instant-inventory-updates/>

我和一个朋友最近集思广益，在 Covid 继续限制他们现场演出的时候，我们如何支持我们的当地乐队。我们想出了一个网站的主意，乐队可以在那里出售限量版海报和 t 恤。该网站旨在通过在社交媒体上推广新产品(也称为“drops ”)来创造一种社区感和紧迫感，这与时尚品牌和运动鞋界的成功做法类似。

我们选择 Algolia 作为我们的主要工具，因为它提供了前端库，可以轻松地为客户提供顶级搜索和发现体验的电子商务网站。我们决定为 React 利用 [的搜索功能和模块性。这里有一个我们想要构建的草图:](https://www.algolia.com/developers/mobile-web-instantsearch-react/)

## [](#the-challenge-of-instant-inventory-updates)挑战即时库存更新

这个项目的目标是在新产品发布时营造一种紧迫感。因此，技术挑战是向我们的用户显示最新的库存。从 Algolia 索引中搜索和检索信息的速度快如闪电，提供即时搜索结果。另一方面，在后端更新数据可能会花费更多的时间。然而，我们发现，如果我们在每次库存发生变化时都更新索引，就会降低搜索结果和一些前端组件的呈现时间。

我们将看看我们是如何使用和调整 InstantSearch 来为一个电子商务网站构建几乎是即时的搜索功能的。

## [](#using-algolia-widgets-for-instantsearch)使用 Algolia 小工具进行即时搜索

[Algolia instant search](https://www.algolia.com/doc/guides/building-search-ui/what-is-instantsearch/js/)提供了令人难以置信的快速搜索结果，它允许我们快速浏览设计，以产生所需的紧迫感。InstantSearch 中的自动完成功能增加了发现功能，因为用户可以看到他们可能没有考虑过的建议。

我想在 React 中建立网站，因为它为电子商务设计和跨平台兼容性提供了很好的选择。React 的 InstantSearch 是一个预先构建的前端 [React 小部件](https://www.algolia.com/doc/guides/building-search-ui/widgets/showcase/js/) 库，它可以让你快速开始一个站点布局。上面的组件图很好地映射了其中的一些小部件:

如果你没有 Algolia 索引，[创建一个账户和 app](https://www.algolia.com/users/sign_up?utm_source=blog&utm_medium=main-blog&utm_campaign=devrel-jamstack&utm_id=11ty-serverless) 。我们创建了一个包含前三个小部件(SearchBox、RefinementList 和 Autocomplete)的`<SearchBox>`组件，以及一个用于 Hits 小部件的`<Hits>`组件，显示我们的产品列表。`<ProductDetail>`组件使用我们自己的数据和代码，因为它不是来自 Algolia 索引。

然而，我们在这个设计上有一个问题，那就是它要求我们不断地更新和检索 Algolia 索引中的数据。正如我们上面提到的，Algolia 索引针对*检索进行了优化。他们的引擎优先搜索，快速索引紧随其后。我们不想让服务器充斥着索引更新，这可能会降低搜索速度、`<Hits>`的渲染速度，或者限制`<SearchBox>`的功能，所有这些都依赖于每次按键的超快速数据检索。我们想出了一个解决方案——将我们的库存数据更新从我们的搜索处理中分离出来，这样我们就可以在不触及我们的主索引的情况下更新库存。*

我们回到我们的网站模型，发现我们可以通过定制即时搜索小部件来解决这个问题

## [](#create-a-second-algolia-index-for-inventory)为库存创建第二个 Algolia 指标

让我们同时拥有 as-you-type 即时搜索结果 *和* 即时库存的解决方案是拥有两个独立的 Algolia 索引。 如果你还没有 Algolia Index，[创建一个账户和 app](https://www.algolia.com/users/sign_up?utm_source=blog&utm_medium=main-blog&utm_campaign=devrel-jamstack&utm_id=11ty-serverless) 。

其中一个指数是典型的产品指数。JSON 看起来是这样的:

```
[
 {
  "objectID": 42,
  "title": "Return to Paranormal",
  "band": “Out In The Cold”,
  "designer": “Jamie Deer”,
  "introduction_date": “2021-09-23T18:25:43.511Z”,
  "sizes": [“adults”, “kids”, “pets”],
  "_tags": ["Seattle", "aliens", “punk”]
 }
]

```

这只是简单的记录。最终，我们将在索引中包含更多信息，如乐队和设计师网站的链接、图片 URL 等。

第二个指标更简单:

```
[
 {
   “objectID”: 42,
   “remaining_editions”: 18
 }
]

```

现在我们可以更新较小的库存指数，而不会影响用于`<Hits>`的主要指数。

使用两个索引也需要为每个点击结果将 UI 分成两个独立的组件，但这是一个简单的调整:

我们将`remaining_editions`数字从`<Hits>`组件中移出，放入它自己单独的 [自定义小部件](https://www.algolia.com/doc/guides/building-search-ui/widgets/create-your-own-widgets/js/) ，我们将其渲染为一个单独的组件，称为`<Inventory>`。两个索引和两个组件使用相同的`objectID`。

需要注意的是，我们没有让`<Inventory>`成为`<Hits>`的子元素，因为我们需要将这两个组件的渲染分开。这就是为什么`<SearchPage>`组件持有`<SearchBox>`、`<Hits>`、`<Inventory>`、`<ProductDetail>`处于同一级别的原因。

## [](#using-react-hooks-to-keep-everything-available)使用 React 钩子保持一切可用

您可能已经注意到，我们依赖三个数据来源:

1.  **Algolia 索引** (在 Algolia 的服务器上)–该索引提供填充`<Hits>`和`<SearchBox>`组件的数据。这里的数据摘自我们的后台数据库(3)，当产品列表发生变化时，该数据库会定期更新。
2.  **Algolia 库存索引** (在 Algolia 的服务器上)——该索引提供了显示在`<Inventory>`组件中的更新后的`remaining_editions`属性。`remaining_editions`属性来自我们的后台数据库(3)。
3.  **我们的后台数据库** (在我们的服务器上)–我们维护着一个详细的产品后台数据库，其中包括更长的产品描述和购买数据。我们从这个数据库中提取库存变化，并异步更新`<ProductDetail>`组件，除了`remaining_editions`、显示给 `<Inventory>`组件。

我们在这个应用程序中有两个目标:第一，提供“即时”库存数字，第二，保持随时输入即时搜索结果。以下是我们实现这两个目标的方法。

### [](#instantsearch-never-slows-down)即时搜索从不减速

保持即时搜索完全可用是没有商量余地的。客户可以理解库存中几秒钟的延迟，但他们不会容忍缓慢或故障的搜索功能。我们是这样构建这部分的:

*   我们使用 InstantSearch 的一个非常标准的实现，其中一个变化是需要我们为`Hits`连接一个 [定制组件](https://www.algolia.com/doc/api-reference/widgets/hits/react/#connector) 。
*   在我们的`<SearchPage>`组件中，我们使用了 React 状态挂钩，并为传递给`<Hits>`组件的`setObjectIDs`创建了一个 setter 方法。
*   当点击渲染时，我们用一个处理程序方法捕获数组中的`objectIDs`，然后将它传递回`<SearchPage>`。
*   在管理 *异步* API 调用方面，这是我们的第一要务，所以唯一的 Hits widget 是 *等待* 的是 InstantSearch API。我们可以用`<Hits>`中的`useEffect()`钩子来完成这个任务，但是 Algolia [在 InstantSearch 中提供了一个方法](https://www.algolia.com/doc/guides/building-search-ui/going-further/routing-urls/react/#onsearchstatechangenextsearchstate) 来实现同样的目标:`onSearchStateChange`。

这些步骤从`<Hits>`中获取我们需要的信息，而不会中断即时搜索功能的 UI 或数据流——我们只需在中间插入我们连接的组件，并使用内置方法来触发更新。

**代码为`SearchBox.js` :**

```
import { InstantSearch } from 'react-instantsearch-dom';
import SearchBox from './SearchBox.js';
import CustomHits from './Hits.js';
import Inventory from './Inventory.js';
import ProductDetail from './ProductDetail.js';

const searchClient = algoliasearch('YourApplicationID', 'YourSearchOnlyAPIKey');

function SearchPage() {
 const [objectIDs, setObjectIDs] = useState([]);
 const [selectedObject, setSelectedObject] = useState({
   objectID: null,
 });

 return (
   <React.Fragment>
     <InstantSearch
       indexName="product_index"
       searchClient={searchClient}
       searchState={{
         query: '',
       }}
       onSearchStateChange={searchState => {
         if (searchState) {
           setObjectIDs();
         }
       }}
     >
       <SearchBox />
       <CustomHits
         objectIDs={objectIDs}
         setObjectIDs={setObjectIDs}
         selectedObject={selectedObject}
         setSelectedObject={setSelectedObject}
       />
       <Inventory objectIDs={objectIDs} />
       <ProductDetail selectedObject={selectedObject} />
     </InstantSearch>
   </React.Fragment>
 );
}

export default SearchPage;

```

**代码为`Hits.js` `:**

```
import React from 'react';
import PropTypes from 'prop-types';
import { connectHits } from 'react-instantsearch-dom';

function Hits({ hits, objectIDs, setObjectIDs }) {
 const handleSearch = () => {
   hits.map(hit => objectIDs.push(hit.objectID));
   setObjectIDs(hits);
 };
 return (
   <ol>
     {handleSearch}
     {hits.map(hit => (
       <li key={hit.objectID}>{hit.name}</li>
     ))}
   </ol>
 );
}

Hits.propTypes = {
 hits: PropTypes.arrayOf(PropTypes.object),
 objectIDs: PropTypes.arrayOf(PropTypes.string),
 setObjectIDs: PropTypes.func,
};

const CustomHits = connectHits(Hits);
export default CustomHits;

```

### [](#%e2%80%9cinstant%e2%80%9d-inventory-updates-are-close-to-real-time)【即时】库存更新接近实时

既然我们已经设置好捕获`objectIDs`，我们需要设置我们的库存更新。我们将在三种不同的情况下调用我们的库存指数:两个用于`<Inventory>`组件，一个用于`<ProductDetail>`组件。

为了显示最新的库存信息，我们需要两个数据点:`<Hits>`中当前结果的数组`objectIDs`,以及库存指数中的最新数字。`objectIDs`现在保存在`<SearchPage>`中，每次`searchState`改变时都会更新，所以将它们传递给`<Inventory>`很容易。为了从我们的库存索引中获取信息，我们使用另一个 [内置的 Algolia 方法](https://www.algolia.com/doc/api-reference/api-methods/get-objects/) : `index.getObjects()`。这让我们可以直接找到我们想要的对象，而不必创建另一个 InstantSearch 实例。

注意，从`algoliasearch@4.0.0`开始，`getObject()`和`getObjects()`方法在`lite`构建中不可用，这是默认的导入——您只需要更新 import 语句。

一旦我们建立了初始连接，建立对库存索引的第二次调用就很简单了。我们在一个`useEffect()`钩子中添加了一个`setInterval()`函数，每两秒钟调用一次库存指数。这样，即使用户不更新他们的搜索词，库存数量也会继续快速刷新，确保实时库存。我们可能会调整生产中的时间间隔，但这对我们的目的来说已经足够快了。

**代码为`Inventory.js` :**

```
import React, { useEffect } from 'react';
import PropTypes from 'prop-types';
import algoliasearch from 'algoliasearch';

const searchClient = algoliasearch('YourApplicationID', 'YourSearchOnlyAPIKey');

function Inventory({ objectIDs }) {
 const index = searchClient.initIndex('inventory_index');
 let results = index.getObjects(objectIDs);
 useEffect(() => {
   const timer = setInterval(() => {
     results = index.getObjects(objectIDs);
   }, 2000);
   return () => clearTimeout(timer);
 }, []);
 return (
   <ol>
     {results.map(result => (
       <li key={result.objectID}>Remaining: {result.remaining_editions}</li>
     ))}
   </ol>
 );
}

Inventory.propTypes = {
 objectIDs: PropTypes.arrayOf(PropTypes.string),
};

export default Inventory;

```

组件内部的代码也是如此。该组件将从我们的产品数据库(非 Algolia 数据源)获取大部分数据。对于库存编号，它会像上面一样调用`inventory_index`，只使用`selectedObject`的`objectID`。采用这种设计，在`<Hits>`中产品结果旁边显示的库存编号和`<ProductDetail>`组件中的库存编号之间应该只有非常小的延迟，不超过两秒钟。这样，`<ProductDetail>`将总是显示更新的数据。

现在我们有了一个`<SearchPage>`，显示近乎即时的库存更新，同时仍然允许充分利用即时搜索结果。

我们设想这个项目是独立艺术家和音乐家将他们的社交媒体与电子商务平台联系起来的一种方式。作为音乐迷，在过去的几年里，我们已经错过了当地音乐界的社区意识和活力。我们的希望是，这个网站设计将感觉真实和令人兴奋，足以捕捉一些能量。

Algolia 的 [快速实现的小部件](https://www.algolia.com/doc/guides/building-search-ui/widgets/showcase/js/) 和适应性强的组件使其易于构建原型。你可能有其他“即时库存”有用的想法——也许是黑色星期五销售！我们希望了解您用 Algolia 构建了什么，以及它如何改变您的项目！