# 添加自动完成搜索到您的 Strapi CMS 

> 原文：<https://www.algolia.com/blog/engineering/add-autocomplete-search-to-your-strapi-cms/>

Strapi 是一个开源的无头 CMS，它构建 api 抽象来检索您的内容，而不管它存储在哪里。它是构建内容驱动的应用程序(如博客和其他媒体网站)的一个很好的工具。

在本文中，您将使用 Algolia 的 Autocomplete.js 库和社区构建的 Strapi 搜索插件，通过 Next.js 前端向 Strapi 博客添加交互式搜索。

## [](#prerequisites)先决条件

Strapi 和 Algolia Autcomplete.js 都是使用 Javascript 构建的，所以您需要安装 node v.12。
。
您将使用 Strapi 和 Next.js 构建本指南[中的基本博客应用程序。在构建您的搜索体验之前，您应该熟悉文章中的步骤。](https://strapi.io/blog/build-a--react-js-strapi)

您还需要一个 Algolia 帐户进行搜索。您可以使用现有的 Algolia 帐户或[注册一个免费帐户](https://www.algolia.com/users/sign_up?utm_source=blog&utm_medium=main-blog&utm_campaign=devrel&utm_id=agency-finder)。

## [](#building-the-back-end)建筑后端

首先创建一个目录来保存项目的前端和后端:

```
mkdir blog-strapi-algolia && cd blog-strapi-algolia
```

Strapi 有一组预烤模板，您可以使用它们快速启动并运行 CMS。这些都包括 Strapi 本身以及预定义的内容类型和样本数据。如果您还没有 Strapi 博客，您可以使用博客模板快速创建一个。

```
npx create-strapi-app backend  --quickstart --template @strapi/template-blog@1.0.0 blog
```

脚本安装完成后，在 https://localhost:1337 上添加一个 admin 用户，这样您就可以登录到 Strapi 仪表板。这个脚本为我们设置了大部分后端，包括一些演示博客帖子。您可以在[快速入门](https://strapi.io/blog/build-a-blog-with-next-react-js-strapi)中了解更多关于设置的信息。

接下来，您需要索引您的演示内容。幸运的是，Strapi 社区为您提供了由社区成员 Mattias van den Belt 构建的搜索插件和 Algolia 索引提供程序。你可以在文档中阅读更多关于 Mattie 的插件，但是启动和运行只需要几项配置。

继续并停止您的 Strapi 服务器，以便您可以使用`npm`(或`yarn`)安装插件。

```
cd backend && npm install @mattie-bundle/strapi-plugin-search @mattie-bundle/strapi-provider-search-algolia --save
```

您需要将 Algolia API 密钥和应用 ID 添加到您的 Strapi 环境中。您可以通过导航 Algolia 仪表盘的“设置>团队和访问> api 密钥”来管理您的密钥，或者直接进入 https://algolia.com/account/api-keys.。由于 Strapi 正在修改您的 Algolia 索引，您需要提供管理 API 密钥(用于演示)或创建一个对您的生产项目具有适当访问权限的密钥。

```
# .env
// ...
ALGOLIA_PROVIDER_APPLICATION_ID=XXXXXXXXXX
ALGOLIA_PROVIDER_ADMIN_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

```

准备好凭证后，最后一步是配置插件。在您的`backend`目录下创建或修改`./config/plugins.js`文件。你想让插件为你的博客索引`article`和`category`内容类型。

```
'use strict';

module.exports = ({ env }) => ({
  // ...
  search: {
    enabled: true,
    config: {
      provider: 'algolia',
      providerOptions: {
        apiKey: env('ALGOLIA_PROVIDER_ADMIN_API_KEY'),
        applicationId: env('ALGOLIA_PROVIDER_APPLICATION_ID'),
      },
      contentTypes: [
        { name: 'api::article.article' },
        { name: 'api::category.category' },
      ],
    },
  },
});

```

重启您的 Strapi 服务器以获取这些环境变量和新插件(`npm run develop`)。当新内容是`Published`时，搜索插件触发，所以你需要或者`Unpublish`和`Publish`演示文章，或者创建一个新的。加载 Strapi 管理面板(https://localhost:1337/admin)并导航到**内容管理器>集合类型>文章**，然后或者点击现有文章并`Unpublish`它，或者点击`Create new Entry`。点击文章上的`Publish`将此条目编入您的 Algolia 应用程序。如果您也想索引那些内容类型，您可以对`category`内容类型做同样的事情。

## [](#building-the-front-end)建筑前端

既然您已经构建了后端并填充了索引，那么是时候为您的用户构建前端了。

Strapi 有一篇很棒的博文，带你构建一个 Next.js 驱动的前端。您将在这些步骤的基础上进行构建。你可以自己浏览他们的[快速入门](https://strapi.io/blog/build-a-blog-with-next-react-js-strapi)，或者如果你想直接加入搜索，你可以直接复制[这个回购](https://github.com/chuckmeyer/blog-strapi-frontend/tree/no-search-version)。

```
git clone --single-branch --branch no-search-version git@github.com:chuckmeyer/blog-strapi-frontend.git frontend
```

别忘了跑步

```
cd frontend && npm install
```

如果你克隆了回购协议的前端。

这足以让基本的博客网站开始运行。您可以通过在`frontend`目录中启动服务器来测试它。

```
npm run dev
```

唯一缺少的就是搜索。您将使用 Algolia Autocomplete.js 库为您的博客导航栏添加自动完成搜索体验。当用户在字段中键入内容时，自动完成功能会通过提供完整的术语或结果来“完成”他们的想法。自动完成库是源代码不可知的，所以您还需要 Algolia InstantSearch 库来连接到后端的索引。

```
npm install @algolia/autocomplete-js algoliasearch @algolia/autocomplete-theme-classic --save
```

要在 React 项目中使用自动完成库，首先需要创建一个自动完成组件来包装该库。您可以在[自动完成文档](https://www.algolia.com/doc/ui-libraries/autocomplete/integrations/using-react/)中找到样板文件。

`./frontend/components/autocomplete.js`

```
import { autocomplete } from '@algolia/autocomplete-js';
import React, { createElement, Fragment, useEffect, useRef } from 'react';
import { render } from 'react-dom';

export function Autocomplete(props) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) {
      return undefined;
    }

    const search = autocomplete({
      container: containerRef.current,
      renderer: { createElement, Fragment },
      render({ children }, root) {
        render(children, root);
      },
      ...props,
    });

    return () => {
      search.destroy();
    };    
  }, [props]);

  return <div ref={containerRef} />;
}

```

就像在后端一样，您需要您的 Algolia 凭证来连接到 API。由于前端只需要*从索引中读取*，您可以使用应用程序的搜索关键字。创建一个`./frontend/.env.local`文件来存储您的凭证。

```
NEXT_PUBLIC_ALGOLIA_APP_ID=XXXXXXXXXX
NEXT_PUBLIC_ALGOLIA_SEARCH_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

```

现在，您可以初始化与 Algolia 的连接，并通过更新`./frontend/components/nav.js`中的代码来添加新的自动完成组件。

```
import React from "react";
import Link from "next/link";

import { getAlgoliaResults } from '@algolia/autocomplete-js';
import algoliasearch from 'algoliasearch';
import { Autocomplete } from './autocomplete';
import SearchItem from './searchItem';
import "@algolia/autocomplete-theme-classic";

const searchClient = algoliasearch(
  process.env.NEXT_PUBLIC_ALGOLIA_APP_ID,
  process.env.NEXT_PUBLIC_ALGOLIA_SEARCH_API_KEY,
);

const Nav = ({ categories }) => {
  return (
    <div>
      <nav className="uk-navbar-container" data-uk-navbar>
        <div className="uk-navbar-left">
          <ul className="uk-navbar-nav">
            <li>
              <Link href="/">
                <a>Strapi Blog</a>
              </Link>
            </li>
          </ul>
        </div>
        <div className="uk-navbar-center">
          <Autocomplete
            openOnFocus={false}
            detachedMediaQuery=''
            placeholder="Search for articles"
            getSources={({ query }) => [
              {
                sourceId: "articles",
                getItemUrl( {item} ) { return `/article/${item.slug}`},
                getItems() {
                  return getAlgoliaResults({
                    searchClient,
                    queries: [
                      {
                        indexName: "development_api::article.article",
                        query,
                      }
                    ]
                  })
                },
                templates: {
                  item({ item, components}) {
                    return <SearchItem hit={item} components={components} />;
                  }
                }
              },
            ]}
          />
        </div>
        <div className="uk-navbar-right">
          <ul className="uk-navbar-nav">
            {categories.map((category) => {
              return (
                <li key={category.id}>
                  <Link href={`/category/${category.attributes.slug}`}>
                    <a className="uk-link-reset">{category.attributes.name}</a>
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      </nav>
    </div>
  );
};

export default Nav;

```

如您所见，您正在向`Autocomplete`组件传递几个参数:

*   `openOnFocus={false}`–告诉您的搜索在用户开始输入之前不要填充结果
*   `detachedMediaQuery=''`–以分离模式打开搜索，为您的结果提供更多空间
*   `placeholder="Search for articles"`–搜索前出现在搜索框中的文本
*   `getSources={({ query }) =>`–为您的自动完成体验定义数据源的地方

请记住，自动完成是源代码不可知的。您可以基于应用程序中的 API、库或静态内容来定义源代码。这里，您使用`autocomplete-js`库中的`getAlgoliaResults`函数将一个名为`articles`的源绑定到您的 Algolia 索引。

```
              {
                sourceId: "articles",
                getItemUrl( {item} ) { return `/article/${item.slug}`},
                getItems() {
                  return getAlgoliaResults({
                    searchClient,
                    queries: [
                      {
                        indexName: "development_api::article.article",
                        query,
                      }
                    ]
                  })
                },
                templates: {
                  item({ item, components}) {
                    return <SearchItem hit={item} components={components} />;
                  }
                }
              },

```

`development_api::article.article`是上面的 Strapi 搜索插件为您的`article`内容类型生成的索引。当您进入生产阶段时，插件将在同一个应用程序中创建一个单独的`production_api::article.article`索引。

`getItemUrl()`部分设置键盘导航，而`getItems()`使用搜索框中的查询词从索引中检索文章。

注意上面的代码引用了一个`SearchItem`组件。这是您将用来告诉 Autocomplete 如何呈现您的搜索结果的`template`。用下面的代码添加一个名为`./frontend/components/searchItem.js`的新组件。

```
import React from "react";

function SearchItem({ hit, components }) {
  return (
    <a className="aa-ItemLink" href={`/article/${hit.slug}`}>
      <div className="aa-ItemContent">
        <div className="ItemCategory">{hit.category.name}</div>
        <div className="aa-ItemContentBody">
          <div className="aa-ItemContentTitle">
            <components.Highlight hit={hit} attribute="title" />
          </div>
          <div className="aa-ItemContentDescription">
            <components.Highlight hit={hit} attribute="description" />
          </div>

        </div>
      </div>
    </a>
  );
};

export default SearchItem;

```

使用这段代码，您将显示与文章、标题和描述相关联的`category`。使用`components.Highlight`组件来强调匹配用户查询的属性部分。

就这样，你完成了！用`npm run dev`启动你的前端服务器。现在，您应该会在页面顶部看到自动完成搜索框。点击它打开模态搜索界面，你可以开始输入你的搜索词。

[![](img/7665551c18b687f25dcadc15cb213b7d.png)](https://www.algolia.com/developers/code-exchange/backend-tools/integrate-strapi-with-algolia/)

你可以在 [codesandbox](https://codesandbox.io/s/github/chuckmeyer/blog-strapi-frontend) 上看到这个前端的托管版本，尽管后端容器启动可能需要一些时间。前端代码的之前的[和](https://github.com/chuckmeyer/blog-strapi-frontend/tree/no-search-version)之后的[版本也可以在 Github 上获得。](https://github.com/chuckmeyer/blog-strapi-frontend/tree/with-search-version)

如果你用这个博客做了一些很酷的东西，请在 Twitter 上与我们分享。