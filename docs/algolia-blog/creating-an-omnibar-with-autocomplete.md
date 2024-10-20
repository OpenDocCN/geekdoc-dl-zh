# 创建一个带有自动完成功能的 omnibar

> 原文：<https://www.algolia.com/blog/engineering/creating-an-omnibar-with-autocomplete/>

什么时候搜索栏不是搜索栏？当它是一个用 Autocomplete 构建的“omnibar”时！

[https://www.youtube.com/embed/ghy7a78JGCQ](https://www.youtube.com/embed/ghy7a78JGCQ)

视频

在《与杰森一起学习》的[一集中](https://www.learnwithjason.dev/javascript-autocomplete)，[莎拉·达扬](https://twitter.com/frontstuff_io)提到了使用自动完成来创造一种充满快捷方式和超级用户启示的体验的想法。

在本教程中，我们将通过设置[自动完成](https://www.algolia.com/doc/ui-libraries/autocomplete/api-reference/autocomplete-js/autocomplete/)来触发与 JavaScript 的交互。具体来说，我们将构建一个 omnibar 来切换网站的明暗模式。omnibar 是一个搜索字段，既有搜索功能，也有可以执行的操作。Chrome 或 Firefox 的搜索和 URL 栏就是一个很好的例子。

在搜索栏中，用户可以输入`/`命令。这些命令将被绑定到特定的 JavaScript 方法来触发。我们还将使自动完成结果有状态。当应用程序处于灯光模式时，灯光模式选项将显示“已启用”标志。启用黑暗模式时，黑暗模式选项将显示标志。

[自己试一试！](https://codesandbox.io/s/autocomplete-actions-finished-eh7xw)

## [](#configuring-autocomplete-for-use-with-react)配置与 React 一起使用的自动完成功能

在其核心，自动完成是一个普通的 JavaScript 库。让我们通过将它作为 React 组件安装在任何基于 React 的框架或站点中来使它更具可重用性。

我们将从 CodeSandbox 的基本 React 沙箱开始。[分叉这个沙箱](https://alg.li/omnibar-starter)以获得所有为我们安装的包的准确起点。

为了创建我们的组件，我们将从添加一个名为`Autocomplete.js`的新文件开始。这个文件将包含自动完成库的所有初始化代码，并导出组件以供我们的应用程序使用。

在新文件的顶部，从 React、React-dom 和 Autocomplete 库中导入必要的元素。

```
import React, { createElement, Fragment, useEffect, useRef } from "react";  
import { render } from "react-dom";  
import { autocomplete } from "@algolia/autocomplete-js";
```

一旦导入，我们需要导出一个新的功能性 React 组件。我们将从创建新的挂载组件的基本样板文件开始。

```
export function Autocomplete(props) {  
  const containerRef = useRef(null);  

  useEffect(() => {  
    if (!containerRef.current) {  
      return undefined;  
    }

    // Space to initialize autocomplete on the newly created container

    // Destroy the search instance in cleanup  
    return () => {  
      search.destroy();  
    };  

  }, [props]);

  return /<div ref={containerRef} //>;  
}
```

这段代码将负责组件在安装和卸载时的基本初始化和分解。

在函数内部，是时候初始化 Autocomplete 实例了。

```
// Creates an Autcomplete component from the JS library
// https://www.algolia.com/doc/ui-libraries/autocomplete/guides/using-react/
export function Autocomplete(props) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) {
      return undefined;
    }

    // Initialize autocomplete on the newly created container
    const search = autocomplete({
      container: containerRef.current,
      renderer: { createElement, Fragment },
      // Autocomplete render()
      // https://www.algolia.com/doc/ui-libraries/autocomplete/api-reference/autocomplete-js/autocomplete/#param-render
      render({ children }, root) {
        // react-dom render
        // https://reactjs.org/docs/react-dom.html#render
        render(children, root);
      },
      ...props
    });

    // Destroy the search instance in cleanup
    return () => {
      search.destroy();
    };
  }, [props]);

  return <div ref={containerRef} />;
}
```

`autocomplete`方法接受一个选项对象。我们将`container`属性设置为该函数创建的元素。通过指定`renderer`函数，我们可以使用 React 的`createElement`方法和`Fragment`组件。

然后，我们需要为 Autocomplete 提供一个`render`函数。该函数将接受一个要呈现的组件对象(`children`)，以及附加到实例的元素(`root`)。

然后，我们可以使用任何方法来呈现这些项目。在我们的例子中，我们将使用`react-dom`的`render()`方法，并向它传递那些相同的元素。最后，我们想通过`autocomplete`方法传递我们使用组件时添加到组件中的任何附加道具。这将允许即时定制。

## [](#using-the-autocomplete-component)使用`<Autocomplete />`组件

转到`App.js`文件，我们可以导入我们的自动完成组件(以及一些默认样式)。

```
// Styles
import "./styles.css";  
import "@algolia/autocomplete-theme-classic";  

// Import algolia and autocomplete needs
import { Autocomplete } from "./Autocomplete";
```

现在，我们准备在页面上放置一个自动完成字段。在`App()`函数的 JSX 返回值中，我们可以将`<Autocomplete />`组件放在对 UI 有意义的任何地方。我建议放在网页正文之后。

```
export default function App() {  
    return (  
      <div className="App">  
           <h1 className="text-xl">  
             Run JS from{" "}  
             <a href="https://www.algolia.com/doc/ui-libraries/autocomplete/api-reference/autocomplete-js/autocomplete/">  
               Autocomplete  
             </a>  
           </h1>  
           <p className="text-base">  
             This demo is based on the amazing idea of{" "}  
             <a href="https://twitter.com/frontstuff_io">Sarah Dayan</a> in her  
             appearance on{" "}  
             <a href="https://www.learnwithjason.dev/javascript-autocomplete">  
               Learn with Jason  
             </a>  
             .  
           </p>  
           <p>  
             Use the Autocomplete box below to toggle dark mode and perform other  
             JS-driven actions on the page.  
           </p>  

            <Autocomplete />

      {/* ... the rest of the function ... */}
      </div>
    )
  }
```

Autocomplete 组件可以接受任何属性，只要这个属性是`autocomplete-js`库可以接受的选项。首先，让我们添加占位符文本。

```
<Autocomplete placeholder="Try /dark" />
```

一个搜索字段应该出现在我们的应用程序与占位符文本集。这个字段还没有做任何事情。让我们添加一些数据来完成。

## [](#adding-an-actions-source-to-the-autocomplete-component)向自动完成组件添加一个`actions`源

自动完成库能够针对多个源创建自动完成功能。在我们的例子中，我们只有一个静态源，但是任何外部数据——包括 Algolia 指数——都可以用来填充这个功能。

要添加一个源，我们将使用`getSources`属性并提供一个接受`query`选项的函数。该查询是用户主动输入到输入中的内容。我们可以用它来检查数据中的项目。

源是 getSources 返回数组中的对象。我们需要的源的基本元素是一个`sourceId`字符串、一个用于呈现的`template`对象和一个返回数据的`getItems()`函数。现在，我们只返回一个带有标签属性的静态数组。这足以填充我们的自动完成功能。让我们也添加`openOnFocus`作为道具，当用户聚焦字段时自动列出我们的项目。

```
<Autocomplete  
  placeholder="Try /dark"
  openOnFocus   
  getSources={({ query }) => [  
    {  
      sourceId: "actions",  
      templates: {  
        item({ item }) {  
          return <h3>{item.label}</h3>  
        }  
      },  
      getItems({ state }) {  
        return [  
          {  
            label: "/dark"  
          },  
          {  
            label: "/light"  
          }  
        ]  
      }  
    }  
  ]}  
/>
```

现在，我们有项目填充我们的领域，但我们不过滤项目，因为我们键入。让我们用几个辅助函数来解决这个问题。

## [](#filtering-and-highlighting-autocomplete-items)过滤并突出显示自动完成的项目

当使用 Algolia 索引时，我们可以使用一些辅助函数来管理过滤和突出显示，但我们没有使用 Algolia 索引。在我们的用例中，我们希望将它完全保留在浏览器中。为此，我们需要几个助手函数来适当地过滤和突出显示我们的选项。

### [](#filtering-autocomplete-items-with-javascript-regexp)用 JavaScript RegExp()过滤自动完成项

JavaScript 提供了基于正则表达式测试过滤数组的能力。要做到这一点，我们需要创建一个模式来测试用户可能向我们抛出的任何组合。让我们基于查询创建一个助手函数，并在 JS `.filter()`方法中使用它。

在`App.js`导出之外，我们将创建新的助手函数`getQueryPattern()`。

```
function getQueryPattern(query, flags = \"i\") {  
  const pattern = new RegExp(  
    `(${query  
      .trim() // Trim leading and ending whitespace 
      .toLowerCase() // convert to lower case
      .split(" ") // Split on spaces for multiple commands 
      .map((token) => `^${token}`) // Map over the resulting array and create Regex_  
      .join("|")})`, // Join those expressions with an OR | 
    flags  
  );

  return pattern;  
}

export default function App() { /* ... */ }
```

一旦创建了 helper 函数，我们将在返回项目数组之前在`getItems()`方法中创建模式。

有了保存的模式，我们可以用它来测试我们的数组。

```
<Autocomplete
  placeholder="Try /dark"
  openOnFocus
  getSources={({ query }) => [
    {
      sourceId: "actions",
      templates: {
        item({ item }) {
          return <h3>{item.label}</h3>
        }
      },
      getItems({ state }) {
        const pattern = getQueryPattern(query);

        return [
          {
            label: "/dark"
          },
          {
            label: "/light"
          }
        ].filter(({ label }) => pattern.test(label)) // tests the label against the pattern
      }
    }
  ]}
/>
```

现在，当我们在字段中键入`/dark`时，只有`/dark`选项。我们没有给用户任何提示，说明为什么会这样。让我们添加一个小的突出显示功能来展示输入的字母。

### [](#highlighting-the-string-being-typed-in-results)高亮显示结果中正在键入的字符串

为了突出显示键入的文本，我们需要获取查询文本和我们在上一步中创建的模式，并生成一个新的字符串，在键入的文本周围添加附加的降价。

在`getQueryPattern`助手函数之后，让我们创建一个新的`highlight`助手函数。

```
function highlight(text, pattern) {

    // Split the text based on the pattern  
    const tokens = text.split(pattern);

    // Map over the split text and test against the pattern  
    return tokens.map((token) => {

      // If the pattern matches the text, wrap the text in <mark>  
      if (!pattern.test("") && pattern.test(token)) {
        return <mark>{token}</mark>;
      }

      // return the token back to the array  
      return token;
    });
  }
```

这个 helper 函数接受要测试的文本和要检查的模式，并返回一个带有附加标记的字符串。

我们首先根据模式分割文本。这将给我们一个由两部分组成的数组——匹配的和不匹配的。当我们映射这个新数组时，我们可以对照模式检查文本，如果匹配，就用一段新的标记包装这个特定的项。如果没有，返回未修改的文本。

```
<Autocomplete
  placeholder="Try /dark"
  openOnFocus
  getSources={({ query }) => [
    {
      sourceId: "actions",

      templates: {
        item({ item }) {
          return <h3>{item.highlighted}</h3>
        }
      },

      getItems({ state }) {
        const pattern = getQueryPattern(query);

        return [
          {
            label: "/dark"
          },
          {
            label: "/light"
          }
        ]
        .filter(({ label }) => pattern.test(label)) // tests the label against the pattern
        .map((action) => ({
          ...action,
          highlighted: highlight(action.label, pattern)
        }));
      }
    }
  ]
  }
/>
```

有了这个助手函数，我们现在可以映射所有被过滤的条目。我们将采取行动项目，并返回一个对象及其所有的初始属性，但一个新的`highlighted`属性，其中包含我们突出显示的文本。这是根据动作的`label`属性和我们之前定义的模式构建的。

现在，我们将使用新的`highlight`属性，而不是在模板中使用`action.label`。当`/dark`被输入到字段中时，该项目将有正确的高亮文本。

过滤 UI 已经完成，但是当我们选择一个项目时，什么也没有发生。让我们解决这个问题。

## [](#firing-a-javascript-function-in-autocomplete-with-onselect)用`onSelect`在自动完成中触发一个 JavaScript 函数

在`getSources`数组中的每个源可以有自己的`onSelect`方法。这个方法定义了当用户通过键盘或点击选择一个选项时的功能。

让我们首先创建一个全局选择函数来记录项目的数据，然后将查询重置为空字符串。

```
getSources = {({ query }) => [
    {
      sourceId: "actions",
      templates: {
        item({ item }) {
          return <h3>{item.highlighted}</h3>
        }
      },
      // Run this code when item is selected  
     onSelect(params) {
        // item is the full item data
        // setQuery is a hook to set the query state
        const { item, setQuery } = params;
        console.log(item)
        setQuery("");
      },
    }
```

对于一个动作，我们可以在这个方法中定义 JavaScript，但是为了使这个方法在将来可以被任何动作重用，让我们在项目的数据上定义这个方法。

为此，我们将为每个项目定义一个名为`onSelect`的方法。这个方法可以处理您需要的任何功能。在这种情况下，我们将创建一个非常简单的黑暗和光明模式，方法是将类`dark`添加到主体以启用黑暗模式，并移除它以启用光明模式。

```
{
  label: "/light",
  onSelect() {
    document.querySelector("body").classList.remove("dark");
    notify("Light Mode enabled");
  }
},
{
  label: "/dark",
  onSelect() {
    document.querySelector("body").classList.add("dark");
    notify("Dark Mode enabled");
  }
},
```

现在，回到主`onSelect`方法，我们可以运行`item.onSelect()`，而不是运行`console.log(item)`。这将启动我们刚刚创建的函数。

我们现在有功能性的行动！

## [](#enhancing-the-omnibar-experience)增强全方位体验

有了工作动作，我们可以专注于为 omnibar 打造强大的用户体验。

### [](#automatic-highlight-and-select)自动高亮并选择

首先，让 Autocomplete 自动突出显示列表中的第一项。这将允许用户只需按回车键就可以选择一个动作。

为了添加这个特性，我们需要向`<Autocomplete />`组件传递一个新的道具。通过给属性`defaultActiveItemId`传递一个值`"0"`，我们可以激活列表中的第一项。任何激活的项目都可以通过按回车键来选择。这带来了坚实的键盘体验。

### [](#creating-a-more-robust-ui-with-a-new-component)使用新组件创建更健壮的 UI

让我们抽象出`template`来使用一个叫做`Action`的独立组件。我们可以在一个单独的文件中构建它，或者在`App.js`中创建它。

为了使用该组件，我们将向它传递一个包含我们的项目数据的`hit`属性。该组件还将使用特定的类名，这些类名与我们在本教程开始时导入的经典主题中的特定项目相匹配。

在标记内部，我们提供了突出显示的文本和两个新项:`hit.icon`和 return 键的 SVG 表示。这增加了一些定制的图标

```
function Action({ hit }) {
    // Component to display the items  
    return (
      <div className="aa-ItemWrapper">
        <div className="aa-ItemContent">
          <div className="aa-ItemIcon">{hit.icon}</div>
          <div className="aa-ItemContentBody">
            <div className="aa-ItemContentTitle">
              <span>{hit.highlighted}</span>
            </div>
          </div>
        </div>
        <div className="aa-ItemActions">
          <button
            className="aa-ItemActionButton aa-DesktopOnly aa-ActiveOnly"
            type="button"
            title="Select"
          >
            <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
              <path d="M18.984 6.984h2.016v6h-15.188l3.609 3.609-1.406 1.406-6-6 6-6 1.406 1.406-3.609 3.609h13.172v-4.031z" />
            </svg>
          </button>
        </div>
      </div>
    );
  }
```

一旦组件被创建，我们需要改变我们的`item`模板来使用它。

```
templates: {
    item({ item }) {
      return <Action hit={item} />;
    }
  }
```

我们还需要为每个操作项添加一个图标属性。在这个例子中，我们有一些手工制作的 SVG，但是任何图标库都可以。

```
return [
    {
      icon: (
        <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
          />
        </svg>
      ),
      label: "/dark",
      enabled: state.context.dark,
      onSelect({ setContext }) {
        document.querySelector("body").classList.add("dark");
      }
    },
    {
      icon: (
        <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
          />
        </svg>
      ),
      label: "/light",
      onSelect() {
        document.querySelector("body").classList.remove("dark");
        notify("Light Mode enabled");
      }
    },
  ]
```

这看起来真的很不错。有点奇怪的是，该网站是在光模式，但光模式选项没有提供这方面的指示。让我们为我们的用户添加一些上下文。

### [](#creating-an-enabled-state-with-setcontext)用`setContext`创建启用状态

自动完成使我们能够访问状态。让我们用它来创建一个`enabled`状态，并在我们的动作被触发时设置这个状态。

让我们从给每个名为`enabled`的动作添加一个新属性开始。

```
{ //...
  label: "/dark",
  enabled: state.context.dark,
  // ...
},
{ //...
  label: "/light",
  enabled: !state.context.dark,
  // ...
}
```

该属性将检查 Autocomplete 的 state 对象中是否有标记为`dark`的上下文项。如果`dark`设置为`true`，暗动作将为真`enabled`状态，如果`false`，亮将为真。

为了获得上下文，我们需要在我们的`onSelect`函数中设置应用程序的上下文。我们可以将`setContext`方法传递给我们的`onSelect`函数，并使用它将`dark`设置为真或假。

我们需要为 sources 方法传递 options 对象中的`setContext`方法。从将`getSources={({ query })}`改为`getSources={({ query, setContext })}`开始。然后我们可以在我们的`onSelect`函数中使用`setContext`。

```
onSelect({ setContext }) {
  document.querySelector("body").classList.remove("dark");
  setContext({ dark: false });
}
```

现在剩下的就是在我们的组件中使用`enabled`布尔值。

```
function Action({ hit }) {
    // Component to display the items
    return (
      <div className="aa-ItemWrapper">
        <div className="aa-ItemContent">
          <div className="aa-ItemIcon">{hit.icon}</div>
          <div className="aa-ItemContentBody">
            <div className="aa-ItemContentTitle">
              <span>{hit.highlighted}</span>
              {hit.enabled && (
                <code className="aa-ItemContentTitleNote">Enabled</code>
              )}
            </div>
          </div>
        </div>
        <div className="aa-ItemActions">
          <button
            className="aa-ItemActionButton aa-DesktopOnly aa-ActiveOnly"
            type="button"
            title="Select"
          >
            <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
              <path d="M18.984 6.984h2.016v6h-15.188l3.609 3.609-1.406 1.406-6-6 6-6 1.406 1.406-3.609 3.609h13.172v-4.031z" />
            </svg>
          </button>
        </div>
      </div>
    );
  }
```

因此，我们的 omnibar 是有状态的。这是黑暗模式的一个相对简单的例子。为了更好地构建它，您可以从应用程序的整体状态或者基于用户的本地存储中的信息来添加和设置 omnibar 的上下文。

## [](#next-steps)下一步

在本教程中，我们构建了不仅仅是搜索的 Autocomplete，但是您也可以使用不同的源对象和它自己的一组模板来添加常规的搜索功能。您还可以扩展这些操作，以匹配应用程序具有的任何潜在操作。

一些想法:

*   添加到待办事项列表或已保存列表
*   时事通讯注册
*   用户配置文件更新

我们很想看看你有什么想法。分叉[启动沙箱](https://alg.li/omnibar-starter)(或者[这个完成了一个](https://codesandbox.io/s/autocomplete-actions-finished-eh7xw))，创造一些新的东西，然后[在 Twitter](https://twitter.com/algolia) 上和我们分享。在我们的开源代码交换平台上查看相关解决方案。

[![](img/7665551c18b687f25dcadc15cb213b7d.png)](https://www.algolia.com/developers/code-exchange/?page=1&refinementList%5Bproduct_features%5D%5B0%5D=Autocomplete)