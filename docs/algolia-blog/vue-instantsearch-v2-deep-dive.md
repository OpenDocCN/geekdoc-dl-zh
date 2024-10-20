# 深入研究 Vue 即时搜索版本 2 

> 原文：<https://www.algolia.com/blog/engineering/vue-instantsearch-v2-deep-dive/>

这篇文章是与[布拉姆·亚当斯](https://github.com/bramses)合作完成的。

在[之前的帖子](https://www.algolia.com/blog/engineering/vue-instantsearch-v2/)中，我们宣布了 Vue 即时搜索 2.0 库的发布。那篇文章向我们介绍了该版本中包含的新特性。这一次，我们想更深入地了解我们是如何更新库的，以及我们为什么做出这样的决定。

## 当你的腿不像以前那样工作的时候🎶

自从我们在 2017 年发布第一个 Vue InstantSearch 以来，我们学到了很多东西。该库开始落后于我们更新的 InstantSearch.js 库，我们能够将学到的许多经验教训应用到 Vue 中。

Vue InstantSearch 的最初版本是由 Raymond 的一个小黑客开始的。我们对 Vue 的发展轨迹感到非常兴奋，并希望能够发布一个库，使 Algolia 用户能够更容易地将其集成到他们的框架中。

我们的 Vue 库的第一次迭代有一些问题。主要问题是我们是在黑暗中建造的！嗯，我是说，不完全是，我们的电脑是背光的。从某种意义上说，我们在没有太多社区投入的情况下建造了这个图书馆，这是黑暗的。

另一个主要问题是，我们的许多组件缺乏可定制性，需要人们经常完全复制我们的组件，以便能够充分地编辑它们。

通过承认我们的弱点，我们能够建立一个更好的图书馆。我们希望你也会喜欢它。如果你已经使用了 Vue InstantSearch 的前一版本，并且想知道增加了哪些新功能，请查看前一篇文章。

## 米罗，墙上的米罗，他们中谁最抽象？

![](img/bbb66d425639a1623809baed883b65a5.png)

Joan Miro – Portrait IV (1938)

说到抽象，Vue InstantSearch 现在已经涵盖了你。我们希望扩展组件的可定制性，让最终用户(就是你！)决定 Algolia 体验的外观和感觉。

在下面的段落中，我们将讨论如何定制即时搜索组件。每一层都比上一层更加可定制，允许您决定是喜欢易于集成的已构建组件，还是完全控制外观、感觉和功能。

想象一个搜索框的例子。这似乎是一个简单的组件，但它实际上是我们更复杂的部件之一，支持所有这些不同级别的定制。

![search box with a submit and reset button, current text is "hello"](img/9f70dad5b2cb978ee0adfd5f0519fc76.png)

The default ais-search-box component

### 作为道具的文本

最容易修改搜索框的部分之一是`placeholder`。这是将在输入元素本身中可见的文本。我们特别选择将这个属性转发到内部输入，因为我们将输入包装在一个表单中(更多关于我们为什么这样设计搜索框的信息，请参见关于搜索框的详细博文[】)。](https://www.algolia.com/blog/engineering/mobile-search-ux-tips/)

当一个 prop 被转发到一个内部组件时，它总是保留在你的外部组件上，或者有相同的名称，或者有一个一致的前缀来指示它被应用到哪个内部元素。

```
<ais-searchbox
  placeholder="Search for flavors: e.g. vanilla, chocolate…"
/> 
```

### 作为插槽的子组件

在搜索框中，我们还有一个带有图标的按钮，让用户清除查询。此时，不仅字符串是有效的，任何 Vue 组件也是有效的。因为它不需要访问依赖于它的渲染的任何状态，所以它是一个常规的插槽。

如果你不熟悉，你可以在这里阅读更多关于 Vue 插槽的信息。

```
<ais-searchbox>
  <template slot="resetIcon">✕</template>
</ais-searchbox> 
```

### 将整个 DOM 重写为作用域槽

最后，我们还在组件的根内添加了一个[作用域槽](https://vuejs.org/v2/guide/components-slots.html#Scoped-Slots)。我们向该槽提供从小部件的数据层(`connector`)检索的信息。该模板中的任何内容现在都可以访问`currentRefinement`、`refine`以及呈现该组件所需的任何内容。

多亏了`slot-scope`,我们现在不用离开模板就可以轻松地重写小部件的外观。

```
<ais-searchbox>
  <template
    slot="default"
    slot-scope="{ currentRefinement, refine }"
  >
    <input
      type="search"
      :value="currentRefinement"
      @input="refine($event.target.value)"
    />
  </template>
</ais-searchbox>
```

### 编写全新的 Vue 组件

有时候,`slot-scope`提供的控制水平仍然不足以满足你想要构建的东西。这有两个不同的原因:需要访问组件生命周期中底层业务逻辑(`connector`)提供的范围，或者自己编写一个定制的连接器。我们通过从连接器中抽象出创建“小部件”的逻辑，并用`createWidgetMixin` mixin 将它注册到其根组件，从而简化了这个过程。

假设您想要呈现自己的搜索框(而不是`ais-searchbox`)。我们可以使用`connectSearchBox`获取数据，使用`createWidgetMixin`将 Algolia 数据直接插入到我们的模板中。从这里开始，天空就是极限。你可以让你的组件看起来像你想要的那样！我们这样做的原因是在`created()`钩子期间数据被填充到`state`键。这允许您完全控制呈现逻辑和转换业务逻辑。

```
<template>
  <form
    v-if="state"
    action=""
    @submit="onFormSubmit"
    @reset="onFormReset"
  >
    <input
      :value="state.currentRefinement"
      @input="onInput"
    />
    <button type="reset">reset</button>
  </form>
</template>

<script>
import { connectSearchBox } from 'instantsearch.js/es/connectors';
import { createWidgetMixin } from 'vue-instantsearch';

export default {
  mixins: [
    createWidgetMixin({ connector: connectSearchBox }),
  ],
  methods: {
    onInput(event) {
      this.state.refine(event.target.value);
    },
    onFormSubmit() {
      const input = this.$el.querySelector('input[type=search]');
      input.blur();
    },
    onFormReset() {
      this.state.refine('');
    },
  },
};
</script> 
```

在这个组件中，没有太多使用完全自定义的小部件的情况。然而，对于包装更高级的第三方组件，或者修改`connector`的工作方式，这是一个非常有用的 API。

但是让我们先来看看为什么我们可能想要拆分逻辑。

## 拆分逻辑是否不理智？

在 Vue InstantSearch 2 中，我们选择在呈现组件和负责业务逻辑的连接器之间进行分离。当您创建自己的自定义组件并希望使用 Algolia 结果中的数据时，这非常有用。通过将业务逻辑与组件本身分离，概念上与另一个现有组件相同的组件不需要从头重写它们的业务逻辑。

> 一个抽象应该能够适应任何具有相同概念的用例

拆分逻辑需要找到一个平衡点。你不能把它分开太多，因为构建不同风格和框架所需的所有样板文件会变得乏味。然而，您也不能让它过于特定于视图，因为这样您就失去了将来添加新组件的灵活性。我们在这里使用的一个经验法则是，任何连接器都应该能够适应所有具有相同概念的组件。

这里的一个例子是显示为列表的菜单(只有一个选择的列表)，或者显示为下拉菜单的菜单。如果在一个业务逻辑抽象中，你用代码来处理选择逻辑，这将超出它的界限，你更有可能为每种情况做一个抽象。

在这一点上的选择是决定逻辑是否足够容易在视图层被重写(使用事件处理程序之类的东西)，从而只进行单一的业务逻辑抽象。另一方面，有时候一个想法变成了两种完全不同的抽象。

在这种情况下，两种完全不同的抽象是命中(即结果)和无限命中。这两种方法都从查询结果中读取数据，以显示当前应用的结果，但是无限命中抽象将包含诸如将以前的命中保留在范围内这样的内容，以便可以将所有可见的命中连接起来。

## 业务逻辑抽象之间的一致性

Algolia 以拥有最好的开发者体验而自豪。为了实现这个目标，我们专注于在不同的 JavaScript 风格之间建立一个一致的基础。无论您是在 Vue、React 还是普通的 JavaScript 中，这些连接器都可以工作。如果您的商业案例超出了我们的小工具所能处理的范围，我们鼓励您查看它们！

事实证明，在创建像 Vue InstantSearch 这样的库时，向一致抽象的转移非常有用。我们只需要在 Vue 组件中考虑一种类型的 API(使用一个`connector`)就可以让它一致地感知业务逻辑(从状态中读取，以及改变精化状态)。

如果我们选择将它分解成函数，每个函数都有自己的 API，那么这些函数会变得更简单——但是更难协同工作。这里的目标是使视图层尽可能简单，使添加新的框架风格更容易。

## 结论

我们希望你喜欢我们新的 Vue 库的内部，我们希望你和我们一样喜欢这个库。