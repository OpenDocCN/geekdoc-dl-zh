<!--yml

分类：未分类

date: 2024-07-01 18:17:03

-->

# 可重用和可组合规范的问题：ezyang 的博客

> 来源：[`blog.ezyang.com/2016/12/the-problem-of-reusable-and-composable-specifications/`](http://blog.ezyang.com/2016/12/the-problem-of-reusable-and-composable-specifications/)

说服人们认为版本边界是特定 API 的不良近似并不太困难。当我们说 `>= 1.0 && < 1.1` 时，我们指的是什么？版本边界是代表某些具有特定语义的模块和函数集合的代理，这些模块需要构建库。版本边界是不精确的；从 1.0 到 1.1 的变化意味着什么？显然，我们应该写下我们实际需要的规范（类型或合同）。

这听起来都是一个好主意，直到你试图将其付诸实践，那时你会意识到版本号有一个很大的优点：它们非常简短。规范却可能变得相当庞大：甚至只是写下你依赖的所有函数类型可能就需要几页纸，更不用说描述更复杂行为的可执行合同了。更糟糕的是，同一个函数会被重复依赖；规范必须在每种情况下提供！

所以我们戴上我们的 PL 帽子说：“啊哈！我们需要的是一种能够*重用*和*组合*规范的机制。类似于……规范的*语言*！”但是在这一点上，关于这种语言应该如何工作存在分歧。

**规范就是代码。** 如果你与一个 Racketeer 聊天，他们会说：“嗯，[合同](https://docs.racket-lang.org/reference/contracts.html)只是[代码](https://docs.racket-lang.org/guide/Building_New_Contracts.html)，而我们知道如何重用和组合代码！”你可以用原始合同描述值，将它们组合成描述函数的合同，然后进一步将这些合同组合成关于模块的合同。你可以将这些合同收集到模块中，并在你的代码中共享它们。

There is one interesting bootstrapping problem: you're using your contracts to represent versions, but your contracts themselves live in a library, so should you version your contracts? Current thinking is that you shouldn't.

**但也许你不应该像通常那样组合它们。** 当我阅读 Clojure 规范文档的前言时，有一件事引起了我的注意，那就是[地图规范应仅包含键集](http://clojure.org/about/spec#_map_specs_should_be_of_keysets_only)，以及[它们如何处理这个问题](http://clojure.org/about/spec#_global_namespaced_names_are_more_important)。

spec 设计的核心原则是记录的规范不应采用`{ name: string, age: int }`的形式。相反，规范分为两部分：一组键 `{ name, age }`，以及从键到规范的映射，一旦注册，将适用于所有地图规范中的所有键。（请注意，键都是命名空间的，因此这并非是全局命名空间中的一场疯狂的自由竞争。）这样做的理由是：

> 在 Clojure 中，我们通过动态组合、合并和构建地图来增强功能。我们通常处理可选和部分数据，由不可靠外部来源产生的数据，动态查询等。这些地图代表了相同键的各种集合、子集、交集和并集，并且通常在使用相同键的情况下应具有相同的语义。在每个子集/并集/交集定义规范，然后冗余地说明每个键的语义，在最动态的情况下既是反模式又不可行。

**回到类型的世界。** 契约可以做所有这些，因为它们是代码，我们知道如何重用代码。但是在（非依赖性）类型化的语言中，类型的语言往往比值的语言要贫乏得多。以 Backpack 作为（异常表现出色的）例子，我们可以对签名执行的唯一操作是定义它们（对类型的完整定义）并将它们合并在一起。因此，Backpack 签名正面临着由规范识别出的冗余问题：因为模块的签名包括其函数的签名，所以每当编写略有不同的模块迭代时，您最终不得不重复这些函数签名。

要采用 Clojure 模型，您需要为每个模块编写一个单独的签名（每个位于自己的包中），然后让用户通过在每个他们想要使用的签名上添加`build-depends`来将它们组合在一起：

```
-- In Queue-push package
signature Queue where
  data Queue a
  push :: a -> Queue a -> Queue a

-- In Queue-pop package
signature Queue where
  data Queue a
  pop :: Queue a -> Maybe (Queue a, a)

-- In Queue-length package
signature Queue where
  data Queue a
  length :: Queue a -> Int

-- Putting them together (note that Queue is defined
-- in each signature; mix-in linking merges these
-- abstract data types together)
build-depends: Queue-push, Queue-pop, Queue-length

```

在我们当前的 Backpack 实现中，这有点不可思议：要为具有一百个方法的模块编写规范，您需要一百个包。在单个包中简明地定义多个公共库的能力可能会有所帮助，但这涉及到尚不存在的设计。（也许治疗比疾病更糟糕。包管理器 - 编译器分层结构再次展现了其丑陋的一面！）（自己的注意：签名包应该受到特殊对待；在实例化时确实不应该构建它们。）

**结论。** 直到我开始阅读像 Clojure 这样的动态语言如何应对规范问题时，我的许多思考才得以凝结：我认为这只是表明我们通过关注其他系统可以学到多少，即使它们的背景可能完全不同。（如果 Clojure 相信数据抽象，我认为他们可以从 Backpack 混入链接抽象数据声明中学到一些东西。）

在 Clojure 中，无法重用规范是一个不可接受的问题，这导致了它们当前的规范设计。在 Haskell 中，无法重用类型签名则接近无法使用的边缘：类型*刚好*足够短并且可以复制粘贴以至于可以容忍。对于这些类型的文档，情况稍好；这正是我寻找更好的签名重用机制的原因。

虽然 Backpack 的当前设计已经足够"好"来完成工作，但我仍然怀疑我们是否不能做得更好。一个诱人的选择是允许下游签名有选择地从较大的签名文件中挑选出某些函数添加到它们的需求中。但是，如果你需要`Queue.push`，你最好也需要`Queue.Queue`（没有它，`push`的类型甚至不能声明：避免问题）；这可能导致对最终需要的内容有很多疑问。值得深思。
