<!--yml

类别：未分类

日期：2024-07-01 18:18:11

-->

# 通用化 API：ezyang 博客

> 来源：[`blog.ezyang.com/2010/08/generalizing-apis/`](http://blog.ezyang.com/2010/08/generalizing-apis/)

*编辑.* ddarius 指出，类型族的例子是反过来的，所以我把它们调整成了与函数依赖相同的方式。

类型函数可用于执行各种精妙的类型级计算，但也许最基本的用途是允许构建通用 API，而不仅仅依赖于模块导出的“大部分相同的函数”。你需要多少类型技巧取决于 API 的属性，也许最重要的是你的数据类型的属性。

* * *

假设我有一个单一数据类型上的单一函数：

```
defaultInt :: Int

```

而我想要通用化它。我可以通过创建一个*类型类*来轻松实现：

```
class Default a where
  def :: a

```

对单个类型的抽象通常只需要普通的类型类。

* * *

假设我有一个在多个数据类型上的函数：

```
data IntSet
insert :: IntSet -> Int -> IntSet
lookup :: IntSet -> Int -> Bool

```

我们希望对`IntSet`和`Int`进行抽象化。由于我们所有的函数都提到了这两种类型，我们所需做的就是编写一个*多参数类型类*：

```
class Set c e where
  insert :: c -> e -> c
  lookup :: c -> e -> Bool

instance Set IntSet Int where ...

```

* * *

如果我们运气不好，一些函数可能不会使用所有的数据类型：

```
empty :: IntSet

```

在这种情况下，当我们尝试使用该函数时，GHC 会告诉我们它无法确定使用哪个实例：

```
No instance for (Set IntMap e)
  arising from a use of `empty'

```

其中一件事要做的就是引入 `IntSet` 和 `Int` 之间的*功能依赖*。依赖意味着某些东西依赖于另一些东西，那么哪种类型依赖于什么？在这里我们没有太多选择：因为我们想要支持函数 `empty`，其签名中并没有任何地方提到 `Int`，因此依赖将从 `IntSet` 到 `Int`，也就是说，给定一个集合（`IntSet`），我可以告诉你它包含的是什么（一个 `Int`）。

```
class Set c e | c -> e where
  empty :: c
  insert :: c -> e -> c
  lookup :: c -> e -> Bool

```

注意，这仍然基本上是一个多参数类型类，我们只是给 GHC 一个小提示，告诉它如何选择正确的实例。如果需要，我们也可以引入反方向的功能依赖。出于教育目的，让我们假设我们的老板真的想要一个“null”元素，它总是集合的成员，并且在插入时不做任何事情：

```
class Set c e | c -> e, e -> c where
  empty :: c
  null :: e
  insert :: c -> e -> c
  lookup :: c -> e -> Bool

```

还要注意，每当我们添加功能依赖时，我们就排除了提供另一个实例的可能性。在最后一个类型类对于 `Set` 是非法的：

```
instance Set IntSet Int where ...
instance Set IntSet Int32 where ...
instance Set BetterIntSet Int where ...

```

这将报告“功能依赖冲突。”

* * *

功能依赖有时会因为与其他某些类型特性的交互而受到诟病。GHC 最近添加的等效功能是*关联类型*（也称为*类型族*或*数据族*）。

而不是告诉 GHC 如何自动从另一个类型中推断（通过依赖），我们创建一个显式的类型族（也称为类型函数），它提供了映射：

```
class Set c where
  data Elem c :: *
  empty :: c
  null :: Elem c
  insert :: c -> Elem c -> c
  lookup :: c -> Elem c -> Bool

```

注意我们的类型类不再是多参数的：它有点像如果我们从 `c -> e` 引入了一个函数依赖。但是，它如何知道 `null` 的类型应该是什么？简单：它让你告诉它：

```
instance Set IntSet where
  data Elem IntSet = IntContainer Int
  empty = emptyIntSet
  null = IntContainer 0

```

注意 `data` 的右侧不是一个类型：它是一个数据构造函数，然后是一个类型。数据构造函数将告诉 GHC 使用哪个 `Elem` 的实例。

* * *

在本文的原始版本中，我定义了相反方向的类型类：

```
class Key e where
  data Set e :: *
  empty :: Set e
  null :: e
  insert :: Set e -> e -> Set e
  lookup :: Set e -> e -> Bool

```

我们的类型函数朝着另一个方向发展，我们可以根据正在使用的类型变体实现*容器*，这可能不是我们拥有的类型。这是数据族的一个主要用例，但与通用化 API 的问题不直接相关，所以我们暂时不考虑它。

* * *

`IntContainer` 看起来很像一个 newtype，并且实际上可以成为一个：

```
instance Set IntSet where
  newtype Elem IntSet = IntContainer Int

```

如果你觉得包装和解包 newtype 很烦人，在某些情况下，你可以只使用类型同义词：

```
class Set c where
  type Elem c :: *

instance Set IntSet where
  type Elem IntSet = Int

```

然而，这样做会排除一些你可能想写的功能，例如自动专门化你的通用函数：

```
x :: Int
x = null

```

GHC 会报错：

```
Couldn't match expected type `Elem e'
       against inferred type `[Int]'
  NB: `Container' is a type function, and may not be injective

```

既然我也可以写成：

```
instance Set BetterIntSet where
  type Elem BetterIntSet = Int

```

GHC 不知道要使用 `null` 的哪个 `Set` 实例：`IntSet` 还是 `BetterIntSet`？你需要通过另一种方式将此信息传递给编译器，如果这完全在幕后进行，你就有点倒霉了。这与函数依赖有着明显的不同，如果你有一个非单射关系，它们会产生冲突。

* * *

另一种方法，如果你有幸定义你的数据类型，是在实例内部定义数据类型：

```
instance Set RecordMap where
  data Elem RecordMap = Record { field1 :: Int, field2 :: Bool }

```

然而，请注意，新 `Record` 的类型不是 `Record`；它是 `Elem RecordMap`。你可能会发现类型同义词有用：

```
type Record = Elem RecordMap

```

与 newtype 方法相比，没有太大区别，只是避免了添加额外的包装和解包层。

* * *

在许多情况下，我们希望规定我们 API 中的数据类型具有某些类型类：

```
instance Ord Int where ...

```

强制执行这一点的一种低技术方式是将其添加到我们所有函数的类型签名中：

```
class Set c where
  data Elem c :: *
  empty :: c
  null :: Ord (Elem c) => Elem c
  insert :: Ord (Elem c) => c -> Elem c -> c
  lookup :: Ord (Elem c) => c -> Elem c -> Bool

```

但更好的方法是只需在 `Set` 上添加一个类约束，使用*灵活的上下文*：

```
class Ord (Elem c) => Set c where
  data Elem c :: *
  empty :: c
  null :: Elem c
  insert :: c -> Elem c -> c
  lookup :: c -> Elem c -> Bool

```

* * *

我们可以使函数和数据类型通用化。我们还可以使类型类通用化吗？

```
class ToBloomFilter a where
  toBloomFilter :: a -> BloomFilter

```

假设我们决定允许多个 `BloomFilter` 的实现，但仍然希望为转换成任何你想要的布隆过滤器提供统一的 API。

不是[直接](http://hackage.haskell.org/trac/ghc/wiki/TypeFunctions/ClassFamilies)，但我们可以伪造它：只需创建一个捕捉所有通用类型类，并将其参数化为真实类型类的参数：

```
class BloomFilter c where
  data Elem c :: *

class BloomFilter c => ToBloomFilter c a where
  toBloomFilter :: a -> c

```

* * *

稍微退后一步，比较函数依赖和类型族产生的类型签名：

```
insertFunDeps :: Set c e => c -> e -> c
insertTypeFamilies :: Set c => c -> Elem c -> c

emptyFunDeps :: Set c e => c
emptyTypeFamilies :: Set c => c

```

因此，类型族（type families）将实现细节隐藏在类型签名之后（你只使用你需要的关联类型，与`Set c e => c`相反，其中`e`是必需的但没有用于任何操作—如果你有 20 个关联数据类型，这更加明显）。然而，当你需要为你的关联数据引入新类型包装器（`Elem`）时，它们可能会显得有些啰嗦。功能依赖（functional dependencies）非常适合自动推断其他类型，而无需重复自己。

（感谢 Edward Kmett 指出这一点。）

* * *

从这里开始要做什么呢？我们只是初步了解了类型级编程的表面，但是为了通用化 API，这基本上就是你需要知道的全部！找到你写过的在多个模块中重复的 API，每个模块提供不同的实现。找出哪些函数和数据类型是基本的。如果你有很多数据类型，就应用这里描述的技巧来确定你需要多少类型机制。然后，让你的 API 变得通用起来吧！
