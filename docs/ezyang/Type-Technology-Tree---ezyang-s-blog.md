<!--yml

类别：未分类

date: 2024-07-01 18:17:57

-->

# 类型技术树：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/03/type-tech-tree/`](http://blog.ezyang.com/2011/03/type-tech-tree/)

## 类型技术树

他们说，你并不是发现了高级类型系统扩展：相反，类型系统扩展发现了你！尽管如此，了解 GHC 的类型扩展的技术树仍然是值得的，这样你可以决定需要多少能力（以及对应的头疼的错误消息）。

1.  一些扩展自动启用其他扩展（蕴含）；

1.  一些扩展提供了另一扩展提供的所有功能（包含）；

1.  一些扩展与其他扩展非常良好地协同工作（协同作用）；

1.  一些扩展提供了与另一扩展相当（但以不同的形式）的功能（等效）。

此外值得注意的是，GHC 手册将这些扩展划分为“数据类型和类型同义词的扩展”、“类和实例声明”、“类型族”和“其他类型系统扩展”。我在这里对它们进行了稍微不同的组织。

### 等级和数据

我们的第一个技术树将任意等级的多态性和广义代数数据类型结合在一起。

简言之：

+   GADTSyntax 允许普通数据类型以 GADT 风格编写（带有显式构造函数签名）：`data C where C :: Int -> C`

+   [显式的 forall](http://hackage.haskell.org/trac/haskell-prime/wiki/ExplicitForall) 允许你显式声明多态类型中的量化器：`forall a. a -> a`

+   [存在量化](http://hackage.haskell.org/trac/haskell-prime/wiki/ExistentialQuantification) 允许将类型隐藏在数据构造器中：`data C = forall e. C e`

+   [GADTs](http://hackage.haskell.org/trac/haskell-prime/wiki/GADTs) 允许显式构造函数签名：`data C where C :: C a -> C b -> C (a, b)`。包含存在量化因此，存在量化的数据类型只是那些类型变量不在结果中的多态构造函数。

+   [多态组件](http://hackage.haskell.org/trac/haskell-prime/wiki/PolymorphicComponents) 允许你在数据类型字段中写入 `forall`：`data C = C (forall a. a)`

+   [Rank2Types](http://hackage.haskell.org/trac/haskell-prime/wiki/Rank2Types) 允许多态参数：`f :: (forall a. a -> a) -> Int -> Int`。与 GADTs 结合，它包含多态组件，因为数据类型字段中的 `forall` 对应于具有二阶类型的数据构造器。

+   [RankNTypes](http://hackage.haskell.org/trac/haskell-prime/wiki/RankNTypes)：`f :: Int -> (forall a. a -> a)`

+   ImpredicativeTypes 允许多态函数和数据结构参数化为多态类型：`Maybe (forall a. a -> a)`

### 实例

我们的下一个技术树涉及类型类实例。

简言之：

+   [TypeSynonymInstances](http://hackage.haskell.org/trac/haskell-prime/wiki/TypeSynonymInstances) 允许在实例声明中类似宏地使用类型同义词：`instance X String`

+   [FlexibleInstances](http://hackage.haskell.org/trac/haskell-prime/wiki/FlexibleInstances) 允许更多有趣的类型表达式的实例，但限制以保持可判定性：`instance MArray (STArray s) e (ST s)`（经常与多参数类型类一起看到，但不在图表中）

+   [UndecidableInstances](http://hackage.haskell.org/trac/haskell-prime/wiki/UndecidableInstances) 允许更有趣的类型表达式的实例，没有限制，但牺牲了可判定性。参见[Oleg](http://okmij.org/ftp/Haskell/types.html#undecidable-inst-defense)作为合法示例。

+   [FlexibleContexts](http://hackage.haskell.org/trac/haskell-prime/wiki/FlexibleContexts) 允许在函数和实例声明的约束中更多的类型表达式：`g :: (C [a], D (a -> b)) => [a] -> b`

+   [OverlappingInstances](http://hackage.haskell.org/trac/haskell-prime/wiki/OverlappingInstances) 允许实例在有最特定实例的情况下重叠：`instance C a; instance C Int`

+   [IncoherentInstances](http://hackage.haskell.org/trac/haskell-prime/wiki/IncoherentInstances) 允许实例任意重叠。

或许在此图表中显著缺失的是 `MultiParamTypeClasses`，它位于以下。

### 类型族和函数依赖

我们最终的技术树涉及类型编程：

简言之：

+   [KindSignatures](http://hackage.haskell.org/trac/haskell-prime/wiki/KindSignatures) 允许声明类型变量的种类：`m :: * -> *`

+   [MultiParamTypeClasses](http://hackage.haskell.org/trac/haskell-prime/wiki/MultiParamTypeClasses) 允许类型类跨越多个类型变量：`class C a b`

+   [FunDeps](http://hackage.haskell.org/trac/haskell-prime/wiki/FunctionalDependencies) 允许限制多参数类型类的实例，有助于解决歧义：`class C a b | a -> b`

+   [TypeFamilies](http://www.haskell.org/ghc/docs/7.0.1/html/users_guide/type-families.html) 允许在类型上进行“函数”操作：`data family Array e`

函数依赖与类型族之间的对应关系众所周知，尽管不完美（类型族可能更啰嗦，无法表达某些相等性，但在广义代数数据类型（GADTs）中更友好）。
