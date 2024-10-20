<!--yml

category: 未分类

date: 2024-07-01 18:17:14

-->

# 类型类：收敛性、一致性和全局唯一性：ezyang's 博客

> 来源：[`blog.ezyang.com/2014/07/type-classes-confluence-coherence-global-uniqueness/`](http://blog.ezyang.com/2014/07/type-classes-confluence-coherence-global-uniqueness/)

今天，我想讨论类型类背后的一些核心设计原则，这是 Haskell 中一个非常成功的特性。这里的讨论受到我们在 MSRC 支持背包中使用类型类的工作的密切影响。在我进行背景阅读时，我惊讶地发现人们在谈论类型类时普遍误用了“收敛性”和“一致性”这两个术语。因此，在这篇博文中，我想澄清这一区别，并提出一个新术语，“全局唯一性实例”，用于描述人们口头上所说的收敛性和一致性的属性。

* * *

让我们从这两个术语的定义开始。收敛性是来自术语重写的一种属性：如果一组实例是**收敛**的，那么无论进行约束求解的顺序如何，GHC 都将以一组规范的约束终止，这些约束必须满足任何给定类型类的使用。换句话说，收敛性表明，我们不会因为使用了不同的约束求解算法而得出程序不通过类型检查的结论。

**一致性**的密切相关者是**收敛性**（在论文“类型类：探索设计空间”中定义）。该性质表明，程序的每个不同有效的类型推导都会导致具有相同动态语义的生成程序。为什么不同的类型推导会导致不同的动态语义呢？答案是上下文缩减，它选择类型类实例，并将其详细解释为生成代码中的具体字典选择。收敛性是一致性的先决条件，因为对于不能通过类型检查的程序，我们几乎不能谈论其动态语义。

那么，当人们将 Scala 类型类与 Haskell 类型类进行比较时，他们通常指的是**全局唯一性实例**，定义如下：在完全编译的程序中，对于任何类型，给定类型类的实例解析最多只有一个。像 Scala 这样具有局部类型类实例的语言通常不具备此属性，但在 Haskell 中，我们发现这一属性在构建诸如集合等抽象时非常方便。

* * *

那么，实际上 GHC 强制执行哪些属性？在没有任何类型系统扩展的情况下，GHC 使用一组规则来确保类型类解析是一致的和完整的。直观地说，它通过具有非重叠*的实例集*来实现这一点，确保只有一种方法来解决想要的约束。重叠是比一致性或完整性更严格的限制，通过 `OverlappingInstances` 和 `IncoherentInstances`，GHC 允许用户放宽这一限制，“如果他们知道自己在做什么的话。”

然而，令人惊讶的是，GHC *并不* 强制全局唯一性的实例。导入的实例在尝试用于实例解析之前不会被检查是否重叠。考虑以下程序：

```
-- T.hs
data T = T
-- A.hs
import T
instance Eq T where
-- B.hs
import T
instance Eq T where
-- C.hs
import A
import B

```

当使用一次性编译时，只有在实际尝试使用 `C` 中的 `Eq` 实例时，`C` 才会报告实例重叠。这是 [有意设计](https://ghc.haskell.org/trac/ghc/ticket/2356)：确保没有重叠实例需要及时读取模块可能依赖的所有接口文件。

* * *

我们可以总结这三个属性如下。在文化上，Haskell 社区期望*实例的全局唯一性*能够保持：实例的隐式全局数据库应该是一致的和完整的。然而，GHC 并不强制实例的唯一性：相反，它仅保证在编译任何给定模块时使用的实例数据库的*子集*是一致的和完整的。当一个实例声明时，GHC 确实会进行一些测试，看看它是否会与可见实例重叠，但检查 [绝不完美](https://ghc.haskell.org/trac/ghc/ticket/9288)；真正的*类型类约束解析*有最终决定权。一个缓解因素是在没有*孤儿实例*的情况下，GHC 保证会及时注意到实例数据库是否有重叠（假设实例声明检查确实有效……）

显然，GHC 的惰性行为对大多数 Haskeller 来说是令人惊讶的，这意味着懒惰检查通常是足够好的：用户很可能会以某种方式发现重叠的实例。然而，相对简单地构造违反实例的全局唯一性的示例程序是可能的：

```
-- A.hs
module A where
data U = X | Y deriving (Eq, Show)

-- B.hs
module B where
import Data.Set
import A

instance Ord U where
compare X X = EQ
compare X Y = LT
compare Y X = GT
compare Y Y = EQ

ins :: U -> Set U -> Set U
ins = insert

-- C.hs
module C where
import Data.Set
import A

instance Ord U where
compare X X = EQ
compare X Y = GT
compare Y X = LT
compare Y Y = EQ

ins' :: U -> Set U -> Set U
ins' = insert

-- D.hs
module Main where
import Data.Set
import A
import B
import C

test :: Set U
test = ins' X $ ins X $ ins Y $ empty

main :: IO ()
main = print test

```

```
-- OUTPUT
$ ghc -Wall -XSafe -fforce-recomp --make D.hs
[1 of 4] Compiling A ( A.hs, A.o )
[2 of 4] Compiling B ( B.hs, B.o )

B.hs:5:10: Warning: Orphan instance: instance [safe] Ord U
[3 of 4] Compiling C ( C.hs, C.o )

C.hs:5:10: Warning: Orphan instance: instance [safe] Ord U
[4 of 4] Compiling Main ( D.hs, D.o )
Linking D ...
$ ./D
fromList [X,Y,X]

```

在本地，所有类型类解析都是一致的：在每个模块可见的实例子集中，类型类解析可以无歧义地完成。此外，`ins` 和 `ins'` 的类型解决了类型类解析，因此在 `D` 中，当数据库现在重叠时，不会发生解析，因此错误永远不会被发现。

这个例子很容易被看作是 GHC 中的一个实现上的瑕疵，继续假装类型类实例的全局唯一性是成立的。然而，类型类实例全局唯一性的问题在于它们本质上是非模块化的：你可能会发现自己无法组合两个组件，因为它们意外地定义了相同的类型类实例，尽管这些实例深深地嵌入在组件的实现细节中。对于 Backpack 或者任何模块系统来说，这是一个很大的问题，它们的分离模块化开发宗旨旨在保证，如果库的编写者和应用的编写者按照共同的签名进行开发，链接将会成功。
