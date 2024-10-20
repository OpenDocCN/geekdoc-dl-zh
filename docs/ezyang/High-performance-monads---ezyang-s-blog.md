<!--yml

分类：未分类

日期：2024-07-01 18:18:08

-->

# 高性能单子：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/09/high-performance-monads/`](http://blog.ezyang.com/2010/09/high-performance-monads/)

延续以其难以使用而闻名：它们是函数式编程世界中的“goto”。它们可以搞砸或者做出惊人的事情（毕竟，异常不过是一个结构良好的非本地 goto）。本文适合那些对延续有一定了解但怀疑它们能否用于日常编程任务的读者：我想展示延续如何让我们以一种相当系统的方式定义高性能单子，如[逻辑单子](http://hackage.haskell.org/package/logict)。一个（可能）相关的帖子是[所有单子之母](http://blog.sigfpe.com/2008/12/mother-of-all-monads.html)。

```
> import Prelude hiding (Maybe(..), maybe)

```

* * *

我们将从一个热身开始：身份单子。

```
> data Id a = Id a
> instance Monad Id where
>     Id x >>= f = f x
>     return = Id

```

这个单子的延续传递风格（CPS）版本是您的标准`Cont`单子，但没有定义`callCC`。

```
> data IdCPS r a = IdCPS { runIdCPS :: (a -> r) -> r }
> instance Monad (IdCPS r) where
>     IdCPS c >>= f =
>         IdCPS (\k -> c (\a -> runIdCPS (f a) k))
>     return x = IdCPS (\k -> k x)

```

虽然解释 CPS 不在本文的范围内，但我想指出这个翻译中的一些习语，我们将在一些更高级的单子中重复使用它们。

1.  为了“提取”`c`的值，我们传递了一个 lambda `(\a -> ...)`，其中`a`是`c`计算的结果。

1.  只有一个成功的延续`k :: a -> r`，它总是最终被使用。在绑定的情况下，它被传递给`runIdCPS`，在返回的情况下，它被直接调用。在后续的单子中，我们会有更多的延续漂浮。

* * *

顺着单子教程的步伐，下一步是看看那古老的 Maybe 数据类型及其相关的单子实例。

```
> data Maybe a = Nothing | Just a
> instance Monad Maybe where
>     Just x >>= f = f x
>     Nothing  >>= f = Nothing
>     return = Just

```

在实现此单子的 CPS 版本时，我们将需要两个延续：一个成功的延续（`sk`）和一个失败的延续（`fk`）。

```
> newtype MaybeCPS r a = MaybeCPS { runMaybeCPS :: (a -> r) -> r -> r }
> instance Monad (MaybeCPS r) where
>     MaybeCPS c >>= f =
>         MaybeCPS (\sk fk -> c (\a -> runMaybeCPS (f a) sk fk) fk)
>     return x = MaybeCPS (\sk fk -> sk x)

```

将此单子与`IdCPS`进行比较：你会注意到它们非常相似。实际上，如果我们从代码中消除所有关于`fk`的提及，它们将是相同的！我们的单子实例大力支持成功。但是如果我们添加以下函数，情况就会改变：

```
> nothingCPS = MaybeCPS (\_ fk -> fk)

```

此函数忽略了成功的延续并调用失败的延续：你应该确信一旦调用失败的延续，它立即退出`MaybeCPS`计算。（提示：看看我们运行`MaybeCPS`延续的任何情况：我们为失败延续传递了什么？我们为成功延续传递了什么？）

为了更好地说明，我们还可以定义：

```
> justCPS x = MaybeCPS (\sk _ -> sk x)

```

其实这只是伪装的`return`。

您可能还会注意到我们的`MaybeCPS`新类型的签名与`maybe`“析构”函数的签名非常相似，因此被称为它破坏了数据结构：

```
> maybe :: Maybe a -> (a -> r) -> r -> r
> maybe m sk fk =
>     case m of
>         Just a  -> sk a
>         Nothing -> fk

```

（为了教学目的，类型已重新排序。）我特意将“默认值”命名为`fk`：它们是同一回事！

```
> monadicAddition mx my = do
>   x <- mx
>   y <- my
>   return (x + y)
> maybeTest    = maybe       (monadicAddition (Just 2)   Nothing)    print (return ())
> maybeCPSTest = runMaybeCPS (monadicAddition (return 2) nothingCPS) print (return ())

```

这两段代码的最终结果相同。然而，`maybeTest` 在单子部分内部构造了一个 `Maybe` 数据结构，然后再次拆除它。`runMaybeCPS` 则完全跳过了这个过程：这就是 CPS 转换获得性能优势的地方：没有数据结构的构建和拆除。

现在，公平地说原始的 Maybe 单子，在许多情况下 GHC 会为您执行此转换。因为代数数据类型鼓励创建大量小数据结构，GHC 将尽最大努力确定何时创建数据结构，然后立即拆除它，从而优化掉这种浪费的行为。前进！

* * *

列表单子（也称为“流”单子）编码了非确定性。

```
> data List a = Nil | Cons a (List a)
> instance Monad List where
>     Nil >>= _ = Nil
>     Cons x xs >>= f = append (f x) (xs >>= f)
>     return x = Cons x Nil
> append Nil ys = ys
> append (Cons x xs) ys = Cons x (append xs ys)

```

`Nil` 本质上等同于 `Nothing`，因此我们的失败延续再次出现。但是，我们必须稍微不同地处理我们的成功延续：虽然我们可以简单地传递列表的第一个 `Cons` 的值给它，但这将阻止我们继续处理列表的其余部分。因此，我们需要向成功延续传递一个恢复延续 (`rk`)，以便在需要时继续其路径。

```
> newtype LogicCPS r a = LogicCPS { runLogicCPS :: (a -> r -> r) -> r -> r }
> instance Monad (LogicCPS r) where
>     LogicCPS c >>= f =
>         LogicCPS (\sk fk -> c (\a rk -> runLogicCPS (f a) sk rk) fk)
>     return x = LogicCPS (\sk fk -> sk x fk)

```

请记住，`return`生成的是单元素列表，因此没有更多的内容可以继续，我们将成功的延续 `fk` 作为恢复的延续。

旧的数据构造函数也可以进行 CPS 变换：`nilCPS` 看起来就像 `nothingCPS`。`consCPS` 调用成功的延续，并且需要生成一个恢复的延续，这恰好可以通过它的第二个参数来方便地完成：

```
> nilCPS =
>     LogicCPS (\_ fk -> fk)
> consCPS x (LogicCPS c) =
>     LogicCPS (\sk fk -> sk x (c sk fk))
> appendCPS (LogicCPS cl) (LogicCPS cr) =
>     LogicCPS (\sk fk -> cl sk (cr sk fk))

```

这些类型看起来应该非常熟悉。稍微调整一下这种类型（并将 b 重命名为 r）：

```
foldr :: (a -> b -> b) -> b -> [a] -> b

```

我得到：

```
fold :: List a -> (a -> r -> r) -> r -> r

```

嘿，这是我的延续。所以我们所做的一切就是一个折叠操作，只是没有实际构造列表！

敏锐的读者可能也会注意到，列表的 CPS 表达式仅仅是列表的[高阶 Church 编码](http://en.wikipedia.org/wiki/Church_encoding#Higher-order_function)。

在几个方面，CPS 转换后的列表单子赢得了巨大的优势：我们从不需要构造和拆除列表，并且连接两个列表只需 `O(1)` 时间。

* * *

最后一个例子：树叶单子（来自 Edward Kmett 的指示树幻灯片）：

```
> data Leafy a = Leaf a | Fork (Leafy a) (Leafy a)
> instance Monad Leafy where
>     Leaf a >>= f = f a
>     Fork l r >>= f = Fork (l >>= f) (r >>= f)
>     return a = Leaf a

```

事实证明，如果我们想要对这种数据类型进行折叠，我们可以重用 `LogicCPS`：

```
> leafCPS x = return x
> forkCPS l r = appendCPS l r

```

要反向进行操作，如果我们结合到目前为止定义的所有关于逻辑的 CPS 操作，并将它们转换回数据类型，我们将得到一个可连接的列表：

```
> data Catenable a = Append (Catenable a) (Catenable a) | List (List a)

```

* * *

总结，我们已经表明，当我们构建一个大数据结构，只有在完成时才会被销毁时，我们最好将这两个过程合并，并且[将我们的数据结构重新转换回代码](http://blog.ezyang.com/2010/09/data-is-code/)。类似地，如果我们想对我们的数据结构执行“数据结构”-样的操作，实际上构建它可能更好：像`tail`这样的 Church 编码因其效率低下而臭名昭著。我并未讨论编码某种状态的单子：在许多方面，它们与控制流单子是不同类别的（或许更准确地说“Cont 是所有控制流单子的鼻祖”）。

引用《星球大战》，下次当你发现自己陷入连续操作的混乱中时，*使用数据结构*！

*附录.* CPS（Continuation Passing Style）转换数据结构遍历与单子（monads）无关。你可以对任何东西进行这种操作。恰巧控制流单子的杀手级特性——非确定性，正好从这种转换中受益良多。

*参考.* 这个主题已经有大量现有的讨论。

我可能还错过了其他显而易见的一些内容。
