<!--yml

category: 未分类

date: 2024-07-01 18:17:26

-->

# 泛化可编程分号：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/10/generalizing-the-programmable-semicolon/`](http://blog.ezyang.com/2012/10/generalizing-the-programmable-semicolon/)

*购买者注意：前方有半成品研究思路。*

什么是单子（monad）？一个答案是，它是在非严格语言中排序操作的一种方式，一种表达“这应该在那之前执行”的方式。但另一个答案是，它是可编程分号，一种在进行计算时实现自定义副作用的方式。这些包括基本的效果，如状态、控制流和非确定性，以及更奇特的效果，比如[labeled IO](http://hackage.haskell.org/package/lio)。即使你不需要单子来排序，这样的功能也是有用的！

让我们来个大逆转：对于按需调用语言来说，可编程分号会是什么样子呢？也就是说，我们能否在不对计算进行排序的情况下实现这种可扩展性呢？

乍一看，答案是否定的。大多数按值调用语言无法抵制副作用的诱惑，但在按需调用中，副作用是足够痛苦的，以至于 Haskell 设法避免了它们（在大多数情况下！）任何使用过带有 `NOINLINE` 修饰的 `unsafePerformIO` 的人都可以证明这一点：依赖于优化，效果可能会执行一次，或者执行多次！正如保罗·列维所说：“第三种评估方法，按需调用，对于实现目的是有用的。但它缺乏干净的指称语义——至少对于除了发散和不规则选择之外的效果来说是如此，它们的特殊属性被利用在[Hen80]中提供按需模型。”所以我们不考虑按需调用。保罗·列维并不是说对于纯按需调用，没有指称语义（这些语义与称为名字调用的语义完全一致），而是当你添加副作用时，事情变得复杂。

但是这里有一个攻击角度的提示：列维继续展示了如何在名字调用中讨论副作用，并且在这里指定指称语义毫无困难。直觉上来看，其原因在于，在名字调用中，所有对带有附加效果的延迟值的使用（例如 case-matches）都会导致效果显现。一些效果可能会被丢弃（因为它们的值从未被使用），但除此之外，效果的发生完全是确定性的。

嗯！

当然，我们可以轻松通过放弃记忆化来实现这一点，但这是一个难以接受的牺牲。因此，我们的新问题是：*如何在保留共享的同时恢复具有影响力的按名字调用语义？*

在`Writer`单子的情况下，我们可以保留所有原始共享。过程非常简单：每个 thunk `a` 现在的类型是`(w, a)`（对于某个固定的单子`w`）。这个元组可以像原始的`a`一样共享，但现在它还有一个嵌入的效果`w`。每当`a`被强制时，我们简单地将效果追加到结果 thunk 的`w`中。下面是一个简单的解释器，实现了这一点：

```
{-# LANGUAGE GADTs #-}
import Control.Monad.Writer

data Expr a where
    Abs :: (Expr a -> Expr b) -> Expr (a -> b)
    App :: Expr (a -> b) -> Expr a -> Expr b
    Int :: Int -> Expr Int
    Add :: Expr Int -> Expr Int -> Expr Int
    Print :: String -> Expr a -> Expr a

instance Show (Expr a) where
    show (Abs _) = "Abs"
    show (App _ _) = "App"
    show (Int i) = show i
    show (Add _ _) = "Add"
    show (Print _ _) = "Print"

type M a = Writer String a
cbneed :: Expr a -> M (Expr a)
cbneed e@(Abs _) = return e
cbneed (App (Abs f) e) =
    let ~(x,w) = run (cbneed e)
    in cbneed (f (Print w x))
cbneed e@(Int _) = return e
cbneed (Add e1 e2) = do
    Int e1' <- cbneed e1
    Int e2' <- cbneed e2
    return (Int (e1' + e2'))
cbneed (Print s e) = do
    tell s
    cbneed e

sample = App (Abs (\x -> Print "1" (Add x x))) (Add (Print "2" (Int 2)) (Int 3))
run = runWriter

```

尽管最终输出是`"122"`（数字`2`出现两次），但将`2`添加到`3`的实际加法只发生了一次（您可以通过添加适当的跟踪调用来验证）。对于`Maybe`，您可以做类似的事情：通过稍微作弊，因为在`Nothing`的情况下，我们没有`x`的值，我们提供 bottom。我们永远不会被追究，因为我们总是在任何人获得值之前就进行了短路。

这里与应用函子有些相似之处，但我们要求更严格的条件：不仅计算的控制流需要固定，计算的值也必须固定！很明显，我们无法为大多数单子做到这一点。昨天在[Twitter](https://twitter.com/ezyang/status/253258690688344064)，我提出了以下签名和定律（回想起逆元），任何您想要对此过程执行的单子都必须实现这些：

```
extract :: Functor f => f a -> (a, f ())
s.t.  m == let (x,y) = extract m in fmap (const x) y

```

但似乎只有`Writer`具有适当的结构来正确执行这一点（既是单子又是余单子）。这很遗憾，因为我想要进行这种理论化的应用需要分配单元的能力。

然而，并非一无所获。即使无法完全共享，您仍可能实现部分共享：一种完全惰性和部分求值的混合体。不幸的是，这将需要对您的运行时进行重大和侵入性的更改（如果您想要将您的代码转换为 CPS，我不确定您将如何做到这一点），因此在这一点上我放下了这个问题，而是写了这篇博客文章。
