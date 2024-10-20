<!--yml

category: 未分类

date: 2024-07-01 18:18:01

-->

# “gin and monotonic”的勘误：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/12/errata-for-gin-and-monotonic/`](http://blog.ezyang.com/2010/12/errata-for-gin-and-monotonic/)

## Errata for gin and monotonic

在忙于 GHC 的打包和黑客之间，我没有足够的时间来准备系列的下一篇文章或编辑上一篇文章的图片，所以今天只有一个小勘误贴。

+   完全列表图表缺少一些排序：★:⊥ ≤ ★:⊥:⊥ 等等。

+   在通常的指称语义中，你无法区分 ⊥ 和 `λ_.⊥`。然而，正如 Anders Kaseorg 和 Haskell 报告指出的那样，使用 `seq` 你可以区分它们。这或许是为何 `seq` 是一种不好的函数的真正原因。我一直假设它有更强的保证（这是 zygoloid 指出的），但对于 Haskell 实际上并非如此。

+   “停滞”图表中的“应该存在”箭头方向错误。

+   在完全列表图表的相同方式中，`head` 缺少一些排序，因此实际上灰色的 blob 是完全连接的。在某些情况下，我们可以有断开的 blob，但对于只有一个最大值的域来说不行。

+   fst 明显的拼写错误。

+   函数的正式偏序没有正确定义：最初它声明对于 f ≤ g，有 f(x) = g(x)；实际上，它比这弱：f(x) ≤ g(x)。

+   非勘误：头部图表的右侧被省略了，因为……添加所有箭头使其看起来相当难看。这是我在决定它不是一个好图之前所做的草图。

感谢 Anders Kaseorg、zygoloid、polux 和指出函数偏序错误的其他人（现在找不到那些信件了）的更正。

*不合逻辑的推论*。这是一个*真的*很简单的多变函数。相同的基本思想是 Text.Printf 的工作原理。希望它能帮助你在多变的旅程中。

```
{-# LANGUAGE FlexibleInstances #-}

class PolyDisj t where
    disj :: Bool -> t

instance PolyDisj Bool where
    disj x = x
instance PolyDisj t => PolyDisj (Bool -> t) where
    disj x y = disj (x || y)

```
