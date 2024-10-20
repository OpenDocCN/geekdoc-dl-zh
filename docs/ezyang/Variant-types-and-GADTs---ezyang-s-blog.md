<!--yml

类别：未分类

日期：2024-07-01 18:17:42

-->

# 变体类型和 GADTs：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/07/variant-types-and-gadts/`](http://blog.ezyang.com/2011/07/variant-types-and-gadts/)

OCaml 支持匿名变体类型，形式为``type a = [`Foo of int | `Bar of bool]``，具有适当的子类型关系。子类型在一般情况下比较棘手，因此我一直比较保守地使用这些变体类型。（即使一个特性给了你太多的灵活性，如果你有纪律地使用它，它也是可控的和有用的。）事实上，它们对于我通常会使用 GADTs 的一个特定用例非常方便。这就是“将多个和类型合并为单个和类型”的用例。

考虑以下在 Haskell 中的程序：

```
data A = Foo Int | Bar Bool
data B = Baz Char | Qux

```

如果你想定义 A 加 B 的道德等价物，最简单的方法是：

```
data AorB = A A | B B

```

但这种方法有点糟糕：我更喜欢一种平坦的命名空间，可以引用`A`和`B`（此编码在惰性存在时不等同于`data AorB = Foo Int | Bar Bool | Baz Char | Qux`）。如果在 OCaml 中使用普通的和类型，你也会遇到类似的问题。但是，如果使用变体类型，你可以轻松管理这些情况：

```
type a = [`Foo of int | `Bar of bool]
type b = [`Baz of char | `Quz]
type a_or_b = [a | b]

```

很好！请注意，我们并未使用变体类型的完整通用性：我只会在`a`、`b`或`a_or_b`的上下文中引用这些变体构造函数。这可以避免强制转换的混乱。

我实际上可以在 Haskell 中使用 GADTs 完成这个，尽管对于初学者来说显然不明显：

```
data A
data B
data AorB t where
  Foo :: Int -> AorB A
  Bar :: Bool -> AorB A
  Baz :: Char -> AorB B
  Quz :: AorB B

```

要匹配所有构造函数，我指定类型`AorB t`；要仅匹配`A`，我使用`AorB A`；要仅匹配`B`，我使用`AorB B`。别问我如何指定超过两个组合和类型的任意子集。（评论区中的解决方案欢迎，但它们的清晰度将会评分。）

Haskell 的方法确实有一个优势，即和类型仍然是封闭的。由于 OCaml 不能做出这样的保证，像`bin-prot`这样的东西需要使用完整的四字节来指定变体类型（它们对名称进行哈希并将其用作唯一标识符），而不是这里所需的两位（但更可能是一个字节）。这也意味着更有效的生成代码。
