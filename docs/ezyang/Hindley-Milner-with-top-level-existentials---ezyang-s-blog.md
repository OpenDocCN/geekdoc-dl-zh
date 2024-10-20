<!--yml

category: 未分类

date: 2024-07-01 18:17:06

-->

# Hindley-Milner with top-level existentials：ezyang's 博客

> 来源：[`blog.ezyang.com/2016/04/hindley-milner-with-top-level-existentials/`](http://blog.ezyang.com/2016/04/hindley-milner-with-top-level-existentials/)

*内容警示：这是一篇半成品的研究文章。*

**摘要.** 存在类型的顶层解包很容易集成到 Hindley-Milner 类型推断中。Haskell 应该支持它们。这个想法也可能适用于存在的内部绑定（例如 F-ing 模块），但我还没有想出如何实现。

**更新.** 而 UHC 是第一个做到这一点的！

**更新 2.** 并且 rank-2 类型推断是可判定的（而 rank-1 的存在类型是一个更弱的系统），尽管 rank-2 推断的算法需要半一致化。

### 背景

**Hindley-Milner 与 System F 的区别。** 尽管在非正式讨论中，Hindley-Milner 常被描述为“类型推断算法”，但其实它应该被正确地描述为比 System F 更为限制的类型系统。两种类型系统都通过变量的全称量化来实现多态性，但在 System F 中，这种多态性是显式的，并且可以出现在任何地方；而在 Hindley-Milner 中，多态性是隐式的，只能发生在“顶层”（在所谓的“多态类型”或“类型方案”中）。这种多态性的限制是使得 Hindley-Milner 的推断（通过算法 W）成为可判定（和实际可行）的关键，而 System F 的推断则是不可判定的。

```
-- Hindley Milner
id :: a -> a
id = λx. x

-- System F
id :: ∀a. a -> a
id = Λa. λ(x : a). x

```

**System F 中的存在类型。** System F 的一个常见泛化是配备存在类型：

```
Types  τ ::= ... | ∃a. τ
Terms  e ::= ... | pack <τ, e>_τ | unpack <a, x> = e in e

```

在 System F 中，从技术上讲，不需要将存在类型作为原始概念添加进来，因为它们可以通过使用全称量词编码来实现，比如说 `∃a. τ = ∀r. (∀a. τ → r) → r`。

**Hindley-Milner 中的存在类型？** 这种策略对 Hindley-Milner 不起作用：编码需要更高阶的类型，而这恰恰是 Hindley-Milner 为了推断而排除的。

无论如何，试图推断存在类型是一个愚蠢的游戏：没有最佳类型！HM 总是为表达式推断出*最*一般的类型：例如，我们会推断 `f :: a -> a` 对于函数 `f = \x -> x`，而不是 `Int -> Int`。但数据抽象的整个点是选择一个更抽象的类型，这不会是最一般的类型，因此不会是唯一的。什么应该是抽象的，什么应该是具体的？只有用户知道。

**Haskell 中的存在类型。** 假设我们愿意在打包存在类型时写下显式类型，Hindley-Milner 是否能完成程序中其余类型的完整且可判定的推断呢？

Haskell 是一个存在（咳咳）证明，这是可行的。实际上，有两种方法可以实现它。第一种是当你搜索“Haskell 存在类型”时会看到的内容。

```
{-# LANGUAGE ExistentialQuantification #-}
data Ex f = forall a. Ex (f a)
pack :: f a -> Ex f
pack = Ex
unpack :: Ex f -> (forall a. f a -> r) -> r
unpack m k = case m of Ex x -> f x

```

`Ex f` 等同于 `∃a. f a`，类似于 System F 语法，它们可以通过`Ex`构造函数打包，并通过模式匹配解包。

第二种方法是直接使用 Haskell 对于秩-n 类型的支持使用 System F 编码：

```
{-# LANGUAGE RankNTypes #-}
type Ex f = forall r. (forall a. f a -> r) -> r
pack :: f a -> Ex f
pack x = \k -> k x
unpack :: Ex f -> (forall a. f a -> r) -> r
unpack m k = m k

```

[盒子类型论文](http://research.microsoft.com/pubs/67445/boxy-icfp.pdf)展示了你*可以*进行推断，只要你的所有高阶类型都有注解。尽管如此，或许事情并不像希望的那样简单，因为不可预测的类型是 GHC 类型检查器中常见的 bug 源。

### 问题

**显式解包很糟糕。** 正如任何试图在 Haskell 中使用存在类型编程的人所能证明的那样，由于需要在使用之前对存在类型进行*解包*（即对其进行模式匹配），使用存在类型仍然可能相当笨拙。也就是说，语法`let Ex x = ... in ...`是不允许的，这是让 GHC 告诉你它的大脑爆炸的简单方法。

[Leijen](http://research.microsoft.com/en-us/um/people/daan/download/papers/existentials.pdf)研究了如何处理存在类型*无需*显式解包。

**没有显式解包会导致主类型的丢失，以及 Leijen 的解决方案。** 不幸的是，天真的类型系统没有主类型。Leijen 给出了一个没有主类型的例子：

```
wrap :: forall a. a -> [a]
key  :: exists b. Key b
-- What is the type of 'wrap key'?
-- [exists b. Key b]?
-- exists b. [key b]?

```

两种类型都不是对方的子类型。在他的论文中，Leijen 建议应尽可能晚地解包存在类型（因为你可以从第一种类型到第二种类型，但反之则不行），因此应优先选择第一种类型。

### 解决方案

**另一种方法。** 如果我们总是将存在类型提升到顶层会怎样？如果你将解包限制在程序的顶层，这实际上是非常容易做到的，并且结果非常好。

**每个顶层的 Haskell 代数数据类型中都有一个存在类型。** 首先，我想说服你这并不是一个那么奇怪的想法。为了做到这一点，我们看一下 Haskell 对代数数据类型的支持。Haskell 中的代数数据类型是*生成的*：每个数据类型必须有一个顶层声明，并且被认为是与任何其他数据类型不同的独立类型。事实上，Haskell 用户利用这种生成性与隐藏构造子的能力来实现数据抽象。尽管实际上并没有存在类型潜藏其中——生成性并不是数据抽象——生成性是数据抽象的一个重要部分，并且 HM 对此没有任何问题。

**顶层生成性对应于在程序的顶层解包的存在量词（类似于 F-ing 模块）。** 我们不需要存在于 Haskell 表达式内部来支持代数数据类型的生成性：我们所需要的只是在顶层包装一个存在类型，然后立即将其解包到顶层上下文中。事实上，F-ing 模块甚至走得更远：存在量词可以始终被提升，直到它们达到程序的顶层。使用适用函子（ML 类型）进行模块化编程可以通过立即解包作为其定义的顶层存在来*编码*。

**建议。** 因此，让我们建议以下类型系统，带有顶层存在量词（其中`a*`表示零到多个类型变量）：

```
Term variables ∈ f, x, y, z
Type variables ∈ a, b, c

Programs
prog ::= let f = e in prog
       | seal (b*, f :: σ) = (τ*, e) in prog
       | {- -}

Type schemes (polytypes)
σ ::= ∀a*. τ

Expressions
e ::= x
    | \x -> e
    | e e

Monotypes
τ ::= a
    | τ -> τ

```

有一个新的顶层绑定形式，`seal`。我们可以给出以下类型规则：

```
Γ ⊢ e :: τ₀[b* → τ*]
a* = free-vars(τ₀[b* → τ*])
Γ, b*, (f :: ∀a*. τ₀) ⊢ prog
---------------------------------------------
Γ ⊢ seal (b*, f :: ∀a*. τ₀) = (τ*, e) in prog

```

它还可以直接展开为带有存在量词的 System F：

```
seal (b*, f :: σ) = (τ*, e) in prog
  ===>
unpack <b*, f> = pack <τ*, e>_{∃b*. σ} in prog

```

几点观察：

1.  在 HM 的传统呈现中，let 绑定允许嵌套在表达式内部（并且在添加到上下文之前被泛化为多态类型）。我们是否可以类似地处理`seal`？这应该是可能的，但绑定的存在量词类型变量必须向上传播。

1.  这导致了第二个问题：天真地，量词的顺序必须是`∃b. ∀a. τ`而不是`∀a. ∃b. τ`，否则我们无法将存在量词添加到顶层上下文。然而，存在“斯克莱姆化”技巧（参见 Shao 和 F-ing 模块），通过这种方式你可以将`b`作为一个高阶类型变量，以`a`作为参数，例如，`∀a. ∃b. b`等价于`∃b'. ∀a. b' a`。这个技巧可以作为支持内部`seal`绑定的方式，但编码往往相当复杂（因为你必须闭合整个环境）。

1.  这条规则并不适用于直接建模机器学习模块，因为“模块”通常被认为是多态函数的记录。也许你可以将这条规则概括为绑定多个多态函数？

**结论。** 到此为止，这就是我所想出来的。我希望有人能告诉我（1）谁已经提出了这个想法，以及（2）为什么它不起作用。
