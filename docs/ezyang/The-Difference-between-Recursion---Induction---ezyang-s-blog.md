<!--yml

category: 未分类

date: 2024-07-01 18:17:21

-->

# 递归与归纳的区别：ezyang 的博客

> 来源：[`blog.ezyang.com/2013/04/the-difference-between-recursion-induction/`](http://blog.ezyang.com/2013/04/the-difference-between-recursion-induction/)

递归和归纳密切相关。当你在初级计算机科学课程中首次学习递归时，你可能被告知使用归纳来证明你的递归算法是正确的。（为了本文的目的，让我们排除像[Collatz 猜想](http://en.wikipedia.org/wiki/Collatz_conjecture)中那样不明显终止的复杂递归函数。）归纳看起来非常像递归：这种相似性来自于归纳假设看起来有点像你正在证明的定理的“递归调用”的结果。如果一个普通的递归计算返回普通的值，你可能会想知道一个“归纳计算”是否返回证明项（根据柯里-霍华德对应，可以将其视为一个值）。

然而，事实证明，当你从范畴论的角度来看递归和归纳时，它们并不等价！直观地说，区别在于当你进行归纳时，你进行归纳的数据类型（例如数字）出现在*类型*级别，而不是术语级别。用范畴论者的话来说，递归和归纳都有关联的初等代数，但载体集和 endo 函子是不同的。在这篇博文中，我希望精确阐明递归和归纳之间的区别。不幸的是，我需要假设读者*某些*对初等代数的了解：如果你不知道折叠和初等代数之间的关系，请查看这篇[列表在初等代数形式中的导出](http://blog.ezyang.com/2012/10/duality-for-haskellers/)。

处理广义抽象无意义时，最重要的第一步是使用具体示例！因此，让我们选择最简单的非平凡数据类型之一：自然数（我们的示例在可能的情况下以 Coq 和 Haskell 编写）：

```
Inductive nat : Set := (* defined in standard library *)
  | 0 : nat
  | S : nat -> nat.

data Nat = Z | S Nat

```

自然数是一个很好的例子：即使是[F-代数的维基百科文章](http://en.wikipedia.org/wiki/F-algebra)也使用它们。简言之，一个 F-代数（有时简称为“代数”）有三个组成部分：一个（endo）函子`f`，一个类型`a`和一个减少函数`f a -> a`。对于自然数的简单递归，我们需要定义一个生成自然数的函子`NatF`；然后我们的类型`a`是`Nat`，减少函数是类型`NatF Nat -> Nat`。该函子定义如下：

```
Inductive NatF (x : Set) : Set :=
  | F0 : NatF x.
  | FS : x -> NatF x.

data NatF x = FZ | FS x

```

本质上，取原始定义，但用多态变量替换任何类型的递归出现。作为练习，展示`NatF Nat -> Nat`是存在的：它是`() -> Nat`和`Nat -> Nat`的（共）积。这个代数的初始性意味着对于任意类型`x`的`NatF x -> x`的函数可以在`Nat -> x`的折叠中使用：这个折叠是从初始代数(`NatF Nat -> Nat`)到另一个代数(`NatF x -> x`)的同态。关键是自然数的初始代数包括了一个关于**集合**的自函子。

现在让我们来看看归纳的 F-代数。作为第一次尝试，让我们尝试使用相同的 F-代数，并看看是否存在与“归纳类型”相适应的适当同态（这里我们只能用 Coq 编写，而不能用 Haskell）。假设我们试图证明某个命题`P : nat -> Prop`对所有自然数都成立；那么最终证明项的类型必须是`forall n : nat, P n`。现在我们可以写出代数的态射：`NatF (forall n : nat, P n) -> forall n : nat, P n`。但这个“归纳原理”既是无意义的，也不是真的：

```
Hint Constructors nat NatF.
Goal ~ (forall (P : nat -> Prop), (NatF (forall n : nat, P n) -> forall n : nat, P n)).
  intro H; specialize (H (fun n => False)); auto.
Qed.

```

（旁注：你可能会说这个证明失败了，因为我提供了一个在所有自然数上都为假的谓词。但归纳仍然“有效”，即使你试图证明的谓词是假的：你应该在尝试提供基础情况或归纳假设时失败！）

我们退后一步，现在想知道，“那么，正确的代数是什么？”很明显，我们的自函子是错误的。幸运的是，我们可以通过检查自然数归纳原理的类型来得出正确自函子的线索：

```
(* Check nat_ind. *)
nat_ind : forall P : nat -> Prop,
  P 0 -> (forall n : nat, P n -> P (S n)) -> forall n : nat, P n

```

`P 0`是基础情况的类型，`forall n : nat, P n -> P (S n)`是归纳情况的类型。就像我们为自然数定义了`NatF nat -> nat`一样，它是`zero : unit -> nat`和`succ : nat -> nat`的组合，我们需要定义一个单一的函数，它结合了基础情况和归纳情况。这似乎很困难：结果类型并不相同。但依赖类型来拯救：我们正在寻找的类型是：

```
fun (P : nat -> Prop) => forall n : nat, match n with 0 => True | S n' => P n' end -> P n

```

你可以这样阅读这个类型：我将为任意的`n`给你一个类型为`P n`的证明对象。如果`n`是 0，我将为你提供这个证明对象而不需要进一步的帮助（`True -> P 0`）。然而，如果`n`是`S n'`，我将要求你提供`P n'`（`P n' -> P (S n')`）。

我们快要接近了。如果这是一个初始代数的态射，那么函子`IndF`必须是：

```
fun (P : nat -> Prop) => forall n : nat, match n with 0 => True | S n' => P n' end

```

这个函子是什么类别上的？不幸的是，这篇文章和我的大脑都没有足够的空间来进行严格的处理，但大致上可以将该类别视为自然数索引的命题。这个类别的对象形式为`forall n : nat, P n`，类别的态射形式为`forall n : nat, P n -> P' n`。[1] 作为练习，展示恒等和复合存在，并遵守适当的法则。

即将发生一些惊人的事情。我们已经定义了我们的函子，并且现在正在寻找初始代数。就像对自然数的情况一样，初始代数由函子的最小不动点定义：

```
Fixpoint P (n : nat) : Prop :=
  match n with 0 => True | S n' => P n' end.

```

但这只是 `True`！

```
Hint Unfold P.
Goal forall n, P n = True.
  induction n; auto.
Qed.

```

绘制我们的图表：

我们范畴的代数（向下箭头）对应于归纳论证。因为我们的态射形式为 `forall n, P n -> P' n`，所以不能仅仅从 `forall n, P n` 得出 `forall n, P' n`；然而，初始代数的存在意味着当我们有一个代数 `forall n, IndF n -> P n` 时，`True -> forall n, P n`。令人惊叹！（顺便提一下，Lambek 引理表明 `Mu P` 同构于 `P (Mu P)`，因此初始代数实际上是*非常非常*平凡的。）

总结：

+   **自然数递归**涉及到与函子 `unit + X` 对应的 F-代数，这些代数定义在**集合**范畴上。这个函子的最小不动点是自然数，而由初始代数诱导的态射对应于*折叠*。

+   **自然数归纳**涉及到与函子 `fun n => match n with 0 => True | S n' => P n'` 对应的 F-代数，这些代数定义在自然数索引命题的范畴上。这个函子的最小不动点是 `True`，而由初始代数诱导的态射*确立了归纳证明命题的真实性*。

所以，下次有人问你归纳和递归的区别是什么，告诉他们：*归纳只是由索引命题上的初始代数诱导的唯一同态，有什么问题吗？*

* * *

特别感谢 Conor McBride，在 ICFP 会议上向我解释了这个问题。我答应要写博客，但是忘记了，最终不得不重新推导一遍。

[1] 关于态射的另一个合理表述是 `(forall n : nat, P n) -> (forall n : nat, P' n)`。然而，在这个范畴中的态射太过*强大*：它们要求你对所有的*n*去证明结果… 这需要归纳，但这种方式并不是重点。此外，这个范畴是命题的普通范畴的子范畴。
