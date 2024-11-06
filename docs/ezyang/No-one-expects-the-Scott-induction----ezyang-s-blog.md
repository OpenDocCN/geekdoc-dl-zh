<!--yml

category: 未分类

date: 2024-07-01 18:18:00

-->

# 没有人预料到斯科特归纳！: ezyang’s 博客

> 来源：[`blog.ezyang.com/2010/12/no-one-expects-the-scott-induction/`](http://blog.ezyang.com/2010/12/no-one-expects-the-scott-induction/)
> 
> 新来这个系列？请从[最开始！](http://blog.ezyang.com/2010/12/hussling-haskell-types-into-hasse-diagrams/)开始。

递归可能是你学习函数式编程（或者说计算机科学，希望如此）时首先了解的概念之一。经典的例子是阶乘：

```
fact :: Int -> Int
fact 0 = 1 -- base case
fact n = n * fact (pred n) -- recursive case

```

自然数上的递归与自然数上的归纳密切相关，如[这里解释的](http://scienceblogs.com/goodmath/2007/01/basics_recursion_and_induction_1.php)。

有趣的一点是，Haskell 中数据类型 `Int` 没有涉及到无穷大，因此这个定义在严格语言和惰性语言中都能完美工作。（记住 `Int` 是一个扁平的数据类型。）然而，请考虑一下，我们之前正在玩耍的 `Omega`，它确实有一个无穷大！因此，我们还需要展示，当阶乘传入无穷大时也会产生一些合理的结果：它输出无穷大。幸运的是，阶乘的定义对于 Omega 类型也是完全相同的（鉴于适当的类型类）。但是它为什么有效呢？

一种操作性的回答是，程序的任何执行都只能处理有限数量：我们永远不能真正“看到”类型 Omega 的值是无穷大。因此，如果我们将一切都限制在某个大数之下（比如，我们计算机的 RAM），我们可以使用同样的推理技术来处理 `Int`。然而，我希望你对这个答案感到深感不满：你想要*将*无限数据类型想象为无限的，即使实际上你永远也不会需要无穷大。这是自然和流畅的推理方式。事实证明，还有一个归纳原理与之对应：超越归纳。

自然数上的递归 - 归纳

Omega 上的递归 - 超越归纳

Omega 或许并不是一个非常有趣的数据类型，它具有无限的值，但在 Haskell 中有许多无限数据类型的例子，无限列表就是其中一个特别的例子。因此，实际上，我们可以将有限和无限情况推广到任意数据结构，如下所示：

在有限数据结构上的递归 - 结构归纳

无限数据结构上的递归 - 斯科特归纳

斯科特归纳是关键：有了它，我们有一个多功能的工具来推理惰性语言中递归函数的正确性。然而，它的定义可能有点难以理解：

> 让 D 是一个 cpo。D 的一个子集 S 如果是链闭的，则对于 D 中的所有链，如果链中的每个元素都在 S 中，则链的上确界也在 S 中。如果 D 是一个 domain，一个子集 S 如果既是链闭的又包含 bottom，则它是可接受的。斯科特的不动点归纳原理表明，要证明 fix(f) 在 S 中，我们只需证明对于 D 中的所有 d，如果 d 在 S 中，则 f(d) 也在 S 中。

当我第一次学习斯科特归纳时，我不明白为什么所有这些可接受性的东西是必要的：有人告诉我，这些是“使归纳原理生效所必需的东西”。最后，我也认同了这个观点，但是要在其全面性上看清楚还是有点困难。

因此，在本文中，我们将展示从自然数归纳到超限归纳的跃迁是如何对应从结构归纳到斯科特归纳的跃迁。

* * *

*自然数归纳.* 这是你在小学学到的归纳法，也许是最简单的归纳形式。简单来说，它规定如果某个性质对 n = 0 成立，并且如果某个性质对 n + 1 成立，那么它对所有自然数都成立。

把基本情况和归纳步骤看作推理规则的一种方式是：我们需要证明它们是正确的，如果它们是正确的，我们就得到了另一个推理规则，让我们能够避开无限应用归纳步骤来满足我们对所有自然数成立的性质的要求。（请注意，如果我们只想证明某个性质对一个任意自然数成立，那只需要有限次应用归纳步骤！）

*Omega 上的超限归纳.* 回忆一下，Omega 是自然数加上最小的无限序数 ω。假设我们想要证明某个性质对所有自然数以及无穷大都成立。如果我们仅仅使用自然数归纳，我们会注意到我们可以证明某个有限自然数具有某个性质，但未必对无穷大成立（例如，我们可能会得出每个自然数都有比它大的另一个数，但在 Omega 中大于无穷大的值却不存在）。

这意味着我们需要一个情况：如果一个性质对所有自然数成立，那么它也对 ω 成立。然后我们可以应用自然数归纳，并推断该性质对无穷大也成立。

在 Omega 上的超限归纳需要证明的情况比自然数归纳多得多，因此能够得出更强的结论。

> *旁白。* 在其全面性中，我们可能有许多无限序数，因此第二种情况推广到*继承者序数*（例如添加 1），而第三种情况推广到极限序数（即，不能通过有限次应用继承者函数达到的序数——例如从零到无穷大）。这听起来熟悉吗？希望是的：这种极限的概念应该让你想起链的最小上界（事实上，ω 是域 Omega 中唯一非平凡链的最小上界）。

* * *

让我们再次看一下 Scott 归纳的定义：

> 让 D 是一个 cpo。D 的子集 S 是链闭的，当且仅当对于 D 中的所有链，如果链中的每个元素都在 S 中，则链的最小上界也在 S 中。如果 D 是一个域，子集 S 是可接受的，如果它是链闭的并且包含底部。Scott 的不动点归纳原理表明，要证明 fix(f) 在 S 中，我们只需证明对于所有 d 属于 D，如果 d 属于 S，则 f(d) 也属于 S。

现在我们可以找出与这个定义中的语句对应的超限归纳的部分。S 对应于具有我们想要展示属性的值的集合，因此 `S = {d | d in D and prop(d)}`。*基础情况* 是底部包含在 S 中。*继承者情况* 是“如果 d 属于 S，则 f(d) 属于 S”（注意现在 *f* 是我们的继承者函数，而不是加一）。*极限情况* 对应于链闭条件。

这里是我们需要展示的所有推理规则！

我们用于证明阶乘在 Omega 上正确的域 D 是函数 `Omega -> Omega` 的域，继承者函数是 `(Omega -> Omega) -> (Omega -> Omega)`，而子集 S 对应于阶乘不断定义版本的链。有了所有这些要素，我们可以看到 `fix(f)` 确实是我们要找的阶乘函数。

* * *

Scott 归纳法有许多有趣的“怪癖”。其中一个是这个属性必须对底部成立，这是一个部分正确性结果（“如果程序终止，则如此如此成立”），而不是一个完全正确性结果（“程序终止且如此如此成立”）。另一个是继承者情况通常不是涉及 Scott 归纳的证明中最困难的部分：显示属性的可接受性是。

这结束了我们关于指称语义的系列。这并不是完整的：通常接下来要看的是一个称为 PCF 的简单函数式编程语言，然后将这种语言的操作语义和指称语义联系起来。但即使你决定不想再听有关指称语义的更多内容，我希望这些对这个迷人世界的一瞥能帮助你在 Haskell 程序中思考惰性。

*后记.* 最初我想将所有这些归纳形式与 TAPL 中提出的广义归纳联系起来：归纳原理是单调函数 F : P(U) -> P(U)（这里 P(U) 表示宇宙的幂集）的最小不动点是 U 的所有 F-闭子集的交集。但这导致了一个非常有趣的情况，即函数的最大不动点需要接受值的集合，而不仅仅是单个值。我对此并不确定应该如何解释，所以我将其略过了。

无关的是，出于教学目的，也很好有一个由于错误（但似乎合理）应用斯科特归纳而产生的“悖论”。可惜，在我写作时，这样的例子让我无法捉摸。