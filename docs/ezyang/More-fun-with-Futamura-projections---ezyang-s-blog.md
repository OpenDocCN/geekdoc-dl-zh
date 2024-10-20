<!--yml

category: 未分类

date: 2024-07-01 18:18:23

-->

# 更多关于 Futamura 投影的乐趣：ezyang's 博客

> 来源：[`blog.ezyang.com/2010/03/more-fun-with-futamura-projections/`](http://blog.ezyang.com/2010/03/more-fun-with-futamura-projections/)

*Anders Kaseorg 编写的代码。*

在[Doctor Futamura 的三个投影](http://blog.sigfpe.com/2009/05/three-projections-of-doctor-futamura.html)中，Dan Piponi 向非程序员解释了 Futamura 投影，这是部分求值的一系列令人费解的应用。如果你还没有读过，请去看一看；这篇文章旨在成为那篇文章的精神继承者，在这篇文章中我们将编写一些 Haskell 代码。

*铸币的图像类型。* 在原始文章中，Piponi 绘制出各种硬币、模板或其他机器作为输入，并输出硬币或机器。让我们用更像 Haskell 类型的东西来重新定义这个定义。

首先，来点简单的：第一个机器，它接收空白硬币并铸造新硬币。

现在我们使用箭头来表示输入输出关系。事实上，这只是一个将空白硬币作为输入并输出镌刻硬币的函数。我们可以用以下类型同义词来概括这个概念：

```
> type Machine input output = input -> output

```

那么我们让我们输入硬币的描述是什么呢？好吧，首先我们需要一个简单的数据类型来表示这个描述：

```
> data Program input output = Program

```

(是的，这种数据类型真的不能做任何有趣的事情。我们实际上不会为这些机器编写实现。) 从这里开始，我们有了我们下一个"类型化"的解释器的图片：

或者，在代码中表示为：

```
> type Interpreter input output = (Program input output, input) -> output

```

从那里开始，看看编译器是什么样子的也不是难事：

```
> type Compiler input output = Program input output -> Machine input output

```

我想指出，我们完全可以像这样完全写出这个类型：

```
type Compiler input output = Program input output -> (input -> output)

```

我们故意保留了不必要的括号，因为 Haskell 诱人地表明你可以把`a -> b -> c`当作一个二元函数处理，而我们希望它与`(a, b) -> c`保持不同。

最后，我们有了专用程序：

```
> type Specializer program input output =
>     ((program, input) -> output, program) -> (input -> output)

```

我们已经用富有启发性的方式命名了我们的 Specializer 类型同义词中的变量，但程序不仅仅是程序：Futamura 投影的整个要点是我们可以在那里放置不同的东西。另一个有趣的事情是，任何给定的 Specializer 都需要在输入和输出之外还根据它操作的程序进行参数化。这意味着 Specializer 假设的具体类型因实际上让`program`变化而变化。它不依赖于 Specializer 的第一个参数，这是由`program`、`input`和`output`强制的`（program，input）-> output`。

那么，这些具体类型是什么呢？对于这个任务，我们可以问问 GHC。

*到第四个投影，以及更远的地方！* 首先，几个准备工作。我们保留了`input`和`output`在我们的类型同义词中完全一般化，但实际上我们应该用具体的数据类型来填充它们。还有一些更空洞的定义：

```
> data In = In
> data Out = Out
>
> type P = Program In Out
> p :: P
> p = undefined
>
> type I = Interpreter In Out
> i :: I
> i = undefined

```

我们实际上不关心如何实现我们的程序或解释器，因此 `undefined`；考虑到我们的虚无数据定义，确实存在这些的有效实例，但它们并不特别增加洞察力。

```
> s :: Specializer program input output
> -- s (x, p) i = x (p, i)
> s = uncurry curry

```

我们对待专用化器有点不同：部分求值和部分应用非常相似：事实上，对外部用户来说，它们确实做着完全相同的事情，只是部分求值最终更快，因为它实际上在做一些工作，而不是形成一个闭包，中间参数无所作为地挂在空中。然而，我们需要取消柯里化，因为 Haskell 函数默认情况下是柯里化的。

现在，Futamura 投影：

```
> type M = Machine In Out
> m :: M
> m = s1 (i, p)

```

没有单态限制，`s` 也可以正常工作，但我们将很快为 `s1` 给出一个显式类型，这将破坏其余投影的乐趣。（实际上，因为我们给 `s` 指定了显式类型，单态限制不适用。）

那么，`s1` 的类型是什么？它绝对不是通用的：`i` 和 `p` 完全明确，并且专门化器不会引入任何其他多态类型。这应该很容易判断，但我们还是问问 GHC 以防万一：

```
Main> :t s1
s1 :: ((P, In) -> Out, P) -> In -> Out

```

当然。它与我们的变量名匹配！

```
> type S1 = Specializer P In Out
> s1 :: S1
> s1 = s

```

是时候进行第二个 Futamura 投影了：

```
> type C = Compiler In Out
> c :: C
> c = s2 (s1, i)

```

请注意，这次我写了 `s2`。那是因为 `` s1 (s1, i)`` 无法通过类型检查；如果你进行统一，你会看到具体类型不匹配。那么 `s2` 的具体类型是什么？稍微多想一会儿，或许快速浏览一下 Piponi 的文章就能阐明答案了。

```
> type S2 = Specializer I P M
> s2 :: S2
> s2 = s

```

第三个 Futamura 投影，解释器到编译器机器：

```
> type IC = I -> C
> ic :: IC
> ic = s3 (s2, s1)

```

（你应该验证 `s2 (s2, s1)` 和 `s1 (s1, s2)` 以及任何其排列都不能通过类型检查。）我们也设法丢失了与具体性的直接基础：看不到 `p` 或 `i`。但 `s2` 和 `s1` 明显是具体类型，正如我们之前展示的那样，而 GHC 可以为我们执行统一：

```
Main> :t s3
s3 :: ((S1, I) -> C, S1) -> I -> Program In Out -> In -> Out

```

事实上，它已经很友好地用相关类型同义词替换了一些更加棘手的类型供我们使用。如果我们加入一些额外的括号并只获取输出：

```
I -> (Program In Out -> (In -> Out))

```

这就是我们的解释器到编译器机器！

```
> type S3 = Specializer S1 I C
> s3 :: S3
> s3 = s

```

但为什么停在这里呢？

```
> s1ic :: S1 -> IC
> s1ic = s4 (s3, s2)
>
> type S4 = Specializer S2 S1 IC
> s4 :: S4
> s4 = s

```

或者甚至在这里？

```
> s2ic :: S2 -> (S1 -> IC)
> s2ic = s5 (s4, s3)
>
> type S5 = Specializer S3 S2 (S1 -> IC)
> s5 :: S5
> s5 = s
>
> s3ic :: S3 -> (S2 -> (S1 -> IC))
> s3ic = s6 (s5, s4)
>
> type S6 = Specializer S4 S3 (S2 -> (S1 -> IC))
> s6 :: S6
> s6 = s

```

我们可以继续，使用我们用于 *n-1* 和 *n-2* 投影的专用化器构造第 *n* 个投影。

这可能看起来像一堆类型奇技淫巧。我认为不仅仅是这样。

部分求值器的实现者关心，因为这代表了部分求值器组合的机制。`S2` 和 `S1` 可能是不同类型的专用化器，各自具有其优势和劣势。这也是部分求值器编写者面临的哲学挑战的生动示范：他们需要编写一段可以在 *Sn* 中任意 *n* 上工作的代码。也许在实践中，它只需要在低 *n* 上表现良好，但它确实能够工作是一个令人印象深刻的技术成就。

对于部分应用的信徒来说，这有点像是一种客厅戏法：

```
*Main> :t s (s,s) s
s (s,s) s
  :: ((program, input) -> output) -> program -> input -> output
*Main> :t s (s,s) s s
s (s,s) s s
  :: ((input, input1) -> output) -> input -> input1 -> output
*Main> :t s (s,s) s s s
s (s,s) s s s
  :: ((input, input1) -> output) -> input -> input1 -> output
*Main> :t s (s,s) s s s s
s (s,s) s s s s
  :: ((input, input1) -> output) -> input -> input1 -> output
*Main> :t s (s,s) s s s s s
s (s,s) s s s s s
  :: ((input, input1) -> output) -> input -> input1 -> output

```

但这是一个有用的客厅戏法：我们设法使一个任意可变参数的函数！我相信这种技术在某些地方被野生使用，尽管在撰写本文时，我找不到任何实际的例子（Text.Printf 可能有，尽管很难将其与它们的类型类技巧区分开来）。
