<!--yml

类别：未分类

日期：2024-07-01 18:16:50

-->

# 动态作用域是一种效果，隐式参数是一种共效：ezyang 的博客

> 来源：[`blog.ezyang.com/2020/08/dynamic-scoping-is-an-effect-implicit-parameters-are-a-coeffect/`](http://blog.ezyang.com/2020/08/dynamic-scoping-is-an-effect-implicit-parameters-are-a-coeffect/)

长久以来，我一直认为[隐式参数](https://downloads.haskell.org/~ghc/latest/docs/html/users_guide/glasgow_exts.html#implicit-parameters)和动态作用域基本上是相同的东西，因为它们都可以用来解决类似的问题（例如所谓的“配置问题”，其中你需要将某些配置深入到函数定义的嵌套体中而不显式地定义它们）。但隐式参数却有一个不被推荐使用的名声（[使用反射代替](https://www.reddit.com/r/haskell/comments/5xqozf/implicit_parameters_vs_reflection/dek9eqg/)），而通过读者单子进行动态作用域是一个有用且被充分理解的构造（除了你需要将一切变成单子的那一点）。为什么会有这样的差异？

[Oleg](http://okmij.org/ftp/Computation/dynamic-binding.html#implicit-parameter-neq-dynvar) 指出隐式参数并不真正是动态作用域，并且举了一个例子，展示了 Lisp 和 Haskell 在此问题上的分歧。而你甚至不希望在 Haskell 中出现 Lisp 的行为：如果你考虑动态作用域的操作概念（沿着堆栈向上走，直到找到动态变量的绑定位置），它与惰性计算并不兼容，因为一个 thunk（访问动态变量）将在程序执行的某个不可预测的点被强制执行。你确实不想要去思考 thunk 将在何处执行以确定它的动态变量将如何绑定，这样做只会导致疯狂。但在严格语言中，没人会觉得需要去理解动态作用域应该如何运行（[好吧](https://blog.klipse.tech/clojure/2018/12/25/dynamic-scope-clojure.html)，[大多数情况下](https://stuartsierra.com/2013/03/29/perils-of-dynamic-scope)--稍后会更详细说明）。

研究界已经发现，隐式参数与**附效**有所不同。我相信这最初是在[Tomas Petricek](http://tomasp.net/academic/papers/coeffects/coeffects-icalp.pdf)的研究中首次观察到的（更现代的呈现在[Tomas Petricek](https://www.doc.ic.ac.uk/~dorchard/publ/coeffects-icfp14.pdf); 更 Haskelly 的呈现可以在[Tomas Petricek](http://tomasp.net/academic/papers/haskell-effects/haskell-effects.pdf)找到）。然而，Tomas 在 2012 年[在我的博客上发表评论](http://blog.ezyang.com/2012/10/generalizing-the-programmable-semicolon/)，探讨了类似的想法，所以这可能已经在酝酿了一段时间了。关键点是，对于一些附效（即隐式参数），按名调用的减少保持类型和附效，因此隐式参数不会像动态作用域（一种效果）那样在使用时出现问题。这些肯定有不同的行为方式！类型类也是附效，这就是为什么 Haskell 中现代隐式参数的使用明确承认了这一点（例如，在反射包中）。

在今年的 ICFP 上，我看到了有关[Koka 中隐式值和函数](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/implicits-tr-v2.pdf)的有趣技术报告，这是动态作用域的一个新变化。我不禁想到了 Haskell 隐式参数可能从这项工作中学到一些东西。隐式值明智地选择在顶层全局定义隐式值，以便它们可以参与正常的模块命名空间，而不是一组没有命名空间的动态作用域名称（这也是反射在隐式参数上的改进）。但实际上，隐式函数似乎正在借鉴隐式参数的一部分！

最大的创新在于隐式函数，它解决了函数中所有动态引用（不仅仅是词法上，而是所有后续的动态调用）到词法范围（函数定义时的动态范围）的问题，生成一个函数，它不依赖于隐式值（也就是说，没有*效果*表明在调用函数时必须定义隐式值）。这正是隐式参数`let ?x = ...`绑定会做的事情，在定义时直接为隐式函数填充字典，而不是等待。非常具有上下文意识！（当然，Koka 使用代数效应实现了这一点，并通过非常简单的转换得到了正确的语义）。结果并不完全是动态作用域，但正如 TR 所述，它导致更好的抽象。

很难想象隐式值/函数如何重新进入 Haskell，至少不是在某种序列构造（例如，一个单子）潜伏的情况下。尽管隐式函数的行为很像隐式参数，但其余的动态作用域（包括隐式函数本身的绑定）仍然是良好的旧有效果动态作用域。你不能在 Haskell 中轻易实现这一点，因为这会破坏在 Beta-还原和 Eta-扩展下的类型保持性。Haskell 别无选择，只能*走到底*，一旦你超越了隐式参数的明显问题（这是反射修复的），事情似乎大部分可以解决。
