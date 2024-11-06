<!--yml

category: 未分类

date: 2024-07-01 18:17:40

-->

# *显然正确*：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/10/obviously-correct/`](http://blog.ezyang.com/2011/10/obviously-correct/)

什么是自动内存管理、静态类型和纯度的共同点？它们是利用我们可以通过视觉检查使程序在某种程度上*显然正确*（某种正确性的部分定义）的方法。使用自动内存管理的代码对于一类内存错误来说是*显然正确*的。使用静态类型的代码对于一类类型错误来说是*显然正确*的。使用纯度（无可变引用或副作用）的代码对于一类并发错误来说是*显然正确*的。当我利用这些技术中的任何一种时，我不必*证明*我的代码没有错误：它只是自动地是！

不幸的是，这里有一个限制。所有这些“显然正确”的方法论所要求你做的是在它们的祭坛上牺牲不同程度的表现力。不能再使用指针技巧了。不能再随意处理数据表示了。不能再有变异了。如果这种表现力实际上是大多数人并不真正想要的（比如内存管理），那么它很乐意被交换掉。但如果这是他们*想要*的东西，作为语言设计者，我们正在使他们做自己想做的事情更加困难，当他们拿起火把和干草叉攻击象牙塔时，对正确性和可维护性的断言就不应该让我们感到惊讶。

在我看来，我们必须以牙还牙：如果我们要剥夺功能，我们最好给他们提供引人入胜的新特性。使用静态类型，你还可以得到模式匹配、QuickCheck 风格的属性测试和性能优势。使用纯度，你会得到软件事务内存和推测性评估。发现和实现更多这样的“杀手级应用程序”是推广的关键。（我目前与亚当·克利帕拉进行的一些研究是利用纯度为 Web 应用程序提供自动缓存。这还不算什么，但我认为它朝着正确的方向发展。）

我对正确性依然有着狂热的虔诚。但是如今，我怀疑对于大多数人来说，这就像是一种苦药，需要加入一些更好口感的特性。那也没关系。作为编程语言研究者，我们的挑战在于利用正确性带来即时的实际好处，而不是稍后的模糊的可维护性好处。

*感谢尼尔森·埃尔哈吉和基根·麦卡利斯特的评论。*

* * *

*附言：静态类型与动态类型的性能比较。* 本文的早期草案指出了[Quora 决定从 Python 转向 Scala](http://www.quora.com/Is-the-Quora-team-considering-adopting-Scala-Why)作为这一事实的明确指示。不幸的是，正如几位预读者指出的那样，有太多混杂因素使得这一主张无法被确立：CPython 从未专门为性能而设计，而 JVM 则已经进行了几十年的工作。因此，我只能用更理论的论据来讨论静态类型的性能优化技术：动态编译器的即时编译器优化技术涉及识别实际上是静态类型的代码段，并将它们编译成静态编译器的形式。因此，如果你提前知道这些信息，你总是会比后来知道这些信息做得更好：这只是程度上的问题。（当然，这并不解决 JIT 可以识别静态编译器难以确定的信息的可能性。）

*附言：共享事务内存。* 乔·达菲在[事务内存回顾](http://www.bluebytesoftware.com/blog/2010/01/03/ABriefRetrospectiveOnTransactionalMemory.aspx)中有一篇很棒的文章，介绍了他尝试为 Microsoft 的技术栈实现事务内存的经历。尽管对这个想法充满热情，但有趣的是注意到这样一句话：

> 在所有这些过程中，我们不断寻找并寻找杀手级 TM 应用程序。把这件事归咎于 TM 是不公平的，因为整个行业仍在寻找一个杀手级并发应用程序。但随着我们在后者中发现更多成功案例，我越来越不认为未来 5 年广泛部署的杀手级并发应用程序需要 TM。大多数享受自然隔离，如令人尴尬的并行图像处理应用程序。如果你需要共享，那么你就做错了。

理查德·蒂贝茨指出，并发通常是在比大多数工作程序员想处理的更低的架构层次上解决的，因此虽然 STM 对于这些平台来说是一个杀手级应用程序，但大多数开发人员根本不想考虑并发。