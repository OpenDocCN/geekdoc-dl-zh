<!--yml

category: 未分类

date: 2024-07-01 18:17:29

-->

# Is Haskell liberal or conservative? : ezyang’s blog

> 来源：[`blog.ezyang.com/2012/08/is-haskell-liberal-or-conservative/`](http://blog.ezyang.com/2012/08/is-haskell-liberal-or-conservative/)

## Haskell 是自由派还是保守派？

Steve Yegge 发表了一篇[有趣的文章](https://plus.google.com/u/0/110981030061712822816/posts/KaSKeg4vQtz)试图将自由派和保守派标签应用于软件工程。当然，这是一个极端简化（Yegge 自己承认）。例如，他得出结论说 Haskell 必须是“极端保守”的，主要是因为它极力强调安全性。这完全忽略了 Haskell 最好的一点，即*我们做一些疯狂的事情，正常情况下没有人会在没有 Haskell 安全功能的情况下这样做*。

所以我想我会借鉴一些 Yegge 的思路，通过提出的标准来评估一个语言用户的保守程度，并尽量穿上我“Haskell 帽子”来回答：

1.  *软件在发布之前应该目标是无缺陷的。* 是的。尽管，“小心以上代码中的错误；我只是证明了它的正确性，而没有尝试它。”

1.  *程序员应该免受错误的影响。* 是的。**但是**，Yegge 接着补充道：“许多语言特性本质上容易出错和危险，应禁止我们编写的所有代码使用。” 这并不是 Haskell 的做法：如果你想要带有可变状态的延续，Haskell 会提供给你。（试试在 Python 中做到这一点。）它并不*禁止*语言特性，只是使它们更啰嗦（`unsafePerformIO`）或更难使用。Haskell 对于逃生口的信念很健康。

1.  *程序员学习新语法有困难。* **不。** Haskell 完全站在了这个围栏的错误一侧，拥有任意的中缀操作符；甚至更极端的语言（例如 Coq）在语法制定上走得更远。当然，这并不是为了语法本身，而是为了紧密模拟数学家和其他从业者已经使用的现有语法。因此，我们允许操作符重载，但只有在支持代数法则的情况下。我们允许元编程，尽管我怀疑它目前很少使用，只因为它非常笨重（但在*文化上*，我认为 Haskell 社区非常愿意接受元编程的概念）。

1.  *生产代码必须经过编译器的安全检查。* 是的。**但是**，任何使用依赖类型语言的人对于“安全检查”的标准要求更高，而我们经常在决定静态编码会非常烦人的不变量时玩得很随意。请注意，Yegge 声称编译器安全检查的对立面是*简洁性*，这是一个完全错误的神话，由于非 Hindley Milner 类型系统缺乏类型推断而流传开来。

1.  *数据存储必须遵循一个明确定义的、公开的架构。* 明确定义的？是的。公开的？不是。Haskell 对静态检查的重视意味着编写数据类型的人更愿意在应用需求变化时更新它们，而且并不介意全局地重构数据库，因为这样做非常容易做到正确。

1.  *公共接口应该严格建模。* 是的。（尽管 *咳咳* “理想情况下应该面向对象” *咳咳*）

1.  *生产系统绝不能有危险或者有风险的后门。* **意外的。** 这里工具的匮乏意味着很难窥视正在运行的编译后可执行文件并且操纵内部数据：这是目前 Haskell 生态系统的一个大问题。但抽象来说，我们非常灵活：例如，XMonad 可以重新启动以运行任意的新代码 *同时保留你的全部工作状态*。

1.  *如果对某个组件的安全性有任何疑问，它不能被允许在生产环境中使用。* 这有点个人问题，实际上取决于你的项目，而不是语言本身。Haskell 对于安全关键项目非常合适，但我也用它写一些临时脚本。

1.  *快速胜于慢速。* **不。** Haskell 代码有机会非常快，而且通常从一开始就很快。但我们强调的特性（惰性和抽象）已知会导致性能问题，大多数 Haskell 程序员的做法是只有在我们（非常棒的）性能分析工具提醒我们时才进行优化。一些 Haskell 程序员本能地在他们的数据类型中加入 `! {-# UNPACK #-}`，但我不会 —— 至少在我认为我的代码太慢之前不会加。

Haskell 有很多功能都出现在 Yegge 的 “Liberal Stuff” 中。这里是其中一些：

+   Eval: 我们喜欢编写解释器，这有点像类型安全的 eval。

+   Metaprogramming: Template Haskell。

+   Dynamic scoping: Reader monad。

+   all-errors-are-warnings: 我们可以[将类型错误延迟到运行时！](http://hackage.haskell.org/trac/ghc/ticket/5624)。

+   Reflection and dynamic invocation: `class Data`。

+   RTTI: 我听说这被称为“字典”。

+   The C preprocessor: 不情愿地不可或缺。

+   Lisp macros: 为什么要使用宏，当你可以在 Template Haskell 中正确地做！

+   Domain-specific languages: Haskell 对 EDSLs 简直游刃有余。

+   Optional parameters: 这被称为组合器库。

+   Extensible syntax: 当然啦中缀表达式！

+   Auto-casting: 数字字面量，有谁不会？

+   Automatic stringification: `class Show` 和 deriving。

+   Sixty-pass compilers: GHC 运行 *非常多* 的编译步骤。

+   Whole-namespace imports: 是的（虽然既方便又有点烦人）。

我从这次对话中得到的感觉是，大多数人认为“Haskell”和“静态类型”，同时想着在 Haskell 中编写传统动态类型代码有多糟糕，却忘了 Haskell 实际上是一种令人惊讶的自由语言，重视可理解性、简洁性和冒险精神。Haskell 是自由派还是保守派？我认为它是设计空间中的一个有趣点，将一些保守观点视为基础，然后看它能走多远。*它折向了极右，结果绕到了极左。*
