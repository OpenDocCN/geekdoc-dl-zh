<!--yml

category: 未分类

date: 2024-07-01 18:18:04

-->

# Intelligence is the ability to make finer distinctions: Another Haskell Advocacy Post : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/10/finer-distinctions/`](http://blog.ezyang.com/2010/10/finer-distinctions/)

我父母喜欢向我推销各种自助书籍，虽然我有时会对此嗤之以鼻，但我确实会阅读（或至少粗略翻阅）它们，并从中提取出有用的信息。这句标题的引用来自罗伯特·清崎在《富爸爸，穷爸爸》中的“富爸爸”。

“Intelligence is the ability to make finer distinctions” really spoke to me. I’ve since found it to be an extremely effective litmus test to determine if I’ve really understood something. A recent example comes from my concurrent systems class, where there are many extremely similar methods of mutual exclusion: mutexes, semaphores, critical regions, monitors, synchronized region, active objects, etc. True knowledge entails an understanding of the conceptual details differentiating these gadgets. What are the semantics of a *signal* on a condition variable with no waiting threads? Monitors and synchronized regions will silently ignore the signal, thus requiring an atomic release-and-wait, whereas a semaphore will pass it on to the next *wait*. Subtle.

* * *

We can frequently get away with a little sloppiness of thinking, indeed, this is the mark of an expert: they know precisely how much sloppiness they can get away with. However, from a learning perspective, we’d like to be able to make as fine a distinction as possible, and hopefully derive some benefit (either in the form of deeper understanding or a new tool) from it.

Since this is, after all, an advocacy piece, how does learning Haskell help you make finer distinctions in software? You don’t have to look hard:

> Haskell is a standardized, general-purpose **purely** functional programming language, with **non-strict semantics** and strong static typing.

These two bolded terms are concepts that Haskell asks you to make a finer distinction on.

* * *

*Purity.* Haskell requires you to make the distinction between pure code and input-output code. The very first time you get the error message “Couldn't match expected type `[a]` against inferred type `IO String`” you are well on your way to learning this distinction. Fundamentally, it is the difference between *computation* and *interaction with the outside world*, and no matter how imperative your task is, both of these elements will be present in a program, frequently lumped together with no indication which is which.

纯代码带来了巨大的好处。它自动线程安全和异步异常安全。它与外部世界没有隐藏的依赖关系。它是可测试和确定性的。系统可以在没有对外部世界承诺的情况下对纯代码进行推测性评估，并且可以缓存结果而无需担忧。Haskellers 痴迷于尽可能多地将代码移到 IO 之外：你不必如此，但即使在小剂量中，Haskell 也会使你意识到被视为良好工程实践的东西如何变得严格。

* * *

*非严格语义。* 有些事情是理所当然的，生活中的一些小恒定，你无法想象会有所不同。也许如果你停下来思考一下，还有另一种方式，但这种可能性从未发生在你身上。你驾驶的道路哪一边是这些事情之一；严格评估是另一种。Haskell 要求你区分严格评估和惰性评估。

Haskell 对这一区别不像对纯度和静态类型那样张扬，因此你可以在不理解这一区别的情况下愉快地进行编程，直到第一次堆栈溢出。此时，如果你不理解这一区别，错误会显得难以解决（“但在我知道的其他语言中是有效的”），但如果你了解，堆栈溢出很容易修复——也许只需明确地使奇怪的参数或数据构造函数严格。

隐式惰性具有许多显著优点。它允许用户级控制结构。它编码流和其他无限数据结构。它比严格评估更一般化。它在摊销持久数据结构的构建中至关重要。（Okasaki）它也不总是适合使用：Haskell 促进了对严格性和惰性优缺点的理解。

> * 嗯，几乎是。只有在你拥有无限内存的情况下，它才完全泛化严格评估，此时任何严格评估的表达式也会惰性评估，而反之则不成立。但在实践中，我们有限制堆栈大小等讨厌的事物。

* * *

*缺点。* 你能够做出更细微的区分表明了你的智力。但同样地，如果这些区分不成为第二天性，每次需要调用它们时都会带来认知负担。此外，这使得那些不理解区别的人难以有效地进行代码开发。（保持简单！）

对于有经验的 Haskell 程序员来说，管理纯度已经是驾轻就熟的事情：他们通过类型检查器长时间训练，知道什么是可接受的，什么是不可接受的。鉴于单子的神秘性，通常在开始学习 Haskell 时人们会积极尝试学习如何管理纯度。管理严格性对有经验的 Haskell 程序员来说也很容易，但我觉得它的学习曲线更高：没有严格性分析器在你做出次优选择时大声警告你，大多数情况下你可以不考虑它而逃避问题。有人可能会说，默认惰性不是正确的选择方式，并且正在[探索严格的设计空间](http://trac.haskell.org/ddc/)。我仍然乐观地认为，我们 Haskell 程序员可以建立起一套知识体系和教学技巧，引导新手进入非严格评估的奥秘和奇迹中去。

* * *

所以，这就是它们：纯度和非严格性，这两个 Haskell 希望你能区分的概念。即使你从未计划在严肃的项目中使用 Haskell，对这两个概念有直观的感受也将极大地影响你的其他编程实践。纯度将在你编写线程安全代码、管理副作用、处理中断等方面提供帮助。惰性将在你使用生成器、处理流、控制结构、记忆化、使用函数指针等高级技巧时发挥作用。这些都是非常强大的工程工具，你应该自己尝试一下。
