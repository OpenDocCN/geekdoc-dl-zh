<!--yml

category: 未分类

date: 2024-07-01 18:17:14

-->

# GHC 和可变数组：一个**脏**小秘密：ezyang 的博客

> 来源：[`blog.ezyang.com/2014/05/ghc-and-mutable-arrays-a-dirty-little-secret/`](http://blog.ezyang.com/2014/05/ghc-and-mutable-arrays-a-dirty-little-secret/)

## GHC 和可变数组：一个**脏**小秘密

Brandon Simmon 最近在 glasgow-haskell-users 邮件列表上发布了一个帖子，问了以下问题：

> 我一直在研究 [一个问题](http://stackoverflow.com/questions/23462004/code-becomes-slower-as-more-boxed-arrays-are-allocated/23557704#23557704)，在这个库中，随着分配更多可变数组，GC 占主导地位（我想我验证了这个？），所有代码的速度与挂起的可变数组数量成正比地变慢。

...对此，我回复道:

> 在当前的 GC 设计中，指针数组的可变数组总是放置在可变列表上。未收集的代的可变代的列表总是被遍历；因此，指针数组的数量对于小 GC 产生了线性的开销。

如果你从传统的命令式语言转过来，你可能会发现这非常令人惊讶：如果你为系统中所有可变数组支付了 Java 中每个 GC 的线性开销... 嗯，你可能永远都不会使用 Java。但大多数 Haskell 用户似乎过得很好；主要因为 Haskell 鼓励不可变性，使得大多数情况下不需要大量的可变指针数组。

当然，当你确实需要时，这可能有点痛苦。我们有一个 [GHC bug](https://ghc.haskell.org/trac/ghc/ticket/7662) 跟踪这个问题，还有一些低 hanging fruit（一种变体的可变指针数组，写操作更昂贵，但只有在写入时才放入可变列表中），以及一些有前途的实现卡标记堆的方向，这是像 JVM 这样的 GC 策略所使用的策略。

更加元层次上，为不可变语言实现一个性能良好的分代垃圾收集器要比为可变语言实现一个更容易得多。这是我个人的假设，解释了为什么 Go 仍然没有一个分代收集器，以及为什么 GHC 在某些突变类别上表现如此糟糕。

*后记。* 标题是一个双关语，因为“DIRTY”用于描述自上次 GC 以来已写入的可变对象。这些对象是记忆集的一部分，必须在垃圾收集期间遍历，即使它们位于旧代中也是如此。
