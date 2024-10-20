<!--yml

category: 未分类

date: 2024-07-01 18:17:15

-->

# PEPM’14: **The HERMIT in the Stream**：ezyang’s 博客

> 来源：[`blog.ezyang.com/2014/01/pepm14-the-hermit-in-the-stream/`](http://blog.ezyang.com/2014/01/pepm14-the-hermit-in-the-stream/)

POPL 就快来了！当会议正式开始时，我将在[Tumblr 上实时更新](http://ezyang.tumblr.com/)，但与此同时，我想在 PEPM'14 程序的一篇论文中写点东西：[The HERMIT in the Stream](http://www.ittc.ku.edu/~afarmer/concatmap-pepm14.pdf)，作者是 Andrew Farmer, Christian Höner zu Sierdissen 和 Andy Gill。该论文提出了一种优化方案的实现，用于在[stream fusion 框架](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.104.7401)中消除对 concatMap 组合子的使用，该框架是使用[HERMIT 优化框架](http://www.ittc.ku.edu/csdl/fpg/software/hermit.html)开发的。HERMIT 项目已经进行了一段时间，各种应用该框架的论文陆续发表（任何参加 Haskell 实现者研讨会的人都能证明这一点）。

“但是等等”，你可能会问，“我们不是已经有了[stream fusion](http://hackage.haskell.org/package/stream-fusion)吗？” 你是对的：但是尽管 stream fusion 作为一个库是可用的，它并没有取代 GHC 默认的 foldr/build 融合系统。什么使得融合方案好呢？一个重要的度量标准是它支持的列表组合子的数量。几乎可以说 stream fusion 几乎完全取代了 foldr/build 融合，除了 concatMap 的情况，这个问题已经持续了七年，阻止了 GHC 将 stream fusion 作为其默认选项。

原来，我们很久以前就知道如何优化 concatMap 了；[Duncan Coutts 在他的论文中给出了一个基本的概述。](http://community.haskell.org/~duncan/thesis.pdf) 这篇论文的主要贡献是[这一优化的原型实现](https://github.com/xich/hermit-streamfusion)，包括重要技术细节的阐述（增加原始规则的适用性，简化器的必要修改以及用于解糖列表推导的规则）。论文还提供了一些微基准测试和真实世界基准测试，论证了优化 concatMap 的重要性。

我很高兴看到这篇论文，因为它是在替换 GHC 标准库中的 foldr/build 融合与流融合之路上的一个重要里程碑。同时，开发这一优化似乎在很大程度上得益于使用 HERMIT，这对于 HERMIT 的验证是一个很好的例证（尽管论文没有详细介绍 HERMIT 如何在开发这一优化过程中起作用）。

论文中所述的优化还有一些令人不太满意的地方，最好通过考虑从流融合实施者的角度来表达。她有两个选择：

+   她可以尝试直接使用**HERMIT 系统**。然而，HERMIT 会导致 5-20 倍的编译减速，这对实际使用来说相当令人泄气。这种减速可能并非根本性的问题，在适当的时候会消失，但今天显然不是那个时候。在原型中有限的流融合实现（它们没有实现所有的组合器，只是足够用来运行他们的数据）也建议不直接使用该系统。

+   她可以直接按照论文中所述的规则将其整合到编译器中。这将需要特殊情况代码，仅适用于应用非语义保持简化的流，并且基本上需要重新实现系统，并且这篇论文提供了指导。但这种特殊情况代码的适用性有限，超出了对 concatMap 的实用性，这是一个负面评价。

因此，至少从普通 GHC 用户的角度来看，我们在手中拥有流融合还需要等待一段时间。尽管如此，我同意微基准测试和[ADPFusion](http://hackage.haskell.org/package/ADPfusion)案例研究显示了这种方法的可行性，而且新的简化规则的一般原则似乎是合理的，尽管有些特殊。

如果你在阅读 nofib 性能部分时要注意一点：实验是将他们的系统与 foldr/build 进行比较的，因此增量主要显示出流融合的好处（在文本中，他们指出哪些基准测试最从 concatMap 融合中受益）。无论如何，这确实是一篇相当棒的论文：一定要看看！
