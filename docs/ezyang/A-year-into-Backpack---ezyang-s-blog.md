<!--yml

类别：未分类

日期：2024-07-01 18:16:57

-->

# 一年后的 Backpack：ezyang 的博客

> 来源：[`blog.ezyang.com/2018/07/a-year-into-backpack/`](http://blog.ezyang.com/2018/07/a-year-into-backpack/)

## 一年后的 Backpack

距我获得我的斗篷和长袍并加入 Facebook（我一直在那里工作 PyTorch）已经一年了，但是在 Facebook，Backpack 并没有停滞不前；事实上，活动比我希望的要多得多。我希望在这篇博文中总结一些近况。

### 使用 Backpack 的库

在使用 Backpack 空间的库中，一直在进行一些非常有趣的工作。以下是我在过去几个月中看到的两个最大的项目：

**unpacked-containers.** 多产的 Edward Kmett 编写了[unpacked-containers](https://github.com/ekmett/unpacked-containers)包，利用了通过 Backpack 签名进行展开的事实，为你提供了比通常的装箱表示性能提高（15-45%）的通用容器实现。在[这篇 Reddit 主题](https://www.reddit.com/r/haskell/comments/8a5w1n/new_package_unpackedcontainers/)中进行了大量讨论。

**hasktorch.** hasktorch，由 Austin Huang 和 Sam Stites 开发，是 Haskell 的张量和神经网络库。它绑定到 TH 库（也支持 PyTorch），但使用了 Backpack，使得 Kaixi Ruan 的文章[深度学习的 Backpack](http://blog.ezyang.com/2017/08/backpack-for-deep-learning/)焕然一新。这可能是我迄今为止见过的最大的 Backpack 实例之一。

### 生态系统中的 Backpack

**Eta 支持 Backpack。** Eta 是 GHC 的 JVM 分支，将 Backpack 支持移植到他们的分支中，这意味着你现在可以在你的 Eta 项目中使用 Backpack。它在[这篇 Twitter 帖子](https://twitter.com/rahulmutt/status/1015956353028747271)中宣布，并且在[这篇帖子](https://twitter.com/rahulmutt/status/1017658825799753728)中有更多讨论。

**GSOC 关于多个公共库。** 作为 Google Summer of Code 的一部分，Francesco Gazzetta 正在添加对 Cabal 中[多个公共库](https://github.com/haskell/cabal/issues/4206)的支持。多个公共库将使许多 Backpack 用例更容易编写，因为您不再需要将 Backpack 单元拆分为单独的包，为每个包编写不同的 Cabal 文件。

### GHC 和 Cabal 中的 Backpack

总体而言，自初始发布以来，我们没有改变 Backpack 的用户界面语法或语义。然而，已经有一些显著的错误修复（也许比预期的少），这些错误修复已经合并并即将到来：

+   [#13955](https://ghc.haskell.org/trac/ghc/ticket/13955)：Backpack 现在支持非*种类，因此您可以使用 Backpack 进行活力多态。

+   [#14525](https://ghc.haskell.org/trac/ghc/ticket/14525)：Backpack 现在支持 CPP 扩展

+   [#15138](https://ghc.haskell.org/trac/ghc/ticket/15138)：Backpack 将很快支持数据 T : Nat 签名，可以用类型 T = 5 来实例化。感谢 Piyush Kurur 发现问题并编写补丁来修复此问题。

+   修复了 Cabal 问题 [#4754](https://github.com/haskell/cabal/issue/4753)：现在 Backpack 与性能分析兼容。

### 需要帮助的事情

**Stack 对 Backpack 的支持。** 在 [Stack issue #2540](https://github.com/commercialhaskell/stack/issues/2540) 中，我自愿为 Stack 实现 Backpack 的支持。然而，在过去的一年中，显而易见的是我实际上没有足够的空闲时间来亲自实现这个功能。现在正在寻找勇敢的人来深入研究这个问题；我很乐意就 Backpack 方面提供咨询。

**为 Backpack 添加模式同义词支持。** 你应该能够用一个合适的双向类型同义词填充签名数据 T = MkT Int，反之亦然！这是 GHC 问题 [#14478](https://ghc.haskell.org/trac/ghc/ticket/14478)。我们认为这并不应该太难；我们必须获取由构造函数诱导的匹配项，并检查它们是否匹配，但确切地如何做还需要一些时间。
