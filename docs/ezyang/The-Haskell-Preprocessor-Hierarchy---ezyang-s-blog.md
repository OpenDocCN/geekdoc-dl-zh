<!--yml

category: 未分类

date: 2024-07-01 18:18:16

-->

# Haskell 预处理器层次结构：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/06/the-haskell-preprocessor-hierarchy/`](http://blog.ezyang.com/2010/06/the-haskell-preprocessor-hierarchy/)

本篇文章是我希望能成为关于使用 [c2hs](http://www.cse.unsw.edu.au/~chak/haskell/c2hs/) 的多部教程/食谱系列之一。（[Hackage](http://hackage.haskell.org/cgi-bin/hackage-scripts/package/c2hs/)）

1.  Haskell 预处理器层次结构（本文）

1.  [配置 Cabal、FFI 和 c2hs](http://blog.ezyang.com/2010/06/setting-up-cabal-the-ffi-and-c2hs/)

1.  [FFI API 设计原则](http://blog.ezyang.com/2010/06/principles-of-ffi-api-design/)

1.  [c2hs 的第一步](http://blog.ezyang.com/2010/06/first-steps-in-c2hs/)

1.  [使用 get 和 set 进行数据整理](http://blog.ezyang.com/2010/06/marshalling-with-get-and-set/)

1.  [调用和乐趣：重新整理马歇尔](http://blog.ezyang.com/2010/06/call-and-fun-marshalling-redux/)

*c2hs 是什么？* c2hs 是一个 Haskell 预处理器，帮助生成 [外部函数接口](http://www.haskell.org/haskellwiki/FFI_Introduction) 绑定，与 [hsc2hs](http://www.haskell.org/ghc/docs/6.12.2/html/users_guide/hsc2hs.html) 和 [GreenCard](http://www.haskell.org/greencard/) 一起。（下图展示了 Cabal 支持的当前预处理器。）（对于好奇的人来说，Cpp 被放在其他 FFI 预处理器中，并不是因为它特别有用于生成 FFI 代码，而是因为许多 FFI 预处理器也实现了一些 Cpp 的功能。我根据 Alex 是一个词法分析器生成器，而 Happy 是一个语法分析器生成器的原则来确定了它们的顺序。）

*c2hs 是做什么的？* 在我告诉你 c2hs 做什么之前，让我告诉你它 *不* 做的事情：它 *不* 会魔法般地消除你理解 FFI 规范的需要。事实上，它可能会让你编写更大、更雄心勃勃的绑定，这反过来会测试你对 FFI 的理解。（稍后详细介绍。）

c2hs 帮助做的事情是消除编写 FFI 绑定时的一些枯燥工作。（那些手写过 FFI 绑定的老手们此刻都在会意地点头。）以下是你将不再需要做的一些事情：

+   将枚举定义移植到纯 Haskell 代码中（这意味着需要编写数据定义以及 Enum 实例），

+   手动计算你要进行数据整理的结构体的大小，

+   手动计算结构体中字段的偏移量（并处理相应的可移植性头疼问题），

+   手动编写 C 指针类型，

+   （在某种程度上）编写实际的 `foreign import` 声明以使用 C 函数

*何时使用 c2hs？* Haskell 有很多预处理器；你应该选择哪一个？简单（虽然有些不太准确）地说，你可以将上述层次结构特征化为：越往下，你需要写的样板越*少*，需要阅读的文档越*多*；因此有人建议，对于小型 FFI 项目应该使用 hsc2hs，而对于更大型的项目则更适合 c2hs。

**c2hs** 支持而 **hsc2hs** 不支持的功能：

+   根据 C 头文件的内容自动生成 `foreign import`，

+   函数调用的半自动封送和解封送，

+   将指针类型和层次结构翻译成 Haskell 类型。

GreenCard 支持而 c2hs 不支持的功能：

+   根据 Haskell 类型签名自动生成 `foreign import`（实际上，这是一个主要的哲学区别），

+   更全面的封送语言，

+   使用数据接口方案自动生成数据封送，

此外，hsc2hs 和 c2hs 被认为是相当成熟的工具；前者与 GHC 打包在一起，而后者（部分）被用于 gtk2hs，这可以说是 Haskell 中最大的 FFI 绑定。GreenCard 则稍微“年轻”一些，但最近经过更新，看起来非常有前途。

*这个教程系列适合我吗？* 幸运的是，我不会假设读者对 FFI 有太多了解（我进入时对它的了解肯定没有现在多）；不过，会假设读者对 C 有一些了解。特别是，你应该了解将数据传递给 C 函数和从 C 函数中取出数据的标准惯例，并且对于处理指针应该感到自如（尽管可能还会简要复习一下）。

*下次再见。* [设置 Cabal、FFI 和 c2hs](http://blog.ezyang.com/2010/06/setting-up-cabal-the-ffi-and-c2hs/)。
