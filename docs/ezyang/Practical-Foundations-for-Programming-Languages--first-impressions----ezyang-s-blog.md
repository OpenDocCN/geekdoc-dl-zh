<!--yml

category:

date: 2024-07-01 18:17:29

-->

# 编程语言实用基础（第一印象）：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/08/practical-foundations-for-programming-languages/`](http://blog.ezyang.com/2012/08/practical-foundations-for-programming-languages/)

## 编程语言实用基础（第一印象）

[罗伯特·哈珀](http://www.cs.cmu.edu/~rwh/)最近（有点）发布了一本[书的预印本（PDF）](http://www.cs.cmu.edu/~rwh/plbook/book.pdf)，他一直在努力完成，*编程语言实用基础*。我在初次发布时下载了一份副本，但一直拖延没有真正深入研究这本大约 590 页的书。直到哈珀最近在[他的一篇最新博文](http://existentialtype.wordpress.com/2012/08/14/haskell-is-exceptionally-unsafe/)中成功引诱我，我才最终坐下来更加仔细地浏览了一下。

立即诱惑的是将 PFPL 与本杰明·皮尔斯的开创性著作*类型与编程语言*进行比较。乍一看，两者似乎在内容和展示方式上有相当多的重叠。两本书都从非常简单的编程语言开始，然后逐步添加功能，以解释编程语言设计中逐渐复杂的主题。但是 PFPL 在许多方面有意与 TAPL 不同。出于意识形态的原因，哈珀完全跳过了无类型语言，直接进入具有变量 let 绑定的类型语言，以立即引入类型、上下文和安全性。这种展示方式更为简洁，对于没有编程语言经验的新手来说，可能会感觉 PFPL 更像是一本参考手册而不是教科书—[一个评论者](http://existentialtype.wordpress.com/2012/08/06/there-and-back-again/#comment-949)将其比作数学教科书。（与此无关的是，哈珀的介绍性课程[15-312 编程语言原理](http://www.cs.cmu.edu/~rwh/courses/ppl/schedule.html)，使用了 PFPL，*确实*从无类型 lambda 演算开始。）

尽管如此，这种简洁性对于 PFPL 是一种资产；首先，它使哈珀能够涵盖大量内容，涉及 TAPL 根本不涉及的主题。简洁性也并不意味着哈珀“遗漏了任何内容”，每一章都是自包含的，并且在其选择涵盖的主题上全面而详尽。这也使得像我这样对讨论的主题有些熟悉但从不同视角看待和思考的人们来说，阅读起来颇具乐趣。

Harper 一直在博客中写他的书，我认为他的博客文章很好地指示了书中哪些部分特别值得关注。Harper 在他的博客文章中采用了“全凭直觉”的风格，并将大部分形式主义留给了他的书。我认为这很遗憾，因为他定义的形式主义是相当容易理解的，会让他的许多读者（至少从评论部分来看如此！）更加清楚。以下是一些配对：

+   [动态语言是静态语言](http://existentialtype.wordpress.com/2011/03/19/dynamic-languages-are-static-languages/) 是第十八章“动态类型”的伴侣。在那里，他开发了动态 PCF（本质上是 Lisp 的核心），并展示了通常的具体语法掩盖了发生的标记，而通常的动态掩盖了在动态类型语言中普遍存在的冗余检查。在这些圣战中，总是有一种诱惑，试图扩展论点的范围，但如果你接受动态 PCF 作为表述辩论一个方面的有效方式，它是*极其*精确的。

+   [引用透明性](http://existentialtype.wordpress.com/2012/02/09/referential-transparency/) 是第三十二章“符号”的伴侣。符号有些奇怪，因为大多数语言甚至没有一种方式来*承认*这个概念的存在。你可以将它视为一个“广义可变单元”的标识符，除非你实际访问它，但实际上你应该只读形式化处理，因为它非常简单。

+   [词语的重要性](http://existentialtype.wordpress.com/2012/02/01/words-matter/) 是第三十六章“可赋值引用”的伴侣。这是一个简单的术语分割，受 Harper 在他的书第一章中对术语“变量”的定义所启发。

+   [Haskell is Exceptionally Unsafe](http://existentialtype.wordpress.com/2012/08/14/haskell-is-exceptionally-unsafe/) 是第三十四章“动态分类”的伴侣。文章认为能够在*运行时*生成异常类别非常重要（这里的“类别”具有非常精确的含义，即它是有限求和的索引，本例中为异常类型；这在第十二章中有详细讨论）。至少在 Haskell 社区中，这并不是一个特别常见的“动态”用法（尽管我同意 Harper 认为这是正确的用法），而 PFPL 确切地解释了它的含义，没有多余也没有少数。

总的来说，*编程语言实用基础*非常值得一读。人们很少真正从头到尾地阅读教科书，但如果你发现自己阅读了 Harper 的博客文章并感到困惑，请给配套的章节一个机会。即使是我读过的书的一小部分，PFPL 也教会了我新的东西，并澄清了我的思绪。
