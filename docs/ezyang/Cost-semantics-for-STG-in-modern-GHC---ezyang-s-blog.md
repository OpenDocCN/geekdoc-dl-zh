<!--yml

category: 未分类

date: 2024-07-01 18:17:18

-->

# 现代 GHC 中的 STG 成本语义：ezyang 的博客

> 来源：[`blog.ezyang.com/2013/09/cost-semantics-for-stg-in-modern-ghc/`](http://blog.ezyang.com/2013/09/cost-semantics-for-stg-in-modern-ghc/)

## 现代 GHC 中的 STG 成本语义

学术出版的一个问题是难以使旧论文保持最新。这对于这篇[1995 年 Sansom 关于非严格高阶函数式语言剖析的论文](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.43.6277)来说显然也是如此。尽管论文的基本思想仍然成立，但在 GHC 中成本中心的实际实现已经发生了相当大的变化，也许最显著的变化是成本中心栈的引入。因此，虽然旧论文很好地向你介绍了 GHC 中剖析的基本思想，但如果你真的想了解详情，这篇论文提供的指导有限。

当你的成本语义过时时，你会怎么做？当然是更新它们！我呈现了一份[现代 GHC 中 STG 的更新成本语义（PDF）](https://github.com/ezyang/stg-spec/raw/master/stg-spec.pdf)（[GitHub](https://github.com/ezyang/stg-spec)）。最终，这些将会放入 GHC 代码库中，与[core-spec](http://typesandkinds.wordpress.com/2012/12/03/a-formalization-of-ghcs-core-language/)类似，后者是 Core 语言的类似文档。然而，我还没有用这些语义做过任何证明，所以它们可能还有些 bug。

尽管没有证明，但形式化已经非常有帮助：我已经发现了当前实现中的一个 bug（在文档中有所记录）。我还根据当前规则的设置方式，确定了一次潜在的重构。请告诉我你发现的其他任何 bug！
