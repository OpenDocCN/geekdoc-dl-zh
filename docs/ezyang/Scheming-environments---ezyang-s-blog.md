<!--yml

category: 未分类

日期：2024-07-01 18:18:27

-->

# 环境策划：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/02/scheming-environment/`](http://blog.ezyang.com/2010/02/scheming-environment/)

环境在 [MIT/GNU Scheme 中是一流对象](http://www.gnu.org/software/mit-scheme/documentation/mit-scheme-ref/Environment-Operations.html#Environment-Operations)。这很棒，因为 Scheme 的词法结构的一个整体部分也是一个数据结构，能够编码数据和行为。事实上，环境数据结构正是 Yegge 所称的 [属性列表](http://steve-yegge.blogspot.com/2008/10/universal-design-pattern.html)，可以与继承链接起来。因此它不仅是一个数据结构，而且还是一个高度通用的数据结构。

即使不能将环境作为一流变量传递，你仍然可以利用其在面向对象方式中[隐藏本地状态](http://mitpress.mit.edu/sicp/full-text/book/book-Z-H-21.html#%_sec_3.2.3)的语法联系。数据只能在指向适当环境框架的过程内部访问，传统上闭包返回一个 lambda（其双泡指向新出生的环境框架），作为封闭状态的唯一接口。这需要相当多的样板代码，因为此 lambda 必须支持你可能希望对函数内部进行的每种操作。

然而，具有一流环境对象，你可以随意处理闭包的绑定。不幸的是，没有直接 `get-current-environent` 的方法（除了顶级 REPL 环境，这不算），所以我们采取以下技巧：

```
(procedure-environment (lambda () '()))

```

`procedure-environment` 可以从某些过程的双泡中抓取环境指针。因此，我们强制使用空 lambda 创建指向我们关心的环境的双泡。

我最近在一个 6.945 项目中使用了这种技术，使用 lambda 生成了一堆带有不同参数的过程（鼓励代码重用），类似于多次包含 C 文件并使用不同宏定义的时间荣誉技巧。不是将这些过程作为哈希表返回然后人们必须显式调用，而是返回环境，因此任何消费者都可以通过使用适当的环境进入"一个不同的宇宙"。

Scheme 在其庆祝环境作为一流对象方面非常独特。我可以尝试在 Python 中使用这种技巧，但函数上的 `func_closure` 属性是只读的，而且 Python 的作用域规则相当弱。这真是遗憾，因为这种技术允许一些可爱的语法简化。

*评论。* Oleg Kiselyov 提到 "*MIT Scheme* 特别是在将环境作为一等环境进行庆祝方面是独一无二的"，并指出即使一些 MIT Scheme 的开发者也对该特性有了[反思](http://people.csail.mit.edu/gregs/ll1-discuss-archive-html/msg03947.html)。这使得代码难以优化，从理论和实际上来说都是危险的：从理论上讲，环境实际上是一种实现细节，从实际上讲，这使得对代码的推理变得非常困难。

从面对面的讨论中，我对 Sussman 偏爱的 Scheme 方言允许这样一个危险的特性并不感到惊讶；Sussman 一直支持让人们接触危险的玩具，并相信他们能正确使用它们。
