<!--yml

category: 未分类

date: 2024-07-01 18:17:15

-->

# Ott ⇔ PLT Redex：ezyang 的博客

> 来源：[`blog.ezyang.com/2014/01/ott-iff-plt-redex/`](http://blog.ezyang.com/2014/01/ott-iff-plt-redex/)

[Ott](http://www.cl.cam.ac.uk/~pes20/ott/) 和 [PLT Redex](http://redex.racket-lang.org/) 是一对互补的工具，适用于工作中的语义学家。Ott 是一个用 ASCII 符号写程序语言定义的工具，可以用 LaTeX 排版，也可用于生成定理证明器（如 Coq）的定义。PLT Redex 是一种用于指定和调试操作语义的工具。这两个工具都很容易安装，这是一个很大的优点。由于这两个工具相似，我觉得对它们进行比较执行各种常见任务可能会很有趣。（而且我认为 Redex 的手册相当糟糕。）

**变量。** 在 Ott 中，变量通过元变量（`metavar x`）定义，然后作为变量使用（可以单独使用元变量，也可以将其后缀为数字、索引变量或 tick）。

在 Redex 中，没有“元变量”的概念；变量只是另一种产生式。有几种不同的方法可以表明一个产生式是变量：最简单的方法是使用 `variable-not-otherwise-mentioned`，这可以自动防止关键字被视为变量。还有几种其他变量模式 `variable`、`variable-except` 和 `variable-prefix`，可以更精确地控制哪些符号被视为变量。如果你有一个分类变量的函数，`side-condition` 也许会很有用。

**语法。** Ott 和 Redex 都可以识别模糊匹配。Ott 在遇到模糊解析时会报错。而 Redex 则会生成所有有效的解析结果；尽管在解析术语时这并不那么有用，但在指定非确定性操作语义时却非常有用（尽管这可能会对性能产生不良影响）。`check-redundancy` 可能对识别模糊模式很有用。

**绑定器。** 在 Ott 中，绑定器通过在语法中明确声明 `bind x in t` 来定义；还有一个用于模式匹配收集绑定器的绑定语言。Ott 还可以为语义生成替换/自由变量函数。在 Redex 中，绑定器不在语法中声明；而是仅在规约语言中实现，通常使用替换（Redex 提供了一个实用的替换函数用于此目的），并明确要求变量是新鲜的。Redex 还在元语言中提供了一个用于立即进行 let 绑定的特殊形式（`term-let`）。

**列表。** Ott 支持两种形式的列表：点形式和列表理解。点形式看起来像`x1 , .. , xn`并需要上限。列表理解看起来像`</ xi // i IN 1 .. n />`；上下限可以省略。目前 Ott 的一个限制是它不理解如何处理嵌套的点形式，可以通过在制品上做理解，然后在其他地方说明制品满足的适当等式来解决这个问题。

Redex 使用省略号模式支持列表，看起来像`(e ...)`。这里没有语义内容：省略号只是匹配零个或多个`e`的副本，当存在多个省略号时可能导致非确定性匹配。支持嵌套的省略号，并且简单地导致嵌套列表。可以使用侧条件指定边界；但是 Redex 支持使用命名省略号进行有限形式的绑定（例如`..._1`），其中具有相同名称的所有省略号必须具有相同的长度。

**语义。** Ott 对您想定义的任何语义都是不可知的；可以指定任意判断。在 Redex 中也可以像通常一样定义判断，但 Redex 专门支持*评估语义*，其中语义是通过评估上下文来给出的，从而允许您避免使用结构规则。因此，通常的用例是定义一个正常的表达式语言，扩展该语言以具有评估上下文，然后使用`in-hole`定义一个`reduction-relation`进行上下文分解。限制在于，如果需要做任何复杂操作（例如[multi-hole evaluation contexts](https://github.com/iu-parfunc/lvars/tree/master/redex/lambdaLVar)），则必须返回到判断形式。

**排版。** Ott 支持通过转换为 LaTeX 进行排版。制品可以有与之关联的自定义 LaTeX，用于生成它们的输出。Redex 有一个`pict`库，可以直接排版成 PDF 或 Postscript；虽然 PLT Redex 似乎不支持定制排版作为预期用例，但它可以生成合理的类似 Lisp 的输出。

**结论。** 如果我必须说 Ott 和 PLT Redex 之间最大的区别是什么，那就是 Ott 主要关注于您定义的抽象语义含义，而 PLT Redex 主要关注于如何*匹配*语法（运行）。可以通过观察到，在 Ott 中，您的语法是 BNF，这被馈送到 CFG 解析器中；而在 PLT Redex 中，您的语法是用于模式匹配机器的模式语言。这不应该令人惊讶：人们期望每个工具的设计理念符合其预期的使用方式。