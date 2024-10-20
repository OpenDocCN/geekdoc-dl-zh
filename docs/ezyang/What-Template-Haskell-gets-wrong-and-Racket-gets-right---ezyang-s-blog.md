<!--yml

category: 未分类

date: 2024-07-01 18:17:05

-->

# [What Template Haskell gets wrong and Racket gets right : ezyang’s blog](http://blog.ezyang.com/2016/07/what-template-haskell-gets-wrong-and-racket-gets-right/)

> 来源：[`blog.ezyang.com/2016/07/what-template-haskell-gets-wrong-and-racket-gets-right/`](http://blog.ezyang.com/2016/07/what-template-haskell-gets-wrong-and-racket-gets-right/)

为什么 [Haskell 中的宏](https://stackoverflow.com/questions/10857030/whats-so-bad-about-template-haskell) 糟糕，而 Racket 中的宏很棒？ GHC 的 Template Haskell 支持确实存在许多小问题，但我认为有一个基本设计点 Racket 做对了而 Haskell 做错了：Template Haskell 没有充分区分 *编译时* 和 *运行时* 阶段。混淆这两个阶段会导致诸如“Template Haskell 不适用于交叉编译”的奇怪说法，以及 `-fexternal-interpreter` 这样更奇怪的特性（通过将宏代码发送到目标平台执行来“解决”交叉编译问题）。

只需比较 Haskell 和 Racket 的宏系统设计差异即可见端倪。本文假设您了解 Template Haskell 或 Racket 的知识，但不一定两者皆通。

**基本宏**。为了建立比较基础，让我们比较一下 Template Haskell 和 Racket 中宏的工作方式。在 Template Haskell 中，调用宏的基本机制是 *splice*：

```
{-# LANGUAGE TemplateHaskell #-}
module A where
val = $( litE (intPrimL 2) )

```

这里，`$( ... )` 表示插入，它运行 `...` 来计算一个 AST，然后将其插入正在编译的程序中。语法树是使用库函数 `litE`（字面表达式）和 `intPrimL`（整数原始字面量）构造的。

在 Racket 中，宏是通过 [transformer bindings](https://docs.racket-lang.org/reference/syntax-model.html#%28part._transformer-model%29) 引入，并在扩展器遇到此绑定的使用时调用：

```
#lang racket
(define-syntax macro (lambda (stx) (datum->syntax #'int 2)))
(define val macro)

```

这里，`define-syntax` 定义了一个名为 `macro` 的宏，它接受其用法的语法 `stx`，并无条件地返回代表文字二的 [语法对象](https://docs.racket-lang.org/guide/stx-obj.html)（使用 `datum->syntax` 将 Scheme 数据转换为构造它们的 AST）。

Template Haskell 宏显然不如 Racket 的表达力强（标识符不能直接调用宏：插入总是在语法上显而易见）；相反，向 Racket 引入插入特殊形式很容易（对于此代码，特别感谢 Sam Tobin-Hochstadt — 如果你不是 Racketeer，不必过于担心具体细节）：

```
#lang racket
(define-syntax (splice stx)
    (syntax-case stx ()
        [(splice e) #'(let-syntax ([id (lambda _ e)]) (id))]))
(define val (splice (datum->syntax #'int 2)))

```

我将在一些进一步的示例中重用 `splice`；它将被复制粘贴以保持代码自包含性，但不需要重新阅读。

**宏帮助函数的阶段**。在编写大型宏时，经常希望将一些代码因子化到一个帮助函数中。现在我们将重构我们的示例，使用外部函数来计算数字二。

在模板哈斯克尔中，您不允许在一个模块中定义一个函数，然后立即在一个片段中使用它：

```
{-# LANGUAGE TemplateHaskell #-}
module A where
import Language.Haskell.TH
f x = x + 1
val = $( litE (intPrimL (f 1)) ) -- ERROR
-- A.hs:5:26:
--     GHC stage restriction:
--       ‘f’ is used in a top-level splice or annotation,
--       and must be imported, not defined locally
--     In the splice: $(litE (intPrimL (f 1)))
-- Failed, modules loaded: none.

```

然而，如果我们将 `f` 的定义放在一个模块中（比如 `B`），我们可以导入然后在一个片段中使用它：

```
{-# LANGUAGE TemplateHaskell #-}
module A where
import Language.Haskell.TH
import B (f)
val = $( litE (intPrimL (f 1)) ) -- OK

```

在 Racket 中，可以在同一个文件中定义一个函数，并在宏中使用它。但是，您必须使用特殊形式 `define-for-syntax` 将函数放入适合宏使用的正确*阶段*中：

```
#lang racket
(define-syntax (splice stx)
    (syntax-case stx ()
        [(splice e) #'(let-syntax ([id (lambda _ e)]) (id))]))
(define-for-syntax (f x) (+ x 1))
(define val (splice (datum->syntax #'int (f 1))))

```

如果我们尝试简单地 `(define (f x) (+ x 1))`，我们会得到一个错误 “f: unbound identifier in module”。原因是 Racket 的阶段区分。如果我们 `(define f ...)`，`f` 是一个*运行时*表达式，而运行时表达式不能在*编译时*使用，这是宏执行时的情况。通过使用 `define-for-syntax`，我们将表达式放置在编译时，以便可以使用它。（但同样地，`f` 现在不能再在运行时使用。从编译时到运行时的唯一通信是通过宏的扩展为语法对象。）

如果我们将 `f` 放在一个外部模块中，我们也可以加载它。但是，我们必须再次指示我们希望将 `f` 作为*编译时*对象引入作用域：

```
(require (for-syntax f-module))

```

与通常的 `(require f-module)` 相反。

**反映和结构类型变换绑定。** 在模板哈斯克尔中，`reify` 函数使模板哈斯克尔代码可以访问有关定义的数据类型的信息：

```
{-# LANGUAGE TemplateHaskell #-}
module A where
import Language.Haskell.TH
data Single a = Single a
$(reify ''Single >>= runIO . print >> return [] )

```

此示例代码在编译时打印有关 `Single` 的信息。编译此模块会给我们关于 `List` 的以下信息：

```
TyConI (DataD [] A.Single [PlainTV a_1627401583]
   [NormalC A.Single [(NotStrict,VarT a_1627401583)]] [])

```

`reify` 函数通过交错插入片段和类型检查实现：在顶层片段之前的所有顶层声明在运行顶层片段之前都已完全类型检查。

在 Racket 中，使用 `struct` 形式定义的结构的信息可以通过 [结构类型转换器绑定](https://docs.racket-lang.org/reference/structinfo.html) 传递到编译时：

```
#lang racket
(require (for-syntax racket/struct-info))
(struct single (a))
(define-syntax (run-at-compile-time stx)
  (syntax-case stx () [
    (run-at-compile-time e)
      #'(let-syntax ([id (lambda _ (begin e #'(void)))]) (id))]))
(run-at-compile-time
  (print (extract-struct-info (syntax-local-value (syntax single)))))

```

输出如下：

```
'(.#<syntax:3:8 struct:single> .#<syntax:3:8 single>
   .#<syntax:3:8 single?> (.#<syntax:3:8 single-a>) (#f) #t)

```

代码有点冗长，但发生的事情是 `struct` 宏将 `single` 定义为*语法转换器*。语法转换器始终与*编译时* lambda 关联，`extract-struct-info` 可以查询以获取有关 `struct` 的信息（尽管我们必须使用 `syntax-local-value` 来获取这个 lambda——在编译时 `single` 是未绑定的！）

**讨论。** Racket 的编译时和运行时阶段是一个非常重要的概念。它们有许多后果：

1.  您不需要在编译时运行您的运行时代码，反之亦然。因此，跨编译被支持得非常简单，因为只有您的运行时代码被跨编译。

1.  模块导入分为运行时和编译时导入。这意味着您的编译器只需加载编译时导入到内存中即可运行它们；与模板哈斯克尔不同，后者会将*所有*导入（包括运行时和编译时）加载到 GHC 的地址空间中，以防它们在片段内部被调用。

1.  信息不能从运行时流向编译时：因此任何编译时声明（`define-for-syntax`）都可以简单地在执行扩展之前编译，只需忽略文件中的其他所有内容。

Racket 是正确的，Haskell 是错误的。让我们停止模糊编译时和运行时之间的界限，并且设计一个可行的宏系统。

*附言.* 感谢来自[Mike Sperber](https://twitter.com/sperbsen/status/740411982726234112)的一条推文，它让我思考了这个问题，还有与 Sam Tobin-Hochstadt 有趣的早餐讨论。同时也感谢 Alexis King 帮助我调试`extract-struct-info`代码。

*进一步阅读.* 想要了解更多关于 Racket 的宏阶段，可以查阅文档[编译和运行时阶段](https://docs.racket-lang.org/guide/stx-phases.html)和[通用阶段级别](https://docs.racket-lang.org/guide/phases.html)。此阶段系统也在论文[可组合和可编译的宏](https://www.cs.utah.edu/plt/publications/macromod.pdf)中有所描述。
