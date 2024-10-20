<!--yml

类别：未分类

日期：2024-07-01 18:17:54

-->

# Hoopl 导览：基础系统：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/04/hoopl-guided-tour-base-system/`](http://blog.ezyang.com/2011/04/hoopl-guided-tour-base-system/)

[Hoopl](http://hackage.haskell.org/package/hoopl) 是一个高阶优化库。我们觉得它非常酷！这系列博文旨在向您介绍这个库，作为对[Hoopl 相关论文](http://research.microsoft.com/en-us/um/people/simonpj/papers/c--/)和源代码的教程式补充。我希望这个系列对那些不想使用 Hoopl 编写优化传递的人也有所帮助，但对 Haskell 中高阶 API 设计感兴趣的人也有所帮助。通过本教程的学习，您将能够理解代码中对`analyzeAndRewriteFwd`和`DataflowLattice`等名称的引用，并能够解读诸如以下的类型签名：

```
analyzeAndRewriteFwd
   :: forall m n f e x entries. (CheckpointMonad m, NonLocal n, LabelsPtr entries)
   => FwdPass m n f
   -> MaybeC e entries
   -> Graph n e x -> Fact e f
   -> m (Graph n e x, FactBase f, MaybeO x f)

```

我们假设您对函数式编程和编译技术有基本的了解，但我会在适当的地方介绍适当的基本概念。

> *旁白：介绍。* 对于已经熟悉正在讨论的主题的人来说，可以自由跳过像这样格式化的部分。

* * *

我们将导览 Hoopl 的`testing`子目录，其中包含一个示例客户端。（你可以通过克隆 Git 仓库`git clone git://ghc.cs.tufts.edu/hoopl/hoopl.git`获取一份副本）。你可以通过查看`README`文件来了解基础情况。我们首先将查看“Base System”，该系统定义了*抽象语法树*和 Hoopl 化的*中间表示*的数据类型。

抽象语法与标准一样（`Ast.hs`）：

```
data Proc = Proc { name :: String, args :: [Var], body :: [Block] }
data Block = Block { first :: Lbl, mids :: [Insn], last :: Control }
data Insn = Assign Var  Expr
          | Store  Expr Expr
data Control = Branch Lbl
             | Cond   Expr   Lbl    Lbl
             | Call   [Var]  String [Expr] Lbl
             | Return [Expr]
type Lbl = String

```

我们有一种命名的过程语言，它由基本块组成。我们支持无条件分支`Branch`，条件分支`Cond`，函数调用`Call`（`[Var]`是存储返回值的变量，`String`是函数名，`[Expr]`是参数，`Lbl`是函数调用完成后跳转的位置），以及函数返回`Return`（支持多返回值，因此使用`[Expr]`而不是`Expr`）。

我们没有任何高级流控制结构（这种语言的控制流思想就是大量的 goto 语句——不用担心，这对我们有利），所以我们可能会期望将这种“高级汇编语言”相对容易地映射到机器代码中，事实上确实如此（但需要注意的是，这种语言不需要考虑寄存器分配，但我们如何使用变量将明显影响寄存器分配）。高级汇编语言的真实世界例子包括 C--。

这里是一个可能在这种语言中编写的代码的简单示例：

> *旁白：基本块。* 完全解释什么是抽象语法树（AST）略有超出本文的范围，但如果你知道如何在 Haskell 中编写 [Scheme 解释器](http://en.wikibooks.org/wiki/Write_Yourself_a_Scheme_in_48_Hours/Parsing)，你已经了解了语言的 *表达式* 组件的大部分内容：例如二元运算符和变量（例如 `a + b`）。然后，我们以明显的方式扩展这个计算器，加入低级别的命令式特性。如果你已经做过任何命令式编程，大多数这些特性也是熟悉的（分支、函数调用、变量赋值）：唯一的新概念是 [基本块](http://en.wikipedia.org/wiki/Basic_block)。基本块是流程控制的原子单位：如果我进入了一个基本块，我知道我会从另一端出来，不管怎样。这意味着在这个块的内部不会有非本地的控制转移（例如异常），也不会有能够跳到这个块 *内部* 的代码（例如 goto）。任何控制流发生在基本块的末尾，我们可能无条件地跳转到另一个块，或者进行函数调用等操作。真实的程序不会以这种方式编写，但我们可以轻松地将它们转换成这种形式，并且我们希望采用这种表示方式，因为它将更容易进行数据流分析。因此，我们的示例抽象语法树实际上并不像你会编程的命令式语言，但它很容易成为代码生成的目标，所以示例抽象语法树以这种方式设置。

Hoopl 是对底层表示的抽象，但不幸的是，我们不能直接使用这个 AST；Hoopl 有自己的图表示。无论如何，我们也不想使用我们自己的表示方式：我们已经将控制流图表示为块列表 `[Block]`。如果我想取出某个特定标签的块，我必须遍历整个列表。与其发明自己更高效的块表示（类似于标签到块的映射），不如使用 Hoopl 给我们提供的表示 `Graph n e x`（毕竟它将要在这个表示上操作）。`n` 代表“节点”，你提供构成图节点的数据结构，而 Hoopl 管理图本身。`e` 和 `x` 参数将用于存储关于节点形状的信息，不代表任何特定数据。

这里是我们的中间表示（`IR.hs`）：

```
data Proc = Proc { name :: String, args :: [Var], entry :: Label, body :: Graph Insn C C }
data Insn e x where
  Label  :: Label  ->                               Insn C O
  Assign :: Var    -> Expr    ->                    Insn O O
  Store  :: Expr   -> Expr    ->                    Insn O O
  Branch :: Label  ->                               Insn O C
  Cond   :: Expr   -> Label   -> Label  ->          Insn O C
  Call   :: [Var]  -> String  -> [Expr] -> Label -> Insn O C
  Return :: [Expr] ->                               Insn O C

```

显著的差异是：

+   `Proc` 的主体是 `Graph Insn C C`，而不是 `[Block]`。此外，由于 `Graph` 没有“第一个”块的概念，我们必须用 `entry` 明确指出入口点是什么。

+   我们不再使用 `String` 作为 `Lbl`，而是切换到了 Hoopl 提供的 `Label` 数据类型。

+   `Insn`、`Control` 和 `Label` 都被合并为一个 `Insn` 广义抽象数据类型（GADT），它处理所有这些情况。

然而，重要的是，我们通过`e`和`x`参数保留了关于节点是什么*形状*的信息。`e`代表进入，`x`代表退出，`O`代表开放，`C`代表关闭。每个指令都有一个形状，你可以想象成一系列的管道，它们是相互连接的。形状为`C O`（进入时关闭，退出时开放）的管道开始了这个块，形状为`O C`（进入时开放，退出时关闭）的管道结束了这个块，而在中间可以有任意数量的`O O`管道。我们可以看到`Insn C O`对应于我们旧的数据类型`Ast.Lbl`，`Insn O O`对应于`Ast.Insn`，而`Insn O C`对应于`Ast.Control`。当我们把节点放在一起时，我们得到了图，它也可以是各种开放或关闭的。

> *另外：广义抽象数据类型.* [GADTs](http://en.wikibooks.org/wiki/Haskell/GADT) 是类型级编程中不可或缺的瑞士军刀。在这个另外的部分中，我们简要描述了一些可以与我们上面提到的`Insn e x`一起使用的技巧（类似子类型化）。
> 
> 第一个“技巧”是你可以完全忽略幻象类型变量，并且像使用普通数据类型`Insn`一样使用它：
> 
> ```
> isLabel :: Insn e x -> Bool
> isLabel Label{} = True
> isLabel _ = False
> 
> ```
> 
> 我可以把一个`Label`传给这个函数，它将返回`True`，或者我可以传一个`Branch`给它，它将返回`False`。在这个特定的例子中，对 GADT 进行模式匹配不会导致我关心的类型细化，因为在任何构造函数或函数返回类型中都没有类型变量`e`或`x`。
> 
> 当然，我可以以一种不可能将不是`Label`的东西传递给它的方式编写这个函数：
> 
> ```
> assertLabel :: Insn C O -> Bool
> assertLabel Label{} = True
> 
> ```
> 
> 如果你尝试调用`assertLabel (Branch undefined)`，你将会从 GHC 得到这个很好的类型错误：
> 
> ```
> <interactive>:1:13:
>     Couldn't match expected type `C' against inferred type `O'
>       Expected type: Insn C O
>       Inferred type: Insn O C
>     In the first argument of `assertLabel', namely `(Branch undefined)'
>     In the expression: assertLabel (Branch undefined)
> 
> ```
> 
> 让我们来解析一下：任何构造函数`Branch`都将得到一个类型为`Insn O C`的值。然而，我们函数的类型签名却声明了`Insn C O`，而且`C ≠ O`。这个类型错误非常直接明了，足以告诉我们出了什么问题！
> 
> 同样地，我可以编写一个不同的函数：
> 
> ```
> transferMiddle :: Insn O O -> Bool
> transferMiddle Assign{} = True
> transferMiddle Store{} = False
> 
> ```
> 
> 在类型级别上没有办法区分`Assign`和`Store`，但是我不必对数据类型中的任何其他内容进行模式匹配：`Insn O O`意味着我只需要处理符合这种形状的构造函数。
> 
> 我甚至可以部分指定允许的形状是什么：
> 
> ```
> transferMiddleOrEnd :: Insn O x -> Bool
> 
> ```
> 
> 对于这个函数，我需要对指令和控制操作进行模式匹配，但是*不*需要针对`IR.Label`进行模式匹配。这并不是我在原始 AST 中容易做到的事情：我本来需要创建一个和类型`Ast.InsnOrControl`对应的求和类型。
> 
> *快速问题.* 如果我有一个以`Insn e x`作为参数的函数，并且我想把这个值传给一个只接受`Insn C x`的函数，我该怎么做？另一种情况呢？
> 
> *练习.* 假设你正在为 Hoopl 设计一个 `Graph` 表示，但不能使用 GADTs。`Graph IR.Insn`（其中 `IR.Insn` 就像我们的 `IR` GADT，但没有幻象类型）和 `Graph Ast.Label Ast.Insn Ast.Control` 之间的表示有什么区别？

我们今天将看的最后一个文件是一些管道工作，用于将抽象语法树转换为中间表示，`Ast2ir.hs`。由于存在一些名称重载，我们使用 `A.` 作为前缀来区分来自 `Ast` 的数据类型和 `I.` 来自 `IR` 的数据类型。主要函数是 `astToIR`：

```
astToIR :: A.Proc -> I.M I.Proc
astToIR (A.Proc {A.name = n, A.args = as, A.body = b}) = run $
  do entry <- getEntry b
     body  <- toBody   b
     return $ I.Proc { I.name  = n, I.args = as, I.body = body, I.entry = entry }

```

代码是单子的，因为当我们将字符串转换为标签（在内部是任意的唯一整数）时，我们需要跟踪我们已经分配的标签。单子本身是一个普通的状态单子变换器，位于“新标签”单子之上。（实际上，堆栈中还有另一个单子；请参阅 `IR.M` 获取更多细节，但在这个阶段它没有被使用，所以我们忽略它。）

`getEntry` 查看过程主体中的第一个块，并使用它确定入口点：

```
getEntry :: [A.Block] -> LabelMapM Label
getEntry [] = error "Parsed procedures should not be empty"
getEntry (b : _) = labelFor $ A.first b

```

`labelFor` 是一个单子函数，如果我们之前没有见过字符串 `Lbl`，它会给我们一个新的标签，否则会返回已存在的标签。

`toBody` 使用了一些更有趣的 Hoopl 函数：

```
toBody :: [A.Block] -> LabelMapM (Graph I.Insn C C)
toBody bs =
  do g <- foldl (liftM2 (|*><*|)) (return emptyClosedGraph) (map toBlock bs)
     getBody g

```

Hoopl 提供的函数包括 `|*><*|` 和 `emptyClosedGraph`。请注意，Hoopl 图不必连接（即它们可以包含多个基本块），因此 `|*><*|` 是一种图连接运算符，将两个封闭的图连接在一起（`Graph n e C -> Graph n C x -> Graph n e x`），可能通过间接控制操作符连接在一起（我们除了在运行时无法知道这一点外，还用红色箭头画出）。这是一个有些笨拙的运算符，因为 Hoopl 希望尽可能使用 `<*>`。

`toBlock` 给出了 `<*>` 的一个例子：

```
toBlock :: A.Block -> LabelMapM (Graph I.Insn C C)
toBlock (A.Block { A.first = f, A.mids = ms, A.last = l }) =
  do f'  <- toFirst f
     ms' <- mapM toMid ms
     l'  <- toLast l
     return $ mkFirst f' <*> mkMiddles ms' <*> mkLast l'

```

我们从底部向上工作。`mkFirst f'`、`mkMiddle ms'` 和 `mkLast l'` 的类型是什么？它们都是 `(Graph I.Insn e x)`，但 `f'` 是 `C O`，`ms'` 是 `O O`，`l'` 是 `O C`。我们建立部分图形，这些图形两侧未封闭，然后使用 `<*>` 将它们连接在一起：`Graph n e O -> Graph n O x -> Graph n e x`。`mkFirst`、`mkMiddles` 和 `mkLast` 是由 Hoopl 提供的函数，将 `I.Insn e x` 提升为 `(Graph I.Insn e x)`（或者在 `mkMiddles` 的情况下是 `[I.Insn O O]`）。

最后，`toFirst`、`toMid` 和 `toLast` 实际上执行了翻译：

```
toFirst :: A.Lbl -> LabelMapM (I.Insn C O)
toFirst = liftM I.Label . labelFor

toMid :: A.Insn -> LabelMapM (I.Insn O O)
toMid (A.Assign v e) = return $ I.Assign v e
toMid (A.Store  a e) = return $ I.Store  a e

toLast :: A.Control -> LabelMapM (I.Insn O C)
toLast (A.Branch l)   = labelFor l >>= return . I.Branch
toLast (A.Cond e t f) = labelFor t >>= \t' ->
                        labelFor f >>= \f' -> return (I.Cond e t' f')
toLast (A.Call rs f as l) = labelFor l >>= return . I.Call rs f as
toLast (A.Return es)      = return $ I.Return es

```

注意，我们仔细指定返回形状，以便可以使用 `mkFirst`、`mkMiddles` 和 `mkLast`。最有趣的事情是，我们必须将 `Lbl` 字符串转换为 `Label`；否则，代码就是琐碎的。

数据表示到此结束，下次我们将看看 Hoopl 中的数据流事实分析。
