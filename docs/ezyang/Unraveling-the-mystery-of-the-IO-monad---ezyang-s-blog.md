<!--yml

category: 未分类

date: 2024-07-01 18:17:52

-->

# 揭示 IO 单子的奥秘：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/05/unraveling-the-mystery-of-the-io-monad/`](http://blog.ezyang.com/2011/05/unraveling-the-mystery-of-the-io-monad/)

当我们向初学者教授 Haskell 时，我们需要讨论的一件事是 IO 单子的工作原理。是的，它是一个单子，是的，它执行 IO 操作，但它不是你可以在 Haskell 中实现的东西，这使得它具有某种神奇的品质。在今天的帖子中，我想通过描述 GHC 如何在基本操作和真实世界令牌的术语中实现 IO 单子来揭示 IO 单子的奥秘。阅读完本文后，你应该能够理解这个票据的解决方案以及这个 Hello World! 程序的 Core 输出：

```
main = do
    putStr "Hello "
    putStrLn "world!"

```

Nota bene: **这不是单子教程**。本文假设读者知道单子是什么！然而，第一部分回顾了严格性作为单子应用的一个关键概念，因为它对 IO 单子的正确功能至关重要。

### 惰性和严格的 State 单子

作为 IO 单子的序曲，我们将简要回顾 State 单子，它构成了 IO 单子的操作基础（IO 单子被实现为一个带有特殊状态的严格 State 单子，尽管有一些重要的区别——这就是其魔力所在）。如果你对惰性和严格状态单子之间的区别感到舒适，可以跳过本节。否则，请继续阅读。State 单子的数据类型构造器如下：

```
newtype State s a = State { runState :: s -> (a, s) }

```

在状态单子中运行计算涉及给它一些输入状态，并从中检索出结果状态和计算的实际值。单子结构涉及通过各种计算来*穿越*状态。例如，状态单子中的这段代码片段：

```
do x <- doSomething
   y <- doSomethingElse
   return (x + y)

```

可以重写（去掉 newtype 构造器后）如下：

```
\s ->
let (x, s')  = doSomething s
    (y, s'') = doSomethingElse s' in
(x + y, s'')

```

现在，我想向读者提出一个相当有趣的实验：假设 `doSomething` 和 `doSomethingElse` 被跟踪：即，在评估时，它们输出一个跟踪消息。也就是说：

```
doSomething s = trace "doSomething" $ ...
doSomethingElse s = trace "doSomethingElse" $ ...

```

在 `doSomething` 的结果被强制执行之后，`doSomethingElse` 的跟踪是否会在其之前触发？在严格语言中，答案显然是否定的；你必须按顺序执行每个状态计算步骤。但 Haskell 是惰性的，在另一种情况下，`doSomethingElse` 的结果可能在 `doSomething` 之前被请求。确实，这里有一个这样的代码示例：

```
import Debug.Trace

f = \s ->
        let (x, s')  = doSomething s
            (y, s'') = doSomethingElse s'
        in (3, s'')

doSomething s = trace "doSomething" $ (0, s)
doSomethingElse s = trace "doSomethingElse" $ (3, s)

main = print (f 2)

```

发生的情况是，我们对状态值是惰性的，因此当我们要求 `s''` 的值时，我们强制执行了 `doSomethingElse` 并得到了一个指向 `s'` 的间接引用，然后导致我们强制执行了 `doSomething`。

假设我们确实希望 `doSomething` 总是在 `doSomethingElse` 之前执行。在这种情况下，我们可以通过使我们的状态严格化来解决问题：

```
f = \s ->
        case doSomething s of
            (x, s') -> case doSomethingElse s' of
                          (y, s'') -> (3, s'')

```

这种从惰性 `let` 到严格 `case` 的微妙转换让我们现在可以保持顺序。事实上，事情会变得明朗：由于原语的工作方式，我们必须按照这种方式来做事情。留意 `case`：当我们开始查看 Core 时，它会再次出现。

*额外内容*。有趣的是，如果你使用不可否认的模式，`case` 代码等同于原始的 `let` 代码：

```
f = \s ->
        case doSomething s of
            ~(x, s') -> case doSomethingElse s' of
                          ~(y, s'') -> (3, s'')

```

### 原语

我们故事的下一部分是 GHC 提供的原始类型和函数。这些机制是 GHC 导出类型和功能的方式，这些功能通常在 Haskell 中是无法实现的：例如，非装箱类型、两个 32 位整数相加，或执行 IO 操作（主要是将位写入内存位置）。它们非常特定于 GHC，普通的 Haskell 用户从不见它们。事实上，它们如此特殊，你需要启用一个语言扩展来使用它们（`MagicHash`）！IO 类型是用 `GHC.Types` 中的这些原语构建的：

```
newtype IO a = IO (State# RealWorld -> (# State# RealWorld, a #))

```

为了理解 `IO` 类型，我们将需要了解这些原语中的一些。但很明显，这看起来非常像状态单子...

第一个原语是 *非装箱元组*，在代码中看到的形式为 `(# x, y #)`。非装箱元组是一种“多返回”调用约定的语法；它们实际上并不是真正的元组，不能像普通元组那样放在变量中。我们将使用非装箱元组来代替我们在 `runState` 中看到的元组，因为如果每次执行 IO 操作都要进行堆分配，那将是非常糟糕的。

下一个原语是 `State# RealWorld`，它将对应于我们状态单子的 `s` 参数。实际上，这是两个原语，类型构造子 `State#` 和魔术类型 `RealWorld`（有趣的是，它没有 `#` 后缀）。之所以将其分为类型构造子和类型参数，是因为 `ST` 单子也重用了这个框架，但这是另一篇博文的事情。你可以将 `State# RealWorld` 视为表示非常神奇值的类型：整个真实世界的值。当你运行一个状态单子时，你可以用任何你准备好的值初始化状态，但只有 `main` 函数接收到真实世界，并且它随后会在你可能要执行的任何 IO 代码中进行线程处理。

你可能会问一个问题：“`unsafePerformIO` 怎么办？”特别是，由于它可能出现在任何纯计算中，而真实世界可能不一定可用，我们如何虚拟出真实世界的副本来执行等同于嵌套 `runState` 的操作？在这些情况下，我们有一个最终的原语，`realWorld# :: State# RealWorld`，它允许您在任何地方获取对真实世界的引用。但由于这不是与 `main` 钩连的，你绝对不会得到任何顺序保证。

### 你好，世界

让我们回到我答应要解释的 Hello World 程序：

```
main = do
    putStr "Hello "
    putStrLn "world!"

```

当我们编译这个程序时，我们会得到一些核心代码，看起来像这样（某些部分，尤其是强制转换（虽然这是展示新类型如何工作的迷人演示，但在运行时没有影响），已经为了您的观看愉快修剪）：

```
Main.main2 :: [GHC.Types.Char]
Main.main2 = GHC.Base.unpackCString# "world!"

Main.main3 :: [GHC.Types.Char]
Main.main3 = GHC.Base.unpackCString# "Hello "

Main.main1 :: GHC.Prim.State# GHC.Prim.RealWorld
              -> (# GHC.Prim.State# GHC.Prim.RealWorld, () #)
Main.main1 =
  \ (eta_ag6 :: GHC.Prim.State# GHC.Prim.RealWorld) ->
    case GHC.IO.Handle.Text.hPutStr1
           GHC.IO.Handle.FD.stdout Main.main3 eta_ag6
    of _ { (# new_s_alV, _ #) ->
    case GHC.IO.Handle.Text.hPutStr1
           GHC.IO.Handle.FD.stdout Main.main2 new_s_alV
    of _ { (# new_s1_alJ, _ #) ->
    GHC.IO.Handle.Text.hPutChar1
      GHC.IO.Handle.FD.stdout System.IO.hPrint2 new_s1_alJ
    }
    }

Main.main4 :: GHC.Prim.State# GHC.Prim.RealWorld
              -> (# GHC.Prim.State# GHC.Prim.RealWorld, () #)
Main.main4 =
  GHC.TopHandler.runMainIO1 @ () Main.main1

:Main.main :: GHC.Types.IO ()
:Main.main =
  Main.main4

```

重要的部分是`Main.main1`。重新格式化并重命名后，它看起来就像我们的去糖化状态单子：

```
Main.main1 =
  \ (s :: State# RealWorld) ->
    case hPutStr1 stdout main3 s  of _ { (# s', _ #) ->
    case hPutStr1 stdout main2 s' of _ { (# s'', _ #) ->
    hPutChar1 stdout hPrint2 s''
    }}

```

单子都消失了，而`hPutStr1 stdout main3 s`，虽然表面上总是返回类型为`(# State# RealWorld, () #)`的值，但却具有副作用。然而，重复的 case 表达式确保我们的优化器不会重新排列 IO 指令（因为那会有非常明显的效果！）

对于那些好奇的人，这里有一些关于核心输出的其他显著部分：

+   我们的`:main`函数（前面带有冒号）实际上并没有直接进入我们的代码：它调用了一个包装函数`GHC.TopHandler.runMainIO`，该函数做一些初始化工作，比如安装顶级中断处理程序。

+   `unpackCString#`的类型是`Addr# -> [Char]`，它的作用是将以空字符结尾的 C 字符串转换为传统的 Haskell 字符串。这是因为我们尽可能地将字符串存储为以空字符结尾的 C 字符串。如果嵌入了空字节或其他恶意的二进制数据，则会使用`unpackCStringUtf8#`。

+   `putStr`和`putStrLn`不见了。这是因为我使用了`-O`进行了编译，所以这些函数调用被内联了。

### 有序的重要性

为了强调顺序的重要性，请考虑当你混淆`seq`（传统上用于纯代码，不提供任何顺序约束）和对于 IO 非常重要的 IO 时会发生什么。也就是说，请考虑[Bug 5129](http://hackage.haskell.org/trac/ghc/ticket/5129)。Simon Peyton Jones 给出了一个很好的解释，所以我只想强调那些没有正确排序的代码是多么诱人（以及错误）。有问题的代码是``x `seq` return ()``。这会编译成以下核心代码：

```
case x of _ {
  __DEFAULT -> \s :: State# RealWorld -> (# s, () #)
}

```

请注意，`seq`编译成一个`case`语句（因为 Core 中的 case 语句是严格的），并且还请注意，此语句中的`s`参数没有涉及。因此，如果此片段包含在较大的片段中，则这些语句可能会被优化。实际上，在某些情况下确实会发生这种情况，正如 Simon 所描述的那样。故事的寓意？不要写``x `seq` return ()``（确实，我认为某些基础库中有这种习惯用法的实例需要修复）。新世界秩序是一个新的 primop：

```
case seqS# x s of _ {
  s' -> (# s', () #)
}

```

更好！

这也说明了为什么`seq x y`绝对不保证`x`或`y`哪个先评估。优化器可能注意到`y`总是引发异常，而由于不精确的异常不关心抛出哪个异常，它可能会丢弃对`x`的任何引用。天哪！

### 进一步阅读

+   大部分定义 IO 的代码位于 `base` 中的 `GHC` 超模块中，虽然实际的 IO 类型在 `ghc-prim` 中。`GHC.Base` 和 `GHC.IO` 特别适合阅读。

+   Primops 的描述在 [GHC Trac](http://hackage.haskell.org/trac/ghc/wiki/Commentary/PrimOps) 上详细说明。

+   ST 单子的实现方式基本上也完全相同：不安全的强制转换函数只是进行一些类型重排，实际上并未改变任何内容。你可以在 `GHC.ST` 中进一步阅读。
