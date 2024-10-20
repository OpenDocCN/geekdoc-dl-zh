<!--yml

category: 未分类

date: 2024-07-01 18:18:13

-->

# 读者单子和隐式参数：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/07/implicit-parameters-in-haskell/`](http://blog.ezyang.com/2010/07/implicit-parameters-in-haskell/)

*当读者单子看起来无望笨拙时*

读者单子（也称为环境单子）和隐式参数非常相似，尽管前者是工作中 Haskell 程序员的标准工具，而后者是 GHC 语言扩展的一个使用较少的功能。两者都允许程序员编写代码，就像他们可以访问一个全局环境，这个环境在运行时仍然可以改变。然而，隐式参数非常适合于那些您本来会使用一堆读者转换器的情况。不幸的是，与许多类型系统扩展不同，GHC 不能建议您启用 `ImplicitParams`，因为您无意中编写的代码不是有效的 Haskell98 代码，但如果您启用了此扩展，它将是有效的。本文旨在演示一种发现隐式参数的方式，并略微推动一下。

*实践中的读者单子。* 读者单子实际上非常简单：毕竟，它与 `(->) r` 同构，唯一的真正区别在于新类型。因此，在工程上下文中，它很少原样使用；特别是：

+   它被用作转换器，为您构建的任何特定应用单子提供“环境”，以及

+   它与记录类型一起使用，因为只有一个原始值的环境通常不是很有趣。

这些选择对 Reader 单子编写的代码如何使用施加了一些约束。特别是，将 `ReaderT r` 的环境类型 `r` 内嵌到您的单子代码中意味着，您的单子代码不能轻易地与其他 `ReaderT r2` 的单子代码配合使用；此外，我无法逐步构建一个复杂的记录类型 `Record { field1 :: Int; field2 :: String; field3 :: Bool}`，并在发现环境值时将其放入。我可以将我的记录类型设计为某种映射，在这种情况下，我可以随意在其中放置值，但在这种情况下，我无法静态保证在特定时间点映射中会有哪些值或不会有哪些值。

*堆叠的 Reader 变换器*。为了允许我们逐步构建环境，我们可以考虑堆叠 Reader Monad 变换器。考虑类型 `ReaderT a (ReaderT b (ReaderT c IO)) ()`。如果我们将其解糖成函数应用，我们会得到 `a -> (b -> (c -> IO ()))`，这可以进一步简化为 `a -> b -> c -> IO ()`。如果 `a`、`b` 和 `c` 恰好是相同类型，我们没有办法区分不同的值，除了参数列表中的位置。然而，与在函数签名中明确写出参数不同（事实上我们正试图通过 Reader Monad 避免这种情况），我们发现自己不得不反复使用 `ask`（对于 `a` 不用，对于 `b` 使用一次，对于 `c` 使用两次）。与具有三个字段的记录不同，每个环境变量都没有名称：我们必须使用某些数量的 `ask` 来引用它们。

> *旁注*。事实上，这是[德布鲁因索引](http://zh.wikipedia.org/wiki/De_Bruijn_index)，Oleg 在我们关于[嵌套循环和延续](http://blog.ezyang.com/2010/02/nested-loops-and-continuation/)的文章后，通过电子邮件友好地指出了这一点。升降机的数量就是索引（嗯，维基百科文章是从 1 开始索引的，所以需要加 1），告诉我们需要弹出多少读者绑定作用域。因此，如果我有：
> 
> ```
> runReaderT (runReaderT (runReaderT (lift ask) c) b) a
> \------- outermost/furthest context (3) ------------/
>            \--- referenced context (2; one lift) -/
>                        \--- inner context (1) -/
> 
> ```
> 
> 我得到了`b`的值。这对λ演算理论家来说非常棒（他们对无障碍的α-转换感到高兴），但对软件工程师来说并不是那么理想，因为德布鲁因索引等同于著名的反模式，即魔法数字。

借助类型类技巧，我们可以在某种程度上恢复名称：例如，Dan Piponi [使用单例数据类型或“标签”重命名变换器](http://blog.sigfpe.com/2010/02/tagging-monad-transformer-layers.html)，在此过程中引入了`OverlappingInstances`的强大功能。Oleg [使用与所属层次相关的词法变量类型化](http://okmij.org/ftp/Haskell/regions.html#light-weight)来标识不同的层次，虽然这种方法对于 Reader Monad 堆栈并不真正有用，因为 Reader Monad 的要点不在于必须传递任何词法变量，无论它们是实际变量还是特别类型化的变量。

*隐式参数*。在许多方面，隐式参数是一种欺骗：虽然 Dan 和 Oleg 的方法利用现有的类型级编程设施，隐式参数定义了一个“全局”命名空间（Lisper 们熟知的动态作用域），我们可以在其中放置变量，并且还扩展了类型系统，以便我们可以表达每个函数调用期望存在的这个命名空间中的变量（而无需使用 Monad，这就是它的魔力！）

而不是一个匿名环境，我们为变量赋予一个名称：

```
f :: ReaderT r IO a
f' :: (?implicit_r :: r) => IO a

```

`f'`仍然是单子的，但单子不再表达环境中的内容：完全依赖于类型签名来确定是否传递隐式变量：

```
f  = print "foobar" >> g 42 -- Environment always passed on
f' = print "foobar" >> g 42 -- Not so clear!

```

实际上，`g`也可以是纯计算：

```
f' = print (g 42)

```

然而，如果类型是：

```
g :: IO a

```

隐式变量丢失，而如果它是：

```
g :: (?implicit_r :: r) => IO a

```

变量是可用的。

虽然`runReader(T)`是我们指定环境的方法，但现在我们有了自定义的`let`语法：

```
runReaderT f value_of_r
let ?implicit_r = value_of_r in f

```

除了放弃了我们的单子限制外，我们现在可以轻松地表达我们的增量环境：

```
run = let ?implicit_a = a
          ?implicit_b = b
          ?implicit_c = c
      in h

h :: (?implicit_a :: a, ?implicit_b :: b, ?implicit_c :: c) => b
h = ?implicit_b

```

你也可以使用`where`。请注意，虽然这看起来像是普通的`let`绑定，但实际上有很大不同：你不能混合隐式和普通的变量绑定，如果右侧有同名的隐式绑定，它们指的是`let`之外的值。你不能递归！（回想一下`runReaderT`：我们在第二个参数中提供的值是纯变量，而不是 Reader 单子中的值，尽管通过`>>=`你可以那样处理。）

*良好的实践。* 随着单子结构的消失，在源码级别上少了一些关于单态性约束和多态递归如何应用的提示。非多态递归*将*编译，并导致意外的结果，例如当你期望时，你的隐式参数没有变化。通过确保始终提供带有所有隐式参数的类型签名，你可以相对安全地处理事务。我希望能做一个后续的帖子，更仔细地解释这些语义，基于[相关论文](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.46.9849)中类型的形式描述。
