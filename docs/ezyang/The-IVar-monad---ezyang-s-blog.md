<!--yml

category: 未分类

date: 2024-07-01 18:17:44

-->

# IVar 单子：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/06/the-iva-monad/`](http://blog.ezyang.com/2011/06/the-iva-monad/)

`IVar`是一个[不可变变量](http://blog.ezyang.com/2011/05/reified-laziness/)；你只需写一次，可以多次读取。在`Par`单子框架中，我们使用一种提示单子风格的构造方式来编码对`IVar`的各种操作，这种框架中的确定性并行代码可能会使用。我在本文中感兴趣的问题是这种功能的另一种编码方式，它支持非确定性并发，并出现在其他上下文中，如 Python Twisted、node.js、任何 JavaScript UI 库和 LWT。[许多博客作者](http://amix.dk/blog/post/19509)已经对此进行了评论，[尽管所有关于本质上是诸多回调函数的单子狂热](http://matthew.yumptious.com/2009/04/javascript/dojo-deferred-is-a-monad/)，但实际上却没有人在涉及 Haskell 时*使用*这种单子。为什么呢？一方面，Haskell 具有廉价而轻松的抢先式绿色线程，因此我们可以在许多线程中以同步方式编写我们的 IO。但另一个原因，我将在稍后的博客文章中探讨，是在这个模型空间中天真地实现 bind 会发生泄漏！（大多数事件库已经以某种方式解决了这个问题，我们也将对此进行调查。）

不过，首先要做的是，在 Haskell 中实现`IVar`单子。我们逐步构建它，从演示`IO (IORef a)`是一个单子开始。这并不特别有趣：我们可以使用`IO`获得它的所有特性。我们对它的主要兴趣在于展示我们将呈现非确定性`IVar`单子的基本结构。

```
import Data.IORef

newtype R a = R { runR :: IO (IORef a) }

instance Functor R where
  fmap f m = R $ do xref <- runR m
                    x <- readIORef xref
                    newIORef (f x)

instance Monad R where
  return x = R (newIORef x)
  m >>= f
    = R $ do xref <- runR m
             x <- readIORef xref
             runR (f x)

```

我们绝对不会传递值：相反，我们把它们放在`IORef`盒子里。绑定操作涉及读取盒子的内容，然后从我们要绑定到的函数`f`中取出一个新盒子。我们始终知道盒子的内容：我们从不调用`writeIORef`。还请注意，检索引用是在`IO`中进行的，因此在此过程中可能发生任意其他副作用。当我们有了实际的`IVar`时，这些副作用可能涉及启动另一个执行线程，最终填充`IVar`。请注意这些“盒子”：我们将关注它们的使用属性以提高性能。

对于`IVar`，我们现在希望有“空”盒子，可能只在以后的某个日期被填充。我们可能会被诱惑使用`IO (IORef (Maybe a))`来实现这一点：

```
newtype S a = S { runS :: IO (IORef (Maybe a)) }

instance Monad S where
  return x = S (newIORef (Just x))
  m >>= f
    = S $ do xref <- runS m
             mx <- readIORef xref
             case mx of
               Just x -> runS (f x)
               Nothing -> ???

```

但我们陷入了一种困境（咳嗽）：如果盒子仍然是空的，我们实际上并不知道需要向`f`传递什么值。我们该怎么办？

传统的解决方案是将 `f` 存储起来，以备将来值真正可用时使用，此时我们会用新值调用所有阻塞回调。由于我们的单子允许任意副作用，这些回调仍然可以执行有用的工作。（顺便说一句，`IORef (IVarContents a)` 本质上是 `Par` 单子用来编码 `IVars` 的方式。）

```
data IVarContents a =
    Empty
  | Blocking [a -> IO ()]
  | Full a

newtype T a = T { runT :: IO (IORef (IVarContents a)) }

```

现在我们可以实现最后一个情况：

```
instance Monad T where
  return x = T (newIORef (Full x))
  m >>= f
    = T $ do xref <- runT m
             mx <- readIORef xref
             r <- newIORef Empty
             let callback x = runT (f x >>= fillIVar r) >> return ()
             case mx of
               Full x      -> callback x
               Empty       -> writeIORef xref (Blocking [callback])
               Blocking cs -> writeIORef xref (Blocking (callback:cs))
             return r

```

`filIVar` 是一些神奇的函数，它填充一个空的 IVar，并重新安排等待该值执行的任何人。一个可能的（有点儿愚蠢的）实现，假设单线程操作，可能是：

```
fillIVar :: IORef (IVarContents a) -> a -> T ()
fillIVar ref x = T $ do
  r <- readIORef ref
  writeIORef ref (Full x)
  case r of
    Empty -> newIORef (Full ())
    Blocking cs -> mapM_ ($x) cs >> newIORef (Full ())
    Full _ -> error "fillIVar: Cannot write twice"

```

这一切都非常直接，而且任何合作的非阻塞异步库基本上都重新实现了这种变体。在我的下一篇文章中，我想详细解释这种天真的单子编码存在的一些问题，正如 LWT 的作者所解释的，并准确指出我们在实际中确实看到了这种模式的哪些变体。
