<!--yml

类别：未分类

日期：2024-07-01 18:18:00

-->

# 关于 MVars 的全部内容：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/02/all-about-mvars/`](http://blog.ezyang.com/2011/02/all-about-mvars/)

我最近花了时间重新编写了[MVar 文档](http://www.haskell.org/ghc/docs/latest/html/libraries/base/Control-Concurrent-MVar.html)，目前文档内容相对较少（简介部分非常简洁地说明了“同步变量”；尽管原始作者在数据类型和其基本操作的内联文档方面做得相当详尽。）我在这里复制了我的新简介。

在研究此文档时，我发现了有关 MVars 如何工作的新内容，这体现在这个程序中。它做什么？

```
import Control.Concurrent.MVar
import Control.Concurrent
main = do
    x <- newMVar 0
    forkIO $ do
        putMVar x 1
        putStrLn "child done"
    threadDelay 100
    readMVar x
    putStrLn "parent done"

```

* * *

`MVar t`是一个可变位置，可以是空的，也可以包含类型为`t`的值。它有两个基本操作：`putMVar`，如果 MVar 为空则填充并阻塞，否则阻塞；`takeMVar`，如果 MVar 为满则清空并阻塞，否则阻塞。它们可以以多种不同的方式使用：

1.  作为同步可变变量，

1.  作为通道，使用`takeMVar`和`putMVar`作为接收和发送，

1.  作为二进制信号量`MVar ()`，使用`takeMVar`和`putMVar`作为等待和信号。

它们是由 Simon Peyton Jones、Andrew Gordon 和 Sigbjorn Finne 在论文“Concurrent Haskell”中引入的，尽管其实现的一些细节已经发生了变化（特别是，对满 MVar 的放置曾经导致错误，但现在仅仅阻塞。）

### 适用性

MVars 比 IORefs 提供更多的灵活性，但比 STM 提供的灵活性更少。它们适用于构建同步原语和执行简单的线程间通信；然而，它们非常简单且容易受到竞争条件、死锁或未捕获的异常的影响。如果需要执行更大的原子操作（例如从多个变量读取），请使用“STM”。

特别是，在本模块中的“大”函数（`readMVar`，`swapMVar`，`withMVar`，`modifyMVar_`和`modifyMVar`）只是一个`takeMVar`后跟一个带有异常安全的`putMVar`的组合。只有当所有其他线程在`putMVar`之前执行`takeMVar`时，它们才具有原子性保证；否则，它们可能会阻塞。

### 公平性

原始论文规定，除非另一个线程无限期地持有该 MVar，否则不能有任何线程在 MVar 上被阻塞。通过以先进先出的方式为阻塞在 MVar 上的线程提供服务，此实现维护了这一公平性质。

### 注意事项

与许多其他 Haskell 数据结构一样，MVars 是惰性的。这意味着如果您将一个昂贵的未求值的 thunk 放入 MVar 中，它将由消费它的线程求值，而不是产生它的线程。确保将要放入 MVar 中的值评估为适当的正常形式，或者利用由[strict-concurrency 包](http://hackage.haskell.org/package/strict-concurrency)提供的严格 MVar。

### 示例

考虑以下并发数据结构，跳过通道。这是用于间歇性高带宽信息源（例如，鼠标移动事件）的通道。写入通道永远不会阻塞，从通道读取仅返回最新值，或者如果没有新值则阻塞。支持多个读取器，有一个`dupSkipChan`操作。

跳过通道是一对 MVars：第二个 MVar 是特定读取器的信号量：如果通道中有该读取器尚未读取的值，则为满，否则为空。

```
import Control.Concurrent.MVar
import Control.Concurrent

data SkipChan a = SkipChan (MVar (a, [MVar ()])) (MVar ())

newSkipChan :: IO (SkipChan a)
newSkipChan = do
    sem <- newEmptyMVar
    main <- newMVar (undefined, [sem])
    return (SkipChan main sem)

putSkipChan :: SkipChan a -> a -> IO ()
putSkipChan (SkipChan main _) v = do
    (_, sems) <- takeMVar main
    putMVar main (v, [])
    mapM_ (\sem -> putMVar sem ()) sems

getSkipChan :: SkipChan a -> IO a
getSkipChan (SkipChan main sem) = do
    takeMVar sem
    (v, sems) <- takeMVar main
    putMVar main (v, sem:sems)
    return v

dupSkipChan :: SkipChan a -> IO (SkipChan a)
dupSkipChan (SkipChan main _) = do
    sem <- newEmptyMVar
    (v, sems) <- takeMVar main
    putMVar main (v, sem:sems)
    return (SkipChan main sem)

```

该示例改编自原始的 Concurrent Haskell 论文。有关使用 MVars 构建更高级同步原语的更多示例，请参见[Control.Concurrent.Chan](http://www.haskell.org/ghc/docs/latest/html/libraries/base/Control-Concurrent-Chan.html)和[Control.Concurrent.QSem](http://www.haskell.org/ghc/docs/latest/html/libraries/base/Control-Concurrent-QSem.html)。
