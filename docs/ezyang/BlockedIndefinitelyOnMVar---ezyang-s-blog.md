<!--yml

类别：未分类

日期：2024-07-01 18:17:42

-->

# BlockedIndefinitelyOnMVar：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/07/blockedindefinitelyonmvar/`](http://blog.ezyang.com/2011/07/blockedindefinitelyonmvar/)

*本文摘自我发表在 glasgow-haskell-users 列表中的一篇帖子。*

根据[Control.Exception](http://haskell.org/ghc/docs/6.12.2/html/libraries/base-4.2.0.1/Control-Exception.html#t%3ABlockedIndefinitelyOnMVar)，`BlockedIndefinitelyOnMVar`异常（以及相关的`BlockedIndefinitelyOnSTM`异常）在“线程被阻塞在 MVar 上，但没有其他引用指向 MVar，因此它永远无法继续执行”时被抛出。描述实际上相当精确，但容易被误解。要完全理解此异常的工作原理，需要一些额外来自[Control.Concurrent](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Control-Concurrent.html)的文档，以及对 Haskell 的绿色线程与 GHC 垃圾收集机制的直觉感受。

这是一个试金石测试：您能预测这三个程序会做什么吗？

```
main1 = do
    lock <- newMVar ()
    forkIO $ takeMVar lock
    forkIO $ takeMVar lock
    threadDelay 1000 -- let threads run
    performGC -- trigger exception
    threadDelay 1000

main2 = do
    lock <- newEmptyMVar
    complete <- newEmptyMVar
    forkIO $ takeMVar lock `finally` putMVar complete ()
    takeMVar complete

main3 = do
    lock <- newEmptyMVar
    forkIO $ takeMVar lock `finally` putMVar lock ()
    let loop = do
        b <- isEmptyMVar lock
        if b
            then yield >> performGC >> loop
            else return ()
    loop

```

不要偷看。要获取提示，请查看[forkIO](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Control-Concurrent.html#v:forkIO)的文档。

* * *

第一个程序没有输出，尽管`threadDelay`表面上让两个分叉线程都得以调度、运行并发生死锁。实际上，会引发`BlockedIndefinitelyOnMVar`，而您没有看到的原因是`forkIO`安装了一个异常处理程序，使该异常静音，以及`BlockedIndefinitelyOnSTM`和`ThreadKilled`。您可以使用`catch`及其相关方法安装自己的异常处理程序。

这个程序末尾有一组有趣的额外咒语，确保线程高概率调度，并且抛出`BlockedIndefinitelyOnMVar`异常。请注意，只有在“没有其他引用指向 MVar”时才会抛出该异常。由于 Haskell 是一种垃圾收集语言，它仅在垃圾回收发生时才会发现引用已消失，因此在看到这些错误之前，您需要确保发生其中一个垃圾回收。

这意味着的一个推论是，GHC 不会神奇地知道要将异常抛给哪个线程来“解开”程序的死锁状态：相反，它会将`BlockedIndefinitelyOnMVar`异常抛给所有死锁线程，包括（如果适用）主线程。这种行为在第二个程序中得到了展示，该程序由于主线程获取了异常的副本而以`BlockedIndefinitelyOnMVar`终止，尽管子线程的`finally`处理程序本来会解决死锁。尝试将最后一行替换为``takeMVar complete `catch` \BlockedIndefinitelyOnMVar -> takeMVar complete >> putStrLn "done"``。这真的很滑稽。

最后一个程序考虑了 `MVar` 被“可达”是什么意思。由于它在死锁时是沉默的，这必须意味着 `MVar` 仍然可达；确实，我们的引用 `isEmptyMVar` 防止了 `MVar` 从未变为死的，并且因此我们无限循环，*即使* `MVar` 没有可能被填充。GHC 只知道如果没有引用指向它，线程就可以被视为垃圾（这会导致异常被抛出）。谁持有线程的引用？`MVar`，因为线程 *阻塞* 在这个数据结构上，并将自身添加到该阻塞列表中。谁保持 `MVar` 存活？嗯，我们的闭包中包含一个对 `isEmptyMVar` 的调用。所以线程保持存在。一般规则如下：如果一个线程被阻塞在一个可从非阻塞线程访问的 `MVar` 上，那么该线程就会一直存在。虽然有一些明显的情况（GHC 无法处理），其中 `MVar` 显然已经死了，即使还有引用指向它，但是在一般情况下找出这一点是不可判定的。（练习：编写一个程序来解决停机问题，如果 GHC 能够在一般情况下弄清楚这一点。）

总结起来，没有一点工作（顺便说一句，看起来会很有趣），`BlockedIndefinitelyOnMVar` 并不是一个显而易见的机制，用来给你的 Haskell 程序提供死锁保护。相反，你被邀请把它看作是一种垃圾收集那些本来会无休止地闲置下去的线程的方法：默认情况下，一个死锁的线程是沉默的（除了内存使用方面）。事实上，异常的出现是方便的，从操作的角度来看，但不应依赖这一点。
