<!--yml

类别：未分类

日期：2024-07-01 18:18:10

-->

# 中断 GHC：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/08/interrupting-ghc/`](http://blog.ezyang.com/2010/08/interrupting-ghc/)

在我的[有关 abcBridge 的技术讲座](http://blog.ezyang.com/2010/08/galois-tech-talk-abcbridge-functional-interfaces-for-aigs-and-sat-solving/)中，我在将 FFI 代码用作普通 Haskell 代码时遇到的一个“未解决”问题是中断处理。在这里，我描述了一种涉及 GHC 运行时系统更改的实验性解决方案，这是由[Simon Marlow](http://permalink.gmane.org/gmane.comp.lang.haskell.glasgow.user/18771)建议的。导论部分可能对寻找代码工作示例的从业者感兴趣，用于捕获信号的代码；后面的部分是我希望能够完全完成的补丁的概念验证。

```
> {-# LANGUAGE ForeignFunctionInterface #-}
> {-# LANGUAGE DeriveDataTypeable #-}
> {-# LANGUAGE ScopedTypeVariables #-}
>
> import qualified Control.Exception as E
>
> import Foreign.C.Types (CInt)
>
> import Control.Monad
> import Control.Concurrent (threadDelay, myThreadId, throwTo, forkIO)
> import Control.Concurrent.MVar (newEmptyMVar, putMVar, readMVar)
>
> import System.IO (hPutStrLn, stderr)
> import System.Posix.Signals (installHandler, sigINT, Handler(..))

```

* * *

在许多交互式应用程序（特别是 REPL）中，您希望能够捕获用户按下`^C`时终止当前计算，而不是整个程序。`fooHs`是一个可能需要很长时间才能运行的函数（在这种情况下，`fooHs`永远不会终止）。

```
> fooHs :: Int -> IO Int
> fooHs n = do
>     putStrLn $ "Arf HS " ++ show n
>     threadDelay 1000000
>     fooHs n

```

默认情况下，GHC 生成一个异步异常，我们可以使用正常的异常处理设施来捕获，以表明“还不要退出”：

```
> reallySimpleInterruptible :: a -> IO a -> IO a
> reallySimpleInterruptible defaultVal m = do
>     let useDefault action =
>             E.catch action
>                 (\(e :: E.AsyncException) ->
>                     return $ case e of
>                         E.UserInterrupt -> defaultVal
>                         _ -> E.throw e
>                         )
>     useDefault m
>
> reallySimpleMain = do
>     r <- reallySimpleInterruptible 42 (fooHs 1)
>     putStrLn $ "Finished with " ++ show r

```

有时，您不希望生成任何异常，并希望在信号到达时立即进行研究。您可能处于程序的某个关键部分，不希望被中断！在这种情况下，您可以使用来自[System.Posix.Signals](http://www.haskell.org/ghc/docs/6.12-latest/html/libraries/unix-2.4.0.1/System-Posix-Signals.html)的`installHandler`安装信号处理程序。

```
> installIntHandler :: Handler -> IO Handler
> installIntHandler h = installHandler sigINT h Nothing

```

在完成后，应确保恢复原始的信号处理程序。

如果您决定从信号处理程序内生成异常，需要小心处理：如果我们仅尝试简单地抛出异常，我们的异常似乎会消失到虚无中！这是因为中断处理程序在不同的线程上运行，我们必须使用来自[Control.Concurrent](http://www.haskell.org/ghc/docs/6.12.2/html/libraries/base-4.2.0.1/Control-Concurrent.html)的`throwTo`确保我们的异常发送到正确的线程。

```
> simpleInterruptible :: a -> IO a -> IO a
> simpleInterruptible defaultVal m = do
>     tid <- myThreadId
>     let install = installIntHandler (Catch ctrlc)
>         ctrlc = do
>             -- This runs in a different thread!
>             hPutStrLn stderr "Caught signal"
>             E.throwTo tid E.UserInterrupt
>         cleanup oldHandler = installIntHandler oldHandler >> return ()
>         useDefault action =
>             E.catch action
>                 (\(e :: E.AsyncException) ->
>                     return $ case e of
>                         E.UserInterrupt -> defaultVal
>                         _ -> E.throw e
>                         )
>     useDefault . E.bracket install cleanup $ const m
>
> simpleMain = do
>     r <- simpleInterruptible 42 (fooHs 1)
>     putStrLn $ "Finished with " ++ show r

```

这段代码对纯 Haskell 工作很好。

* * *

然而，我们的问题是，我们是否可以中断处于 FFI 中的 Haskell 线程，而不仅仅是纯 Haskell 代码。也就是说，我们想用`fooHs`替换：

```
> foreign import ccall "foo.h" foo :: CInt -> IO ()

```

其中`foo.h`包含：

```
void foo(int);

```

而`foo.c`包含：

```
#include <stdio.h>
#include "foo.h"

void foo(int d) {
    while (1) {
        printf("Arf C %d!\n", d);
        sleep(1);
    }
}

```

在实际应用中，`foo`将是一些在 C 语言中编写的高度优化的函数，可能需要很长时间才能运行。我们也不能随意终止函数：我们应该能够随时强制终止它，而不会破坏一些全局状态。

如果我们尝试使用现有的`interruptible`函数，我们发现它们不起作用：

+   `reallySimpleInterruptible` 注册了 SIGINT，但外部调用仍在继续。在第二个 SIGINT 上，程序终止。这是运行时系统的 [默认行为](http://hackage.haskell.org/trac/ghc/wiki/Commentary/Rts/Signals)：RTS 会试图优雅地中止计算，但无法终止 FFI 调用，并在第二个 SIGINT 到来时强制终止程序。

+   `simpleInterruptible` 的表现甚至更糟：没有“在第二个信号上退出”的行为，我们发现无法通过按 `^C` 来终止程序！请求 FFI 调用的线程正在忽略我们的异常。

* * *

*Nota bene.* 请告知作者本节中的任何事实错误。

是时候深入了解运行时系统了！管理异步异常的代码位于 `rts` 目录下的 `RaiseAsync.c` 中。特别是这个函数：

```
nat throwToMsg (Capability *cap, MessageThrowTo *msg)

```

当线程调用 `throwTo` 在另一个线程中创建异常时会调用。

首先看一下没有任何幽默的情况发生时会发生什么，也就是说，当线程没有被阻塞时：

```
case NotBlocked:
{
    if ((target->flags & TSO_BLOCKEX) == 0) {
        // It's on our run queue and not blocking exceptions
        raiseAsync(cap, target, msg->exception, rtsFalse, NULL);
        return THROWTO_SUCCESS;
    } else {
        blockedThrowTo(cap,target,msg);
        return THROWTO_BLOCKED;
    }
}

```

如果线程正常运行，我们使用 `raiseAsync` 来引发异常，然后完成！然而，线程可能已调用 `block`（来自 [Control.Exception](http://haskell.org/ghc/docs/6.12.2/html/libraries/base-4.2.0.1/Control-Exception.html#v%3Ablock)），在这种情况下，我们将异常添加到目标的阻塞异常队列，并等待目标解除阻塞。

另一个 Haskell 线程可能处于的状态是这样的：

```
case BlockedOnCCall:
case BlockedOnCCall_NoUnblockExc:
{
    blockedThrowTo(cap,target,msg);
    return THROWTO_BLOCKED;
}

```

运行时系统等待线程停止在 FFI 调用上的阻塞，然后再传递异常——它最终会到达那里！但如果 FFI 调用时间太长，这将为时已晚。我们可以用 `raiseAsync` 替换此调用，但我们发现，虽然异常被引发并且 Haskell 线程恢复正常执行，*FFI 计算继续进行*！

* * *

如果这看起来很神秘，回顾一下 [GHC 运行时系统的多线程调度器](http://hackage.haskell.org/trac/ghc/wiki/Commentary/Rts/Scheduler) 的工作方式将会很有帮助。Haskell 线程是轻量级的，与操作系统线程没有一对一的对应关系。相反，Haskell 线程使用 TSO（线程状态对象）表示，在 RTS 中被调度为一小部分操作系统线程，被抽象为任务。每个操作系统线程与一个 CPU 核心相关联，在 RTS 中被抽象为能力。

在执行的最初阶段，操作系统线程的数量与虚拟核心的数量相同（由 `-N` RTS 选项指定）：就 Haskell 代码而言，我们通过拥有多个能力而获得并行性，*而不是*多个任务！能力一次只能属于一个任务。然而，如果一个任务在操作系统上阻塞，它可能会放弃它的能力给另一个任务，后者可以继续运行 Haskell 代码，因此我们经常将这些任务称为工作线程。

任务（操作系统线程）通过执行由 TSO（Haskell 线程）在运行队列中请求的 InCall 来执行工作，以循环轮换的方式进行调度。在执行过程中，可能会遇到 FFI 调用。这里的行为会根据 FFI 调用是安全还是不安全而有所不同。

+   如果调用是不安全的，我们直接进行调用，不放弃能力！这意味着没有其他 Haskell 代码可以运行这个虚拟核心，如果 FFI 调用花费很长时间或者阻塞，这是个坏消息，但如果速度真的很快，我们不必放弃能力只为了随后再夺回来。

+   如果调用是安全的，我们释放能力（允许其他 Haskell 线程继续），而 Haskell 线程则暂停在一个外部调用上。当前的操作系统线程接着运行 FFI 调用。

因此，如果我们试图通过向其抛出异常来直接唤醒原始的 Haskell 线程，它最终会被调度到*不同*的操作系统线程（而原始线程继续运行 FFI 调用！）

诀窍是终止正在运行 FFI 调用的操作系统线程。

```
   case BlockedOnCCall:
   case BlockedOnCCall_NoUnblockExc:
   {
#ifdef THREADED_RTS
       Task *task = NULL;
       if (!target->bound) {
           // walk all_tasks to find the correct worker thread
           for (task = all_tasks; task != NULL; task = task->all_link) {
               if (task->incall->suspended_tso == target) {
                   break;
               }
           }
           if (task != NULL) {
               raiseAsync(cap, target, msg->exception, rtsFalse, NULL);
               pthread_cancel(task->id);
               task->cap = NULL;
               task->stopped = rtsTrue;
               return THROWTO_SUCCESS;
           }
       }
#endif
       blockedThrowTo(cap,target,msg);
       return THROWTO_BLOCKED;
   }

```

无论它是哪个操作系统线程？它不可能是试图抛出异常的线程，也与暂停的 Haskell 线程无关，后者正在等待被唤醒但不知道它在等待被唤醒的原因。然而，运行 FFI 调用的任务知道哪个 Haskell 线程正在等待它，因此我们可以遍历所有任务列表，查找与我们的异常目标匹配的任务。一旦找到它，我们就使用 `pthread_cancel` 杀死线程，并用异常唤醒原始的 Haskell 线程。

有一个微妙之处是 Marlow 指出的：我们不想销毁绑定线程，因为它们可能包含线程本地状态。工作线程是相同的，因此是可以牺牲的，但不能轻视绑定线程。

* * *

我们有点不友好：在中断时，我们没有给库一个清理的机会。幸运的是，库可以使用 `pthread_setcancelstate` 和 `pthread_setcanceltype` 在退出之前给它一个清理的机会。

* * *

结果表明，即使使用了 RTS 补丁，我们仍然无法完全中断 FFI 调用。但如果我们添加一个明确的新 Haskell 线程，事情就会起作用：

```
> interruptible :: a -> IO a -> IO a
> interruptible defaultVal m = do
>     mresult <- newEmptyMVar -- transfer exception to caller
>     mtid    <- newEmptyMVar
>     let install = installIntHandler (Catch ctrlc)
>         cleanup oldHandler = installIntHandler oldHandler >> return ()
>         ctrlc = do
>             hPutStrLn stderr "Caught signal"
>             tid <- readMVar mtid
>             throwTo tid E.UserInterrupt
>         bracket = reportBracket . E.bracket install cleanup . const
>         reportBracket action = do
>             putMVar mresult =<< E.catches (liftM Right action)
>                 [ E.Handler (\(e :: E.AsyncException) ->
>                     return $ case e of
>                         E.UserInterrupt -> Right defaultVal
>                         _ -> Left (E.toException e)
>                     )
>                 , E.Handler (\(e :: E.SomeException) -> return (Left e))
>                 ]
>     putMVar mtid =<< forkIO (bracket m)
>     either E.throw return =<< readMVar mresult -- one write only
>
> main = main' 3
>
> main' 0 = putStrLn "Quitting"
> main' n = do
>     interruptible () $ do
>         (r :: Either E.AsyncException ()) <- E.try $ foo n
>         putStrLn $ "Thread " ++ show n ++ " was able to catch exception"
>     main' (pred n)

```

当以修补后的 RTS 编译并使用 `-threaded` 选项时，这个 Literate Haskell 文件的输出如下：

```
Arf C 3!
Arf C 3!
^CCaught signal
Thread 3 was able to catch exception
Arf C 2!
Arf C 2!
Arf C 2!
^CCaught signal
Thread 2 was able to catch exception
Arf C 1!
Arf C 1!
^CCaught signal
Thread 1 was able to catch exception
Quitting

```

概念证明成功！现在让它在 Windows 上工作…
