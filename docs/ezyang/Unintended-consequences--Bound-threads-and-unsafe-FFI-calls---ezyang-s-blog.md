<!--yml

category: 未分类

date: 2024-07-01 18:17:11

-->

# 意外后果：绑定线程和不安全的 FFI 调用：ezyang 的博客

> 来源：[`blog.ezyang.com/2014/12/unintended-consequences-bound-threads-and-unsafe-ffi-calls/`](http://blog.ezyang.com/2014/12/unintended-consequences-bound-threads-and-unsafe-ffi-calls/)

不久前，我写了一篇文章描述了[不安全的 FFI 调用如何可能阻塞整个系统](http://blog.ezyang.com/2010/07/safety-first-ffi-and-threading/)，并且给出了以下这种行为的例子：

```
/* cbit.c */
#include <stdio.h>
int bottom(int a) {
    while (1) {printf("%d\n", a);sleep(1);}
    return a;
}

```

```
/* cbit.h */
int bottom(int a);

```

```
/* UnsafeFFITest.hs */
{-# LANGUAGE ForeignFunctionInterface #-}

import Foreign.C
import Control.Concurrent

main = do
    forkIO $ do
        safeBottom 1
        return ()
    yield
    print "Pass (expected)"
    forkIO $ do
        unsafeBottom 2
        return ()
    yield
    print "Pass (not expected)"

foreign import ccall "cbit.h bottom" safeBottom :: CInt -> IO CInt
foreign import ccall unsafe "cbit.h bottom" unsafeBottom :: CInt -> IO CInt

```

在这篇文章中，我解释了发生这种情况的原因是因为不安全的 FFI 调用是不可抢占的，所以当`unsafeBottom`无限循环时，Haskell 线程无法继续。

这个解释看起来很合理，但有一个问题：即使在多线程运行时系统中，代码也会挂起。David Barbour 曾经写信询问我关于不安全调用会阻塞整个系统的说法是否过时。但是，根据这篇文章的标题，你能猜到原因吗？如果你认为你知道，请问这些程序的变体会做什么？

1.  将`main =`改为`main = runInUnboundThread`

1.  将第二个`forkIO`改为`forkOn 2`

1.  在`unsafeBottom`之前加上一个`yield`，在`print "Pass (not expected)"`之前再加一个`yield`

* * *

代码阻塞的原因，或者更具体地说，主线程阻塞的原因是因为不安全的 FFI 调用不可抢占地在操作系统线程上运行，而主线程绑定到该线程上。回想一下，默认情况下，主线程在一个绑定的操作系统线程中运行。这意味着必须使用特定的操作系统线程来运行主线程中的代码。如果该线程被 FFI 调用阻塞，即使有其他工作线程可用，主线程也无法运行。

因此，我们可以解释这些变体：

1.  `main`在一个未绑定的线程中运行，不会发生阻塞，因此第二个打印语句会运行。

1.  默认情况下，一个分支线程在与生成它的线程相同的能力上运行（这很好，因为这意味着不需要同步），因此强制糟糕的 FFI 调用在不同的工作线程上运行可以防止它阻塞主线程。

1.  或者，如果一个线程让出，它可能会被重新调度到另一个工作线程上，这也可以防止主线程被阻塞。

所以，也许这个故事的真正教训是：如果你有绑定的线程，请小心处理不安全的 FFI 调用。请注意：每个 Haskell 程序都有一个绑定的线程：主线程！
