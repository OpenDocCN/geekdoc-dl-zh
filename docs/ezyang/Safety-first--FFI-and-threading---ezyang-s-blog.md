<!--yml

category: 未分类

date: 2024-07-01 18:18:14

-->

# 安全第一：FFI 和线程：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/07/safety-first-ffi-and-threading/`](http://blog.ezyang.com/2010/07/safety-first-ffi-and-threading/)

**更新。** 虽然这篇博文列出了两个事实，但它错误地解释了这两个事实之间的因果关系。[这里是更正链接。](http://blog.ezyang.com/2014/12/unintended-consequences-bound-threads-and-unsafe-ffi-calls/)

*注意保守使用。* 在 FFI 导入中不要使用`unsafe`！我们是认真的！

考虑以下来自旧版 Haskellwiki 的[FFI 介绍示例](http://www.haskell.org/haskellwiki/?title=FFI_Introduction&oldid=33660)：

```
{-# INCLUDE <math.h> #-}
{-# LANGUAGE ForeignFunctionInterface #-}
module FfiExample where
import Foreign.C -- get the C types

-- pure function
-- "unsafe" means it's slightly faster but can't callback to haskell
foreign import ccall unsafe "sin" c_sin :: CDouble -> CDouble
sin :: Double -> Double
sin d = realToFrac (c_sin (realToFrac d))

```

评论轻松指出该函数无法“回调到 Haskell”。初学 FFI 的人可能会想：“哦，这意味着我可以在大多数 FFI 声明上使用大多数`unsafe`，因为我不会做任何像回调到 Haskell 那样高级的事情。”

哦，朋友，如果事情能这么简单就好了！

请记住，在 Haskell 中使用`forkIO`创建线程时，并不是真正创建操作系统线程；你创建的是一个绿色线程，由 Haskell 的运行时系统在其操作系统线程池中管理。这通常是很好的：真正的线程很重，但 Haskell 线程很轻，你可以使用很多而不用付出太多代价。但问题来了：

运行时系统无法抢占不安全的 FFI 调用！

特别是，当你调用一个`unsafe`的 FFI 导入时，你实际上暂停了系统中的其他所有操作：Haskell 无法抢占它（特别是`unsafe`表示不需要保存运行时系统的状态），并且外部代码将独自运行，直到完成。

不相信？自己试试（我在 6.12.1 上进行了测试）。你需要一些文件：

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

还有`UnsafeFFITest.hs`：

```
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

使用以下命令编译和运行相关文件：

```
gcc -c -o cbit.o cbit.c
ghc -threaded --make UnsafeFFITest.hs cbit.o
./UnsafeFFITest +RTS -N4

```

你看到的输出应该类似于这样：

```
ezyang@javelin:~/Dev/haskell/unsafe-ffi$ ./UnsafeFFITest +RTS -N2
1
"Pass (expected)"
2
1
2
1
2

```

第一个调用友好，允许 Haskell 继续前进，但第二个调用没有。一些值得尝试的事情包括交换 fork 的顺序，使用`forkOS`（许多人，包括我自己，错误地认为它会创建另一个操作系统调用），以及更改 RTS 选项`-N`。

这意味着什么？本质上，只有当你*非常*确定 Haskell 永远不会中断你的 C 调用时（我不会对除了最小、最纯净的 C 函数以外的情况做出这种断言），才不要使用`unsafe`。不值得冒险。安全第一！

*附录。* 感谢`#haskell`帮助我梳理这条思路（我之前遇到过这种行为，但没想到可以写成博客。）

*附录 2.* 感谢 Simon Marlow 在澄清我在原始处理该主题时所犯的一些错误时提供的帮助。如果你对并发和外部函数接口（FFI）的相互作用更多细节感兴趣，请查阅他指向的论文：[扩展 Haskell 外部函数接口与并发性](http://www.haskell.org/~simonmar/bib/concffi04_abstract.html)。
