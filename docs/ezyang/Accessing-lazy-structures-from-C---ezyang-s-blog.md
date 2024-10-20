<!--yml

分类：未分类

日期：2024-07-01 18:17:39

-->

# 从 C 中访问惰性结构：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/12/accessing-lazy-structures-from/`](http://blog.ezyang.com/2011/12/accessing-lazy-structures-from/)

## 从 C 中访问惰性结构

[最近有人在 haskell-beginners 上询问](http://comments.gmane.org/gmane.comp.lang.haskell.beginners/9109)，如何在 C 中访问一个惰性（可能是无限的）数据结构。我未能找到一些关于如何做到这一点的示例代码，因此我自己写了一些。希望这能帮助你在你的 C 调用 Haskell 的努力中！

主文件 `Main.hs`：

```
{-# LANGUAGE ForeignFunctionInterface #-}

import Foreign.C.Types
import Foreign.StablePtr
import Control.Monad

lazy :: [CInt]
lazy = [1..]

main = do
    pLazy <- newStablePtr lazy
    test pLazy -- we let C deallocate the stable pointer with cfree

chead = liftM head . deRefStablePtr
ctail = newStablePtr . tail <=< deRefStablePtr
cfree = freeStablePtr

foreign import ccall test :: StablePtr [CInt] -> IO ()
foreign export ccall chead :: StablePtr [CInt] -> IO CInt
foreign export ccall ctail :: StablePtr [CInt] -> IO (StablePtr [CInt])
foreign export ccall cfree :: StablePtr a -> IO ()

```

C 文件 `export.c`：

```
#include <HsFFI.h>
#include <stdio.h>
#include "Main_stub.h"

void test(HsStablePtr l1) {
    int x = chead(l1);
    printf("first = %d\n", x);
    HsStablePtr l2 = ctail(l1);
    int y = chead(l2);
    printf("second = %d\n", y);
    cfree(l2);
    cfree(l1);
}

```

以及一个简单的 Cabal 文件来构建它全部：

```
Name:                export
Version:             0.1
Cabal-version:       >=1.2
Build-type:          Simple

Executable export
  Main-is:             Main.hs
  Build-depends:       base
  C-sources:           export.c

```

祝愉快编程！
