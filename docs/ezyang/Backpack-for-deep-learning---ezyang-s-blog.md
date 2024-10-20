<!--yml

category: 未分类

date: 2024-07-01 18:16:59

-->

# 深度学习的背包：ezyang 的博客

> 来源：[`blog.ezyang.com/2017/08/backpack-for-deep-learning/`](http://blog.ezyang.com/2017/08/backpack-for-deep-learning/)

*这是由阮开曦撰写的客座文章。*

[背包](https://ghc.haskell.org/trac/ghc/wiki/Backpack) 是 Haskell 的一个模块系统，最近在 [GHC 8.2.1](https://ghc.haskell.org/trac/ghc/blog/ghc-8.2.11-released) 中发布。作为一项新功能，我想知道人们如何使用它。因此，我每天在 Twitter 上搜索，前些天看到了 [这条推文](https://twitter.com/Profpatsch/status/897993951852015616)：

> 除了 String/Bytestring/Text 外，还有其他的示例吗？到目前为止我还没有看到任何其他示例；看起来背包只是用来赞美的字符串洞。

有一些 [良好的回应](https://twitter.com/ezyang/status/897999430196101120)，但我想给出另一个来自深度学习的用例。

在深度学习中，人们对在 *张量* 上进行计算感兴趣。张量可以具有不同的值类型：整数、浮点数、双精度等。此外，张量计算可以在 CPU 或 GPU 上进行。尽管有许多不同类型的张量，但每种类型的张量计算是相同的，即它们共享相同的接口。由于背包允许您针对可以有多个实现的单个接口进行编程，因此它是实现张量库的理想工具。

Torch 是一个广泛使用的用于深度学习的库，用 C 实现。亚当·帕什克有一篇关于 Torch 的好文章 [文章](https://apaszke.github.io/torch-internals.html)。我们可以为 Torch 编写一些 Haskell 绑定，然后使用背包在浮点和整数张量之间切换实现。这是一个通过背包签名使用张量的程序：

```
unit torch-indef where
  signature Tensor where
    import Data.Int
    data Tensor
    data AccReal
    instance Show AccReal
    instance Num AccReal
    read1dFile :: FilePath -> Int64 -> IO Tensor
    dot :: Tensor -> Tensor -> IO AccReal
    sumall :: Tensor -> IO AccReal
  module App where
    import Tensor
    app = do
        x <- read1dFile "x" 10
        y <- read1dFile "y" 10
        d <- dot x y
        s <- sumall x
        print (d + s)
        return ()

```

我们有一个简单的主函数，从文件中读取两个一维张量，计算两者的点积，对第一个张量的所有条目求和，然后最后打印出这两个值的和。（这个程序是从亚当的文章中转录的，不同之处在于亚当的程序使用浮点张量，而我们保持张量类型抽象，因此使用背包可以同时处理浮点数和整数）。程序使用像点积这样的函数，在签名中定义。

这里是 `dot` 的实现以及浮点张量的类型。使用 Haskell 的 FFI 调用 C 函数：

```
import Foreign
import Foreign.C.Types
import Foreign.C.String
import Foreign.ForeignPtr

foreign import ccall "THTensorMath.h THFloatTensor_dot"
    c_THFloatTensor_dot :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO CDouble

type Tensor = FloatTensor
type AccReal = Double

dot :: Tensor -> Tensor -> IO AccReal
dot (FT f) (FT g) = withForeignPtr f $ \x ->
                    withForeignPtr g $ \y -> do
                    d <- c_THFloatTensor_dot x y
                    return (realToFrac d)

```

正如您所见，背包可以用于构建一个深度学习库，该库具有多个不同类型操作的实现。如果您为 Torch 中的所有函数编写绑定，您将拥有一个用于 Haskell 的深度学习库；通过背包，您可以轻松编写对张量类型和处理单元（CPU 或 GPU）不可知的模型。

你可以在 [GitHub 上找到完整的示例代码](https://github.com/ezyang/backpack-examples/tree/master/torch)。
