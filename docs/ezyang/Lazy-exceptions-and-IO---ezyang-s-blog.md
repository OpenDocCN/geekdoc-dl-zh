<!--yml

category: 未分类

date: 2024-07-01 18:18:19

-->

# 懒惰异常与 IO : ezyang 的博客

> 来源：[`blog.ezyang.com/2010/05/imprecise-exceptions-and-io/`](http://blog.ezyang.com/2010/05/imprecise-exceptions-and-io/)

## 懒惰异常与 IO

考虑下面的代码片段：

```
import Prelude hiding (catch)
import Control.Exception

main :: IO ()
main = do
    t <- safeCall
    unsafeCall t
    putStrLn "Done."

safeCall :: IO String
safeCall = do
    return alwaysFails `catch` errorHandler

--alwaysFails = throw (ErrorCall "Oh no!")
alwaysFails = error "Oh no!"

errorHandler :: SomeException -> IO String
errorHandler e = do
    putStrLn "Caught"
    return "Ok."
errorHandler_ e = errorHandler e >> return ()

unsafeCall :: String -> IO ()
unsafeCall = putStrLn

```

你可能期望的输出是什么？直接转录到 Python 可能看起来像：

```
def main():
    t = safeCall()
    unsafeCall(t)
    print "Done"

def safeCall():
    try:
        return alwaysFails()
    except:
        return errorHandler()

def alwaysFails():
    raise Exception("Oh no!")

def errorHandler():
    print "Caught."
    return "Ok."

def unsafeCall(output):
    print output

```

任何对任何严格语言有一定了解的人都会说：“当然，它会输出：”

```
Caught.
Ok.
Done.

```

当然，懒惰异常（`error`发出的就是这种）并非无缘无故地被称为懒惰；Haskell 代码输出：

```
*** Exception: Oh no!

```

发生了什么？Haskell 是懒惰的，直到它需要为 unsafeCall 评估 IO `return alwaysFails` 的纯内部代码时，它才会这样做。在那时，没有更多的`catch`调用保护代码了。如果你不相信我，可以在`alwaysFails`周围添加一个追踪。您也可以尝试在`unsafeCall`上安装`errorHandler_`。

这个故事的寓意是什么？嗯，其中一个是`错误`是邪恶的，但我们早已知道这一点…

+   对于大多数基于 IO 的错误，您可以以显而易见的方式安装异常处理程序。（如果我们用`return alwaysFails`替换了`alwaysFails`，结果就会是严格的。）对于源自纯代码的错误，您不能安装异常处理程序，因为 GHC 保留在执行代码的时间上任意调度的权利。

+   如果纯代码正在抛出异常，而您希望它停止这样做，您可能需要使用`$!` `deepseq`或`rnf`来强制严格性，这将迫使 GHC 在受保护区域内执行计算。正如我的读者指出的那样，一个很好的思考方式是，*调用*不是异常的，*结构*才是。

+   如果您从纯代码中获得不精确的异常，但是无法弄清楚原因，祝您好运！我还没有找到解决这个问题的好办法。（给我的博客读者的一个小提示。）

*附言.* 请注意，我们需要使用`Control.Exception.catch`。`Prelude.catch`，按照 Haskell98 的定义，仅捕获基于 IO 的错误。
