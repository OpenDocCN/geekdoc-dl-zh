<!--yml

category: 未分类

date: 2024-07-01 18:18:27

-->

# 嵌套循环和延续：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/02/nested-loops-and-continuation/`](http://blog.ezyang.com/2010/02/nested-loops-and-continuation/)

一位命令式程序员的主要技能是循环。从 C/汇编的角度来看，循环只是一个结构化的跳转，如果某些条件不满足，则跳回到指定位置。通常，这种循环遍历某些列表数据结构的元素。在 C 语言中，你可能会对数组的元素进行指针算术运算，或者沿着链表的指针进行操作，直到获得`NULL`；在 Python 和其他高级语言中，你会用到`for x in xs`的结构，它巧妙地抽象了这种功能。在循环内部，你还可以使用流程控制操作符`break`和`continue`，它们也是高度结构化的跳转。更紧凑的循环形式和嵌套循环是列表推导式，它们不允许使用这些流程控制操作符。

Haskell 鼓励您使用诸如`map`和`fold`之类的高阶形式，这进一步限制了数据的操作。在 Haskell 中，您肯定不会看到`for`循环... 然而，作为一个有害的小练习，同时也是更深入了解`callCC`可能有用的一种方式，我决定使用`continue`和`break`关键字来实现`for...in`循环。最终的希望是能够编写如下代码：

```
import Prelude hiding (break)

loopLookForIt :: ContT () IO ()
loopLookForIt =
    for_in [0..100] $ \loop x -> do
        when (x `mod` 3 == 1) $ continue loop
        when (x `div` 17 == 2) $ break loop
        lift $ print x

```

以及：

```
loopBreakOuter :: ContT () IO ()
loopBreakOuter =
    for_in [1,2,3] $ \outer x -> do
        for_in [4,5,6] $ \inner y -> do
            lift $ print y
            break outer
        lift $ print x

```

后者通过显式标记每个循环来解决经典的“嵌套循环”问题。我们可以使用以下代码运行这些片段：

```
runContT loopBreakOuter return :: IO ()

```

由于延续表示程序流的“延续”，我们应该有某种作为`break`的延续的概念，以及作为`continue`的延续的概念。我们将存储与在循环“标签”内部跳出和继续相对应的延续，这是我们悬挂的 lambda 函数的第一个参数：

```
data (MonadCont m) => Label m = Label {
    continue :: m (),
    break :: m ()
}

```

然后只需在单子内部调用`continue label`或`break label`来提取和跟随继续。

接下来要做的是实现实际的`for_in`构造。如果我们不必提供任何继续，这实际上只是一个反转的`mapM_`：

```
for_in' :: (Monad m) => [a] -> (a -> m ()) -> m ()
for_in' xs f = mapM_ f xs

```

当然，示例代码中，`f`的类型是`Label m -> a -> m ()`，所以这行不通！考虑这第一种转换：

```
for_in'' :: (MonadCont m) => [a] -> (a -> m ()) -> m ()
for_in'' xs f = callCC $ \c -> mapM_ f xs

```

这个函数与`for_in'`做了同样的事情，但我们把它放在了延续单子内部，并明确了一个变量`c`。在这种情况下，当前的延续`c`对应的是什么呢？嗯，它位于非常外部的上下文中，这意味着“当前的延续”完全不在循环内部。这必须意味着它是`break`的延续。酷！

考虑这第二种替代转换：

```
for_in''' :: (MonadCont m) => [a] -> (a -> m ()) -> m ()
for_in''' xs f = mapM_ (\x -> callCC $ \c -> f x) xs

```

这一次，我们用一个包装器 lambda 替换了`f`，在实际调用`f`之前使用了`callCC`，当前的延续结果是调用`mapM_`的下一步。这是`continue`的延续。

只剩下把它们粘在一起，并将它们打包到`Label`数据类型中。

```
for_in :: (MonadCont m) => [a] -> (Label m -> a -> m ()) -> m ()
for_in xs f = callCC $ \breakCont ->
    mapM_ (\x -> callCC $ \continueCont -> f (Label (continueCont ()) (breakCont ())) x) xs

```

*Et voila!* Haskell 中的命令式循环结构。（尽管你可能永远不想使用它们，挤眼、眨眼）

*附录.* 感谢 Nelson Elhage 和 Anders Kaseorg 指出一个风格上的错误：将延续作为`() -> m ()`存储是不必要的，因为 Haskell 是惰性语言（我为此辩护，命令式范式正在泄漏！）

*附录 2.* 添加了类型签名和运行最初两个示例的代码。

*附录 3.* Sebastian Fischer 指出附录 1 引入的错误。这就是我因未测试我的修改而遭遇的后果！
