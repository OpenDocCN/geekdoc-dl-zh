<!--yml

category: 未分类

date: 2024-07-01 18:18:27

-->

# 嵌套循环和异常（Oleg Kiselyov）：ezyang’s 博客

> 来源：[`blog.ezyang.com/2010/02/nested-loops-and-exceptions-oleg-kiselyov/`](http://blog.ezyang.com/2010/02/nested-loops-and-exceptions-oleg-kiselyov/)

*编者按.* 今天我们中断常规节目，为您带来 [Oleg Kiselyov](http://okmij.org/ftp/) 的客座文章，重新解释我们之前关于 [嵌套循环和延续](http://blog.ezyang.com/2010/02/nested-loops-and-continuation/) 的文章，使用异常。

* * *

你好！

我注意到您最近关于嵌套循环和延续的文章。我应该使用提供的表单进行评论，但我不确定格式化结果如何。评论包含大量代码。请随意发布全部或部分代码，或做任何其他操作。

我的评论论点是，在单个和嵌套循环中，`callCC` 是不必要的用于实现 break 和 continue。我们观察到每次迭代和整个循环的延续被调用 0 次或 1 次（但从不超过一次）。这是异常的模式。因此，您的文章提出的问题可以用异常解决。这里有几种解决方案的变体。

首先，一些预备知识：这条消息是完整的文学化 Haskell 代码。

```
> import Prelude hiding (break, catch)
> import Control.Monad
> import Control.Monad.Trans

```

唉，`Control.Monad.Error` 中的 `ErrorT` 有一个愚蠢的 `Error` 约束。因此，我们不得不编写我们自己的异常 monad transformer。下面的代码是标准的。

```
> newtype ExT e m a = ExT{unExT :: m (Either e a)}
>
> instance Monad m => Monad (ExT e m) where
>     return  = ExT . return . Right
>     m >>= f = ExT $ unExT m >>= either (return . Left) (unExT . f)
>
> instance MonadTrans (ExT e) where
>     lift m = ExT $ m >>= return . Right
>
> instance MonadIO m => MonadIO (ExT e m) where
>     liftIO m = ExT $ liftIO m >>= return . Right
>
> raise :: Monad m => e -> ExT e m a
> raise = ExT . return . Left
>
> catch :: Monad m => ExT e m a -> (e -> ExT e' m a) -> ExT e' m a
> catch m h = ExT $ unExT m >>= either (unExT . h) (return . Right)
>
> runExT :: Monad m => ExT e m a -> m a
> runExT m = unExT m >>= either (const $ fail "Unhandled exc") return

```

我们准备编码第一个解决方案，用于简单的非嵌套循环。其思想是将 'break' 和 'continue' 视为异常。毕竟，这两个控制运算符都会导致跳过计算，这正是异常所做的。我们定义了我们 'exceptions' 的数据类型：

```
> data BC = Break | Cont
>
> break, continue :: Monad m => ExT BC m a
> break    = raise Break
> continue = raise Cont

```

这是循环的代码：它在某些点上捕捉异常：

```
> for_in :: Monad m => [a] -> (a -> ExT BC m ()) -> m ()
> for_in xs f = runExT $ mapM_ iter xs `catch` hbreak
>  where
>  iter x = catch (f x) hcont
>  hcont  Cont  = return ()     -- handle Cont, re-raise Break
>  hcont  e     = raise e
>  hbreak Break = return ()
>  hbreak Cont  = return ()     -- Shouldn't happen actually

```

这是您的测试：

```
> loopLookForIt1 :: IO ()
> loopLookForIt1 =
>     for_in [0..100] $ \x -> do
>         when (x `mod` 3 == 1) $ continue
>         when (x `div` 17 == 2) $ break
>         lift $ print x

```

运行它：

```
> tf1 = loopLookForIt1 :: IO ()

```

打印从 0、2、3 开始，以 30、32、33 结束的 23 个数字。

我们必须推广到嵌套循环。显然有两种解决方案。我称第一种为 'dynamic'。我们通过级别（自然数）对异常进行索引。级别 0 适用于当前循环，级别 1 适用于父循环，依此类推。

```
> data BCN = BCN BC Int                 -- Add the level of breaking

```

现在，break 和 continue 运算符需要带上数字：中断多少个循环作用域。我认为 Perl 有类似的带数字中断操作符。

```
> breakN    = raise . BCN Break
> continueN = raise . BCN Cont

```

新的迭代器：

```
> for_inN :: Monad m => [a] -> (a -> ExT BCN m ()) -> ExT BCN m ()
> for_inN xs f = mapM_ iter xs `catch` hbreak
>  where
>  iter x = catch (f x) hcont
>  hcont  (BCN Cont 0)  = return ()     -- handle Cont, re-raise Break
>  hcont  e             = raise e
>  -- If the exception is for a parent, re-raise it, decrementing its level
>  hbreak (BCN Break 0) = return ()
>  hbreak (BCN Cont 0)  = return ()     -- Shouldn't happen actually
>  hbreak (BCN exc n)   = raise (BCN exc (n-1))

```

现在单循环测试如下所示。

```
> loopLookForItN :: ExT BCN IO ()
> loopLookForItN =
>     for_inN [0..100] $ \x -> do
>         when (x `mod` 3 == 1) $ continueN 0
>         when (x `div` 17 == 2) $ breakN 0
>         lift $ print x
>
> tfN = runExT loopLookForItN :: IO ()

```

现在我们可以编写嵌套循环测试了。我稍作修改您文章中的示例，以涵盖所有情况：

```
> loopBreakOuter1 :: ExT BCN IO ()
> loopBreakOuter1 =
>     for_inN [1,2,3] $ \x -> do
>         lift $ print x
>         for_inN [4,5,6] $ \y -> do
>             lift $ print y
>             when (y == 4) $ continueN 0
>             when (x == 1) $ breakN 0
>             when (x == 3) $ breakN 1
>             when (y == 5) $ continueN 1
>             breakN 1
>         lift $ print x
>
> tbN1 = runExT loopBreakOuter1 :: IO ()

```

结果是数字序列：1 4 5 1 2 4 5 3 4 5

存在另一种解决嵌套循环问题的方法，我称之为 'static'。如果我们只是迭代单循环解决方案会怎样？我们可以将 `ExT` `BC` monad transformers 嵌套到任意给定的深度。要引用转换器堆栈中的特定层，我们使用 lift。我们可以使用之前定义的 for_in 迭代器和操作符 break、continue。我们将嵌套测试编写如下：

```
> loopBreakOuterS1 :: IO ()
> loopBreakOuterS1 =
>     for_in [1,2,3] $ \x -> do
>         liftIO $ print x
>         for_in [4,5,6] $ \y -> do
>             liftIO $ print y
>             when (y == 4) $ continue
>             when (x == 1) $ break
>             when (x == 3) $ lift $ break
>             when (y == 5) $ lift $ continue
>             lift $ break
>         liftIO $ print x
> tbS1 = loopBreakOuterS1 :: IO ()

```

我猜这里的教训可能是`callCC`通常是不需要的（我会争辩说`callCC`从来不需要，但那是另一个时间的论点）。这里是另一个简单异常足够的例子，而在那里人们认为需要`call/cc`：

[`okmij.org/ftp/Computation/lem.html`](http://okmij.org/ftp/Computation/lem.html)
