<!--yml

category: 未分类

date: 2024-07-01 18:17:39

-->

# monad-control 很棘手：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/01/monadbasecontrol-is-unsound/`](http://blog.ezyang.com/2012/01/monadbasecontrol-is-unsound/)

*编辑注*。我已经减少了这篇文章中一些修辞。原标题是“monad-control 是不稳定的”。

来自 [monad-control](http://hackage.haskell.org/package/monad-control) 包的 MonadBaseControl 和 MonadTransControl 指定了一种吸引人的方式，自动提升在 IO 中采用“回调”函数的函数到基于 IO 的任意单子栈。它们的吸引力在于，它们似乎提供了比替代方案更通用的机制：选择一些函数，提升它们，然后手动重新实现所有在其上构建的函数的通用版本。

不幸的是，monad-control 对于您可能提升的许多函数具有相当令人惊讶的行为。

例如，它不能在多次调用回调函数的函数上工作：

```
{-# LANGUAGE FlexibleContexts #-}

import Control.Monad.Trans.Control
import Control.Monad.State

double :: IO a -> IO a
double m = m >> m

doubleG :: MonadBaseControl IO m => m a -> m a
doubleG = liftBaseOp_ double

incState :: MonadState Int m => m ()
incState = get >>= \x -> put (x + 1)

main = execStateT (doubleG (incState)) 0 >>= print

```

结果是`1`，而不是我们预期的`2`。如果你还不信，请假设 double 的签名是 `Identity a -> Identity a`，比如 `a -> a`。在这种情况下，这个签名只有一个可能的实现：`id`。这种情况下会发生什么应该是显而易见的。

如果你仔细查看 MonadBaseControl 中涉及的类型，其中的原因应该变得显而易见：我们依赖于要提升的函数的多态性，以便在单子变换器中传递 `StM m`，这是封装的单子变换器的“状态”。如果这个返回值被 `IO` 丢弃，就像我们的函数 `double` 中那样，那么恢复这个状态的方式就不存在了。（甚至在 `liftBaseDiscard` 函数中也提到了这一点！）

我的结论是，尽管 monad-control 可能是提升函数的方便实现机制，但它导出的函数存在严重的语义不一致性。最终用户，请注意！

*附言*。对于之前的 MonadBaseControl/MonadTransControl 的版本，其名称分别为 MonadPeel 和 MonadMorphIO，也有类似的限制。
