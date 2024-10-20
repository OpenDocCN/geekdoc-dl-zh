<!--yml

类别：未分类

时间：2024-07-01 18:17:36

-->

# 模拟 IO：MonadIO 及其进一步的应用：ezyang 博客

> 来源：[`blog.ezyang.com/2012/01/modelling-io/`](http://blog.ezyang.com/2012/01/modelling-io/)

`MonadIO`问题表面上看起来很简单：我们希望获取一些包含`IO`的函数签名，并用另一个基于`IO`的单子`m`替换所有`IO`的实例。`MonadIO`类型类本身允许我们将形如`IO a`的值转换为`m a`（并通过组合，任何结果为`IO a`的函数）。这个接口既没有争议，也非常灵活；自从[2001 年创建](https://github.com/ghc/packages-base/commit/7f1f4e7a695c402ddd3a1dc2cc7114e649a78ebc)以来，它一直在引导库中（最初在 base 中，后来迁移到 transformers）。然而，很快发现，当存在许多形如`IO a -> IO a`的函数时，我们希望将它们转换为`m a -> m a`时，`MonadIO`没有处理函数负位置参数的规定。这在异常处理的情况下尤其麻烦，这些高阶函数是原始的。因此，社区开始寻找一个更能捕捉 IO 更多特性的新类型类。

尽管升力的语义已经很清楚（通过变换器定律），但更强大的机制是什么并不清楚。因此，早期解决这个问题的方法是挑选一些我们想要的特定函数，将它们放入一个类型类中，并手动实现它们的升力版本。这导致已经存在的`MonadError`类发展成为更专业的`MonadCatchIO`类。然而，安德斯·卡塞奥格意识到这些函数升力版本的实现有一个共同的模式，他将其提取出来形成了`MonadMorphIO`类。这种方法被进一步完善成了`MonadPeelIO`和`MonadTransControlIO`类型类。然而，只有`MonadError`位于核心，并且由于一些根本性问题而未能获得广泛认可。

我认为社区库编写者收敛到其中一个这些类型类是重要且可取的，因为如果希望导出仅需要`MonadIO`接口的界面，则无法正确实现异常处理任务。我完全预期 monad-control 将成为“赢家”，它是类型类的长期系列的终点。然而，我认为将`MonadError`和`MonadCatchIO`描述为一种思想流派，将`MonadMorphIO`、`MonadPeelIO`和`MonadTransControlIO`描述为另一种流派更为准确。

在本博客文章中，我想研究并对比这两种思想流派。类型类是一个接口：它定义了某些对象支持的操作，以及该对象遵循的法则。类型类的实用性既在于其通用性（支持多个实现的单一接口）也在于其精确性（通过*法则*对可接受的实现进行限制，使得使用接口的代码更易于推理）。这是一个基本的张力：这两种流派在如何解决这一问题上有非常不同的结论。

### 对异常建模

这种一般的技术可以描述为在一个类型类中选择几个函数进行泛化。由于通用性的原因，一个功能较少的类型类更可取于一个功能较多的类型类，因此 `MonadError` 和 `MonadCatchIO` 对异常有着非常特殊的强调：

```
class (Monad m) => MonadError e m | m -> e where
  throwError :: e -> m a
  catchError :: m a -> (e -> m a) -> m a

class MonadIO m => MonadCatchIO m where
  catch   :: Exception e => m a -> (e -> m a) -> m a
  block   :: m a -> m a
  unblock :: m a -> m a

```

不幸的是，这些函数被一些问题所困扰：

+   `MonadError` 封装了一个关于错误的抽象概念，并不一定包括异步异常。也就是说，`catchError undefined h` 不一定会运行异常处理程序 `h`。

+   `MonadError` 对于强大的异步异常处理来说是不足的，因为它不包含 `mask` 的接口；这使得编写健壮的括号函数变得困难。

+   `MonadCatchIO` 明确只处理异步异常，这意味着任何纯粹的错误处理不由它处理。这就是“最终器有时被跳过”的问题。

+   通过 `MonadIO` 约束，`MonadCatchIO` 要求 API 支持将任意 IO 操作提升到该单子（而单子设计者可能创建一个限制了用户访问的 IO 支持单子）。

+   `MonadCatchIO` 导出了过时的 `block` 和 `unblock` 函数，而现代代码应该使用 `mask`。

+   `MonadCatchIO` 导出了 `ContT` 变换器的一个实例。然而，续体和异常之间有[已知的非平凡交互](http://hpaste.org/56921)，需要额外的注意来正确处理。

从某种意义上说，`MonadError` 是一个不合逻辑的论断，因为它与 IO 没有任何关联；它对于非 IO 支持的单子也存在完全有效的实例。`MonadCatchIO` 更接近；后三点并不致命，可以很容易地加以考虑：

```
class MonadException m where
  throwM  :: Exception e => e -> m a
  catch   :: Exception e => m a -> (e -> m a) -> m a
  mask    :: ((forall a. m a -> m a) -> m b) -> m b

```

(去除了 `ContT` 实例。) 然而，“最终器有时被跳过”问题更为棘手。事实上，可能存在某些实例的 `MonadCatchIO` 不知道的零存在。有人认为[这些零与 `MonadCatchIO` 无关](http://www.haskell.org/pipermail/haskell-cafe/2010-October/085079.html)；从中可以推断出，如果你想要通过使用 `MonadException` 安装的最终器来尊重短路，则应该使用异步异常来实现。换句话说，`ErrorT` 是一个糟糕的想法。

然而，您可以采取另一种观点：`MonadException`不仅仅与异步异常有关，而是与任何遵循异常规则的零值有关。这些异常的语义在文章[Asynchronous Exceptions in Haskell](http://community.haskell.org/~simonmar/papers/async.pdf)中有详细描述。它们确切地规定了掩码、抛出和捕获的互动方式，以及其他线程如何引入中断。从这个角度来看，无论这种行为是由运行时系统规定还是通过传递纯值来实现，都是实现细节：只要实例编写正确，零值将得到正确处理。这也意味着，如果我们没有内层单子的基础`MonadException`，为`ErrorT e`提供`MonadException`实例就不再可接受：我们不能忽略低层的异常！

采用这种方法还存在一个最后的问题：一旦选择了原语，标准库的大部分内容就必须通过“复制粘贴”其定义来重新定义，但是它们必须引用广义版本。这对基于这一原则实现库来说是一个重大的实际障碍：仅仅在函数开头加上`liftIO`是远远不够的！

我认为强调定义类型类语义将对这一类型类谱系的未来至关重要；这是过去并没有真正存在的一种强调。从这个角度来看，我们定义了我们的类型类，不仅可以访问 IO 中否则无法访问的函数，还可以定义这些函数的行为方式。实际上，我们正在对 IO 的一个子集进行建模。我认为 Conal Elliott 会为此感到自豪。

> 关于异步异常原始语义扩展的[激烈辩论](http://comments.gmane.org/gmane.comp.lang.haskell.cafe/93834)正在进行中，允许“可恢复”和“不可恢复”错误的概念。（这是线程末尾附近的内容。）

### 线程纯效果

这种技术可以描述为概括了一个常见的实现技术，用于实现`MonadCatchIO`中许多原始函数。这些是一组相当奇怪的签名：

```
class Monad m => MonadMorphIO m where
  morphIO :: (forall b. (m a -> IO b) -> IO b) -> m a

class MonadIO m => MonadPeelIO m where
  peelIO :: m (m a -> IO (m a))

class MonadBase b m => MonadBaseControl b m | m -> b where
  data StM m :: * -> *
  liftBaseWith :: (RunInBase m b -> b a) -> m a
  restoreM :: StM m a → m a
type RunInBase m b = forall a. m a -> b (StM m a)

```

这些类型类的关键直觉是它们利用了在被提升的 IO 函数中的*多态性*，以便*在 IO 的顶部线索纯粹效果*。你可以把这看作是`morphIO`中的普遍量化，`peelIO`的返回类型（这是`IO (m a)`，而不是`IO a`），以及`MonadBaseControl`中的`StM`关联类型。例如，`Int -> StateT s IO a`相当于类型`Int -> s -> IO (s, a)`。我们可以部分应用这个函数与当前状态，得到`Int -> IO (s, a)`；很明显，只要我们提升的 IO 函数让我们秘密地传出任意值，我们就能传出我们更新的状态，并在提升的函数完成时重新整合它。能够适用于这种技术的函数集合恰好是那些能够进行这种线索的函数集合。

正如我在[这篇文章](http://blog.ezyang.com/2012/01/monadbasecontrol-is-unsound/)中描述的，这意味着如果它们不是由函数返回的话，你将无法获得任何变换器堆叠效果。因此，MonadBaseControl 的一个更好的词可能不是它是不安全的（尽管它确实允许奇怪的行为），而是它是不完整的：它无法将所有 IO 函数提升到一个形式，其中基础 monad 效果和变换器效果总是同步进行的。

这有一些有趣的含义。例如，这种遗忘性实际上正是为什么一个提升的括号函数将始终运行的精确原因，无论是否存在其他的零值：`finally`根据定义只能察觉异步异常。这使得 monad-control 提升的函数非常明确地只处理异步异常：提升的`catch`函数将不会捕获`ErrorT`的零值。然而，如果您使用更原始函数的提升版本手动实现`finally`，则可能会丢弃最终器。

它还建议了一种 monad-control 的替代实现策略：与其通过函数的返回类型将状态线索化，不如将其嵌入到隐藏的 IORef 中，并在计算结束时读取出来。实际上，我们希望*嵌入*纯 monad 变换器堆栈的语义到 IO 中。然而，在`forkIO`情况下需要注意一些细节：IORefs 需要适当地复制，以保持线程本地性，或者使用 MVars 代替，以允许一致的非局部通信。

众所周知，MonadBaseControl 不允许 ContT 有一个合理的实例。Mikhail Vorozhtsov 认为这太过于限制性。困难在于，虽然无限制的继续与异常不兼容，但在有限的延续传递风格中，可以以一种明智的方式结合异常。不幸的是，monad-control 对此情况没有作任何处理：它要求用户实现的功能太过强大。似乎明确建模 IO 子集的类型类，在某种意义上更为一般化！这也突显了这些类型类首先和主要地是受到通用实现模式抽象的驱动，而不是任何语义上的考量。

### 结论

我希望这篇文章已经清楚地表明了，我为什么将 MonadBaseControl 视为一种实现策略，而不是一个合理的*编程接口*。MonadException 是一个更合理的接口，它具有语义，但面临重要的实现障碍。
