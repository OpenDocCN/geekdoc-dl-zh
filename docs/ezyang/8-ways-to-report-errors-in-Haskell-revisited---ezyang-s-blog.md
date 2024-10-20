<!--yml

category: 未分类

date: 2024-07-01 18:17:41

-->

# 8 种在 Haskell 中报告错误的方式再访：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/08/8-ways-to-report-errors-in-haskell-revisited/`](http://blog.ezyang.com/2011/08/8-ways-to-report-errors-in-haskell-revisited/)

2007 年，Eric Kidd 写了一篇相当流行的文章，名为[8 ways to report errors in Haskell](http://www.randomhacks.net/articles/2007/03/10/haskell-8-ways-to-report-errors/)。然而，自原文发表以来已经过去四年了。这是否会影响原文章的真实性？一些名称已经更改，原来给出的建议可能有些...靠不住。我们将审视原始文章中的每一个建议，并提出一种新的概念来理解 Haskell 的所有错误报告机制。

我建议你将这篇文章与旧文章并列阅读。

### 1\. 使用 error

没有改变。我个人的建议是，只有在涉及*程序员*错误的情况下才使用`error`；也就是说，你有一些不变量只有程序员（而不是最终用户）可能会违反。不要忘记，你可能应该看看能否在类型系统中强制执行这个不变量，而不是在运行时。在与`error`相关联的函数名称中包含函数的名称也是良好的风格，这样你可以说“myDiv：除以零”而不仅仅是“除以零”。

另一个重要的事情要注意的是，`error e`实际上是`throw (ErrorCall e)`的缩写，所以你可以显式地模式匹配这类错误，例如：

```
import qualified Control.Exception as E

example1 :: Float -> Float -> IO ()
example1 x y =
  E.catch (putStrLn (show (myDiv1 x y)))
          (\(ErrorCall e) -> putStrLn e)

```

然而，测试错误消息的字符串相等性是不好的做法，所以如果你确实需要区分特定的`error`调用，你可能需要更好的东西。

### 2\. 使用 Maybe a

没有改变。Maybe 是一个方便的、通用的机制，用于在只有一种可能的失败模式和用户可能会想要在纯代码中处理的情况下报告失败。你可以很容易地将返回的 Maybe 转换成一个错误，使用`fromMaybe (error "bang") m`。Maybe 不提供错误的指示，所以对于像`head`或`tail`这样的函数是个好主意，但对于`doSomeComplicatedWidgetThing`这样的函数就不是那么好了。

### 3\. 使用 Either String a

在任何情况下，我实际上不能推荐使用这种方法。如果你不需要区分错误，你应该使用`Maybe`。如果你不需要在纯代码中处理错误，使用异常。如果你需要在纯代码中区分错误，请不要使用字符串，而是制作一个可枚举类型！

然而，在 base 4.3 或更高版本（GHC 7）中，这个 monad 实例在`Control.Monad.Instances`中是免费的；你不再需要做丑陋的`Control.Monad.Error`导入。但是改变这一点也有一些成本：请参见下文。

### 4\. 使用 Monad 和 fail 来泛化 1-3

如果你是一个理论家，你会拒绝将`fail`作为不应该属于`Monad`的憎恶，并拒绝使用它。

如果您比这更加实际，那么情况就更加复杂。我已经提到，在纯代码中捕获字符串异常不是一个特别好的主意，如果您在`Maybe` monad 中，`fail`会简单地吞噬您精心编写的异常。如果您运行的是 base 4.3，`Either`也不会对`fail`特别处理：

```
-- Prior to base-4.3
Prelude Control.Monad.Error> fail "foo" :: Either String a
Loading package mtl-1.1.0.2 ... linking ... done.
Left "foo"

-- After base-4.3
Prelude Control.Monad.Instances> fail "foo" :: Either String a
*** Exception: foo

```

所以，您有这种不真正做您大部分时间所需的奇怪泛化。它可能会（即使如此，仅仅是勉强）在您有一个自定义错误处理应用 monad 时有点用处，但仅此而已。

值得注意的是，`Data.Map`不再使用这种机制。

### 5\. 使用 MonadError 和自定义错误类型

在新的世界秩序中，`MonadError`变得更加合理了，如果您正在构建自己的应用 monad，这是一个相当合理的选择，无论是作为堆栈中的转换器还是要实现的实例。

与旧建议相反，您可以在`IO`的顶部使用`MonadError`：您只需转换`IO` monad 并提升所有 IO 操作。尽管如此，我不太确定为什么您会想要这样做，因为 IO 有自己的良好、高效且可扩展的错误抛出和捕获机制（见下文）。

我还要注意，规范化与您正在交互的库中的错误是一件好事：它让您考虑您关心的信息以及您希望向用户展示信息的方式。您总是可以创建一个`MyParsecError`构造器，它直接使用 parsec 错误，但是对于真正良好的用户体验，您应该考虑每种情况。

### 6\. 在 IO monad 中使用 throw

现在不再称为`throwDyn`和`catchDyn`（除非您导入`Control.OldException`），只是`throw`和`catch`。您甚至不需要一个 Typeable 实例；只需要一个微不足道的 Exception 实例。我强烈推荐在 IO 中用这种方法处理未经检查的异常：尽管随着时间的推移这些库的变化，Haskell 和 GHC 的维护者们对这些异常应该如何工作已经进行了深思熟虑，并且它们具有广泛的适用性，从正常的同步异常处理到*异步*异常处理，这非常巧妙。有很多 bracketing、masking 和其他函数，如果您在传递 Eithers 的话，您根本做不到这些。

确保您在 IO monad 中使用`throwIO`而不是`throw`，因为前者保证了顺序；后者则不一定。

### 7\. 使用 ioError 和 catch

没有理由使用这个，它存在是因为有些历史原因。

### 8\. 在 monad transformers 中大肆发挥

在所有情况下，这与第 5 种情况是相同的；只是在一个情况中，您自己编写，而在这种情况下，您使用 transformers 组合它。同样的注意事项适用。Eric 在这里给出了一个很好的建议，劝阻您不要在 IO 中使用这个。

* * *

这里有一些自原始文章发布以来涌现的新机制。

### 9\. 已检查的异常

Pepe Iborra 写了一个非常巧妙的 [checked exceptions library](http://hackage.haskell.org/package/control-monad-exception)，它允许你明确指出一段代码可能抛出哪些 `Control.Exception` 风格的异常。我以前从未使用过它，但知道 Haskell 的类型系统可以（被滥用）这样使用是令人满意的。如果你不喜欢很难确定是否捕获了所有你关心的异常，可以去看看它。

### 10\. 失败

[Failure typeclass](http://hackage.haskell.org/packages/archive/failure/0.1.0.1/doc/html/Control-Failure.html) 是一个非常简单的库，试图通过简化包装和解包第三方错误来解决互操作性问题。我用过一点点，但不足以对此事发表权威意见。还值得看一看 [Haskellwiki 页面](http://www.haskell.org/haskellwiki/Failure)。

### 结论

你需要考虑两个错误处理领域：*纯错误* 和 *IO 错误*。对于 IO 错误，有一个非常明确的选择：`Control.Exception` 中指定的机制。如果错误明显是由于外部环境的不完善造成的，请使用它。对于纯错误，需要稍微有些品味。如果只有一个失败的情况（也许甚至不是什么大不了的事），应该使用 `Maybe`；如果编码了一个*不可能*的条件，可以使用 `error`；在不需要对错误做出反应的小应用程序中，字符串错误可能是可以接受的；而在需要对其做出反应的应用程序中，则可以使用自定义错误类型。对于互操作性问题，你可以很容易地通过自定义错误类型来解决它们，或者尝试使用一些其他人正在构建的框架：也许某一天会达到关键质量。

应该清楚地指出，Haskell 的错误报告有很多选择。但是，我认为这种选择并非没有道理：每种工具都有适合其使用的情况，而在高级语言中工作的一大乐趣就是错误转换并不是那么难。
