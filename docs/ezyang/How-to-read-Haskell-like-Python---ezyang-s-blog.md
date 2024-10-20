<!--yml

类别: 未分类

date: 2024-07-01 18:17:40

-->

# 如何像 Pythonista 一样阅读 Haskell 代码 : ezyang’s blog

> 来源：[`blog.ezyang.com/2011/11/how-to-read-haskell/`](http://blog.ezyang.com/2011/11/how-to-read-haskell/)

**tl;dr** — 保存此页面以供将来参考。

你是否曾经处于需要快速理解某种陌生语言代码功能的情况？如果该语言看起来很像你熟悉的语言，通常你可以猜出大部分代码的作用；即使你可能不完全熟悉所有语言特性的工作方式。

对于 Haskell 来说，这有点困难，因为 Haskell 语法看起来与传统语言非常不同。但这里没有真正的深层区别；你只需要适当地看待它。以下是一个快速的、大部分不正确但希望对解释 Haskell 代码有用的指南，就像一个 Python 程序员一样。最后，你应该能够解释这段 Haskell 代码片段（某些代码被省略为`...`）：

```
runCommand env cmd state = ...
retrieveState = ...
saveState state = ...

main :: IO ()
main = do
    args <- getArgs
    let (actions, nonOptions, errors) = getOpt Permute options args
    opts <- foldl (>>=) (return startOptions) actions
    when (null nonOptions) $ printHelp >> throw NotEnoughArguments
    command <- fromError $ parseCommand nonOptions
    currentTerm <- getCurrentTerm
    let env = Environment
            { envCurrentTerm = currentTerm
            , envOpts = opts
            }
    saveState =<< runCommand env command =<< retrieveState

```

* * *

*类型.* 忽略`::`后的所有内容（同样，你可以忽略`type`, `class`, `instance`和`newtype`）。有些人声称类型帮助他们理解代码；如果你是完全的初学者，像`Int`和`String`可能会有所帮助，而像`LayoutClass`和`MonadError`则不会。不要太担心这些。

* * *

*参数.* `f a b c` 翻译成 `f(a, b, c)`。Haskell 代码省略括号和逗号。这导致我们有时需要用括号来表示参数：`f a (b1 + b2) c` 翻译成 `f(a, b1 + b2, c)`。

* * *

*美元符号.* 因为像`a + b`这样的复杂语句很常见，而 Haskell 程序员不太喜欢括号，所以美元符号用于避免括号：`f $ a + b` 等同于 Haskell 代码 `f (a + b)`，翻译成 `f(a + b)`。你可以把它想象成一个大的左括号，自动在行尾关闭（不再需要写`))))))`！特别是，如果你堆叠它们，每一个都会创建更深的嵌套：`f $ g x $ h y $ a + b` 等同于 `f (g x (h y (a + b)))`，翻译成 `f(g(x,h(y,a + b))`（尽管有些人认为这是不良实践）。

在某些代码中，你可能会看到 `<$>` 的变体（带有尖括号）。你可以将 `<$>` 看作与 `$` 同样的方式处理。（你可能还会看到 `<*>`；假装它是一个逗号，所以 `f <$> a <*> b` 翻译成 `f(a, b)`。对于普通的 `$`，没有真正的等价物）

* * *

*反引号.* ``x `f` y`` 翻译成 `f(x,y)`。反引号中的内容通常是一个函数，通常是二元的，左右两边是参数。

* * *

*等号.* 有两种可能的含义。如果它在代码块的开头，它只是表示你正在定义一个函数：

```
doThisThing a b c = ...
  ==>
def doThisThing(a, b, c):
  ...

```

或者如果你看到它靠近`let`关键字，它就像一个赋值操作符：

```
let a = b + c in ...
  ==>
a = b + c
...

```

* * *

*左箭头.* 也起到了赋值操作符的作用：

```
a <- createEntry x
  ==>
a = createEntry(x)

```

为什么不使用等号？骗局。（更准确地说，`createEntry x` 有副作用。更确切地说，这意味着表达式是单子的。但这只是小把戏。现在先忽略它。）

* * *

*右箭头。* 它很复杂。我们稍后会回头再说。

* * *

*Do 关键字。* 线噪声。你可以忽略它。（它确实提供一些信息，即下面存在副作用，但你在 Python 中看不到这种区别。）

* * *

*返回。* 线噪声。也可以忽略。（你永远不会看到它用于控制流。）

* * *

*点。* `f . g $ a + b` 翻译成 `f(g(a + b))`。实际上，在 Python 程序中，你可能更容易看到：

```
x = g(a + b)
y = f(x)

```

但 Haskell 程序员对额外的变量过敏。

* * *

*绑定和鱼操作符。* 你可能会看到类似 `=<<`, `>>=`, `<=<` 和 `>=>` 的东西。这些基本上只是更多摆脱中间变量的方法：

```
doSomething >>= doSomethingElse >>= finishItUp
  ==>
x = doSomething()
y = doSomethingElse(x)
finishItUp(y)

```

有时，Haskell 程序员决定如果变量在某处被赋值，将其在另一方向中进行可能更漂亮：

```
z <- finishItUp =<< doSomethingElse =<< doSomething
  ==>
x = doSomething()
y = doSomethingElse(x)
z = finishItUp(y)

```

最重要的是通过查看 `doSomething`、`doSomethingElse` 和 `finishItUp` 的定义来反向工程实际发生的事情：这将给你一个线索，指出鱼操作符“流动”的方式。如果你这样做，你可以以相同的方式读取 `<=<` 和 `>=>`（它们实际上执行函数组合，就像点操作符一样）。将 `>>` 看作分号（例如，没有赋值涉及）：

```
doSomething >> doSomethingElse
  ==>
doSomething()
doSomethingElse()

```

* * *

*部分应用。* 有时，Haskell 程序员会调用一个函数，但是他们*没有传足够的参数*。不要担心；他们可能已经在别处安排了剩余的参数给函数。忽略它，或者寻找接受匿名函数作为参数的函数。一些常见的罪魁祸首包括 `map`、`fold`（及其变体）、`filter`、组合操作符`.`、鱼操作符（`=<<` 等）。这在数值操作符上经常发生：`(+3)`翻译成`lambda x: x + 3`。

* * *

*控制操作符。* 凭直觉使用它们：它们做你想要的事情！（即使你认为它们不应该那样做。）所以如果你看到：`when (x == y) $ doSomething x`，它读起来像是“当 x 等于 y 时，调用带有 x 作为参数的 doSomething。”

忽略你无法真正将其翻译成 `when(x == y, doSomething(x))`（因为那样会导致 `doSomething` 总是被调用）。事实上，`when(x == y, lambda: doSomething x)` 更准确，但也许假装 `when` 也是一种语言构造更舒服。

`if` 和 `case` 是内置关键字。它们的工作方式符合你的预期。

* * *

*右箭头（真的！）* 右箭头与左箭头无关。把它们看作冒号：它们总是靠近`case`关键字和反斜杠符号，后者是 lambda 函数：`\x -> x`翻译成`lambda x: x`。

使用 `case` 进行模式匹配是一个非常好的功能，但在这篇博文中有点难以解释。可能最容易的近似是带有一些变量绑定的 `if..elif..else` 链：

```
case moose of
  Foo x y z -> x + y * z
  Bar z -> z * 3
  ==>
if isinstance(moose, Foo):
  x = moose.x # the variable binding!
  y = moose.y
  z = moose.z
  return x + y * z
elif isinstance(moose, Bar):
  z = moose.z
  return z * 3
else:
  raise Exception("Pattern match failure!")

```

* * *

*括号。* 如果一个函数以 `with` 开头，你可以知道它是一个括号函数。它们的工作方式类似于 Python 中的上下文：

```
withFile "foo.txt" ReadMode $ \h -> do
  ...
  ==>
with open("foo.txt", "r") as h:
  ...

```

（你可能还记得前面的反斜杠。是的，那是一个 lambda 表达式。是的，`withFile` 是一个函数。是的，你可以定义你自己的。）

* * *

*异常。* `throw`、`catch`、`catches`、`throwIO`、`finally`、`handle` 等看起来像这样的函数实际上都按你预期的方式工作。然而它们看起来可能有点奇怪，因为这些都不是关键字：它们都是函数，并遵循所有这些规则。例如：

```
trySomething x `catch` \(e :: IOException) -> handleError e
  ===
catch (trySomething x) (\(e :: IOException) -> handleError e)
  ==>
try:
  trySomething(x)
except IOError as e:
  handleError(e)

```

* * *

*也许吧。* 如果你看到 Nothing，可以将其视为 `None`。因此 `isNothing x` 用于测试 `x` 是否为 `None`。它的反义词是什么？`Just`。例如，`isJust x` 用于测试 `x` 是否不为 `None`。

你可能会看到很多与保持 `Just` 和 `None` 有关的噪音。这是其中一个最常见的：

```
maybe someDefault (\x -> ...) mx
  ==>
if mx is None:
  x = someDefault
else:
  x = mx
...

```

这里有一个特定的变体，用于当 null 是一个错误条件时：

```
maybe (error "bad value!") (\x -> ...) x
  ==>
if x is None:
  raise Exception("bad value!")

```

* * *

*记录。* 它们的工作方式符合你的预期，尽管 Haskell 允许你创建没有名称的字段：

```
data NoNames = NoNames Int Int
data WithNames = WithNames {
  firstField :: Int,
  secondField :: Int
}

```

所以 `NoNames` 在 Python 中可能被表示为元组 `(1, 2)`，而 `WithNames` 则是一个类：

```
class WithNames:
  def __init__(self, firstField, secondField):
    self.firstField = firstField
    self.secondField = secondField

```

然后创建是非常简单的 `NoNames 2 3` 翻译为 `(2, 3)`，而 `WithNames 2 3` 或 `WithNames { firstField = 2, secondField = 3 }` 翻译为 `WithNames(2,3)`。

访问器有点不同。最重要的记住的是 Haskeller 把他们的访问器放在变量之前，而你可能更熟悉它们放在之后。所以 `field x` 翻译为 `x.field`。如何拼写 `x.field = 2`？嗯，你真的做不到。不过你可以复制一个并进行修改：

```
return $ x { field = 2 }
  ==>
y = copy(x)
y.field = 2
return y

```

或者，如果你用数据结构的名称（以大写字母开头）替换 `x`，你也可以从头开始创建一个。为什么我们只允许你复制数据结构？这是因为 Haskell 是一种 *纯* 函数语言；但不要让这太让你担心。这只是 Haskell 的另一个怪癖。

* * *

*列表推导式。* 它们最初来自 Miranda-Haskell 衍生！只是符号更多。

```
[ x * y | x <- xs, y <- ys, y > 2 ]
  ==>
[ x * y for x in xs for y in ys if y > 2 ]

```

原来 Haskeller 经常更喜欢以多行形式书写列表推导式（也许他们觉得更容易阅读）。它们看起来像这样：

```
do
  x <- xs
  y <- ys
  guard (y > 2)
  return (x * y)

```

因此，如果你看到一个左箭头，它看起来并不像在执行副作用，也许它是一个列表推导式。

* * *

*更多的符号*。列表在 Python 中的工作方式与您期望的相同；`[1, 2, 3]`实际上是一个包含三个元素的列表。冒号，如`x:xs`表示构造一个以`x`开头、`xs`结尾的列表（对于 Lisp 爱好者来说是`cons`）。`++`是列表连接操作。`!!`表示索引。反斜杠表示 lambda。如果您看到一个您不理解的符号，请尝试在[Hoogle](http://haskell.org/hoogle/)上查找它（是的，它适用于符号！）。

* * *

*更多的行噪声*。以下函数可能是行噪声，可以忽略不计。`liftIO`、`lift`、`runX`（例如`runState`）、`unX`（例如`unConstructor`）、`fromJust`、`fmap`、`const`、`evaluate`、参数前的感叹号（`f !x`）、`seq`、井号（例如`I# x`）。

* * *

*汇总所有信息*。让我们回到原始代码片段：

```
runCommand env cmd state = ...
retrieveState = ...
saveState state = ...

main :: IO ()
main = do
    args <- getArgs
    let (actions, nonOptions, errors) = getOpt Permute options args
    opts <- foldl (>>=) (return startOptions) actions
    when (null nonOptions) $ printHelp >> throw NotEnoughArguments
    command <- fromError $ parseCommand nonOptions
    currentTerm <- getCurrentTerm
    let env = Environment
            { envCurrentTerm = currentTerm
            , envOpts = opts
            }
    saveState =<< runCommand env command =<< retrieveState

```

通过一些猜测，我们可以得出这个翻译：

```
def runCommand(env, cmd, state):
   ...
def retrieveState():
   ...
def saveState(state):
   ...

def main():
  args = getArgs()
  (actions, nonOptions, errors) = getOpt(Permute(), options, args)
  opts = **mumble**
  if nonOptions is None:
     printHelp()
     raise NotEnoughArguments
  command = parseCommand(nonOptions)

  currentTerm = getCurrentTerm()
  env = Environment(envCurrentTerm=currentTerm, envOpts=opts)

  state = retrieveState()
  result = runCommand(env, command, state)
  saveState(result)

```

对于对 Haskell 语法的非常肤浅的理解来说，这并不差（只有一个明显无法翻译的部分，需要知道 fold 是什么。并非所有的 Haskell 代码都是折叠；我会再次重申，请不要过多担心它！）

我称之为“行噪声”的大多数东西实际上都有深刻的原因，如果您对这些区别背后的真正原因感到好奇，我建议您学习如何*编写*Haskell。但如果您只是阅读 Haskell，我认为这些规则应该已经足够了。

*感谢*Keegan McAllister、Mats Ahlgren、Nelson Elhage、Patrick Hurst、Richard Tibbetts、Andrew Farrell 和 Geoffrey Thomas 的评论。还要感谢两位 Python 使用者`#python`的友好居民，`asdf`和`talljosh`，因为他们是 Python 使用的试验品。

*附言*。如果您真的好奇`foldl (>>=) (return startOptions) actions`做了什么，它实现了[责任链](http://en.wikipedia.org/wiki/Chain-of-responsibility_pattern)模式。当然。
