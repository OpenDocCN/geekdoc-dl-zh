<!--yml

category: 未分类

date: 2024-07-01 18:18:13

-->

# 延迟隐式参数绑定：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/07/delaying-implicit-parameter-binding/`](http://blog.ezyang.com/2010/07/delaying-implicit-parameter-binding/)

今天，我们将更详细地讨论丹·多尔在[周一的文章](http://blog.ezyang.com/2010/07/implicit-parameters-in-haskell/)评论中提到的动态绑定的一些要点。我们的第一步是巩固我们对延迟绑定的定义，如在惰性语言（使用 Reader 单子的 Haskell）和严格语言（使用有错误的元循环评估器的 Scheme）中所见。然后我们回到隐式参数，并问一个问题：隐式参数执行动态绑定吗？（忽略单型限制，[奥列格说不](http://okmij.org/ftp/Computation/dynamic-binding.html#implicit-parameter-neq-dynvar)，但在 GHC 可能存在 bug 的情况下，答案是肯定的。）最后，我们展示如何将隐式参数的方便性与 Reader 单子的显式性结合起来，使用奥列格在他的单子区域中使用的标准技巧。

> *旁注.* 对于那些注意力不集中的人，要点是：使用隐式参数的表达式的类型确定了隐式参数绑定的时间。对于大多数项目，隐式参数往往会尽快解析，这并不是很动态；关闭单型限制将导致更加动态的行为。如果您只设置一次隐式参数并且不再更改它们，您将看不到太多差异。

冒着听起来像是一个破碎的记录的风险，我想复习有关 Reader 单子的一个重要区别。在 Reader 单子中，以下两行之间存在很大的区别：

```
do { x <- ask; ... }
let x = ask

```

如果我们在`Reader r`单子中，第一个`x`将具有类型`r`，而第二个`x`将具有类型`Reader r r`；可以称第二个`x`为“延迟”，因为我们尚未使用`>>=`来查看谚语单子包装器并在其结果上采取行动。我们可以通过以下代码看到这意味着什么：

```
main = (`runReaderT` (2 :: Int)) $ do
  x <- ask
  let m = ask
  liftIO $ print x
  m3 <- local (const 3) $ do
    liftIO $ print x
    y <- m
    liftIO $ print y
    let m2 = ask
    return m2
  z <- m3
  liftIO $ print z

```

which outputs:

```
2
2
3
2

```

虽然我们通过调用`local`改变了底层环境，但原始的`x`保持不变，而当我们将`m`的值强制到`y`时，我们发现了新的环境。`m2`表现类似，但方向相反（在内部`ReaderT`声明，但采用外部`ReaderT`的值）。语义不同，因此语法也不同。

请记住这一点，因为我们即将离开（我敢说“熟悉的”？）单子的世界，转向 Lisp 的领域，那里大部分代码*不*是单子的，动态绑定是意外发明的。

在这里，我有在 SICP 中找到的精简版元循环评估器（去除了变异和顺序控制；如果添加这些内容，[理论是可行的](http://okmij.org/ftp/Computation/dynamic-binding.html#DDBinding)，但我们在此帖子中忽略它们）：

```
(define (eval exp env)
  (cond ((self-evaluating? exp) exp)
        ((variable? exp) (lookup-variable-value exp env))
        ((lambda? exp)
         (make-procedure (lambda-parameters exp)
                         (lambda-body exp)))
        ((application? exp)
         (apply (eval (operator exp) env)
                (list-of-values (operands exp) env))
                env)
        ))
(define (apply procedure arguments env)
  (eval
    (procedure-body procedure)
    (extend-environment
      (procedure-parameters procedure)
      arguments
      env)))

```

这里是另一个版本的评估器：

```
(define (eval exp env)
  (cond ((self-evaluating? exp) exp)
        ((variable? exp) (lookup-variable-value exp env))
        ((lambda? exp)
         (make-procedure (lambda-parameters exp)
                         (lambda-body exp)
                         env))
        ((application? exp)
         (apply (eval (operator exp) env)
                (list-of-values (operands exp) env)))
        ))
(define (apply procedure arguments)
  (eval
    (procedure-body procedure)
    (extend-environment
      (procedure-parameters procedure)
      arguments
      (procedure-environment procedure))))

```

如果你对 SICP 的知识有点生疏，在[查阅源代码](http://mitpress.mit.edu/sicp/full-text/book/book-Z-H-26.html#%_sec_4.1)之前，试着弄清楚哪个版本实现了词法作用域，哪个版本实现了动态作用域。

这两个版本之间的主要区别在于 `make-procedure` 的定义。第一个版本本质上是 lambda 定义的直译，只接受参数和主体，而第二个版本添加了额外的信息，即 lambda 创建时的环境。相反，当 `apply` 解开过程以运行其内部时，第一个版本需要额外的信息——当前环境——作为我们将使用 `eval` 运行时的基础环境，而第二个版本则只使用过程中存储的环境。对于没有被“双泡泡” lambda 模型击败的学生来说，这两种选择都似乎是合理的，他们可能会简单地遵循 `make-procedure` 的定义（请注意：给学生一个不正确的 `make-procedure` 是非常邪恶的！）

第一个版本是动态作用域的：如果我尝试引用一个未在 lambda 参数中定义的变量，我会在调用 lambda 的环境中寻找它。第二个版本是词法作用域的：我会在创建 lambda 的环境中寻找缺失的变量，这恰好是 lambda 源代码所在的地方。

那么，“延迟”变量引用意味着什么？如果是词法作用域，意义不大：过程要使用的环境从创建时就已经确定，如果环境是不可变的（即我们不允许 `set!` 等操作），则在尝试解引用变量时根本不重要。

另一方面，如果变量是动态作用域的，则调用引用变量的函数的时间至关重要。由于 Lisp 是严格评估的，一个简单的 `variable` 表达式将立即导致在当前调用环境中查找，但是以 `(lambda () variable)` 形式的“惰性求值”将延迟查找变量，直到我们使用 `(thunk)` 强制求值 `thunk` 为止。`variable` 在 Haskell 中直接类比于类型为 `r` 的值，而 `(lambda () variable)` 类比于类型为 `Reader r r` 的值。

回到 Haskell，再谈隐式参数。百万美元问题是：我们能区分强制和延迟隐式参数吗？如果我们尝试直译原始代码，我们很快就会陷入困境：

```
main = do
  let ?x = 2 :: Int
  let x = ?x
      m = ?x
  ...

```

隐式参数的语法似乎没有任何区分`x`和`m`的内置语法。因此，人们必须要问，什么是默认行为，另外一种方法可以实现吗？

对于 Haskell 来说，这是一种罕见的情况，*类型*实际上改变了表达式的语义。考虑这个带注释的版本：

```
main =
  let ?x = 2 :: Int
  in let x :: Int
         x = ?x
         m :: (?x :: Int) => Int
         m = ?x
     in let ?x = 3 :: Int
        in print (x, m)

```

`x`的类型是`Int`。回顾一下，`(?x :: t)`约束指示表达式使用该隐式变量。这怎么可能：当我们约定不使用隐式变量时，我们是否在非法地使用隐式变量？在这个困境中有一种解决办法：我们强制`?x`的值，并将其赋给`x`，这样我们就已经解析了`?x`，不需要在使用`x`的任何地方要求它。因此，*从表达式的类型约束中移除隐式变量会强制该表达式中的隐式变量。*

另一方面，`m`不执行这种特化：它声明你需要`?x`才能使用表达式`m`。因此，推迟隐式变量的评估。*在类型约束中保持隐式变量会延迟该变量。*

因此，如果简单地写`let mystery = ?x`，那么 mystery 的类型是什么？在这里，可怕的[单型限制](http://www.haskell.org/ghc/docs/6.12.2/html/users_guide/monomorphism.html)就出现了。你可能已经见过单型限制：在大多数情况下，它使得你的函数比你想要的更不通用。然而，这是非常明显的——你的程序无法通过类型检查。在这里，无论单型限制是否开启，都不会导致你的程序无法通过类型检查；它只会改变其行为。我建议不要猜测，在使用隐式参数时明确指定你的类型签名。这样可以清楚地显示出隐式参数是被强制还是推迟的视觉线索。

> *旁注.* 对于那些好奇的人，如果单型限制被启用（默认情况下是启用的），并且你的表达式是合格的（如果它不带参数，它肯定是合格的，否则，请[查阅你最近的 Haskell 报告](http://www.haskell.org/onlinereport/decls.html#sect4.5.5)），所有隐式参数将从你的类型中特化出来，所以`let mystery = ?x`将立即强制`?x`。即使你已经为你的隐式参数精心编写了类型，单型 Lambda 或函数也可能导致你的表达式变为单型化。如果通过`NoMonomorphismRestriction`禁用单型限制，推断算法将保留你的隐式参数，直到它们在一个特殊化的上下文中使用而没有隐式参数。 GHC 也试验性地使模式绑定单型化，可以通过`NoMonoPatBinds`进行调整。

然而，这个故事并没有完全结束：我忽略了`m2`和`m3`！

```
main =
  let ?x = (2 :: Int)
  in do m3 <- let x :: Int
                  x = ?x
                  m :: (?x :: Int) => Int
                  m = ?x
              in let ?x = 3
                 in let m2 :: (?x :: Int) => Int
                        m2 = ?x
                    in print (x, m) >> return m2
        print m3

```

但是`m3`打印的是`3`而不是`2`！我们已经指定了完整的签名，正如我们应该做的那样：出了什么问题？

麻烦的是，**一旦我们试图使用 `m2` 将其从内部作用域传递回外部作用域，我们强制隐式参数，并且出现的 `m3` 只不过是一个 `m3 :: Int`。即使我们尝试指定 `m3` 应该使用隐式参数 `?x`，该参数也会被忽略。你可以将其类比为以下链条：**

```
f :: (?x :: Int) => Int
f = g

g :: Int
g = let ?x = 2 in h

h :: (?x :: Int) => Int
h = ?x

```

`g` 是单态的：再怎么劝说，`?x` 也不会再次未绑定。

我们在 Scheme 世界的简短旅行中，然而，暗示了一种防止 `m2` 过早使用的可能方法：将其放在一个 thunk 中。

```
main =
  let ?x = (2 :: Int)
  in let f2 :: (?x :: Int) => () -> Int
         f2 = let ?x = 3
              in let f1 :: (?x :: Int) => () -> Int
                     f1 = \() -> ?x
                 in f1
     in print (f2 ())

```

但我们发现当我们运行 `f2 ()` 时，签名再次变成了单态，时间点太早了。虽然在 Scheme 中，创建一个 thunk 起作用是因为动态绑定与 *执行模型* 密切相关，但在 Haskell 中，隐式参数由类型控制，而类型却不对。

Dan Doel [发现](http://hackage.haskell.org/trac/ghc/ticket/4226) 有一种方法使事情工作：将 `?x` 约束移到签名的右侧：

```
main =
  let ?x = (2 :: Int)
  in let f2 :: () -> (?x :: Int) => Int
         f2 = let ?x = (3 :: Int)
              in let f1 :: () -> (?x :: Int) => Int
                     f1 = \() -> ?x
                 in f1
     in print (f2 ())

```

以高阶等级的风格来说，这非常脆弱（最微小的触碰，比如一个 `id` 函数，可能使高阶特性消失）。Simon Peyton Jones 对此行为感到惊讶，所以不要对它太过依赖。

这里有另一种获得“真正”动态绑定的方法，以及一个在我看来使绑定时机更加清晰的单子接口。它的模式是基于 Oleg 的 [单子区域](http://okmij.org/ftp/Haskell/regions.html)。

```
{-# LANGUAGE ImplicitParams, NoMonomorphismRestriction,
   MultiParamTypeClasses, FlexibleInstances #-}

import Control.Monad
import Control.Monad.Reader

-- How the API looks

f = (`runReaderT` (2 :: Int)) $ do
    l1 <- label
    let ?f = l1
    r1 <- askl ?f
    liftIO $ print r1
    g

g = (`runReaderT` (3 :: Int)) $ do
    l <- label
    let ?g = l
    r1 <- askl ?f
    r2 <- askl ?g
    liftIO $ print r1
    liftIO $ print r2
    delay <- h
    -- change our environment before running request
    local (const 8) $ do
        r <- delay
        liftIO $ print r

h = (`runReaderT` (4 :: Int)) $ do
    l3 <- label
    let ?h = l3
    r1 <- askl ?f
    r2 <- askl ?g
    r3 <- askl ?h
    -- save a delayed request to the environment of g
    let delay = askl ?g
    liftIO $ print r1
    liftIO $ print r2
    liftIO $ print r3
    return delay

-- How the API is implemented

label :: Monad m => m (m ())
label = return (return ())

class (Monad m1, Monad m2) => LiftReader r1 m1 m2 where
    askl :: ReaderT r1 m1 () -> m2 r1

instance (Monad m) => LiftReader r m (ReaderT r m) where
    askl _ = ask

instance (Monad m) => LiftReader r m (ReaderT r1 (ReaderT r m)) where
    askl = lift . askl

instance (Monad m) => LiftReader r m (ReaderT r2 (ReaderT r1 (ReaderT r m))) where
    askl = lift . askl

```

这是一种混合方法：每次我们以 `ReaderT` 单子的形式添加新参数时，我们生成一个“标签”，这个标签允许我们回到那个单子（通过使用标签的类型来提升我们回到原始单子的方式）。然而，不是通过词法传递标签，而是将它们塞进隐式参数中。然后有一个定制的 `askl` 函数，它以标签作为参数，并返回对应于那个单子的环境。即使你用 `local` 改变环境，这个处理也能正常工作：

```
*Main> f
2
2
3
2
3
4
8

```

更详细地解释这个机制可能是另一篇文章的主题；它非常方便且非常轻量级。

*结论.* 如果你计划将隐式变量仅仅用作更接近程序顶部的可变静态变量，单态性限制是你的朋友。然而，为了安全起见，强制所有隐式参数。你不需要担心让隐式变量通过函数输出逃逸的困难。

如果你计划为更复杂的事情使用动态作用域，使用 [Oleg 风格的动态绑定](http://okmij.org/ftp/Computation/dynamic-binding.html#DDBinding) 并使用隐式参数作为传递标签的便捷方式可能更好。

> *后记.* 或许解释单态性和隐式参数交互如此久，可能表明对两者的高级使用可能并非普通程序员的菜。
