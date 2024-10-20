<!--yml

category: 未分类

date: 2024-07-01 18:17:27

-->

# Y 组合子和严格正性：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/09/y-combinator-and-strict-positivity/`](http://blog.ezyang.com/2012/09/y-combinator-and-strict-positivity/)

无类型 λ 演算最令人费解的特征之一是固定点组合子，即一个具有性质 `fix f == f (fix f)` 的函数 `fix`。编写这些组合子除了 lambda 之外不需要任何东西；其中最著名的之一是 Y 组合子 `λf.(λx.f (x x)) (λx.f (x x))`。

现在，如果你像我一样，在像 Haskell 这样的类型化函数式编程语言中看到了这个，并试图实现它：

```
Prelude> let y = \f -> (\x -> f (x x)) (\x -> f (x x))

<interactive>:2:43:
    Occurs check: cannot construct the infinite type: t1 = t1 -> t0
    In the first argument of `x', namely `x'
    In the first argument of `f', namely `(x x)'
    In the expression: f (x x)

```

糟糕！类型检查不通过。

有一个解决方案流传开来，你可能通过 [维基百科文章](http://en.wikipedia.org/wiki/Fixed_point_combinator#Example_of_encoding_via_recursive_types) 或者 [Russell O'Connor 的博客](http://r6.ca/blog/20060919T084800Z.html) 遇到过，它通过定义一个新类型来打破无限类型：

```
Prelude> newtype Rec a = In { out :: Rec a -> a }
Prelude> let y = \f -> (\x -> f (out x x)) (In (\x -> f (out x x)))
Prelude> :t y
y :: (a -> a) -> a

```

这里发生了一些非常奇怪的事情，Russell 指出 `Rec` 被称为“非单调”。事实上，任何合理的依赖类型语言都会拒绝这个定义（在 Coq 中是这样的）：

```
Inductive Rec (A : Type) :=
  In : (Rec A -> A) -> Rec A.

(* Error: Non strictly positive occurrence of "Rec" in "(Rec A -> A) -> Rec A". *)

```

“非严格正的出现”是什么？它让人想起 [子类型化中的“协变”和“逆变”](http://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science))，但更严格（毕竟是严格的！）基本上，类型的递归出现（例如 `Rec`）不能出现在构造函数参数的函数箭头的左侧。`newtype Rec a = In (Rec a)` 是可以的，但 `Rec a -> a` 不行（即使 `Rec a` 处于正位置，`Rec a -> a` 也不行）。

拒绝这类定义有很充分的理由。最重要的原因之一是排除定义 Y Combinator 的可能性（搞砸派！），这将允许我们创建一个非终止的术语，而不是显式地使用固定点。在 Haskell 中这并不是大问题（非终止大行其道），但在定理证明语言中，一切都应该是终止的，因为非终止的术语对于任何命题都是有效的证明（通过 Curry-Howard 对应）。因此，通过 Y Combinator 潜入非终止将使类型系统非常不完善。此外，有一种类型（非严格正的类型）“太大”，即它们没有集合论解释（集合不能包含自己的幂集，这基本上是 `newtype Rec = In (Rec -> Bool)` 声称的内容）。

结论是，像 `newtype Rec a = In { out :: Rec a -> a }` 这样的类型看起来相当无害，但实际上它们相当讨厌，应该谨慎使用。对于希望编写如下类型的高阶抽象语法（HOAS）的支持者来说，这有点麻烦：

```
data Term = Lambda (Term -> Term)
          | App Term Term

```

啊！`Lambda` 中 `Term` 的非正出现问题又来了！（可以感觉到在场受过匹兹堡训练的类型理论家们的紧张。）幸运的是，我们有像参数化高阶抽象语法（PHOAS）这样的东西来拯救情况。但这是另一个帖子的话题了...

* * *

多亏了亚当·克里帕拉，他在去年秋天的时候首次向我介绍了[他的 Coq 课程](http://adam.chlipala.net/cpdt/html/InductiveTypes.html) 中的正性条件。康纳·麦克布赖德做了一个旁敲侧击的评论，让我真正理解了这里发生的事情，丹·多尔告诉我非严格正的数据类型在集合论模型中没有。
