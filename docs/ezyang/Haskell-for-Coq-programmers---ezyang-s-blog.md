<!--yml

category: 未分类

date: 2024-07-01 18:17:15

-->

# [Coq 程序员的 Haskell](http://blog.ezyang.com/2014/03/haskell-for-coq-programmers/)：ezyang 的博客

> 来源：[`blog.ezyang.com/2014/03/haskell-for-coq-programmers/`](http://blog.ezyang.com/2014/03/haskell-for-coq-programmers/)

所以你可能听说过这个流行的新编程语言叫做 Haskell。Haskell 是什么？Haskell 是一种非依赖类型的编程语言，支持一般递归、类型推断和内置副作用。诚然，依赖类型被认为是现代、表现力强的类型系统的一个基本组成部分。然而，放弃依赖性可能会对软件工程的其他方面带来某些好处，本文将讨论 Haskell 为支持这些变化而做出的省略。

### 语法

在本文中，我们将指出 Coq 和 Haskell 之间的一些句法差异。首先，我们注意到在 Coq 中，类型用单冒号表示（`false : Bool`）；而在 Haskell 中，使用双冒号（`False :: Bool`）。此外，Haskell 有一个句法限制，构造子必须大写，而变量必须小写。

类似于我的[OCaml for Haskellers](http://blog.ezyang.com/2010/10/ocaml-for-haskellers/)文章，代码片段将采用以下形式：

```
(* Coq *)

```

```
{- Haskell -}

```

### 宇宙/类型分类

宇宙是一种其元素为类型的类型。最初由 Per Martin-Löf 引入构造型理论。Coq 拥有无限的宇宙层次结构（例如，`Type (* 0 *) : Type (* 1 *)`，`Type (* 1 *) : Type (* 2 *)`等）。

因此，很容易将宇宙与 Haskell 类型的`*`（发音为“star”）之间的类比，这种类型分类方式与 Coq 中的`Type (* 0 *)`类似原始类型。此外，*box*类别也可以分类种类（`* : BOX`），尽管这种类别严格来说是内部的，不能在源语言中书写。然而，这里的相似之处仅仅是表面的：把 Haskell 看作只有两个宇宙的语言是误导性的。这些差异可以总结如下：

1.  在 Coq 中，宇宙纯粹作为一个尺寸机制使用，以防止创建过大的类型。在 Haskell 中，类型和种类兼具以强制*阶段区分*：如果`a`的种类是`*`，那么`x :: a`保证是一个运行时值；同样地，如果`k`具有 box 类别，那么`a :: k`保证是一个编译时值。这种结构是传统编程语言中的常见模式，尽管像[Conor McBride](https://twitter.com/pigworker/status/446784239754022912)这样的知识渊博的人认为，最终这是一个设计错误，因为不真正需要种类化系统来进行类型擦除。

1.  在 Coq 中，宇宙是累积的：具有类型`Type (* 0 *)`的术语也具有类型`Type (* 1 *)`。在 Haskell 中，类型和种类之间没有累积性：如果`Nat`是一个类型（即具有类型`*`），它不会自动成为一种。然而，在某些情况下，可以使用[datatype promotion](http://www.haskell.org/ghc/docs/latest/html/users_guide/promotion.html)实现部分累积性，它构造了类型级别的构造函数的独立种级别副本，其中数据构造函数现在是类型级别的构造函数。提升还能够将类型构造函数提升为种构造函数。

1.  在 Coq 中，所有级别的宇宙都使用共同的术语语言。在 Haskell 中，有三种不同的语言：用于处理基本术语（运行时值）的语言，用于处理类型级术语（例如类型和类型构造函数）的语言，以及用于处理种级术语的语言。在某些情况下，此语法是重载的，但在后续章节中，我们经常需要说明如何在种系统的每个级别上单独制定构造。

进一步说明：在 Coq 中，`Type`是预测的；在 Haskell 中，`*`是*非预测的*，遵循 System F 和 lambda 立方体中其他语言的传统，在这些风格的种系统中，这种类型的系统易于建模。

### 函数类型

在 Coq 中，给定两种类型`A`和`B`，我们可以构造类型`A -> B`表示从 A 到 B 的函数（对于任何宇宙的 A 和 B）。与 Coq 类似，使用柯里化本地支持具有多个参数的函数。Haskell 支持类型（`Int -> Int`）和种类（`* -> *`，通常称为*类型构造器*）的函数类型，并通过并置应用（例如`f x`）。（函数类型被 pi 类型所包含，但我们将此讨论推迟到以后。）然而，Haskell 对如何构造函数有一些限制，并在处理类型和种类时使用不同的语法：

对于*表达式*（类型为`a -> b`，其中`a, b :: *`），支持直接定义和 lambda。直接定义以等式风格书写：

```
Definition f x := x + x.

```

```
f x = x + x

```

而 lambda 使用反斜杠表示：

```
fun x => x + x

```

```
\x -> x + x

```

对于*类型族*（类型为`k1 -> k2`，其中`k1`和`k2`是种类），不支持 lambda 语法。实际上，在类型级别不允许高阶行为；虽然我们可以直接定义适当种类的类型函数，但最终，这些函数必须完全应用，否则它们将被类型检查器拒绝。从实现的角度来看，省略类型 lambda 使得类型推断和检查变得更容易。

1.  *类型同义词*：

    ```
    Definition Endo A := A -> A.

    ```

    ```
    type Endo a = a -> a

    ```

    类型同义词在语义上等同于它们的扩展。正如在介绍中提到的，它们不能被部分应用。最初，它们旨在作为一种有限的语法机制，使类型签名更易读。

1.  *封闭类型（同义词）族*：

    ```
    Inductive fcode :=
      | intcode : fcode
      | anycode : fcode.
    Definition interp (c : fcode) : Type := match c with
      | intcode -> bool
      | anycode -> char
    end.

    ```

    ```
    type family F a where
      F Int = Bool
      F a   = Char

    ```

    尽管封闭类型家族看起来像是类型案例的添加（并且可能会违反参数性），但实际情况并非如此，因为封闭类型家族只能返回类型。事实上，封闭类型家族对应于 Coq 中的一个众所周知的设计模式，其中编写表示类型*代码*的归纳数据类型，然后具有解释函数，将代码解释为实际类型。正如我们之前所述，Haskell 没有直接的机制来定义类型上的函数，因此必须直接在类型家族功能中支持这种有用的模式。再次强调，封闭类型家族不能部分应用。

    实际上，封闭类型家族的功能性比归纳代码更具表现力。特别是，封闭类型家族支持*非线性模式匹配*（`F a a = Int`），有时可以在没有 iota 缩减可用时减少术语，因为一些输入是未知的。其原因是封闭类型家族使用统一和约束求解进行“评估”，而不是像 Coq 中的代码那样进行普通术语缩减。事实上，在 Haskell 中进行的几乎所有“类型级计算”实际上只是约束求解。封闭类型家族尚未在 GHC 的发布版本中可用，但有一篇[Haskell 维基页面详细描述了封闭类型家族](http://www.haskell.org/haskellwiki/GHC/Type_families#Closed_family_simplification)。

1.  *开放类型（同义词）家族*：

    ```
    (* Not directly supported in Coq *)

    ```

    ```
    type family F a
    type instance F Int = Char
    type instance F Char = Int

    ```

    与封闭类型家族不同，开放类型家族在开放的宇宙中运行，在 Coq 中没有类似物。开放类型家族不支持非线性匹配，并且必须完全统一以减少。此外，在维持可决定类型推断的情况下，左侧和右侧的这类家族还有一些限制。GHC 手册的部分[类型实例声明](http://www.haskell.org/ghc/docs/latest/html/users_guide/type-families.html#type-instance-declarations)详细说明了这些限制。

封闭和类型级家族均可用于在数据构造函数的类型级别上实现计算，这些函数通过提升转换到了类型级别。不幸的是，任何此类算法必须实现两次：一次在表达级别，一次在类型级别。使用元编程可以减少一些必要的样板代码；例如，请参阅[singletons](https://hackage.haskell.org/package/singletons)库。

### 依赖函数类型（Π-类型）

Π-类型是一个函数类型，其目标类型可以根据应用函数的域中的元素而变化。在任何有意义的意义上，Haskell 都没有Π-类型。然而，如果您仅想单纯地使用Π-类型进行多态性，Haskell 确实支持。对于类型的多态性（例如具有类型`forall a : k, a -> a`，其中`k`是一种类型），Haskell 有一个技巧：

```
Definition id : forall (A : Type), A -> A := fun A => fun x => x.

```

```
id :: a -> a
id = \x -> x

```

特别是，在 Haskell 中，标准的表示法是省略类型 lambda（在表达级别）和量化（在类型级别）。可以使用显式的全称量化扩展来恢复类型级别的量化：

```
id :: forall a. a -> a

```

然而，没有办法直接显式地声明类型 lambda。当量化不在顶层时，Haskell 需要一个明确的类型签名，并在正确的位置放置量化。这需要排名-2（或排名-n，取决于嵌套）多态性扩展：

```
Definition f : (forall A, A -> A) -> bool := fun g => g bool true.

```

```
f :: (forall a. a -> a) -> Bool
f g = g True

```

类型级别的多态性也可以使用 [kind polymorphism extension](http://www.haskell.org/ghc/docs/latest/html/users_guide/kind-polymorphism.html) 支持。然而，对于种类变量，没有显式的 forall；你只需在种类签名中提到一种种类变量。

不能直接支持适当的依赖类型，但可以通过首先将数据类型从表达级别提升到类型级别来模拟它们。然后使用运行时数据结构称为*单例*来将运行时模式匹配的结果细化为类型信息。这种在 Haskell 中的编程模式并不标准，尽管最近有学术论文描述了如何使用它。其中特别好的一篇是 [Hasochism: The Pleasure and Pain of Dependently Typed Haskell Program](https://personal.cis.strath.ac.uk/conor.mcbride/pub/hasochism.pdf)，由 Sam Lindley 和 Conor McBride 编写。

### 乘积类型

Coq 支持类型之间的笛卡尔乘积，以及一个称为空元的空乘类型。非常类似的构造也实现在 Haskell 标准库中：

```
(true, false) : bool * bool
(True, False) :: (Bool, Bool)

```

```
tt : unit
() :: ()

```

对偶可以通过模式匹配来解构：

```
match p with
  | (x, y) => ...
end

```

```
case p of
  (x, y) -> ...

```

有血性的类型理论家可能会对这种认同提出异议：特别是，Haskell 的默认对偶类型被认为是一个*负*类型，因为它对其值是惰性的。（更多内容请参阅[polarity](http://existentialtype.wordpress.com/2012/08/25/polarity-in-type-theory/)。）由于 Coq 的对偶类型是归纳定义的，即正的，更准确的认同应该是与严格对偶类型，定义为 `data SPair a b = SPair !a !b`；即，在构造时，两个参数都被评估。这种区别在 Coq 中很难看到，因为正对偶和负对偶在逻辑上是等价的，而 Coq 并不区分它们。（作为一种总语言，它对评估策略的选择是漠不关心的。）此外，在进行代码提取时，将对偶类型提取为它们的惰性变体是相对常见的做法。

### 依赖对偶类型（Σ-类型）

依赖对偶类型是将乘积类型推广为依赖形式的一般化。与之前一样，Σ-类型不能直接表达，除非第一个分量是一个类型。在这种情况下，有一种利用数据类型的编码技巧，可以用来表达所谓的*存在类型*：

```
Definition p := exist bool not : { A : Type & A -> bool }

```

```
data Ex = forall a. Ex (a -> Bool)
p = Ex not

```

正如在多态性的情况下一样，依赖对的类型参数是隐式的。可以通过适当放置的类型注释来显式指定它。

### 递归

在 Coq 中，所有递归函数必须有一个结构上递减的参数，以确保所有函数都终止。在 Haskell 中，这个限制在表达级别上被解除了；结果是，表达级函数可能不会终止。在类型级别上，默认情况下，Haskell 强制执行类型级计算是可判定的。但是，可以使用`UndecidableInstances`标志解除此限制。通常认为不可判定的实例不能用于违反类型安全性，因为非终止实例只会导致编译器无限循环，并且由于在 Haskell 中，类型不能（直接）引起运行时行为的改变。

### 归纳类型/递归类型

在 Coq 中，可以定义归纳数据类型。Haskell 有一个类似的机制来定义数据类型，但是有许多重要的区别，这导致许多人避免在 Haskell 数据类型中使用 *归纳数据类型* 这个术语（尽管对于 Haskeller 来说使用这个术语是相当普遍的）。

在两种语言中都可以轻松定义基本类型，例如布尔值（在所有情况下，我们将使用[Haskell 数据类型扩展中的 GADT 语法](http://www.haskell.org/ghc/docs/latest/html/users_guide/data-type-extensions.html#gadt)，因为它更接近 Coq 的语法形式，且严格更强大）：

```
Inductive bool : Type :=
  | true : bool
  | false : bool.

```

```
data Bool :: * where
  True :: Bool
  False :: Bool

```

两者也支持正在定义的类型的递归出现：

```
Inductive nat : Type :=
  | z : nat
  | s : nat -> nat.

```

```
data Nat :: * where
  Z :: Nat
  S :: Nat -> Nat

```

但是必须小心：我们在 Haskell 中对 `Nat` 的定义接受了一个额外的术语：无穷大（一个无限的后继链）。这类似于产品的情况，并且源于 Haskell 是惰性的这一事实。

Haskell 的数据类型支持参数，但这些参数只能是类型，而不能是值。（尽管，记住数据类型可以提升到类型级别）。因此，可以定义向量的标准类型族，假设适当的类型级 nat（通常情况下，显式的 forall 已被省略）：

```
Inductive vec (A : Type) : nat -> Type :=
  | vnil  : vec A 0
  | vcons : forall n, A -> vec A n -> vec A (S n)

```

```
data Vec :: Nat -> * -> * where
  VNil  :: Vec Z a
  VCons :: a -> Vec n a -> Vec (S n) a

```

由于类型级λ不支持，但数据类型的部分应用是支持的（与类型族相反），因此必须谨慎选择类型中参数的顺序。（可以定义类型级的 flip，但不能部分应用它。）

Haskell 数据类型定义不具有 [严格正性要求](http://blog.ezyang.com/2012/09/y-combinator-and-strict-positivity/)，因为我们不要求终止；因此，可以编写在 Coq 中不允许的奇怪的数据类型：

```
data Free f a where
   Free :: f (Free f a) -> Free f a
   Pure :: a -> Free f a

data Mu f where
   Roll :: f (Mu f) -> Mu f

```

### 推断

Coq 支持请求通过统一引擎推断术语，可以通过在上下文中放置下划线或将参数指定为 *implicit*（在 Coq 中实现像 Haskell 中看到的省略多态函数的类型参数）。通常不可能期望在依赖类型语言中解决所有推断问题，Coq 的统一引擎（复数！）的内部工作被认为是黑魔法（别担心，受信任的内核将验证推断的参数是类型良好的）。

Haskell 如同 Haskell'98 规定的那样，在 Hindley-Milner 下享有主类型和完整类型推断。然而，为了恢复 Coq 所享有的许多高级特性，Haskell 添加了许多扩展，这些扩展不易适应于 Hindley-Milner，包括类型类约束、多参数类型类、GADTs 和类型族。当前的最新算法是一种名为 [OutsideIn(X)](http://research.microsoft.com/en-us/um/people/simonpj/papers/constraints/jfp-outsidein.pdf) 的算法。使用这些特性，没有完整性保证。然而，如果推断算法接受一个定义，那么该定义具有一个主类型，并且该类型就是算法找到的类型。

### 结论

这篇文章最初是在 OPLSS'13 开玩笑时开始的，我在那里发现自己向 Jason Gross 解释了 Haskell 类型系统的一些复杂方面。它的构建曾经中断了一段时间，但后来我意识到我可以按照同伦类型论书的第一章的模式来构建这篇文章。虽然我不确定这篇文档对学习 Haskell 有多大帮助，但我认为它提出了一种非常有趣的方式来组织 Haskell 更复杂的类型系统特性。合适的依赖类型更简单吗？当然是。但考虑到 Haskell 在大多数现有依赖类型语言之外的地方更进一步，这也值得思考。

### 后记

Bob Harper [在 Twitter 上抱怨](http://storify.com/ezyang/bob-harper-comments-on-haskell-for-coq-programmers)，指出这篇文章在某些情况下提出了误导性的类比。我尝试修正了他的一些评论，但在某些情况下我无法推测出他评论的全部内容。我邀请读者看看是否能回答以下问题：

1.  由于阶段区分，Haskell 的 *类型族* 实际上不是像 Coq、Nuprl 或 Agda 那样的类型族。为什么？

1.  这篇文章对推导（类型推断）和语义（类型结构）之间的区别感到困惑。这种困惑出现在哪里？

1.  对种类的量化不同于对类型的量化。为什么？
