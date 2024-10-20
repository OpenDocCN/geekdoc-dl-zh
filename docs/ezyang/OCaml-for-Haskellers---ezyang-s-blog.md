<!--yml

category: 未分类

date: 2024-07-01 18:18:05

-->

# Haskell 程序员的 OCaml：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/10/ocaml-for-haskellers/`](http://blog.ezyang.com/2010/10/ocaml-for-haskellers/)

我开始正式学习 OCaml（我一直在阅读 ML，自 Okasaki 以来，但从未实际写过），这里是关于与 Haskell 不同的一些笔记，来源于 Jason Hickey 的《Objective Caml 简介》。最显著的两个区别是 OCaml 是*不纯的*和*严格的*。

* * *

*特性.* 这里是 OCaml 具有而 Haskell 没有的一些特性：

+   OCaml 有命名参数（`~x:i` 绑定到命名参数 `x` 的值 `i`，`~x` 是 `~x:x` 的简写）。

+   OCaml 有可选参数（`?(x:i = default)` 将 `i` 绑定到带有默认值 `default` 的可选命名参数 `x`）。

+   OCaml 有开放联合类型（`[> 'Integer of int | 'Real of float]`，其中类型保存了实现；你可以将其分配给具有 `type 'a number = [> 'Integer of int | 'Real of float] as a` 的类型。匿名闭合联合类型也被允许（`[< 'Integer of int | 'Real of float]`）。

+   OCaml 有可变记录（在定义中用`mutable`作为字段的前缀，然后使用`<-`运算符赋值）。

+   OCaml 有一个模块系统（今天只是简单提及）。

+   OCaml 有本地对象（本文未涵盖）。

* * *

*语法.* 省略意味着相关语言特性的工作方式相同（例如，`let f x y = x + y` 是相同的）

组织：

```
{- Haskell -}
(* OCaml *)

```

类型：

```
()   Int Float Char String Bool (capitalized)
unit int float char string bool (lower case)

```

运算符：

```
  == /= .&.  .|. xor  shiftL shiftR complement
= == != land lor lxor [la]sl [la]sr lnot

```

（在 Haskell 中，算术与逻辑移位取决于位的类型。）

OCaml 中的浮点运算符：用点号作为前缀（即`+.`）

浮点数转换：

```
floor fromIntegral
int_of_float float_of_int

```

字符串操作符：

```
++ !!i
^  .[i] (note, string != char list)

```

复合类型：

```
(Int, Int)  [Bool]
int * int   bool list

```

列表：

```
x :  [1, 2, 3]
x :: [1; 2; 3]

```

数据类型：

```
data Tree a = Node a (Tree a) (Tree a) | Leaf
type 'a tree = Node of 'a * 'a tree * 'a tree | Leaf;;

```

（请注意，在 OCaml 中，你需要 `Node (v,l,r)` 来匹配，尽管实际上并不存在这样的元组。）

记录：

```
data MyRecord = MyRecord { x :: Int, y :: Int }
type myrecord = { x : int; y : int };;
Field access:
    x r
    r.x
Functional update:
    r { x = 2 }
    { r with x = 2 }

```

（OCaml 记录也支持破坏性更新。）

Maybe：

```
data Maybe a = Just a | Nothing
type 'a option = None | Some of 'a;;

```

数组：

```
         readArray a i  writeArray a i v
[|1; 3|] a.(i)          a.(i) <- v

```

引用：

```
newIORef writeIORef readIORef
ref      :=         !

```

顶层定义：

```
x = 1
let x = 1;;

```

Lambda：

```
\x y -> f y x
fun x y -> f y x

```

递归：

```
let     f x = if x == 0 then 1 else x * f (x-1)
let rec f x = if x == 0 then 1 else x * f (x-1)

```

互递归（请注意，Haskell 中的`let`始终是递归的）：

```
let f x = g x
    g x = f x
let rec f x = g x
and     g x = f x

```

函数模式匹配：

```
let f 0 = 1
    f 1 = 2
let f = function
    | 0 -> 1
    | 1 -> 2

```

（注意：你可以在 OCaml 的参数中放置模式匹配，但由于缺乏等式函数定义风格，这种方式并不实用）

Case：

```
case f x of
    0 -> 1
    y | y > 5 -> 2
    y | y == 1 || y == 2 -> y
    _ -> -1
match f x with
    | 0 -> 1
    | y when y > 5 -> 2
    | (1 | 2) as y -> y
    | _ -> -1

```

异常：

```
Definition
    data MyException = MyException String
    exception MyException of string;;
Throw exception
    throw (MyException "error")
    raise (MyException "error")
Catch exception
    catch expr $ \e -> case e of
        x -> result
    try expr with
        | x -> result
Assertion
    assert (f == 1) expr
    assert (f == 1); expr

```

Build：

```
ghc --make file.hs
ocamlopt -o file file.ml

```

运行：

```
runghc file.hs
ocaml file.ml

```

* * *

*类型签名.* Haskell 支持使用双冒号为表达式指定类型签名。OCaml 有两种指定类型的方式，可以内联进行：

```
let intEq (x : int) (y : int) : bool = ...

```

或者它们可以放置在接口文件中（扩展名为 `mli`）：

```
val intEq : int -> int -> bool

```

后一种方法更为推荐，类似于 GHC 支持的`hs-boot`文件（http://www.haskell.org/ghc/docs/6.10.2/html/users_guide/separate-compilation.html#mutual-recursion）。

* * *

*Eta 展开。* 以`'_a`形式的多态类型可以被视为类似于 Haskell 的单态化限制：它们只能被实例化为一个具体类型。然而，在 Haskell 中，单态化限制旨在避免用户不期望的额外重新计算值；在 OCaml 中，值限制要求在面对副作用时保持类型系统的完整性，并且也适用于函数（只需查找签名中的`'_a`）。更根本地，`'a`表示广义类型，而`'_a`表示一个在此时未知的具体类型—在 Haskell 中，所有类型变量都是隐式地普遍量化的，因此前者始终成立（除非单态化限制介入，即使这时也不会显示任何类型变量给你看。但 OCaml 要求单态类型变量不会从编译单元中逃逸，因此存在一些相似性。这听起来没有意义吗？不要惊慌。）

在 Haskell 中，我们可以通过指定显式类型签名来使我们的单态值再次变成多态。在 OCaml 中，我们通过 eta 展开来泛化类型。典型的例子是 `id` 函数，当应用于自身 (`id id`) 时，结果是一个类型为 `'_a -> '_a` 的函数（即受限制的）。我们可以通过编写 `fun x -> id id x` 来恢复 `'a -> 'a`。

还有一个细微之处需要处理 OCaml 的不纯和严格性：eta 展开类似于一个延迟计算，因此如果你 eta 展开的表达式具有副作用，它们将被延迟执行。当然，你可以编写 `fun () -> expr` 来模拟一个经典的延迟计算。

* * *

*尾递归。* 在 Haskell 中，当计算是惰性时，你不必担心尾递归；相反，你要努力将计算放入数据结构中，以便用户不会强制获取比他们所需更多的计算（受限递归），并且“堆栈帧”在你深入模式匹配结构时会被高兴地丢弃。然而，如果你正在实现像`foldl'`这样的严格函数，你需要注意这一点（并且不要建立一个非常大的延迟计算。）

好吧，OCaml 默认是严格的，所以你总是要注意确保有尾调用。一个有趣的地方是在 [map 的实现中](http://ocaml.janestreet.com/?q=node/71)，其中的简单版本无法进行尾调用优化。在 Haskell 中，这不是问题，因为我们的 map 是惰性的，并且递归隐藏在 cons 构造函数中；在 OCaml 中，存在一个权衡：复制整个列表以实现尾调用优化，或者不复制并在处理大列表时可能耗尽堆栈空间。（注意，Haskell 中的严格 map 函数也会遇到相同的问题；这是惰性和严格性之间的区别，而不是 Haskell 和 OCaml 的区别。）

* * *

*文件组织。* 单个文件的 OCaml 脚本包含一系列按顺序执行的语句（没有 `main` 函数）。

Haskell 模块的道德等价物在 OCaml 中被称为*编译单元*，命名约定为`foo.ml`（小写！）对应于`Foo`模块，或者`Foo.foo`指的是`Foo`中的`foo`函数。

按照上述描述编写接口文件`mli`被认为是一种良好的实践；这些文件类似于导出列表。接口文件还将包含数据定义（构造函数被省略以实现隐藏）。

默认情况下，所有模块都像`import qualified Foo`一样自动“导入”（无需导入列表）。在 OCaml 中，可以通过`open Foo`进行传统的`import Foo`风格的导入（以便可以不加限定地使用名称）。

* * *

*模块系统。* OCaml 没有类型类，但它有模块系统，您可以通过它们[实现相似的效果](http://okmij.org/ftp/ML/ML.html#typeclass)。 （获得类型类风格效果的另一种经典方法是使用对象，但我今天不涵盖它们。）我本来想今天讨论这个问题，但这篇文章变得很长，所以也许我会把它留到另一天。

* * *

*开放问题。* 我不确定这些内容在 OCaml 特定情况下有多少通用性，以及它们如何推广到所有 ML 语言。

*更新。* ocamlrun 并不同于 runghc；我已经相应地更新了文章。

*更新 2。* Raphael Poss 写了一篇反向的好文章：[Haskell for OCaml programmers](http://staff.science.uva.nl/~poss/haskell-for-ocaml-programmers.html)
