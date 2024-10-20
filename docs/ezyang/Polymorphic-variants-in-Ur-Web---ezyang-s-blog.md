<!--yml

类别：未分类

日期：2024-07-01 18:17:29

-->

# Ur/Web 中的多态变体：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/07/polymorphic-variants-in-urweb/`](http://blog.ezyang.com/2012/07/polymorphic-variants-in-urweb/)

本文解释了[Ur/Web](http://www.impredicative.com/ur/)中**多态变体**的工作原理。编写本文的原因是官方教程没有提及它们，手册只在该主题上留下了一段话，而我在[Logitext](http://logitext.mit.edu/main)中使用它们时学到了一些有用的技巧。

### 什么是多态变体？

对于 OCaml 用户来说，多态变体可能会很熟悉：它们允许您在多种类型中使用变体的标签，而不仅仅是在原始代数数据类型中定义构造函数时需要保持名称唯一：

```
datatype yn = Yes1 | No1
datatype ynm = Yes2 | No2 | Maybe2

```

我们可以简单地重用它们：

```
con yn = variant [Yes = unit, No = unit]
con ynm = variant [Yes = unit, No = unit, Maybe = unit]

```

如果您有很多构造函数想要在多个逻辑类型之间共享，则这非常方便。不幸的是，它们有许多[讨厌的](http://t-a-w.blogspot.com/2006/05/variant-types-in-ocaml-suck.html) [影响](https://ocaml.janestreet.com/?q=node/99)，这主要源于它们向语言引入了子类型化。在 Ur 中，这种固有的子类型化通过与 Ur 记录系统驱动相同的行类型进行调节，因此处理多态变体非常类似于处理记录，并且两者都基于 Ur/Web 的类型级记录。

### 如何创建多态变体？

要创建多态变体类型，不要应用`$`运算符，而是将`variant`应用于类型级记录。因此：

```
$[A = int, B = bool]

```

生成一个具有两个字段的记录，`A`包含一个`int`，`B`包含一个`bool`，而：

```
variant [A = int, B = bool]

```

生成一个具有两个构造函数的变体，`A`仅包含一个`int`，或者`B`仅包含一个`bool`。

要创建多态变体值，请使用`make`函数，该函数需要一个标签（指示构造函数）和值：

```
make [#A] 2

```

从技术上讲，在构建多态变体时，您还需要知道此值将与哪些完整集合的构造函数一起使用。通常 Ur/Web 会为您推断出这一点，但这是一个重要的限制，将影响对变体进行操作的代码。`make`的完整签名如下：

```
val make : nm :: Name
        -> t ::: Type
        -> ts ::: {Type}
        -> [[nm] ~ ts]
        => t -> variant ([nm = t] ++ ts)

```

函数`nm`和`t`的功能应该是不言自明的，而`ts`是类型级记录，用于包含变体中其余值的拼接，附加`[nm = t]`以生成一个保证包含`nm`的类型级记录。

### 如何解构多态变量？

使用`match`函数，该函数接受一个变量和一个函数记录，指示如何处理该变量的每个可能构造函数：

```
match t { A = fn a => a + 2,
          B = fn b => if b then 3 else 6 }

```

实际上，变体和记录使用*相同*类型级记录，尽管记录的类型有些不同，如在匹配类型中所见：

```
val match : ts ::: {Type}                (* the type-level record *)
         -> t ::: Type
         -> variant ts                   (* the variant *)
         -> $(map (fn t' => t' -> t) ts) (* the record *)
         -> t

```

### 我可以对变体执行哪些其他操作？

`make`和`match`是您唯一需要的基本操作：其他所有操作都可以派生出来。但是，[meta](http://hg.impredicative.com/meta/)库中有一个`Variant`模块，其中包含用于处理变体的多个有用的派生函数。例如，这对函数：

```
val read : r ::: {Unit} -> t ::: Type -> folder r
        -> $(mapU t r) -> variant (mapU {} r) -> t
val write : r ::: {Unit} -> t ::: Type -> folder r
        -> $(mapU t r) -> variant (mapU {} r) -> t -> $(mapU t r)

```

允许您将变体用作标签，以从同类型记录中投影和编辑值。签名并不难阅读：`r`是定义变体的类型级记录，`t`是同类型记录的类型，`folder r`是记录的折叠器（通常会被推断出），`$(mapU t r)`是同类型记录的类型（我们没有写`$r`，因为那将是仅包含单元的记录），而`variant (mapU {} r)`是充当“标签”的变体。以下是这个库中一些简单函数的一些示例用法：

```
read {A = 1, B = 2} (make [#A] ())
== 1

write {A = 1, B = 2} (make [#B] ()) 3
== {A = 1, B = 3}

search (fn v => match v {A = fn () => None, B = fn () => Some 2})
== Some 2

find (fn v => match v {A = fn () => True, B = fn () => False})
== Some (make [#A] ())

test [#A] (make [#A] 2)
== Some 2

weaken (make [#A] 2 : variant [A = int])
== make [#A] 2 : variant [A = int, B = int]

eq (make [#A] ()) (make [#B] ())
== False

mp (fn v => match v {A = fn () => 2, B = fn () => True})
== {A = 2, B = True}

fold (fn v i => match v {A = fn () => i + 1, B = fn () => i + 2}) 0
== 3

mapR (fn v x => match v {A = fn i => i * 2, B = fn i => i * 3}) { A = 2 , B = 3 }
== { A = 4, B = 9 }

```

### `destrR`做什么？

这个函数的类型有点令人生畏：

```
val destrR : K --> f :: (K -> Type) -> fr :: (K -> Type) -> t ::: Type
          -> (p :: K -> f p -> fr p -> t)
          -> r ::: {K} -> folder r -> variant (map f r) -> $(map fr r) -> t

```

但实际上，它只是一个更一般的`match`。`match`可以很容易地用`destrR`实现：

```
match [ts] [t] v fs =
  destrR [ident] [fn p => p -> t] (fn [p ::_] f x => f x) v fs

```

当记录不完全是函数，而是包含函数、类型类甚至是当变体是函数而记录是数据时，`destrR`提供了更多的灵活性。

### 是否有更简洁的方法来匹配多个构造函数？

多态变体经常有很多构造函数，它们看起来基本相同：

```
con tactic a =
  [Cut = logic * a * a,
   LExact = int,
   LConj = int * a,
   LDisj = int * a * a,
   LImp = int * a * a,
   LIff = int * a,
   LBot = int,
   LTop = int * a,
   LNot = int * a,
   LForall = int * universe * a,
   LExists = int * a,
   LContract = int * a,
   LWeaken = int * a,
   RExact = int,
   RConj = int * a * a,
   RDisj = int * a,
   RImp = int * a,
   RIff = int * a * a,
   RTop = int,
   RBot = int * a,
   RNot = int * a,
   RForall = int * a,
   RExists = int * universe * a,
   RWeaken = int * a,
   RContract = int * a]

```

快速填充以与`match`匹配的记录很快变得老套，特别是对于任何两个具有相同类型数据的构造函数而言：

```
let fun empty _ = True
    fun single (_, a) = proofComplete a
    fun singleQ (_, _, a) = proofComplete a
    fun double (_, a, b) = andB (proofComplete a) (proofComplete b)
in match t {Cut       = fn (_, a, b) => andB (proofComplete a) (proofComplete b),
            LExact    = empty,
            LBot      = empty,
            RExact    = empty,
            RTop      = empty,
            LConj     = single,
            LNot      = single,
            LExists   = single,
            LContract = single,
            LWeaken   = single,
            LTop      = single,
            RDisj     = single,
            RImp      = single,
            LIff      = single,
            RNot      = single,
            RBot      = single,
            RForall   = single,
            RContract = single,
            RWeaken   = single,
            LForall   = singleQ,
            RExists   = singleQ,
            LDisj     = double,
            LImp      = double,
            RIff      = double,
            RConj     = double
            }
end

```

Adam Chlipala 和我开发了一种不错的方法，通过滥用*局部类型类*来减少这种样板代码，这允许我们依赖 Ur/Web 的推理引擎自动填写处理特定类型元素的函数。这里是使用我们的新方法进行的递归遍历：

```
let val empty   = declareCase (fn _ (_ : int) => True)
    val single  = declareCase (fn _ (_ : int, a) => proofComplete a)
    val singleQ = declareCase (fn _ (_ : int, _ : Universe.r, a) => proofComplete a)
    val double  = declareCase (fn _ (_ : int, a, b) => andB (proofComplete a) (proofComplete b))
    val cut     = declareCase (fn _ (_ : Logic.r, a, b) => andB (proofComplete a) (proofComplete b))
in typeCase t end

```

对于每个变体中的“类型”，您需要编写一个`declareCase`函数，它接受该类型并将其转换为所需的返回类型。（作为第一个构造函数，您还会得到一个构造函数，用于创建原始构造函数；例如，`declareCase (fn f x => f x)` 就是恒等变换。然后您运行`typeCase`，并观察魔法发生。更详细的使用说明请参阅[variant.urs](http://hg.impredicative.com/meta/file/f55f66c6fdee/variant.urs)。）

### 我该如何扩展我的变体类型？

在编写创建变体类型的元程序时，一个常见的问题是您刚刚创建的变体太窄了：也就是说，在`variant ts`中的`ts`中的条目不足。当`ts`是您正在折叠的记录时，特别是这种情况尤为常见。考虑一个简单的例子，我们想要编写这个函数，它为变体的每个构造函数生成一个构造函数的记录：

```
fun ctors : ts ::: {Type} -> fl : folder ts -> $(map (fn t => t -> variant ts) ts)

```

Ur/Web 并不聪明到足以理解这种天真的方法：

```
fun ctors [ts] fl =
  @fold [fn ts' => $(map (fn t => t -> variant ts) ts')]
      (fn [nm ::_] [v ::_] [r ::_] [[nm] ~ r] n => n ++ {nm = make [nm]})
      {} fl

```

因为它并不知道 `nm` 是类型级记录 `ts` 的成员（Ur/Web 并不直接具有字段包含的编码方式）。

修复此问题的方法是使用一种在变体元程序中反复出现的技巧：使累加器在已处理的字段中多态化。这与值级程序中使用的技巧相同，当通过 `foldr` 反转列表时，可以观察到您希望将累加器作为一个函数：

```
con accum r = s :: {Type} -> [r ~ s] => $(map (fn t => t -> variant (r ++ s)) r)
fun ctors [ts] fl =
  @fold [accum]
        (fn [nm::_] [v::_] [r::_] [[nm] ~ r]
            (k : accum r)
            [s::_] [[nm = v] ++ r ~ s] => k [[nm = v] ++ s] ++ {nm = make [nm]})
        (fn [s::_] [[] ~ s] => {}) fl [[]] !

```

`accum` 是累加器的类型，并且我们可以看到它具有新的类型参数 `s :: {Type}`。此参数与要处理的字段 `r` 和当前字段 `nm` 进行连接，以提供完整的字段集 `ts`。在对类似 `[A = int, B = bool, C = string]` 的记录进行折叠时，我们可以看到：

```
r = [],                  nm = A, s = [B = bool, C = string]
r = [A = int],           nm = B, s = [C = string]
r = [A = int, B = bool], nm = C, s = []

```

`r` 按照通常的折叠方式构建字段，但 `s` 则反向构建其字段，因为类似于列表反转，只有在整个结构折叠后，`s` 才能确定，并且现在在外部逐层评估类型函数的堆栈。因此，很容易看出 `k [[nm = v] ++ s]` 总是具有正确的类型。

### 结论

Ur/Web 中的多态变体非常有用，并且避免了与无限制子类型化相关的许多问题。Logitext 最初并不打算使用多态变体，但当发现它们是通过元编程快速实现 JSON 序列化的最可靠方法时，我们采用了它们，并且我们也开始欣赏它们在各种其他情境中的元编程能力。与传统代数数据类型相比，它们可能最大的缺点是缺乏递归，但这也可以通过在 Ur/Web 的模块系统中手动实现 mu 操作符来模拟。我希望本教程已经为您提供了足够的知识，以便自己使用多态变体，并且也可以通过它们进行一些元编程。
