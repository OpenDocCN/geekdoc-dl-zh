<!--yml

类别：未分类

日期：2024-07-01 18:18:01

-->

# ω：I’m lubbin’ it：ezyang's blog

> 来源：[`blog.ezyang.com/2010/12/omega-i-m-lubbin-it/`](http://blog.ezyang.com/2010/12/omega-i-m-lubbin-it/)
> 
> 新来这个系列？从[开始吧！](http://blog.ezyang.com/2010/12/hussling-haskell-types-into-hasse-diagrams/)

今天我们将更详细地了解一种有些不寻常的数据类型，Omega。在此过程中，我们将讨论[lub](http://hackage.haskell.org/package/lub)库的工作原理以及如何使用它。这对于懒惰的程序员很有实际意义，因为在 Conal 的话中，lub 是*模块化*懒惰的一个好方法。

Omega 很像自然数，但是没有显式的`Z`（零）构造器，而是使用 bottom 代替。毫不奇怪，这使得理论更容易，但实践更困难（但多亏了 Conal 的 lub 库，并没有太多困难）。我们将展示如何在这种数据类型上实现加法、乘法和阶乘，还将展示如何证明减法和等式（甚至对垂直布尔值）是不可计算的。

这是一个文学化的 Haskell 文章。因为我们要实现的类型类的并非所有方法都是可计算的，我们关闭了丢失方法警告：

```
> {-# OPTIONS -fno-warn-missing-methods #-}

```

一些初步工作：

```
> module Omega where
>
> import Data.Lub (HasLub(lub), flatLub)
> import Data.Glb (HasGlb(glb), flatGlb)

```

这里再次是 Omega 的定义，以及它的两个显著元素，zero 和 omega（无穷）。Zero 是 bottom；我们也可以写作`undefined`或`fix id`。Omega 是 Omega 的最小上界，是一个无穷的 W 的堆栈。

```
> data Omega = W Omega deriving (Show)
>
> zero, w :: Omega
> zero = zero
> w    = W w -- the first ordinal, aka Infinity

```

这里有两个`w`的备选定义：

```
w = fix W
w = lubs (iterate W zero)

```

第一个备选定义使用显式的固定点递归，正如我们在图表中看到的那样。第二个备选定义直接计算ω作为链的最小上界`[⊥, W ⊥, W (W ⊥) ...] = iterate W ⊥`。

`Data.Lub`中的 lub 运算符做什么？到目前为止，我们只看到 lub 运算符用于定义链的最小上界：我们能有用地谈论两个值的 lub 吗？是的：最小上界简单地是两个值都“在顶部”的值。

如果顶部没有值，则 lub 是未定义的，lub 运算符可能会给出虚假的结果。

如果一个值比另一个值更严格定义，它可能只是 lub 的结果。

一种直观的理解 lub 运算符的方式是它结合了两个表达式的信息内容。因此，`(1, ⊥)`知道对偶的第一个元素，`(⊥, 2)`知道第二个元素，所以 lub 将这些信息结合起来得到`(1, 2)`。

我们如何计算最小上界？要意识到的一件事是，在 Omega 的情况下，最小上界实际上是两个数字的最大值，因为该域是完全有序的。

```
> instance Ord Omega where
>     max = lub
>     min = glb

```

相应地，两个数字的最小值是最大下界：一个比两个值的信息内容少的值。

如果我们考虑实现`case lub x y of W a -> case a of W _ -> True`的对话，它可能会像这样进行：

Me

Lub，请给我你的值。

Lub

稍等片刻。X 和 Y，请给我你们的值。

X

我的值是 W 和另一个值。

Lub

好的 Edward，我的值是 W 和另一个值。

Me

谢谢！Lub，你的下一个值是什么？

Lub

稍等片刻。X，请给我你的下一个值。

Y

（过了一会儿。）我的值是 W 和另一个值。

Lub

好的。Y，请给我你的下一个值。

Y

我的下一个值是 W 和另一个值。

Lub

好的 Edward，我的值是 W 和另一个值。

Me

谢谢！

X

我的值是 W 和另一个值。

Lub

好的。

这是这次对话的时间线：

这个对话有几个有趣的特点。第一个是 lub 本身是惰性的：它将开始返回答案而不知道完整的答案。第二个是 X 和 Y“竞赛”返回特定的 W，而 lub 不会对第二个返回结果进行操作。然而，顺序并不重要，因为最终结果总是相同的（当最小上界未定义时，情况将*不*相同！）

驱动`lub`的`unamb`库为我们处理了所有这些混乱的并发业务，通过`flatLub`操作符将其公开，用于计算平面数据类型的最小上界。我们需要为非平面数据类型稍加帮助来计算它（尽管人们不禁想知道这是否可以自动推导出来）。

```
> instance Enum Omega where
>     succ = W
>     pred (W x) = x -- pred 0 = 0
>     toEnum n = iterate W zero !! n
>
> instance HasLub Omega where
>     lub x y = flatLub x y `seq` W (pred x `lub` pred y)

```

等价的、更冗长但更明显正确的定义是：

```
isZero (W _) = False -- returns ⊥ if zero (why can’t it return True?)
lub x y = (isZero x `lub` isZero y) `seq` W (pred x `lub` pred y)

```

将此定义与自然数的普通最大值进行比较可能也很有用：

```
data Nat = Z | S Nat

predNat (S x) = x
predNat Z = Z

maxNat Z Z = Z
maxNat x y = S (maxNat (predNat x) (predNat y))

```

我们可以将`lub`的定义分为两部分：零-零情况和否则情况。在`maxNat`中，我们对两个参数进行模式匹配，然后返回 Z。我们不能直接对底部进行模式匹配，但如果承诺在模式匹配成功时返回底部（这里就是这种情况），我们可以使用`seq`来进行模式匹配。我们使用`flatLub`和`lub`来进行多个模式匹配：如果任一值不是底部，则 lub 的结果为非底部，并继续执行`seq`的右侧。

在备选定义中，我们将`Omega`展平为`Bool`，然后在其上使用先前定义的`lub`实例（我们也可以使用`flatLub`，因为`Bool`是一个平坦的域）。为什么我们可以在`Omega`上使用`flatLub`，而`Omega`并不是一个平坦的域？原因有两个：第一个是`seq`只关心其第一个参数是否为底或非底：它隐式地将所有域展平为“底或非底”。第二个原因是`flatLub = unamb`，虽然`unamb`要求其两边的值相等（以便它可以在两者之间做一个*无歧义*的选择），但是对于`Omega`来说，无法证明其不等性：`Omega`的等号和不等号都是不可计算的。

```
> instance Eq Omega where -- needed for Num

```

`glb`实例相当简单，我们将不再深入讨论。建议读者为此实例绘制对话图。

```
> instance HasGlb Omega where
>    glb (W x') (W y') = W (x' `glb` y')

```

现在是思考的好时机，为什么加法、乘法和阶乘在 Omega 上是可计算的，而减法和等式则不是。如果你选择使用游戏语义的方法，你可能会相当自信地认为对于后者的任何情况都没有合理的对话可以完成工作。我们来做一些更有说服力的事情：画一些图片。我们将拆分二元运算符以帮助绘制图表。

这里是加法的图示：

Omega 的成对形成一个矩阵（通常，向上和向右在偏序上更高），而蓝线将输入集分隔为它们的输出。乘法类似，虽然稍微不那么漂亮（有更多的切片）。

我们可以看到这个函数是单调的：一旦我们跟随偏序进入下一个“步骤”，通过蓝线，我们就再也不能回去了。

现在考虑减法：

在这里，函数不是单调的：如果我在偏序上向右移动并进入下一个步骤，我可以通过向上移动“后退”（红线）。因此，它必须是不可计算的。

这是等式的图片。我们立即注意到，将（⊥，⊥）映射到 True 将意味着每个值都必须映射到 True，所以我们不能使用普通的布尔值。但是，我们也不能使用垂直布尔值（其中⊥表示 False，()表示 True）：

再次可以清楚地看到这个函数不是单调的。

现在是实际实现加法和乘法的时候了：

```
> instance Num Omega where
>     x + y = y `lub` add x y
>         where add (W x') y = succ (x' + y)
>     (W x') * y = (x' * y) + y
>     fromInteger n = toEnum (fromIntegral n)

```

这些函数看起来与定义在 Peano 自然数上的加法和乘法非常相似：

```
natPlus Z y = y
natPlus (S x') y = S (natPlus x' y)

natMul Z y = Z
natMul (S x') y = natPlus y (natMul x' y)

```

这里是以前对第一个零进行模式匹配的模式。但是`natPlus`有点令人烦恼：我们模式匹配到零，但返回`y`：我们的`seq`技巧在这里不起作用！然而，我们可以观察到，如果第一个参数是底部，`add`将会是底部，因此如果 x 为零，返回值将为 y。如果 x 不是零怎么办？我们知道`add x y`必须大于或等于`y`，所以这也符合预期。

对于乘法，我们不需要这种技巧，因为零乘以任何数都是零，模式匹配将自动执行此操作。

最后，壮举——*阶乘*：

```
factorial n = W zero `lub` (n * factorial (pred n))

```

我们使用了与加法相同的技巧，注意到 0! = 1。对于阶乘 1，lub 的两边实际上是相等的，而对于任何更大的值，右侧占优势。

* * *

总结一下将模式匹配对零的规则转换为 lub（假设函数是可计算的）：

```
f ⊥ = ⊥
f (C x') = ...

```

变成：

```
f (C x') = ...

```

（正如你可能已经注意到的，这只是通常的严格计算）。更有趣的情况：

```
g ⊥ = c
g (C x') = ...

```

变成：

```
g x = c `lub` g' x
  where g' (C x') = ...

```

假设原始函数`g`是可计算的（特别是单调的）。当 x 为⊥时，情况显而易见；而由于⊥处于偏序的底部，对于任何 x 不为底部的情况，g x 的任何可能值必须大于或等于底部，从而满足第二种情况。

* * *

*一个轻松的片段。* 量子 bogosort 是一种排序算法，涉及创建列表所有可能的排列的宇宙，然后摧毁所有列表未排序的宇宙。

事实证明，使用`lub`时，很容易在你的算法中意外地实现量子 bogosort 的等效物。我将使用我的加法算法的早期版本来演示：

```
x + y = add x y `lub` add y x
  where add (W x') y = succ (x' + y)

```

或者，`(+) = parCommute add`，其中：

```
parCommute f x y = f x y `lub` f y x

```

这个定义可以得到正确答案，但需要指数级的线程才能弄清楚。以下是正在发生的情况的示意图：

关键在于我们在每次递归时重复交换加法的参数，并且非确定性路径之一导致了 x 和 y 都为零的结果。树中的任何其他“早期”终止的分支都会小于真实结果，因此`lub`不会选择它。正如你可能猜到的那样，探索所有这些分支是低效的。

* * *

[下一次](http://blog.ezyang.com/2010/12/no-one-expects-the-scott-induction/)，我们将探讨 Scott 归纳作为一种关于像这样的不动点的推理方法，将其与自然数的归纳和广义归纳联系起来。如果我在下一篇文章中设法理解共归纳，可能也会有一点内容。
