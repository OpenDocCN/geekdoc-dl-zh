<!--yml

category: 未分类

date: 2024-07-01 18:17:21

-->

# Category theory for loop optimizations : ezyang’s blog

> 来源：[`blog.ezyang.com/2013/05/category-theory-for-loop-optimizations/`](http://blog.ezyang.com/2013/05/category-theory-for-loop-optimizations/)

克里斯托弗·德·萨（Christopher de Sa）和我一直在研究一种类别论方法来优化类似 MapReduce 的管道。实际上，我们开始并不涉及任何范畴论——最初我们只是试图在[Delite 编译器](http://stanford-ppl.github.io/Delite/)执行的一些现有循环优化上施加一些结构，顺便发现了范畴论与循环优化之间丰富的关系。

一方面，我认为这种方法非常酷；但另一方面，该领域有许多先前的工作，很难弄清楚自己在研究景观中的位置。当我与约翰·米切尔讨论这个想法时，他对我说：“循环优化，难道你不能只用表查找来解决吗？”我们从现有工作中获得了很多灵感，特别是由伯德、梅尔滕斯、马尔科姆、迈耶等人在 90 年代初开创的*程序计算*文献。这篇博客文章的目的是讨论我们已经研究出的一些想法，并从你这位亲爱的读者那里获得一些反馈。

有几种思考我们试图做的事情的方式：

+   我们希望*实现*一个基于计算的优化器，针对一个真实的项目（Delite），其中循环优化的应用可以对任务的性能产生 drastc 的影响（其他具有类似目标的系统包括[Yicho](http://takeichi.ipl-lab.org/yicho/doc/Yicho.html)，[HYLO](http://lecture.ecc.u-tokyo.ac.jp/~uonoue/hylo/)）。

+   我们希望冒险涉足理论家通常不会涉足的领域。例如，有许多“无聊”的函子（如数组），它们具有重要的性能特性。虽然它们可能与适当定义的代数数据类型同构，但我们认为在计算优化器中，我们希望*区分*这些不同的表示方式。同样，许多不是自然变换本身的函数可以通过部分应用变成自然变换。例如，当`map p xs`被作为函数定义的一部分（结果函数可以应用于任何列表，而不仅限于原始的`xs`）时，`filter p xs`就是一个自然变换。这种结果的自然变换*丑陋*但*有用*。

+   对于股票优化器（例如 Haskell），一些计算优化可以通过使用 *重写规则* 支持。虽然重写规则是一个非常强大的机制，但它们只能描述“始终有效”的优化方式；例如对于森林化，我们总是希望尽可能消除中间数据结构。在我们希望优化的许多应用中，只有通过 *增加* 中间数据结构才能达到最佳性能：现在我们有一个可能的程序空间，而重写规则对于指定哪个程序最好是明显不足的。我们希望能够使用范畴论来解释带有结构的重写规则，并利用领域特定知识来选择最佳程序。

我想通过一个例子来阐述这些想法。这里有一些用 Delite 写的示例代码，用于计算（一维）k 均值聚类的一个迭代：

```
(0 :: numClusters, *) { j =>
  val weightedPoints = sumRowsIf(0,m){i => c(i) == j}{i => x(i)};
  val points = c.count(_ == j);
  val d = if (points == 0) 1 else points
  weightedPoints / d
}

```

可以这样理解它：我们正在计算一个结果数组，其中包含每个簇的位置，最外层的块正在通过索引变量 `j` 循环遍历簇。要计算簇的位置，我们必须获取分配给簇 `j` 的所有点 `x`（即 `c(i) == j` 的条件），将它们加在一起，最后除以簇中点的数量来获取真实位置。

这段代码的主要问题在于，它在整个数据集上 *numClusters* 次进行迭代，而我们只想执行一次迭代。优化后的版本看起来是这样的：

```
val allWP = hashreduce(0,m)(i => c(i), i => x(i), _ + _)
val allP = hashreduce(0,m)(i => c(i), i => 1, _ + _)
(0::numClusters, *) { j =>
    val weightedPoints = allWP(j);
    val points = allP(j);
    val d = if (points == 0) 1 else points
    return weightedpoints / d
}

```

换句话说，我们必须预先计算加权点和点数（请注意两个 hashreduce 可以和应该融合在一起）之后，才能为每个簇生成新的坐标：在这种情况下，生成 *更多* 中间数据结构是有利的。

现在让我们计算优化程序的方式。但是首先，我们必须定义一些函子：

+   `D_i[X]` 是大小由 `i` 指定的 `X` 数组（具体来说，我们将使用 `D_i` 来表示大小为 `numPoints` 的数组，以及 `D_j` 来表示大小为 `numClusters` 的数组）。这组函子也被称为 [对角函子](http://en.wikipedia.org/wiki/Diagonal_functor)，适用于任意大小的乘积。我们还将依赖于 `D` 是可表示的事实，即 `D_i[X] = Loc_D_i -> X` 对于某些类型 `Loc_D_i`（在这种情况下，它是索引集 `{0 .. i}`）。

+   `List[X]` 是 `X` 的标准列表。它是函子 `F[R] = 1 + X * R` 的初始代数。任何 `D_i` 都可以嵌入到 `List` 中；我们将隐式地进行这些转换（请注意反过来则不成立）。

还有一些函数，我们将在下面描述：

+   `tabulate` 见证了 `Loc_D_i -> X` 和 `D_i[X]` 之间同构的一个方向，因为 `D_i` 是可表示的。另一个方向是 `index`，它接受 `D_i[X]` 和一个 `Loc_D_i`，并返回一个 `X`。

+   `fold` 是在 `List` 上的初始代数唯一确定的函数。此外，假设我们有一个函数 `*`，通过取它们的笛卡尔积来组合两个代数，

+   `bucket` 是一个自然变换，它接受 `D_i[X]` 并基于某些函数将其分桶到 `D_j[List[X]]` 中，该函数将 `D_i` 中的元素分配到 `D_j` 的插槽中。这是一个自然变换的示例，在部分应用之前并不是自然变换：如果我们计算 `D_i[Loc_D_j]`，那么我们可以创建一个永远不会查看 `X` 的自然变换；它只是“知道”每个 `D_i` 插槽在结果结构中应该去的位置。

现在让我们用更功能化的术语重新写循环：

```
tabulate (\j ->
  let weightedPoints = fold plus . filter (\i -> c[i] == j) $ x
      points = fold inc . filter (\i -> c[i] == j) $ x
  in divide (weightedPoints, points)
)

```

（其中 `divide` 只是一个函数，它除以它的参数，但检查除数不为零。）消除一些共同的子表达式并将两个折叠融合在一起，我们得到：

```
tabulate (\j -> divide . fold (plus * inc) . filter (\i -> c[i] == j) $ x)

```

在这一点上，仍然完全不清楚我们可以执行任何重写：`filter` 对我们造成了问题。然而，因为 `filter` 在进行相等性测试，我们可以以不同的方式重写它：

```
tabulate (\j -> divide . fold (plus * inc) . index j . bucket c $ x)

```

这里发生了什么？不直接筛选仅在群集 `j` 中的项目，我们可以将其视为在 `c` 上进行 *分桶* `x`，然后索引出我们关心的单个桶。这种视角的转变对整体优化至关重要。

现在我们可以应用自然变换的基本规则。设 `phi = index j` 和 `f = divide . fold (plus * inc)`，那么我们可以将 `f` 推到 `phi` 的另一侧：

```
tabulate (\j -> index j . fmap (divide . fold (plus * inc)) . bucket c $ x)

```

现在我们可以消除 `tabulate` 和 `index`：

```
fmap (divide . fold (plus * inc)) . bucket c $ x

```

最后，因为我们知道如何高效实现 `fmap (fold f) . bucket c`（作为 `hashreduce`），我们分解 `fmap` 并加入折叠和桶：

```
fmap divide . hashreduce (plus * inc) c $ x

```

我们已经实现了我们的完全优化程序。

所有这些都是正在进行的研究，有许多未解决的问题。尽管如此，我希望这篇文章能让你感受到我们推崇的方法。我对你的评论很感兴趣，无论是“太棒了！”还是“这在 20 年前就被 X 系统完成了。” 期待听到你的看法！
