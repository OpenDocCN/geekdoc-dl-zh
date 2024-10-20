<!--yml

category: 未分类

date: 2024-07-01 18:16:53

-->

# 有关功能线性映射的简短笔记：ezyang 的博客

> 来源：[`blog.ezyang.com/2019/05/a-short-note-about-functional-linear-maps/`](http://blog.ezyang.com/2019/05/a-short-note-about-functional-linear-maps/)

从对 Conal Elliot 的[编译为范畴](http://conal.net/papers/compiling-to-categories/compiling-to-categories.pdf)和[自动微分的简单本质](https://arxiv.org/pdf/1804.00746.pdf)的仔细阅读中收集的一些笔记。

有位同事试图定义张量的“树结构”，希望从而将该概念推广到具有“不规则维度”的张量上。让我们来看看：

假设我们有一个（2，3）矩阵：

```
tensor([[1, 2, 3],
        [4, 5, 6]])

```

想一种方式来思考这个问题，我们有一种某种类型的“树”，其中树的根分支到两个子节点，然后每个子节点再分支到三个节点：

```
       /- ROOT -\
  ROW 1          ROW 2
 /  |  \        /  |  \
1   2   3      4   5   6

```

假设您想在 Haskell 中定义此数据结构。一个显而易见的方法是说矩阵只是一堆嵌套的列表，`[[Float]]`。这确实有效，但并不是很详细，并且肯定不是类型安全的。使用大小向量可以实现类型安全，但我们仍然想知道，“这意味着什么？”

常常，归纳定义源于我们如何*组合*事物，就像编程语言的归纳数据结构告诉我们如何将较小的程序组合起来形成更大的程序一样。对于矩阵，我们可以考虑一种图解方式将它们组合起来，无论是垂直附加还是水平附加。这为我们提供了将矩阵组合起来的词汇表，这使我们能够（非唯一地）表示每个矩阵（[编译为范畴，第八部分](http://conal.net/papers/compiling-to-categories/compiling-to-categories.pdf)）：

```
data Matrix
  = Scalar Float
  | Horizontal Matrix Matrix
  | Vertical Matrix Matrix

```

但是这意味着什么呢？好吧，每个矩阵表示一个线性映射（如果`A：（n，m）`是您的矩阵，则线性映射是函数`R^m -> R^n`，定义为`f（x）= A x`。我们将从 a 到 b 的线性映射称为`Linear a b`）。所以我们现在要问的问题是，将两个矩阵“粘”在一起意味着什么？这是将两个线性映射组合成一个新线性映射的一种方法：

```
-- A function definition does not a category make!  You have to
-- prove that the resulting functions are linear.

horizontal :: Linear a c -> Linear b c -> Linear (a, b) c
horizontal f g = \(a, b) -> f a + g b

-- In matrix form:
--
--              [ a ]
-- [ F  |  G ]  [ - ] = [ F a + G b ]
--              [ b ]

vertical :: Linear a c -> Linear a d -> Linear a (c, d)
vertical f g = \a -> (f a, g a)

-- In matrix form:
--
-- [ F ]         [ F a ]
-- [ - ] [ a ] = [  -  ]
-- [ G ]         [ G a ]

```

现在我们开始了！请注意，粘贴在线性映射的类型中*显示出来*：如果我们水平粘贴，那只意味着这个线性映射接收的向量必须被粘在一起（使用元组构造函数）；同样地，如果我们垂直粘贴，我们将产生输出向量，这些向量是粘贴结果。

很棒，我们可以添加一些类型索引，并将 Linear 写成一个 GADT，以在应用构造函数时精细化索引：

```
data Linear a b where
  Scalar :: Float -> Linear Float Float
  Horizontal :: Linear a c -> Linear b c -> Linear (a, b) c
  Vertical :: Linear a c -> Linear a d -> Linear a (c, d)

```

这就是故事的结局吗？还没有。有很多方法可以组合线性映射；例如，你可以（字面上）将两个线性映射组合在一起（与函数组合的意义相同）。确实，你可以用上述数据类型粘贴任何你喜欢的矩阵；我们如何决定什么应该放入我们的*线性映射语言*中，什么不应该？

为此，Conal Elliot 借助*范畴论*的语言来裁决。一个范畴应该定义身份和函数组合：

```
identity :: Linear a a
identity a = a

-- In matrix form: the identity matrix

compose :: Linear b c -> Linear a b -> Linear a c
compose g f = \a -> g (f a)

-- In matrix form: matrix multiply

```

我们发现水平和垂直是余笛卡尔和笛卡尔范畴的消除和引入操作（分别）。

但我们应该只是在我们的数据类型中添加**Identity**和**Compose**构造函数吗？线性映射组合是一个计算上有趣的操作：如果我们只是保留它作为语法（而不是像道义上的矩阵乘法那样做），那么在最终线性映射上进行操作将会非常昂贵。可表示函子又在哪里？我不太确定如何解释这一点，而且我在这篇文章中已经没有时间了；请继续关注后续。
