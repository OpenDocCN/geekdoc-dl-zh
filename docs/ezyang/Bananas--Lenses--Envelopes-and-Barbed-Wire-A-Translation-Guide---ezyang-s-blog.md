<!--yml

类别：未分类

日期：2024-07-01 18:18:23

-->

# 香蕉、透镜、信封和铁丝网——翻译指南：ezyang's 博客

> 来源：[`blog.ezyang.com/2010/05/bananas-lenses-envelopes-and-barbed-wire-a-translation-guide/`](http://blog.ezyang.com/2010/05/bananas-lenses-envelopes-and-barbed-wire-a-translation-guide/)

自从夏天开始以来，我一直在缓慢地重新阅读一篇论文，这篇论文是 Erik Meijer、Maarten Fokkinga 和 Ross Paterson 的《香蕉、透镜、信封和铁丝网的函数式编程》。如果你想知道 {cata,ana,hylo,para}morphisms 是什么，这篇论文是必读的：第二节为所爱的链表提供了一个非常易读的形式化定义。

然而上次，当他们开始讨论代数数据类型时，我的眼睛有点发直，尽管我在 Haskell 中已经使用和定义了它们；部分原因是我感到自己淹没在三角形、圆形和波浪线的海洋中，当他们讨论基本组合子的定律时，我甚至可能会说：“这全都是数学！”

更仔细地阅读揭示了实际情况，所有这些代数运算符都可以用简单的 Haskell 语言书写出来，对于那些在 Haskell 中已经有一些时间的人来说，这可以提供更流畅（尽管更冗长）的阅读体验。因此，我呈现这份翻译指南。

*类型运算符。* 按照惯例，类型用 ![A, B, C\ldots](img/ldots") 表示在左边，而 `a, b, c...` 表示在右边。我们将其与函数运算符区分开来，尽管本文没有并且依赖于惯例来区分这两者。

```
 Bifunctor t => t a b
 Functor f => f a
 [a]
 (d, d')
 Either d d'
 Identity
 Const d
 (Functor f, Functor g) => g (f a)
 (Bifunctor t, Functor f, Functor g) => Lift t f g a
 ()

```

（对于学究们来说，你需要在所有的 Bifunctors 后面加上 `Hask Hask Hask`。）

*函数运算符。* 按照惯例，函数用 ![f, g, h\ldots](img/ldots") 表示在左边，而 `f :: a -> b, g :: a' -> b', h...` 表示在右边（类型根据需要统一）。

```
 bimap f g :: Bifunctor t => t a a' -> t b b'
 fmap f :: Functor f => f a -> f b
 f *** g :: (a, a') -> (b, b')
    where f *** g = \(x, x') -> (f x, g x')
 fst :: (a, b) -> a
 snd :: (a, b) -> b
 f &&& g :: a -> (b, b')        -- a = a'
    where f &&& g = \x -> (f x, g x)
 double :: a -> (a, a)
    where double x = (x, x)
 asum f g :: Either a a' -> Either b b'
    where asum f g (Left x)  = Left (f x)
          asum f g (Right y) = Right (g y)
 Left :: a -> Either a b
 Right :: b -> Either a b
 either f g :: Either a a' -> b        -- b = b'
 extract x :: a
    where extract (Left x) = x
          extract (Right x) = x
 (f --> g) h = g . h . f
    (-->) :: (a' -> a) -> (b -> b') -> (a -> b) -> a' -> b'
 (g <-- f) h = g . h . f
    (<--) :: (b -> b') -> (a' -> a) -> (a -> b) -> a' -> b'
 (g <-*- f) h = g . fmap h . f
    (<-*-) :: Functor f => (f b -> b') -> (a' -> f a) -> (a -> b) -> a' -> b'
 id f :: a -> b
 const id f :: a -> a
 (fmap . fmap) x
 const ()
 fix f

```

现在，让我们来看看 *abides law*：

被翻译成 Haskell 后，这一句是：

```
either (f &&& g) (h &&& j) = (either f h) &&& (either g j)

```

对我来说（至少是这样）更有意义：如果我想从 Either 中提取一个值，然后对其运行两个函数并返回结果的元组，我也可以立即将该值分成一个元组，并使用不同的函数从 either 中“两次”提取值。（尝试手动在 `Left x` 和 `Right y` 上运行该函数。）
