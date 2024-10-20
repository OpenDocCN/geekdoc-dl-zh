<!--yml

category: 未分类

date: 2024-07-01 18:17:29

-->

# 应用函子：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/08/applicative-functors/`](http://blog.ezyang.com/2012/08/applicative-functors/)

*关于主要来源的重要性。*

（前言资料。）这篇博客的大多数读者应该至少对[适用函子](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Control-Applicative.html)有所了解：

```
class Applicative f where
  pure :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b

```

这个接口非常方便日常编程（特别是，它使得漂亮的`f <$> a <*> b <*> c`习语变得容易），但它遵守的法律非常糟糕：

```
    [identity] pure id <*> v = v
 [composition] pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
[homomorphism] pure f <*> pure x = pure (f x)
 [interchange] u <*> pure y = pure ($ y) <*> u

```

所以，如果你（像我在二十四小时前一样）还没有看过它，你应该展示这个接口等价于 Applicative：

```
class Functor f => Monoidal f where
  unit :: f ()
  (**) :: f a -> f b -> f (a,b)

```

（顺便说一句，如果你还没有证明对于单子`join :: m (m a) -> m a`等价于`bind :: m a -> (a -> m b) -> m b`，你也应该这样做。）这种表述的定律要好得多：

```
    [naturality] fmap (f *** g) (u ** v) = fmap f u ** fmap g v
 [left identity] unit ** v ≅ v
[right identity] u ** unit ≅ u
 [associativity] u ** (v ** w) ≅ (u ** v) ** w

```

其中 `f *** g = \(x,y) -> (f x, g y)`。我稍微美化了一下，通过使用“等同于”来抑制 `((), a)` 和 `a` 之间的差异，以及 `(a,(b,c))` 和 `((a,b),c)` 之间的差异，对于严格的相等性，你需要一些额外的函数来将结果转换为正确的类型。看起来有一个一般模式，即具有良好法律表述的 API 并不方便编程，而良好编程的表述则没有良好的法律。C’est la vie...但至少它们是等价的！

有了这种表述，可以轻松地陈述交换适用函子遵循的法律：

```
[commutativity] u ** v ≅ v ** u

```

原始论文[使用效果的适用性编程](http://www.soi.city.ac.uk/~ross/papers/Applicative.html)非常值得一读。快去看看吧！这就是本次公共服务通知的结束。
