<!--yml

category: 未分类

date: 2024-07-01 18:17:47

-->

# **Reified laziness**：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/05/reified-laziness/`](http://blog.ezyang.com/2011/05/reified-laziness/)

## Reified laziness

短篇，长篇正在进行中。

[Par monad](http://hackage.haskell.org/packages/archive/monad-par/0.1.0.1/doc/html/Control-Monad-Par.html)中真正的亮点之一是它如何显式地实现了惰性，使用了一个被称为`IVar`的小结构（文献中也称为*I-structures*）。一个`IVar`有点像`MVar`，但是一旦你把一个值放进去，就再也取不出来了（也不允许放入另一个值）。事实上，这恰恰对应于惰性求值。

关键区别在于`IVar`将一个惰性变量的命名（创建`IVar`）和指定将产生该变量结果的代码（在`IVar`上执行`put`操作）分开。对空`IVar`的任何`get`操作都会阻塞，就像再次尝试评估正在评估的 thunk 会阻塞一样（这个过程称为 blackholing），并且一旦“惰性计算”完成（即发生`put`时），就会被满足。

有趣的是，这种构造方式之所以被采纳，正是因为惰性使得并行性的推理变得非常困难。它还为那些希望提供惰性作为内置结构的语言提供了一些指导（提示：将其实现为记忆化的 thunk 可能不是最佳想法！）
