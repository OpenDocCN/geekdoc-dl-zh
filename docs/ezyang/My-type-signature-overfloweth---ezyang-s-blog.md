<!--yml

category: 未分类

date: 2024-07-01 18:18:10

-->

# My type signature overfloweth : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/09/my-type-signature-overfloweth/`](http://blog.ezyang.com/2010/09/my-type-signature-overfloweth/)

我最近开始研究使用 *会话类型* 进行实际编码，这个想法从我曾参与构建网络协同文本编辑器团队开始，我就一直在思考。当时我花了大量时间仔细审查服务器和客户端，以确保它们实现了正确的协议。这些协议的本质通常相对简单，但在错误流（例如断开连接后的重新同步）存在的情况下很快变得复杂起来。错误条件也很难进行自动化测试！因此，静态类型似乎是解决这一任务的一种吸引人的方式。

Haskell 中有三种会话类型的实现：[sessions](http://hackage.haskell.org/package/sessions)，[full-sessions](http://hackage.haskell.org/package/full-sessions) 和 [simple-sessions](http://hackage.haskell.org/package/simple-sessions)。如果你感到特别天真，你可能会尝试访问 [Haddock 页面](http://hackage.haskell.org/packages/archive/sessions/2008.7.18/doc/html/Control-Concurrent-Session.html) 来了解 API 的外观。在继续阅读之前，请检查那个页面。

* * *

眼睛剜出来了吗？我们继续吧。

在《Coders at Work》的采访中，Simon Peyton Jones 提到类型的一个显著好处是它提供了函数可能做什么的简明、清晰描述。但那个 API 根本不是简明和清晰的，我仅仅通过查看相应的函数定义就无法理解它。因此，当前会话类型编码的一个关键卖点是它们不会破坏类型推断：我们放弃用户理解一堆类型类代表的含义，只期待传输一个信息位，“协议是否匹配？”

这个问题并不是会话类型的根本问题：任何大量使用类型类的功能都很容易陷入这些冗长的类型签名中。对于如何更好地向用户展示这种复杂性，我有两个（相当未完成的）想法，尽管并不能完全消除：

+   类型系统黑客的一种喜爱的消遣是使用 Peano 数（`Z`和`S a`）对自然数进行类型级编码，附加到类似于`Vector (S (S Z))`的东西。Vector 是一个类型构造器，类型为`* -> *`。然而，由于 Haskell 中只有一个原始种类，我们实际上可以将任何类型传递给 Vector，比如说`Vector Int`，这将是荒谬的。防止这种情况发生的一种方法是声明我们的 Peano 数是类型类`Nat`的实例，然后声明`Nat a => Vector a`。但是，由于在任何这样的语句中`a`只使用一次，如果我们能够写成`Vector :: Nat -> *`，那不是更好吗？如果需要指定类型相等性，可以想象某种类型模式匹配`concat :: Vector a -> Vector b -> Vector c with c ~ a :+: b`。[类型和种类的折叠](http://byorgey.wordpress.com/2010/08/05/typed-type-level-programming-in-haskell-part-iv-collapsing-types-and-kinds/)是朝这个方向迈出的有趣一步。

+   当数学家提出证明时，他们可能会明确地指定“对于所有的 F，使得 F 是一个字段……”，但更频繁地，他们会说类似于“在以下证明中，假设以下变量命名约定。” 这样一来，他们就避免了反复显式地重新声明所有变量名的含义。对于类型变量的类似系统将大大减少长类型签名的需求。

但实际上，这与我当前正在研究的内容无关。

* * *

我正在看的是：会话类型还受到另一种类型签名爆炸现象的困扰：协议中的任何函数在其类型中包含从该时刻起整个协议的完整规范。正如[Neubauer and Thiemann 承认（PDF）](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.70.7370&rep=rep1&type=pdf)，“对完整 SMTP 的会话类型相当难以阅读。” 我正在追求的两条研究路线如下：

+   是否可以通过在会话类型中构建异常支持（目前是一个未解决的问题），允许通过省略与错误情况对应的会话类型来实现更简单的会话类型？

+   是否可以使用`type`来允许协议的单一全局规范，然后个别函数简单地引用它？我们需要更强大的一些东西吗？

到目前为止，我只是在进行思考和阅读论文，但我希望很快开始编写代码。不过我很乐意听听你的想法。
