<!--yml

category: 未分类

date: 2024-07-01 18:17:14

-->

# 用动机进行消除（在 Coq 中）：ezyang 的博客

> 来源：[`blog.ezyang.com/2014/05/elimination-with-a-motive-in-coq/`](http://blog.ezyang.com/2014/05/elimination-with-a-motive-in-coq/)

## 用动机进行消除（在 Coq 中）

在像 Coq 这样的证明助手中，消除规则在数据类型的计算中起着重要作用。在他的论文《用动机进行消除》中，Conor McBride 论述道：“我们应该利用假设，不是仅仅在其直接后果上，而是在其对任意目标产生的影响上：我们应该给消除一个动机。” 换句话说，在细化设置中的证明（向后推理）应该利用它们的目标来指导消除。

最近我有机会重新阅读这篇历史性的论文，在此过程中，我想将示例移植到 Coq 中。以下是结果：

> [`web.mit.edu/~ezyang/Public/motive/motive.html`](http://web.mit.edu/~ezyang/Public/motive/motive.html)

这基本上是一个激励约翰·梅杰相等性（也称为异构相等性）的简短教程。链接的文本实质上是论文第一部分的注释版本——我大部分文本重复使用，并在必要时添加了评论。源代码也可以在以下链接找到：

> [`web.mit.edu/~ezyang/Public/motive/motive.v`](http://web.mit.edu/~ezyang/Public/motive/motive.v)
