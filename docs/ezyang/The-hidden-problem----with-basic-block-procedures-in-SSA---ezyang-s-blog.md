<!--yml

category: 未分类

date: 2024-07-01 18:16:57

-->

# SSA 中基本块过程的隐藏问题：ezyang 博客

> 来源：[`blog.ezyang.com/2020/10/the-hidden-problem-with-basic-block-procedures-in-ssa/`](http://blog.ezyang.com/2020/10/the-hidden-problem-with-basic-block-procedures-in-ssa/)

多年前，Nadav Rotem 向我讲述了关于为什么 Swift 中的基本块过程并不像它们看起来那么好的故事。Nelson Elhage 在 Twitter 上提醒我这件事，所以我觉得这应该被记录在公众记录中。

基本块过程使得某些优化变得更加困难。考虑以下程序：

```
block j3 (%y1, %y2) { ... }
block j1 () { jump j3(%x1, %x2) }
block j2 () { jump j3(%x3, %x4) }

```

这个程序比传统的带有 phi 节点公式的 SSA 更容易还是更难优化？

```
L1:
   goto L3
L2:
   goto L3
L3:
   %y1 = phi [%x1, %L1] [%x3, %L2]
   %y2 = phi [%x2, %L1] [%x4, %L2]

```

假设优化器确定 j3/L3 内的 y1 未被使用并且可以被消除。在基本块环境中，通过删除 "y1 = phi x1 x3" 就可以简单地消除 y1。然而，在连接点环境中，你不仅需要消除 y1，还要更新 j3 的所有调用点，因为你已经改变了函数签名。在可变 AST 中，改变函数签名很麻烦；特别是，你必须做的突变包括一些无效的中间状态（这很容易意外触发断言）。

当我看到这个例子时，我想知道为什么 GHC（其具有基本块过程道德等价的连接点形式）没有这个问题。嗯，事实证明，这种优化可以作为一系列局部转换来完成。首先，我们进行一个工作者/包装器转换，引入一个中间块（工作者），丢弃无用参数：

```
block j3 (%y1, %y2) { jump wj3(%y2) }
block j1 () { jump j3(%x1, %x2) }
block j2 () { jump j3(%x3, %x4) }
block wj3 (%y2) { ... }

```

稍后，我们内联 j3，移除包装器。工作者/包装器是函数式程序的一个非常重要的优化，但很容易想象为什么在可变编译器环境中它不那么受欢迎。
