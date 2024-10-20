<!--yml

category: 未分类

date: 2024-07-01 18:17:16

-->

# 可视化块分配器：ezyang 的博客

> 来源：[`blog.ezyang.com/2013/10/visualizing-a-block-allocator/`](http://blog.ezyang.com/2013/10/visualizing-a-block-allocator/)

## 可视化块分配器

GHC 的[block allocator](http://ghc.haskell.org/trac/ghc/wiki/Commentary/Rts/Storage/BlockAlloc)是一个非常棒的低级基础设施。它提供了一种更灵活的管理堆的方式，而不是试图把所有内容都塞进一块连续的内存块中，可能对于像运行时这样实现低级代码的任何人都应该是一件通常感兴趣的事情。其核心思想相当古老（BIBOP: 大袋子页），对于任何对象都标有相同描述符的情况并且不想为每个对象支付标签的成本都非常有用。

管理比页面大的对象有些棘手，因此我写了一篇文档来可视化这种情况，以帮助自己理解。我想这可能会引起一般兴趣，所以你可以在这里获取它：[`web.mit.edu/~ezyang/Public/blocks.pdf`](http://web.mit.edu/~ezyang/Public/blocks.pdf)

总有一天我会把它转换成可维基形式，但今天我不想处理图像...
