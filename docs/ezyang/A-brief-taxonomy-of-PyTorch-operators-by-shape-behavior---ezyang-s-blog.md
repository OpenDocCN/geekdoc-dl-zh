<!--yml

category: 未分类

date: 2024-07-01 18:16:52

-->

# PyTorch 操作符的简要分类：ezyang’s 博客

> 来源：[`blog.ezyang.com/2020/05/a-brief-taxonomy-of-pytorch-operators-by-shape-behavior/`](http://blog.ezyang.com/2020/05/a-brief-taxonomy-of-pytorch-operators-by-shape-behavior/)

最近，我一直在重新设计如何在 PyTorch 中指定张量形状公式的方法。作为这个过程的一部分，我按其形状行为对每一个 PyTorch 操作符进行了分类；是的，总共有 1364 个（这包括每个操作符的变体，例如原地操作和`out=`关键字的变体）。在这个过程中，我试图提出一些类别来帮助分类操作符的功能。过程中的一个意外是发现我之前认为不常见的形状行为，实际上出现的频率比预期要高一些。

这些类别本身非常有趣，并且可以帮助理解 PyTorch API 的各个部分是如何结合在一起的。以下是我设计的所有类别。

**TensorIterator**（505，例如 add、sum）操作符是 PyTorch 的核心操作；这些操作符执行逐点操作和减少，并支持 [broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html) 和 [type promotion](https://pytorch.org/docs/stable/tensor_attributes.html)。名称 [TensorIterator](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorIterator.h) 指的是 PyTorch 中用于实现这些操作的内部抽象；您可以在 [wiki](https://github.com/pytorch/pytorch/wiki/How-to-use-TensorIterator) 和 [这篇博客文章](https://labs.quansight.org/blog/2020/04/pytorch-tensoriterator-internals/) 上了解更多信息。TensorIterator 在 PyTorch 中是一个真正的工作马：大部分（虽然不是大多数）操作符都是以这种方式实现的！请注意，此类别包括一些使用等效的传统功能的函数（但并非完全使用 TensorIterator）。

**Fixed**（273，例如卷积、addbmm）操作符是仅适用于固定数量维度的操作符。这一假设使得编写高效的内核变得更加容易，因为固定维度的索引计算非常简单。（例如，[TensorAccessor](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/TensorAccessor.h) 是一个内部类，允许您在编译时已知的固定维度上查看张量）。有时，第一个维度被视为批处理维度，但并非总是如此（不幸的是，我在数据集中没有区分这些情况）。有些固定操作符实际上支持多个维度，但只支持固定数量的维度；例如，因为我们只支持 1-3 维卷积，这被视为固定。（与下面的 FeatureBatched 进行比较！）

**N-Dimensional**（107，例如，squeeze，index_add，tensordot）运算符是通用于任意维度张量的运算符。这些是很难以符号形式编写通用形状规则的操作，因为你需要一个能够处理列表操作的语言。N 维运算符的一个重要子类是**Identity**（42，例如，clone，contiguous；不包括上述计数）运算符，可以在任意维度上工作，但它们始终返回与其输入大小相同的张量。另一个子类是**Flatten**（11，例如，take，bucketize）运算符，可以接受任意维度的张量，但在内部始终将它们视为 1D 张量。

**Composite**（95，例如，kl_div，isfinite）运算符是在其他运算符中实现的，它们本身不进行形状检查（而是依赖于它们调用的操作来检查形状）。请注意，这一类别可能有些被低估，因为在某些情况下，当运算符的基本行为显而易见时，我将其分类为该类别，而不是复合类别。

**Batched**（94，例如，nll_loss，adaptive_avg_pool2d）运算符类似于固定维度运算符，但在其开始处接受任意数量的批处理维度。许多固定运算符应该是批处理运算符；其他则不能转换为批处理运算符，否则会引入关于批处理维度结束位置的歧义。与之相比，**FeatureBatched**（19，例如，batch_norm，embedding）运算符类似于批处理运算符，但不是在开始处接受批处理维度，而是在结束处接受任意数量的特征维度。

**Factory**（90，例如，empty）运算符在没有任何张量输入的情况下生成新的张量。

**Trivial**（59，例如，size，is_floating_point）运算符并非实际的张量操作，而是返回非张量信息或访问内部数据结构的方式。

**Sparse**（40）运算符很特殊，因为它们的大小计算考虑了密集和稀疏维度。

**Dynamic**（15，例如，unique）运算符产生的输出形状取决于其输入张量的数据。

**Variadic**（14，例如，cat）运算符接受多个输入张量；与 n 维操作类似，它们难以以符号形式捕捉。

您可以在[`docs.google.com/spreadsheets/d/e/2PACX-1vQQFW0T_bucT5KZn0BHYTC1KYhkL6ZMG5ZxQWc6UmAkHUDYpqkpzXnsb59uv2TB0Jgc1Q6qO63bx6WQ/pubhtml`](https://docs.google.com/spreadsheets/d/e/2PACX-1vQQFW0T_bucT5KZn0BHYTC1KYhkL6ZMG5ZxQWc6UmAkHUDYpqkpzXnsb59uv2TB0Jgc1Q6qO63bx6WQ/pubhtml) 上查看完整数据集。
