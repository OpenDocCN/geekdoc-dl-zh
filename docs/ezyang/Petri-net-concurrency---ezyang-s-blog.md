<!--yml

category: 未分类

date: 2024-07-01 18:17:57

-->

# Petri 网并发 : ezyang's 博客

> 来源：[`blog.ezyang.com/2011/03/petri-net-concurrency/`](http://blog.ezyang.com/2011/03/petri-net-concurrency/)

## Petri 网并发

一个 [Petri 网](http://en.wikipedia.org/wiki/Petri_net) 是一种有趣的小型图形建模语言，用于并发控制流。几周前在这次演讲中提到了它们：[Petri-nets as an Intermediate Representation for Heterogeneous Architectures](http://talks.cam.ac.uk/talk/index/29894)，但我觉得有趣的是我可以用这种建模语言描述一些常见的并发结构。

例如，这里是备受推崇的锁：

解释图的方式是这样的：每个圆圈是一个“Petri 碟”（位置），可能包含一些令牌。方形框（转换）是希望触发的操作，但为了执行这些操作，所有输入它们的 Petri 碟必须有令牌。这种表示方法可以说是可以变成某种棋盘游戏的形式！

如果多个转换可以触发，我们选择其中一个，只有那一个成功；一个令牌沿着一个或另一个箭头流动的能力在这个模型中编码了非确定性。在锁图中，只有一个分支可以获取中间的锁令牌，但它们在退出关键区域（解锁）时将其归还。

这里是一个信号量：

它和之前的完全相同，只是中间的位置可能包含多个令牌。当然，没有人说独立的进程必须在发出信号之前等待。我们可以像这样实现一个简单的生产者-消费者链：

注意，Petri 网中的位置类似于 `MVar ()`，尽管在 Haskell 中需要小心确保我们不是在空中制造令牌，这是由于线性类型的缺失。你可能还注意到，Petri 网对于 *数据流* 并没有说太多；我们可以想象这些令牌代表数据，但形式主义并没有详细说明这些令牌实际上代表什么。
