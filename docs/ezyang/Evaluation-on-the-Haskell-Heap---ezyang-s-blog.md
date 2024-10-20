<!--yml

category: 未分类

日期：2024-07-01 18:17:54

-->

# Evaluation on the Haskell Heap : ezyang’s blog

> 来源：[`blog.ezyang.com/2011/04/evaluation-on-the-haskell-heap/`](http://blog.ezyang.com/2011/04/evaluation-on-the-haskell-heap/)

*The Ghost of the Christmas Present*

In today’s post, we’ll do a brief survey of the various things that can happen when you open haunted presents in the Haskell heap. Asides from constants and things that have already been evaluated, mostly everything on the Haskell heap is haunted. The real question is what the ghost haunting the present does.

在最简单的情况下，几乎什么也不会发生！

与礼品卡不同，你必须打开下一个礼物（Haskell 不允许你评估一个 thunk，然后决定不跟随间接引用...）

More commonly, the ghost was *lazy* and, when woken up, has to open other presents to figure out what was in your present in the first place!

简单的原始操作需要打开所有涉及的礼物。

但是鬼魂也可能会无故地打开另一个礼物...

或者执行一些 IO 操作...

Note that any presents he opens may trigger more ghosts:

结果是一场真正的鬼魂盛会，全都是为了打开一个礼物！

打开一个礼物（thunk）可能会引起这样的级联效应，这正是让惯于认为堆中所有对象都已解包（评估）的人感到惊讶的原因。因此，摆脱这种惊讶的关键在于了解鬼魂何时决定需要打开一个礼物（严格性分析），以及你的礼物是否已经被解包（摊销分析）。

Last time: [The Haskell Heap](http://blog.ezyang.com/2011/04/the-haskell-heap/)

Next time: [IO evaluates the Haskell Heap](http://blog.ezyang.com/2011/04/io-evaluates-the-haskell-heap/)

This work is licensed under a [Creative Commons Attribution-ShareAlike 3.0 Unported License](http://creativecommons.org/licenses/by-sa/3.0/).
