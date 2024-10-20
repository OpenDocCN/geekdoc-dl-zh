<!--yml

类别: 未分类

日期: 2024-07-01 18:18:20

-->

# Spring 2010: A Random Walk : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/05/spring-2010-a-random-walk/`](http://blog.ezyang.com/2010/05/spring-2010-a-random-walk/)

## Spring 2010: A Random Walk

在 2010 年春季学期的前夕，我决定在我的笔记本电脑上运行这个小实验：在过去的六个月内，我修改了哪些文件？

```
find . \( -path '*/.*' \) -prune -o -mtime -180 -print

```

结果是修改了超过一百五十万个文件。以下是（稍微）删节的版本：

+   LaTeX 文件，用于["Adventures in Three Monads"](http://blog.ezyang.com/2010/01/adventures-in-three-monads/)，这篇文章发表在 Monad Reader 上。还有我在 Advanced Typeclasses 课上的黑板图表，我最终没有能够使用为 Reader 准备的材料。

+   `valk.txt`，其中包含我对 Valkyrie Nethack 角色的笔记。我在 3 月{24,25}日首次升级。

+   作为本科研究项目的一部分，一个 Eclipse Java 项目，用作我的[HAMT 实验的跳板](http://blog.ezyang.com/2010/03/the-case-of-the-hash-array-mapped-trie/)。

+   `htmlpurifier-web`和`htmlpurifier`，感谢我在过去一个月内推出的[HTML Purifier 4.1](http://htmlpurifier.org/)版本。这也意味着我为我的超级神奇的 PHP [多版本农场](http://repo.or.cz/w/phpv.git)编译了新版本的 PHP 5.2.11, 5.2.12, 5.2.13, 5.3.1 和 5.3.2。自言自语，下次记得从自动备份中排除*构建目录*，kthxbai。

+   一个`qemu`的检出，我试图在他们的 DHCP 代码中修复同一 MAC 地址请求两个不同 IP 地址的问题，但放弃了，并为我们用于演示实时进程迁移的虚拟机分配了静态地址。嗯... [6.828 final project](http://pdos.csail.mit.edu/6.828/)。

+   `hackage-server`和`holumbus`的检出，源自于让 Holombus 和 Hackage 合作，实现所有 Haskell 函数最新索引的未竞成功梦想。听说 Holumbus 团队一直在做出改变，以便 Hayoo 能够增量更新其索引。

+   更新以整理`extort`，这是一个用 Haskell 编写的会费追踪应用程序，因为刺客公会的领导最近发生了变化。在换届选举期间，有一位候选人的建议问题是“你懂 Haskell 吗？”我们将看看这个程序能坚持多久...

+   一个`abc`源目录，在那里我展示了我的 C 源码技能，并搜索了如何使用该库的信息。我可能会在 Galois 实习期间与它密切合作。有趣的是，这几乎与为 6.005 编写的 SAT 求解器以及我在计算复杂性课程 6.045 中研究 SAT 的时间重叠。

+   一个`mit-scheme`的检出，用于分析他们的红黑树实现，以弄清楚它是否可以轻松持久化（答案是否定的，并且我不得不根据 Okasaki 的笔记编写了自己的实现），以及弄清楚为什么`--batch-mode`没有按照[它所说的去做](http://blog.ezyang.com/2010/04/art-code-math-and-mit-scheme/)。

+   一个`log4j`源码树，我在我的软件工程 6.005 项目中使用过两次。它大多数时候使用起来很顺利，如果你在 Java 中开发软件，我强烈推荐它。

+   有很多用于`wizard`的测试目录（注意备份这些目录也是个坏主意！）。有一天，我将[释放这个软件](http://scripts.mit.edu/wizard/)到世界上，但目前它在 MIT 内部的使用正在增长。

真正的精简版本：

+   半年的语言：Java、Haskell 和 Scheme

+   最大的目录：我没有严格计数，但`linux`、`wizard`和`hackage`都相当大。

+   最佳文件名：`./Dev/exploits/jessica_biel_naked_in_my_bed.c`

经历了漫长而随机的学习之旅，涉及了许多学科、软件和自我研究。在一个月后真正专注于某个领域有一些权衡：一方面，一个月的时间确实足够深入学习任何领域（我对我的博客文章也有同样的感觉；它们是做一些小实验和过渡的借口，但没什么大作为），另一方面，这意味着我继续看到计算机科学的许多具体子领域。随着夏天即将来临，我可能会找到另一个有雄心的项目来利用我的空闲时间（或者给我现有的一些项目一些它们需要的关注）。
