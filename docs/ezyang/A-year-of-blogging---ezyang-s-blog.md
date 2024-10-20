<!--yml

category: 未分类

日期：2024-07-01 18:18:00

-->

# 一年的博客：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/12/a-year-of-blogging/`](http://blog.ezyang.com/2010/12/a-year-of-blogging/)

## 博客的一年

在这里庆祝一年的博客。感谢大家的阅读。仅仅是[一年前](http://blog.ezyang.com/2009/12/iron-blogger/)，我第一次在 Iron Blogger 的翅膀下开设了这个博客。Iron Blogger 在此时大部分已经解散，但我很自豪地说这个博客没有停止发布，每周三次（除了那次我错过了一篇文章，后来进行了补偿性的额外发布），这是我与自己打赌并且很高兴赢得的赌注。

这个博客在过去的一年里走过了哪些地方？根据 Google Analytics 的数据，这里是十篇最受欢迎的文章：

1.  [图而非网格：缓存如何腐蚀年轻算法设计师并修复方法](http://blog.ezyang.com/2010/07/graphs-not-grids/)

1.  [你本可以发明拉链（zippers）](http://blog.ezyang.com/2010/04/you-could-have-invented-zippers/)

1.  [中世纪医学与计算机](http://blog.ezyang.com/2010/11/medieval-medicine-and-computers/)

1.  [数据库即分类](http://blog.ezyang.com/2010/06/databases-are-categories/)

1.  [Haskell 中的设计模式](http://blog.ezyang.com/2010/05/design-patterns-in-haskel/)

1.  [每日静态分析（非博士）人士](http://blog.ezyang.com/2010/06/static-analysis-mozilla/)

1.  [MVC 与纯度](http://blog.ezyang.com/2010/07/mvc-and-purity/)

1.  [Galois 实习生一天的生活](http://blog.ezyang.com/2010/08/day-in-the-life-of-a-galois-intern/)

1.  [用 Haskell 替换小型 C 程序](http://blog.ezyang.com/2010/03/replacing-small-c-programs-with-haskell/)

1.  [如何像专家一样使用 Vim 的 textwidth](http://blog.ezyang.com/2010/03/vim-textwidth/)

可能还有一些我个人最喜欢的不那么显眼的文章，但到这个点我写了这么多，有点难以数清：包括这篇文章，我将已发布 159 篇文章，总字数约为 120,000 字。 （此数据包含标记，但用于比较，一本书大约有 80,000 字。哇哦，我写了一本半的内容。不过我不觉得自己是个更好的作家——这可能是因为我在“修订”这部分的过程中偷懒了。）

这个博客将在一月份进行短暂的休息。不是因为我在假期期间不能发布文章（如果有机会，我可能会... 实际上，这是一个有点难做出的决定），而是因为我应该花一个月的大部分空闲时间集中在除了博客之外的事情上。祝大家新年快乐，我们二月见！

*附言.* 这是我用来计数的 SQL 查询：

```
select
  sum( length(replace(post_content, '  ', '')) - length(replace(post_content, ' ', ''))+1)
from wp_posts
where post_status = 'publish';

```

或许有更精确的方法，但我懒得写脚本。
