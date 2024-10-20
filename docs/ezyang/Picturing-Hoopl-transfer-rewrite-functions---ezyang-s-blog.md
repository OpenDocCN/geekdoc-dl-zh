<!--yml

category: 未分类

日期：2024-07-01 18:17:58

-->

# 描绘 Hoopl 的传输/重写函数：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/02/picturing-hoopl-transferrewrite-functions/`](http://blog.ezyang.com/2011/02/picturing-hoopl-transferrewrite-functions/)

## 描绘 Hoopl 的传输/重写函数

[Hoopl](http://hackage.haskell.org/package/hoopl) 是一个“高阶优化库”。为什么称之为“高阶”？因为使用 Hoopl 的用户只需编写优化的各种片段，而 Hoopl 将把它们组合在一起，就像使用 fold 的人只需编写函数在一个元素上的操作，而 fold 将把它们组合在一起一样。

不幸的是，如果您对问题结构不熟悉，那么以这种风格编写的代码可能有点难以理解。幸运的是，Hoopl 的两个主要高阶组成部分：传输函数（收集程序数据）和重写函数（利用数据重写程序），相对较容易可视化。
