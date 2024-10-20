<!--yml

category: 未分类

date: 2024-07-01 18:17:41

-->

# 约瑟夫与令人惊叹的彩色箱子：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/08/joseph-and-the-amazing-technicolor-box/`](http://blog.ezyang.com/2011/08/joseph-and-the-amazing-technicolor-box/)

## 约瑟夫与令人惊叹的彩色箱子

在 Haskell 中考虑以下数据类型：

```
data Box a = B a

```

有多少类型为 `Box a -> Box a` 的可计算函数？如果我们严格使用表义语义，有七个：

但如果我们进一步区分底部的*源头*（一个非常操作性的概念），一些具有相同表达的函数有更多的实现方式……

1.  *不可反驳的模式匹配:* `f ~(B x) = B x`。无多余内容。

1.  *身份:* `f b = b`。无多余内容。

1.  *严格:* `f (B !x) = B x`。无多余内容。

1.  *常数箱底:* 有三种可能性：`f _ = B (error "1")`；`f b = B (case b of B _ -> error "2")`；以及 `f b = B (case b of B !x -> error "3")`。

1.  *不存在:* 有两种可能性：`f (B _) = B (error "4")`；以及 ``f (B x) = B (x `seq` error "5")``。

1.  *严格的常数箱底:* `f (B !x) = B (error "6")`.

1.  *底部:* 有三种可能性：`f _ = error "7"`；`f (B _) = error "8"`；以及 `f (B !x) = error "9"`。

列表按彩虹颜色排序。如果这对你来说像象形文字一样难懂，我可以向你介绍[这篇博文？](http://blog.ezyang.com/2010/12/gin-and-monotonic/)

*附录.* GHC 可以且将优化 `f b = B (case b of B !x -> error "3")`，``f (B x) = B (x `seq` error "5")`` 和 `f (B !x) = error "9"` 到替代形式，因为一般来说我们不会说 `seq (error "1") (error "2")` 在语义上等同于 `error "1"` 或 `error "2"`：由于不精确的异常，任何一个都有可能。但是如果你真的在乎，你可以使用 `pseq`。然而，即使使用异常集语义，这个“精炼”视图中仍有更多函数的正常表示语义。
