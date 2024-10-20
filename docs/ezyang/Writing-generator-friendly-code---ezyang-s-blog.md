<!--yml

category: 未分类

日期：2024-07-01 18:18:26

-->

# 编写生成器友好的代码：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/03/writing-generator-friendly-code/`](http://blog.ezyang.com/2010/03/writing-generator-friendly-code/)

## 编写生成器友好的代码

我从[向 html5lib 列表抱怨 Python 版本过度使用生成器，导致难以移植到 PHP](http://www.mail-archive.com/html5lib-discuss@googlegroups.com/msg00241.html)走了很远。 现在已经沉迷于 Haskell 的惰性编程，我喜欢尝试使我的代码符合生成器习惯。 虽然 Python 生成器与无限惰性列表相比有显著缺点（例如，将它们分叉以供多次使用并不简单），但它们非常不错。

不幸的是，我看到的大多数期望看到列表的代码对生成器的接受程度不够高，当我不得不说`list(generator)`时，我很伤心。 如果你的内部代码期望 O(1)访问任意索引，我会原谅你，但我经常看到的是只需要顺序访问却因为调用`len()`而搞砸一切。 鸭子类型在这种情况下救不了你。

制作代码生成器友好的技巧很简单：**使用迭代接口。** 不要改变列表。 不要请求任意项。 不要请求长度。 这也是`for range(0, len(l))`是*绝对*错误遍历列表的提示； 如果你需要索引，请使用`enumerate`。

**更新（2012 年 9 月 1 日）。** 令人发笑的是，PHP **终于引入了生成器**。
