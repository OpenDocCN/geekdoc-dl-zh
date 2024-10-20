<!--yml

category: 未分类

date: 2024-07-01 18:17:55

-->

# HTML 净化器 4.3.0 发布：ezyang's 博客

> 来源：[`blog.ezyang.com/2011/03/html-purifier-4-3-0-released/`](http://blog.ezyang.com/2011/03/html-purifier-4-3-0-released/)

## HTML 净化器 4.3.0 发布

发布周期变得越来越长……可能会让所有下游开心吧。

* * *

[HTML 净化器](http://htmlpurifier.org) 4.3.0 是一个重要的安全发布版本，解决了与用户提交的代码和合法客户端脚本相关的各种安全漏洞。它还包含了半年来的新功能和错误修复累积。新的配置选项包括%CSS.Trusted、%CSS.AllowedFonts 和%Cache.SerializerPermissions。为了定制原始定义，API 发生了不兼容的变化，请参阅[定制文档](http://htmlpurifier.org/docs/enduser-customize.html#optimized)获取详细信息。

HTML 净化器是一个用 PHP 编写的符合标准的 HTML 过滤器库。

*无关逻辑。* 在研究这个版本的 HTML 净化器中修复的安全漏洞时，我有了一个想法：在 JavaScript 中使用高阶函数编程有多容易？JavaScript 在传递函数方面非常流畅（有人可能会说它的面向对象编程设施只是在某些基础结构上放置函数），但由于缺乏类型系统，可能会很烦人，需要说明某个特定函数具有类似`Direction -> DataflowLattice -> (Block -> Fact -> (DG, Fact)) -> [Block] -> (Fact -> DG, Fact)`（简化的真实例子，我不是在开玩笑！）。我在 Python 中的经验是，向同事解释这种事情需要花费太多时间，调试也是个头疼的问题（检查函数以查看实际内容很困难），所以最好不要涉及。
