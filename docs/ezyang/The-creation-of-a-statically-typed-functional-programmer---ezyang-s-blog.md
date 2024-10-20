<!--yml

分类：未分类

日期：2024-07-01 18:17:57

-->

# 一个静态类型的函数式程序员的诞生：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/03/the-creation-of-a-statically-typed-functional-programmer/`](http://blog.ezyang.com/2011/03/the-creation-of-a-statically-typed-functional-programmer/)

## 一个静态类型的函数式程序员的诞生

早在 2009 年初，我被 MIT 独立活动期间所影响；实际上，是两个影响。第一个是 [6.184](http://web.mit.edu/alexmv/6.001/)，在 Scheme 中重新开设的入门计算机科学课程——出于显而易见的原因。但我觉得这还不够：我记得当时觉得 Scheme 很有趣，但并不是我真正想编程的语言。第二个是 Anders Kaseorg 在我结束一个讲座 [Introduction to Web Application Security](http://mit.edu/~ezyang/Public/iap/intro-to-was.html) 后的评论（作为 MIT 新生，我认为自己能够讲一些东西）。讲座的重点之一是所有关于*类型*的事情：也就是说，“字符串”并不能充分表达今天我们应用程序中流动的大多数文本的语义内容。Haskell 出现是为了让你的编译器确保你不会混淆 HTML 和纯文本。

某些事情一定是触动了什么。那年二月，我写道：

> 哇，Haskell 真漂亮。

有人回答道：

> 不要盯太久太阳，你的眼睛会被烧伤。

因此，一个静态类型的函数式程序员诞生了。

*后记。* 我在 Haskell 中的第一个应用是一个拉普拉斯求解器，通过它我也学到了单子（因为一个映射查找返回了一个`Maybe`值，安德斯决定谈一谈 do-notation 和 bind 如何处理它是个好主意。也许我第一次听解释时并没有理解，但最终我确实让程序运行起来了。）
