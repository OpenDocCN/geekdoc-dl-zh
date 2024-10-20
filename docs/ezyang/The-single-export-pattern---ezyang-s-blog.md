<!--yml

category: 未分类

date: 2024-07-01 18:17:22

-->

# 单一导出模式 : 艾德华·杨的博客

> 来源：[艾德华·杨的博客](http://blog.ezyang.com/2013/03/the-single-export-pattern/)

## 单一导出模式

*来自 ECMAScript TC39 会议记录的文件*

**单一导出**指的是一种设计模式，其中模块标识符被重载为表示模块内部的函数或类型。据我所知，“单一导出”这个术语在 ECMAScript TC39 委员会之外并不特别广泛使用；然而，这个想法在其他上下文中也有出现，所以我希望推广这个特定的名称（因为名称具有力量）。

这个基本概念非常简单。在 JavaScript 中，模块经常表示为一个对象：

```
var sayHello = require('./sayhello.js');
sayHello.run();

```

`sayHello`的方法是模块导出的函数。但是`sayHello`本身呢？因为函数也是对象，我们可以想象`sayHello`也是一个函数，因此：

```
sayHello()

```

这将是一个有效的代码片段，也许相当于 `sayHello.run()`。只能以这种方式导出一个符号，但在许多模块中，有一个明显的选择（例如 jQuery 的`$`对象等）。

这种模式在 Haskell 中也很常见，利用了类型和模块存在不同命名空间的事实：

```
import qualified Data.Map as Map
import Data.Map (Map)

```

`Map`现在被重载为类型和模块两者。
