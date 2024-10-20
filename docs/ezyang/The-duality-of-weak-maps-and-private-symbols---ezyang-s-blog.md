<!--yml

category: 未分类

date: 2024-07-01 18:17:23

-->

# 弱映射和私有符号的二元性：ezyang 的博客

> 来源：[`blog.ezyang.com/2013/03/duality-of-weak-maps-and-private-symbols/`](http://blog.ezyang.com/2013/03/duality-of-weak-maps-and-private-symbols/)

*来自 ECMAScript TC39 会议记录文件*

我想讨论马克·米勒指出的一个有趣的二元性，即弱映射和私有符号之间的二元性，它们本质上是不同语言特性！

[弱映射](http://wiki.ecmascript.org/doku.php?id=harmony:weak_maps)是一种普通的关联映射，其特点是如果任何条目的键变得不可访问，那么值也将变得不可访问（尽管您必须忽略值本身对键的引用！）弱映射具有多种用途，包括记忆化，我们希望记住计算的结果，但仅当可能再次请求时！弱映射支持`get(key)`和`set(key, value)`操作。

[私有符号](http://tc39wiki.calculist.org/es6/symbols/)是对象上字段的不可伪造标识符。符号很有用，因为它们可以“新鲜”生成；也就是说，它们确保不会与对象上已有的字段冲突（而使用 `_private_identifier_no_really` 可能会不够幸运）；私有符号有额外的规定，即在不具备符号的情况下无法发现它存在于对象上—例如，在枚举对象属性时，对象将拒绝透露私有符号的存在。可以创建一个私有符号 `psym`，然后像普通属性名一样使用它来获取（`obj[psym]`）和设置（`obj[psym] = value`）值。

要了解它们为何相同，请使用私有符号来实现弱映射，反之亦然（警告，前面是伪代码）：

```
function WeakMap() {
  var psym = PrivateSymbol();
  return {
    get: function(key) { return key[psym]; },
    set: function(key, value) { key[psym] = value; }
  }
}

function PrivateSymbol() {
  return WeakMap();
}
// pretend that get/set are magical catch-all getters and setters
Object.prototype.get = function(key) {
  if (key instanceof PrivateSymbol) { return key.get(this); }
  else { return this.old_get(key); }
}
Object.prototype.set = function(key, value) {
  if (key instanceof PrivateSymbol) { return key.get(this, value); }
  else { return this.old_set(key, value); }
}

```

特别注意，枚举弱映射的所有条目是没有意义的；这样的枚举会因垃圾收集器的运行而任意更改。

如果你更仔细地观察这一点，会发现有一些非常有趣的事情正在发生：弱映射和私有符号的*实现*策略是相反的。对于弱映射，你可能想象一个类似于实际映射的数据结构，它从键到值的映射（加上一些垃圾收集的技巧）；而对于私有符号，你期望的是将值存储在对象本身上。也就是说，如果我们说“WeakMap = PrivateSymbol”和“key = this”，那么主要的区别在于关系是存储在 WeakMap/PrivateSymbol 上，还是存储在 key/this 上。WeakMap 暗示前者；PrivateSymbol 暗示后者。

其中一个实现比另一个更好吗？如果系统中的对象是不可变的或不能任意扩展的，那么私有符号的实现可能是不可能的。但如果两种实现都可能，那么哪种更好取决于相关对象的*生命周期*。垃圾收集弱引用是一件昂贵的事情（它的效率远低于普通的垃圾收集），因此如果你可以通过正常的垃圾收集使你的弱映射死去，那就是一种胜利。因此，最好将映射存储在*生命周期较短*的对象上。在记忆表的情况下，键将比映射更短暂，这导致了一个非常奇怪的结果：*对于弱映射的最佳实现策略根本不涉及创建映射！*

不幸的是，与许多优雅的结果一样，ECMAScript 规范的其他部分复杂性导致了一些困难。特别是，“只读弱映射”意味着什么完全不清楚，而“只读私有符号”却有明显的含义。此外，将这两个相当不同的概念合并为一种语言可能只会使 web 开发人员感到困惑；这是一个提案过于巧妙以至于不利于自身的情况。最后，关于如何[将私有状态与代理结合](http://wiki.ecmascript.org/doku.php?id=strawman:proxy_symbol_decoupled)仍在进行讨论。这个提案被引入来解决这个问题的一个特定方面，但据我们了解，它只解决了一个具体的子问题，并且*只有*在相关的代理是[膜](http://blog.ezyang.com/2013/03/what-is-a-membran/)时才有效。
