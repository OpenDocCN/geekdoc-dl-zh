<!--yml

category: 未分类

date: 2024-07-01 18:18:22

-->

# data-accessor 不太重要：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/04/inessential-guide-to-data-accessor/`](http://blog.ezyang.com/2010/04/inessential-guide-to-data-accessor/)

[data-accessor](http://hackage.haskell.org/package/data-accessor-0.2.1.2) 是一个使记录*更有用*的包。与这段代码不同：

```
newRecord = record {field = newVal}

```

你可以写这样：

```
newRecord = field ^= newVal
          $ record

```

特别是 `(field ^= newVal)` 现在是一个值，而不是额外语法的一部分，你可以将其视为一级公民。

当我尝试使用[Chart](http://hackage.haskell.org/package/Chart)（以 criterion 著称）绘制一些数据时，我遇到了这个模块。起初我并没有意识到这一点，直到我尝试了一些代码示例后才意识到，`^=` 并不是 Chart 为自己发明的一个组合子（与你可能在 *xmonad.hs* 中看到的 `-->`、`<+>`、`|||` 和其他朋友们不同）。在使用 Template Haskell 时，Data.Accessor 代表了普通记录系统的一种*替代*，因此了解一个模块何时使用这种其他语言是很有用的。使用 Data.Accessor 的模块的迹象包括：

+   在代码示例中使用 `^=` 运算符

+   所有的记录都有下划线后缀，例如 `plot_lines_title_`

+   Template Haskell 魔法（包括看起来像 `x[acGI]` 的类型变量，尤其是 Template Haskell 生成的“真实”访问器中）。

+   浮动的不合格的 `T` 数据类型。 （正如 Brent Yorgey 告诉我的，这是 Henning-ism，他将定义一个类型 T 或类型类 C，只用于带有限定导入，但 Haddock 会丢弃此信息。如果你不确定，可以在 GHC 中使用 `:t` 来获取此信息。）

一旦确认一个模块确实使用 Data.Accessor，你已经赢得了大部分战斗。这是一个关于如何使用使用 data-accessor 记录的快速教程。

*解释类型。* 一个*访问器*（由类型 `Data.Accessor.T r a` 表示）被定义为一个获取器（`r -> a`）和设置器（`a -> r -> r`）。`r` 是记录的类型，`a` 是可以检索或设置的值的类型。如果使用 Template Haskell 生成定义，`a` 和 `r` 中的多态类型通常会使用看起来像 `x[acGI]` 的类型变量普遍量化，不用太担心它们；你可以假装它们是普通的类型变量。对于好奇的人来说，这些是由 Template Haskell 中的引用单子生成的。

*访问记录字段。* 旧的方法：

```
fieldValue = fieldName record

```

使用 Data.Accessor 可以有多种方式：

```
fieldValue = getVal fieldname record
fieldValue = record ^. fieldname

```

*设置记录字段。* 旧的方法：

```
newRecord = record {fieldName = newValue}

```

新的方法：

```
newRecord = setVal fieldName newValue record
newRecord = fieldName ^= newValue $ record

```

*访问和设置子记录字段。* 旧的方法：

```
innerValue = innerField (outerField record)
newRecord = record {
  outerField = (outerField record) {
    innerField = newValue
  }
}

```

新的方法（这有点像[语义编辑器组合子](http://conal.net/blog/posts/semantic-editor-combinators/)）：

```
innerValue = getVal (outerField .> innerField) record
newRecord = setVal (outerField .> innerField) newValue record

```

还有一些用于修改状态单子内部记录的函数，但我会把这些解释留给[Haddock 文档](http://hackage.haskell.org/packages/archive/data-accessor/0.2.1.2/doc/html/Data-Accessor.html)。现在，继续前行，并以*时尚方式*访问你的数据！
