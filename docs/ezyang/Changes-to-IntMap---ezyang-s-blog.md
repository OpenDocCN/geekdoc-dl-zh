<!--yml

category: 未分类

date: 2024-07-01 18:17:41

-->

# IntMap 的更改：ezyang's 博客

> 来源：[`blog.ezyang.com/2011/08/changes-to-intmap/`](http://blog.ezyang.com/2011/08/changes-to-intmap/)

## IntMap 的更改

目前，使用当前 containers API 无法对 [IntMaps](http://hackage.haskell.org/packages/archive/containers/0.4.0.0/doc/html/Data-IntMap.html) 定义某些严格值操作。例如，读者可以尝试高效地实现 `map :: (a -> b) -> IntMap a -> IntMap b`，使得对于非底部和非空映射 `m`，`Data.IntMap.map (\_ -> undefined) m == undefined`。

现在，我们本可以简单地在现有的 API 上添加大量带有撇号后缀的操作，这样会大大增加其大小，但根据[libraries@haskell.org 上的讨论](http://www.haskell.org/pipermail/libraries/2011-May/016362.html)，我们决定将模块拆分为两个模块：`Data.IntMap.Strict` 和 `Data.IntMap.Lazy`。为了向后兼容，`Data.IntMap` 将成为模块的惰性版本，而当前存放在此模块中的值严格函数将被弃用。

发生的细节有点微妙。这是读者文摘版本：

+   `Data.IntMap.Strict` 中的 `IntMap` 和 `Data.IntMap.Lazy` 中的 `IntMap` 是完全相同的映射；在两者之间没有运行时或类型级别的差异。用户可以通过导入一个模块或另一个模块来在“实现”之间切换，但我们不会阻止您在严格映射上使用惰性函数。您可以使用 `seqFoldable` 将惰性映射转换为严格映射。

+   类似地，如果将具有惰性值的映射传递给严格函数，则该函数将在映射上执行最大惰性操作，以确保在严格情况下仍然正确操作。通常情况下，这意味着惰性值可能不会被评估……除非它是。

+   大多数类型类实例对严格和惰性映射都有效，但 `Functor` 和 `Traversable` 没有遵守适当法律的有效“严格”版本，因此我们选择了它们的惰性实现。

+   惰性和严格折叠保持不变，因为折叠是否严格独立于数据结构是值严格还是脊椎严格。

我在 Hac Phi 上星期日为严格模块编写了第一个版本，您可以[在此处查看](http://hpaste.org/49733)。完整实现可以[在此处找到](https://github.com/ezyang/packages-containers)
