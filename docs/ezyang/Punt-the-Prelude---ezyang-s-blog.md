<!--yml

category: 未分类

日期：2024-07-01 18:18:17

-->

# Punt the Prelude : ezyang’s 博客

> 来源：[`blog.ezyang.com/2010/05/punt-the-prelude/`](http://blog.ezyang.com/2010/05/punt-the-prelude/)

*注意保护注意事项。* Haskell 98 Prelude 中哪些定义容易被隐藏？我非正式地检查了 Prelude 并提到了一些候选项。

`(.)` 在 Prelude 中是函数组合，即 `(b -> c) -> (a -> b) -> a -> c`。但是 #haskell 的用户知道它可能远比这更多：函数 `a -> b` 实际上只是函子，因此更一般化的类型是 `Functor f => (b -> c) -> f b -> f c`，即 fmap。更一般地说，`(.)` 可以表示态射的组合，就像在 [Control.Category](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Control-Category.html) 中一样。

`all`、`and`、`any`、`concat`、`concatMap`、`elem`、`foldl`、`foldl1`、`foldr`、`foldr1`、`mapM_`、`maximum`、`minimum`、`or`、`product`、`sequence_`。这些都是操作列表的函数，可以很容易地泛化为 `Foldable` 类型类；只需用 `Foldable t => t a` 替换 `[a]`。它们可以在 [Data.Foldable](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Data-Foldable.html) 中找到。

`mapM`、`sequence`。这些函数泛化为 `Traversable` 类型类。它们可以在 [Data.Traversable](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Data-Traversable.html) 中找到。

*任何数字函数或类型类。* 瑟斯顿（Thurston）、蒂勒曼（Thielemann）和约翰逊（Johansson）编写了 [numeric-prelude](http://hackage.haskell.org/package/numeric-prelude-0.1.3.4)，它显著重新组织了数字类的层次结构，并且通常更接近它们的数学根源。虽然被称为实验性的，但它已经在更多面向数学的 Haskell 模块中得到应用，如约吉（Yorgey）的 [species](http://hackage.haskell.org/package/species) 软件包。

*任何列表函数。* 许多数据结构看起来和使用起来像列表，并支持一些类似于列表的操作。大多数模块依赖于命名约定，因此像向量、流、字节流等列表样的结构要求您通过限定导入来使用它们自己。不过，有 [Data.ListLike](http://hackage.haskell.org/packages/archive/ListLike/latest/doc/html/Data-ListLike.html)，试图编码这些结构之间的相似之处。[Prelude.Listless](http://hackage.haskell.org/packages/archive/list-extras/0.3.0/doc/html/Prelude-Listless.html) 提供了一个不包含列表函数的 Prelude 版本。

`Monad`、`Functor`。普遍认为 Monad 可能应该是 `Applicative` 的一个实例（而且类别理论家们也许还要您在层次结构中插入 `Pointed` 函数）。[The Other Prelude](http://www.haskell.org/haskellwiki/The_Other_Prelude) 包含了这种另一种组织形式，尽管实际使用起来很笨拙，因为新的类意味着大多数现有的 Monad 库都无法使用。

`repeat`, `until`。这两个函数在[Control.Monad.HT](http://hackage.haskell.org/packages/archive/utility-ht/latest/doc/html/Control-Monad-HT.html)中有一个确实奇怪的泛化。特别是，`repeat`泛化了 identity monad（需要明确的（解）包装），而`until`泛化了`(->) a` monad。

`map`。这是列表的`fmap`。

`zip`, `zipWith`, `zipWith3`, `unzip`。Conal 的[Data.Zip](http://hackage.haskell.org/packages/archive/TypeCompose/latest/doc/html/Data-Zip.html)将 zip 操作泛化为`Zip`类型类。

*IO.* 这里会看到最多的变化，有多个模块在多个不同层次上工作，提供额外的功能。（不幸的是，它们并不真正可组合...）

+   [System.IO.Encoding](http://hackage.haskell.org/packages/archive/encoding/latest/doc/html/System-IO-Encoding.html) 使 IO 函数支持编码，并使用隐式参数允许设置“默认编码”。相关地，[System.UTF8IO](http://hackage.haskell.org/packages/archive/utf8-prelude/latest/doc/html/System-UTF8IO.html) 导出仅针对 UTF-8 的函数。

+   [System.IO.Jail](http://hackage.haskell.org/packages/archive/jail/latest/doc/html/System-IO-Jail.html) 允许您强制输入输出仅在白名单目录和/或句柄上进行。

+   [System.IO.Strict](http://hackage.haskell.org/packages/archive/strict-io/latest/doc/html/System-IO-Strict.html) 提供了 IO 函数的严格版本，因此您不必担心文件句柄用完的问题。

+   [System.Path.IO](http://hackage.haskell.org/packages/archive/pathtype/latest/doc/html/System-Path-IO.html)，虽然不完全是 IO 本身，但提供了类型安全的文件名操作和相应的 IO 函数来使用这些类型。

+   [System.IO.SaferFileHandles](http://hackage.haskell.org/packages/archive/safer-file-handles/latest/doc/html/System-IO-SaferFileHandles.html) 允许在单子区域中使用句柄，并根据它们打开时的 IO 模式对句柄进行参数化。[System.IO.ExplicitIOModes](http://hackage.haskell.org/packages/archive/explicit-iomodes/latest/doc/html/System-IO-ExplicitIOModes.html) 只处理 IOMode。
