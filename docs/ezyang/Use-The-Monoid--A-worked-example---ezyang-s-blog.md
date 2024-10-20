<!--yml

类别：未分类

日期：2024-07-01 18:18:18

-->

# 使用 Monoid：一个实例：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/05/use-the-monoid/`](http://blog.ezyang.com/2010/05/use-the-monoid/)

*注意保存*。等效的 Haskell 和 Python 程序用于使用状态从数据结构中检索值。然后，我们将 Haskell 程序重构为没有状态，只有一个 monoid 的程序。

一个工作程序员经常需要做的事情是从一些数据结构中提取一些值（可能是多个），可能同时跟踪额外的元数据。有一天我发现自己写下了这段代码：

```
getPublicNames :: Module -> State (Map Name (Set ModuleName)) ()
getPublicNames (Module _ m _ _ (Just exports) _ _) = mapM_ handleExport exports
    where handleExport x = case x of
            EVar (UnQual n) -> add n
            EAbs (UnQual n) -> add n
            EThingAll (UnQual n) -> add n
            EThingWith (UnQual n) cs -> add n >> mapM_ handleCName cs
            _ -> return ()
          handleCName x = case x of
            VarName n -> add n
            ConName n -> add n
          add n = modify (Map.insertWith Set.union n (Set.singleton m))
getPublicNames _ = return ()

```

简而言之，`getPublicNames`遍历`Module`数据结构，查找“公共名称”，每次找到一个名称时，它记录当前模块包含该名称的记录。这使我能够高效地提出问题：“多少（以及哪些）模块使用 FOO 名称？”

Python 中的转录可能如下所示：

```
def getPublicNames(module, ret=None):
    if not ret:
        ret = defaultdict(set)
    if module.exports is None:
        return ret
    for export in module.exports:
        if isinstance(export, EVar) or \
           isinstance(export, EAbs) or \
           isinstance(export, EThingAll):
            ret[export.name].add(module.name)
        elif isinstance(export, EThingWith):
            ret[export.name].add(module.name)
            for cname in export.cnames:
                ret[export.name].add(cname.name)
    return ret

```

这两个版本之间有一些视觉上的差异：

1.  Python 版本可以选择接受预先存在的状态；否则，它将进行初始化并具有引用透明性。然而，Haskell 版本没有默认状态的概念；我们相信用户可以用简单的`runState`运行状态单子。

1.  Python 版本利用鸭子类型来减少代码；我还与假设的面向对象等价数据结构玩得很快。

1.  Python 版本并没有将其代码分离成`handleExport`和`handleCname`，尽管我们确实可以通过几个更多的内联函数来实现。

但除此之外，它们几乎完全以*完全*相同的方式读取和操作，通过改变状态。Python 版本也几乎是尽头；除了将函数推入其成员对象之外，我相信没有更多*“Pythonic”*的方法来做到这一点。然而，Haskell 版本让我觉得痒痒的…

*我们从来没有读出状态！* 这是我们应该使用 Writer 单子而不是 State 单子的明显迹象。然而，有一个轻微的技术困难：Writer 要求被“记录”的值是一个 Monoid，而理论上，`Map k (Set a)` 确实有一个做我们想要的事情的 Monoid 实例，但是对于 `Map k v` 的一般 Monoid 实例则不够。回想一下，一个 monoid 描述了可以“附加”在一起形成该数据的另一个版本的数据。对于 `SetMap`，

1.  *我们想要*一个 monoid 实例，它接受两个 `SetMap` 结构并将映射并集，通过并集那些集合来解决重复。

1.  *默认情况下*，我们得到一个 monoid 实例，它接受两个 `Map` 结构并将映射并集，在冲突发生时更喜欢原始值并丢弃其余的值。

*新类型来拯救*。`newtype` 来了。我们将其称为`SetMap`。用于烹饪新类型的配方如下：

首先，你需要一个新类型声明。在记录语法中显式命名字段为`unDataType`是惯用法，并调用"解包"对象的新类型包装：

```
newtype SetMap k v = SetMap { unSetMap :: Map k (Set v) }

```

接下来，你需要编写你感兴趣的特殊类型类实例。（并可能使用`deriving ...`来[导入任何旧的、默认的实例](http://hackage.haskell.org/trac/haskell-prime/wiki/NewtypeDeriving)，这些实例仍然很好。）

```
instance (Ord k, Ord v) => Monoid (SetMap k v) where
    mempty = SetMap Map.empty
    mappend (SetMap a) (SetMap b) = SetMap $ Map.unionWith Set.union a b
    mconcat = SetMap . Map.unionsWith Set.union . map unSetMap

```

或许需要一些辅助函数：

```
setMapSingleton :: (Ord k, Ord v) => k -> v -> SetMap k v
setMapSingleton k v = SetMap $ Map.singleton k (Set.singleton v)

```

然后就完成了！

```
getPublicNames :: Module -> Writer (SetMap Name ModuleName) ()
getPublicNames (Module _ m _ _ (Just exports) _ _) = mapM_ handleExport exports
    where handleExport x = case x of
            EVar (UnQual n) -> add n
            EAbs (UnQual n) -> add n
            EThingAll (UnQual n) -> add n
            EThingWith (UnQual n) cs -> add n >> mapM_ handleCName cs
            _ -> return ()
          handleCName x = case x of
            VarName n -> add n
            ConName n -> add n
          add n = tell (setMapSingleton n m) -- *
getPublicNames _ = return ()

```

等等，我们使我们的代码更加具体，但它的长度却增加了！也许，亲爱的读者，你可能会因为新的`SetMap`支持代码的存在（它构成了我们所写内容的主要部分，并且高度通用和可重用），而稍微感到安心；不过，除了该代码，我们稍微减少了从`add n = modify (Map.insertWith Set.union n (Set.singleton m))`到`add n = tell (setMapSingleton n m)`的代码量。

更重要的是，我们现在向最终用户表明了这个函数的新契约：我们只会写出值，而不会改变它们。

*我们为什么再次使用了 monad 呢？* 进一步检查显示，我们从未使用过绑定(`>>=`)。事实上，我们并没有真正使用 monad 的任何功能。让我们使我们的代码更加具体：

```
-- This operator is going into base soon, I swear!
(<>) = mappend

getPublicNames :: Module -> SetMap Name ModuleName
getPublicNames (Module _ m _ _ (Just exports) _ _) = foldMap handleExport exports
    where handleExport x = case x of
            EVar (UnQual n) -> make n
            EAbs (UnQual n) -> make n
            EThingAll (UnQual n) -> make n
            EThingWith (UnQual n) cs -> make n <> foldMap handleCName cs
            _ -> mempty
          handleCName x = case x of
            VarName n -> make n
            ConName n -> make n
          make n = setMapSingleton n m
getPublicNames _ = mempty

```

函数的使用者现在不再需要`execWriter`，虽然他们可能最终需要用`unSetMap`来解包输出，但这里并没有太多的空间变化。

*技术上，我们从未需要 monoid.* 特别是，`setMapSingleton`强迫我们的代码迎合`SetMap`，而不是一般的 Monoids（这也不太合理）。也许"Pointed" Monoid 的概念会有用。所以我们本可以直接写出所有函数；更有可能的是，我们可以定义另一组辅助函数以减少代码大小。*但你仍然应该使用 monoid.* Monoids 有一些特定的行为方式（例如，monoid 法则）和一组规范的操作函数。通过使用这些函数，即使他们不熟悉你的特定 monoid，其他人也可以快速推理你的代码。

*后记.* 在写这篇博客文章时，我重构了真实的代码；所有的例子都不是虚构的。我最初计划写一些关于"You ain't gonna need it"和 Haskell 抽象的内容，但是完善这个例子的过程比我预期的要长一些。也许下次吧...

*后脚注.* Anders Kaseorg 指出，SetMap 已在几个地方（Criterion.MultiMap, Holumbus.Data.MultiMap）实现了，但尚未放入一个特别通用的库中。
