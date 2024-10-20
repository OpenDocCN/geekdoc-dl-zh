<!--yml

类别：未分类

日期：2024-07-01 18:17:54

-->

# Hoopl：数据流分析：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/04/hoopl-dataflow-analysis/`](http://blog.ezyang.com/2011/04/hoopl-dataflow-analysis/)

一旦确定了你将收集的[数据流事实](http://blog.ezyang.com/2011/04/hoopl-dataflow-lattices/)，下一步就是编写实际执行此分析的*传递函数*！

记住你的数据流事实的含义，这一步应该相对容易：编写传递函数通常涉及浏览语言中的每个可能语句，并思考它如何改变你的状态。我们将详细介绍常量传播和活跃性分析的传递函数。

* * *

这是活跃性分析的传递函数（再次在`Live.hs`中）：

```
liveness :: BwdTransfer Insn Live
liveness = mkBTransfer live
  where
    live :: Insn e x -> Fact x Live -> Live
    live   (Label _)       f = f
    live n@(Assign x _)    f = addUses (S.delete x f) n
    live n@(Store _ _)     f = addUses f n
    live n@(Branch l)      f = addUses (fact f l) n
    live n@(Cond _ tl fl)  f = addUses (fact f tl `S.union` fact f fl) n
    live n@(Call vs _ _ l) f = addUses (fact f l `S.difference` S.fromList vs) n
    live n@(Return _)      _ = addUses (fact_bot liveLattice) n

    fact :: FactBase (S.Set Var) -> Label -> Live
    fact f l = fromMaybe S.empty $ lookupFact l f

    addUses :: S.Set Var -> Insn e x -> Live
    addUses = fold_EN (fold_EE addVar)
    addVar s (Var v) = S.insert v s
    addVar s _       = s

```

`live`是我们传递函数的核心：它接受一个指令和当前的事实，然后根据这些信息修改事实。因为这是一个向后传递（`BwdTransfer`），传递给`live`的`Fact x Live`是此指令之后的数据流事实，我们的任务是计算这些指令之前的数据流事实（数据流向后流动）。

如果你仔细观察这个函数，会发现有一些非常奇怪的地方：在`live (Label _) f = f`这一行中，我们简单地将`f`（显然具有类型`Fact x Live`）作为结果传递出去。这是如何工作的呢？嗯，`Fact`实际上是一个类型族：

```
type family   Fact x f :: *
type instance Fact C f = FactBase f
type instance Fact O f = f

```

看，又是 O 和 C 虚类型！如果我们回顾一下`Insn`的定义（在`IR.hs`中）：

```
data Insn e x where
  Label  :: Label  ->                               Insn C O
  Assign :: Var    -> Expr    ->                    Insn O O
  Store  :: Expr   -> Expr    ->                    Insn O O
  Branch :: Label  ->                               Insn O C
  Cond   :: Expr   -> Label   -> Label  ->          Insn O C
  Call   :: [Var]  -> String  -> [Expr] -> Label -> Insn O C
  Return :: [Expr] ->                               Insn O C

```

这意味着对于任何在退出时是*开放的*指令（对于 Label、Assign 和 Store 是`x = O`），我们的函数得到`Live`，而对于在退出时是*封闭的*指令（对于 Branch、Cond、Call 和 Return 是`x = C`），我们得到`FactBase Live`，这是一个标签到事实的映射（`LabelMap Live`）——我们稍后会讨论原因。

由于我们接收的指令的参数类型实际上会根据指令的形状而改变，因此一些人（包括 GHC 开发人员在内）更喜欢使用长形式`mkBTransfer3`，它接受三个函数，分别对应每种节点形状。因此，重写后的代码如下所示：

```
liveness' :: BwdTransfer Insn Live
liveness' = mkBTransfer3 firstLive middleLive lastLive
  where
    firstLive :: Insn C O -> Live -> Live
    firstLive (Label _) f = f

    middleLive :: Insn O O -> Live -> Live
    middleLive n@(Assign x _) f = addUses (S.delete x f) n
    middleLive n@(Store _ _)  f = addUses f n

    lastLive :: Insn O C -> FactBase Live -> Live
    lastLive n@(Branch l)      f = addUses (fact f l) n
    lastLive n@(Cond _ tl fl)  f = addUses (fact f tl `S.union` fact f fl) n
    lastLive n@(Call vs _ _ l) f = addUses (fact f l `S.difference` S.fromList vs) n
    lastLive n@(Return _)      _ = addUses (fact_bot liveLattice) n

```

(使用相同的定义来`fact`、`addUses`和`addVar`)。

有了这个理解，解析`firstLive`和`middleLive`的代码应该相当容易。标签不会改变活跃库的集合，因此我们的事实`f`保持不变。对于赋值和存储，表达式中对寄存器的任何使用都会使该寄存器变为活跃（`addUses`是一个计算这一点的实用函数），但如果我们对寄存器赋值，则会*失去*其先前的值，因此它不再是活跃的。以下是一些演示伪代码：

```
// a is live
x = a;
// a is not live
foo();
// a is not live
a = 2;
// a is live
y = a;

```

如果你对`addUses`的实现好奇，`fold_EE`和`fold_EN`函数可以在`OptSupport.hs`中找到：

```
fold_EE :: (a -> Expr -> a) -> a -> Expr      -> a
fold_EN :: (a -> Expr -> a) -> a -> Insn e x -> a

fold_EE f z e@(Lit _)         = f z e
fold_EE f z e@(Var _)         = f z e
fold_EE f z e@(Load addr)     = f (f z addr) e
fold_EE f z e@(Binop _ e1 e2) = f (f (f z e2) e1) e

fold_EN _ z (Label _)       = z
fold_EN f z (Assign _ e)    = f z e
fold_EN f z (Store addr e)  = f (f z e) addr
fold_EN _ z (Branch _)      = z
fold_EN f z (Cond e _ _)    = f z e
fold_EN f z (Call _ _ es _) = foldl f z es
fold_EN f z (Return es)     = foldl f z es

```

命名约定如下：`E` 代表 `Expr`，而 `N` 代表 `Node`（`Insn`）。左边的字母表示传递给结合函数的值的种类，而右边的字母表示正在折叠的内容。因此，`fold_EN` 折叠节点中的所有 `Expr` 并对其调用结合函数，而 `fold_EE` 折叠 `Expr` 中的所有 `Expr`（注意像 `Load` 和 `Binop` 中可能包含内部表达式！）。因此 `fold_EN (fold_EE f)` 的效果是，如果我们正在检查 `Var` 的使用情况，那么 `f` 将在节点中的每个表达式上调用，这正是我们想要的。

我们也可以明确地写出递归：

```
addUses :: S.Set Var -> Insn e x -> Live
addUses s (Assign _ e)      = expr s e
addUses s (Store e1 e2)     = expr (expr s e1) e2
addUses s (Cond e _ _)      = expr s e
addUses s (Call _ _ es _)   = foldl expr s es
addUses s (Return es)       = foldl expr s es
addUses s _                 = s

expr :: S.Set Var -> Expr -> Live
expr s e@(Load e') = addVar (addVar s e) e'
expr s e@(Binop _ e1 e2) = addVar (addVar (addVar s e) e1) e2
expr s e = addVar s e

```

但是正如你所见，递归结构中涉及很多无用的内容，你可能会无意中忘记某个 `Expr`，因此使用预定义的折叠操作符更可取。然而，如果你对复杂数据类型上的折叠不太熟悉，在至少完整写出一次整个内容后也是一个不错的练习。

最后要看的部分是 `lastLives`：

```
lastLive :: Insn O C -> FactBase Live -> Live
lastLive n@(Branch l)      f = addUses (fact f l) n
lastLive n@(Cond _ tl fl)  f = addUses (fact f tl `S.union` fact f fl) n
lastLive n@(Call vs _ _ l) f = addUses (fact f l `S.difference` S.fromList vs) n
lastLive n@(Return _)      _ = addUses (fact_bot liveLattice) n

```

有几个问题需要问。

1.  为什么它接收 `FactBase Live` 而不是 `Live`？这是因为作为向后分析的结束节点，我们可能从多个位置接收事实：控制流可能经过的每个可能路径。

    在 `Return` 的情况下，没有更多的路径，因此我们使用 `fact_bot liveLattice`（无活跃变量）。在 `Branch` 和 `Call` 的情况下，只有一条进一步路径 `l`（我们正在分支或返回的标签），因此我们简单地调用 `fact f l`。最后，在 `Cond` 的情况下，有两条路径：`tl` 和 `fl`，因此我们必须获取它们的事实并将它们与数据流格的结合操作结合。

1.  为什么我们仍然需要调用 `addUses`？因为基本块末尾的指令可以使用变量（`Cond` 在其条件语句中可能使用它们，`Return` 在指定其返回内容时可能使用它们等）。

1.  `Call` 中为什么要调用 `S.difference`？记住，`vs` 是函数调用写入其返回结果的变量列表。因此，我们需要从活跃变量集中移除这些变量，因为它们将被此指令覆盖：

    ```
    f (x, y) {
    L100:
      goto L101
    L101:
      if x > 0 then goto L102 else goto L104
    L102:
      // z is not live here
      (z) = f(x-1, y-1) goto L103
    L103:
      // z is live here
      y = y + z
      x = x - 1
      goto L101
    L104:
      ret (y)
    }

    ```

你应该已经弄清楚了 `fact` 的作用：它查找与标签相关联的数据流事实集，并且如果该标签尚未在我们的映射中，则返回一个空集（无活跃变量）。

* * *

一旦你看过一个 Hoopl 分析，你就看过它们全部了！常量传播的传递函数看起来非常相似：

```
-- Only interesting semantic choice: values of variables are live across
-- a call site.
-- Note that we don't need a case for x := y, where y holds a constant.
-- We can write the simplest solution and rely on the interleaved optimization.
--------------------------------------------------
-- Analysis: variable equals a literal constant
varHasLit :: FwdTransfer Node ConstFact
varHasLit = mkFTransfer ft
 where
  ft :: Node e x -> ConstFact -> Fact x ConstFact
  ft (Label _)            f = f
  ft (Assign x (Lit k))   f = Map.insert x (PElem k) f
  ft (Assign x _)         f = Map.insert x Top f
  ft (Store _ _)          f = f
  ft (Branch l)           f = mapSingleton l f
  ft (Cond (Var x) tl fl) f
      = mkFactBase constLattice
           [(tl, Map.insert x (PElem (Bool True))  f),
            (fl, Map.insert x (PElem (Bool False)) f)]
  ft (Cond _ tl fl) f
      = mkFactBase constLattice [(tl, f), (fl, f)]
  ft (Call vs _ _ bid)      f = mapSingleton bid (foldl toTop f vs)
      where toTop f v = Map.insert v Top f
  ft (Return _)             _ = mapEmpty

```

显著的区别在于，与活跃性分析不同，常量传播分析是一种向前分析 `FwdTransfer`。 这也意味着函数的类型是 `Node e x -> f -> Fact x f`，而不是 `Node e x -> Fact x f -> f`：当控制流分裂时，我们可以为可能的出口标签提供不同的事实集。 这在 `Cond (Var x)` 中得到了很好的应用，我们知道如果我们采取第一个分支，条件变量为真，反之亦然。 其余是管道：

+   `Branch`: 无条件分支不会导致我们的任何变量停止为常量。 Hoopl 将自动注意到，如果到该标签的不同路径具有矛盾的事实，并将映射转换为 `Top` 作为通知，使用我们格的连接函数。 `mapSingleton` 从标签 `l` 到事实 `f` 创建一个单例映射。

+   `Cond`: 我们需要创建一个包含两个条目的映射，可以方便地通过 `mkFactBase` 完成，其中最后一个参数是标签到映射的列表。

+   `Call`: 函数调用相当于将所有返回变量分配给许多未知变量，因此我们用 `toTop` 将它们全部设置为未知。

+   `Return`: 不会前进任何地方，因此空映射就足够了。

下次，我们将讨论一些关于传递函数和连接函数的更精细的细微差别，并讨论图重写，并用一些 Hoopl 的调试工具来总结如何观察 Hoopl 如何重写图。
