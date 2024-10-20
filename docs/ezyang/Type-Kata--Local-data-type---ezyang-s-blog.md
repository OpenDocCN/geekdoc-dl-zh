<!--yml

类别：未分类

日期：2024-07-01 18:17:54

-->

# 类型 Kata：本地数据类型：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/04/type-kata-local-data-type/`](http://blog.ezyang.com/2011/04/type-kata-local-data-type/)

*命令式。* 什么时候应该创建自定义数据类型，而不是重用预先存在的数据类型如`Either`、`Maybe`或元组？以下是重用通用类型的一些理由：

+   它节省了输入（在声明和模式匹配中），使其对于一次性的事务非常有用。

+   它为你提供了一个预定义函数库，用于处理该类型的函数。

+   其他开发者对类型做什么使理解更快的期望。

硬币的另一面：

+   你可能会失去相同但具有不同含义的类型之间的语义区分（[newtype](http://blog.ezyang.com/2010/08/type-kata-newtypes/) 论据），

+   现有的类型可能允许比你打算允许的更多的值，

+   其他开发者对类型的期望可能会导致问题，如果你的意思不同的话。

在本文中，我想谈谈关于重用自定义类型的最后两个问题，使用 GHC 代码库中的两个案例研究，以及如何通过定义仅在代码库的一小部分中使用的数据类型来缓解这些问题。

*大家期待。* `Maybe` 类型本身有一个非常直观的解释：你要么有值，要么没有。即使 `Nothing` 意味着类似 `Wildcard`、`Null` 或 `UseTheDefault`，其含义通常也很明确。

然而，更有趣的是，当 `Maybe` 放置在另一个具有其*自己*无意义概念的数据类型中时。一个非常简单的例子是 `Maybe (Maybe a)`，它允许值 `Nothing`、`Just Nothing` 或 `Just (Just a)`。在这种情况下，`Just Nothing` 意味着什么？在这种情况下，我们真正掩盖的是一个具有三个构造函数的数据类型案例：`data Maybe2 a = Nothing1 | Nothing2 | Just a`。如果我们打算区分 `Nothing` 和 `Just Nothing`，我们需要为它们分配一些不同的语义含义，这些含义不是从拼凑在一起的数据类型中显而易见的。

另一个例子来自[Hoopl 和 GHC](http://blog.ezyang.com/2011/04/hoopl-dataflow-lattices/)，是`Map Var (Maybe Lit)`的奇怪情况。映射已经有了它自己的空值概念：也就是说，当键-值对根本不在映射中时！所以一个遇到这段代码的开发者可能会问的第一个问题是，“为什么不只是`Map Var Lit`呢？”对于那些已经阅读了数据流格点帖子的人来说，这个问题的答案是，在这种情况下，`Nothing`表示 top（变量绝对不是常量），这与映射中的缺失不同，后者表示 bottom（变量是常量或非常量）。我成功地用这段奇怪的代码搞糊涂了两位西蒙斯，经过一些时间解释这种情况后，他们立刻建议我为此制作一个自定义数据类型。更好的是，我发现 Hoopl 已经提供了这种类型：`HasTop`，它还具有一系列反映这组语义的实用函数。真是幸运！

*不速之客.* 我们的一个过多值的数据类型例子来自于 GHC 中的线性寄存器分配器（`compiler/nativeGen/RegAlloc/Linear/Main.hs`）。别担心，你不需要知道如何实现线性寄存器分配器来跟进。

线性寄存器分配器是一个相当庞大且笨拙的家伙。这是实际分配和溢出寄存器的函数：

```
allocateRegsAndSpill reading keep spills alloc (r:rs)
 = do   assig <- getAssigR
        case lookupUFM assig r of
                -- case (1a): already in a register
                Just (InReg my_reg) ->
                        allocateRegsAndSpill reading keep spills (my_reg:alloc) rs

                -- case (1b): already in a register (and memory)
                -- NB1\. if we're writing this register, update its assignment to be
                -- InReg, because the memory value is no longer valid.
                -- NB2\. This is why we must process written registers here, even if they
                -- are also read by the same instruction.
                Just (InBoth my_reg _)
                 -> do  when (not reading) (setAssigR (addToUFM assig r (InReg my_reg)))
                        allocateRegsAndSpill reading keep spills (my_reg:alloc) rs

                -- Not already in a register, so we need to find a free one...
                loc -> allocRegsAndSpill_spill reading keep spills alloc r rs loc assig

```

这里有些噪音，但需要注意的重要事情是，这是一个大部分递归的函数。`lookupUFM`的前两种情况直接调用`allocateRegsAndSpill`，但最后一种情况需要执行一些复杂操作，并调用辅助函数`allocRegsAndSpill_spill`。事实证明，这个函数最终总是会调用`allocateRegsAndSpill`，所以我们有两个[互递归函数](http://en.wikipedia.org/wiki/Mutual_recursion)。

这段代码正在重用一个数据类型！你能看到吗？这非常微妙，因为类型的原始用途是合法的，但随后以不恰当的方式*重复使用*了。答案就是`loc`，在最后的 case 语句中。特别是因为我们已经在`loc`上进行了 case 匹配，我们知道它不可能是`Just (InReg{})`或`Just (InBoth{})`。如果我们查看`Loc`的声明，我们会发现只剩下两种情况：

```
data Loc
        -- | vreg is in a register
        = InReg   !RealReg

        -- | vreg is held in a stack slot
        | InMem   {-# UNPACK #-}  !StackSlot

        -- | vreg is held in both a register and a stack slot
        | InBoth   !RealReg
                   {-# UNPACK #-} !StackSlot
        deriving (Eq, Show, Ord)

```

也就是说，唯一剩下的情况是`Just (InMem{})`和`Nothing`。这相当重要，因为我们稍后在`allocRegsAndSpill_spill`中依赖这一不变量：

```
let new_loc
        -- if the tmp was in a slot, then now its in a reg as well
        | Just (InMem slot) <- loc
        , reading
        = InBoth my_reg slot

        -- tmp has been loaded into a reg
        | otherwise
        = InReg my_reg

```

如果你没有看到 `allocateRegsAndSpill` 中的原始情况分割，这段特定的代码可能会让你想知道最后的保护条件是否也适用于结果是 `Just (InReg{})` 的情况，这种情况下行为会非常错误。实际上，如果我们在 `reading` 时，那个最后的分支中 `loc` 必须是 `Nothing`。但是代码现在无法表达这一点：你必须添加一些紧急情况处理，而且变得非常混乱：

```
let new_loc
        -- if the tmp was in a slot, then now its in a reg as well
        | Just (InMem slot) <- loc
        = if reading then InBoth my_reg slot else InReg my_reg

        -- tmp has been loaded into a reg
        | Nothing <- loc
        = InReg my_reg

        | otherwise = panic "Impossible situation!"

```

此外，我们注意到一个非常有趣的额外不变量：如果我们正在从一个以前从未分配过的位置读取（也就是，`reading` 是 `True` 而 `loc` 是 `Nothing`），会发生什么？这显然是错误的，因此实际上我们应该检查是否出现了这种情况。请注意，原始代码*没有*强制执行这个不变量，这是通过使用 `otherwise` 掩盖了出来。

与其在不可能的情况下恐慌，我们应该*静态地排除这种可能性*。我们可以通过引入一个新的数据类型来实现这一点，并适当地进行模式匹配：

```
-- Why are we performing a spill?
data SpillLoc = ReadMem StackSlot  -- reading from register only in memory
              | WriteNew           -- writing to a new variable
              | WriteMem           -- writing to register only in memory
-- Note that ReadNew is not valid, since you don't want to be reading
-- from an uninitialized register.  We also don't need the location of
-- the register in memory, since that will be invalidated by the write.
-- Technically, we could coalesce WriteNew and WriteMem into a single
-- entry as well. -- EZY

allocateRegsAndSpill reading keep spills alloc (r:rs)
 = do       assig <- getAssigR
        let doSpill = allocRegsAndSpill_spill reading keep spills alloc r rs assig
        case lookupUFM assig r of
                -- case (1a): already in a register
                Just (InReg my_reg) ->
                        allocateRegsAndSpill reading keep spills (my_reg:alloc) rs

                -- case (1b): already in a register (and memory)
                -- NB1\. if we're writing this register, update its assignment to be
                -- InReg, because the memory value is no longer valid.
                -- NB2\. This is why we must process written registers here, even if they
                -- are also read by the same instruction.
                Just (InBoth my_reg _)
                 -> do      when (not reading) (setAssigR (addToUFM assig r (InReg my_reg)))
                        allocateRegsAndSpill reading keep spills (my_reg:alloc) rs

                -- Not already in a register, so we need to find a free one...
                Just (InMem slot) | reading   -> doSpill (ReadMem slot)
                                  | otherwise -> doSpill WriteMem
                Nothing | reading   -> pprPanic "allocateRegsAndSpill: Cannot read from uninitialized register" (ppr r)
                        | otherwise -> doSpill WriteNew

```

现在，在 `allocateRegsAndSpill_spill` 内部的模式匹配变得清晰简洁：

```
-- | Calculate a new location after a register has been loaded.
newLocation :: SpillLoc -> RealReg -> Loc
-- if the tmp was read from a slot, then now its in a reg as well
newLocation (ReadMem slot) my_reg = InBoth my_reg slot
-- writes will always result in only the register being available
newLocation _ my_reg = InReg my_reg

```
