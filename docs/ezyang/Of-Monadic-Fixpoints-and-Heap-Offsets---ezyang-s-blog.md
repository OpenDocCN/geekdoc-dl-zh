<!--yml

类别：未分类

日期：2024-07-01 18:17:17

-->

# Of Monadic Fixpoints and Heap Offsets : ezyang’s blog

> 来源：[`blog.ezyang.com/2013/09/of-monadic-fixpoints-and-heap-offsets/`](http://blog.ezyang.com/2013/09/of-monadic-fixpoints-and-heap-offsets/)

在 ICFP，有时所谓的“走廊轨道”有时比普通轨道还要重要。Johan Tibell 希望在 GHC 中避免对 `allocate` 函数的非行内调用，当静态已知大小的小数组被分配时。但他发现 GHC 的新代码生成器处理堆分配的方式有点令人困惑，所以今天我们中有人放弃了一场会议来解决这个问题。在这篇文章中，我想解释一下代码生成单子如何通过一种有趣（同时也有点烦人）的技巧来计算代码中的堆偏移量，这涉及到一个“单子”修复点的方式。

首先，关于代码生成器的一些背景。 GHC 需要为其生成代码的函数的大致模式如下：

1.  检查是否有足够的堆空间，如果没有则进行垃圾回收，

1.  将一堆数据写入堆，

1.  将一些东西推到栈上，

1.  跳转到适当的继续。

具体来说，代码将是这样的：

```
c2EP:
    _s28e::P64 = R2;
    // Check if there is enough heap space
    Hp = Hp + 40;
    if (Hp > HpLim) goto c2ET; else goto c2ES;
c2ET:
    // If not enough space, GC
    HpAlloc = 40;
    R2 = _s28e::P64;
    R1 = withEmpty_riC_static_closure;
    call (stg_gc_fun)(R2, R1) args: 8, res: 0, upd: 8;
c2ES:
    // Write a bunch of data to the heap
    I64[Hp - 32] = sat_s28f_info;
    _c2EG::P64 = Hp - 32;
    I64[Hp - 16] = :_con_info;
    P64[Hp - 8] = _c2EG::P64;
    P64[Hp] = _s28e::P64;
    _c2EO::P64 = Hp - 14;
    R1 = _c2EO::P64;
    // No stack updates this time
    // Jump to the continuation
    call (P64[Sp])(R1) args: 8, res: 0, upd: 8;

```

这看起来是合理的，但如何实际生成这段代码呢？代码是按顺序生成的，但在完成布局其余代码之前，我们并不知道需要检查多少堆。如果我们戴上突变帽子，我们可能会说：“好吧，暂时略过它，等你知道实际值时再进行突变”，但是仍然有一个棘手的问题，即当我们向堆写入值时偏移量应该是多少。请注意，在上面的代码中，我们只增加了堆指针一次；如果我们反复增加堆指针，那么偏移量就很容易计算，但我们会浪费指令；x86 寻址模式支持直接将写入寄存器加上一些偏移量。

让我们看看当 GHC 将动态闭包分配到堆时的操作（简化版）：

```
allocDynClosureCmm info_tbl args_offsets
  = do  { virt_hp <- getVirtHp
        ; let rep = cit_rep info_tbl -- cit = c info table
              info_offset = virt_hp + 1 -- virtual heap offset of first word of new object
              info_ptr = CmmLit (CmmLabel (cit_lbl info_tbl))
        ; base <- getHpRelOffset (virt_hp + 1)
        ; emitSetDynHdr base info_ptr
        ; let (args, offsets) = unzip args_offsets
        ; hpStore base args offsets
        ; setVirtHp (virt_hp + heapClosureSize rep)
        ; getHpRelOffset info_offset
        }

```

换句话说，它：

1.  检索一个“虚拟堆指针”（稍后详细介绍），

1.  使用虚拟堆指针 (`getHpRelOffset`，注意偏差一个单位) 获取真正的 `Hp - n` 表达式 (`base`)，

1.  发出一系列写入到 `base` 内存的操作（`emitSetDynHdr` 和 `hpStore`），

1.  将虚拟 Hp 的位置上升到刚刚分配的闭包的大小，

1.  返回 `Hp - n` 表达式。

正如事实证明的那样，虚拟堆指针只是代码生成单子 `FCode` 中的普通状态变量（查看您正在使用的单子的实现是件好事！）：

```
newtype FCode a = FCode (CgInfoDownwards -> CgState -> (# a, CgState #))

data CgState
  = MkCgState { ...
     cgs_hp_usg  :: HeapUsage,
     ... }

data HeapUsage =
  HeapUsage {
        virtHp :: VirtualHpOffset, -- Virtual offset of highest-allocated word
                                   --   Incremented whenever we allocate
        realHp :: VirtualHpOffset  -- realHp: Virtual offset of real heap ptr
                                   --   Used in instruction addressing modes
  }

```

因此，`virtHp` 只需在我们分配东西时向上移动；实际上，它是我们低效的重新增加实现中 `Hp` 寄存器的内容。

这留给我们一个紧迫的问题，`realHp` 是什么？嗯，它最初是零（因为真实堆指针的偏移量只是零），但一旦我们推动堆指针进行栈检查，它现在*恰好是我们进行堆检查的堆量*。回顾我们的例子：

```
c2EP:
    _s28e::P64 = R2;
    // Check if there is enough heap space
    // virtHp = 0; realHp = 0
    Hp = Hp + 40;
    // virtHp = 0; realHp = 40
    if (Hp > HpLim) goto c2ET; else goto c2ES;
c2ET:
    // If not enough space, GC
    HpAlloc = 40;
    R2 = _s28e::P64;
    R1 = withEmpty_riC_static_closure;
    call (stg_gc_fun)(R2, R1) args: 8, res: 0, upd: 8;
c2ES:
    // Write a bunch of data to the heap
    // First closure
    // virtHp = 0; realHp = 40
    I64[Hp - 32] = sat_s28f_info;
    _c2EG::P64 = Hp - 32;
    // virtHp = 8; realHp = 40
    I64[Hp - 16] = :_con_info;
    P64[Hp - 8] = _c2EG::P64;
    P64[Hp] = _s28e::P64;
    _c2EO::P64 = Hp - 14;
    // virtHp = 32; realHp = 40
    R1 = _c2EO::P64;
    // No stack updates this time
    // Jump to the continuation
    call (P64[Sp])(R1) args: 8, res: 0, upd: 8;

```

（实际上，内部偏移量记录为单词，所以在这个 64 位代码中，一切都要除以八。顺便说一句，virtHp + 8 == realHp，这就是偏差为一的原因。）数学有点复杂，但 `getHpRelOffset` 会为你计算偏移量；你只需确保虚拟偏移量正确即可！

好的，但我们仍然没有弄清楚最初这个神奇数字 40 是从哪里来的！关键是要看负责堆检查的代码生成器 `heapCheck`，它包裹了对 `code` 的调用，后者实际上负责代码生成：

```
heapCheck :: Bool -> Bool -> CmmAGraph -> FCode a -> FCode a
heapCheck checkStack checkYield do_gc code
  = getHeapUsage $ \ hpHw ->

```

嘿，那个神奇的 `getHeapUsage` 函数是什么？

```
-- 'getHeapUsage' applies a function to the amount of heap that it uses.
-- It initialises the heap usage to zeros, and passes on an unchanged
-- heap usage.
--
-- It is usually a prelude to performing a GC check, so everything must
-- be in a tidy and consistent state.
--
-- Note the slightly subtle fixed point behaviour needed here

getHeapUsage :: (VirtualHpOffset -> FCode a) -> FCode a
getHeapUsage fcode
  = do  { info_down <- getInfoDown
        ; state <- getState
        ; let   fstate_in = state { cgs_hp_usg  = initHpUsage }
                (r, fstate_out) = doFCode (fcode hp_hw) info_down fstate_in
                hp_hw = heapHWM (cgs_hp_usg fstate_out)        -- Loop here!

        ; setState $ fstate_out { cgs_hp_usg = cgs_hp_usg state }
        ; return r }

```

在这里，我们看到了单子的不动点。为了将堆使用情况提供给 `fcode`，GHC 为自己编写了一个检查：`hp_hw`。检查借鉴了生成 `fcode` 的*结果*，并附加的字符串是：“只要在生成代码之前不兑现这个检查，一切都会没问题！”（在某种程度上有点像一个大银行。）可爱—我们只需要进行一次代码生成！

这种技术并非没有其阴暗面。`hp_hw` 是危险的；如果你在错误的地方强制它，你将陷入无限循环。这个变量有两个用途，都在 `compiler/codeGen/StgCmmLayout.hs` 中，它们都小心地不强制使用它。如果能够显式地将 `hp_hw` 标记为黑洞，并附加自定义错误消息，以便在无限循环发生时发出，那将是非常好的。如何实现这一点留给读者作为练习。

* * *

顺便提一句，在你还不知道的情况下，我一直在实时转发 ICFP 的报道，链接在[这里](http://ezyang.tumblr.com) — 虽然报道并非百分之百完整，编辑也比较草率，但你可以看看！
