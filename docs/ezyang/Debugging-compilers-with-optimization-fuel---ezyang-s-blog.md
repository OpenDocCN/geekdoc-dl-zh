<!--yml

category: 未分类

date: 2024-07-01 18:17:44

-->

# 使用优化燃料调试编译器：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/06/debugging-compilers-with-optimization-fuel/`](http://blog.ezyang.com/2011/06/debugging-compilers-with-optimization-fuel/)

今天我想描述一下我如何精确定位编译器错误，具体来说，是被优化触发的错误，使用了一个叫做*优化燃料*的巧妙功能，这个功能是由 Hoopl 引入的。不幸的是，这不是一个特别容易在 Google 上找到的术语，所以希望这篇文章也能帮助到一些人。优化燃料最初是由 David Whalley 在 1994 年的一篇论文*自动隔离编译器错误*中提出的。基本思想是编译器执行的所有优化可以被限制（例如通过限制燃料），所以当我们怀疑优化器行为异常时，我们进行二分搜索，找到在引入错误之前能够给予编译器的最大燃料量。然后我们可以检查有问题的优化并修复错误。优化燃料是新代码生成器的一个特性，只有当你向 GHC 传递`-fuse-new-codegen`参数时才可用。

### 缺陷

当我尝试使用新代码生成器构建 GHC 本身时，bug 就出现了。构建 GHC 是发现 bug 的一个好方法，因为它有这么多代码，它成功覆盖了很多情况：

```
"inplace/bin/ghc-stage1" (...) -o compiler/stage2/build/FastString.o
ghc-stage1: panic! (the 'impossible' happened)
  (GHC version 7.1 for i386-unknown-linux):
       RegAllocLinear.makeRegMovementGraph

Please report this as a GHC bug:  http://www.haskell.org/ghc/reportabug

```

我们迅速地在代码库中使用 grep 命令来找到相关错误，它位于`compiler/nativeGen/RegAlloc/Linear/JoinToTargets.hs`文件中：

```
-- | Construct a graph of register\/spill movements.
--
--  Cyclic components seem to occur only very rarely.
--
--  We cut some corners by not handling memory-to-memory moves.
--  This shouldn't happen because every temporary gets its own stack slot.
--
makeRegMovementGraph :: RegMap Loc -> RegMap Loc -> [(Unique, Loc, [Loc])]
makeRegMovementGraph adjusted_assig dest_assig
 = let
        mkNodes src vreg
         = expandNode vreg src
         $ lookupWithDefaultUFM_Directly
                dest_assig
                (panic "RegAllocLinear.makeRegMovementGraph")
                vreg

   in       [ node  | (vreg, src) <- ufmToList adjusted_assig
                , node <- mkNodes src vreg ]

```

但是源代码并没有特别指出问题可能在哪里。现在是开始使用优化燃料的时候了！

### 二分搜索

我们可以通过改变`-dopt-fuel`的值来修改 GHC 用于运行优化的优化燃料数量。如果我们发现 bug 在没有优化燃料的情况下出现，我们首先要做的是：

```
$ "inplace/bin/ghc-stage1" (...) -o compiler/stage2/build/FastString.o -dopt-fuel=0

```

太棒了，成功了！我们选择一个较大的数字作为我们二分搜索的起点（并传递`-fforce-recomp`，这样 GHC 实际上会编译程序）。

```
$ "inplace/bin/ghc-stage1" (...) -o compiler/stage2/build/FastString.o -dopt-fuel=1000 -fforce-recomp
ghc-stage1: panic! (the 'impossible' happened)
  (GHC version 7.1 for i386-unknown-linux):
       RegAllocLinear.makeRegMovementGraph

Please report this as a GHC bug:  http://www.haskell.org/ghc/reportabug

```

然后我进行二分搜索（测试 500，如果失败则测试 750 等），直到找到添加一个燃料单元导致失败的点。

```
$ "inplace/bin/ghc-stage1" (...) -o compiler/stage2/build/FastString.o -dopt-fuel=709 -fforce-recomp
$ "inplace/bin/ghc-stage1" (...) -o compiler/stage2/build/FastString.o -dopt-fuel=710 -fforce-recomp
ghc-stage1: panic! (the 'impossible' happened)
  (GHC version 7.1 for i386-unknown-linux):
       RegAllocLinear.makeRegMovementGraph

```

### 查看罪魁祸首

如何说服 GHC 告诉我们它在第 710 个燃料单位时做了什么优化呢？我最喜欢的方法是从两次运行中输出优化后的 C--代码，然后进行比较。我们可以使用`-ddump-opt-cmm -ddump-to-file`将 C--代码输出到文件，然后进行比较：

```
@@ -10059,7 +10059,6 @@
         }
     c45T:
         _s3es::I32 = I32[Sp + 4];
-        _s3eu::I32 = I32[Sp + 0];
         // deleted: if (0) goto c460;
         // outOfLine should follow:
         _s3er::I32 = 0;
@@ -10093,1354 +10092,3 @@
         jump (I32[Sp + 0]) ();
 }

```

优化正在删除一个赋值。这有效吗？这是完整的代码，带有一些注释：

```
FastString.$whashStr_entry()
        { [const 131081;, const 0;, const 15;]
        }
    c45T:
        _s3es::I32 = I32[Sp + 4];
        _s3eu::I32 = I32[Sp + 0]; // deleted assignment
        _s3er::I32 = 0;
        _s3ex::I32 = 0;
        goto c463;
    c460:
        R1 = FastString.$whashStr_closure;
        jump (I32[BaseReg - 4]) ();
    c463:
        if (I32[GHC.Types.Bool_closure_tbl + ((_s3er::I32 == _s3es::I32) << 2)] & 3 >= 2) goto c46d;
        // uh oh, assignment used here
        _s3IC::I32 = %MO_S_Rem_W32(%MO_UU_Conv_W8_W32(I8[_s3eu::I32 + (_s3er::I32 << 0)]) + _s3ex::I32 * 128,
                                   4091);
        _s3er::I32 = _s3er::I32 + 1;
        _s3ex::I32 = _s3IC::I32;
        goto c463;
    c46d:
        R1 = _s3ex::I32;
        Sp = Sp + 8;
        jump (I32[Sp + 0]) ();
}

```

似乎不是：变量在`MO_S_Rem_W32`中被使用：这不好。我们得出结论，bug 在一个优化过程中，并且不是寄存器分配器未能处理我们的优化现在正在触发的情况。

### 修复 bug

有了这些信息，我们还可以提取导致此 bug 的程序片段：

```
hashStr  :: Ptr Word8 -> Int -> Int
hashStr (Ptr a#) (I# len#) = loop 0# 0#
   where
    loop h n | n GHC.Exts.==# len# = I# h
             | otherwise  = loop h2 (n GHC.Exts.+# 1#)
          where !c = ord# (indexCharOffAddr# a# n)
                !h2 = (c GHC.Exts.+# (h GHC.Exts.*# 128#)) `remInt#` 4091#

```

我们还可以看到我们的流水线如何处理程序，并准确观察在过程中坏优化发生的确切位置：

```
==================== Post Proc Points Added ====================
{offset
  c43r:
      _s3es::I32 = I32[(old + 8)];
      _s3eu::I32 = I32[(old + 12)];
      if (Sp - <highSp> < SpLim) goto c43y; else goto c43u;

==================== Post spills and reloads ====================
{offset
  c43r:
      _s3es::I32 = I32[(old + 8)];
      _s3eu::I32 = I32[(old + 12)];
      if (Sp - <highSp> < SpLim) goto c43y; else goto c43u;

==================== Post rewrite assignments ====================
{offset
  c43r:
      _s3es::I32 = I32[(old + 8)];
      if (Sp - <highSp> < SpLim) goto c43y; else goto c43u;

```

由于这是代码移除的一个虚假实例，我们在重写赋值优化步骤中寻找所有对`emptyGraph`的提及：

```
usageRewrite :: BwdRewrite FuelUniqSM (WithRegUsage CmmNode) UsageMap
usageRewrite = mkBRewrite3 first middle last
    where first  _ _ = return Nothing
          middle :: Monad m => WithRegUsage CmmNode O O -> UsageMap -> m (Maybe (Graph (WithRegUsage CmmNode) O O))
          middle (Plain (CmmAssign (CmmLocal l) e)) f
                     = return . Just
                     $ case lookupUFM f l of
                            Nothing    -> emptyGraph
                            Just usage -> mkMiddle (AssignLocal l e usage)
          middle _ _ = return Nothing
          last   _ _ = return Nothing

```

看起来这应该是无可非议的死赋值消除案例，结合存活性分析，但出于某种原因，向后事实未能正确传播。事实上，问题在于我试图优化 Hoopl 数据流函数，结果搞错了。（不动点分析很棘手！）在恢复我的更改后，不合理的优化问题消失了。*呼~*
