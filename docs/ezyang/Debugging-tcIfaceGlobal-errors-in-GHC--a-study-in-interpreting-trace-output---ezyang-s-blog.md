<!--yml

category: 未分类

date: 2024-07-01 18:17:06

-->

# 调试 GHC 中的 tcIfaceGlobal 错误：解读跟踪输出研究：ezyang 的博客

> 来源：[`blog.ezyang.com/2016/05/debugging-tcifaceglobal-errors-in-ghc-a-study-in-interpreting-trace-output/`](http://blog.ezyang.com/2016/05/debugging-tcifaceglobal-errors-in-ghc-a-study-in-interpreting-trace-output/)

最近我解决了一个 bug，其中 GHC 表现得不够懒惰（是的，*更*多的懒惰是需要的！）我想这可能成为一个很好的博客文章，介绍我如何解决这类懒惰 bug，并可能引发关于如何使调试这类问题对人们更容易的有用讨论。

### 哎呀！一个 bug！

我们的故事始于一个[待处理的补丁](https://phabricator.haskell.org/D2213)，涉及到我之前正在进行的一些相关更改。补丁的内容并不重要——它只是修复了一个 bug，即 `ghc --make` 在具有 `hs-boot` 文件的程序中与 `ghc -c` 没有相同的行为。

在验证对 GHC 测试套件的补丁时，我发现这导致 `prog006` 测试在 GHCi 上开始失败，并显示以下错误：

```
ghc-stage2: panic! (the 'impossible' happened)
  (GHC version 8.1.20160512 for x86_64-unknown-linux):
        tcIfaceGlobal (global): not found
  You are in a maze of twisty little passages, all alike.
  While forcing the thunk for TyThing Data
  which was lazily initialized by initIfaceTcRn,
  I tried to tie the knot, but I couldn't find Data
  in the current type environment.
  If you are developing GHC, please read Note [Tying the knot]
  and Note [Type-checking inside the knot].
  Consider rebuilding GHC with profiling for a better stack trace.
  Contents of current type environment: []

```

`tcIfaceGlobal` 错误是 GHC 如何实现 hs-boot 文件的“黑暗”角落，但因为我过去一周一直在看这部分编译器，所以我决定大胆地前进。

### 如果你的测试案例放不进一张幻灯片，那还不够小

`prog006` 并不是一个简单的测试案例，因为它涉及在 GHCi 会话中运行以下命令：

```
:! cp Boot1.hs Boot.hs
:l Boot.hs
:! sleep 1
:! cp Boot2.hs Boot.hs
:r

```

虽然所涉及的源文件*相对*较短，但我第一个想法仍然是简化测试案例。我最初的想法是，这个 bug 可能涉及到 GHCi 如何重新加载模块的某些方面，因此我的第一个想法是尝试最小化涉及的源代码：

```
-- Boot.hs-boot
module Boot where
data Data

-- A.hs
module A where
import {-# SOURCE #-} Boot
class Class a where
  method :: a -> Data -> a

-- Boot1.hs
module Boot where
data Data

-- Boot2.hs
{-# LANGUAGE ExistentialQuantification #-}
module Boot where
import A
data Data = forall n. Class n => D n

```

这个示例使用了一个花哨的语言特性 `ExistentialQuantification`，如果这些使用与手头的问题无关，通常最好尝试消除它们。因此，我最初的想法是用更普通的东西替换模块 A 中的类型类，例如，一个类型同义词。（注意：为什么不试着消除 `hs-boot`？在这种情况下，我碰巧知道，在编译 `hs-boot` 文件时，`tcIfaceGlobal` 错误*只*会发生。）

我进行了这个转换，得到了以下较小的程序：

```
-- Boot.hs-boot
module Boot
data Data

-- A.hs
module A
import {-# SOURCE #-} Boot
type S = Data

-- Boot.hs
module Boot
import A
x :: S

```

这个程序确实也产生了一个 `tcIfaceGlobal` 错误...但后来我意识到 `Boot.hs` 本身就不是良好类型化的：它缺少了 `Data` 的声明！事实上，当我插入了缺少的声明时，恐慌消失了。

在调试中的一个重要事项是要知道何时意外触发了不同的 bug。事实上，这确实是一个不同的 bug，[我在这里报告了](https://ghc.haskell.org/trac/ghc/ticket/12063)。

在减少这个测试用例的过程中，我发现这个 bug 与 GHCi 无关；例如，如果我只是运行`ghc --make Boot2.hs`，就足以触发这个 bug。（或者，对于一个没有我的补丁的 GHC 版本，在构建其余部分后运行`ghc -c Boot2.hs`，`ghc --make`在引发问题的补丁之前具有不同的行为，这一切都掩盖了问题的本质。）因此，这是最终的测试用例（为了避免一些混乱的消息使用了一些更短的名称）：

```
-- Boot.hs-boot
module Boot where
data D

-- A.hs
module A where
import {-# SOURCE #-} Boot
class K a where
  method :: a -> D -> a

-- Boot.hs
{-# LANGUAGE ExistentialQuantification #-}
module Boot where
import A
data Data = forall n. K n => D n

```

### 当你知道问题所在时，调试就更容易

在调试这样的问题时，了解为什么 bug 会发生是有帮助的。而要有假设，我们必须首先问自己一个问题：`tcIfaceGlobal`到底在做什么？

每当你遇到这样的恐慌时，你应该搜索错误消息并查看周围的源代码。这里是关于`tcIfaceGlobal`的（在一个稍旧版本的 GHC 上，这也表现出了 bug）：

```
; case if_rec_types env of {    -- Note [Tying the knot]
    Just (mod, get_type_env)
        | nameIsLocalOrFrom mod name
        -> do           -- It's defined in the module being compiled
        { type_env <- setLclEnv () get_type_env         -- yuk
        ; case lookupNameEnv type_env name of
                Just thing -> return thing
                Nothing   -> pprPanic "tcIfaceGlobal (local): not found:"
                                        (ppr name $$ ppr type_env) }

  ; _ -> do

```

如果你看到与代码相关联的注释，你绝对应该去找到它并阅读它：

```
-- Note [Tying the knot]
-- ~~~~~~~~~~~~~~~~~~~~~
-- The if_rec_types field is used in two situations:
--
-- a) Compiling M.hs, which indirectly imports Foo.hi, which mentions M.T
--    Then we look up M.T in M's type environment, which is splatted into if_rec_types
--    after we've built M's type envt.
--
-- b) In ghc --make, during the upsweep, we encounter M.hs, whose interface M.hi
--    is up to date.  So we call typecheckIface on M.hi.  This splats M.T into
--    if_rec_types so that the (lazily typechecked) decls see all the other decls
--
-- In case (b) it's important to do the if_rec_types check *before* looking in the HPT
-- Because if M.hs also has M.hs-boot, M.T will *already be* in the HPT, but in its
-- emasculated form (e.g. lacking data constructors).

```

所以情况（a）正是这里正在发生的事情：当我们正在对`Boot.hs`进行类型检查并加载接口`A.hi`时，当我们对`D`的引用进行类型检查时，我们不会去对`Boot.hi-boot`进行类型检查；相反，我们试图与模块中本地定义的`Data`打成一片。如果类型环境中没有`Data`，我们会看到我们之前遇到的恐慌。

使情况复杂的是，并没有显式调用“对`D`的类型检查”；相反，这一堆工作被不安全地封装在表示`D`的`TyThing`的 thunk 中，而这个 thunk 嵌入在对`K`描述中。当我们强制求值这个 thunk 时，GHC 将*然后*忙于尝试对与`D`相关联的类型进行类型检查。

回到我们最初的问题：为什么本地类型环境中没有定义`D`？一般来说，这是因为我们在实际将`D`添加到类型环境之前就强制求值了`K`的 thunk（因此导致调用`tcIfaceGlobal D`）。但为什么会这样呢？有两种可能的解释：

1.  第一个解释是，我们在强制求值 thunk 之前忘记更新类型环境。修复方法是在全局类型环境中添加一些额外的更新，这样当我们强制求值 thunk 时，就能看到缺失的类型。

1.  第二个解释是，我们强制求值 thunk 的时间过早，有些代码需要变得*更懒*，这样我们才能在类型环境已经充分更新时才强制求值 thunk。

所以，问题究竟出在哪里？

### 读茶叶脉络

在这两种情况下，知道我们实际上在类型检查过程中什么时候强制求值 thunk 似乎是有用的。现在是时候重建带有分析工具的 GHC 并获得`tcIfaceGlobal`的堆栈跟踪了，但我有点懒，所以我决定改用 GHC 的跟踪工具。

GHC 具有现有标志 `-ddump-tc-trace`，`-ddump-rn-trace` 和 `-ddump-if-trace`，它们分别倾倒了与类型检查、重命名和接口加载相关的*大量*调试跟踪信息。大多数这些消息非常简洁，不会详细说明消息应该如何解释；如果您想要解释这些消息，您将不得不搜索源代码，看看哪段代码输出了这些跟踪信息。

这是我们在编译 `Boot.hs` 时得到的跟踪的结尾：

```
Tc2 (src)
Tc3
txExtendKindEnv []
txExtendKindEnv []
tcTyAndCl start kind checking ()
kcTyClGroup
  module Boot
    data D = forall n_anU. K n_anU => D
<<some log elided here>>
tc_lhs_type:
  K n_anU
  Constraint
tc_infer_lhs_type: K
lk1 K
Starting fork { Declaration for K
Loading decl for K
updating EPS_
Considering whether to load GHC.Prim {- SYSTEM -}
Reading interface for GHC.Prim;
    reason: Need home interface for wired-in thing TYPE
updating EPS_
tc-iface-class1 K
tc-iface-class2 K
tc-iface-class3 K
tc-iface-class4 K
buildClass
newGlobalBinder A C:K <no location info>
                C:K
newGlobalBinder A $tcK <no location info>
                $tcK
Starting fork { Class op method D -> a
ghc-stage2: panic! (the 'impossible' happened)
<<rest of the panic message>>

```

神奇的是，这个跟踪实际上告诉你*确切地*你需要知道什么来解决这个 bug……但我们得先知道如何解释这个跟踪。

每条跟踪消息，例如 `Tc2 (src)`，`Tc3` 等，都带有一个唯一的字符串，您可以用它来找到跟踪的来源。例如，使用 `Tc2` 进行 grep 会导航到 `TcRnDriver.hs`，就在我们即将开始对源文件中所有声明进行重命名和类型检查的地方。类似地，`lk1` 会导航到 `TcHsType.hs`，在这里我们试图查找与 `K` 关联的 `TyThing`。

`Starting fork` 消息特别值得关注：这是 `-ddump-if-trace` 的方式表达“我正在评估一个带有某些延迟工作类型检查接口的 thunk”。因此，我们可以看到，在跟踪 `lk1` 之后不久，我们强制执行了类型类声明 `K` 的 thunk；此外，在我们强制执行此 thunk 时，我们进一步强制执行了类操作 `method :: D -> a` 的 thunk，这实际上导致了 panic。

### 鲁布·戈尔德堡机器

我没有仔细阅读跟踪，因此在类型检查期间，我花了一些时间手动添加额外的跟踪声明和跟踪代码的流程。从 `Tc2 (src)` 开始，我们实际上可以使用跟踪来跟随类型检查的流程（这里使用 `hasktags` 是必不可少的！）

1.  `tcRnModuleTcRnM` 是重命名和类型检查模块的主要入口点。处理导入后，它调用 `tcRnSrcDecls` 对主体进行类型检查。

1.  `tcRnSrcDecls` 调用 `tc_rn_src_decls` 来对所有顶层声明进行类型检查；然后简化所有顶层约束并完成所有类型。

1.  `tc_rn_src_decls` 是模板 Haskell / 类型检查/重命名舞蹈的主循环。我们首先通过 `rnTopSrcDecls` 进行重命名，然后通过 `tcTopSrcDecls` 进行类型检查，直到第一个 splice，然后运行 splice 并递归。

1.  `tcTopSrcDecls` 输出 `Tc2 (src)`。它逐个检查所有不同类型的顶层声明。其中一个重要的是 `tcTyClsInstDecls`，它对类型和类声明进行类型检查，并处理推导子句。

1.  `tcTyClsInstDecls` 调用 `tcTyAndClassDecls` 对顶层类型和类声明进行类型检查，然后调用 `tcInstDeclsDeriv` 处理推导。

1.  `tcTyAndClassDecls` 处理每个互递类型/类声明组，并在它们上调用 `tcTyClGroup`。

1.  `tcTyClGroup`调用`tcTyClDecls`来对组进行类型检查，然后检查一切是否良好形式。

1.  `tcTyClDecls`实际上是类型检查声明组。它首先用`kcTyClGroup`对组进行种类检查，然后将所有组一起进行类型检查，绑定结节。

1.  `kcTyClGroup`输出（适当命名的）`kcTyClGroup`追踪。在这一点上，我停止了追踪。

通过观察`kcTyClGroup`的追踪，但没有终止的`kcTyClGroup result`追踪（这在函数末尾），我们可以得知在我们进行种类检查时，坏的延迟计算被触发了。

知道恐慌发生在我们进行种类检查时实际上是非常有用的：种类检查发生在我们实际构造这些顶层声明的结节绑定`TyThing`结构之前。所以我们知道，我们没有失败更新全局类型环境，因为它在这一点上肯定没有构建。必须是我们太早强制了一个延迟计算。

### AAAAAAAA 是 GHC 消失在一个黑洞中的声音

此时，我非常确定`lk1`，即`tcTyVar`是导致最终引发恐慌的延迟计算的责任所在，但我并不确定。以下是该函数的代码：

```
tcTyVar :: TcTyMode -> Name -> TcM (TcType, TcKind)
-- See Note [Type checking recursive type and class declarations]
-- in TcTyClsDecls
tcTyVar mode name         -- Could be a tyvar, a tycon, or a datacon
  = do { traceTc "lk1" (ppr name)
       ; thing <- tcLookup name
       ; case thing of
           ATyVar _ tv -> return (mkTyVarTy tv, tyVarKind tv)

           ATcTyCon tc_tc -> do { check_tc tc_tc
                                ; tc <- get_loopy_tc name tc_tc
                                ; handle_tyfams tc tc_tc }
                             -- mkNakedTyConApp: see Note [Type-checking inside the knot]
                 -- NB: we really should check if we're at the kind level
                 -- and if the tycon is promotable if -XNoTypeInType is set.
                 -- But this is a terribly large amount of work! Not worth it.

           AGlobal (ATyCon tc)
             -> do { check_tc tc
                   ; handle_tyfams tc tc }

```

对`K`的`tcTyVar`应该导致`AGlobal (ATyCon tc)`，在该分支上添加一个追踪并没有额外的输出。但我通过添加``thing `seq` traceTc "lk2" (ppr name)``并观察没有出现`lk2`来确定了这件事。

显然在这一点上强制`K`应该对我们来说没问题，因为它是一个外部声明。所以某些东西在延迟计算本身出错了。

### 回到茶叶上

让我们再次看一下追踪的结尾：

```
Starting fork { Declaration for K
Loading decl for K
updating EPS_
Considering whether to load GHC.Prim {- SYSTEM -}
Reading interface for GHC.Prim;
    reason: Need home interface for wired-in thing TYPE
updating EPS_
tc-iface-class1 K
tc-iface-class2 K
tc-iface-class3 K
tc-iface-class4 K
buildClass
newGlobalBinder A C:K <no location info>
                C:K
newGlobalBinder A $tcK <no location info>
                $tcK
Starting fork { Class op method D -> a
ghc-stage2: panic! (the 'impossible' happened)
<<rest of the panic message>>

```

以人类可读的方式来说，这个追踪告诉了一个这样的故事：

1.  有人强制了代表类型类`K`的`TyThing`的延迟计算（`Starting fork { Declaration for K`）

1.  我正在对`K`的`IfaceDecl`的内容进行类型检查（`tc-iface-class`等）

1.  我正在构建代表这个类型类的实际`Class`（`buildClass`）

1.  我为所讨论的类分配了一些全局名称。（`newGlobalBinder`）

1.  糟糕！我强制了代表类操作`method`的延迟计算（其类型为`D -> a`）

1.  不久之后，恐慌发生。

所以，去读`TcIface`的代码。以下是类型检查`IfaceDecl`的代码体：

```
= bindIfaceTyConBinders binders $ \ tyvars binders' -> do
  { tc_name <- lookupIfaceTop tc_occ
  ; traceIf (text "tc-iface-class1" <+> ppr tc_occ)
  ; ctxt <- mapM tc_sc rdr_ctxt
  ; traceIf (text "tc-iface-class2" <+> ppr tc_occ)
  ; sigs <- mapM tc_sig rdr_sigs
  ; fds  <- mapM tc_fd rdr_fds
  ; traceIf (text "tc-iface-class3" <+> ppr tc_occ)
  ; mindef <- traverse (lookupIfaceTop . mkVarOccFS) mindef_occ
  ; cls  <- fixM $ \ cls -> do
            { ats  <- mapM (tc_at cls) rdr_ats
            ; traceIf (text "tc-iface-class4" <+> ppr tc_occ)
            ; buildClass tc_name tyvars roles ctxt binders' fds ats sigs mindef tc_isrec }
  ; return (ATyCon (classTyCon cls)) }

```

类型类的方法在`sigs <- mapM tc_sig rdr_sigs`中处理。看一下这个辅助函数，我们可以看到：

```
tc_sig :: IfaceClassOp -> IfL TcMethInfo
tc_sig (IfaceClassOp occ rdr_ty dm)
  = do { op_name <- lookupIfaceTop occ
       ; ~(op_ty, dm') <- forkM (mk_op_doc op_name rdr_ty) $
                          do { ty <- tcIfaceType rdr_ty
                             ; dm' <- tc_dm dm
                             ; return (ty, dm') }
             -- Must be done lazily for just the same reason as the
             -- type of a data con; to avoid sucking in types that
             -- it mentions unless it's necessary to do so
       ; return (op_name, op_ty, dm') }

```

太好了！已经有一些代码提到了签名类型需要懒惰地完成。如果我们强制`op_ty`或`dm'`，我们将导致这里的类型被加载。现在我们只需要找到在`buildClass`中它们被强制的地方。以下是`buildClass`的头部：

```
buildClass tycon_name tvs roles sc_theta binders
           fds at_items sig_stuff mindef tc_isrec

```

所以让我们来看看`sig_stuff`的出现。它们第一次被使用的地方是：

```
; op_items <- mapM (mk_op_item rec_clas) sig_stuff
                -- Build the selector id and default method id

```

让我们看看这个辅助函数：

```
mk_op_item :: Class -> TcMethInfo -> TcRnIf n m ClassOpItem
mk_op_item rec_clas (op_name, _, dm_spec)
  = do { dm_info <- case dm_spec of
                      Nothing   -> return Nothing
                      Just spec -> do { dm_name <- newImplicitBinder op_name mkDefaultMethodOcc
                                      ; return (Just (dm_name, spec)) }
       ; return (mkDictSelId op_name rec_clas, dm_info) }

```

在这里！`dm_spec`上的这个案例将迫使`dm'`，进而导致类型被强制，结果引发了恐慌。这肯定不对。

看起来`mk_op_item`只关心`dm_spec`上的顶层包装；`dm_info`内部懒惰地使用`spec`，并且似乎在`mkClass`后期不会被强制执行。因此修复的方法将是使得我们可以在不强制`dm`内容的情况下剥离外部的`Maybe`：

```
--- a/compiler/iface/TcIface.hs
+++ b/compiler/iface/TcIface.hs
@@ -429,20 +429,23 @@ tc_iface_decl _parent ignore_prags
    tc_sig :: IfaceClassOp -> IfL TcMethInfo
    tc_sig (IfaceClassOp occ rdr_ty dm)
      = do { op_name <- lookupIfaceTop occ
-          ; ~(op_ty, dm') <- forkM (mk_op_doc op_name rdr_ty) $
-                             do { ty <- tcIfaceType rdr_ty
-                                ; dm' <- tc_dm dm
-                                ; return (ty, dm') }
+          ; let doc = mk_op_doc op_name rdr_ty
+          ; op_ty <- forkM (doc <+> text "ty") $ tcIfaceType rdr_ty
                 -- Must be done lazily for just the same reason as the
                 -- type of a data con; to avoid sucking in types that
                 -- it mentions unless it's necessary to do so
+          ; dm'   <- tc_dm doc dm
           ; return (op_name, op_ty, dm') }

-   tc_dm :: Maybe (DefMethSpec IfaceType) -> IfL (Maybe (DefMethSpec Type))
-   tc_dm Nothing               = return Nothing
-   tc_dm (Just VanillaDM)      = return (Just VanillaDM)
-   tc_dm (Just (GenericDM ty)) = do { ty' <- tcIfaceType ty
-                                    ; return (Just (GenericDM ty')) }
+   tc_dm :: SDoc
+         -> Maybe (DefMethSpec IfaceType)
+         -> IfL (Maybe (DefMethSpec Type))
+   tc_dm _   Nothing               = return Nothing
+   tc_dm _   (Just VanillaDM)      = return (Just VanillaDM)
+   tc_dm doc (Just (GenericDM ty))
+        = do { -- Must be done lazily to avoid sucking in types
+             ; ty' <- forkM (doc <+> text "dm") $ tcIfaceType ty
+             ; return (Just (GenericDM ty')) }

```

我们检查了修复，是的！它奏效了！

### 分手的酒杯

我不会声称我的调试过程是可能的最有效过程——在这篇博文中没有提到的是我花了一天时间阅读提交历史，试图说服自己我们并没有忘记更新全局类型环境中的错误。但是这里似乎有一些可推广的经验教训：

1.  如果你看到一些跟踪输出，使跟踪对你最有用的方法是确定*代码中*跟踪消息来自何处，并且在那个时间点编译器正在做什么。通常，使用 grep 搜索跟踪消息就足以弄清楚这一点。

1.  你的测试案例越小，你的跟踪就会越小，这样解释跟踪就更容易。当我运行我的测试案例时使用`ghc --make`而不是`ghc -c`时，输出的日志要多得多。确实，结束的跟踪是一样的，但如果在早期跟踪中有重要内容，那么挖掘出来就更加困难。

1.  如果你可以信任你的跟踪，调试就会更容易。如果我相信跟踪输出，我本可以更快地找到错误。但我没有，而是花了大量时间确保代码表现出我期望的行为。好的一面是，我现在对这里的代码路径了解得比以前深多了。

GHC 如何使调试这些类型的错误更容易？有自己的与惰性相关的调试故事吗？我很想知道你的想法。
