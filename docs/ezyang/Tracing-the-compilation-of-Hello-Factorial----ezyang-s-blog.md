<!--yml

category: 未分类

date: 2024-07-01 18:17:54

-->

# 追踪编译 Hello Factorial! : ezyang’s 博客

> 来源：[`blog.ezyang.com/2011/04/tracing-the-compilation-of-hello-factorial/`](http://blog.ezyang.com/2011/04/tracing-the-compilation-of-hello-factorial/)

在函数式编程语言世界中，*阶乘函数*常被称为函数式编程语言中的“Hello World!”。确实，阶乘是测试模式匹配和递归功能的一种非常有用的方式：我们不用操心像输入输出这样“琐碎”的问题。在本博文中，我们将追踪阶乘函数在 GHC 的编译过程中的详细步骤。您将学习如何阅读 Core、STG 和 Cmm，希望您能体验一下编译函数式程序的过程。想要参与 GHC 源码的朋友可以查看[GHC wiki 上一个模块编译的描述。](http://hackage.haskell.org/trac/ghc/wiki/Commentary/Compiler/HscMain)为了保持简单，我们不会进行优化编译；或许优化后的阶乘函数会成为另一篇文章的主题！

本文示例使用 GHC 6.12.1 在一个 32 位 Linux 机器上编译。

### Haskell

```
$ cat Factorial.hs

```

我们从 Haskell 这个温暖舒适的国度开始：

```
module Factorial where

fact :: Int -> Int
fact 0 = 1
fact n = n * fact (n - 1)

```

为了保持代码简单，我们不再检查输入是否为负数，并且还将此函数特化为 `Int`，以便最终生成的代码更清晰。但除此之外，这就是标准的阶乘函数。将其放入名为 `Factorial.hs` 的文件中，您就可以开始体验了。

### Core

```
$ ghc -c Factorial.hs -ddump-ds

```

Haskell 是一种大而复杂的语言，具有许多特性。这对于编码来说很重要，但对于机器处理来说就不那么好了。因此，一旦我们完成了大多数用户可见的错误处理（如类型检查等），我们将 Haskell 转换成一个称为 Core 的小语言。在这一点上，程序仍然是函数式的，但比我们最初写的要冗长一些。

我们首先看到我们阶乘函数的 Core 版本：

```
Rec {
Factorial.fact :: GHC.Types.Int -> GHC.Types.Int
LclIdX
[]
Factorial.fact =
  \ (ds_dgr :: GHC.Types.Int) ->
    let {
      n_ade :: GHC.Types.Int
      LclId
      []
      n_ade = ds_dgr } in
    let {
      fail_dgt :: GHC.Prim.State# GHC.Prim.RealWorld -> GHC.Types.Int
      LclId
      []
      fail_dgt =
        \ (ds_dgu :: GHC.Prim.State# GHC.Prim.RealWorld) ->
          *_agj n_ade (Factorial.fact (-_agi n_ade (GHC.Types.I# 1))) } in
    case ds_dgr of wild_B1 { GHC.Types.I# ds_dgs ->
    letrec { } in
    case ds_dgs of ds_dgs {
      __DEFAULT -> fail_dgt GHC.Prim.realWorld#; 0 -> GHC.Types.I# 1
    }
    }

```

这可能看起来有点陌生，因此这里是将 Core 重新编写成更像 Haskell 的形式。特别是我省略了绑定器信息（类型签名、`LclId` 和 `[]`，这些都在每个绑定之前），删除了一些类型签名并重新缩进：

```
Factorial.fact =
    \ds_dgr ->
        let n_ade = ds_dgr in
        let fail_dgt = \ds_dgu -> n_ade * Factorial.fact (n_ade - (GHC.Int.I# 1)) in
        case ds_dgr of wild_B1 { I# ds_dgs ->
            case ds_dgs of ds_dgs {
                __DEFAULT -> fail_dgt GHC.Prim.realWorld#
                0 -> GHC.Int.I# 1
            }
        }

```

这仍然是一段有趣的代码，让我们来逐步分析一下它。

+   不再有 `fact n = ...` 风格的绑定：一切都被转换成 lambda。我们引入了匿名变量，前缀为 `ds_` 用于此目的。

+   第一个 let 绑定是为了确保我们的变量 `n`（在末尾附加了一些额外的东西，以防我们定义了另一个遮盖原始绑定的 `n`）确实与 `ds_dgr` 相同。它很快会被优化掉。

+   我们对 `fact` 的递归调用已神秘地放置在一个名为 `fail_dgt` 的 lambda 中。这是什么意思呢？这是我们正在做的模式匹配的产物：如果我们所有的其他匹配失败（我们只有一个零的情况），我们调用 `fail_dgt`。它接受的值是一个伪 token `GHC.Prim.realWorld#`，你可以把它看作是单位。

+   我们看到我们的模式匹配已经被解糖成了对 `ds_dgr` 的 *unboxed* 值 `ds_dgs` 的 `case` 语句。我们做一个情况切换来解箱它，然后再做另一个情况切换来进行模式匹配。与 `case` 语句附加的一个额外的语法是 `of` 关键字右边的一个变量，它指示评估后的值（在这种特殊情况下，没有人使用它）。

+   最后，我们看到我们递归的每一个分支，我们看到我们必须手动构造一个装箱的整数 `GHC.Int.I# 1` 作为字面量。

然后我们看到一堆额外的变量和函数，它们表示我们从 Prelude 隐式使用的函数和值，比如乘法、减法和相等性：

```
$dNum_agq :: GHC.Num.Num GHC.Types.Int
LclId
[]
$dNum_agq = $dNum_agl
*_agj :: GHC.Types.Int -> GHC.Types.Int -> GHC.Types.Int
LclId
[]
*_agj = GHC.Num.* @ GHC.Types.Int $dNum_agq
-_agi :: GHC.Types.Int -> GHC.Types.Int -> GHC.Types.Int
LclId
[]
-_agi = GHC.Num.- @ GHC.Types.Int $dNum_agl
$dNum_agl :: GHC.Num.Num GHC.Types.Int
LclId
[]
$dNum_agl = GHC.Num.$fNumInt
$dEq_agk :: GHC.Classes.Eq GHC.Types.Int
LclId
[]
$dEq_agk = GHC.Num.$p1Num @ GHC.Types.Int $dNum_agl
==_adA :: GHC.Types.Int -> GHC.Types.Int -> GHC.Bool.Bool
LclId
[]
==_adA = GHC.Classes.== @ GHC.Types.Int $dEq_agk
fact_ado :: GHC.Types.Int -> GHC.Types.Int
LclId
[]
fact_ado = Factorial.fact
end Rec }

```

因为 `+`、`*` 和 `==` 是从类型类来的，我们必须为每种类型 `dNum_agq` 和 `dEq_agk` 查找字典，然后使用它们来获取我们实际的函数 `*_agj`、`-_agi` 和 `==_adA`，这是我们的 Core 引用的内容，*不是* 完全通用的版本。如果我们没有提供 `Int -> Int` 类型签名，这将有所不同。

### 简化的 Core

```
ghc -c Factorial.hs -ddump-simpl

```

从这里开始，我们对核心进行了多次优化。敏锐的读者可能已经注意到，在 `n = 0` 时，未优化的 Core 分配了一个不必要的 thunk，即 `fail_dgt`。这种低效性，以及其他因素，都被优化掉了：

```
Rec {
Factorial.fact :: GHC.Types.Int -> GHC.Types.Int
GblId
[Arity 1]
Factorial.fact =
  \ (ds_dgr :: GHC.Types.Int) ->
    case ds_dgr of wild_B1 { GHC.Types.I# ds1_dgs ->
    case ds1_dgs of _ {
      __DEFAULT ->
        GHC.Num.*
          @ GHC.Types.Int
          GHC.Num.$fNumInt
          wild_B1
          (Factorial.fact
             (GHC.Num.-
                @ GHC.Types.Int GHC.Num.$fNumInt wild_B1 (GHC.Types.I# 1)));
      0 -> GHC.Types.I# 1
    }
    }
end Rec }

```

现在，我们进入时的第一件事是对输入 `ds_dgr` 进行拆箱并对其进行模式匹配。所有的字典混乱已经内联到 `__DEFAULT` 分支中，因此 `GHC.Num.* @ GHC.Types.Int GHC.Num.$fNumInt` 对应于 `Int` 的乘法，而 `GHC.Num.- @ GHC.Types.Int GHC.Num.$fNumInt` 对应于 `Int` 的减法。由于我们可以直接对 unboxed 的 `Int` 进行模式匹配，所以找不到相等性。

关于装箱（boxing）和拆箱（unboxing）有几点需要说明。一个重要的事情要注意的是，`ds_dgr` 上的 `case` 语句迫使这个变量：它可能是一个 thunk，因此在我们进一步进行之前可能会运行一些（潜在的大量）代码。这也是为什么在 Haskell 中获取回溯（backtraces）如此困难的原因之一：我们关心的是 `ds_dgr` 的 thunk 分配位置，而不是它被评估的位置！但是在我们评估它之前，我们不知道它会出错。

另一个重要的事情要注意的是，尽管我们将整数解包，结果 `ds1_dgs` 并未用于除了模式匹配之外的任何事情。事实上，每当我们使用 `n` 时，我们都会使用 `wild_B1`，它对应于 `ds_dgr` 的完全求值版本。这是因为所有这些函数都期望 *装箱* 的参数，而由于我们已经有了整数的装箱版本，重新装箱未装箱版本就没有意义。

### STG

```
ghc -c Factorial.hs -ddump-stg

```

现在我们将 Core 转换为无脊柱、无标签的 G 机器，在生成更像传统命令式程序的代码之前的最后表示。

```
Factorial.fact =
    \r srt:(0,*bitmap*) [ds_sgx]
        case ds_sgx of wild_sgC {
          GHC.Types.I# ds1_sgA ->
              case ds1_sgA of ds2_sgG {
                __DEFAULT ->
                    let {
                      sat_sgJ =
                          \u srt:(0,*bitmap*) []
                              let {
                                sat_sgI =
                                    \u srt:(0,*bitmap*) []
                                        let { sat_sgH = NO_CCS GHC.Types.I#! [1];
                                        } in  GHC.Num.- GHC.Num.$fNumInt wild_sgC sat_sgH;
                              } in  Factorial.fact sat_sgI;
                    } in  GHC.Num.* GHC.Num.$fNumInt wild_sgC sat_sgJ;
                0 -> GHC.Types.I# [1];
              };
        };
SRT(Factorial.fact): [GHC.Num.$fNumInt, Factorial.fact]

```

结构上，STG 与 Core 非常相似，尽管在为代码生成阶段准备的时候有很多额外的杂项：

+   所有变量都已重命名，

+   现在所有的 lambda 表达式都具有形式 `\r srt:(0,*bitmap*) [ds_sgx]`。参数位于最右边的列表中：如果没有参数，则只是一个惰性求值。反斜杠后的第一个字符指示闭包是否可重入（r）、可更新（u）或单入口（s，在本例中未使用）。可更新的闭包在求值后可以重写为其结果（因此带有参数的闭包不能是可更新的！）然后显示[静态引用表](http://hackage.haskell.org/trac/ghc/wiki/Commentary/Rts/CAFs)，尽管在我们的程序中没有有趣的静态引用。

+   `NO_CCS` 是一个用于性能分析的注释，表示此闭包未附加任何成本中心堆栈。由于我们没有使用性能分析进行编译，这并不是很有趣。

+   构造函数应用使用方括号来接收它们的参数：`GHC.Types.I# [1]`。这不仅是风格上的变化：在 STG 中，构造函数需要 *所有* 的参数（例如，它们是饱和的）。否则，构造函数将被转换为一个 lambda 表达式。

还有一个有趣的结构变化，现在所有的函数应用现在只接受变量作为参数。特别是，我们已经创建了一个新的 `sat_sgJ` 惰性求值，传递给 `factorial` 的递归调用。因为我们没有使用优化编译，GHC 没有注意到 `fact` 的参数将立即被求值。这将导致一些极其迂回的中间代码！

### Cmm

```
ghc -c Factorial.hs -ddump-cmm

```

Cmm（读作“C 减减”）是 GHC 的高级汇编语言。它在范围上类似于 LLVM，尽管看起来更像 C 而不是汇编语言。在这里，输出开始变得很大，因此我们将它分块处理。Cmm 输出包含许多数据部分，主要编码自 STG 中的额外注释信息和入口点：`sgI_entry`、`sgJ_entry`、`sgC_ret` 和 `Factorial_fact_entry`。还有两个额外的函数 `__stginit_Factorial_` 和 `__stginit_Factorial`，用于初始化模块，我们不会详细讨论。

因为我们一直在查看 `STG`，所以我们可以在这些入口点和 `STG` 中的名称之间建立直接的对应关系。简言之：

+   `sgI_entry` 对应于从 `wild_sgC` 减去 `1` 的 thunk。我们预计它将设置调用将 `Int` 减去的函数。

+   `sgJ_entry` 对应于调用 `Factorial.fact` 在 `sat_sgI` 上的 thunk。我们预计它将设置调用 `Factorial.fact`。

+   `sgC_ret` 有点不同，在末尾带有 `ret` 标记。这是一个返回点，在成功评估 `ds_sgx`（即 `wild_sgC`）后我们将返回到这里。我们预计它将检查结果是否为 `0`，并返回一个一（根据“返回”的某种定义）或设置一个调用将 `Int` 与 `sgJ_entry` 及其参数相乘的函数。

该到代码时间了！这是 `sgI_entry`：

```
sgI_entry()
        { has static closure: False update_frame: <none>
          type: 0
          desc: 0
          tag: 17
          ptrs: 1
          nptrs: 0
          srt: (Factorial_fact_srt,0,1)
        }
    ch0:
        if (Sp - 24 < SpLim) goto ch2;
        I32[Sp - 4] = R1; // (reordered for clarity)
        I32[Sp - 8] = stg_upd_frame_info;
        I32[Sp - 12] = stg_INTLIKE_closure+137;
        I32[Sp - 16] = I32[R1 + 8];
        I32[Sp - 20] = stg_ap_pp_info;
        I32[Sp - 24] = base_GHCziNum_zdfNumInt_closure;
        Sp = Sp - 24;
        jump base_GHCziNum_zm_info ();
    ch2: jump stg_gc_enter_1 ();
}

```

函数顶部给出了一些元数据，这是将存储在此函数实际代码旁边的 *信息表* 的描述。如果您对值的含义感兴趣，可以查看 `cmm/CmmDecl.hs` 中的 `CmmInfoTable`；特别是标签 17 对应于 `THUNK_1_0`：这是一个 thunk，其环境中（自由变量：在本例中是 `wild_sgC`）有一个单指针和没有非指针。

不需要试图理解代码，我们可以看到一些有趣的东西：我们跳到了`base_GHCziNum_zm_info`，这是一个[Z 编码的名称](http://hackage.haskell.org/trac/ghc/wiki/Commentary/Compiler/SymbolNames)，代表`base GHC.Num - info`：嘿，这是我们的减法函数！在这种情况下，一个合理的猜测是我们写入栈的值是这个函数的参数。让我们再次看一下 STG 调用：`GHC.Num.- GHC.Num.$fNumInt wild_sgC sat_sgH`（回想起```sat_sgH was a constant 1). ``base_GHCziNum_zdfNumInt_closure```是 Z 编码的 `base GHC.Num $fNumInt`，所以这是我们的字典函数。`stg_INTLIKE_closure+137`是一个相当奇特的常量，它指向一个表示数字 `1` 的静态分配闭包。这意味着最后我们有 `I32[R1 + 8]`，必须指向 `wild_sgC`（事实上 `R1` 是指向这个 thunk 在栈上闭包的指针。）

您可能会问，`stg_ap_pp_info` 和 `stg_upd_frame_info` 是什么，为什么 `base_GHCziNum_zdfNumInt_closure` 在栈的最底部？关键是要意识到实际上，我们在栈上放置了三个不同的实体：`base_GHCziNum_zm_info` 的参数、一个包含 `I32[R1 + 8]` 和 `stg_INTLIKE_closure+137` 的 `stg_ap_pp_info` 对象的闭包，以及一个包含 `R1` 的 `stg_upd_frame_info` 对象的闭包。我们精心设计了一个鲁布·戈尔德堡机器，当运行时，将执行以下操作：

1.  在`base_GHCziNum_zm_info`内部，使用参数 `base_GHCziNum_zdfNumInt_closure` 并找出这个字典的正确减法函数，将这个函数放入栈中，然后跳转到它的返回点，栈上的下一个信息表 `stg_ap_pp_info`。

1.  在`stg_ap_pp_info`内部，消耗了`base_GHCziNum_zm_info`创建的参数，并使用`I32[R1 + 8]`和`stg_INTLIKE_closure+137`这两个参数进行应用。 （正如你可以想象的那样，`stg_ap_pp_info`非常简单。）

1.  减法函数运行并执行实际的减法操作。然后，它使用这个参数调用了堆栈上的下一个信息表`stg_upd_frame_info`。

1.  因为这是一个可更新的闭包（还记得 STG 中的`u`字符吗？），`stg_upd_frame_info`将步骤 3 的结果使用来覆盖`R1`指向的闭包（延迟求值的原始闭包），用一个新的只包含新值的闭包来替换它。然后它将调用堆栈上的下一个信息表，这个信息表是我们进入`sgI_Entry`时堆栈上的内容。

哦，现在还有一个小问题，即`if (Sp - 24 < SpLim) goto ch2;`，它检查我们是否会溢出堆栈，并在如此时跳转到垃圾收集器。

`sgJ_entry`做了类似的事情，但这次的继续执行链是从`Factorial_fact`到`stg_upd_frame_info`再到更远的地方。我们还需要在堆上分配一个新的闭包（`sgI_info`），它将作为参数传递进来：

```
sgJ_entry()
        { has static closure: False update_frame: <none>
          type: 0
          desc: 0
          tag: 17
          ptrs: 1
          nptrs: 0
          srt: (Factorial_fact_srt,0,3)
        }
    ch5:
        if (Sp - 12 < SpLim) goto ch7;
        Hp = Hp + 12;
        if (Hp > HpLim) goto ch7;
        I32[Sp - 8] = stg_upd_frame_info;
        I32[Sp - 4] = R1;
        I32[Hp - 8] = sgI_info;
        I32[Hp + 0] = I32[R1 + 8];
        I32[Sp - 12] = Hp - 8;
        Sp = Sp - 12;
        jump Factorial_fact_info ();
    ch7:
        HpAlloc = 12;
        jump stg_gc_enter_1 ();
}

```

最后，`sgC_ret`实际上进行了计算：

```
sgC_ret()
        { has static closure: False update_frame: <none>
          type: 0
          desc: 0
          tag: 34
          stack: []
          srt: (Factorial_fact_srt,0,3)
        }
    ch9:
        Hp = Hp + 12;
        if (Hp > HpLim) goto chb;
        _sgG::I32 = I32[R1 + 3];
        if (_sgG::I32 != 0) goto chd;
        R1 = stg_INTLIKE_closure+137;
        Sp = Sp + 4;
        Hp = Hp - 12;
        jump (I32[Sp + 0]) ();
    chb:
        HpAlloc = 12;
        jump stg_gc_enter_1 ();
    chd:
        I32[Hp - 8] = sgJ_info;
        I32[Hp + 0] = R1;
        I32[Sp + 0] = Hp - 8;
        I32[Sp - 4] = R1;
        I32[Sp - 8] = stg_ap_pp_info;
        I32[Sp - 12] = base_GHCziNum_zdfNumInt_closure;
        Sp = Sp - 12;
        jump base_GHCziNum_zt_info ();
}

```

...虽然内容不是很多。我们从`I32[R1 + 3]`（R1 是一个标记指针，所以偏移量看起来有些奇怪）处获取分支情况的结果。然后检查它是否为零，如果是，则将`stg_INTLIKE_closure+137`（即字面值 1）推入我们的寄存器，并跳转到我们的继续执行点；否则，我们在堆栈上设置参数以执行乘法`base_GHCziNum_zt_info`。同样进行字典传递的操作。就是这样！

当我们在这里的时候，简要提一下“优化的 Cmm”，这只是在 Cmm 上应用了一些轻微的优化。如果你真的对底层汇编的对应关系感兴趣，那么看看这个是很好的。

```
ghc -c Factorial.hs -ddump-opt-cmm

```

### 汇编语言

```
ghc -c Factorial.hs -ddump-asm

```

最后，我们来看看汇编语言。它与 Cmm 几乎相同，除了一些优化、指令选择和寄存器分配。特别是，Cmm 中的所有名称都被保留了下来，这在你用 GDB 调试编译后的 Haskell 代码时非常有用，如果你不想深入研究汇编语言：你可以查看 Cmm，了解函数的大致操作。

这里有一个摘录，显示了 Haskell 在 x86-32 上的一些更为显著的方面：

```
sgK_info:
.Lch9:
        leal -24(%ebp),%eax
        cmpl 84(%ebx),%eax
        jb .Lchb
        movl $stg_upd_frame_info,-8(%ebp)
        movl %esi,-4(%ebp)
        movl $stg_INTLIKE_closure+137,-12(%ebp)
        movl 8(%esi),%eax
        movl %eax,-16(%ebp)
        movl $stg_ap_pp_info,-20(%ebp)
        movl $base_GHCziNum_zdfNumInt_closure,-24(%ebp)
        addl $-24,%ebp
        jmp base_GHCziNum_zm_info
.Lchb:
        jmp *-8(%ebx)

```

一些寄存器被固定在我们在 Cmm 中看到的寄存器上。前两行是栈检查，我们可以看到 `%ebp` 总是设置为 `Sp` 的值。`84(%ebx)` 应该是 `SpLim` 所在的地方；确实，`%ebx` 存储了指向 `BaseReg` 结构的指针，在程序执行过程中我们将各种“类似寄存器”的数据存储在其中（以及垃圾收集函数，见 `*-8(%ebx)`）。之后，大量代码将值移动到栈上，我们可以看到 `%esi` 对应于 `R1`。实际上，一旦你分配了所有这些寄存器，就没有多少通用寄存器可以用于实际计算了：只有 `%eax` 和 `%edx`。

### 结论

就是这样：从阶乘一直到汇编级别！您可能会有几个想法：

+   *天啊！下次我需要参加混淆 C 程序设计竞赛时，我只需要让 GHC 为我生成代码就好了。* GHC 的内部运行模型确实与您可能见过的任何命令式语言非常不同，但它非常规律，一旦掌握，就相当容易理解。

+   *天啊！我简直无法相信 Haskell 居然能运行！* 记住，我们完全没有进行优化编译。使用 `-O` 编译的同一模块要聪明得多。

感谢您一路阅读！请继续关注不久的将来，我将以漫画形式展示 Haskell 堆上的操作。
