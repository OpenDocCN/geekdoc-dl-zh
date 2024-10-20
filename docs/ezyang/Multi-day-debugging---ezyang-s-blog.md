<!--yml

category: 未分类

date: 2024-07-01 18:17:57

-->

# 多日调试：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/02/multi-day-debugging/`](http://blog.ezyang.com/2011/02/multi-day-debugging/)

目前，我大部分的编程时间都投入到调试 GHC 的新代码生成器上。GHC 的代码生成阶段将 Spineless Tagless G-machine（STG）中间表示转换为 C-- 高级汇编表示；旧代码生成器基本上一次性完成了这个步骤。新代码生成器则是多功能的。它是一个更模块化、更易理解和更灵活的代码库。它是控制流优化高阶框架研究的客户端。

调试也非常难。

过去，如果我在经过几个小时的深入分析后仍然无法弄清楚 bug 的原因，我会感到沮丧并放弃。但在 GHC 上的工作使我对多日调试有了更深入的了解：一个明确定义的 bug，尽管经过数日的强烈分析仍然存在。 （过去我只成功过几次，我很自豪地说我设法收集足够的信息来解决[“这个 bug”](http://bugs.php.net/42362)。什么是“这个 bug”？你曾经在浏览 MediaWiki 站点时神秘地被要求下载 PHP 文件吗？是的，那就是“这个 bug”）。 这大大提高了我在 gdb 上的熟练程度，并且在编译器构造的理论和实践中进行了惊险的冒险。我觉得自己愚蠢，因为没有立即理解回顾来看似乎完全清晰和显而易见的概念。我感到一种惊人的冲动，不是因为问题解决了（尽管当然这会带来好的感觉），而是因为我的攻击计划正在取得进展。我看到我的理论从一个到另一个再到另一个发展，并且学会了从不信任任何第一眼的实验观察。

虽然调试过程尚未完成（尽管我认为我接近拥有正确但慢的新代码生成流水线），我想抽出一些时间来描述这段旅程。

### 为什么调试 GHC 如此容易？

有趣的是，虽然错误导致编译程序表现出极其奇怪的行为，需要花费很长时间才能解析，但一旦完全理解了错误行为，修复通常只需一行代码。正是这个事实使得调试 GHC 既令人沮丧又光辉：有时你正在调试的代码基本上是错的，你必须重写整个东西。 GHC 的代码基本上是清晰的（这是写它的人的证明），而 bug 通常只是有人忘记处理的一个小细节。解决方案就像是神秘的寻宝解：简短，你找到它时就知道。没有杂乱的情况，比如，“这实际上应该做什么？”

我有一个现有的代码生成管道，可以用来与我的结果进行比较，尽管这样做并不容易，因为新的代码生成器在编译过程中采用了基本不同的方式，所以代码部分经常无法比较。

我也有一个奇妙的测试套件，可以轻松地生成能够在单线程情况下导致段错误的程序，并且相对幸运地遇到只在单线程情况下显示的 bug。我的程序有明确定义的输入和输出，并且我有复杂的机制来检查多通道编译器的内部状态。

### 我到目前为止修复了什么？

警告：接下来是繁琐的技术细节。

#### 构建它

我的第一个工作是使最新的代码生成代码与最新的 GHC 分支编译（在这段时间内有些陈旧）。这基本上进行得很顺利，除了一个问题，Norman Ramsey 真的喜欢多态本地定义，MonoLocalBinds 在 Hoopl 和其他几个模块中显现了它丑陋的一面。

#### 测试 4030

[测试 4030](http://hackage.haskell.org/trac/ghc/ticket/4030)是这个“简单”程序（简单用引号，因为正如 Simon Peyton-Jones 所说的那样，“这看起来像一个难以开始的... 线程、异常等”）。

```
main = do tid <- block $ forkIO $ let x = x in x
        killThread tid

```

尝试解引用“something”时，生成的代码在 stg_BLACKHOLE_info 处导致段错误。

```
0x822a6e0 <stg_CAF_BLACKHOLE_info>:  jmp    0x822a620 <stg_BLACKHOLE_info>
0x822a620 <stg_BLACKHOLE_info>:      mov    0x4(%esi),%eax
0x822a623 <stg_BLACKHOLE_info+3>:    test   $0x3,%eax
0x822a628 <stg_BLACKHOLE_info+8>:    jne    0x822a663 <stg_BLACKHOLE_info+67>
0x822a62a <stg_BLACKHOLE_info+10>:   mov    (%eax),%ecx -- SEGFAULT!

```

这个“something”最终成为了 Simon Marlow 在他[重写黑洞方案](http://hackage.haskell.org/trac/ghc/ticket/3838)时引入的一个新的栈插槽。解决方案是将这些更改移植到新的代码生成器上。我最终在合并时间窗口内手动审核了每个补丁，以确保所有更改都已移植，并在这个过程中可能消灭了一些潜在的 bug。没有补丁，因为我最终将这个改动合并到了一起（因为新的黑洞方案在新代码生成器分支冻结时还不存在）。

#### 测试 ffi021

测试 ffi021 包括创建指向导入的 FFI 函数的指针，然后动态执行它。（我甚至不知道你可以用 FFI 做到这一点！）

```
type Malloc = CSize -> IO (Ptr ())

foreign import ccall unsafe "&malloc" pmalloc:: FunPtr Malloc
foreign import ccall unsafe "dynamic" callMalloc :: FunPtr Malloc -> Malloc

```

这最终是内联语句优化器中的潜在 bug（不是新代码生成器中的 bug，而是新代码生成器触发的优化 bug）。我得出结论认为这是本地代码生成器中的优化 bug，然后 Simon Marlow 辨认出了这个 bug，并且我们得到了[一个单行补丁](http://www.mail-archive.com/cvs-ghc@haskell.org/msg24392.html)。

```
hunk ./compiler/cmm/CmmOpt.hs 156
-   where infn (CmmCallee fn cconv) = CmmCallee fn cconv
+   where infn (CmmCallee fn cconv) = CmmCallee (inlineExpr u a fn) cconv

```

#### 测试 4221

这个问题花了三周时间解决。原始的测试代码相当复杂，对代码变更非常敏感。我最初的理论是，我们试图访问一个从未溢出到栈上的变量，但在与 Simon Peyton Jones 讨论栈溢出工作原理后，我开始怀疑这可能并不是问题，并停止试图理解做溢出的 Hoopl 代码，重新进行分析。关于优化燃料还有另一个错误的尝试，我希望它能帮助我找到错误点，但事实上并不起作用。（优化燃料允许您逐步增加应用的优化数量，因此您可以二分搜索引入错误的优化。不幸的是，大部分所谓的“优化”实际上是通往机器码的关键程序转换。）

突破口在于我意识到，当我将输入程序中的类型从 CDouble 改为 CInt64 时，错误仍然存在，但当我将类型更改为 CInt32 时却不存在。这使我能够识别出涉及 *垃圾收集* 的错误 C-- 代码，并将测试用例缩减为一个非常小的程序，它不会崩溃，但显示出错误的代码（因为程序需要运行一段时间才能在正确的位置触发堆栈溢出）：

```
{-# LANGUAGE ForeignFunctionInterface #-}
module Main(main) where

import Foreign.C

foreign import ccall safe "foo" foo :: CLLong -> CLLong
-- Changing to unsafe causes stg_gc_l1 to not be generated
-- Changing to IO causes slight cosmetic changes, but it's still wrong

main = print (foo 0)

```

在对调用约定产生了巨大误解并在栈布局代码中找不到 bug 的徒劳尝试之后（我认为 `slot<foo> + 4` 表示更高的内存位置；实际上它表示比 `slot<foo>` 更低的内存位置），我最终确认问题出在 `stg_gc_*` 的调用约定上。

我的第一个修补程序是将被调用者（`stg_gc_*` 函数）更改为使用新代码生成器发出的观察到的调用约定，因为我看不出那段代码有什么问题。但有一个异常的地方：按照这个理论，所有调用 GC 的地方都应该使用错误的调用约定，然而只有双精度和 64 位整数表现出了这种行为。我的修补程序起了作用，但有些不对劲。这个不对劲实际上是 32 位 x86 没有通用目的的非 32 位寄存器，这就是代码生成器只将这些类型的参数溢出到栈上的原因。我对 GHC 的虚拟寄存器有了更多了解，并确定了另一个一行修复方案。

```
hunk ./compiler/cmm/CmmCallConv.hs 50
-               (_,   GC)               -> getRegsWithNode
+               (_,   GC)               -> allRegs

```

#### 测试 2047（bagels）

这个正在进行中。修复了 GC bug 后，所有剩余的神秘测试套件失败问题都解决了（万岁），我也能够用新的代码生成器重新编译 GHC 和所有库。这导致了 [test 2047](http://hackage.haskell.org/trac/ghc/ticket/2047) 开始出现段错误。

我花了一点时间确认我没有在用新的代码生成器编译第二阶段编译器时引入错误（我做得过于热情了），并确认哪个库代码有错误，但一旦我这样做了，我就设法将它减少到以下程序（我曾经贴心地命名为“begals”）：

```
import Bagel

main = do
    l <- getContents
    length l `seq` putStr (sort l)

```

在模块 Bagel 中，sort 定义如下：

```
module Bagel where
-- a bastardized version of sort that still exhibits the bug
sort :: Ord a => [a] -> [a]
sort = mergeAll . sequences
  where
    sequences (a:xs) = compare a a `seq` []:sequences xs
    sequences _ = []

    mergeAll [x] = x
    mergeAll xs  = mergeAll (mergePairs xs)

    mergePairs (a:b:xs) = merge a b: mergePairs xs
    mergePairs xs       = xs

    merge (a:as') (b:bs') = compare a a `seq` merge as' as'
    merge _ _ = []

```

并使用以下数据运行：

```
$ hexdump master-data
0000000 7755 7755 7755 7755 7755 7755 7755 7755
*
000b040

```

该程序具有一些奇怪的特性。如果我：

+   关闭紧凑式 GC

+   减少主数据的大小

+   关闭优化

+   使用旧的代码生成器

+   将所有代码放在一个文件中

+   从'sort'中删除 seqs（实际上不是一个排序）

+   从'main'中删除 seqs

+   使 sort 函数在 Char 上具有单态性

当前的理论是某人（可能是新的代码生成器或紧凑式 GC）没有正确处理标签位，但我还没有完全弄清楚具体是哪里。这是新代码生成器唯一的突出问题。
