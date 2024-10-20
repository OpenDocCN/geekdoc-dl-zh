<!--yml

分类：未分类

日期：`2024-07-01 18:17:18`

-->

# 没有语法？没问题！：ezyang 的博客

> 来源：[`blog.ezyang.com/2013/07/no-grammar-no-problem/`](http://blog.ezyang.com/2013/07/no-grammar-no-problem/)

有一天，当你漫步在代码的领域时，突然间你发现一个你不理解的语法结构。

也许你会问你的同事，他会立刻告诉你它是什么。

也许你的编程工具链可以告诉你。也许 IDE 会在鼠标悬停在构造上时告诉你，或者你正在使用 Coq，它允许你`Locate`自定义表示。

也许你会拉起手册（或者更可能的是，众多的教程之一），并且扫描寻找所讨论的语法结构。

但是当所有这些都失败时，该怎么办呢？如果所讨论的代码是写在编译器的内部语言中的，并且其细节自从上次文档化以来就发生了变化，而文档又已经过时了呢？

*没问题。* 只要你愿意卷起袖子，查看相关编译器的源代码，你通常可以在比查阅手册更少的时间内解决你的问题（而且它保证是最新的！）。关键在于现代编译器都使用解析器生成器，而这些输入本质上是可执行的规范。

* * *

我将从 GHC 中给出两个例子。第一个来自 C--，GHC 的高级汇编语言。考虑这个函数：

```
INFO_TABLE_RET(stg_maskUninterruptiblezh_ret, RET_SMALL, W_ info_ptr)
    return (P_ ret)
{
    StgTSO_flags(CurrentTSO) =
       %lobits32(
        (TO_W_(StgTSO_flags(CurrentTSO))
          | TSO_BLOCKEX)
          & ~TSO_INTERRUPTIBLE
       );

    return (ret);
}

```

这个定义的一些方面对于之前写过 C 语言的人来说很熟悉，但还有一些神秘的部分。例如，在导言中`return (P_ ret)`是什么意思？

首要任务是找到相关的文件。当所讨论的代码具有非常独特的关键字（就像这个例子一样）时，grep 通常可以解决问题：

```
ezyang@javelin:~/Dev/ghc-clean/rts$ grep -R INFO_TABLE_RET ../compiler/
../compiler/cmm/CmmParse.y:INFO_TABLE_RET ( label, FRAME_TYPE, info_ptr, field1, ..., fieldN )
../compiler/cmm/CmmParse.y:        'INFO_TABLE_RET'{ L _ (CmmT_INFO_TABLE_RET) }
../compiler/cmm/CmmParse.y:        | 'INFO_TABLE_RET' '(' NAME ',' INT ')'
../compiler/cmm/CmmParse.y:        | 'INFO_TABLE_RET' '(' NAME ',' INT ',' formals0 ')'
../compiler/cmm/CmmParse.y:-- is.  That is, for an INFO_TABLE_RET we want the return convention,
../compiler/cmm/CmmLex.x:  | CmmT_INFO_TABLE_RET
../compiler/cmm/CmmLex.x:   ( "INFO_TABLE_RET",     CmmT_INFO_TABLE_RET ),

```

文件扩展名也可能是明显的线索；GHC 使用一个名为 Happy 的解析器生成器，而 Happy 文件的文件扩展名是`.y`：

```
ezyang@javelin:~/Dev/ghc-clean/rts$ find ../compiler -name *.y
../compiler/cmm/CmmParse.y
../compiler/parser/ParserCore.y

```

从这里，我们可以搜索文件中的关键字或符号（检查是否使用了字符串标记名称的词法分析器；还要确保引用了字母数字文本）。符号可能会出现在多个地方，就像`return`一样：

```
maybe_conv :: { Convention }
           : {- empty -}        { NativeNodeCall }
           | 'return'           { NativeReturn }

```

以及：

```
stmt    :: { CmmParse () }
        : ';'                                   { return () }
...
        | 'goto' NAME ';'
                { do l <- lookupLabel $2; emit (mkBranch l) }
        | 'return' '(' exprs0 ')' ';'
                { doReturn $3 }

```

根据产生式的名称和上下文猜测，`maybe_conv`似乎是相关的产生式。它在这里使用：

```
cmmproc :: { CmmParse () }
        : info maybe_conv maybe_formals maybe_body
                { do ((entry_ret_label, info, stk_formals, formals), agraph) <-
                       getCodeR $ loopDecls $ do {
                         (entry_ret_label, info, stk_formals) <- $1;
                         formals <- sequence (fromMaybe [] $3);
                         $4;
                         return (entry_ret_label, info, stk_formals, formals) }
                     let do_layout = isJust $3
                     code (emitProcWithStackFrame $2 info
                                entry_ret_label stk_formals formals agraph
                                do_layout ) }

```

现在，如果你真的需要*准确*了解它是如何布局的，你可以去查看`emitProcWithStackFrame`是如何实现的。或者，你可能希望源文件中有一个有用的注释来解释这是什么：

```
A stack frame is written like this:

INFO_TABLE_RET ( label, FRAME_TYPE, info_ptr, field1, ..., fieldN )
               return ( arg1, ..., argM )
{
  ... code ...
}

where field1 ... fieldN are the fields of the stack frame (with types)
arg1...argN are the values returned to the stack frame (with types).
The return values are assumed to be passed according to the
NativeReturn convention.

```

* * *

第二个例子是针对 STG 的，你可以要求 GHC 使用`-ddump-stg`打印出来。现在，STG 没有解析器，所以你将不得不查看*pretty-printer*。这不太困难。看这个简单的函数：

```
Gnam.$WKST =
    \r [tpl_sl4 tpl_sl6]
        case tpl_sl4 of tpl_sl8 {
          __DEFAULT ->
              case tpl_sl6 of tpl_sl9 {
                __DEFAULT -> Gnam.KST [tpl_sl8 tpl_sl9];
              };
        };

```

有些方面很熟悉。但`\r`是什么意思呢？

再次，我们必须找到相关的源文件。由于只有在通过`-ddump-stg`标志时才打印出 STG，追踪该标志通过源代码是一个好的开始：

```
ezyang@javelin:~/Dev/ghc-clean/compiler$ grep -R ddump-stg .
./main/DynFlags.hs:  , Flag "ddump-stg"               (setDumpFlag Opt_D_dump_stg)
ezyang@javelin:~/Dev/ghc-clean/compiler$ grep -R Opt_D_dump_stg .
./main/DynFlags.hs:   | Opt_D_dump_stg
./main/DynFlags.hs:  , Flag "ddump-stg"               (setDumpFlag Opt_D_dump_stg)
./simplStg/SimplStg.lhs:        ; dumpIfSet_dyn dflags Opt_D_dump_stg "STG syntax:"

```

这是一个好迹象！打开`SimpleStg.lhs`给了我们：

```
; dumpIfSet_dyn dflags Opt_D_dump_stg "STG syntax:"
                (pprStgBindings un_binds)

```

而`pprStgBindings`的位置（`compiler/stgSyn/StgSyn.lhs`）实际上就是关键。

STG 相当简单，事实证明，如果你只是快速浏览文件，你很可能会找到你需要的东西。但是如果没有找到，你仍然可以有意识地找出答案。假设我们搜索带引号的反斜杠：

```
pprStgExpr (StgLam bndrs body)
  = sep [ char '\\' <+> ppr_list (map (pprBndr LambdaBind) bndrs)
            <+> ptext (sLit "->"),
         pprStgExpr body ]
  where ppr_list = brackets . fsep . punctuate comma

...

-- general case
pprStgRhs (StgRhsClosure cc bi free_vars upd_flag srt args body)
  = sdocWithDynFlags $ \dflags ->
    hang (hsep [if gopt Opt_SccProfilingOn dflags then ppr cc else empty,
                pp_binder_info bi,
                ifPprDebug (brackets (interppSP free_vars)),
                char '\\' <> ppr upd_flag, pprMaybeSRT srt, brackets (interppSP args)])
         4 (ppr body)

```

这是什么？事实证明：

```
StgLam is used *only* during CoreToStg's work. Before CoreToStg has
finished it encodes (\x -> e) as (let f = \x -> e in f)

```

因为`-ddump-stg`是在 CoreToSTG 之后，我们必须看看`StgRhsClosure`，`ppr upd_flag`看起来像是关键。`r`必须是一个`upd_flag`，不管那是什么。正如事实证明的那样，是`UpdateFlag`：

```
data UpdateFlag = ReEntrant | Updatable | SingleEntry

instance Outputable UpdateFlag where
    ppr u = char $ case u of
                       ReEntrant   -> 'r'
                       Updatable   -> 'u'
                       SingleEntry -> 's'

```

`r`表示函数是可重入的！（当然，关于这是什么意思，你得查阅其他文档。）

* * *

当然，在理想的世界中，所有这些都将有文档记录。但即使没有，也没有理由不能自己帮助自己。如果你的代码库像 GHC 那样好，将会有很多线索和注释来帮助你。希望这能为你在遇到不熟悉的东西和不知道如何学习时的思考过程提供一些见解。（当然，有时最好还是忽略它！）
