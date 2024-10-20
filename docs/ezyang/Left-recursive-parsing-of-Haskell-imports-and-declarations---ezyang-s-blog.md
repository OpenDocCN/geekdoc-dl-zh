<!--yml

类别: 未分类

日期: 2024-07-01 18:17:03

-->

# [Haskell 导入和声明的左递归解析](http://blog.ezyang.com/2016/12/left-recursive-parsing-of-haskell-imports-and-declarations/)

> 来源：[`blog.ezyang.com/2016/12/left-recursive-parsing-of-haskell-imports-and-declarations/`](http://blog.ezyang.com/2016/12/left-recursive-parsing-of-haskell-imports-and-declarations/)

假设您想解析一个由换行符分隔的列表，但希望自动忽略额外的换行符（就像 Haskell 文件中的 `import` 声明可以由一个或多个换行符分隔一样）。从历史上看，GHC 使用了一种奇怪的语法来执行这种解析（这里，分号表示换行符）：

```
decls : decls ';' decl
      | decls ';'
      | decl
      | {- empty -}

```

需要一点努力才能理解，但这个语法的要点是接受一个由 decls 组成的列表，其中夹杂着一个或多个分号，并且可以有零个或多个前导/尾随分号。例如，`;decl;;decl;` 解析为：

```
{- empty -}                             (rule 4)
{- empty -} ';' decl                    (rule 1)
{- empty -} ';' decl ';'                (rule 2)
{- empty -} ';' decl ';' ';' decl       (rule 1)
{- empty -} ';' decl ';' ';' decl ';'   (rule 2)

```

（如果没有前导分号，则执行规则 3。）

此语法有两个优点：首先，它只需要一个单一状态，这减少了解析器的大小；其次，它是左递归的，这意味着 LALR 解析器（如 Happy）可以在恒定的堆栈空间中解析它。

这段代码很长时间以来运行得很好，但当我们向 GHC 添加注释时，它最终变得复杂起来。注释是一种跟踪源代码中所有关键字/标点符号/空白位置的功能，因此我们可以逐字节地从抽象语法树重建源代码（通常情况下，格式信息在抽象语法中丢失）。有了注释，我们需要保存关于每个分号的信息；出于我不太理解的原因，我们花了很大力气将每个分号与前面的声明关联起来（前导分号传播到封闭元素）。

这导致了一些非常恶心的解析器代码：

```
importdecls :: { ([AddAnn],[LImportDecl RdrName]) }
        : importdecls ';' importdecl
                                {% if null (snd $1)
                                     then return (mj AnnSemi $2:fst $1,$3 : snd $1)
                                     else do
                                      { addAnnotation (gl $ head $ snd $1)
                                                      AnnSemi (gl $2)
                                      ; return (fst $1,$3 : snd $1) } }
        | importdecls ';'       {% if null (snd $1)
                                     then return ((mj AnnSemi $2:fst $1),snd $1)
                                     else do
                                       { addAnnotation (gl $ head $ snd $1)
                                                       AnnSemi (gl $2)
                                       ; return $1} }
        | importdecl             { ([],[$1]) }
        | {- empty -}            { ([],[]) }

```

你能说出这段代码做了什么吗？我花了一段时间才明白代码在做什么：空测试是为了检查是否有*前面的*元素可以附加分号注释：如果没有，则将分号传播到顶层。

问题的关键在于，一旦添加了注释，**语法就不再与语法树的逻辑结构匹配了。**这很糟糕。让我们让它们匹配起来。以下是一些约束条件：

1.  前导分号与*封闭的* AST 元素相关联。因此，我们希望在开始时解析它们一次，然后在递归规则中不再处理它们。称解析零个或多个分号的规则为 `semis`：

    ```
    semis : semis ';'
          | {- empty -}

    ```

1.  如果有重复的分号，我们希望一次解析它们全部，然后将它们与前面的声明关联起来。因此，我们还需要一条规则来解析一个或多个分号，我们将其称为 `semis1`；然后当我们解析单个声明时，我们想将其解析为 `decl semis1`：

    ```
    semis1 : semis1 ';'
           | ';'

    ```

然后，我们可以以下列方式建立我们的解析器：

```
-- Possibly empty decls with mandatory trailing semicolons
decls_semi : decls_semi decl semis1
           | {- empty -}

-- Non-empty decls with no trailing semicolons
decls : decls_semi decl

-- Possibly empty decls with optional trailing semicolons
top1 : decls_semi
     | decls

-- Possibly empty decls with optional leading/trailing semicolons
top : semi top1

```

我们特别注意不引入任何移位-归约冲突。实际上，如何做到这一点有点不明显，因为在 Haskell 源文件中，我们需要解析一系列导入声明（`importdecl`），然后是一系列顶层声明（`topdecl`）。在不引入移位-归约冲突的情况下定义这两个列表的语法有些困难，但似乎这样做是有效的：

```
top : importdecls_semi topdecls_semi
    | importdecls_semi topdecls
    | importdecls

```

看起来很简单，但有很多看似合理的替代方案，这些方案会引入移位/归约冲突。这里有一个重要的元教训，就是在尝试像这样做某事时，最好先在一个较小的语法上进行实验，这样重新检查是即时的（happy 花费了相当长的时间来处理所有的 GHC，这使得编辑-重新编译周期有点痛苦。）

我很想知道是否有更简单的方法来做到这一点，或者是否我犯了错误并改变了我接受的语言集。在评论中告诉我。我附上了一个简单的 Happy 语法供你玩耍（使用`happy filename.y; ghc --make filename.hs`构建）。

```
{
module Main where

import Data.Char
}

%name parse
%expect 0
%tokentype { Token }
%error { parseError }

%token
      import          { TokenImport }
      decl            { TokenDecl }
      ';'             { TokenSemi }

%%

top     : semis top1                        { $2 }
top1    : importdecls_semi topdecls_semi    { (reverse $1, reverse $2) }
        | importdecls_semi topdecls         { (reverse $1, reverse $2) }
        | importdecls                       { (reverse $1, []) }

id_semi : importdecl semis1                 { $1 }
importdecls
        : importdecls_semi importdecl       { $2:$1 }
importdecls_semi
        : importdecls_semi id_semi          { $2:$1 }
        | {- empty -}                       { [] }

topdecls
        : topdecls_semi topdecl             { $2:$1 }
topdecls_semi
        : topdecls_semi topdecl semis1      { $2:$1 }
        | {- empty -}                       { [] }

semis   : semis ';'                         { () }
        | {- empty -}                       { () }

semis1  : semis1 ';'                        { () }
        | ';'                               { () }

importdecl
        : import                            { "import" }
topdecl : decl                              { "decl" }

{
parseError :: [Token] -> a
parseError p = error ("Parse error: " ++ show p)

data Token
      = TokenImport
      | TokenDecl
      | TokenSemi
 deriving Show

lexer :: String -> [Token]
lexer [] = []
lexer (c:cs)
      | isSpace c = lexer cs
      | isAlpha c = lexVar (c:cs)
lexer (';':cs) = TokenSemi : lexer cs

lexVar cs =
   case span isAlpha cs of
      ("import",rest) -> TokenImport : lexer rest
      ("decl",rest) -> TokenDecl : lexer rest

main = print . parse . lexer $ "import;;import;;decl"
}

```
