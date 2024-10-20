<!--yml

类别：未分类

日期：2024-07-01 18:17:14

-->

# Parsec：“try a <|> b”被认为是有害的：ezyang 的博客

> 来源：[`blog.ezyang.com/2014/05/parsec-try-a-or-b-considered-harmful/`](http://blog.ezyang.com/2014/05/parsec-try-a-or-b-considered-harmful/)

*tl;dr 应该将回溯 try 的范围最小化，通常是将其放置在解析器定义的内部。*

你是否曾经编写过一个 Parsec 解析器并得到了一个非常不具信息性的错误消息？

```
"test.txt" (line 15, column 7):
unexpected 'A'
expecting end of input

```

行号和列号在文档中随机地某个地方，并且你非常确定应该在某些解析器组合器的堆栈中间。但是等等！Parsec 已经以某种方式得出结论，文档应该立即结束。你继续思考并发现真正的错误在实际报告的行号和列号之后一段距离。

你想：“难怪 Parsec 的错误处理声名狼藉。”

* * *

假设你所询问的语法并不太奇怪，通常对于这样的错误消息有一个简单的解释：程序员在他们的代码中撒入了太多的回溯`try`语句，并且回溯已经破坏了有用的错误状态。实际上，在某些时候，解析器因为我们想向用户报告的原因而失败，但是一个封闭的`try`语句迫使解析器回溯并尝试另一种（徒劳无功的）可能性。

这可以通过一个例子来说明。一个 Haskell 程序员正在使用解析组合器玩耍，并决定通过编写 Haskell 模块导入的解析器来测试他们的解析技能：

```
stmt ::= import qualified A as B
       | import A

```

利用 Parsec 内置的[token 组合器](http://hackage.haskell.org/package/parsec-3.0.0/docs/Text-Parsec-Token.html)(以及示例代码)，他们的第一个版本可能看起来像这样：

```
import Text.Parsec
import qualified Text.Parsec.Token as P
import Text.Parsec.Language (haskellDef)

data Stmt = QualifiedImport String String | Import String
    deriving (Show)

pStmt = pQualifiedImport <|> pImport

pQualifiedImport = do
    reserved "import"
    reserved "qualified"
    i <- identifier
    reserved "as"
    i' <- identifier
    return (QualifiedImport i i')

pImport = do
    reserved "import"
    i <- identifier
    return (Import i)

lexer = P.makeTokenParser (haskellDef
    { P.reservedNames = P.reservedNames haskellDef ++ ["qualified", "as"] })
identifier = P.identifier lexer
reserved = P.reserved lexer

parseStmt input = parse (pStmt >> eof) "(unknown)" input

```

不幸的是，该解析器对于常规的导入不起作用，它们会收到以下错误消息：

```
*Main> parseStmt "import Foo"
Left "(unknown)" (line 1, column 8):
unexpected "F"
expecting "qualified"

```

经过一番搜索，他们发现 [Parsec 不默认回溯](http://stackoverflow.com/questions/9976388/haskell-text-parsec-combinator-choice-doesnt-backtrack)。好吧，那很好；为什么不只是在解析器中插入一个 try 呢。

```
pStmt = try pQualifiedImport <|> pImport

```

这既修复了两种解析，并提出了撰写未来解析器的以下规则：

> 如果我需要在多个解析器之间做出选择，但其中一些解析器可能会消耗输入，我最好在每个解析器上添加一个 `try`，这样我就可以回溯。

用户并不知道，他们已经引入了不良的错误报告行为：

```
*Main> parseStmt "import qualified Foo s B"
Left "(unknown)" (line 1, column 17):
unexpected reserved word "qualified"
expecting letter or digit or "#"

```

等一下！我们想要的错误是，当我们期望 `as` 时，出现了意外的标识符`s`。但是，Parsec 没有在此发生时报告错误，而是*回溯*，并尝试匹配 `pImport` 规则，仅在该规则失败后才失败。到那时，我们的选择分支失败的知识已经永远丢失了。

我们该如何修复它？问题在于我们的代码在我们，开发者，知道会徒劳无功时回溯。特别是，一旦我们解析了`import qualified`，我们就知道这个语句实际上是一个有资格的导入，我们就不应该再回溯了。我们如何让 Parsec 理解这一点？简单：*减少 try 回溯操作符的作用范围*：

```
pStmt = pQualifiedImport <|> pImport

pQualifiedImport = do
    try $ do
        reserved "import"
        reserved "qualified"
    i <- identifier
    reserved "as"
    i' <- identifier
    return (QualifiedImport i i')

```

在这里，我们将`try`从`pStmt`移动到`pQualifiedImport`，只有在无法解析`import qualified`时才回溯。一旦解析成功，我们消耗这些标记，现在我们已经选择了一个有资格的导入。错误消息相应地变得更好：

```
*Main> parseStmt "import qualified Foo s F"
Left "(unknown)" (line 1, column 22):
unexpected "s"
expecting "as"

```

故事的寓意：应该尽量减少回溯 try 的范围，通常是通过将其放置在解析器的定义内部来实现。需要一定的技巧：您必须能够确定需要多少前瞻来承诺到一个分支，这通常取决于解析器的使用方式。幸运的是，许多语言专门设计，以便所需的前瞻不会太大，对于我可能使用 Parsec 的项目类型，我愿意牺牲这种模块化。

* * *

另一种看待这场灾难的方式是 Parsec 的问题：它不应该提供一个 API，使得错误消息混乱变得如此容易——为什么它不能自动确定必要的前瞻呢？虽然传统的解析器生成器可以做到这一点（并通过在我们之前的例子中完全避免回溯来提高效率），但是有一些基本原因解释了为什么 Parsec（以及像它一样的单子解析器组合库）不能自动地确定[需要什么前瞻](http://stackoverflow.com/a/7863380/23845)。这是为什么（众多原因之一），许多 Haskeller 更喜欢更快的解析器，它们根本就[不试图进行任何错误处理。](https://hackage.haskell.org/package/attoparsec)

为什么我首先写这篇帖子呢？目前还有大量的文档建议使用 Parsec，并且初学者更有可能在 Parsec 中实现他们的第一个解析器。如果有人要写一个 Parsec 解析器，那么最好花点时间来限制回溯：这样可以使得与 Parsec 解析器一起工作变得*更加愉快*。
