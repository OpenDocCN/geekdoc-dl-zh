<!--yml

category: 未分类

date: 2024-07-01 18:17:05

-->

# 尝试 Backpack：ghc –backpack：ezyang's 博客

> 来源：[`blog.ezyang.com/2016/10/try-backpack-ghc-backpack/`](http://blog.ezyang.com/2016/10/try-backpack-ghc-backpack/)

[Backpack](https://ghc.haskell.org/trac/ghc/wiki/Backpack)，一个用于 Haskell 中混合包的新系统，已经随着 GHC 8.2 发布。虽然 Backpack 与 Cabal 包系统紧密集成，但仍然可以使用一个新命令 `ghc --backpack` 玩耍。在开始之前，请确保你有一个足够新的 [GHC 版本](https://ghc.haskell.org/trac/ghc/blog/ghc-8.2.11-released)：

```
ezyang@sabre:~$ ghc-8.2 --version
The Glorious Glasgow Haskell Compilation System, version 8.2.1

```

顺便说一句，如果你想真正地开始使用 Backpack（包括 Cabal 包等），跳过本教程直接参阅 [Try Backpack: Cabal packages](http://blog.ezyang.com/2017/01/try-backpack-cabal-packages/)。

### Hello World

GHC 支持一种新的文件格式，`bkp` 文件，允许你在单个源文件中轻松定义多个模块和包，这样就可以轻松地使用 Backpack 进行实验。这种格式不适合大规模编程（`bkp` 文件与 Cabal 没有集成，我们也不打算添加这样的集成），但我们会在教程中使用它，因为它非常方便在不与大量 Cabal 包混淆的情况下玩转 Backpack。

这是一个简单的 "Hello World" 程序：

```
unit main where
  module Main where
    main = putStrLn "Hello world!"

```

我们定义了一个单元（类似于包），具有特殊名称 `main`，在其中定义了一个 `Main` 模块（同样是特殊名称），包含我们的 `main` 函数。将其放入名为 `hello.bkp` 的文件中，然后运行 `ghc --backpack hello.bkp`（使用您的 GHC nightly）。这将在 `main/Main` 处生成一个可执行文件，您可以运行它；您还可以使用 `-o filename` 显式指定所需的输出文件名。请注意，默认情况下，`ghc --backpack` 创建一个与每个单元同名的目录，因此 `-o main` 不起作用（它会给出链接器错误；请使用其他名称！）

### A Play on Regular Expressions

让我们写一些真正使用 Backpack 的非平凡代码。在本教程中，我们将按照 [A Play on Regular Expressions](https://sebfisch.github.io/haskell-regexp/regexp-play.pdf)（Sebastian Fischer, Frank Huch, Thomas Wilke）中描述的简单正则表达式匹配器写一个简单的示例。匹配器本身效率低下（通过测试所有指数级字符串分解来检查匹配），但足以说明 Backpack 的许多关键概念。

要开始，让我们复制粘贴功能珍珠中的代码到 Backpack 文件的 `Regex` 模块中，并写一个小测试程序来运行它：

```
unit regex where
    module Regex where
        -- | A type of regular expressions.
        data Reg = Eps
                 | Sym Char
                 | Alt Reg Reg
                 | Seq Reg Reg
                 | Rep Reg

        -- | Check if a regular expression 'Reg' matches a 'String'
        accept :: Reg -> String -> Bool
        accept Eps       u = null u
        accept (Sym c)   u = u == [c]
        accept (Alt p q) u = accept p u || accept q u
        accept (Seq p q) u =
            or [accept p u1 && accept q u2 | (u1, u2) <- splits u]
        accept (Rep r) u =
            or [and [accept r ui | ui <- ps] | ps <- parts u]

        -- | Given a string, compute all splits of the string.
        -- E.g., splits "ab" == [("","ab"), ("a","b"), ("ab","")]
        splits :: String -> [(String, String)]
        splits [] = [([], [])]
        splits (c:cs) = ([], c:cs):[(c:s1,s2) | (s1,s2) <- splits cs]

        -- | Given a string, compute all possible partitions of
        -- the string (where all partitions are non-empty).
        -- E.g., partitions "ab" == [["ab"],["a","b"]]
        parts :: String -> [[String]]
        parts [] = [[]]
        parts [c] = [[[c]]]
        parts (c:cs) = concat [[(c:p):ps, [c]:p:ps] | p:ps <- parts cs]

unit main where
    dependency regex
    module Main where
        import Regex
        nocs = Rep (Alt (Sym 'a') (Sym 'b'))
        onec = Seq nocs (Sym 'c')
        -- | The regular expression which tests for an even number of cs
        evencs = Seq (Rep (Seq onec onec)) nocs
        main = print (accept evencs "acc")

```

如果你将这段代码放在 `regex.bkp` 中，可以再次使用 `ghc --backpack regex.bkp` 编译它，并在 `main/Main` 处调用生成的可执行文件。它应该会打印出 `True`。

### Functorizing the matcher

先前显示的代码并不好，因为它将`String`硬编码为用于正则表达式匹配的类型。一个合理的泛化（你可以在原始论文中看到）是在任意符号列表上进行匹配；然而，我们可能也希望在非列表类型（如`ByteString`）上进行匹配。为了支持所有这些情况，我们将使用 Backpack 来“泛型化”（在 ML 术语中）我们的匹配器。

我们将通过创建一个新单元`regex-indef`并编写一个提供字符串类型的签名（我们决定称其为`Str`，以避免与`String`混淆）来完成这个任务。以下是我所采取的步骤：

1.  首先，我将旧的`Regex`实现复制粘贴到新的单元中。我用`Str`替换了所有`String`的出现，并删除了`splits`和`parts`：我们需要在签名中实现这些。

1.  接下来，我们创建一个新的`Str`签名，它由`Regex`引入，并定义了我们需要支持的类型和操作（`splits`和`parts`）：

    ```
    signature Str where
      data Str
      splits :: Str -> [(Str, Str)]
      parts :: Str -> [[Str]]

    ```

1.  在这一点上，我运行了`ghc --backpack`来对新单元进行类型检查。但我得到了两个错误！

    ```
    regex.bkp:90:35: error:
        • Couldn't match expected type ‘t0 a0’ with actual type ‘Str’
        • In the first argument of ‘null’, namely ‘u’
          In the expression: null u
          In an equation for ‘accept’: accept Eps u = null u

    regex.bkp:91:35: error:
        • Couldn't match expected type ‘Str’ with actual type ‘[Char]’
        • In the second argument of ‘(==)’, namely ‘[c]’
          In the expression: u == [c]
          In an equation for ‘accept’: accept (Sym c) u = u == [c]

    ```

    除了遍历`null`的无意义外，这些错误非常明显：`Str`是一个完全抽象的数据类型：我们不能假设它是一个列表，也不知道它有什么实例。为了解决这些类型错误，我引入了组合子`null`和`singleton`，一个`instance Eq Str`，并重写了`Regex`以使用这些组合子（这是一个非常谨慎的改变）。 （注意，我们不能写`instance Traversable Str`；这是一种类型不匹配。）

这是我们最终的正则表达式单元的不定版本：

```
unit regex-indef where
    signature Str where
        data Str
        instance Eq Str
        null :: Str -> Bool
        singleton :: Char -> Str
        splits :: Str -> [(Str, Str)]
        parts :: Str -> [[Str]]
    module Regex where
        import Prelude hiding (null)
        import Str

        data Reg = Eps
                 | Sym Char
                 | Alt Reg Reg
                 | Seq Reg Reg
                 | Rep Reg

        accept :: Reg -> Str -> Bool
        accept Eps       u = null u
        accept (Sym c)   u = u == singleton c
        accept (Alt p q) u = accept p u || accept q u
        accept (Seq p q) u =
            or [accept p u1 && accept q u2 | (u1, u2) <- splits u]
        accept (Rep r) u =
            or [and [accept r ui | ui <- ps] | ps <- parts u]

```

（为了简单起见，现在我还没有将`Char`参数化。）

### 实例化这个函数（String）

这一切都很好，但我们实际上不能运行这段代码，因为没有`Str`的实现。让我们写一个新单元，提供一个模块，其中包含所有这些类型和函数的实现，使用`String`，将旧的`splits`和`parts`实现复制粘贴进来：

```
unit str-string where
    module Str where
        import Prelude hiding (null)
        import qualified Prelude as P

        type Str = String

        null :: Str -> Bool
        null = P.null

        singleton :: Char -> Str
        singleton c = [c]

        splits :: Str -> [(Str, Str)]
        splits [] = [([], [])]
        splits (c:cs) = ([], c:cs):[(c:s1,s2) | (s1,s2) <- splits cs]

        parts :: Str -> [[Str]]
        parts [] = [[]]
        parts [c] = [[[c]]]
        parts (c:cs) = concat [[(c:p):ps, [c]:p:ps] | p:ps <- parts cs]

```

当为函数编写 Backpack 实现时，一个怪癖是 Backpack 在多态函数上不执行子类型匹配，因此你不能使用多态函数`Traversable t => t a -> Bool`实现`Str -> Bool`（添加这个将是一个有趣的扩展，但并不是完全平凡的）。所以我们必须写一个稍微增加阻抗匹配的绑定，将`null`单态化到预期的类型。

为了用`str-string:Str`实例化`regex-indef`，我们在`main`中修改了依赖项：

```
-- dependency regex -- old
dependency regex-indef[Str=str-string:Str]

```

Backpack 文件要求显式指定实例化（这与 Cabal 文件不同，后者使用混合链接来确定实例化）。在这种情况下，实例化指定`regex-indef`的名为`Str`的签名应由`str-string`中的`Str`模块填充。

进行这些更改后，运行`ghc --backpack`；你应该会得到一个完全相同的结果。

### 实例化这个函数（ByteString）

参数化 `regex` 的整个目的是使我们能够有第二个 `Str` 的实现。所以让我们继续编写一个 `bytestring` 实现。经过一点工作，你可能最终得到这个：

```
unit str-bytestring where
    module Str(module Data.ByteString.Char8, module Str) where
        import Prelude hiding (length, null, splitAt)
        import Data.ByteString.Char8
        import Data.ByteString

        type Str = ByteString

        splits :: Str -> [(Str, Str)]
        splits s = fmap (\n -> splitAt n s) [0..length s]

        parts :: Str -> [[Str]]
        parts s | null s    = [[]]
                | otherwise = do
                    n <- [1..length s]
                    let (l, r) = splitAt n s
                    fmap (l:) (parts r)

```

关于这个实现，有两点需要注意：

1.  与 `str-string` 不同，它在其模块体中显式定义了每个所需的方法，`str-bytestring` 通过重新导出来自 `Data.ByteString.Char8` 的所有实体（适当地单态化）来提供 `null` 和 `singleton`。我们聪明地选择了我们的命名，以符合现有字符串包的命名约定！

1.  我们的 `splits` 和 `parts` 的实现比原始的 `String` 实现中的 consing 和 unconsing 要优化得多。我经常听到人们说 `String` 和 `ByteString` 的性能特性非常不同，因此你不应该在同一个实现中混合它们。我认为这个例子表明，只要你对字符串有足够高级的操作，这些性能差异最终会平滑化；并且仍然有相当大的代码块可以在不同的实现之间重用。

要使用 `bytestring-string:Str` 实例化 `regex-indef`，我们再次修改 `main` 中的依赖项：

```
-- dependency regex -- oldest
-- dependency regex-indef[Str=str-string:Str] -- old
dependency regex-indef[Str=str-bytestring:Str]

```

我们还需要粘贴 `{-# LANGUAGE OverloadedStrings #-}` 命令，以便将 `"acc"` 解释为 `ByteString`（不幸的是，`bkp` 文件格式仅支持适用于所有定义的模块的语言命令，因此将此命令放在文件顶部）。但除此之外，一切都按预期工作！

### 同时使用两个实例

没有任何阻碍我们同时使用 `regex-indef` 的两个实例，只需取消注释两个 `dependency` 声明，除了每个依赖项提供的模块名称之间冲突且不明确外。因此，Backpack 文件为模块提供了*重命名*语法，让你为每个导出的模块指定一个不同的名称：

```
dependency regex-indef[Str=str-string:Str]     (Regex as Regex.String)
dependency regex-indef[Str=str-bytestring:Str] (Regex as Regex.ByteString)

```

我们应该如何修改 `Main` 来在 `String` 和 `ByteString` 上运行我们的正则表达式？但是 `Regex.String.Reg` 和 `Regex.ByteString.Reg` 是一样的吗？编译器的快速查询将揭示它们*不*是一样的。这是因为 Backpack 的类型标识规则：所有在一个单元中定义的类型的标识都取决于*所有*签名的实例化方式，即使该类型实际上并不依赖于来自签名的任何类型。如果我们希望只有一个 `Reg` 类型，我们将不得不从 `reg-indef` 中提取它，并为它单独创建一个单元，*没有*签名。

重构后，这是最终的完整程序：

```
{-# LANGUAGE OverloadedStrings #-}

unit str-bytestring where
    module Str(module Data.ByteString.Char8, module Str) where
        import Prelude hiding (length, null, splitAt)
        import Data.ByteString.Char8
        import Data.ByteString

        type Str = ByteString

        splits :: Str -> [(Str, Str)]
        splits s = fmap (\n -> splitAt n s) [0..length s]

        parts :: Str -> [[Str]]
        parts s | null s    = [[]]
                | otherwise = do
                    n <- [1..length s]
                    let (l, r) = splitAt n s
                    fmap (l:) (parts r)

unit str-string where
    module Str where
        import Prelude hiding (null)
        import qualified Prelude as P

        type Str = String

        null :: Str -> Bool
        null = P.null

        singleton :: Char -> Str
        singleton c = [c]

        splits :: Str -> [(Str, Str)]
        splits [] = [([], [])]
        splits (c:cs) = ([], c:cs):[(c:s1,s2) | (s1,s2) <- splits cs]

        parts :: Str -> [[Str]]
        parts [] = [[]]
        parts [c] = [[[c]]]
        parts (c:cs) = concat [[(c:p):ps, [c]:p:ps] | p:ps <- parts cs]

unit regex-types where
    module Regex.Types where
        data Reg = Eps
                 | Sym Char
                 | Alt Reg Reg
                 | Seq Reg Reg
                 | Rep Reg

unit regex-indef where
    dependency regex-types
    signature Str where
        data Str
        instance Eq Str
        null :: Str -> Bool
        singleton :: Char -> Str
        splits :: Str -> [(Str, Str)]
        parts :: Str -> [[Str]]
    module Regex where
        import Prelude hiding (null)
        import Str
        import Regex.Types

        accept :: Reg -> Str -> Bool
        accept Eps       u = null u
        accept (Sym c)   u = u == singleton c
        accept (Alt p q) u = accept p u || accept q u
        accept (Seq p q) u =
            or [accept p u1 && accept q u2 | (u1, u2) <- splits u]
        accept (Rep r) u =
            or [and [accept r ui | ui <- ps] | ps <- parts u]

unit main where
    dependency regex-types
    dependency regex-indef[Str=str-string:Str]     (Regex as Regex.String)
    dependency regex-indef[Str=str-bytestring:Str] (Regex as Regex.ByteString)
    module Main where
        import Regex.Types
        import qualified Regex.String
        import qualified Regex.ByteString
        nocs = Rep (Alt (Sym 'a') (Sym 'b'))
        onec = Seq nocs (Sym 'c')
        evencs = Seq (Rep (Seq onec onec)) nocs
        main = print (Regex.String.accept evencs "acc") >>
               print (Regex.ByteString.accept evencs "acc")

```

### 还有更多！

继续阅读下一篇博客文章，[尝试 Backpack：Cabal packages](http://blog.ezyang.com/2017/01/try-backpack-cabal-packages/)，我将告诉你如何将这个原型转化为一组 Cabal packages 中的 `bkp` 文件。

**后记。** 如果你感到冒险的话，尝试进一步参数化`regex-types`，使其不再将`Char`硬编码为元素类型，而是某种任意的元素类型`Elem`。了解到，你可以使用语法`dependency regex-indef[Str=str-string:Str,Elem=str-string:Elem]`来实例化多个签名，而且如果你依赖一个带有签名的包，你必须通过使用语法`dependency regex-types[Elem=<Elem>]`来传递该签名。如果这听起来用户不友好，那就是真的！这就是为什么在 Cabal 包的宇宙中，实例化是*隐式*完成的，使用混合链接。
