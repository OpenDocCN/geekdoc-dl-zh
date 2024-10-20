<!--yml

category: 未分类

date: 2024-07-01 18:17:04

-->

# Haskell 的字符串理论基础：ezyang 的博客

> 来源：[`blog.ezyang.com/2016/09/the-base-of-a-string-theory-for-haskell/`](http://blog.ezyang.com/2016/09/the-base-of-a-string-theory-for-haskell/)

这个博客的早期文章之一，来自 2010 年，是关于[Haskell 中如何选择你的字符串库](http://blog.ezyang.com/2010/08/strings-in-haskell/)的主题。半个世纪后，Haskell 生态系统在很大程度上仍处于与半个世纪前相同的情况下，大部分与 GHC 一起发货的引导库（例如，`base`）仍然使用`String`类型，尽管存在更优秀的字符串类型。问题是双重的：

1.  没有人想要破坏所有现有的代码，这意味着像`base`这样的库必须保持对所有代码的`String`版本。你不能只是搜索替换每个`String`出现的地方为`Text`。

1.  没有人想要维护*两个*代码副本，它们彼此复制粘贴但略有不同。在实践中，我们必须：例如，[unix](https://hackage.haskell.org/package/unix)有所有函数的`ByteString`变体（通过复制粘贴完成）；[text](https://hackage.haskell.org/package/text)提供了一些核心 IO 功能（同样是通过复制粘贴完成）。但这非常糟糕且扩展性差：现在每个下游库想要支持两种或更多的字符串类型都必须发布两个自己的副本，并且任何新的字符串实现都必须重新实现整个世界以使自己有用。

Backpack 通过允许您对签名进行*参数化*而不是对字符串类型的具体实现进行实例化来解决这些问题。这解决了这两个问题：

1.  因为你可以随时实例化一个不定的库，我们可以急切地使用`String`来实例化`posix-indef`，并将其作为`posix`发布，以保持对所有不了解 Backpack 的包的向后兼容性。

1.  与此同时，如果包直接依赖于`posix-indef`，它们本身可以对字符串类型进行参数化。整个库生态系统可以将字符串类型的选择推迟到最终用户，在 GHC 的足够新版本上，这为库添加对新字符串类型的支持提供了向后兼容的方式。（我不想说，支持*多种*字符串类型，因为这本身并不一定是一种优点。）

为此，我想提出一个字符串理论，用于 GHC Haskell 的基础：即今天随 GHC 分发的核心引导库。这些包将为最终使生态系统的其余部分 Backpack 化设定基调。

但首先，我们正在对什么进行参数化？字符串并不那么简单...

### 一个关于文件路径（和操作系统字符串）的离题讨论

文件路径（`FilePath`）是一种重要的 `String` 形式，实际上并不是 Unicode 字符串。POSIX 指定文件路径可以是任意的 C 字符串，因此，解码文件路径为 Unicode 的代码必须意识到底层的 `ByteString` 可能包含任意的、无法解码的无意义字符。更糟糕的是，甚至编码也可能不同：在 Windows 中，文件路径编码为 UTF-16（可能存在未配对的代理项），而在现代 Linux 环境中，编码由区域设置决定（`base` 使用 `locale_charset` 来确定如何解释文件路径；区域设置通常是 UTF-8，但不总是）。

因此，定义 `type FilePath = String` 实际上是非常值得怀疑的。已经有一个现有的提案，即 [抽象 FilePath 提案](https://ghc.haskell.org/trac/ghc/wiki/Proposal/AbstractFilePath)，将 `FilePath` 转换为抽象类型，而不仅仅是 `String` 的类型同义词。不幸的是，这样的改变会破坏向后兼容性，因此需要一些时间来实现，因为 GHC 必须首先被教导在 `FilePath` 被错误使用时发出警告，以帮助人们发现他们的错误用法。

Backpack 提供了一种更加分散的方式来迎接未来：只需定义一个*抽象签名*，让 `FilePath` 依赖于它。低级别的签名可能看起来像这样：

```
signature FilePath where

-- | File and directory names, whose precise
-- meaning is operating system dependent. Files can be opened, yielding a
-- handle which can then be used to operate on the contents of that file.
data FilePath

-- | A C string (pointer to an array of C characters terminated by NUL)
-- representing a file path, suitable for use with the operating system
-- C interface for file manipulation.  This exact type is architecture
-- dependent.
type CFilePath =
#ifdef mingw32_HOST_OS
        CWString
#else
        CString
#endif

withFilePath :: FilePath -> (CFilePath -> IO a) -> IO a
newFilePath  :: FilePath -> IO CFilePath
peekFilePath :: CFilePath -> IO FilePath
-- peekFilePath >=> newFilePath should be identity
-- (this is tricky to achieve if FilePath is a
-- Unicode-based type, like String)

```

当然，你会希望所有的 `FilePath` [操作函数](https://hackage.haskell.org/package/filepath-1.4.1.0/docs/System-FilePath-Posix.html) 都能被人们使用。

为了与现有生态系统保持兼容性，你可能会用 `type FilePath = String` 来实例化你的库。但是没有什么可以阻止你选择自己的抽象 `FilePath` 类型并使用它。

从这个意义上说，文件路径并不是唯一的；还有其他具有类似属性的字符串（例如环境变量的值）：我习惯称之为 [OSStrings](https://doc.rust-lang.org/std/ffi/struct.OsString.html)（就像在 Rust 中称呼它们一样）。

### 参数化的轴

考虑到这一点，任何给定的库都可以参数化为三种“字符串变体”：

1.  它们可以参数化为 `FilePath`，适用于处理文件系统的模块（例如，[System.Posix.Directory](https://hackage.haskell.org/package/unix-2.7.2.0/docs/System-Posix-Directory.html)）

1.  它们可以参数化为 `OSString`，因为它们涉及各种操作系统特定的 API（例如，[System.Posix.Env](https://hackage.haskell.org/package/unix-2.7.2.0/docs/System-Posix-Env.html)）

1.  它们可以参数化为 `String`，因为有时候一个字符串就是一个字符串。（例如，[Text.ParserCombinators.ReadP](https://hackage.haskell.org/package/base-4.9.0.0/docs/Text-ParserCombinators-ReadP.html)）

有些库可能以多种方式进行参数化：例如，`readFile` 需要同时参数化 `FilePath` 和 `String`。

### 为 Backpack 拆分 base（和友好组件）

由于技术原因，Backpack 不能用于对特定*模块*进行参数化；必须对整个库进行参数化。因此，Backpack 化核心库的副作用是它们将被拆分为多个较小的库。使用模块重导出，您仍然可以保留旧库作为 shims。

有四个 GHC 引导库将从对字符串的模块化中获益最多：

+   [base](https://hackage.haskell.org/package/base)

    +   `base-io`（`System.IO` 及其子模块；参数化为 `FilePath` 和 `String`）

    +   还有一些其他模块可以字符串化，但边际效益可能不足以为每个模块制作一个新包（`Data.String`、`System.Console.GetOpt`、`Text.ParserCombinators.ReadP`、`Text.Printf`）。每个模块只需要参数化为 String。

    +   `Control.Exception`、`Text.Read` 和 `Text.Show` 是明确的*非目标*，它们目前与 GHC 的深层次连接太紧密，因此不太可能更改。

+   [unix](https://hackage.haskell.org/package/unix)

    +   `unix-env`（`System.Posix.Env`，参数化为 `OSString`）

    +   `unix-fs`（`System.Posix.Directory`、`System.Posix.Files`、`System.Posix.Temp`，参数化为 `FilePath`）

    +   `unix-process`（`System.Posix.Process`，参数化为 `FilePath` 和 `OSString`）

+   [pretty](https://hackage.haskell.org/package/pretty)（参数化为 String；然后 GHC 可以使用它而不是自己制作副本！）

+   [process](https://hackage.haskell.org/package/process)（参数化为 String、OSString 和 FilePath）

我提议的命名方案是，例如，`unix` 包继续使用传统的字符串进行实例化。然后 `unix-indef` 是一个未实例化的包（用户可以根据需要进行实例化，或者将决策传递给他们的用户）。一些包可能会选择还提供其包的 shims，这些 shims 使用特定类型进行实例化，例如 `base-io-bytestring`，它将使用 `ByteString` 而不是 `String` 进行 `base-io` 的实例化，尽管这些名称可能会变得相当长，所以不确定这对你有多大用处。
