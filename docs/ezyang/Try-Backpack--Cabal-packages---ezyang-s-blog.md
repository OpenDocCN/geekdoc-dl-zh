<!--yml

category: 未分类

日期：2024-07-01 18:17:03

-->

# 尝试 Backpack：Cabal 包：ezyang’s 博客

> 来源：[`blog.ezyang.com/2017/01/try-backpack-cabal-packages/`](http://blog.ezyang.com/2017/01/try-backpack-cabal-packages/)

本文是关于如何尝试使用 Backpack，一个新的 Haskell 混合包系统的系列文章的第二部分。在[上一篇文章](http://blog.ezyang.com/2016/10/try-backpack-ghc-backpack/)中，我们描述了如何使用 GHC 的新`ghc --backpack`模式快速尝试 Backpack 的新签名特性。不幸的是，目前没有办法将输入文件分发到这种模式作为 Hackage 上的包。因此，在本文中，我们将介绍如何组装具有相同功能的等效 Cabal 包。

### GHC 8.2，cabal-install 2.0

在开始本教程之前，您需要确保您已经安装了最新版本的[GHC 8.2](https://ghc.haskell.org/trac/ghc/blog/ghc-8.2.11-released)和[cabal-install 2.0](https://www.haskell.org/cabal/download.html)。当它们更新后，您应该看到：

```
ezyang@sabre:~$ ghc-8.2 --version
The Glorious Glasgow Haskell Compilation System, version 8.2.1
ezyang@sabre:~$ /opt/cabal/2.0/bin/cabal --version
cabal-install version 2.0.0.0
compiled using version 2.0.0.2 of the Cabal library

```

### 我们的目标

这是我们在上篇文章中开发的代码摘录，我已经删除了所有的模块/签名内容：

```
unit str-bytestring where
    module Str

unit str-string where
    module Str

unit regex-types where
    module Regex.Types

unit regex-indef where
    dependency regex-types
    signature Str
    module Regex

unit main where
    dependency regex-types
    dependency regex-indef[Str=str-string:Str]     (Regex as Regex.String)
    dependency regex-indef[Str=str-bytestring:Str] (Regex as Regex.ByteString)
    module Main

```

将此文件翻译为 Cabal 包的一种明显方法是定义每个单元的包。然而，我们也可以定义一个包含许多*内部库*的单个包——这是一种独立于 Backpack 的新功能，允许您在单个包内定义私有辅助库。由于这种方法涉及的模板代码较少，我们将在将库“生产化”为单独的包之前首先描述它。

对于所有这些示例，我们假设模块和签名的源代码已经复制粘贴到适当的 `hs` 和 `hsig` 文件中。您可以在[backpack-regex-example 的 source-only 分支](https://github.com/ezyang/backpack-regex-example/tree/source-only)中找到这些文件。

### Single package layout

在本节中，我们将逐步介绍将每个单元定义为内部库的 Cabal 文件。您可以在[backpack-regex-example 的 single-package 分支](https://github.com/ezyang/backpack-regex-example/tree/single-package)找到此版本的所有文件。此包可以使用传统的 `cabal configure -w ghc-8.2`（将 `ghc-8.2` 替换为 GHC 8.2 安装路径，或者如果 `ghc` 已经是 GHC 8.2，则省略它）构建，然后进行 `cabal build`。

包文件的标题非常普通，但由于 Backpack 使用了新的 Cabal 功能，`cabal-version` 必须设置为 `>=1.25`（请注意，Backpack 不支持 `Custom` 设置）：

```
name:                regex-example
version:             0.1.0.0
build-type:          Simple
cabal-version:       >=1.25

```

**私有库**。`str-bytestring`，`str-string` 和 `regex-types` 都是完全传统的 Cabal 库，只包含模块。在早期的 Cabal 版本中，我们需要为它们中的每一个制作一个包。然而，通过私有库，我们可以简单地列出多个带有库内部名称注释的库段：

```
library str-bytestring
  build-depends:       base, bytestring
  exposed-modules:     Str
  hs-source-dirs:      str-bytestring

library str-string
  build-depends:       base
  exposed-modules:     Str
  hs-source-dirs:      str-string

library regex-types
  build-depends:       base
  exposed-modules:     Regex.Types
  hs-source-dirs:      regex-types

```

为了保持每个内部库的模块分开，我们为每个给出了一个不同的 `hs-source-dirs`。这些库可以在此包内部依赖，但对外部客户端是隐藏的；只有 *公共库*（用无名称的 `library` 段表示）是公开可见的。

**不定库。** `regex-indef` 稍有不同，因为它有一个签名。但编写它的库并不完全不同：签名放在名为 `signatures` 的适当命名的字段中：

```
library regex-indef
  build-depends:       base, regex-types
  signatures:          Str
  exposed-modules:     Regex
  hs-source-dirs:      regex-indef

```

**实例化。** 我们如何实例化 `regex-indef`？在我们的 `bkp` 文件中，我们必须明确指定如何填写包的签名：

```
dependency regex-indef[Str=str-string:Str]     (Regex as Regex.String)
dependency regex-indef[Str=str-bytestring:Str] (Regex as Regex.ByteString)

```

使用 Cabal，这些实例化可以通过更间接的 *mix-in linking* 过程来指定，其中一个包的依赖关系被 "混合在一起"，一个依赖的所需签名被另一个依赖的暴露模块填充。在编写 `regex-example` 可执行文件之前，让我们编写一个 `regex` 库，它类似于 `regex-indef`，但专门用于 `String`：

```
library regex
  build-depends:       regex-indef, str-string
  reexported-modules:  Regex as Regex.String

```

这里，`regex-indef` 和 `str-string` 通过 *mix-in linking* 混合链接在一起：来自 `str-string` 的 `Str` 模块填充了 `regex-indef` 的 `Str` 要求。然后，这个库重新导出 `Regex`，并使用新名称以明确表示它是 `String` 的实例化。

我们可以轻松地为 `regex-indef` 的 `ByteString` 实例化版本做同样的事情：

```
library regex-bytestring
  build-depends:       regex-indef, str-bytestring
  reexported-modules:  Regex as Regex.ByteString

```

**将所有这些联系起来。** 添加可执行文件非常简单，然后构建代码：

```
executable regex-example
  main-is:             Main.hs
  build-depends:       base, regex, regex-bytestring, regex-types
  hs-source-dirs:      regex-example

```

在包的根目录下，您可以使用 `cabal configure; cabal build` 来构建包（确保您传递了 `-w ghc-head`！）。或者，您可以使用 `cabal new-build` 以同样的效果。

### 有多种方法可以做到这一点

在前面的代码示例中，我们使用 `reexported-modules` 在 *声明时间* 重命名模块，以避免它们互相冲突。但是，这仅在我们创建了额外的 `regex` 和 `regex-bytestring` 库时才可能。在某些情况下（尤其是如果我们实际上正在创建新的包而不是内部库），这可能会非常麻烦，因此 Backpack 提供了一种在 *使用时间* 重命名模块的方式，使用 `mixins` 字段。它的工作方式如下：在 `build-depends` 中声明的任何包可以在 `mixins` 中指定，使用显式的重命名，指定应该将哪些模块引入作用域，并使用什么名称。

例如，`str-string` 和 `str-bytestring` 都导出一个名为 `Str` 的模块。为了不使用包限定的导入来引用这两个模块，我们可以如下重命名它们：

```
executable str-example
  main-is:             Main.hs
  build-depends:       base, str-string, str-bytestring
  mixins:              str-string     (Str as Str.String),
                       str-bytestring (Str as Str.ByteString)
  hs-source-dirs:      str-example

```

`mixins` 字段的语义是我们仅将导入规范中明确列出的模块（`Str as Str.String`）引入到导入范围内。如果一个包在 `mixins` 中从不出现，则默认将所有模块引入范围内（给出 `build-depends` 的传统行为）。这确实意味着，如果你说 `mixins: str-string ()`，你可以强制一个组件依赖于 `str-string`，但不会引入其任何模块。

有人认为包作者应避免定义具有[冲突模块名称](http://www.snoyman.com/blog/2017/01/conflicting-module-names)的包。因此，假设我们重构 `str-string` 和 `str-bytestring` 以具有唯一的模块名称：

```
library str-string
  build-depends:       base
  exposed-modules:     Str.String
  hs-source-dirs:      str-string

library str-bytestring
  build-depends:       base, bytestring
  exposed-modules:     Str.ByteString
  hs-source-dirs:      str-bytestring

```

然后我们需要重写 `regex` 和 `regex-bytestring`，将 `Str.String` 和 `Str.ByteString` 重命名为 `Str`，以填补 `regex-indef` 的空缺：

```
library regex
  build-depends:       regex-indef, str-string
  mixins:              str-string (Str.String as Str)
  reexported-modules:  Regex as Regex.String

library regex-bytestring
  build-depends:       regex-indef, str-bytestring
  mixins:              str-bytestring (Str.ByteString as Str)
  reexported-modules:  Regex as Regex.ByteString

```

实际上，通过 `mixins` 字段，我们可以完全避免定义 `regex` 和 `regex-bytestring` 的外壳库。我们可以通过在 `mixins` 中两次声明 `regex-indef`，分别重命名其要求来做到这一点：

```
executable regex-example
  main-is:             Main.hs
  build-depends:       base, regex-indef, str-string, str-bytestring, regex-types
  mixins:              regex-indef (Regex as Regex.String)
                          requires (Str as Str.String),
                       regex-indef (Regex as Regex.ByteString)
                          requires (Str as Str.ByteString)
  hs-source-dirs:      regex-example

```

这个特定示例的完整代码在[backpack-regex-example 的更好单包分支](https://github.com/ezyang/backpack-regex-example/tree/better-single-package)中给出。

注意，要求的重命名在语法上由 `requires` 关键字引导。

编写 Backpack 包的艺术仍处于起步阶段，因此尚不清楚最终会采用哪些约定。但这是我的建议：在定义意图实现签名的模块时，遵循现有的无冲突模块名称约定。但是，将您的模块重新导出到签名名称。这个技巧利用了 Cabal 只有在实际使用时才会报告模块冗余的事实。所以，假设我们有：

```
library str-string
  build-depends:       base
  exposed-modules:     Str.String
  reexported-modules:  Str.String as Str
  hs-source-dirs:      str-string

library str-bytestring
  build-depends:       base, bytestring
  exposed-modules:     Str.ByteString
  reexported-modules:  Str.ByteString as Str
  hs-source-dirs:      str-bytestring

```

现在所有以下组件都可以工作：

```
library regex
  build-depends:       regex-indef, str-string
  reexported-modules:  Regex as Regex.String

library regex-bytestring
  build-depends:       regex-indef, str-bytestring
  reexported-modules:  Regex as Regex.ByteString

-- "import Str.String" is unambiguous, even if "import Str" is
executable str-example
  main-is:             Main.hs
  build-depends:       base, str-string, str-bytestring
  hs-source-dirs:      str-example

-- All requirements are renamed away from Str, so all the
-- instantiations are unambiguous
executable regex-example
  main-is:             Main.hs
  build-depends:       base, regex-indef, str-string, str-bytestring, regex-types
  mixins:              regex-indef (Regex as Regex.String)
                          requires (Str as Str.String),
                       regex-indef (Regex as Regex.ByteString)
                          requires (Str as Str.ByteString)
  hs-source-dirs:      regex-example

```

### 独立的包

好的，那么我们如何将其扩展成一个无限制包的生态系统，每个包都可以单独使用并由不同的个人维护呢？库模块基本与上述相同；只需为每个模块创建一个独立的包。不再在此复制所有样板内容，完整的源代码可在[backpack-regex-example 的多包分支](https://github.com/ezyang/backpack-regex-example/tree/multiple-packages)中找到。

有一个重要的陷阱：包管理器需要知道如何实例化和构建这些 Backpack 包（在单个包情况下，智能完全封装在 `Cabal` 库中）。截至目前，唯一知道如何做到这一点的命令是 `cabal new-build`（我计划最终支持 `stack`，但要在完成论文后才会，而且我不打算永远支持旧式的 `cabal install`。）

幸运的是，使用`cabal new-build`构建`regex-example`非常简单；只需说`cabal new-build -w ghc-head regex-example`。完成！

### 结论

如果你真的想要真正地使用 Backpack，你可以做什么？有几种可能性：

1.  如果你只想使用 GHC 8.2，并且只需要在内部参数化代码（其中公共库看起来像普通的非 Backpack 包）时，使用内部库与 Backpack 非常合适。生成的包可以使用 Stack 和 cabal-install 构建，只要你使用的是 GHC 8.2。这可能是你能够实际应用 Backpack 的最实用方式；主要问题是 Haddock 不知道如何处理[重新导出的模块](https://github.com/haskell/haddock/issues/563)，但这应该可以解决。

1.  如果你只想使用`cabal new-build`，那么你也可以编写有要求的包，并让客户决定如何实现他们的包。

除了潜在的任何潜在错误外，实际世界中使用 Backpack 的最大障碍可能是对 Haddock 的支持不足。但如果你愿意暂时忽略这一点，请试试看！
