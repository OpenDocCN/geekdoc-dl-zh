<!--yml

category: 未分类

date: 2024-07-01 18:17:14

-->

# 一个 Cabal 化 Backpack 的体验：ezyang 的博客

> 来源：[`blog.ezyang.com/2014/08/a-taste-of-cabalized-backpack/`](http://blog.ezyang.com/2014/08/a-taste-of-cabalized-backpack/)

**更新。** 想了解更多关于 Backpack 的信息？阅读[规范](https://github.com/ezyang/ghc-proposals/blob/backpack/proposals/0000-backpack.rst)

所以也许你已经[接受了模块和模块化](http://blog.ezyang.com/2014/08/whats-a-module-system-good-for-anyway/)，并希望立即开始使用 Backpack。你如何做到？在这篇博文中，我想给出一个教程风格的 Cabal 编程示例。这些示例是可执行的，但你需要构建自定义版本的[GHC](https://github.com/ezyang/ghc/tree/ghc-backpack)和[Cabal](https://github.com/ezyang/cabal/tree/backpack)来构建它们。非常感谢您的评论和建议；尽管这里的设计在理论上是基础良好的，但由于显而易见的原因，我们还没有太多实际的程序员反馈。

* * *

### 在今天的 Cabal 中，有一个简单的包

首先，让我们简要回顾一下 Haskell 模块和 Cabal 包如何工作。我们的运行示例将是`bytestring`包，尽管我会内联、简化和省略定义以增强清晰度。

假设你正在编写一个库，你想为一些二进制处理使用高效的紧凑字符串。幸运的是，著名的 Don Stewart 已经为你编写了一个`bytestring`包，为你实现了这个功能。这个包包含几个模块：一个严格`ByteStrings`的实现...

```
module Data.ByteString(ByteString, empty, singleton, ...) where
  data ByteString = PS !(ForeignPtr Word8) !Int !Int
  empty :: ByteString
  empty = PS nullForeignPtr 0 0
  -- ...

```

...和一个惰性`ByteStrings`的实现：

```
module Data.ByteString.Lazy(ByteString, empty, singleton, ...) where
  data ByteString = Empty | Chunk !S.ByteString ByteString
  empty :: ByteString
  empty = Empty
  -- ...

```

这些模块被打包成一个包，并使用 Cabal 文件指定：

```
name: bytestring
version: 0.10.4.0
library
  build-depends: base >= 4.2 && < 5, ghc-prim, deepseq
  exposed-modules: Data.ByteString, Data.ByteString.Lazy, ...
  other-modules: ...

```

接着我们可以创建一个简单的模块和依赖于`bytestring`包的包：

```
module Utils where
  import Data.ByteString.Lazy as B
  blank :: IO ()
  blank = B.putStr B.empty

```

```
name: utilities
version: 0.1
library
  build-depends: base, bytestring >= 0.10
  exposed-modules: Utils

```

关于这个完全标准的模块设置，值得注意的几点：

1.  不能简单地将`Utils`从使用惰性`ByteStrings`切换到严格`ByteStrings`，除非直接编辑`Utils`模块。即使如此，也不能使`Utils`依赖于严格`ByteString`和惰性`ByteString`的同一程序，而不复制整个模块文本。（这并不令人太惊讶，因为代码*确实*不同。）

1.  尽管在这里有一些间接性：虽然`Utils`包括一个特定的`ByteString`模块，但未指定会使用*哪个*版本的`ByteString`。假如（假设性地），`bytestring`库发布了一个新版本，其中惰性字节字符串实际上是严格的，那么当用户重新运行依赖解析时，`Utils`的功能将相应改变。

1.  我使用了限定导入来引用`Data.ByteString.Lazy`中的标识符。这在开发 Haskell 代码时是一种相当常见的模式：我们将`B`视为实际模型的*别名*。从文本上讲，这也是有帮助的，因为它意味着我只需要编辑导入语句即可更改我引用的`ByteString`。

* * *

### 通过签名泛化 Utils

要通过一些 Backpack 魔法泛化`Utils`，我们需要为`ByteString`创建一个*签名*，指定提供`ByteStrings`模块的接口。这里是一个这样的签名，它放置在`utilities`包内的文件`Data/ByteString.hsig`中：

```
signature Data.ByteString where
  import Data.Word
  data ByteString
  instance Eq ByteString
  empty :: ByteString
  singleton :: Word8 -> ByteString
  putStr :: ByteString -> IO ()

```

签名的格式本质上与`hs-boot`文件相同：我们有普通的 Haskell 声明，但省略了值的实际实现。

`utilities`包现在需要一个新字段来记录签名：

```
name: utilities
library
  build-depends: base
  exposed-modules: Utils
  signatures: Data.ByteString

```

注意，这里发生了三个变化：(1) 我们移除了对`bytestring`包的直接依赖，和 (2) 我们新增了一个名为**签名**的字段，其中简单列出了我们需要填充的签名文件（也称为**空洞**）的名称。

那么我们实际上如何使用 utilities 包呢？假设我们的目标是生成一个新模块`Utils.Strict`，它是`Utils`，但使用严格的`ByteStrings`（这是由 bytestring 包在模块名`Data.ByteString`下导出的）。为此，我们需要创建一个新的包：

```
name: strict-utilities
library
  build-depends: utilities, bytestring
  reexported-modules: Utils as Utils.Strict

```

就是这样！`strict-utilities`导出了一个单一模块`Utils.Strict`，它使用了来自`bytestring`的`Data.ByteString`（这是其严格实现）。这被称为*混合*：在相同的依赖列表中，我们简单地混合在一起：

+   `utilities`，它*要求*一个名为`Data.ByteString`的模块，并且

+   `bytestring`提供了一个名为`Data.ByteString`的模块。

Cabal 会自动找出如何通过匹配*模块名称*来实例化 utilities 包。具体而言，上述两个包通过模块名`Data.ByteString`连接在一起。这使得包实例化变得非常方便（事实证明，也很表达）。顺便说一句，**reexported-modules**是一个新的（正交的）特性，它允许我们重新导出一个模块，从当前包或依赖关系到外部世界，使用不同的名称。导出的模块和重新导出的模块区分开来是为了明确哪些模块在包中有源代码（exposed-modules）。

不寻常的是，`strict-utilities`是一个不包含任何代码的包！它的唯一目的是混合现有的包。

现在，你可能会想：我们如何使用懒惰的`ByteString`实现来实例化 utilities 呢？该实现放在了`Data.ByteString.Lazy`中，因此名称不匹配。在这种情况下，我们可以使用另一个新特性，即模块瘦身和重命名：

```
name: lazy-utilities
library
  build-depends: utilities, bytestring
  backpack-includes:
    bytestring (Data.ByteString.Lazy as Data.ByteString)
  reexported-modules: Utils as Utils.Lazy

```

新的`backpack-includes`字段表示只应该将`Data.ByteString.Lazy`模块引入到范围内，使用名称`Data.ByteString`。这足以将`utilities`与`ByteString`的延迟实现混合链接。

有趣的二元性在于，你可以以另一种方式进行重命名：

```
name: lazy-utilities
library
  build-depends:
    utilities (Utils, Data.ByteString as Data.ByteString.Lazy),
    bytestring

```

我没有重命名实现，而是重命名了洞！这是等效的：重要的是，签名和实现需要在*同一*名称下混合，以便进行链接（签名与实现的实例化）。

有几点关于签名的使用需要注意：

1.  如果你正在使用一个签名，那么在导入它时指定显式导入列表没有太大意义：你保证*只能*看到签名中的类型和定义（除了类型类... 这是另一个话题）。签名文件就像一个类型安全的导入列表，你可以跨模块共享它。

1.  一个签名可以（而且通常必须）导入其他模块。在`Data/ByteString.hsig`中`singleton`的类型签名中，我们需要引用`Word8`类型，因此必须通过导入`Data.Word`来将其引入范围内。

    现在，当我们编译`utilities`包中的签名时，我们需要知道`Data.Word`来自哪里。它可能来自另一个签名，但在这种情况下，它由*明确*的包基础提供：它是一个具有实现的适当的具体模块！签名可以依赖于实现：由于我们只能引用那些模块中的类型，实际上我们在说：`singleton`函数的任何实现和`ByteString`类型的任何表示都是可以接受的，但是关于`Word8`，你必须使用`prelude`中来自`Data.Word`的*特定*类型。

1.  如果，独立于我的`strict-utilities`包，其他人也用`Data.ByteString`实例化了`utilities`，会发生什么？背包足够聪明，可以重复使用`utilities`的实例化：这个属性称为模块系统的**适用性**。我们用来决定实例化是否相同的具体规则是看所有*包*所需的所有洞是如何实例化的，如果它们用完全相同的模块实例化，那么实例化的包被认为是类型相等的。因此，实际上不需要创建`strict-utilities`或`lazy-utilities`：你可以随时在现场实例化`utilities`。

**迷你测验：** 这个软件包做什么？

```
name: quiz-utilities
library
  build-depends:
    utilities (Utils, Data.ByteString as B),
    bytestring (Data.ByteString.Lazy as B)

```

* * *

### 共享签名

能够为`Data.ByteString`显式编写签名非常好，但是如果我必须为每个我依赖的软件包都这样做，那会很烦人。如果我能够将所有签名放在一个包中，并在需要时包含它，那会更好。我希望所有 Hackage 机制都适用于我的签名以及我的普通软件包（例如版本控制）。好吧，你可以！

`bytestring` 的作者可以编写一个 `bytestring-sig` 包，其中只包含签名：

```
name: bytestring-sig
version: 1.0
library
  build-depends: base
  signatures: Data.ByteString

```

现在，`utilities` 可以包含这个包来指示它对签名的依赖：

```
name: utilities
library
  build-depends: base, bytestring-sig-1.0
  exposed-modules: Utils

```

与普通依赖不同，签名依赖应该是*精确的*：毕竟，虽然你可能想要一个升级的实现，但你不希望签名随意更改！

我们可以总结所有字段如下：

1.  **exposed-modules** 表示在本包中定义了一个公共模块。

系统消息：警告/2 (`<stdin>`, 第 189 行)

枚举列表在没有空行的情况下结束；意外的缩进错误。

2\. **other-modules** 表示在本包中定义了一个私有模块 4\. **signatures** 表示在本包中定义了一个公共签名（没有私有签名；它们总是公共的，因为签名*总是*必须被实现） 5\. **reexported-modules** 表示在依赖中定义了一个公共模块或签名。

在这个列表中，公共意味着它对客户端是可用的。注意前四个字段列出了本包中的所有源代码。以下是客户端的一个简单示例：

```
name: utilities-extras
library
  build-depends: utilities
  exposed-modules: Utils.Extra

```

* * *

### 总结

我们已经涵盖了很多内容，但归根结底，Backpack 真正出色是因为一组正交特性的良好互动：

1.  **模块签名**：模块系统的核心，使我们能够编写不定的包并混合实现，

1.  **模块重新导出**：能够将本地可用的模块重新导出为不同的名称，并

1.  **模块精简和重命名**：有选择地从依赖中公开模块的能力。

要编译一个 Backpack 包，我们首先运行传统的版本依赖解决，获取所有涉及包的精确版本，然后计算如何将这些包链接在一起。就是这样！在未来的博客文章中，我计划更全面地描述这些新特性的语义，特别是有时可能会有微妙的模块签名。
