<!--yml

category: 未分类

date: 2024-07-01 18:17:06

-->

# ghc-shake：重新实现 ghc --make：ezyang 的博客

> 来源：[`blog.ezyang.com/2016/01/ghc-shake-reimplementing-ghc-make/`](http://blog.ezyang.com/2016/01/ghc-shake-reimplementing-ghc-make/)

## ghc-shake：重新实现 ghc --make

`ghc --make` 是 GHC 中的一种有用模式，它会自动确定需要编译的模块，并为您编译它们。它不仅是构建 Haskell 项目的便捷方式，其单线程性能也很好，通过重用读取和反序列化外部接口文件的工作。然而，`ghc --make` 也存在一些缺点：

1.  具有大模块图的项目在重新编译开始之前有相当长的延迟。这是因为 `ghc --make` 在实际进行任何工作之前会重新计算完整的模块图，解析每个源文件的头文件。如果您使用预处理器，情况会更糟（[参见这里](https://ghc.haskell.org/trac/ghc/ticket/1290)）。

1.  这是一个单体构建系统，如果需要比 GHC 默认功能更复杂的东西，将其与其他构建系统集成起来会很困难。（例如，GHC 精心设计的构建系统知道如何在包边界之间并行构建，而 Cabal 不知道如何做。）

1.  它无法提供有关构建性能的洞察，例如哪些模块需要很长时间构建，或者哪些“阻塞”模块很大。

[ghc-shake](https://github.com/ezyang/ghc-shake) 是使用 [Shake 构建系统](http://shakebuild.com/) 重新实现的 `ghc --make`。它可以作为 `ghc` 的替代品。ghc-shake 具有以下特性：

1.  大大减少了重新编译的延迟。这是因为 Shake 不会通过解析每个文件的头文件来重新计算模块图；它会重用缓存的信息，仅重新解析已更改的源文件。

1.  如果重新构建文件（并更新其时间戳），但构建输出未更改，我们就不会重新编译任何依赖于它的内容。这与 `ghc --make` 相比，后者必须在确定没有要做的工作之前运行每个下游模块的重新编译检查，有所不同。事实上，ghc-shake 从不运行重新编译测试，因为我们在 Shake 中本地实现了这种依赖结构。

1.  使用 `-ffrontend-opt=--profile`，你可以获得有关构建的详细分析信息，包括每个模块构建所花费的时间，以及更改一个模块的成本。

1.  在单线程构建上与 `ghc --make` 一样快。与另一个使用 Shake 构建 Haskell 的构建系统 [ghc-make](https://github.com/ndmitchell/ghc-make) 相比，ghc-make 并不使用 GHC API，并且必须使用（慢速的）`ghc -M` 来获取项目的初始依赖信息。

1.  它是准确的。它正确处理许多边缘情况（如 `-dynamic-too`），因为它是使用 GHC API 编写的，原则上可以与 `ghc --make` 功能完全兼容。（当前情况不是这样，只是因为我还没有实现它们。）

也有一些缺点：

1.  Shake 构建系统需要一个 `.shake` 目录来实际存储有关构建的元数据。这与 `ghc --make` 相反，后者完全依赖于目录中构建产品的时间戳。

1.  因为它直接使用了 GHC API 实现，所以只能与特定版本的 GHC（即即将发布的 GHC 8.0 版本）一起使用。

1.  它需要一个修补过的 Shake 库版本，因为我们有一个基于 Shake 的（未导出的）文件表示的自定义模块构建规则。我已经[在这里报告了](https://github.com/ndmitchell/shake/issues/388)。

1.  仍然存在一些缺失的功能和 bug。我遇到的问题是（1）在某些情况下[我们忘记了重新链接](https://ghc.haskell.org/trac/ghc/ticket/10161)，以及（2）它不能用于[构建分析代码](https://ghc.haskell.org/trac/ghc/ticket/11293)。

如果你想今天就使用 `ghc-shake`（不适合心脏虚弱的人），试试 `git clone https://github.com/ezyang/ghc-shake`，然后按照 `README` 中的说明操作。但即使你不打算使用它，我认为 `ghc-shake` 的代码对任何想编写涉及 Haskell 代码的构建系统的人来说都有一些好的教训。其中最重要的架构决策之一是使 `ghc-shake` 中的规则不是围绕输出文件（例如 `dist/build/Data/Foo.hi`，如 `make` 中那样）组织，而是围绕 Haskell 模块（例如 `Data.Foo`）组织的。语义化的构建系统比强制将一切都放入“文件抽象”中要好得多（尽管 Shake 在我希望的模式下使用上并不完全支持）。还有一些其他有趣的经验教训... 但那应该是另一篇博客文章的主题！

这个项目的未来方向在哪里？在不太近的未来，我考虑做一些事情：

1.  为了支持多个 GHC 版本，我们应该将 GHC 特定的代码分离出来成为一个单独的可执行文件，并通过 IPC 进行通信（向 Duncan Coutts 致敬）。这也将使我们能够支持独立进程的并行 GHC 构建，仍然可以重用读取接口文件。无论如何，`ghc-shake` 可以作为 GHC 需要使构建系统更易于访问所需信息的蓝图。

1.  我们可以考虑将这些代码移回 GHC。不幸的是，Shake 是一个太大的依赖项，无法实际让 GHC 依赖它，但可以考虑设计一些抽象接口（你好，Backpack！），用于表示类似 Shake 的构建系统，并让 GHC 提供 `--make` 的简单实现（但用户可以选择切换到 Shake）。

1.  我们可以将这段代码扩展到`ghc --make`以了解如何构建整个 Cabal 项目（或更大的项目），比如[ToolCabal](https://github.com/TiborIntelSoft/ToolCabal)，这是使用 Shake 重新实现的 Cabal。这将允许我们捕捉类似于 GHC 构建系统的模式，该系统可以并行构建所有引导包中的模块（而不必等待包完全构建完成）。

P.S. ghc-shake 不应与[shaking-up-ghc](https://github.com/snowleopard/shaking-up-ghc)混淆，后者是一个旨在用 Shake 基础构建系统替换 GHC 基于 Makefile 的构建系统的项目。
