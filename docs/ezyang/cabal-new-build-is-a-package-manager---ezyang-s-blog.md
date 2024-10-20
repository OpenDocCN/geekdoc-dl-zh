<!--yml

类别：未分类

日期：2024-07-01 18:17:05

-->

# cabal new-build 是一个包管理器：ezyang 的博客

> 来源：[`blog.ezyang.com/2016/08/cabal-new-build-is-a-package-manager/`](http://blog.ezyang.com/2016/08/cabal-new-build-is-a-package-manager/)

今天偶尔会看到引用的一篇旧文章是[重复一遍："Cabal 不是一个包管理器"](https://ivanmiljenovic.wordpress.com/2010/03/15/repeat-after-me-cabal-is-not-a-package-manager/)。很多批评不适用于 cabal-install 1.24 的新[Nix 风格本地构建](http://blog.ezyang.com/2016/05/announcing-cabal-new-build-nix-style-local-builds/)。让我们澄清一下。

### 事实：cabal new-build 不处理非 Haskell 依赖项

好的，这是自 Ivan 的文章以来没有改变的一点。与 Stack 不同，`cabal new-build`不会帮助你下载和安装 GHC，并且像 Stack 一样，它也不会下载和安装系统库或编译器工具链：这些都需要你自己来做。在这种情况下，你应该依赖于系统包管理器来启动 Cabal 和 GHC 的工作安装。

### 事实：Cabal 文件格式可以记录非 Haskell 的 pkg-config 依赖项。

自 2007 年起，Cabal 文件格式有一个`pkgconfig-depends`字段，可用于指定对由[pkg-config](https://en.wikipedia.org/wiki/Pkg-config)工具理解的库的依赖关系。它不会为您安装非 Haskell 依赖项，但它可以让您提前知道某个库是否不可用。

实际上，cabal-install 的依赖解决器了解`pkgconfig-depends`字段，并会选择版本并设置标志，以便我们不会得到一个具有不可满足的 pkg-config 依赖的包。

### 事实：cabal new-build 可以升级包而不会破坏你的数据库

假设你正在开发一个依赖于几个依赖项的项目。你决定通过放宽项目配置中的版本约束来升级其中一个依赖项。做出这些改变后，只需运行`cabal new-build`重新构建相关的依赖项并开始使用它。就是这样！更好的是，如果你有一个使用旧依赖项的旧项目，它仍然能够正常工作，就像你希望的那样。

`cabal new-build`实际上并不像传统的升级那样工作。安装到`cabal new-build`全局存储的包是通过类似 Nix 的标识符唯一标识的，该标识符包含了影响构建的所有信息，包括构建所依赖的具体版本。因此，“升级”一个包实际上只是安装一个使用不同唯一标识符的包，这个包可以与旧版本共存。你永远不会因为输入`new-build`而导致包数据库损坏。

目前没有*删除*软件包的机制，除非删除您的存储（`.cabal/store`），但值得注意的是，删除您的存储是完全安全的操作：如果存储不存在，`cabal new-build` 不会决定以不同方式构建您的软件包；存储仅仅是一个缓存，*不*影响依赖解决过程。

### 事实：除了软件包作者外，Hackage 受托人还可以编辑已发布软件包的 Cabal 文件以修复错误。

如果上传的软件包带有错误的版本范围，并且随后的新版本破坏了这些范围，[Hackage 受托人](https://www.haskell.org/wiki/Hackage_trustees) 可以介入，修改 Cabal 文件以根据新信息更新版本范围。这是一种比 Linux 发行版补丁更有限的干预形式，但其本质类似。

### 事实：如果可能的话，请使用您的系统包管理器。

`cabal new-build` 很棒，但并非人人都适用。如果您只需在系统上安装一个可工作的`pandoc`二进制文件，并且不介意是否有最新的版本，您应该通过操作系统的包管理器下载和安装它。发行版软件包非常适合二进制文件；对于开发人员来说，它们的库通常太旧（尽管这通常是获得工作 OpenGL 安装的最简单方法）。`cabal new-build` 面向的是 Haskell 软件包的开发人员，他们需要构建并依赖于操作系统未分发的软件包。

我希望这篇文章能消除一些误解！
