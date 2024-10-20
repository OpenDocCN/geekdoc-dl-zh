<!--yml

category: 未分类

date: 2024-07-01 18:17:03

-->

# [如何将 GHC API 程序与 Cabal 集成](http://blog.ezyang.com/2017/02/how-to-integrate-ghc-api-programs-with-cabal/)：ezyang 的博客

> 来源：[`blog.ezyang.com/2017/02/how-to-integrate-ghc-api-programs-with-cabal/`](http://blog.ezyang.com/2017/02/how-to-integrate-ghc-api-programs-with-cabal/)

GHC 不仅是一个编译器：它还是一个库，提供了多种功能，任何对 Haskell 源代码进行分析感兴趣的人都可以使用。Haddock、hint 和 ghc-mod 都是使用 GHC API 的包。

对于希望使用 GHC API 的任何程序而言，与 Cabal（以及通过它的 cabal-install 和 Stack）集成是一个挑战。最明显的问题是，在构建时针对由 Cabal 安装的包时，需要向 GHC 传递适当的标志，告诉它应该使用哪些包数据库和实际包。在这一点上，人们往往采用 [某些不太正规的策略](https://groups.google.com/forum/#!topic/haskell-cafe/3ZgLB2khhcI) 来获取这些标志，并希望一切顺利。对于常用的包，这种策略可以完成任务，但对于需要额外处理的罕见包（例如预处理、额外的 GHC 标志、构建 C 源码），不太可能得到正确处理。

与 Cabal 集成 GHC API 程序的一个更可靠的方法是*控制反转*：让 Cabal 调用你的 GHC API 程序，而不是反过来！我们要如何让 Cabal/Stack 调用我们的 GHC API 程序？我们将替换掉经过所有命令的普通 GHC 的 GHC 可执行文件，除了 `ghc --interactive`，我们将将其传递给 GHC API 程序。然后，我们将使用我们重载的 GHC 调用 `cabal repl`/`stack repl`，在我们本来会打开 GHCi 提示符的地方，我们将运行我们的 API 程序。

通过这种方式，所有应该传递给 `ghc --interactive` 调用的标志都传递给我们的 GHC API 程序。我们如何解析这些标志？最方便的方法是创建一个 [前端插件](https://downloads.haskell.org/~ghc/master/users-guide/extending_ghc.html#frontend-plugins)，这样你可以为 GHC 创建一个新的主要模式。当你的代码被调用时，所有标志已经被处理过了（无需与 `DynFlags` 纠缠！）。

言归正传，是时候写一些代码了。首先，让我们看一个简单的前端插件：

```
module Hello (frontendPlugin) where

import GhcPlugins
import DriverPhases
import GhcMonad

frontendPlugin :: FrontendPlugin
frontendPlugin = defaultFrontendPlugin {
  frontend = hello
  }

hello :: [String] -> [(String, Maybe Phase)] -> Ghc ()
hello flags args = do
    liftIO $ print flags
    liftIO $ print args

```

这个前端插件直接来自 GHC 文档（但导入了足够的内容以使其能够编译；-)）。它打印出传递给它的参数。

接下来，我们需要一个围绕 GHC 的包装程序，当以 `--interactive` 标志调用时，将调用我们的插件而不是常规的 GHC。以下是适用于类 Unix 系统的简单脚本：

```
import GHC.Paths
import System.Posix.Process
import System.Environment

main = do
  args <- getArgs
  let interactive = "--interactive" `elem` args
      args' = do
        arg <- args
        case arg of
          "--interactive" ->
            ["--frontend", "Hello",
             "-plugin-package", "hello-plugin"]
          _ -> return arg
  executeFile ghc False (args' ++ if interactive then ["-user-package-db"] else []) Nothing

```

给这个 Cabal 文件，并使用 `cabal install` 将其安装到用户包数据库中（如果你想使用非标准的 GHC，请参阅下面的第二个要点）：

```
name:                hello-plugin
version:             0.1.0.0
license:             BSD3
author:              Edward Z. Yang
maintainer:          ezyang@cs.stanford.edu
build-type:          Simple
cabal-version:       >=1.10

library
  exposed-modules:     Hello
  build-depends:       base, ghc >= 8.0
  default-language:    Haskell2010

executable hello-plugin
  main-is:             HelloWrapper.hs
  build-depends:       base, ghc-paths, unix
  default-language:    Haskell2010

```

现在，要运行你的插件，你可以执行以下任意一种方法：

+   `cabal repl -w hello-plugin`

+   `cabal new-repl -w hello-plugin`

+   `stack repl --system-ghc --with-ghc hello-plugin`

要在特定包上运行插件，请将适当的标志传递给`repl`命令。

此示例的完整代码可以在 GitHub 上的[ezyang/hello-plugin](https://github.com/ezyang/hello-plugin)检索到。

这里有一些杂项提示和技巧：

+   如有必要，可以添加`--ghc-options=-ffrontend-opt=arg`来向插件传递额外的标志（如果愿意，可以围绕这一点编写另一个包装脚本！）

+   如果您使用的 GHC 不是来自您的 PATH 的那个安装了`hello-plugin`，您需要将正确的`ghc`/`ghc-pkg`/等可执行文件放在 PATH 的最前面；如果您仅使用`-w`，Cabal 的自动检测将会混淆。如果您正在运行`cabal`，解决此问题的另一种方法是通过传递`--with-ghc-pkg=PATH`来指定`ghc-pkg`的位置（Stack 不支持此功能）。

+   您不必将插件安装到用户包数据库中，但是需要调整包装程序以便能够找到包实际安装的位置。我不知道有什么方法可以在不编写自定义设置脚本的情况下获取此信息；希望将插件安装到用户包数据库中对于普通用户来说不会太麻烦。

+   `cabal-install`和`stack`在如何传递主模块给 GHCi 的调用上略有不同：`cabal-install`将为每个主包模块调用 GHC；Stack 将传递一个 GHCi 脚本以加载这些内容。我不确定哪种方法更方便，但如果您已经知道要查看哪个模块（可能是从前端选项中获得的），那可能并不太重要。
