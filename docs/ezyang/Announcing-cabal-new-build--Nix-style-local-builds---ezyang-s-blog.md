<!--yml

category: 未分类

date: 2024-07-01 18:17:06

-->

# 宣布 cabal new-build：Nix 风格的本地构建：ezyang 的博客

> 来源：[`blog.ezyang.com/2016/05/announcing-cabal-new-build-nix-style-local-builds/`](http://blog.ezyang.com/2016/05/announcing-cabal-new-build-nix-style-local-builds/)

`cabal new-build`，也称为“Nix 风格的本地构建”，是受 Nix 启发的一个新命令，随 cabal-install 1.24 一同发布。Nix 风格的本地构建结合了非沙盒化和沙盒化 Cabal 的优点：

1.  像今天的沙盒化 Cabal 一样，我们以确定性和独立于任何全局状态的方式构建一组独立的本地包。`new-build` 永远不会告诉你，它不能构建你的包，因为这将导致“危险的重新安装”。在特定状态的 Hackage 索引下，你的构建是完全可重复的。例如，你不再需要预先使用性能分析编译包；只需请求性能分析，`new-build` 将自动重新构建所有依赖项以进行性能分析。

1.  像今天的非沙盒化 Cabal 一样，外部包的构建全局缓存，因此一个包只需构建一次，然后可以在任何其他使用它的地方重复使用。不需要在每次创建新沙盒时不断重新构建依赖项：可以共享的依赖项会被共享。

Nix 风格的本地构建与 cabal-install 1.24 支持的所有 GHC 版本兼容，目前包括 GHC 7.0 及更高版本。此外，cabal-install 的发布周期与 GHC 不同，因此我们计划以比 GHC 每年一次的发布周期更快的速度推送 bug 修复和更新。

尽管此功能目前仅处于测试版（存在一些 bug，请参见“已知问题”，并且文档有点稀少），但我一直成功地使用 Nix 风格的本地构建来进行我的 Haskell 开发。难以形容我对这一新功能的热情：它“只是有效”，而且你不需要假设有一个版本固定的、经过认可的包分发来构建（例如 Stackage）。最终，`new-build` 将简单地取代现有的 `build` 命令。

### 快速入门

Nix 风格的本地构建“只是有效”：几乎不需要任何配置即可开始使用它。

1.  下载并安装 cabal-install 1.24：

    ```
    cabal update
    cabal install cabal-install

    ```

    确保新安装的 `cabal` 已添加到你的路径中。

1.  要构建单个 Cabal 包，不需要运行 `cabal configure; cabal build`，可以通过在这些命令前加上 `new-` 来使用 Nix 风格的构建；例如 `cabal new-configure; cabal new-build`。`cabal new-repl` 也受支持。（不幸的是，其他命令尚未支持，例如 `new-clean` ([#2957](https://github.com/haskell/cabal/issues/2957)) 或 `new-freeze` ([#2996](https://github.com/haskell/cabal/issues/2996))。）

1.  要构建多个 Cabal 包，需要首先在某个根目录下创建 `cabal.project` 文件。例如，在 Cabal 存储库中，有一个根目录，其中每个包都有一个文件夹，例如，文件夹 `Cabal` 和 `cabal-install`。然后在 `cabal.project` 中，指定每个文件夹：

    ```
    packages: Cabal/
              cabal-install/

    ```

    然后，在包的目录中，您可以使用 `cabal new-build` 来构建该包中的所有组件；或者，您可以指定要构建的目标列表，例如，`package-tests cabal` 要求构建 `package-tests` 测试套件和 `cabal` 可执行文件。组件可以从任何目录构建；您不必 cd 到包含要构建的包的目录中。此外，您可以按其来自的包合格目标，例如 `Cabal:package-tests` 具体要求从 Cabal 获取 `package-tests` 组件。无需手动配置沙盒：添加 `cabal.project` 文件，它就可以工作了！

不像沙盒，无需 `add-source`；只需将包目录添加到您的 `cabal.project` 中。而且与传统的 `cabal install` 不同，无需显式请求安装包；`new-build` 将自动获取并构建依赖项。

还有一个方便的[脚本](https://github.com/hvr/multi-ghc-travis/blob/master/make_travis_yml_2.hs)，您可以用它来连接 `new-build` 到您的[Travis 构建](https://github.com/hvr/multi-ghc-travis)。

### 工作原理

Nix 风格的本地构建采用了以下两个重要的思想：

1.  对于外部包（来自 Hackage），在编译之前，我们将会影响包编译的所有输入（标志，依赖选择等）进行哈希处理，生成一个标识符。就像在 Nix 中一样，这些哈希唯一地标识构建的结果；如果我们计算出此标识符，并发现我们已经构建了这个 ID，我们可以直接使用已经构建好的版本。这些包存储在全局的 `~/.cabal/store` 中；您可以使用 `ghc-pkg list --package-db=$HOME/.cabal/store/ghc-VERSION/package.db` 列出所有全局可用的 Nix 包。

1.  对于本地包，我们使用 `inplace` 标识符，例如 `foo-0.1-inplace`，这是针对给定 `cabal.project` 的本地包。这些包存储在本地的 `dist-newstyle/build` 目录中；您可以使用 `ghc-pkg list --package-db=dist-newstyle/packagedb` 列出所有按项目分组的包。这种处理方式适用于任何依赖于本地包的远程包（例如，如果您嵌入某些依赖项，而其他依赖项依赖于它们）。

此外，Nix 的本地构建采用确定性依赖解决策略，通过独立于本地安装包进行依赖解决。一旦我们解决了要使用的版本，并确定了编译过程中将使用的所有标志，我们生成标识符，然后检查是否可以改进我们需要构建的包，使其成为已经在数据库中的包。

### 命令

#### `new-configure FLAGS`

基于 FLAGS 覆盖 `cabal.project.local`。

#### `new-build [FLAGS] [COMPONENTS]`

构建一个或多个组件，自动构建任何本地和非本地依赖项（本地依赖项是指我们在开发过程中可以修改的现有源代码目录）。不具有本地包的传递依赖关系的非本地依赖项安装到 `~/.cabal/store`，而所有其他依赖项安装到 `dist-newstyle`。

从 `cabal.project` 中读取本地包的集合；如果不存在，则假定默认项目包括本地目录中的所有 Cabal 文件（即 `packages: *.cabal`），以及每个子目录中的可选包（即 `optional-packages: */*.cabal`）。

*本地* 包的构建配置是通过以下来源读取标志来计算的（后续来源具有优先级）：

1.  `~/.cabal/config`

1.  `cabal.project`

1.  `cabal.project.local`（通常由 `new-configure` 生成）

1.  命令行的 `FLAGS`

非本地包的配置只受这些来源中特定于包的标志的影响；全局选项不适用于构建。（例如，如果 `--disable-optimization`，则仅适用于本地的 inplace 包，而不适用于它们的远程依赖项。）

`new-build` 不从 `cabal.config` 中读取配置。

#### 短语手册

这里是一个便捷的短语手册，说明如何使用 Nix 本地构建来执行现有的 Cabal 命令：

| old-style | new-style |
| --- | --- |
| `cabal configure` | `cabal new-configure` |
| `cabal build` | `cabal new-build` |
| `cabal clean` | `rm -rf dist-newstyle cabal.project.local` |
| `cabal run EXECUTABLE` | `cabal new-build; ./dist-newstyle/build/PACKAGE-VERSION/build/EXECUTABLE/EXECUTABLE` |
| `cabal repl` | `cabal new-repl` |
| `cabal test TEST` | `cabal new-build; ./dist-newstyle/build/PACKAGE-VERSION/build/TEST/TEST` |
| `cabal benchmark BENCH` | `cabal new-build; ./dist-newstyle/build/PACKAGE-VERSION/build/BENCH/BENCH` |
| `cabal haddock` | 目前不存在 |
| `cabal freeze` | 目前不存在 |
| `cabal install --only-dependencies` | 不必要的（由 `new-build` 处理） |
| `cabal install` | 目前不存在（对于库，`new-build` 应该足够；对于可执行文件，它们可以在 `~/.cabal/store/ghc-GHCVER/PACKAGE-VERSION-HASH/bin` 中找到） |

### cabal.project 文件

`cabal.project` 文件实际上支持多种选项，用于配置构建的详细信息。以下是一个简单的示例文件，展示了一些可能性：

```
-- For every subdirectory, build all Cabal files
-- (project files support multiple Cabal files in a directory)
packages: */*.cabal
-- Use this compiler
with-compiler: /opt/ghc/8.0.1/bin/ghc
-- Constrain versions of dependencies in the following way
constraints: cryptohash < 0.11.8
-- Do not build benchmarks for any local packages
benchmarks: False
-- Build with profiling
profiling: true
-- Suppose that you are developing Cabal and cabal-install,
-- and your local copy of Cabal is newer than the
-- distributed hackage-security allows in its bounds: you
-- can selective relax hackage-security's version bound.
allow-newer: hackage-security:Cabal

-- Settings can be applied per-package
package cryptohash
  -- For the build of cryptohash, instrument all functions
  -- with a cost center (normally, you want this to be
  -- applied on a per-package basis, as otherwise you would
  -- get too much information.)
  profiling-detail: all-functions
  -- Disable optimization for this package
  optimization: False
  -- Pass these flags to GHC when building
  ghc-options: -fno-state-hack

package bytestring
  -- And bytestring will be built with the integer-simple
  -- flag turned off.
  flags: -integer-simple

```

运行 `cabal new-configure` 时，它会输出一个 `cabal.project.local` 文件，其中保存了从命令行输入的额外配置选项；如果想知道如何将命令行参数转换为 `cabal.project` 文件，请运行 `new-configure` 并检查输出。

### 已知问题

作为技术预览，这段代码仍然[有些粗糙](https://github.com/haskell/cabal/labels/nix-local-build)。以下是一些可能遇到的更重要的问题：

+   虽然依赖关系解析是确定性的，如果使用 `cabal update` 更新你的 Hackage 索引，[依赖关系解析也会改变](https://github.com/haskell/cabal/issues/2996)。没有 `cabal new-freeze`，所以你必须手动构建所需约束的集合。

+   new-build 的一个新功能是，当包没有变化时，它避免重新构建包，通过跟踪它们内容的哈希值。然而，这种依赖跟踪不是百分之百准确（具体来说，它依赖于你的 Cabal 文件准确地报告所有文件依赖项，就像 `sdist`，并且不知道搜索路径）。目前没有 UI 强制重新编译一个包；不过你可以通过删除适当的缓存文件相对容易地诱发重新编译：特别是对于名为 `p-1.0` 的包，删除文件 `dist-newstyle/build/p-1.0/cache/build`。

+   在 Mac OS X 上，Haskell 平台，你可能会收到“警告：'hackage.haskell.org' 的包列表不存在。运行 'cabal update' 下载它。”这是[问题 #3392](https://github.com/haskell/cabal/issues/3392)；查看链接的票证以获取解决方法。

如果你遇到其他 bug，请在[Cabal 的问题跟踪器](https://github.com/haskell/cabal/issues/new?labels=nix-local-build)上告诉我们。
