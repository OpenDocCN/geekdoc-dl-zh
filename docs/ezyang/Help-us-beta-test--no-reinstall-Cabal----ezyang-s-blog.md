<!--yml

category: 未分类

date: 2024-07-01 18:17:06

-->

# 帮助我们进行“无需重新安装 Cabal”的 Beta 测试：ezyang 博客

> 来源：[`blog.ezyang.com/2015/08/help-us-beta-test-no-reinstall-cabal/`](http://blog.ezyang.com/2015/08/help-us-beta-test-no-reinstall-cabal/)

## 帮助我们进行“无需重新安装 Cabal”的 Beta 测试

在今年夏天，Vishal Agrawal 正在进行一个 GSoC 项目，将 Cabal 移动到更类似 Nix 的包管理系统中。更简单地说，他正在努力确保您将不会再从 cabal-install 中遇到这类错误：

```
Resolving dependencies...
In order, the following would be installed:
directory-1.2.1.0 (reinstall) changes: time-1.4.2 -> 1.5
process-1.2.1.0 (reinstall)
extra-1.0 (new package)
cabal: The following packages are likely to be broken by the reinstalls:
process-1.2.0.0
hoogle-4.2.35
haskell98-2.0.0.3
ghc-7.8.3
Cabal-1.22.0.0
...

```

但是，这些补丁改变了 Cabal 和 cabal-install 中许多复杂的部分，因此在将其合并到 Cabal HEAD 之前，有意愿的小白鼠帮助我们消除一些错误将非常有帮助。作为奖励，您将能够运行“无需重新安装 Cabal”：Cabal **永远** 不会告诉您无法安装包，因为需要一些重新安装。

以下是您可以提供帮助的方式：

1.  确保你正在运行 GHC 7.10。早期版本的 GHC 存在一个严格的限制，不允许你针对不同的依赖多次重新安装同一个包。（实际上，如果你能测试旧版本的 GHC 7.8，这将非常有用，主要是为了确保我们在这方面没有引入任何退化。）

1.  `git clone https://github.com/ezyang/cabal.git`（在我的测试中，我已经在 Vishal 的版本基础上添加了一些额外的修正补丁），然后 `git checkout cabal-no-pks`。

1.  在 `Cabal` 和 `cabal-install` 目录中，运行 `cabal install`。

1.  尝试在没有沙盒的情况下构建项目，看看会发生什么！（在我的测试中，我曾尝试同时安装多个版本的 Yesod。）

在测试之前不需要清除您的包数据库。如果您完全破坏了您的 Haskell 安装（可能性不大，但确实可能发生），您可以使用旧版的 `cabal-install` 清理掉您的 `.ghc` 和 `.cabal` 目录（不要忘记保存您的 `.cabal/config` 文件），然后重新引导安装。

请在此处报告问题，或者在 [Cabal 跟踪器中的此 PR](https://github.com/haskell/cabal/pull/2752) 中报告。或者下周在 ICFP 会议上与我面对面交流。 :)
