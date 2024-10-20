<!--yml

category: 未分类

date: 2024-07-01 18:18:20

-->

# 无所不在的 Cabal：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/05/omnipresent-cabal/`](http://blog.ezyang.com/2010/05/omnipresent-cabal/)

## 无所不在的 Cabal

一个简短的公共服务声明：你可能认为你不需要 Cabal。哦，你可能只是在编写一个小型的临时脚本或一个你根本不打算分发的小应用程序。*Cabal？那不是你打算将包发布到 Hackage 上时才会做的事情吗？* 但是 Cabal 总是知道的。Cabal 总是在那里。即使你认为自己不重要，你也应该拥抱 Cabal。以下是为什么：

1.  编写一个`cabal`文件会迫使你记录在最初编写脚本时使用的模块和版本。如果你决定在另一个环境中运行或构建你的脚本，cabal 文件将大大简化获取依赖项并快速运行的过程。如果你更新了模块，cabal 文件将在一定程度上保护你免受 API 更改的影响（假设包遵循[Hackage 的 PVP](http://www.haskell.org/haskellwiki/Package_versioning_policy)）。这比 GHC 的包限定导入要好得多。

1.  你可能对编写`Makefile`或`ant`文件来构建你的项目感到不爽；只要是一个或两个文件，与这些构建语言相关的痛苦似乎比运行`gcc foo.c -o foo`还要大。编写 Cabal 文件非常简单。甚至有一个[cabal init](http://byorgey.wordpress.com/2010/04/15/cabal-init/)来为你完成脚手架工作。抛弃你一直用来运行`ghc --make`的微小 shell 脚本，改用`cabal configure && cabal build`。

1.  它可以免费为你提供很多好东西！你想要 Haddock 文档吗？传统的 GNU 风格的 Makefile？代码着色？Cabal 都可以为你做到，而且在编写完`cabal`文件后，只需要很少的努力。
