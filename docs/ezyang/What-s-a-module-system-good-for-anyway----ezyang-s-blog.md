<!--yml

category: 未分类

date: 2024-07-01 18:17:14

-->

# 到底模块系统有什么好处呢？：ezyang 的博客

> 来源：[`blog.ezyang.com/2014/08/whats-a-module-system-good-for-anyway/`](http://blog.ezyang.com/2014/08/whats-a-module-system-good-for-anyway/)

今年夏天，我在微软研究院工作，实现了[Haskell 的 Backpack](http://plv.mpi-sws.org/backpack/)，一个模块系统。有趣的是，Backpack 并不是一个单一的庞大特性，而是一系列小的基础设施改进，这些改进以一种有趣的方式结合在一起。在这一系列博文中，我想讨论这些个别特性是什么，以及整体如何大于部分的总和。

但首先，有一个重要的问题需要回答：**到底模块系统有什么好处呢？** 为什么你作为一名普通的 Haskell 程序员，要关心诸如*模块系统*和*模块化*这样朦胧的东西。归根结底，你希望你的工具能解决你现有的具体问题，有时候很难理解像 Backpack 这样的模块系统到底解决了什么问题。正如[tomejaguar 所说](http://www.reddit.com/r/haskell/comments/28v6c9/backpack_an_mllike_module_system_for_haskell/cierxc1)：“有人能清楚地解释 Backpack 解决的确切问题吗？我读过论文，我知道问题是‘模块化’，但我担心我缺乏想象力，无法真正理解问题的实质是什么。”

不用再找了。在这篇博文中，我想具体讨论 Haskell 程序员今天面临的问题，解释这些问题的根本原因，并说明为什么一个模块系统可以帮助解决问题。

### 字符串、Text、ByteString 的问题

如有经验的 Haskell 程序员们所[知](http://blog.ezyang.com/2010/08/strings-in-haskell/)，[了解](http://stackoverflow.com/questions/19608745/data-text-vs-string)多种 Haskell 字符串类型：String、ByteString（延迟与严格）、Text（同样也有延迟与严格）。更加复杂的是，并没有一个“正确”的字符串类型选择：不同的情况适合不同的类型。String 方便且是 Haskell'98 的本地类型，但非常慢；ByteString 快速但只是字节的数组；Text 慢一些但支持 Unicode。

在理想世界中，程序员可以根据他们的应用选择最合适的字符串表示，并相应地编写所有的代码。然而，对于库编写者来说，这并不能解决问题，因为他们不知道用户会使用哪种字符串类型！那么库编写者该怎么办呢？他们只有几种选择：

1.  当存在不匹配时，它们会“承诺”使用一种特定的字符串表示，让用户在不同表示之间手动转换。或者更可能的是，库的编写者因为默认方式易于使用而选择了默认方式。例如：[base](https://hackage.haskell.org/package/base)（使用 Strings，因为它完全在其他表示之前存在），[diagrams](https://hackage.haskell.org/package/diagrams)（使用 Strings，因为它实际上不做大量字符串操作）。

1.  它们可以为每个变体提供单独的函数，可能命名相同但放置在不同模块中。这种模式经常用于支持严格/惰性变体 Text 和 ByteStringExamples：[aeson](http://hackage.haskell.org/package/aeson)（为惰性/严格 ByteString 提供 decode/decodeStrict）、[attoparsec](https://hackage.haskell.org/package/attoparsec)（提供 Data.Attoparsec.ByteString/Data.Attoparsec.ByteString.Lazy）、[lens](http://hackage.haskell.org/package/lens)（提供 Data.ByteString.Lazy.Lens/Data.ByteString.Strict.Lens）。

1.  它们可以使用类型类来重载函数，以便与多种表示形式一起工作。使用的特定类型类大不相同：有[ListLike](https://hackage.haskell.org/package/ListLike)，被少数包使用，但大部分包通常自行开发。例如：[HDBC](http://hackage.haskell.org/package/HDBC) 中的 SqlValue，[tagsoup](https://hackage.haskell.org/package/tagsoup) 中的内部 StringLike，以及[web-encodings](http://hackage.haskell.org/package/web-encodings) 中的另一个内部 StringLike。

最后两种方法有不同的权衡。像第（2）种方式那样定义单独的函数是一种简单易懂的方法，但您仍在拒绝模块化：支持多种字符串表示的能力。尽管为每种表示提供了实现，用户在导入时仍需选择特定表示。如果他们想要更改字符串表示，他们必须遍历所有模块并重命名导入；如果他们想要支持多种表示，他们仍必须为每种表示编写单独的模块。

使用类型类（3）来恢复模块性似乎是一个吸引人的方法。但这种方法既有实际问题，也有理论问题。首先，你如何选择哪些方法放入类型类中呢？理想情况下，你会选择一个最小的集合，其他所有操作都可以从中派生出来。然而，许多操作在直接实现时效率最高，这导致了一个臃肿的类型类，并且对于那些拥有自己的字符串类型并需要编写自己实例的人来说非常困难。其次，类型类使得你的类型签名更加丑陋 `String -> String` 变成了 `StringLike s => s -> s`，并且可能会使类型推断变得更加困难（例如，引入歧义）。最后，类型类 `StringLike` 与类型类 `Monad` 的性质截然不同，后者具有一组最小操作和规定其操作的法则。很难（或者说不可能）描述这种接口的法则应该是什么样的。总而言之，与具体实现相比，编写针对类型类的程序要不那么愉快。

如果我能够 `import String`，给我提供 `String` 类型和相关操作，然后稍后再决定要实例化的具体实现，这是件多么好的事情啊！这是模块系统可以为你做到的事情！这篇 [Reddit 线程](http://www.reddit.com/r/haskell/comments/28v6c9/backpack_an_mllike_module_system_for_haskell/cierxc1) 描述了另外一些情况下 ML 风格模块将会很方便的情况。

（附注：为什么不能只写一堆预处理器宏来交换你想要的实现呢？答案是，“是的，你可以；但是如何在没有尝试每个实现的情况下对其进行类型检查呢？”）

### 破坏性的包重新安装

当你尝试安装新包时是否遇到过这个错误消息？

```
$ cabal install hakyll
cabal: The following packages are likely to be broken by the reinstalls:
pandoc-1.9.4.5
Graphalyze-0.14.0.0
Use --force-reinstalls if you want to install anyway.

```

不知何故，Cabal 得出结论说安装 hakyll 的唯一方法是重新安装某些依赖项。以下是可能导致这种情况发生的一种情况：

1.  pandoc 和 Graphalyze 都是针对最新的 unordered-containers-0.2.5.0 进行编译的，这本身又是针对最新的 hashable-1.2.2.0 进行编译的。

1.  hakyll 也依赖于 unordered-containers 和 hashable，但对 hashable 有一个排除最新版本的上限限制。Cabal 决定我们需要安装旧版本的 hashable，比如 hashable-0.1.4.5。

1.  如果安装了 hashable-0.1.4.5，我们还需要将 unordered-containers 针对这个旧版本进行构建，以便 Hakyll 可以看到一致的类型。然而，生成的版本与现有版本相同：因此，需要重新安装！

此错误的根本原因是 Cabal 当前对包数据库强制执行的不变式：对于任何给定的包名称和版本，只能有*一个*实例的包。特别地，这意味着不可能安装同一个包的多个实例，编译时使用不同的依赖关系。这有点麻烦，因为有时您确实希望安装具有不同依赖关系的相同包多次：正如上文所示，这可能是满足所有涉及包的版本界限的唯一方法。目前唯一解决此问题的方法是使用 Cabal 沙箱（或清除您的包数据库并重新安装所有内容，这基本上是相同的事情）。

您可能会想，模块系统如何可能帮助解决这个问题？实际上，并不是直接帮助。相反，包的非破坏性重新安装是实现类似 Backpack 的模块系统的关键功能（一个包可以安装多次，具有不同的模块具体实现）。实施 Backpack 需要解决这个问题，将 Haskell 的包管理更接近于 Nix 或 NPM。

### 版本界限和被忽视的 PVP

当我们讨论`cabal-install`出现错误时，您是否曾经尝试安装新包时遇到这个错误？

```
$ cabal install hledger-0.18
Resolving dependencies...
cabal: Could not resolve dependencies:
# pile of output

```

出现这种情况可能有很多原因，但通常是因为某些涉及的包有过度约束的版本界限（尤其是上界），导致一组不可满足的约束条件。更让人沮丧的是，通常这些界限没有实际依据（包作者仅仅是猜测了范围），去掉它可能会导致可以工作的编译。这种情况非常普遍，以至于 Cabal 有一个`--allow-newer`标志，允许您覆盖包的上界。管理界限的烦恼导致了诸如[cabal-bounds](https://github.com/dan-t/cabal-bounds)之类的工具的开发，这些工具试图让保持上界最新变得不那么繁琐。

尽管我们经常批评它们，版本界限有一个非常重要的功能：它们防止您尝试针对根本无法工作的依赖关系编译包！版本界限不足的一组版本范围可以很容易地使您针对无法通过类型检查的依赖版本进行编译。

一个模块系统如何帮助？归根结底，版本号试图捕捉有关包导出的 API 的一些信息，这由 [包版本控制策略](http://www.haskell.org/haskellwiki/Package_versioning_policy) 描述。但是当前的技术水平要求用户将 API 的变化手动转换为版本号：即使在 [各种工具](http://code.haskell.org/gtk2hs/tools/apidiff/) 的辅助下，这也是一个容易出错的过程。另一方面，一个模块系统将 API 转变为编译器本身理解的一流实体：一个 *模块签名*。如果包依赖于签名而不是版本号，那不是很棒吗？那么你将不再需要担心版本号与类型检查不准确。当然，版本号仍然对于记录类型中未见的语义变化是有用的，但在这里它们的角色次要而重要。这里需要一些充分的披露：我不打算在实习结束时实现这个功能，但我希望能对其做出一些重要的基础性贡献。

### 结论

如果你只是粗略地读了《背包论文》的介绍部分，可能会给你留下这样的印象：背包是关于随机数生成器、递归链接和适用语义的某种东西。虽然这些都是关于背包的真实“事实”，但它们低估了一个良好模块系统对工作程序员日常问题可能产生的影响。在这篇文章中，我希望我已经阐明了其中的一些问题，即使我还没有说服你像《背包》这样的模块系统实际上是如何解决这些问题的：这将在接下来的一系列文章中讨论。请继续关注！
