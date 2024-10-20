<!--yml

category: 未分类

date: 2024-07-01 18:18:16

-->

# Static Analysis for everyday (not-PhD) man : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/06/static-analysis-mozilla/`](http://blog.ezyang.com/2010/06/static-analysis-mozilla/)

*Bjarne Stroustrup 曾自豪地说：“C++是一种支持面向对象和其他有用编程风格的多范式编程语言。如果你在寻找一种强制你只能用一种方式做事情的语言，那么 C++不是。” 但正如 Taras Glek 讽刺地指出的那样，大多数用于 C++的静态分析和编码标准主要是为了确保开发人员不使用其他范式。*

星期二，[Taras Glek](http://blog.mozilla.com/tglek/)在[Mozilla 进行大规模静态分析](http://www.galois.com/blog/2010/06/03/tech-talk-large-scale-static-analysis-at-mozilla/)上做了演讲。你可以在[Galois 的视频流](http://vimeo.com/12614626)上观看视频。讲座的主题很简单直接：Mozilla 如何利用静态分析来管理其数百万行的 C++和 JavaScript 代码？但背后还有一个潜在的主题：静态分析不仅仅是形式方法专家和博士生的专利；任何人都可以并且应该利用静态分析带来的能力。

由于 Mozilla 是一个 C++的工作室，这次讲座集中讨论了构建在 C++语言之上的工具。然而，Taras 讨论的静态分析的四个部分是广泛适用于你可能进行静态分析的任何语言：

1.  *解析.* 如何将文本文件转换为源代码的抽象表示（AST）？

1.  *语义 grep.* 如何向 AST 询问信息？

1.  *分析.* 如何限制有效的 AST？

1.  *重构.* 如何改变 AST 并将其写回？

*解析.* 解析 C++很困难。从历史上看，这是因为它继承了许多来自 C 的语法；从技术上讲，这是因为它是一个极其模糊的语法，需要语义知识来消除歧义。Taras 使用[Elsa](http://scottmcpeak.com/elkhound/sources/elsa/)（在 Taras 修复了一堆 bug 之后，可以解析所有 Mozilla 的代码），并急切地期待[Clang](http://clang.llvm.org/)的稳定（因为它还不能完全解析所有 Mozilla 的代码）。当然，[GCC 4.5 的插件接口](http://gcc.gnu.org/wiki/plugins)的添加意味着你可以利用 GCC 的解析能力，并且许多后期工具都是基于此构建的。

*语义 grep.* `grep` 已经老掉牙了！如果你幸运的话，你的代码库维护者会遵循使代码“更易搜索”的古怪规则，但否则你会用一个标识符来 grep，却得到一页页的结果，却错过了你要找的。如果你有源代码的内存表示，你可以智能地询问信息。进一步来说，这意味着更好的代码浏览，比如[DXR](http://dxr.proximity.on.ca/dxr/)。

请考虑以下的[源代码视图](http://dxr.proximity.on.ca/dxr/mozilla-central/xpcom/io/nsLocalFileUnix.h.html)。这是您从源代码浏览器期望看到的：右边是源代码，左边是声明。

让我们从左边选一个标识符，比如`CopyTo`。让我们先浏览源代码，看看能否找到它。

哎呀！那真是一个很短的课程，但找不到它在哪里。好吧，让我们试着点击一下。

啊哈！它在*宏定义*中。这个工具*可能比*未经培训的眼睛更聪明。

*分析.* 对我来说，这真正是演讲的核心。Mozilla 有一个非常复杂的构建过程；通过使用[Dehydra](https://developer.mozilla.org/zh-CN/Dehydra)和[Treehydra](https://developer.mozilla.org/zh-CN/Treehydra)与 GCC 进行交互。Dehydra 项目的想法是利用 GCC 提供给插件的内部结构，并将其转换为类似 JSON 的结构（类似 JSON，因为 JSON 是非循环的，但这些结构是循环的），Dehydra 脚本（用*JavaScript*编写）可以在其上运行。这些脚本可以生成错误和警告，看起来就像 GCC 的构建错误和警告一样。Treehydra 是 Dehydra 的高级版本，为分析脚本编写者提供了更多灵活性。

那么，Dehydra/Treehydra 有什么有趣之处呢？

1.  *JavaScript.* GCC 的插件接口本来只支持 C 代码，这可能让没有静态分析经验的开发人员望而却步。将这些结构转换为 JavaScript 意味着您可以使用高级语言进行操作，也能让您告诉完全没有静态分析经验的初级开发人员：“这就像在一个 Web 应用程序上进行黑客攻击一样。” 这意味着您可以直接打印出类似 JSON 的结构，并查看所需数据的结果；这意味着当您的代码崩溃时，您会得到漂亮的回溯信息。就像 Firefox 的插件接口一样，Dehydra 将 GCC 扩展带给了大众。

1.  *语言的胶带.* 我在他的帖子开头批评了 Stroustrup，这就是原因。我们可以为类（带有属性`__attribute__((user("NS_final")))`，在宏`NS_FINAL_CLASS`中包装）和其他[限制](http://hg.mozilla.org/mozilla-central/file/86cdcd6616de/xpcom/analysis)，像`final`这样的额外语言特性附加功能，纯 C++不提供这些。

1.  *需要时有力量.* Dehydra 是一个简化的接口，适合没有静态分析或编译器背景的人；Treehydra 则更为强大，面向具有这些背景的人，可以让您执行诸如控制流分析之类的操作。

所有这些都透明地集成到构建系统中，因此开发人员无需摸索外部工具来获取这些分析结果。

*重构。* 或许是其中最雄心勃勃的一个，Taras 讨论了超越像 Java IDEs（比如 Eclipse）提供的简单*提取方法*的重构，使用[Pork](https://wiki.mozilla.org/Pork)。这种重构，比如“重写 Mozilla 所有代码以使用垃圾回收而不是引用计数”。当你拥有像 Mozilla 这样活跃的代码库时，你没有豪华的机会去做“停止所有开发并开始重构...长达六年”的风格重构。分支也会带来类似的问题；一旦主要重构在一个分支上落地，要保持该分支与其他分支同步更新是困难的，最终一个分支会淘汰另一个分支。

关键在于自动化重构工具，因为它让你把重构当作“只是另一个补丁”来对待，并持续从主干重建分支，应用你的自定义补丁并运行重构工具生成一个多兆字节的补丁来应用在堆栈中。

重构 C++很难，因为开发者不仅仅写 C++代码；他们写的是 C++和 CPP（C 预处理器）的结合体。重构工具在写回时需要能够重建 CPP 宏，而不像 ML 等语言那样仅仅进行漂亮的 AST 打印。技术包括尽可能少的漂亮打印，以及强制解析器给出所有预处理器修改的日志。

*开源软件。* Taras 留给我们一些关于开源协作的话语，至少[SIPB](http://sipb.mit.edu)群体应该深知。不要把你依赖的工具当作黑盒子：它们是开源的！如果你在 GCC 中发现了一个 bug，不要仅仅绕过它，查看源代码，编写补丁并提交到上游。这是修复 bug 的最佳方式，而且你为后续提交的 bug 赢得了即时的可信度。在 Web 应用到浏览器、浏览器到编译器、编译器到操作系统的层级中，开源生态系统的优势在于你可以一路阅读源代码。利用源代码，卢克。
