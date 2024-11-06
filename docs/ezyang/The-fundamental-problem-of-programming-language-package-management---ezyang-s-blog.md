<!--yml

category: 未分类

date: 2024-07-01 18:17:14

-->

# 编程语言包管理的根本问题：ezyang 的博客

> 来源：[`blog.ezyang.com/2014/08/the-fundamental-problem-of-programming-language-package-management/`](http://blog.ezyang.com/2014/08/the-fundamental-problem-of-programming-language-package-management/)

为什么有这么多该死的包管理器？它们横跨操作系统（apt、yum、pacman、Homebrew）以及编程语言（Bundler、Cabal、Composer、CPAN、CRAN、CTAN、EasyInstall、Go Get、Maven、npm、NuGet、OPAM、PEAR、pip、RubyGems 等等等等）。“普遍认为，每一种编程语言都必须需要一个包管理器。” 是什么致使每一种编程语言都跳入这个悬崖？我们为什么不能，你知道的，[重复利用](http://www.standalone-sysadmin.com/blog/2014/03/just-what-we-need-another-package-manager/)一个现有的包管理器？

你可能想到了几个理由，为什么试图使用 apt 管理你的 Ruby gems 会以泪水收场。“系统和语言包管理器完全不同！分发是经过审查的，但对于 GitHub 上投放的大多数库来说，这完全不合理。分发移动速度太慢了。每种编程语言都不同。不同的社区彼此之间不交流。分发全局安装软件包。我想控制使用哪些库。” 这些理由都是*正确*的，但它们错过了问题的本质。

编程语言包管理的根本问题在于其**去中心化**。

这种去中心化始于包管理器的核心前提：即安装软件和库，否则这些软件和库将无法在本地使用。即使有一个理想化的集中式分发来管理这些软件包，依然涉及到两个主体：分发和*构建应用程序的程序员*。然而，在现实生活中，库生态系统进一步分裂，由众多开发者提供的各种软件包组成。当然，这些软件包可能都被上传并在一个地方索引，但这并不意味着任何一个作者知道其他任何一个软件包的情况。然后就有了 Perl 世界所称的 DarkPAN：可能存在的无法计数的代码行，但我们对此一无所知，因为它们锁在专有的服务器和源代码仓库中。只有当你完全控制你应用程序中的*所有*代码行时，去中心化才能避免...但在那种情况下，你几乎不需要一个包管理器，对吧？（顺便说一句，我的行业朋友告诉我，对于像 Windows 操作系统或 Google Chrome 浏览器这样的软件项目来说，这基本上是强制性的。）

去中心化系统很难。真的非常难。除非你根据此设计你的包管理器，否则你的开发者们*一定会*陷入依赖地狱。解决这个问题没有一种“正确”的方式：我至少可以辨认出在新一代包管理器中有三种不同的方法来处理这个问题，每一种方法都有其利与弊。

**固定版本。** 或许最流行的观点是，开发者应该积极地固定软件包的版本；这种方法由 Ruby 的 Bundler、PHP 的 Composer、Python 的 virtualenv 和 pip 倡导，一般来说，任何自称受到 Ruby/node.js 社区启发的包管理器（例如 Java 的 Gradle、Rust 的 Cargo）都采用这种方法。构建的可复现性至关重要：这些包管理器通过简单地假装一旦固定了版本，生态系统就不存在了来解决去中心化问题。这种方法的主要好处在于，你始终控制着你正在运行的代码。当然，这种方法的缺点是，你始终控制着你正在运行的代码。一个很常见的情况是将依赖固定下来，然后就把它们忘记了，即使其中有重要的安全更新。保持捆绑依赖项的最新状态需要开发者的时间——通常这些时间都花在其他事情上（比如新功能）。

**稳定的发行版。** 如果捆绑要求每个应用程序开发者花时间保持依赖项的最新状态并测试它们是否与他们的应用程序正常工作，我们可能会想知道是否有一种方法可以集中这种努力。这导致了第二种思路：*集中管理*软件包仓库，创建一组已知能良好协作的软件包，并且将获得 bug 修复和安全更新，同时保持向后兼容性。在编程语言中，这种模式并不常见：我所知道的只有 Anaconda 用于 Python 和 Stackage 用于 Haskell。但是如果我们仔细观察，这个模型与大多数操作系统发行版的模型*完全相同*。作为系统管理员，我经常建议用户尽可能使用操作系统提供的库。他们不会在我们进行发布升级之前采用不向后兼容的更改，同时您仍将获得您的代码的 bug 修复和安全更新。（您将无法得到最新的炫酷功能，但这与稳定性本质上是相矛盾的！）

**拥抱去中心化。** 直到现在，这两种方法都抛弃了去中心化，需要一个中央权威，无论是应用开发者还是分发管理者，来进行更新。这是不是舍弃孩子而保留水中的？中心化的主要缺点是维护稳定分发或保持单个应用程序更新所需的大量*工作*。此外，人们可能不会期望整个宇宙都能彼此兼容，但这并不能阻止某些软件包的子集彼此一起使用。一个理想的去中心化生态系统将问题分布到参与系统的每个人身上，以确定哪些软件包的子集*能够*共同使用。这也引出了编程语言包管理的根本未解之谜：

> *我们如何创建一个能够运作的去中心化包生态系统？*

这里有几件事情可以帮助：

1.  **更强的依赖封装。** 依赖地狱之所以如此阴险，其中一个原因是一个软件包的依赖通常是其外部面向 API 的不可分割的一部分：因此，依赖的选择不是一个局部选择，而是一个全局选择，影响整个应用程序。当然，如果一个库在内部使用某些库，但这个选择完全是实现细节，这*不应该*导致任何全局约束。Node.js 的 NPM 将这种选择推向其逻辑极限：默认情况下，它根本不会对依赖进行去重，使每个库都拥有其依赖的副本。虽然我对复制所有内容（它在 Java/Maven 生态系统中确实存在）有些怀疑，但我完全同意保持依赖约束局部化可以提高*可组合性*。

1.  **推进语义化版本控制。** 在去中心化系统中，图书馆作者提供*准确*的信息尤为重要，以便工具和用户能够做出知情决策。虚构的版本范围和艺术化的版本号增加了已经存在的难题（正如我在[上一篇文章](http://blog.ezyang.com/2014/08/whats-a-module-system-good-for-anyway/)中提到的）。如果你可以[强制执行语义化版本控制](http://bndtools.org/)，或者更好地说，放弃语义版本并记录真实的、*类型级*的接口依赖关系，我们的工具可以做出更好的选择。在去中心化系统中信息的黄金标准是，“软件包 A 与软件包 B 兼容”，这种信息通常很难（或者对于动态类型系统来说是不可能的）计算。

1.  **中心化是一种特例。** 分布式系统的要点是每个参与者都可以为自己选择合适的策略。这包括维护自己的中央权威，或者推迟到别人的中央权威：中心化只是一种特例。如果我们怀疑用户将尝试创建自己的操作系统风格的稳定发行版，我们需要给予他们相应的工具……并且让这些工具易于使用！

长期以来，源代码控制管理生态系统完全集中在中心化系统上。分布式版本控制系统如 Git 从根本上改变了这一格局：尽管对于非技术用户而言，Git 可能比 Subversion 更难使用，但去中心化的好处却是多样化的。包管理的 Git 尚不存在：如果有人告诉你包管理问题已解决，只需重新实现 Bundler，我恳求你：也要考虑去中心化！