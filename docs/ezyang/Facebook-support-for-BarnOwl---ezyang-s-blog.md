<!--yml

category: 未分类

date: 2024-07-01 18:17:42

-->

# BarnOwl 的 Facebook 支持：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/07/facebook-support-for-barnowl/`](http://blog.ezyang.com/2011/07/facebook-support-for-barnowl/)

## BarnOwl 的 Facebook 支持

这是专为 MIT 的人群准备的。今天早上，我满意地完成了我对 [BarnOwl 的 Facebook 模块](https://github.com/ezyang/barnowl) 的修改（我的满意度表现在对 Facebook API 调用的异步支持上，即不再随机冻结！）。但是，让它在 Linerva 上运行有点复杂，所以这里有个详细的步骤。

1.  使用 [MIT 网站上的说明](http://sipb.mit.edu/doc/cpan/) 设置本地 CPAN 安装，使用 `local::lib`。不要忘记将设置代码添加到 `.bashrc.mine`，而不是 `.bashrc`，然后进行源操作。不要忘记遵循先决条件：否则，CPAN 将会提示很多信息。

1.  安装所有你需要的 CPAN 依赖项。对于 Facebook 模块，这意味着需要安装 `Facebook::Graph` 和 `AnyEvent::HTTP`。我建议使用 `notest`，因为 `Any::Moose` 在 Linerva 上似乎会失败一个无害的测试。`Facebook::Graph` 失败了几个测试，但不用担心，因为我们将使用预打包版本。如果你想使用其他模块，你也需要在 CPAN 中安装它们。

1.  克隆 BarnOwl 到本地目录 (`git clone git://github.com/ezyang/barnowl.git barnowl`)，然后运行 `./autogen.sh`，`configure` 和 `make`。

1.  使用 `./barnowl` 运行，然后输入命令 `:facebook-auth` 并按照说明操作！

欢迎使用 Facebook！

*附言.* 我真的很惊讶，竟然没有一种流行的命令式语言有绿色线程和抢占式调度，允许你实际上编写看起来是阻塞的代码，尽管它在内部使用事件循环。也许这是因为在保证安全性的同时进行抢占是很难的……

*已知的 bug.* 读/写验证 bug 已修复。我们似乎在 BarnOwl 的事件循环实现中触发了一些 bug，这导致每天都会出现崩溃（这使得调试变得困难）。保持备份的 BarnOwl 实例是个好主意。
