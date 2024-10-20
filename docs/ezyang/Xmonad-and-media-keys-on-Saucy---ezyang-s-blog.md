<!--yml

category: 未分类

date: 2024-07-01 18:17:16

-->

# Xmonad 和 Saucy 上的媒体键：ezyang 的博客

> 来源：[`blog.ezyang.com/2013/10/xmonad-and-media-keys-on-saucy/`](http://blog.ezyang.com/2013/10/xmonad-and-media-keys-on-saucy/)

Ubuntu 继续破坏完全正常的软件，在我最近升级到 Saucy Salamander 时，我惊讶地发现我的媒体键（例如音量键，fn（功能）键，挂起按钮等）停止工作。当然，如果我使用 Unity 登录我的用户，它就可以正常工作，但是谁愿意使用那样一个愚蠢的窗口管理器呢...

根本问题在于，根据 [这些 Arch Linux 论坛帖子](https://bbs.archlinux.org/viewtopic.php?pid=1262471)，Gnome 已经将媒体键支持从 `gnome-settings-daemon`（任何自尊的 Xmonad 用户都会生成）移到他们的窗口管理器中。当然，这是不好的，因为我不想使用他们的窗口管理器！

目前看来，恢复此功能的最简单方法是运行 3.6 版本的 gnome-settings-daemon。幸运的是，至少对于 Saucy，有一些适用于你的架构的 3.6 构建版本可用（你还需要 gnome-control-center，因为它依赖于 gnome-settings-daemon）：

一旦你下载了适当的 deb 文件，运行 `dpkg -i $DEBFILE` 然后运行 `apt-mark hold gnome-control-center gnome-settings-daemon` 应该能解决问题。你应该运行 `aptitude upgrade` 来确保没有破坏其他依赖关系（例如 `gnome-shell`）。（高级用户可以将 deb 文件添加到本地仓库，然后通过 `apt-get` 明确降级。）

未来，我们可能会被迫在其他软件包中重新实现媒体键绑定，并且如果能以某种方式标准化这一点将是很好的。Linux Mint 已经分支了 gnome-settings-daemon，使用他们的 [cinnamon-settings-daemon](https://github.com/linuxmint/cinnamon-settings-daemon)，但我没有尝试过，也不知道它的工作情况如何。

**更新。** Trusty 版本更新了这个软件包的版本，恢复了支持，所以我通过我的 PPA 提供后端支持 [via my PPA.](https://launchpad.net/~ezyang/+archive/ppa)
