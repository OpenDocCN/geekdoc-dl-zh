<!--yml

category: 未分类

date: 2024-07-01 18:17:25

-->

# Ubuntu Quantal 升级（Thinkpad/Xmonad）：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/10/ubuntu-quantal-upgrade-thinkpadxmonad/`](http://blog.ezyang.com/2012/10/ubuntu-quantal-upgrade-thinkpadxmonad/)

## Ubuntu Quantal 升级（Thinkpad/Xmonad）

十月已至，带来了另一个 Ubuntu 发布版（12.10）。我终于屈服并重新安装了我的系统为 64 位（告别 32 位），主要是因为我的升级系统上的图形出现了问题。据我所知，lightdm 在启动后立即崩溃，我无法确定在我的大量配置中哪里出了问题。我还开始加密我的家目录。

+   所有 [fstab 挂载项](http://askubuntu.com/questions/193524/how-to-hide-bind-mounts-in-nautilus) 现在都显示在 Nautilus 中。正确的解决方法似乎是不要将这些挂载项放在 `/media`、`/mnt` 或 `/home` 中，这样它们就不会被检测到。

+   在 rxvt-unicode 中，字体问题仍然是一个棘手的问题。我不得不从 `URxvt.letterSpace: -1` 切换到 `URxvt.letterSpace: -2` 以保持一切正常运作，但字体看起来仍然有不可思议的差异。（我还没搞清楚原因，但新的世界秩序并不是完全的眼中钉，所以我暂时放弃了。）还有 [一个补丁](http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=628167) 可以修复这个问题（参考 [这个 libxft2 的 bug](https://bugs.freedesktop.org/show_bug.cgi?id=47178)），但我发现对于 DejaVu 字体来说，letterSpace 的小技巧是等效的。

+   当你手动暂停你的笔记本并过快关闭盖子时，Ubuntu 也会注册关闭笔记本事件，所以当你恢复时，它会重新暂停！幸运的是，这没什么大问题；如果你再次按下电源按钮，它就会正确恢复。你也可以通过在电源设置中关闭盖子关闭后恢复来解决这个问题。

+   在恢复后，网络管理器小程序不再准确反映你连接到哪个网络（它认为你已连接，但不知道连接到哪里，或者信号强度是多少）。这基本上是无害的，但有点烦人；如果有人解决了这个问题，请告诉我！

+   休眠功能依然无法正常工作，虽然我并没有特别努力地去解决这个问题。

+   Firefox 一度*非常*缓慢，所以我 [重置了它](http://support.mozilla.org/en-US/kb/reset-preferences-fix-problems)。然后它又变快了。天哪！如果你发现 Firefox 非常慢，这值得一试。

+   GHC 现在是 7.4.2，所以你需要重新构建。“我们什么时候可以获得我们的 7.6 新功能呢！”

我的实验室同事们继续取笑我没有转向使用 Arch。我们看看吧…
