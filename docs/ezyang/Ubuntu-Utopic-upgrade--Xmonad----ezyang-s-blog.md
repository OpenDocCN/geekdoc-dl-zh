<!--yml

category: 未分类

date: 2024-07-01 18:17:11

-->

# [Ubuntu Utopic 升级（Xmonad）](http://blog.ezyang.com/2014/12/ubuntu-utopic-upgrade-xmonad/)

> 来源：[`blog.ezyang.com/2014/12/ubuntu-utopic-upgrade-xmonad/`](http://blog.ezyang.com/2014/12/ubuntu-utopic-upgrade-xmonad/)

## Ubuntu Utopic 升级（Xmonad）

我终于升级到了 Utopic 版本。[一年前](http://blog.ezyang.com/2013/10/xmonad-and-media-keys-on-saucy/)我报告说 gnome-settings-daemon 不再提供按键抓取支持。这最终在 Trusty 版本中被撤销，保留了所有人的媒体键。

很抱歉在 Ubuntu Utopic 中报告，传统的按键抓取器不再存在：

```
------------------------------------------------------------
revno: 4015 [merge]
author: William Hua <william.hua@canonical.com>
committer: Tarmac
branch nick: trunk
timestamp: Tue 2014-02-18 18:22:53 +0000
message:
  Revert the legacy key grabber. Fixes: https://bugs.launchpad.net/bugs/1226962.

```

看起来 Unity 团队已经将 gnome-settings-daemon 分叉成 unity-settings-daemon（实际上这个分叉已经发生在 Trusty 版本），截至到 Utopic 版本，gnome-settings-daemon 和 gnome-control-center 已经被剔除，改为使用 unity-settings-daemon 和 unity-control-center。这使我们重新回到一年前的情况。

我目前还没有解决这个（相当大的）问题的方法。但是，我已经为升级中出现的一些较小问题提供了解决方案：

+   如果你的鼠标光标不可见，尝试运行 `gsettings set org.gnome.settings-daemon.plugins.cursor active false`

+   如果你不喜欢 GTK 文件对话框不再将文件夹排序在前面，尝试运行 `gsettings set org.gtk.Settings.FileChooser sort-directories-first true`。（[感谢](http://gexperts.com/wp/gnome-3-12-filesnautilus-sort-folders-before-files-issues/)）

+   并且需要重申的是，替换所有对 gnome-settings-daemon 的调用为 unity-settings-daemon，并使用 unity-control-panel 进行一般配置。
