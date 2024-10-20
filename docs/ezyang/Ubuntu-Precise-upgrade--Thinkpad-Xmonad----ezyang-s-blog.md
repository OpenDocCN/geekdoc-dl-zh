<!--yml

category: 未分类

date: 2024-07-01 18:17:30

-->

# Ubuntu Precise 升级（Thinkpad/Xmonad）：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/05/ubuntu-precise-upgrade-thinkpad-xmonad/`](http://blog.ezyang.com/2012/05/ubuntu-precise-upgrade-thinkpad-xmonad/)

## Ubuntu Precise 升级（Thinkpad/Xmonad）

又到了 Ubuntu 升级的时候。我从 Ubuntu Oneiric Ocelot 升级到了 Ubuntu Precise Pangolin（12.04），这是一个 LTS 版本。几乎没有什么东西出了问题（万岁！）

+   Monospace 字体变成了一个全新的字体，字形非常宽。旧字体是 DejaVuSansMono，我又切换回去了。

+   Xournal 停止编译；不知何故链接器行为发生了变化，现在需要手动指定链接器标志。

+   [gnome-keyring](https://bugs.launchpad.net/ubuntu/+source/gnome-keyring/+bug/932177) 对于非 Unity 用户来说启动不正常。根本问题似乎是由于 [Gnome 的打包错误](http://lists.debian.org/debian-lint-maint/2009/07/msg00129.html)，但将 `` eval `gnome-keyring-daemon -s` `` 添加到我的 `.xsession` 文件后问题解决了。

+   电池图标消失了！我猜是某个守护程序未能正常运行，但由于我有一个很好的 xmobar 显示，我并不为它的失去而感到悲伤。

+   默认的 GHC 版本是 GHC 7.4.1！是时候重新构建了；暂时还没有 Haskell 平台。 （请注意，GHC 7.4.1 不支持 gold 链接器；这是 `chunk-size` 错误。）

我还从之前的 LTS Lucid Lynx 升级了我的桌面。

+   我遇到了很多无效签名错误，这导致升级脚本无法运行。我通过卸载几乎所有的 PPAs 来解决了这个问题。

+   Offlineimap 需要更新，因为它依赖的一些 Python 库有不兼容的改动（即 imap 库）。

+   VirtualBox 搞乱了它的版本号，里面包含一个被禁止的 [下划线](https://bugs.launchpad.net/ubuntu/+source/dpkg/+bug/613018)。手动编辑文件将其删除似乎解决了问题。
