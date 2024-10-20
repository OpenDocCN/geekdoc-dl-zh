<!--yml

category: 未分类

date: 2024-07-01 18:17:39

-->

# Ubuntu Oneiric 升级（Thinkpad/Xmonad）：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/11/ubuntu-oneiric-thinkpad-xmonad/`](http://blog.ezyang.com/2011/11/ubuntu-oneiric-thinkpad-xmonad/)

## Ubuntu Oneiric 升级（Thinkpad/Xmonad）

我今天从 Ubuntu Natty Narwhal 升级到 Oneiric Ocelot（11.10）。很多东西都出问题了。具体来说：

+   “无法计算升级。” 没有指出错误的迹象；在我的情况下，错误最终是由于旧的孤儿 OpenAFS 内核模块（没有相应的内核模块存在）。我也趁机清理了我的 PPA。

+   “阅读变更日志。” `apt-listchanges`并不特别有用，我也不知道为什么我安装了它。但是当阅读变更日志的时间比安装软件还长时，真的很痛苦。Geoffrey 建议`` gdb -p `pgrep apt-listchanges` ``然后强制它调用`exit(0)`，这个方法奏效。我不得不多次这样做；以为它会无限循环。

+   图标无法工作，菜单很丑陋。去“系统设置 > 外观”设置一个新主题；很可能你的旧主题已经消失了。这个[AskUbuntu](http://askubuntu.com/questions/59791/how-do-i-fix-my-theme)问题给了一个线索。

+   网络管理器停止工作。由于某种难以理解的原因，默认的 NetworkManager 配置文件`/etc/NetworkManager/NetworkManager.conf`中对`ifupdown`有`managed=false`的设定。切换回 true。

+   新的窗口管理器，默认会至少让你试用 Unity 一次。只需确保你从小齿轮图标中选择正确的窗口管理器。

+   `gnome-power-manager`消失了。如果你修复了图标，加载`gnome-settings-daemon`时会出现一个不太有用的图标。

+   “等待网络配置。” 这里有很多建议。我的`/var/run`和`/var/lock`被损坏了，所以我[按照这些说明操作了](http://uksysadmin.wordpress.com/2011/10/14/upgrade-to-ubuntu-11-10-problem-waiting-for-network-configuration-then-black-screen-solution/)。我还听说你应该从`/etc/network/interfaces`中移除`wlan0`并从`/etc/udev/rules.d70-persistent-net.rules`中删除它。我还为了保险起见注释了`/init/failsafe.conf`中的休眠。

+   默认的 GHC 版本是 7.0.3！清除你的`.cabal`（但保留`.cabal/config`），重新安装 Haskell 平台。别忘了确保安装了性能分析库，并获取`xmonad`和`xmonad-contrib`。请注意，之前的 haskell-platform 安装可能会相当混乱，因为缺少 GHC 6 二进制文件（你可以重新安装它们，但看起来它们已经被替换了。）

+   ACPI 停止了关于 X 的知识，所以如果你有处理旋转的脚本，请执行`/usr/share/acpi-support/power-funcs`并运行`getXuser`和`getXconsole`

+   DBUS 没有启动。这是由于残留的 pid 和 socket 文件引起的，请参见[此 bug](https://bugs.launchpad.net/ubuntu/+source/dbus/+bug/811441)

+   每次启动时神秘地在我的根目录驱动上执行 fsck。检查你在`/etc/fstab`中的`pass`参数；应该是`0`。

+   Redshift 神秘地被 xrandr 调用重置；通过在运行 xrandr 后立即调用 oneshot 来解决。

+   不确定是否与升级有关，但修复了一个令人讨厌的问题，即在启动时暂停检查（以防从休眠中恢复）需要很长时间。在`/etc/initramfs-tools/conf.d/resume`中设置`resume`为正确的交换区，并使用极大的决心`update-initramfs -u`）。

未解决的烦恼：[X11 在 DBUS 中自动启动](https://bugs.launchpad.net/ubuntu/+source/dbus/+bug/812940)，电源图标不始终正确显示 AC 信息，在 stalonetray 中太小，xmobar 不支持同时百分比电池和 AC 着色（我有一个补丁），从头构建的 totem 会段错误。
