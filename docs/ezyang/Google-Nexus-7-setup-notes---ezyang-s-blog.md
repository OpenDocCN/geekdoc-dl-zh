<!--yml

category: 未分类

date: 2024-07-01 18:17:24

-->

# Google Nexus 7 设置笔记：ezyang's 博客

> 来源：[`blog.ezyang.com/2012/12/googl-nexus-7-setup-notes/`](http://blog.ezyang.com/2012/12/googl-nexus-7-setup-notes/)

## Google Nexus 7 设置笔记

我在寒假期间购买了一台 Google Nexus 7（仅 Wi-Fi 版）。我不太喜欢购买新设备：它们通常需要大量工作来按我的喜好设置。以下是一些笔记：

+   在 Linux 上越狱设备仍然有些麻烦。最终，最简单的方法可能是找一台 Windows 电脑并使用 [Nexus Root Toolkit](http://www.wugfresh.com/nrt/)。这个工具有些不稳定；如果第一次检测失败，可以再试一次检测代码。

+   在 Linux 上进行文件传输真是痛苦。我已经通过 SSHDroid 使用 SCP 进行了工作；我还尝试了 DropBear SSH 服务器，但它们没有附带 scp 二进制文件，因此对文件传输目的来说几乎没有用。SSHDroid 并没有 out-of-the-box 解决：我需要应用 [comment 14](http://code.google.com/p/droidsshd/issues/detail?id=2#c14) 来使真正的 scp 二进制文件在路径中被找到。默认情况下，这些应用程序配置为接受密码验证（甚至不是键盘交互式！）使用极其弱的默认密码：确保您禁用了这一功能。仍在寻找一个好的 rsync 实现。在 USB 方面，Ubuntu/Gnome/Nautilus 在 PTP 模式下本地识别 Nexus，但当我尝试复制文件时却挂起了。Ubuntu 12.10 对 MTP 支持较少，但是 go-mtpfs 在现代 libmtp 的支持下表现还算不错。Adam Glasgall 已经为 Quantal 打包了 [libmtp](https://launchpad.net/~aglasgall/+archive/libmtp)，所以添加他的 PPA，然后 [按照 go-mtpfs 的安装说明](https://github.com/hanwen/go-mtpfs) 进行安装。**更新：** 直接向可移动媒体传输文件也效果不错。

+   这款平板确实感觉像一部手机，因为两者都在 Android 平台上。但是没有 3G 意味着离线功能变得更加重要，而更大的屏幕使某些类型的应用程序使用起来更加愉快（**更新：** 我已经选择了 MX Player 作为我的视频播放器，因为它支持高级字幕 Alpha 和 MKV 文件。不幸的是，它不支持深色（例如 10 位）。）

+   微型 USB 到 USB OTG 电缆非常方便，特别是用于连接键盘或外部媒体。我敢说，它比外壳更为重要。请注意，微型 USB 端口无法为具有高功率需求的 USB 设备供电（例如旋转碟外置硬盘），因此您需要使用带电 USB 集线器连接它们。这会导致挂载时的一个症状是如果您尝试挂载功率不足的硬盘，目录列表会持续为空。它还可能发出点击声：这对硬盘可能不是好事。我使用 USB-OTG 进行挂载。

+   我试图将我的 Mendeley 论文数据库镜像到我的平板电脑上，但这相当困难。我一直在尝试使用 Referey，这是一个适用于安卓设备的 Mendeley 客户端，但它要求我以某种方式传播我的 Mendeley SQLite 数据库和所有的 PDF 文件。Dropbox 在这里看起来是一个很好的选择，但官方 Dropbox 客户端不支持保持整个文件夹同步（只支持收藏的文件）。如果你和我一样，不确定将阅读哪些论文，你必须使用其他方法，比如 Dropsync。（顺便说一句，如果你像我一样，有个聪明的主意，把 SQLite 数据库和 PDF 放在一起，这样它们就会在一个文件夹中同步，永远不要“整理”：Mendeley 会高兴地将你的 SQLite 数据库删除为“外来物”）。Mendeley 和 Dropbox 在各种方面似乎互动不良（区分大小写；此外，Mendeley 喜欢生成过长的文件名，而 Dropbox 则愚蠢而乐意接受它们）。

+   “打开窗口”按钮似乎没有正确尊重应用程序通过其自己的意愿关闭时的状态（即通过应用程序本身支持的退出按钮）。这有点恼人。

哦对了，祝你新年快乐。 :)

**更新：** 我的 Nexus 7 突然变砖了。幸运的是，一旦手机解锁，重新刷新镜像非常容易（并且我没有丢失数据，这通常会在首次解锁手机时发生）。我是在手机处于引导程序状态时（同时按住两个音量键并开机），通过 `fastboot update image-nakasi-jop40d.zip` 来完成的，然后按照[这里](http://web.archive.org/web/20120727033322/http://robertpitt.me/2012/07/rooting-the-nexus-7-via-linux/)的最后一组步骤来重新安装 SuperSu（即通过 fastboot 进入 ClockworkMod，然后通过 sideload 安装 SuperSu）。
