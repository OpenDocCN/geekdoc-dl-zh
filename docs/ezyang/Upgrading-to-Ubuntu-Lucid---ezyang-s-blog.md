<!--yml

category: 未分类

date: 2024-07-01 18:18:18

-->

# Upgrading to Ubuntu Lucid : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/05/upgrading-to-ubuntu-lucid/`](http://blog.ezyang.com/2010/05/upgrading-to-ubuntu-lucid/)

现在学期结束了，我终于升级了我的笔记本电脑到 Ubuntu 10.04 LTS，即 Lucid Lynx。这个过程比[Karmic 的情况](http://ezyang.com/karmic.html)要顺利得多，但仍然有一些小问题。

*Etckeeper.* 一如既往，您在尝试升级发布之前应将`AVOID_COMMIT_BEFORE_INSTALL`设置为 0，因为 etckeeper 钩子将被多次调用，而最令人恼火的莫过于收到通知：“etckeeper 中止安装因为存在未提交的更改，请您自行提交它们”，因为那根本行不通。

这一次，出现了一个不同但又搞笑的错误：

```
/etc/etckeeper/post-install.d/50vcs-commit: 20:
etckeeper: Argument list too long

```

这已被[报告为 Bug #574244](https://bugs.launchpad.net/ubuntu/+source/etckeeper/+bug/574244)。尽管这是一个不祥的警告，但实际上相当无害，您可以使用以下方式完成升级：

```
aptitude update
aptitude full-upgrade

```

我因为破碎的 DNS 而不得不重新启动网络；效果因人而异。

*无线密钥管理.* 我还没有解决这个问题，但基本症状是 Ubuntu 网络管理器无法记住您为受保护网络提供的 WEP 密钥。（我知道您还在校园的麻省理工学院的学生们可能对此并不太关心。）这似乎是一个相当普遍的问题，因为有人在复活早期的[这个](https://bugs.launchpad.net/ubuntu/+source/network-manager/+bug/271097) [bug](https://bugs.launchpad.net/ubuntu/+source/network-manager/+bug/36651)，虽然这些问题早就存在了。 （典型的糟糕 bug 报告风格，用户们附加在旧的 bug 报告上，而他们实际上应该为 Lucid 提出新的回归。）

从我的调查中，我已经验证了[与密钥环守护程序的连接无法正常工作](http://ubuntuforums.org/showthread.php?t=1459804)。有[一种解决方法正在流传](https://bugs.launchpad.net/ubuntu/+source/seahorse/+bug/553032/comments/8)，其中您可以将启动命令从“gnome-keyring-daemon --start --components=pkcs11”更改为只是“gnome-keyring-daemon”，尽管我怀疑这并不是真正的“正确”方法，而且在我这里也不起作用。

*PHP.* Ubuntu Lucid 最显著地升级了 PHP 5.3.2，但他们还调整了一些默认设置。在我的情况下，`log_errors`为我的脚本引起了相当有趣的行为，因此我已经将我的脚本编码为显式关闭此 ini 设置。您应该在升级前保存`php -i`的输出副本，并与升级后的输出进行比较。
