<!--yml

category: 未分类

日期：2024-07-01 18:18:26

-->

# 三步曲的第三方无人值守升级：ezyang’s 博客

> 来源：[`blog.ezyang.com/2010/03/third-party-unattended-upgrade/`](http://blog.ezyang.com/2010/03/third-party-unattended-upgrade/)

## 三步曲的第三方无人值守升级

[无人值守升级](http://packages.ubuntu.com/karmic/unattended-upgrades) 是一个非常方便的小包，它会在启用后自动安装更新。没有严肃的系统管理员会使用这个（你确实在将更新推送到服务器之前进行测试，对吧？），但对于许多个人用途来说，自动更新确实是你想要的；如果你运行 `sudo aptitude full-upgrade` 而不阅读变更日志，那么你最好开启无人值守升级。你可以通过向 `/etc/apt/apt.conf.d/10periodic` 添加行 `APT::Periodic::Unattended-Upgrade "1"` 来实现这一点（感谢 Ken！）

当然，默认配置是在 `/etc/apt/apt.conf.d/50unattended-upgrades` 中从他们的安全仓库拉取更新，而且他们只为普通更新提供了一行注释掉的配置。人们[已经问过](http://ubuntuforums.org/showthread.php?t=1401845)，“那么，我如何从其他仓库拉取自动更新？”也许你已经安装了 Chromium 每日构建版；每天看到“您有更新”的图标可能有点烦人。

好吧，这就是如何做到的：

1.  找出你感兴趣的 PPA 指向的网址。你可以通过查看 `/etc/apt/sources.list` 或 `/etc/apt/sources.list.d/` 来找到这些信息（如果你曾手动添加过一个 PPA，则查看前者；如果你使用了 `add-apt-repository`，则查看后者）。

1.  在浏览器中导航到该网址。导航到 `dists`，然后导航到你正在运行的发行版名称（对我来说是 `karmic`）。最后，点击 `Release`。（对于那些想要直接输入整个网址的人，它是 [`example.com/apt/dists/karmic/Release`](http://example.com/apt/dists/karmic/Release)）。

1.  你将会看到一些字段 `Fieldname: Value`。找到 `Origin` 和 `Suite` 字段。这两个值就是放入 Allowed-Origins 中的值。

例如，[Ksplice 仓库](http://www.ksplice.com/apt/dists/karmic/Release) 包含以下的 `Release` 文件：

```
Origin: Ksplice
Label: Ksplice
Suite: karmic
Codename: karmic
Version: 9.10
Date: Sun, 07 Feb 2010 20:51:12 +0000
Architectures: amd64 i386
Components: ksplice
Description: Ksplice packages for Ubuntu 9.10 karmic

```

这翻译成以下配置：

```
Unattended-Upgrade::Allowed-Origins {
       "Ksplice karmic";
};

```

就是这样！前去通过及时更新使你的系统更加安全。

*额外小贴士*。你可以通过编辑 `/etc/uptrack/uptrack.conf` 并设置 `autoinstall = yes` 来开启 Ksplice 的无人值守[内核更新](http://www.ksplice.com/)。
