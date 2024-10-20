<!--yml

类别：未分类

日期：2024-07-01 18:18:29

-->

# Sup：极客的邮件：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/01/sup/`](http://blog.ezyang.com/2010/01/sup/)

## Sup：极客的邮件

**更新（2012 年 9 月 1 日）：** 本文已有些过时。我计划写一篇更新文章，但主要新点是：如果你有 SSD，Sup 的启动时间非常快，所以你可以轻松在笔记本上运行它，并且你应该使用 maildir-sync 分支，它提供了标签的反向同步（或者我的 patchset，非常棒但需要打磨和发布）。

* * *

我使用 [Sup](http://sup.rubyforge.org/) 并且我喜欢它；不要在意朋友们的嘲笑，他们发现当他们的收件箱超过十万封邮件或者管理索引被抹掉时，收件箱变得非常缓慢。通往电子邮件极乐的道路并不轻松，以及一个十封邮件的收件箱，所以这里有一个为你设置 Sup 的逐步指南。我们将使用顶尖的一切，这意味着从下一个分支的 Git 检出运行，使用 Xapian 索引，并使用 OfflineImap。

1.  获取一个可以 SSH 连接并且运行 screen 的服务器。Sup 的启动时间不算短，所以绕过它的最佳方法是永远不关闭这个进程。这还可以避免你需要因为 ISP 的敏感 SMTP 切换而麻烦。

1.  设置 [OfflineIMAP](http://software.complete.org/software/projects/show/offlineimap) 来下载你的邮件。IMAP 通常很慢，我发现我对我的邮件很重视，希望有一个本地备份。`.offlineimaprc` 的配置稍微麻烦（我在得到正确设置之前两次搞砸了）；看本文末尾获取我最终使用的模板。由于导入过程会花费很长时间，请在运行之前仔细检查你的配置。

1.  设置 Ruby 环境；Sup 在 Ruby 1.8 上可以工作，但在 1.9 上不行。如果你使用的是 Ubuntu Jaunty，你需要[手动安装 RubyGems](http://docs.rubygems.org/read/chapter/3)；在 Karmic 上，打包版本可以正常工作。

1.  获取依赖 gems。这就像使用 `gem install sup` 安装 Sup gem，然后仅移除 Sup gem 一样简单。

1.  使用`git clone git://gitorious.org/sup/mainline.git sup`从 Git 获取 Sup 的副本。在你的 shell 配置文件（Bash 用户为 `.bashrc`）中，设置 PATH 包括 $SUPDIR/bin 和 RUBYLIB 包括 $SUPDIR/lib。添加示例行的一组可以在本帖子底部找到。

1.  运行 `sup-config` 来设置通用配置。当它提示你添加一个新源时，添加一个 Maildir 源，指定一个目录内的文件夹，这个目录是你要求 OfflineImap 同步到的（例如，我要求 OfflineImap 下载我的邮件到 ~/Mail/MIT，所以 ~/Mail/MIT/INBOX 将是我的 Maildir 的有效文件夹）。当我转换到 Sup 后，我停止使用服务器端文件夹，所以这是我唯一为之设置源的文件夹；如果你仍然想使用它们，你需要将它们分别添加为独立的源。

1.  打开你喜欢的编辑器中的`.sup/config.yaml`文件，在新的一行中添加`:index: xapian`。作为更为可靠的方法，另一种选择是设置一个环境变量，但我更倾向于这种方法。

1.  当你开始使用 Sup 时，我强烈建议你设置一些[钩子](http://sup.rubyforge.org/wiki/wiki.pl?Hooks)。由于你在使用 OfflineImap，执行 OfflineImap 的 `before-poll` 钩子在进行轮询前是必要的。同时，“自动备份你的标签” `startup` 钩子也是必须的。

1.  在 screen 会话中加载 `sup` 并享受吧！

`.offlineimaprc` 模板：

```
[general]
accounts = MIT

[Account MIT]
localrepository = LocalMIT
remoterepository = RemoteMIT

[Repository LocalMIT]
type = Maildir
localfolders = ~/Mail/MIT

[Repository RemoteMIT]
type = IMAP
ssl = yes
remotehost = $HOST
remoteuser = $USERNAME
remotepass = $PASSWORD

```

`.bashrc` 模板（假设 Sup 存在于 `$HOME/Dev/sup` 中）：

```
export PATH=$HOME/Dev/sup/bin:$PATH
export RUBYLIB=$HOME/Dev/sup/lib

```
