<!--yml

分类：未分类

日期：2024-07-01 18:17:15

-->

# 如何维护配置文件的原始副本：ezyang 的博客

> 来源：[`blog.ezyang.com/2014/01/how-to-maintain-a-pristine-copy-of-your-configuration-files/`](http://blog.ezyang.com/2014/01/how-to-maintain-a-pristine-copy-of-your-configuration-files/)

[etckeeper](http://joeyh.name/code/etckeeper/) 是一个非常好的工具，用于将你的 /etc 目录置于版本控制之下，但它不会告诉你一件事情，即你的配置与配置的原始版本之间的差异（如果你在系统上安装了相同的软件包，但没有更改任何配置）。[人们曾希望实现这一点](https://blueprints.launchpad.net/ubuntu/+spec/foundations-q-dpkg-pristine-conffiles)，但我找不到真正能够实现这一点的工具。一个月前，我找到了在 etckeeper 中用 Git 仓库实现这一目标的一个简单而好的方法。这个想法是维护一个原始分支，在升级发生时，自动将补丁（自动生成的）应用到原始分支上。这个过程最适合在新安装的系统上运行，因为如果你没有从一开始跟踪原始分支，我没有很好的方法来重建历史。

这是一个示例：

1.  安装 etckeeper。最好使用 etckeeper 1.10 或更高版本，但如果没有，你应该从最新版本中替换 [30store-metadata](https://github.com/joeyh/etckeeper/blob/master/pre-commit.d/30store-metadata) 的副本。这很重要，因为在 1.10 之前的版本中，元数据存储包含了被忽略的文件，这意味着你将会得到很多假冲突。

1.  初始化 Git 仓库，使用 `etckeeper init` 命令，并进行首次提交 `git commit`。

1.  创建一个原始分支：`git branch pristine`（但保留在主分支上）

1.  修改 etckeeper 配置，使得 `VCS="git"`，`AVOID_DAILY_AUTOCOMMITS=1` 和 `AVOID_COMMIT_BEFORE_INSTALL=1`：

    ```
    diff --git a/etckeeper/etckeeper.conf b/etckeeper/etckeeper.conf
    index aedf20b..99b4e43 100644
    --- a/etckeeper/etckeeper.conf
    +++ b/etckeeper/etckeeper.conf
    @@ -1,7 +1,7 @@
     # The VCS to use.
     #VCS="hg"
    -#VCS="git"
    -VCS="bzr"
    +VCS="git"
    +#VCS="bzr"
     #VCS="darcs"

     # Options passed to git commit when run by etckeeper.
    @@ -18,7 +18,7 @@ DARCS_COMMIT_OPTIONS="-a"

     # Uncomment to avoid etckeeper committing existing changes
     # to /etc automatically once per day.
    -#AVOID_DAILY_AUTOCOMMITS=1
    +AVOID_DAILY_AUTOCOMMITS=1

     # Uncomment the following to avoid special file warning
     # (the option is enabled automatically by cronjob regardless).
    @@ -27,7 +27,7 @@ DARCS_COMMIT_OPTIONS="-a"
     # Uncomment to avoid etckeeper committing existing changes to
     # /etc before installation. It will cancel the installation,
     # so you can commit the changes by hand.
    -#AVOID_COMMIT_BEFORE_INSTALL=1
    +AVOID_COMMIT_BEFORE_INSTALL=1

     # The high-level package manager that's being used.
     # (apt, pacman-g2, yum, zypper etc)

    ```

1.  将 [这个补丁应用到 etckeeper/commit.d/50vcs-commit](http://web.mit.edu/~ezyang/Public/etckeeper-pristine.patch)。这个补丁负责保持原始分支的最新状态（下面有更多解释）。

1.  创建一个 `.gitattributes` 文件，内容为 `.etckeeper merge=union`。这样做可以使元数据文件上的合并使用联合策略，显著减少假冲突：

    ```
    diff --git a/.gitattributes b/.gitattributes
    new file mode 100644
    index 0000000..b7a1f4d
    --- /dev/null
    +++ b/.gitattributes
    @@ -0,0 +1 @@
    +.etckeeper merge=union

    ```

1.  提交这些更改。

1.  允许推送到已检出的 `/etc` 目录，通过运行 `git config receive.denyCurrentBranch warn` 来配置。

1.  完成！尝试安装一个包含一些配置的软件包，然后在 `/etc` 目录中运行 `sudo gitk` 来查看结果。你可以通过运行 `sudo git diff pristine master` 来进行差异比较。

所以，底层是怎么回事呢？过去我想要设置这样一个环境的最大问题是，你希望包管理器将其更改应用到原始的`/etc`，这样你就可以在生产版本上自己合并这些更改，但如何说服`dpkg`让它相信`/etc`其实在别的地方呢？也不希望恢复系统配置到原始版本，应用更新，然后再恢复：这样只会麻烦不断。所以，想法是正常应用（生成的）补丁，然后*重新应用*补丁（使用`cherry-pick`）到原始分支，并且重写历史记录，使得父指针正确。所有这些都发生在`/etc`之外，因此生产环境的配置文件副本永远不会被触及。

当然，有时`cherry-pick`可能会失败。在这种情况下，你会收到这样的错误信息：

```
Branch pristine set up to track remote branch pristine from origin.
Switched to a new branch 'pristine'
error: could not apply 4fed9ce... committing changes in /etc after apt run
hint: after resolving the conflicts, mark the corrected paths
hint: with 'git add <paths>' or 'git rm <paths>'
hint: and commit the result with 'git commit'
Failed to import changes to pristine
TMPREPO = /tmp/etckeeper-gitrepo.CUCpBEuVXg
TREEID = 8c2fbef8a8f3a4bcc4d66d996c5362c7ba8b17df
PARENTID = 94037457fa47eb130d8adfbb4d67a80232ddd214

```

不要担心：发生的一切只是`pristine`分支不是最新的。你可以通过查看`$TMPREPO/etc`解决此问题，在那里你会看到某种合并冲突。解决冲突并提交。现在你需要手动完成剩余的脚本，可以这样做：

```
git checkout master
git reset --hard HEAD~ # this is the commit we're discarding
git merge -s ours pristine
git push -f origin master
git push origin pristine

```

要确保你做得对，回到`/etc`并运行`git status`：它应报告工作目录干净。否则，可能存在差异，合并可能没有做对。

我已经测试了这个设置一个月了，进展非常顺利（尽管我从未尝试过使用这种设置进行完整的发布升级）。不幸的是，正如我之前所说的，我没有一种方法可以从头构建一个原始分支，如果你想将这个技巧应用到现有系统。当然没有什么能阻止你：你总是可以决定开始，然后你只记录从你开始记录原始的差异。试试看吧！
