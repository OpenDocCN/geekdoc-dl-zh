<!--yml

类别：未分类

日期：2024-07-01 18:17:42

-->

# 合成 Git 合并：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/07/synthetic-git-merges/`](http://blog.ezyang.com/2011/07/synthetic-git-merges/)

理论上，Git 支持使用`merge`配置属性自定义低级别合并驱动程序。实际上，没有人真的想从头开始编写自己的合并驱动程序。对于许多情况下需要自定义合并驱动程序的案例，你不必从头开始编写自己的合并驱动程序！考虑这些情况：

+   你想要合并具有不同换行符样式的文件，

+   你想要合并一个删除了大量尾随空白的文件，

+   当一个分支替换了某些字符串为自定义字符串时，你想要合并文件（例如，一个配置文件实例化了`PASSWORD`，或者需要在合并冲突时匿名化文件），

+   你想要合并一个具有稳定文本格式的二进制文件，或

+   你想要掌握关于特定类型冲突的知识以及如何解决它们（一个超智能的`rerere`）。

对于所有这些情况，你可以通过修改输入文件（构建合成合并输入），调用 Git 的`git merge-file`来执行实际合并，然后可能编辑结果，再将其交还给你的合并驱动程序的原始调用者。这真的很简单。这里有一个处理具有不同换行符样式文件的示例驱动程序，将它们规范化为 UNIX 的方式：

```
#!/bin/sh
CURRENT="$1"
ANCESTOR="$2"
OTHER="$3"
dos2unix "$CURRENT"
dos2unix "$ANCESTOR"
dos2unix "$OTHER"
exec git merge-file "$CURRENT" "$ANCESTOR" "$OTHER"

```

你可以通过调整你的`.git/config`来设置它：

```
[merge.nl]
        name = Newline driver
        driver = /home/ezyang/merge-fixnewline.sh %A %O %B

```

以及你的`.git/info/attributes`：

```
*.txt merge=nl

```

在[Wizard](http://scripts.mit.edu/wizard/)中，我们实现了更聪明的换行符规范化、配置值去替换（这减少了上游和下游之间的差异，减少了由于接近性而导致的冲突量），以及自定义的`rerere`行为。我也看到我的一位同事在处理包含尾随空白字符的合并冲突时手动使用了这种技术（在 Mercurial 中，更不用说了！）

实际上，我们进一步发展了这个概念：不仅仅创建合成文件，我们创建了完全合成的树，然后适当地调用`git merge`。这有几个好处：

+   现在我们可以选择一个任意的祖先提交来执行合并（令人惊讶的是，这对我们的用例非常有用），

+   Git 更容易检测到文件移动和更改换行符样式等，

+   它使用起来更容易一些，因为你只需调用一个自定义命令，而不必记住如何正确设置你的 Git 配置和属性（并保持它们的最新状态！）

合并只是元数据——多个父提交。Git 不在乎你如何获取合并提交的内容。祝合并愉快！