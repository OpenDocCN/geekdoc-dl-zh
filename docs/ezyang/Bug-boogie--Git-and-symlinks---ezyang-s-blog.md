<!--yml

category: 未分类

date: 2024-07-01 18:18:17

-->

# Bug boogie: Git 和符号链接 : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/06/bug-boogie-git-and-symlinks/`](http://blog.ezyang.com/2010/06/bug-boogie-git-and-symlinks/)

Git 对你的文件非常小心：除非你明确告诉它要进行破坏性操作，否则它将拒绝覆盖它不认识的文件，并显示如下错误：

> 未跟踪的工作树文件'foobar'将被合并覆盖。

在我的工作中，经常需要对[Wizard](http://scripts.mit.edu/wizard)上那些“维护不良”的工作副本执行合并操作，例如，在旧目录上解压了新版本的源代码树，却忘记添加新加入的文件。当 Wizard 尝试以正确方式自动升级它们到新版本时，这将导致各种未跟踪的工作树文件投诉，然后我们必须手动检查这些未跟踪的文件，并在它们正常后移除它们。

对此有一个简单的解决方法：虽然我们不想将所有未跟踪的文件添加到 Git 仓库中，但我们可以只添加那些可能会被覆盖的文件。Git 将停止对这些文件的投诉，并且我们仍然可以在历史记录中找到它们的记录：

```
def get_file_set(rev):
    return set(shell.eval("git", "ls-tree", "-r", "--name-only", rev).split("\n"))
old_files = get_file_set("HEAD")
new_files = get_file_set(self.version)
added_files = new_files - old_files
for f in added_files:
    if os.path.lexists(f): # *
        shell.call("git", "add", f)

```

以前，代码中的星号行为 `if os.path.exists(f)`。你能猜到这有什么错误吗？回想一下 `exists` 和 `lexists` 之间的区别；如果涉及的文件是符号链接，`exists` 会跟随它，而 `lexists` 则不会。因此，如果将要被覆盖的文件是一个损坏的符号链接，旧版本的代码将不会将其删除。在许多情况下，你无法区分这些情况：如果文件符号链接指向的父目录存在，我可以通过符号链接创建一个文件，以及其他正常的“文件操作”。

然而，Git 非常清楚符号链接和普通文件之间的区别，并且如果它将会覆盖一个符号链接，它会相应地投诉。保留了这些好的老信息！

*附言。* 昨天是我在 Galois 工作的第一天！如此令人兴奋，以至于我没能整理思绪写一篇博客。敬请期待更多。
