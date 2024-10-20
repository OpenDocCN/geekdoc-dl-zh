<!--yml

category: 未分类

date: 2024-07-01 18:18:13

-->

# [System.Posix.Redirect](http://blog.ezyang.com/2010/07/system-posix-redirect/)：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/07/system-posix-redirect/`](http://blog.ezyang.com/2010/07/system-posix-redirect/)

[System.Posix.Redirect](http://hackage.haskell.org/package/system-posix-redirect) 是一个[众所周知的、巧妙而有效的 POSIX 黑客](http://homepage.ntlworld.com/jonathan.deboynepollard/FGA/redirecting-standard-io.html) 的 Haskell 实现。它也完全不符合软件工程的标准。大约一周前，我从我的工作代码中去除了这个失败的实验，并将其上传到 Hackage 以供严格学术目的使用。

*它是用来做什么的？* 当你在 shell 脚本中运行一个命令时，你可以选择将其输出*重定向*到另一个文件或程序：

```
$ echo "foo\n" > foo-file
$ cat foo-file
foo
$ cat foo-file | grep oo
foo

```

许多用于[创建新进程](http://www.haskell.org/ghc/docs/6.12.2/html/libraries/process-1.0.1.2/System-Process.html)的 API 允许自定义 stdin/stdout/stderr 句柄；System.Posix.Redirect 允许您在不创建新进程的情况下重定向 stdout/stderr：

```
redirectStdout $ print "foo"

```

*它是如何做到的？* 在 POSIX 系统上，事实证明，几乎与创建子进程时发生的事情完全相同。我们可以通过跟踪使用不同句柄创建子进程的过程来得到一些线索。考虑这个简单的 Haskell 程序：

```
import System.Process
import System.IO

main = do
    -- redirect stdout to stderr
    h <- runProcess "/bin/echo" ["foobar"] Nothing Nothing Nothing (Just stderr) Nothing
    waitForProcess h

```

当我们运行`strace -f`（使用`-f`标志启用对子进程跟踪）时，我们看到：

```
vfork(Process 26861 attached
)                                 = 26861
[pid 26860] rt_sigprocmask(SIG_SETMASK, [], NULL, 8) = 0
[pid 26860] ioctl(0, SNDCTL_TMR_TIMEBASE or TCGETS, {B38400 opost isig icanon echo ...}) = 0
[pid 26860] ioctl(1, SNDCTL_TMR_TIMEBASE or TCGETS, {B38400 opost isig icanon echo ...}) = 0
[pid 26860] waitpid(26861, Process 26860 suspended
 <unfinished ...>
[pid 26861] rt_sigprocmask(SIG_SETMASK, [], NULL, 8) = 0
[pid 26861] dup2(0, 0)                  = 0
[pid 26861] dup2(2, 1)                  = 1
[pid 26861] dup2(2, 2)                  = 2
[pid 26861] execve("/bin/echo", ["/bin/echo", "foobar"], [/* 53 vars */]) = 0

```

`dup2`调用是关键，因为在向`vfork`或`execve`传递特殊参数时没有必要（“53 vars”是继承的环境变量）去捣鼓子进程的标准句柄，我们需要自己修复它们。`dup2`将文件描述符`2`（保证是 stderr，`0`是 stdin，`1`是 stdout）复制到`stdout`，这正是我们在原始代码中请求的内容。文件描述符表是进程全局的，因此当我们改变文件描述符`1`时，所有人都会注意到。

当我们不打算在`dup2`调用后使用`execve`时存在一个复杂情况：你的标准库可能会缓冲输出，这种情况下可能仍然存在一些未写入文件描述符的程序中的数据。如果在普通的 POSIX C 应用程序中尝试这种技巧，你只需要刷新`stdio.h`中的`FILE`句柄。如果你在 Haskell 应用程序中，还需要刷新 Haskell 的缓冲区。（请注意，如果你`execve`，则不需要这样做，因为此系统调用会清除新程序的内存空间。）

*我为什么写这个模块？* 我在编写这个模块时有一个非常具体的用例：我有一个用 C 编写的外部库，它将错误条件写入标准输出。想象一下，如果 `hPutStr` 在无法写入字符串时打印错误消息，而不是引发 `IOException`，那对于想要捕获和处理错误条件的客户端代码将是一件非常糟糕的事情。在调用这些函数之前临时重定向标准输出意味着我可以将这些错误条件传递给 Haskell，同时避免修改外部库或将其降级为子进程（这将导致更慢的进程间通信）。

*为什么不能在生产环境中使用它？* “它在 Windows 上不起作用！” 这并非完全正确：在某些情况下，你可能可以让它的变体工作。

在 Windows 平台上，主要问题是大量选择的运行时和标准库。幸运的是，Unix 上的大多数应用程序都使用一个标准库：libc，因此可以合理地确定你和你的同事都在使用同一个`FILE`抽象。由于文件描述符位于内核层面，无论你使用哪个库，它们都能保证正常工作。在 Windows 上则没有这样的奢侈：你链接的 DLL 可能是由其他编译器工具链编译的，具有自己的运行时。特别是 GHC 在 Windows 上使用 MingW 工具链进行链接，而本地代码更有可能是使用 Microsoft 工具（MSVC++）。如果库能使用 MingW 重新编译，也许就能解决问题，但我决定更简单地修改库，以另一种方式返回错误代码。于是，这个模块从代码库中被完全移除了。
