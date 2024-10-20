<!--yml

category: 未分类

date: 2024-07-01 18:18:03

-->

# It’s just a longjmp to the left : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/11/its-just-a-longjmp-to-the-left/`](http://blog.ezyang.com/2010/11/its-just-a-longjmp-to-the-left/)

*然后向右边发出信号。*

一个与 readline 相关的明显缺陷是，如果在提示期间按下`^C`，什么也不会发生，如果再次按下`^C`（并且没有在信号处理程序中搞砸），整个程序将不情不愿地终止。那不太好！幸运的是，`readline`似乎是少数几个确实投入了一些工作来确保您可以从信号处理程序中 longjmp 出来而不完全破坏库内部状态的 C 库之一（他们通过自由掩码和解除掩码，以及他们自己的信号处理程序来做到这一点，清理然后重新抛出信号）。

所以我决定看看能否修补 readline，使其能从信号处理程序（由[yours truly](http://blog.ezyang.com/2010/09/towards-platform-agnostic-interruptibility/)提供的信号）中的 longjmp 中恢复控制权，并将控制权交还给 Haskell。这个怪物就这样诞生了。

```
static jmp_buf hs_readline_jmp_buf;
static struct sigaction old_sigpipe_action;

void hs_readline_sigaction (int signo, siginfo_t *info, void *data)
{
    sigaction(SIGALRM, &old_sigpipe_action, NULL);
    siglongjmp(hs_readline_jmp_buf, signo);
}

char *hs_readline (const char *s)
{
    struct sigaction action;
    int signo;
    sigset_t mask;
    memset(&action, 0, sizeof(struct sigaction));
    sigemptyset(&mask);
    sigaddset(&mask, SIGALRM);
    action.sa_sigaction = hs_readline_sigaction;
    action.sa_mask = mask;
    action.sa_flags = SA_RESETHAND;
    sigaction(SIGALRM, &action, &old_sigpipe_action);
    if (signo = sigsetjmp(hs_readline_jmp_buf, 1)) {
        return NULL;
    }
    char *r = readline(s);
    sigaction(SIGALRM, &old_sigpipe_action, NULL);
    return r;
}

```

它实际上运行得非常好，尽管信号的路径有些迂回：SIGINT 首先由*readline*安装的信号处理程序处理，它会清理终端的更改，然后重新抛出到 GHC 的信号处理程序。 GHC 将告诉 IO 管理器发生了信号，然后返回到 readline 的内部（它重新安装了终端的所有更改）。然后，IO 管理器读取信号，并发送一个`ThreadKilled`异常，这将导致 RTS 尝试中断外部调用。`SIGALRM`（实际上，这是一个谎言，GHC 中的代码发送了一个`SIGPIPE`，但 readline 认为`SIGPIPE`不是应该清理的信号，所以我改变了它——欢迎更好的建议）再次命中 readline 的信号处理程序，我们清理终端，然后命中我们的信号处理程序，它 longjmp 到一个`return NULL`，这将把我们带回 Haskell。然后捕获信号，大家都很高兴。

不幸的是，几乎所有的代码都是样板，我不能把它放进一个漂亮的 Haskell 组合器中，因为当 Haskell 在执行时，几乎没有堆栈可言，我敢打赌一个`setjmp`的 FFI 调用会让 RTS 非常困惑。它也不是可重入的，尽管我怀疑`readline`也不是可重入的。当然，从信号处理程序进行非本地控制转移是你妈妈告诉你不要做的事情。所以这种方法可能不通用。但它相当有趣。
