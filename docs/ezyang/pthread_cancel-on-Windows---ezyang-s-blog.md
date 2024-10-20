<!--yml

category: 未分类

date: 2024-07-01 18:18:09

-->

# pthread_cancel on Windows : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/09/pthread-cancel-on-window/`](http://blog.ezyang.com/2010/09/pthread-cancel-on-window/)
> 
> Edward，很抱歉，我有些坏消息。你的 [可中断 GHC 补丁](http://blog.ezyang.com/2010/08/interrupting-ghc/)；在移植到 Windows 途中遇到了一场可怕的事故。希望你理解：我们正在尽力修复它，但出现了一些复杂情况...

小测验！这段 pthreads 代码做什么？

```
#include <pthread.h>
#include <stdio.h>

void *thread1(void *arg) { sleep(10000); }
void *thread2(void *arg) { while (1) {} }

void *psycho_killer(void *arg) {
    pthread_t *id = (pthread_t*)arg;
    pthread_cancel(*id);
    printf("[%p] Psycho killer...\n", id);
    pthread_join(*id, NULL);
    printf("[%p] ...qu'est-ce que c'est.\n", id);
}

int main(char* argv, int argc) {
    pthread_t t1, t2, k1, k2;
    pthread_create(&t1, NULL, thread1, NULL);
    printf("[%p] I can't sleep 'cause my bed's on fire\n", &t1);
    pthread_create(&t2, NULL, thread2, NULL);
    printf("[%p] Don't touch me I'm a real live wire\n", &t2);
    pthread_create(&k1, NULL, psycho_killer, &t1);
    pthread_create(&k2, NULL, psycho_killer, &t2);
    pthread_join(k1, NULL);
    pthread_join(k2, NULL);
    printf("Run run run away!\n");
    return 0;
}

```

它从未成功终止第二个线程...

```
ezyang@javelin:~/Desktop$ ./test
[0xbf900b4c] I can't sleep 'cause my bed's on fire
[0xbf900b48] Don't touch me I'm a real live wire
[0xbf900b4c] Psycho killer...
[0xbf900b4c] ...qu'est-ce que c'est.
[0xbf900b48] Psycho killer...
^C

```

如果你只有 `pthread_cancel` 和 `pthread_setcancelstate` 的手册页，这可能有点神秘。但 `pthreads` 页面很清楚：`sleep` 是一百零二个“可取消”函数之一，如果线程的取消状态为 `PTHREAD_CANCEL_DEFERRED`，则必须在其中显式允许延迟取消，使用 `pthread_testcancel`。早期的 POSIX 规范版本在系统调用入口或系统调用运行期间是否应进行取消有些不清楚，但 [2008 规范](http://www.opengroup.org/onlinepubs/9699919799/functions/V2_chap02.html#tag_15_09_05_02) 比较明确：

> 当线程执行以下函数时应发生取消点...

* * *

百万美元问题是：“我们能在 Windows 上实现相同的语义吗？”实际上，因为看起来很多人都希望在 Windows 上拥有 pthreads 的功能，你会认为这已经由 [pthreads-win32](http://sourceware.org/pthreads-win32/) 实现了。我们去看看源代码！

```
if (tp->cancelType == PTHREAD_CANCEL_ASYNCHRONOUS
    && tp->cancelState == PTHREAD_CANCEL_ENABLE
    && tp->state < PThreadStateCanceling)
  {
    /* snip  */
  }
else
  {
    /*
     * Set for deferred cancellation.
     */
    if (tp->state < PThreadStateCancelPending)
      {
        tp->state = PThreadStateCancelPending;
        if (!SetEvent (tp->cancelEvent))
          {
            result = ESRCH;
          }
      }
    else if (tp->state >= PThreadStateCanceling)
      {
        result = ESRCH;
      }

    (void) pthread_mutex_unlock (&tp->cancelLock);
  }

```

有趣的是，pthreads-win32 似乎并没有做任何特殊处理：当我们将我们的测试程序翻译并在 pthreads-win32 上运行时，它也在 `Sleep` 调用上卡住了：

```
C:\Users\ezyang\pthreads-win32\Pre-built.2\lib>test.exe
[0022FF40] I can't sleep 'cause my bed's on fire
[0022FF38] Don't touch me I'm a real live wire
[0022FF40] Psycho killer...
[0022FF38] Psycho killer...
^C

```

* * *

此时，值得稍作停顿，问一问：“我们到底想做什么？”如果你问如何在 Stack Overflow 上终止线程，你会得到一大堆回复告诉你：“停止那样做，用正确的方式来做”；也就是说，通过另一种消息传递机制在线程本身上显式处理线程终止。

因此，中断调用有许多不同的需求：

1.  GHC 希望能够将阻塞的 IO 调用放在工作线程上，但稍后取消它们；目前它可以在 Linux 上做到这一点，但在 Windows 上不行，

1.  用户希望编写友好的中断 C 库，并让它们与 Haskell 的异常机制无缝集成，

1.  我们希望拥有 IO 世界的黄金触摸，即将阻塞 IO 代码即时转换为良好行为的非阻塞代码。

下次我将讨论针对每个目标可能需要的不同方法。
