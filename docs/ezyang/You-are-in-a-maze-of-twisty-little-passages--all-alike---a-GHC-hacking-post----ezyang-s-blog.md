<!--yml

category: 未分类

date: 2024-07-01 18:18:03

-->

# 你正处在一个迷宫般的扭曲小通道中……（一个关于 GHC 的黑客帖子）：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/11/another-ghc-hacking-post/`](http://blog.ezyang.com/2010/11/another-ghc-hacking-post/)

大约一个月前，我决定如果我能解决[GHC 的运行时从不终止未使用的工作线程](http://hackage.haskell.org/trac/ghc/ticket/4262)的 bug，那将会很酷。好吧，今天我终于抽出时间看了看它，经过大约一个小时在 GHC RTS 这个弯弯曲曲的迷宫里四处游荡后，我终于看到了一个希望之光，以一种让人心情愉悦的简单补丁的形式。我已经给 Simon Marlow 发了封邮件，确认这光明其实不是一列火车，但我突然意识到，查看我的命令历史并记录我是如何得出在`Capability.c`的第 464 行添加修改是正确的地方的过程，会是件有趣的事情，因为这种心智旅程实际上从未在任何地方以任何形式被记录过。

* * *

*跑迷宫前的热身。* 在像 GHC 这样不稳定的迷宫中，你希望在尝试任何复杂操作之前，确保引导路线（即干净的构建）是正常工作的。我使用[源码树和构建树分离](http://hackage.haskell.org/trac/ghc/wiki/Building/Using#Sourcetreesandbuildtrees)，因此，更新所有内容包括：

```
cd ghc-clean
./darcs-all get
./darcs-all pull -a
cd ../ghc-build
lndir ../ghc-clean
perl boot && ./configure && make
inplace/bin/ghc-stage2 --interactive

```

当这个问题在一个令人满意的方式下解决（对于 Windows 平台来说是一个非平凡的任务）后，代码猎取就可以开始了。

* * *

*准备好你的设备。* 什么？你是说你迷失在这个迷宫中，连知道如何确认你已到达目的地的方法都没有？那可不行... 你需要某种寻找正确方向的工具... 一些能告诉你什么时候找对了的东西。

在这个特定情况下，原始的 bug 报告者已经写了一个小的不完整的测试脚本，所以我做的第一件事就是把它[完善](http://hackage.haskell.org/trac/ghc/ticket/4262#comment:3)成一个不需要人为交互的脚本。新脚本的基准很明确：`/proc/PID/task` 应该报告一个远小于 200 的数字。为了看到当前实现存在问题：

```
ezyang@javelin:~/Dev/ghc-build/testsuite/tests/ghc-regress/ffi/should_run$ ./cleanupThreads
203

```

* * *

*摸清方向。* 好的，我们想要什么？我们希望线程不再闲置而是结束掉。有两种方法可以做到这一点：当线程意识到不需要它时，让它自行了断，或者在必要时由某个管理器杀死线程。后者通常被认为是不好的做法，因为你希望确保线程不在死亡时执行任何可能导致数据损坏的关键任务。因此，自行了断是最好的选择。现在，有两个问题：

1.  线程何时决定进入等待池？这大概是我们希望它终止自身的地方。

1.  线程如何决定它是否应该继续停留或者退出？

* * *

*绘制地图.* GHC 有一个叫做 `-Ds` 的小运行时标志。它非常有用：它会输出关于线程的一大堆调试信息，这正是我们想要查找的。我们的行动计划是查看我们的测试脚本中线程活动的情况，并确定线程应该死亡而不是徘徊的点。

日志的开头看起来像这样：

```
b75006d0: allocated 1 capabilities
b75006d0: new task (taskCount: 1)
b75006d0: returning; I want capability 0
b75006d0: resuming capability 0
b75006d0: starting new worker on capability 0
b75006d0: new worker task (taskCount: 2)
b75006d0: task exiting
b75006d0: new task (taskCount: 2)
b75006d0: returning; I want capability 0
b71ffb70: cap 0: schedule()
b71ffb70: giving up capability 0

```

注意数字 `b75006d0`；那是我们的主线程，它将非常忙碌。这是我们首次启动的线程，用来进行一个外部调用，但完成得相当快，并不是我们正在寻找的外部调用：

```
b75006d0: cap 0: created thread 1
b75006d0: cap 0: thread 1 appended to run queue
b75006d0: new bound thread (1)
b75006d0: cap 0: schedule()
b75006d0: cap 0: running thread 1 (ThreadRunGHC)
b75006d0: cap 0: thread 1 stopped (suspended while making a foreign call)
b75006d0: freeing capability 0
b75006d0: returning; I want capability 0
b75006d0: resuming capability 0
b75006d0: cap 0: running thread 1 (ThreadRunGHC)
b75006d0: cap 0: thread 1 stopped (suspended while making a foreign call)
b75006d0: freeing capability 0
b75006d0: returning; I want capability 0
b75006d0: resuming capability 0
b75006d0: cap 0: running thread 1 (ThreadRunGHC)
b75006d0: cap 0: created thread 2
b75006d0: cap 0: thread 2 appended to run queue
b75006d0: cap 0: thread 1 stopped (finished)

```

不久之后，我们看到一大堆新线程被创建并添加到运行队列中——这些就是我们的线程：

```
b75006d0: woken up on capability 0
b75006d0: resuming capability 0
b75006d0: cap 0: running thread 3 (ThreadRunGHC)
b75006d0: cap 0: created thread 4
b75006d0: cap 0: thread 4 appended to run queue
b75006d0: cap 0: created thread 5
b75006d0: cap 0: thread 5 appended to run queue
b75006d0: cap 0: created thread 6
b75006d0: cap 0: thread 6 appended to run queue
b75006d0: cap 0: created thread 7
b75006d0: cap 0: thread 7 appended to run queue
b75006d0: cap 0: created thread 8
b75006d0: cap 0: thread 8 appended to run queue
b75006d0: cap 0: created thread 9
b75006d0: cap 0: thread 9 appended to run queue
b75006d0: cap 0: created thread 10
b75006d0: cap 0: thread 10 appended to run queue
b75006d0: cap 0: created thread 11
b75006d0: cap 0: thread 11 appended to run queue
b75006d0: cap 0: created thread 12
b75006d0: cap 0: thread 12 appended to run queue
b75006d0: cap 0: created thread 13

```

这个过程一直持续，直到我们把它们都生成出来：

```
54139b70: starting new worker on capability 0
54139b70: new worker task (taskCount: 201)
53938b70: cap 0: schedule()
53938b70: cap 0: running thread 202 (ThreadRunGHC)
53938b70: cap 0: thread 202 stopped (suspended while making a foreign call)
53938b70: starting new worker on capability 0
53938b70: new worker task (taskCount: 202)
53137b70: cap 0: schedule()
53137b70: cap 0: running thread 203 (ThreadRunGHC)
53137b70: cap 0: thread 203 stopped (suspended while making a foreign call)
53137b70: starting new worker on capability 0
53137b70: new worker task (taskCount: 203)
52936b70: cap 0: schedule()

```

然后，因为没有什么可做的（我们所有的线程都在 FFI land 中），我们进行了一次大的 GC：

```
   52936b70: woken up on capability 0
   52936b70: resuming capability 0
   52936b70: deadlocked, forcing major GC...
   52936b70: cap 0: requesting parallel GC
   52936b70: ready_to_gc, grabbing GC threads
all threads:
threads on capability 0:
other threads:
       thread  203 @ 0xb72b5c00 is blocked on an external call (TSO_DIRTY)
       thread  202 @ 0xb72b5800 is blocked on an external call (TSO_DIRTY)
       thread  201 @ 0xb72b5400 is blocked on an external call (TSO_DIRTY)
       thread  200 @ 0xb72b5000 is blocked on an external call (TSO_DIRTY)
       thread  199 @ 0xb72b4c00 is blocked on an external call (TSO_DIRTY)
       thread  198 @ 0xb72b4800 is blocked on an external call (TSO_DIRTY)
       thread  197 @ 0xb72b4400 is blocked on an external call (TSO_DIRTY)
       thread  196 @ 0xb72b4000 is blocked on an external call (TSO_DIRTY)
       thread  195 @ 0xb72b3c00 is blocked on an external call (TSO_DIRTY)
       thread  194 @ 0xb72b3800 is blocked on an external call (TSO_DIRTY)
       thread  193 @ 0xb72b3400 is blocked on an external call (TSO_DIRTY)
       [snip (you get the idea)]

```

（我一直在想 FFI 调用是否应该被视为死锁。）

现在线程开始从 FFI-land 回来并处于空闲状态：

```
b69feb70: cap 0: running thread 4 (ThreadRunGHC)
b69feb70: cap 0: waking up thread 3 on cap 0
b69feb70: cap 0: thread 3 appended to run queue
b69feb70: cap 0: thread 4 stopped (finished)
b69feb70: giving up capability 0
b69feb70: there are 2 spare workers
b69feb70: passing capability 0 to bound task 0xb75006d0
b61fdb70: returning; I want capability 0
b61fdb70: resuming capability 0
b61fdb70: cap 0: running thread 5 (ThreadRunGHC)
b59fcb70: returning; I want capability 0
b61fdb70: cap 0: thread 5 stopped (finished)
b61fdb70: giving up capability 0
b61fdb70: there are 3 spare workers
b61fdb70: passing capability 0 to worker 0xb59fcb70
b75006d0: woken up on capability 0
b75006d0: capability 0 is owned by another task
b51fbb70: returning; I want capability 0
b59fcb70: resuming capability 0
b59fcb70: cap 0: running thread 6 (ThreadRunGHC)
b59fcb70: cap 0: thread 6 stopped (finished)
b59fcb70: giving up capability 0
b49fab70: returning; I want capability 0
b59fcb70: there are 4 spare workers
b59fcb70: passing capability 0 to worker 0xb51fbb70
b51fbb70: resuming capability 0
b51fbb70: cap 0: running thread 7 (ThreadRunGHC)
b51fbb70: cap 0: thread 7 stopped (finished)
b51fbb70: giving up capability 0
b41f9b70: returning; I want capability 0
b51fbb70: there are 5 spare workers

```

我其实有点作弊：`there are X spare workers` 的调试语句是我自己添加的。但这一部分很重要；我们特别感兴趣的是这些行：

```
b61fdb70: cap 0: thread 5 stopped (finished)
b61fdb70: giving up capability 0

```

线程停止了，但它并没有死，它只是放弃了能力。这两个地方非常适合线程可能会选择杀死自己的地方。

* * *

*地标.* 是时候用信赖的 `grep` 搞清楚这些调试消息是从哪里发出的了。不幸的是，`5` 和 `finished` 可能是动态生成的消息，所以 `stopped` 是我们能找到的唯一真正的标识符。幸运的是，这足够具体，让我能找到 RTS 中正确的行：

```
ezyang@javelin:~/Dev/ghc-clean/rts$ grep -R stopped .
./Capability.c:    // list of this Capability.  A worker can mark itself as stopped,
./Capability.c:    if (!isBoundTask(task) && !task->stopped) {
./RaiseAsync.c:     - all the other threads in the system are stopped (eg. during GC).
./RaiseAsync.c:    // if we got here, then we stopped at stop_here
./Task.c:        if (task->stopped) {
./Task.c:    task->stopped       = rtsFalse;
./Task.c:    task->stopped = rtsFalse;
./Task.c:    task->stopped = rtsTrue;
./Task.c:    task->stopped = rtsTrue;
./Task.c:   debugBelch("task %p is %s, ", taskId(task), task->stopped ? "stopped" : "alive");
./Task.c:   if (!task->stopped) {
./sm/GC.c:      // The other threads are now stopped.  We might recurse back to
./Schedule.c:                  "--<< thread %ld (%s) stopped: requesting a large block (size %ld)\n",
./Schedule.c:                  "--<< thread %ld (%s) stopped to switch evaluators",
./Schedule.c:       // stopped.  We need to stop all Haskell threads, including
./Trace.c:        debugBelch("cap %d: thread %lu stopped (%s)\n",   ### THIS IS THE ONE
./Task.h:    rtsBool    stopped;         // this task has stopped or exited Haskell
./Task.h:// Notify the task manager that a task has stopped.  This is used
./Task.h:// Put the task back on the free list, mark it stopped.  Used by
./Interpreter.c:                  // already stopped at just now
./Interpreter.c:            // record that this thread is not stopped at a breakpoint anymore
./win32/Ticker.c:            // it still hasn't stopped.

```

`Trace.c` 中的那一行实际上是在一个通用的调试函数 `traceSchedEvent_stderr` 中，但幸运的是，其中有一个关于其参数 `tag` 的大 `case` 语句：

```
case EVENT_STOP_THREAD:     // (cap, thread, status)
    debugBelch("cap %d: thread %lu stopped (%s)\n",·
               cap->no, (lnat)tso->id, thread_stop_reasons[other]);
    break;

```

所以 `EVENT_STOP_THREAD` 是一个很好的下一个 `grep`。确实如此：

```
ezyang@javelin:~/Dev/ghc-clean/rts$ grep -R EVENT_STOP_THREAD .
./Trace.c:    case EVENT_STOP_THREAD:     // (cap, thread, status)
./eventlog/EventLog.c:  [EVENT_STOP_THREAD]         = "Stop thread",
./eventlog/EventLog.c:        case EVENT_STOP_THREAD:     // (cap, thread, status)
./eventlog/EventLog.c:    case EVENT_STOP_THREAD:     // (cap, thread, status)
./Trace.h:    HASKELLEVENT_STOP_THREAD(cap, tid, status)
./Trace.h:    traceSchedEvent(cap, EVENT_STOP_THREAD, tso, status);

```

它看起来是 `Trace.h` 中的一个内联函数：

```
INLINE_HEADER void traceEventStopThread(Capability          *cap    STG_UNUSED,
                                        StgTSO              *tso    STG_UNUSED,
                                        StgThreadReturnCode  status STG_UNUSED)
{
    traceSchedEvent(cap, EVENT_STOP_THREAD, tso, status);
    dtraceStopThread((EventCapNo)cap->no, (EventThreadID)tso->id,
                     (EventThreadStatus)status);
}

```

Classy. 所以 `traceEventStopThread` 就是那个魔法词，确实：

```
ezyang@javelin:~/Dev/ghc-clean/rts$ grep -R traceEventStopThread .
./Schedule.c:    traceEventStopThread(cap, t, ret);
./Schedule.c:  traceEventStopThread(cap, tso, THREAD_SUSPENDED_FOREIGN_CALL);
./Trace.h:INLINE_HEADER void traceEventStopThread(Capability          *cap    STG_UNUSED,

```

在 `Schedule.c` 中有两个可能的位置。

* * *

*开始挖掘.* 首先我们得选一个站点仔细检查。幸运的是，我们注意到第二个跟踪事件对应于在进行安全的 FFI 调用之前暂停线程；那肯定不是我们这里要找的。此外，第一个在调度器中，这很有道理。但在这附近并没有明显的东西，你可能会把它与由于工作不足而保存工作任务关联起来。

那个 `giving up` 能力消息呢？再多找一下发现它在 `yieldCapability` 函数里（正如我们所料）。如果我们追踪 `yieldCapability` 的调用，我们看到它是由 `scheduleYield` 调用的，而 `scheduleYield` 又被调度循环调用：

```
scheduleYield(&cap,task);

if (emptyRunQueue(cap)) continue; // look for work again

// Get a thread to run
t = popRunQueue(cap);

```

这非常非常有趣。它表明能力本身将告诉我们是否要进行工作，而`yieldCapability`是一个有希望进一步探索的函数：

```
debugTrace(DEBUG_sched, "giving up capability %d", cap->no);

// We must now release the capability and wait to be woken up
// again.
task->wakeup = rtsFalse;
releaseCapabilityAndQueueWorker(cap);

```

最后那个调用看起来很有趣：

```
static void
releaseCapabilityAndQueueWorker (Capability* cap USED_IF_THREADS)
{
    Task *task;

    ACQUIRE_LOCK(&cap->lock);

    task = cap->running_task;

    // If the current task is a worker, save it on the spare_workers
    // list of this Capability.  A worker can mark itself as stopped,
    // in which case it is not replaced on the spare_worker queue.
    // This happens when the system is shutting down (see
    // Schedule.c:workerStart()).
    if (!isBoundTask(task) && !task->stopped) {
       task->next = cap->spare_workers;
       cap->spare_workers = task;
    }
    // Bound tasks just float around attached to their TSOs.

    releaseCapability_(cap,rtsFalse);

    RELEASE_LOCK(&cap->lock);
}

```

我们找到了！

* * *

*检查区域。* `spare_workers` 队列看起来像是那些没有工作可做的工作线程去放松的队列。我们应该验证这是否属实：

```
int i;
Task *t;
for (i = 0, t = cap->spare_workers; t != NULL; t = t->next, i++) {}
debugTrace(DEUBG_sched, "there are %d spare workers", i);

```

确实，正如我们在上面的调试语句中看到的那样，情况确实如此：空闲工作者的数量不断增加：

```
54139b70: there are 199 spare workers
54139b70: passing capability 0 to worker 0x53938b70
53938b70: resuming capability 0
53938b70: cap 0: running thread 202 (ThreadRunGHC)
53938b70: cap 0: thread 202 stopped (blocked)
    thread  202 @ 0xb727a400 is blocked on an MVar @ 0xb72388a8 (TSO_DIRTY)
53938b70: giving up capability 0
53938b70: there are 200 spare workers
53938b70: passing capability 0 to worker 0x53137b70
53137b70: resuming capability 0
53137b70: cap 0: running thread 203 (ThreadRunGHC)
53137b70: cap 0: thread 203 stopped (blocked)
    thread  203 @ 0xb727a000 is blocked on an MVar @ 0xb72388a8 (TSO_DIRTY)
53137b70: giving up capability 0
53137b70: there are 201 spare workers

```

* * *

*撰写解决方案。* 因此，从这里的补丁很简单，因为我们已经找到了正确的位置。我们检查一下空闲工作者的队列是否在某个数量，并且如果是的话，我们不再保存自己到队列中，而是清理然后自杀：

```
for (i = 1; t != NULL && i < 6; t = t->next, i++) {}
if (i >= 6) {
        debugTrace(DEBUG_sched, "Lots of spare workers hanging around, terminating this thread");
        releaseCapability_(cap,rtsFalse);
        RELEASE_LOCK(&cap->lock);
        pthread_exit(NULL);
}

```

然后我们测试看到这确实起作用了：

```
ezyang@javelin:~/Dev/ghc-build/testsuite/tests/ghc-regress/ffi/should_run$ ./cleanupThreads
7

```

* * *

*附言*。这个概念验证中存在一些明显的不足。它不具备可移植性。我们需要确信这确实完成了 RTS 期望工作者执行的所有清理工作。也许我们的数据表示可以更有效（如果我们存储的值数量是固定的，我们显然不需要链表）。但这些问题最好由更了解 RTS 的人来回答，因此我目前已经[提交了这个概念验证](http://www.haskell.org/pipermail/glasgow-haskell-users/2010-November/019503.html)以供进一步审查。祈祷顺利！
