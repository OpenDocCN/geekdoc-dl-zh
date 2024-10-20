<!--yml

类别：未分类

日期：2024-07-01 18:17:21

-->

# `MVar` 操作解剖：ezyang 的博客

> 来源：[`blog.ezyang.com/2013/05/anatomy-of-an-mvar-operation/`](http://blog.ezyang.com/2013/05/anatomy-of-an-mvar-operation/)

Adam Belay（[Dune](http://dune.scs.stanford.edu/) 的知名人物）最近在思考为什么 Haskell 的 `MVar` 如此缓慢。“缓慢？”我想，“Haskell 的 `MVar` 不是应该很快吗？” 所以我研究了一下 `MVar` 的工作原理，看看能否解释清楚。

让我们考虑在 [Control.Concurrent.MVar](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Control-Concurrent-MVar.html#v:takeMVar) 中函数 `takeMVar` 的操作。此函数非常简单，它解包 `MVar` 以获取基础的 `MVar#` 原始值，然后调用 primop `takeMVar#`：

```
takeMVar :: MVar a -> IO a
takeMVar (MVar mvar#) = IO $ \ s# -> takeMVar# mvar# s#

```

[Primops](http://hackage.haskell.org/trac/ghc/wiki/Commentary/PrimOps) 导致在 `PrimOps.cmm` 中调用 `stg_takeMVarzh`，这是魔术发生的地方。为简单起见，我们只考虑*多线程*情况。

第一步是**锁定闭包**：

```
("ptr" info) = ccall lockClosure(mvar "ptr");

```

在 GHC 堆上的对象具有信息表头，指示它们是什么类型的对象，通过指向对象的相关信息表来实现。这些表头还用于同步：由于它们是字大小的，因此它们可以原子地与其他值交换。`lockClosure` 实际上是信息表头上的自旋锁：

```
EXTERN_INLINE StgInfoTable *lockClosure(StgClosure *p)
{
    StgWord info;
    do {
        nat i = 0;
        do {
            info = xchg((P_)(void *)&p->header.info, (W_)&stg_WHITEHOLE_info);
            if (info != (W_)&stg_WHITEHOLE_info) return (StgInfoTable *)info;
        } while (++i < SPIN_COUNT);
        yieldThread();
    } while (1);
}

```

`lockClosure` 用于一些其他对象，即线程状态对象（`stg_TSO_info`，通过 `lockTSO`）和线程消息，即异常（`stg_MSG_THROWTO_info`，`stg_MSG_NULL_info`）。

下一步是在 `MVar` 上**应用 GC 写屏障**：

```
if (info == stg_MVAR_CLEAN_info) {
    ccall dirty_MVAR(BaseReg "ptr", mvar "ptr");
}

```

正如我之前[写过的](http://blog.ezyang.com/2013/01/the-ghc-scheduler/)，由于 `MVar` 是可变对象，可以变异以指向第 0 代中的对象；因此，当发生变异时，必须通过可变列表将其添加到根集中。由于每个能力都有一个可变对象，这归结为一堆指针修改，并不需要任何同步。请注意，即使我们最终阻塞在其上，我们也需要将 `MVar` 添加到可变列表中，因为 `MVar` 是阻塞在其上的 *线程*（TSO）的保留者！（然而，我怀疑在某些情况下，我们可以不这样做。）

接下来，我们根据 `MVar` 是否满或空进行分割。如果 `MVar` 为空，我们需要*阻塞线程，直到 `MVar` 为满*：

```
/* If the MVar is empty, put ourselves on its blocking queue,
 * and wait until we're woken up.
 */
if (StgMVar_value(mvar) == stg_END_TSO_QUEUE_closure) {

    // We want to put the heap check down here in the slow path,
    // but be careful to unlock the closure before returning to
    // the RTS if the check fails.
    ALLOC_PRIM_WITH_CUSTOM_FAILURE
        (SIZEOF_StgMVarTSOQueue,
         unlockClosure(mvar, stg_MVAR_DIRTY_info);
         GC_PRIM_P(stg_takeMVarzh, mvar));

    q = Hp - SIZEOF_StgMVarTSOQueue + WDS(1);

    SET_HDR(q, stg_MVAR_TSO_QUEUE_info, CCS_SYSTEM);
    StgMVarTSOQueue_link(q) = END_TSO_QUEUE;
    StgMVarTSOQueue_tso(q)  = CurrentTSO;

    if (StgMVar_head(mvar) == stg_END_TSO_QUEUE_closure) {
        StgMVar_head(mvar) = q;
    } else {
        StgMVarTSOQueue_link(StgMVar_tail(mvar)) = q;
        ccall recordClosureMutated(MyCapability() "ptr",
                                         StgMVar_tail(mvar));
    }
    StgTSO__link(CurrentTSO)       = q;
    StgTSO_block_info(CurrentTSO)  = mvar;
    StgTSO_why_blocked(CurrentTSO) = BlockedOnMVar::I16;
    StgMVar_tail(mvar)             = q;

    jump stg_block_takemvar(mvar);
}

```

解码 C-- primop 代码时的一个有用提示是 `StgTSO_block_info(...)` 及其关联部分是我们如何访问对象字段的。C-- 对 C 结构布局一无所知，因此这些“函数”实际上是由 `utils/deriveConstants` 生成的宏。阻塞线程包括三个步骤：

1.  我们必须将线程添加到附加到 MVar 的阻塞队列中（这就是为什么在 MVar 上阻塞会改变 MVar 的原因！）这包括为链表节点进行堆分配以及变更旧链表尾部。

1.  我们必须将线程标记为阻塞状态（`StgTSO` 的修改）。

1.  我们需要为线程设置一个栈帧，以便线程唤醒时执行正确的操作（即对 `stg_block_takemvar` 的调用）。这个调用还负责解锁闭包。虽然这里的机制非常复杂，但它并不是这篇博文的重点。

如果 MVar 是满的，则可以从 MVar 中**取出值**。

```
/* we got the value... */
val = StgMVar_value(mvar);

```

但这还不是全部。如果有其他阻塞的 `putMVar` 在 MVar 上（记住，当线程尝试放置一个已满的 MVar 时，它会阻塞直到 MVar 清空），那么我们应立即解除其中一个线程的阻塞状态，以便 MVar 始终保持满状态：

```
    q = StgMVar_head(mvar);
loop:
    if (q == stg_END_TSO_QUEUE_closure) {
        /* No further putMVars, MVar is now empty */
        StgMVar_value(mvar) = stg_END_TSO_QUEUE_closure;
        unlockClosure(mvar, stg_MVAR_DIRTY_info);
        return (val);
    }
    if (StgHeader_info(q) == stg_IND_info ||
        StgHeader_info(q) == stg_MSG_NULL_info) {
        q = StgInd_indirectee(q);
        goto loop;
    }

```

有一件有趣的事情与检查阻塞线程的代码有关，那就是对 *indirectees*（`stg_IND_info`）的检查。在什么情况下，队列对象会被间接替换为间接引用呢？事实证明，当我们从链表中 *删除* 一个项时会发生这种情况。这非常好，因为在单链表中，除非我们也有指向前一项的指针，否则我们没有简单的方法来删除项。采用这种方案，我们只需用一个间接引用覆盖当前项，以便在下次垃圾回收时进行清理。（顺便说一句，这就是为什么我们不能仅仅直接链起 TSO，而不需要额外的链表节点。[1]）

当我们找到其他线程时，立即运行它们，这样 MVar 就永远不会变为空：

```
// There are putMVar(s) waiting... wake up the first thread on the queue

tso = StgMVarTSOQueue_tso(q);
StgMVar_head(mvar) = StgMVarTSOQueue_link(q);
if (StgMVar_head(mvar) == stg_END_TSO_QUEUE_closure) {
    StgMVar_tail(mvar) = stg_END_TSO_QUEUE_closure;
}

ASSERT(StgTSO_why_blocked(tso) == BlockedOnMVar::I16); // note: I16 means this is a 16-bit integer
ASSERT(StgTSO_block_info(tso) == mvar);

// actually perform the putMVar for the thread that we just woke up
W_ stack;
stack = StgTSO_stackobj(tso);
PerformPut(stack, StgMVar_value(mvar));

```

这里有一个细节：`PerformPut` 实际上并没有运行线程，它只是查看线程的堆栈以确定它打算 *执行* 什么。一旦 MVar 被放置，我们就唤醒线程，这样它就可以继续它的工作了。

```
// indicate that the MVar operation has now completed.
StgTSO__link(tso) = stg_END_TSO_QUEUE_closure;

// no need to mark the TSO dirty, we have only written END_TSO_QUEUE.

ccall tryWakeupThread(MyCapability() "ptr", tso);

unlockClosure(mvar, stg_MVAR_DIRTY_info);
return (val);

```

总结一下，当你执行 `takeMVar` 时，你需要付出以下成本：

+   一个自旋锁，

+   大约数十个内存操作（写障碍、队列操作），以及

+   当 MVar 为空时，进行（小）堆分配和栈写入。

亚当和我对此有些困惑，然后意识到循环次数之所以如此之多的原因：我们的数字是关于 *往返* 的，即使在如此轻量级的同步（和缺乏系统调用）中，当所有事情都说完时，你仍然需要经过调度器，这会增加循环次数。

* * *

[1] 曾经并非如此，请参见：

```
commit f4692220c7cbdadaa633f50eb2b30b59edb30183
Author: Simon Marlow <marlowsd@gmail.com>
Date:   Thu Apr 1 09:16:05 2010 +0000

    Change the representation of the MVar blocked queue

    The list of threads blocked on an MVar is now represented as a list of
    separately allocated objects rather than being linked through the TSOs
    themselves.  This lets us remove a TSO from the list in O(1) time
    rather than O(n) time, by marking the list object.  Removing this
    linear component fixes some pathalogical performance cases where many
    threads were blocked on an MVar and became unreachable simultaneously
    (nofib/smp/threads007), or when sending an asynchronous exception to a
    TSO in a long list of thread blocked on an MVar.

    MVar performance has actually improved by a few percent as a result of
    this change, slightly to my surprise.

    This is the final cleanup in the sequence, which let me remove the old
    way of waking up threads (unblockOne(), MSG_WAKEUP) in favour of the
    new way (tryWakeupThread and MSG_TRY_WAKEUP, which is idempotent).  It
    is now the case that only the Capability that owns a TSO may modify
    its state (well, almost), and this simplifies various things.  More of
    the RTS is based on message-passing between Capabilities now.

```
