<!--yml

category: 未分类

date: 2024-07-01 18:17:44

-->

# IVars 调度：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/07/scheduling-ivar/`](http://blog.ezyang.com/2011/07/scheduling-ivar/)

## IVars 调度

我在先前的[IVar monad post](http://blog.ezyang.com/2011/06/the-iva-monad/)中提到的愚蠢调度程序的一个缺点是，由于它将所有待处理操作存储在堆栈上，因此很容易堆栈溢出。我们可以通过实现执行计划来明确地将所有这些待处理的回调移动到堆上。这涉及在我们的单子中添加`Schedule`状态（我已经用`IORef Schedule`这样做了）。这里是一个稍微聪明一些的调度程序（我还简化了一些代码片段，并添加了一个新的`addCallback`函数）：

```
import Data.IORef

data IVarContents a =
    Blocking [a -> IO ()]
  | Full a

type Schedule = [IO ()]
type IVar a = IORef (IVarContents a)

newtype T a = T { runT :: IORef Schedule -> IO (IVar a) }

instance Monad T where
  return x = T (\_ -> newIORef (Full x))
  m >>= f  = T $ \sched ->
        do xref <- runT m sched
           mx <- readIORef xref
           case mx of
             Full x      -> runT (f x) sched
             Blocking cs -> do
                    r <- newIORef (Blocking [])
                    let callback x = do
                        y <- runT (f x) sched
                        addCallback y (fillIVar sched r)
                    addCallback xref callback
                    return r

addCallback :: IVar a -> (a -> IO ()) -> IO ()
addCallback r c = do
    rc <- readIORef r
    case rc of
        Full x -> c x
        Blocking cs -> writeIORef r (Blocking (c:cs))

fillIVar :: IORef Schedule -> IVar a -> a -> IO ()
fillIVar sched ref x = do
  r <- readIORef ref
  writeIORef ref (Full x)
  case r of
    Blocking cs -> schedule sched (map ($x) cs)
    Full _ -> error "fillIVar: Cannot write twice"

-- FIFO scheduler
schedule :: IORef Schedule -> [IO ()] -> IO ()
schedule sched to_sched = do
    cur <- readIORef sched
    writeIORef sched (cur ++ to_sched)

run :: T () -> IO ()
run initial_job = do
    sched <- newIORef []
    writeIORef sched [runT initial_job sched >> return ()]
    let go = do
        jobs <- readIORef sched
        case jobs of
            [] -> return ()
            (job:rest) -> writeIORef sched rest >> job >> go
    go

```

这里是演示基本思想的一些示例代码：

```
-- Does more work than return (), but semantically the same
tick :: T ()
tick = T $ \sched ->
        do r <- newIORef (Blocking [])
           schedule sched [fillIVar sched r ()]
           return r

main = run loop
loop = tick >> loop

```

实际上，这个简单的无限循环会泄漏空间。（读者可以自行尝试。）这正是 LWT 的作者们遇到的问题。我不喜欢把博客文章分成小块，但是这段代码的正确编写花了比我预期的时间长一些，而我也没有时间了——所以请等下次再详细处理吧！
