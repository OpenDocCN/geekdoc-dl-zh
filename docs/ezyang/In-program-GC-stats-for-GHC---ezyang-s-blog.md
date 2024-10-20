<!--yml

category: 未分类

date: 2024-07-01 18:17:42

-->

# In-program GC stats for GHC : ezyang’s blog

> 来源：[`blog.ezyang.com/2011/07/in-program-gc-stats-for-ghc/`](http://blog.ezyang.com/2011/07/in-program-gc-stats-for-ghc/)

-   我将参加今年的[Hac Phi](http://www.haskell.org/haskellwiki/Hac_%CF%86)（将在一周半后举行），我计划在 GHC 的程序中工作，收集垃圾收集器的统计信息。这个任务并不是技术上的难题（我们只需要在运行时系统中暴露一些函数），但迄今尚未完成。我知道许多注重性能和长期运行服务器的人都希望看到这样的功能。

-   我想问你的问题是：你希望这样的 API 看起来如何？它应该提供哪些功能，你希望如何与之交互？

Here’s one sample API to get the ball rolling:

```
module GHC.RTS.Stats where

-- Info is not collected unless you run with certain RTS options.  If
-- you are planning on using this on a long-running server, costs of the
-- options would be good to have (we also probably need to add extra
-- options which record, but have no outwardly visible effect.)

-- Read out static parameters that were provided via +RTS

generations :: IO Int

---------------------------------------------------------------------
-- Full statistics

-- Many stats are internally collected as words. Should be publish
-- words?

-- Names off of machine readable formats

bytesAllocated :: IO Int64
numGCs :: IO Int64
numByteUsageSamples :: IO Int64
averageBytesUsed :: IO Int64 -- cumulativeBytesUsed / numByteUsageSamples
maxBytesUsed :: IO Int64
-- peakMemoryBlocksAllocated :: IO Int64
peakMegabytesAllocated :: IO Int64
initCpuSeconds :: IO Double
initWallSeconds :: IO Double
mutatorCpuSeconds :: IO Double
mutatorWallSeconds :: IO Double
gcCpuSeconds :: IO Double
gcWallSeconds :: IO Double

-- Wouldn't be too unreasonable to offer a data structure with all of
-- this?  Unclear.  At least, it would prevent related data from
-- desynchronizing.

data GlobalStats = GlobalStats
    { g_bytes_allocated :: Int64
    , g_num_GCs :: Int64
    , g_num_byte_usage_samples :: Int64
    , g_average_bytes_used :: Int64
    , g_max_bytes_used :: Int64
    , g_peak_megabytes_allocated :: Int64
    , g_init_cpu_seconds :: Double
    , g_init_wall_seconds :: Double
    , g_mutator_cpu_seconds :: Double
    , g_mutator_wall_seconds :: Double
    , g_gc_cpu_seconds :: Double
    , g_gc_wall_seconds :: Double
    }
globalStats :: IO GlobalStats
generationStats :: Int -> IO GlobalStats

---------------------------------------------------------------------
-- GC statistics

-- We can't offer a realtime stream of GC events, because they come
-- to fast. (Test? eventlog comes to fast, maybe GC is manageable,
-- but you don't want to trigger GC in your handler.)

data GCStats = GCStats
    { gc_alloc :: Int64
    , gc_live :: Int64
    , gc_copied :: Int64
    , gc_gen :: Int
    , gc_max_copied :: Int64
    , gc_avg_copied :: Int64
    , gc_slop :: Int64
    , gc_wall_seconds :: Int64
    , gc_cpu_seconds :: Int64
    , gc_faults :: Int
    }
lastGC :: IO GCStats
lastMajorGC :: IO GCStats
allocationRate :: IO Double

---------------------------------------------------------------------
-- Parallel GC statistics

data ParGCStats = ParGCStats
    { par_avg_copied :: Int64
    , par_max_copied :: Int64
    }
parGCStats :: IO ParGCStats
parGCNodes :: IO Int64

---------------------------------------------------------------------
-- Threaded runtime statistics
data TaskStats = TaskStats
    -- Inconsistent naming convention here: mut_time or mut_cpu_seconds?
    -- mut_etime or mut_wall_seconds? Hmm...
    { task_mut_time :: Int64
    , task_mut_etime :: Int64
    , task_gc_time :: Int64
    , task_gc_etime :: Int64
    }

---------------------------------------------------------------------
-- Spark statistics

data SparkStats = SparkStats
    { s_created :: Int64
    , s_dud :: Int64
    , s_overflowed :: Int64
    , s_converted :: Int64
    , s_gcd :: Int64
    , s_fizzled :: Int64
    }
sparkStats :: IO SparkStats
sparkStatsCapability :: Int -> IO SparkStats

```
