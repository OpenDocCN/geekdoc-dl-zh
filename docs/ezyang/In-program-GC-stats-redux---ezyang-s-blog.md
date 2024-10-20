<!--yml

category: 未分类

date: 2024-07-01 18:17:41

-->

# 程序内 GC 统计 redux : ezyang’s 博客

> 来源：[`blog.ezyang.com/2011/08/in-program-gc-stats-redux/`](http://blog.ezyang.com/2011/08/in-program-gc-stats-redux/)

## 程序内 GC 统计 redux

Hac Phi 还是相当富有成效的（因为我成功地写了两篇博客文章！）周六，我提交了一个新的模块 `GHC.Stats` 到 base，它实现了我之前提出的 [API 的修改子集。](http://blog.ezyang.com/2011/07/in-program-gc-stats-for-ghc/) 这里是 API；要使用它，你需要从 Git 编译 GHC。请测试并告诉我是否需要做出更改或澄清！

```
-- | Global garbage collection and memory statistics.
data GCStats = GCStats
    { bytes_allocated :: Int64 -- ^ Total number of bytes allocated
    , num_gcs :: Int64 -- ^ Number of garbage collections performed
    , max_bytes_used :: Int64 -- ^ Maximum number of live bytes seen so far
    , num_byte_usage_samples :: Int64 -- ^ Number of byte usage samples taken
    -- | Sum of all byte usage samples, can be used with
    -- 'num_byte_usage_samples' to calculate averages with
    -- arbitrary weighting (if you are sampling this record multiple
    -- times).
    , cumulative_bytes_used :: Int64
    , bytes_copied :: Int64 -- ^ Number of bytes copied during GC
    , current_bytes_used :: Int64 -- ^ Current number of live bytes
    , current_bytes_slop :: Int64 -- ^ Current number of bytes lost to slop
    , max_bytes_slop :: Int64 -- ^ Maximum number of bytes lost to slop at any one time so far
    , peak_megabytes_allocated :: Int64 -- ^ Maximum number of megabytes allocated
    -- | CPU time spent running mutator threads.  This does not include
    -- any profiling overhead or initialization.
    , mutator_cpu_seconds :: Double
    -- | Wall clock time spent running mutator threads.  This does not
    -- include initialization.
    , mutator_wall_seconds :: Double
    , gc_cpu_seconds :: Double -- ^ CPU time spent running GC
    , gc_wall_seconds :: Double -- ^ Wall clock time spent running GC
    -- | Number of bytes copied during GC, minus space held by mutable
    -- lists held by the capabilities.  Can be used with
    -- 'par_max_bytes_copied' to determine how well parallel GC utilized
    -- all cores.
    , par_avg_bytes_copied :: Int64
    -- | Sum of number of bytes copied each GC by the most active GC
    -- thread each GC.  The ratio of 'par_avg_bytes_copied' divided by
    -- 'par_max_bytes_copied' approaches 1 for a maximally sequential
    -- run and approaches the number of threads (set by the RTS flag
    -- @-N@) for a maximally parallel run.
    , par_max_bytes_copied :: Int64
    } deriving (Show, Read)

-- | Retrieves garbage collection and memory statistics as of the last
-- garbage collection.  If you would like your statistics as recent as
-- possible, first run a 'performGC' from "System.Mem".
getGCStats :: IO GCStats

```
