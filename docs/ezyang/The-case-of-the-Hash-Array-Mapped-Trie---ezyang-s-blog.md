<!--yml

category: 未分类

date: 2024-07-01 18:18:24

-->

# The case of the Hash Array Mapped Trie : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/03/the-case-of-the-hash-array-mapped-trie/`](http://blog.ezyang.com/2010/03/the-case-of-the-hash-array-mapped-trie/)

高效的关联映射长期以来一直是函数式编程社区的圣杯。如果你在命令式语言中需要这样的抽象数据结构，那么毫无疑问地，你会使用哈希表。但是哈希表建立在破坏性更新之上，这使得它在纯代码中难以使用。

我们正在寻找的是一个严格更强大的关联映射，它实现了*非破坏性*更新（即“持久性”）。在 Haskell 的世界中，[Data.Map](http://www.haskell.org/ghc/docs/6.10.4/html/libraries/containers/Data-Map.html) 是一个相当引人注目的通用结构，只需要其键具有 `Ord` 类型类。对于能够清晰映射到机器大小整数的键来说，[IntMap](http://hackage.haskell.org/packages/archive/containers/0.1.0.1/doc/html/Data-IntMap.html) 是一个极快的纯函数数据结构，它在大端 Patricia tries 之上使用了位操作技巧。

其他函数式编程语言推广了自己的数据结构：Clojure 的许多关键数据结构都是由 Phil Bagwell 发明的，其中包括[hash-array mapped trie（PDF）](http://lampwww.epfl.ch/papers/idealhashtrees.pdf)，这些数据结构驱动了 Clojure 的持久性关联映射。

纸面上，这些实现具有以下的渐近性能：

+   *Data.Map.* 让 *n* 和 *m* 分别表示地图中的元素数量。*O(log n)* 查找、插入、更新和删除操作。*O(n+m)* 并集、差集和交集操作。

+   *Data.IntMap.* 让 *n* 和 *m* 分别表示地图中的元素数量和机器大小整数（例如 32 或 64）中的位数。*O(min(n,W))* 查找、插入、更新和删除操作。*O(n+m)* 并集、差集和交集操作。

+   *Hash array mapped trie.* 让 *n* 表示地图中的元素数量。由于 [Hickey's implementation](http://github.com/richhickey/clojure/blob/master/src/jvm/clojure/lang/PersistentHashMap.java) 没有子树池或根重新调整，我们将在渐近性能中省略它们。*O(log(n))* 查找、插入、更新和删除操作。未描述并集、差集和交集的实现。

不幸的是，这些数字实际上并没有告诉我们关于这些数据结构的实际性能有多少，因为关联的世界足够竞争，常数因子真的很重要。因此，我构建了以下基准测试：生成 *N* 个随机数，并将它们插入映射中。然后，在这些随机数中进行 *N/2* 次查找，并对另外 *N/2* 个未使用的数字进行查找（这将构成未命中）。竞争者是 IntMap 和 HAMT（Java 和 Haskell 中的实现）。初步结果表明，IntMap 比 Java HAMT 更快，Java HAMT 比 Haskell HAMT 快得多。

当然，这完全是胡说八道。

我转向 [Clojure 邮件列表](http://groups.google.com/group/clojure/browse_thread/thread/776943086de213f9)，并向他们展示了一个奇怪的（不正确的）结果：Haskell 的 IntMap 比 Clojure 内置的 HAMT 实现快了多达五倍。Rich Hickey 立即指出了我的方法论存在三个问题：

+   我使用的是 Java 的默认堆大小（公平地说，我也使用了 Haskell 的默认堆大小），

+   没有使用 `-server` 标志进行测试，

+   我没有考虑 JVM 的基于配置文件的优化。

（还有一些关于随机数生成和交错的评论，但进一步测试表明这些成本可以忽略不计。）Rich 给了我一些新代码，使用 `(apply hash-map list-of-vals)` 构建哈希映射，在修复了一个问题后，Rich 只将 N/2 个条目插入哈希表，我继续前行。

通过改进的测试用例集，我得出了以下统计数据（源代码请查看[此处的 IntMap criterion 测试](http://github.com/ezyang/hamt/blob/master/IntMapTest.hs)，以及此博客文章的后记中的 Clojure 测试）：

```
        IntMap       Java HAMT (32K-512K)  Java HAMT (512K-32K)
 32K     .035s       .100s                  .042s
 64K     .085s       .077s                  .088s
128K     .190s       .173s                  .166s
256K     .439s       .376s                  .483s
512K    1.047s      1.107s                 1.113s

```

然而，令人困惑的是，我重新实现的 [HAMT 的 Haskell 版本](http://github.com/ezyang/hamt) 表现极差，即使在我用位操作技巧、GHC 的装箱和解箱折腾了自己之后，也比原版慢了三到四倍。然后，我有了一个顿悟：

```
public static PersistentHashMap create(List init){
  ITransientMap ret = EMPTY.asTransient();
  for(Iterator i = init.iterator(); i.hasNext();)
  {
    Object key = i.next();
    if(!i.hasNext())
      throw new IllegalArgumentException(String.format("No value supplied for key: %s", key));
      Object val = i.next();
      ret = ret.assoc(key, val);
    }
    return (PersistentHashMap) ret.persistent();
  }
}

```

Hickey 可能是个棘手的家伙：他正在使用变异（请注意 `asTransient` 调用），以优化 `(apply hash-map ...)` 的调用！稍作调整后强制使用函数接口，Voila：

```
        Haskell     Clojure
128K    0.56s       0.33s
256K    1.20s       0.84s
512K    2.62s       2.80s

```

更加可比的性能（如果您仔细观察 JVM 的数字，它们从与 Haskell 大致相同的速度开始，然后随着 HotSpot 的启动而加快。）

不幸的是，在 Haskell 的世界中我不能使用类似的技巧。首先，GHC 没有基于运行时配置文件的优化。此外，虽然我确实**可以**在 GHC 中不安全地冻结单个数组（这是许多包中的标准操作过程），但我不能递归地冻结指向数组的数组，而不是遍历整个结构。因此，使用变异进行递归数据结构的快速构建对于 Haskell 来说仍然是不可能的......暂时是。

这个故事还在不断发展之中。特别是，我还需要：

+   进行更加细致的基准测试，区分插入、查找和其他操作的成本；并

+   在 Java 中实现 IntMap 并观察 JVM 对算法的影响，统一垃圾收集策略也将会很有启发性。

*附言.* 你可以在 [Clojure 邮件列表](http://groups.google.com/group/clojure/browse_thread/thread/776943086de213f9) 上查看基准测试的详细内容。这是用于测试 Java 的 HAMT 实现的测试代码。

首先是变异版本：

```
(ns maptest (:gen-class))

(defn mk-random-stream []
  (let [r (new ec.util.MersenneTwisterFast)]
    (repeatedly (fn [] (. r (nextInt))))))

(defn main [i]
  (let [vals (vec (take (* i 2) (mk-random-stream)))
        dvals (take (* i 2) (doall (interleave vals vals)))]
    (dotimes [_ 10]
      (time
       (let [m (apply hash-map dvals)]
         (reduce (fn [s k] (+ s (m k 0)))
           0
           (take i (drop (/ i 2) vals))))))))

(doseq [n (range 5 10)]
  (let [i (* 1000 (int (Math/pow 2 n)))]
    (println " I = " i)
    (main i)))

```

这里是强制使用函数接口的备选主要定义：

```
(defn main [i]
  (let [vals (vec (take (* i 2) (mk-random-stream)))]
    (dotimes [_ 10]
      (time
       (let [m (reduce (fn [m x] (assoc m x x)) (hash-map) vals)]
         (reduce (fn [s k] (+ s (m k 0)))
           0
           (take i (drop (/ i 2) vals))))))))

```
