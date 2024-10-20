<!--yml

分类：未分类

日期：2024-07-01 18:17:29

-->

# 表示完美二叉树的两种方式：ezyang’s 博客

> 来源：[`blog.ezyang.com/2012/08/statically-checked-perfect-binary-trees/`](http://blog.ezyang.com/2012/08/statically-checked-perfect-binary-trees/)

讨论许多分治算法时的一个常见简化假设是输入列表的大小是二的幂。因此，人们可能会想：*我们如何对具有二次幂大小的列表进行编码*，以一种不可表示其他属性的方式呢？一个观察是这样的列表是 *完美二叉树*，因此如果我们有一个完美二叉树的编码，我们也有一个二次幂列表的编码。以下是在 Haskell 中实现此类编码的两种众所周知的方法：一种使用 GADTs，另一种使用嵌套数据类型。我们声称嵌套数据类型的解决方案更为优越。

这篇文章是文学的，但你需要一些类型系统的特性：

```
{-# LANGUAGE ScopedTypeVariables, GADTs, ImpredicativeTypes #-}

```

### GADTs

一种方法是将树的大小编码到类型中，然后断言两棵树的大小相同。这在 GADTs 中相当容易实现：

```
data Z
data S n

data L i a where
    L :: a -> L Z a
    N :: L i a -> L i a -> L (S i) a

```

通过重用类型变量 `i`，`N` 的构造函数确保我们组合的任意两棵树必须具有相同的大小。这些树可以像普通的二叉树一样解构：

```
exampleL = N (N (L 1) (L 2)) (N (L 3) (L 4))

toListL :: L i a -> [a] -- type signature is necessary!
toListL (L x) = [x]
toListL (N l r) = toListL l ++ toListL r

```

从普通列表创建这些树有点微妙，因为 `i` 类型变量需要小心处理。对列表的存在性也相当有效：

```
data L' a = forall i. L' { unL' :: L i a }
data Ex a = forall i. Ex [L i a]

fromListL :: [a] -> L' a
fromListL xs = g (Ex (map L xs))
  where
    g (Ex [x]) = L' x
    g (Ex xs)  = g (Ex (f xs))
    f (x:y:xs) = (N x y) : f xs
    f _ = []

```

### 嵌套数据类型

另一种方法是直接构建一个等同于 2^n 大小元组的类型（考虑惰性）。例如，在 4-元组的情况下，我们只需写成 `((1, 2), (3, 4))`。然而，还有一个棘手的问题，即如何对这样的结构进行递归。这里使用的技术是引导，由 Adam Buchsbaum 在他的论文中描述，并由 Chris Okasaki 推广：

```
data B a = Two (B (a, a)) | One a
    deriving Show

```

注意递归提到 `B` 的情况并不持有 `a`，而是 `(a, a)`：这就是所谓的“非均匀”递归。

```
exampleB = Two (Two (One ((1,2), (3,4))))

fromListB :: [a] -> B a
fromListB [x] = One x
fromListB xs = Two (fromListB (pairs xs))
    where pairs (x:y:xs) = (x,y) : pairs xs
          pairs _ = []

toListB :: B a -> [a]
toListB (One x) = [x]
toListB (Two c) = concatMap (\(x,y) -> [x,y]) (toListB c)

```

### 哪个更好？

乍一看，GADT 方法似乎更有吸引力，因为在解构时，数据类型看起来和感觉上很像普通的二叉树。然而，将用户数据解析成嵌套数据类型比解析成 GADTs 要容易得多（由于 Haskell 不是依赖类型语言）。Ralf Hinze 在他的论文 [Perfect Trees and Bit-reversal Permutations](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.46.1095) 中，提出了另一个支持嵌套数据类型的论点：

> 比较[完美树和二叉树的通常定义]，很明显第一个表示比第二个更为简洁。如果我们估计一个*k*-ary 构造器的空间使用量为*k+1*个单元，我们可以看到，第一个完美树的排名*n*消耗了*(2^n-1)3+(n+1)2*个单元，而第二个则消耗了*(2^n-1)3+2*2^n*个单元。 [这一差异源于所有额外的叶节点。]

尽管如此，解构嵌套数据类型树非常奇怪，如果有一个从传统树上的卡范略图`(n :: t a -> t a -> t a , z :: a -> t a)`到嵌套数据类型的有效转换，我们可能会对“异国情调”的嵌套数据类型感到更满意：

```
cataL :: (t a -> t a -> t a, a -> t a) -> L i a -> t a
cataL (n,z) (N l r) = n (cataL (n,z) l) (cataL (n,z) r)
cataL (n,z) (L x) = z x

```

对于我们的嵌套数据类型树进行一个卡范略图`(f :: a -> t a, g :: t (a, a) -> t a)`：

```
cataB :: (forall a. a -> t a, forall a. t (a, a) -> t a) -> B a -> t a
cataB (f,g) (One a) = f a
cataB (f,g) (Two t) = g (cataB (f,g) t)

```

尽管如此，这种转换是可能的，但遗憾的是，它不是一个卡范略图：

```
cataLB :: forall t a. (t a -> t a -> t a, a -> t a) -> B a -> t a
cataLB (n,z) t = f t z
  where
    f :: forall b. B b -> (b -> t a) -> t a
    f (One a) z = z a
    f (Two t) z = f t (\(l,r) -> n (z l) (z r))

```

思路是创建一个函数`(a -> t a) -> t a`，然后我们传入`z`以获取最终结果。这是一种历时已久的差异列表/延续传递技巧，我们在其中建立一系列函数调用链，而不是直接尝试建立结果，因为通常嵌套数据类型树上的卡范略图是朝错误的方向进行的。但现在，我们可以轻松地对我们的嵌套数据类型树执行任何我们在普通树上做过的折叠操作，这解决了我们可能有的任何未解决的问题。无论如何，从表示大小的角度来看，嵌套数据类型是优越的。 （有关该问题的另一种看法，请参阅 Jeremy 的评论。）

欲了解更多信息，请查看[嵌套数据类型的广义折叠](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.1517)（Richard Bird，Ross Paterson）。
