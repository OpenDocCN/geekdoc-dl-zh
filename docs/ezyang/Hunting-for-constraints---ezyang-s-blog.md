<!--yml

类别：未分类

日期：2024-07-01 18:18:27

-->

# 寻找约束：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/02/hunting-for-constraints/`](http://blog.ezyang.com/2010/02/hunting-for-constraints/)

```
> import Data.List
> import Control.Monad

```

以下问题作为[基于数字的谜题](http://www.mit.edu/~puzzle/10/puzzles/2010/fun_with_numbers/)的一部分出现在[2010 年 MIT 神秘猎](http://www.mit.edu/~puzzle/10/)中：

> 他在[Wizard of Wor](http://hackage.haskell.org/package/dow)的最终级别等于以精确 9 种方式写成 4 个非零平方数之和的最小数。

我想探索在 Haskell 中使用约束搜索来解决这个问题。希望能找到一个（搜索）程序，直接反映出所提出的问题，并给出一个答案！

因为我们正在寻找最小的数，所以从一个小数开始测试并逐渐计数是有意义的。我们假设这个问题的答案不会导致 Int 溢出。

现在，我们需要测试是否可以将其写成精确 9 种方式的 4 个非零平方数的和。这个问题归结为“n 可以用多少种方式写成平方和”，这是另一个搜索问题。

我们假设 `4+1+1+1` 和 `1+4+1+1` 在我们的九宫格目的中不构成不同的格局。这带来了第一个巧妙之处：如果我们对我们的九宫格施加严格的顺序，我们再次得到唯一性。

我们还需要限定我们的搜索空间；虽然公平搜索可以在某种程度上帮助我们处理无限失败，但如果我们可以进行一些早期终止，那么我们的实现将会简单得多。一个非常简单的终止条件是如果平方和超过我们正在寻找的数字。

考虑我们匹配 x 的情况，并且我们有候选根 a、b 和 c。然后，剩余平方的最大值可以是 x - a² - b² - c²，d 的最大值是平方根的底部。平方根很便宜，我们使用机器大小的整数，所以情况很好。

```
> floorSqrt :: Int -> Int
> floorSqrt = floor . sqrt . fromIntegral
>
> sumSquares :: [Int] -> Int
> sumSquares as = sum (map (²) as)
>
> rootMax :: Int -> [Int] -> Int
> rootMax x as = floorSqrt (x - sumSquares as)

```

从那里开始，我们仅需列出搜索非重复的平方数和的方法：

```
> searchSumFourSquares :: Int -> [(Int, Int, Int, Int)]
> searchSumFourSquares x = do
>       a <- [1..(rootMax x [])]
>       b <- [a..(rootMax x [a])]
>       c <- [b..(rootMax x [a,b])]
>       d <- [c..(rootMax x [a,b,c])]
>       guard $ sumSquares [a,b,c,d] == x
>       return (a,b,c,d)

```

从那里，解决方案自然而然地得出：

```
> search :: Maybe Int
> search = findIndex (==9) (map (length . searchSumFourSquares) [0..])

```

我们巧妙地使用`[0..]`，这样索引就与数字本身相同。其他方法可能使用元组。
