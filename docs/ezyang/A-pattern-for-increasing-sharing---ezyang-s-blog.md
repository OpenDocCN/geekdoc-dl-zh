<!--yml

category: 未分类

date: 2024-07-01 18:17:45

-->

# 增加共享的模式：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/06/a-pattern-for-increasing-sharing/`](http://blog.ezyang.com/2011/06/a-pattern-for-increasing-sharing/)

我最近在编写一些 Haskell 代码时遇到了以下模式，并惊讶地发现标准库中实际上没有为其提供支持。我不知道它叫什么（当我向 Simon Peyton-Jones 提到它时，他也不知道），所以如果有人知道，请告诉我。这个模式是这样的：很多时候，一个自同态映射（`map`函数是`a -> a`）对底层数据结构不会做出太多改变。如果我们直接实现`map`，我们将不得不重建递归数据结构的整个脊柱。然而，如果我们使用`a -> Maybe a`的函数，如果没有更改，我们可以重用旧的映射部分。（我博客的常读者可能会从[this post.](http://blog.ezyang.com/2011/06/pinpointing-space-leaks-in-big-programs/)中认出这种情况。）那么这样的替代`map`函数`(a -> Maybe a) -> f a -> Maybe (f a)`叫什么？

有一个猜测它可能是`Data.Traversable`中的`traverse`函数：它的类型签名确实非常相似：`Applicative f => (a -> f b) -> t a -> f (t b)`。然而，语义上有微妙的不同，你可以从这个例子中看出来：

```
Data.Traversable> traverse (\x -> if x == 2 then Just 3 else Nothing) [1,2,3]
Nothing

```

请记住，我们的函数只在没有变化时返回`Nothing`。因此，我们*应该*得到结果`Just [1,3,3]`：列表的第一和第三个元素不变，而列表的第二个元素有新值。

我们如何为列表实现这样的函数？这里是一个简单的实现：

```
nonSharingMap :: (a -> Maybe a) -> [a] -> Maybe [a]
nonSharingMap f xs = let (b, r) = foldr g (False, []) (zip xs (map f xs))
                     in if b then Just r else Nothing
    where g (y, Nothing) (b, ys) = (b,     y:ys)
          g (_, Just y)  (_, ys) = (True,  y:ys)

```

但是我们可以做得更好。考虑一种情况，列表中除头部外所有元素保持不变：

我们希望在旧版本和新版本之间共享列表的尾部。稍加思索后，意识到`tails`可以实现共享，我们可以写出这个版本：

```
sharingMap :: (a -> Maybe a) -> [a] -> Maybe [a]
sharingMap f xs = let (b, r) = foldr g (False, []) (zip3 (tails xs) xs (map f xs))
                     in if b then Just r else Nothing
    where g (_,   y, Nothing) (True, ys)  = (True,  y:ys)
          g (_,   _, Just y)  (True, ys)  = (True,  y:ys)
          g (ys', _, Nothing) (False, _)  = (False, ys')
          g (_,   _, Just y)  (False, ys) = (True,  y:ys)

```

未解决的问题：这种模式叫什么？为什么它不遵循通常的应用结构？它是否满足某种高阶模式？此外，这种方案并非完全组合：如果我传递给你一个`Nothing`，你就无法访问原始版本，以防数据结构的其他地方发生变化：`(Bool, a)`可能更具有组合性。这是否意味着这是状态单子的一个示例？分享又如何呢？

*更新.* Anders Kaseorg 提供了一个更直接递归版本的函数：

```
sharingMap f [] = Nothing
sharingMap f (x : xs) = case (f x, sharingMap f xs) of
  (Nothing, Nothing) -> Nothing
  (y, ys) -> Just (fromMaybe x y : fromMaybe xs ys)

```

我还没有检查过，但是用`foldr`和`zip3`来表达这个函数的一个希望是能够进行融合。当然，对于实际的递归脊柱严格的数据类型，通常无法融合，因此更直接的展示方式更为正常。
