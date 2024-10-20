<!--yml

分类：未分类

日期：2024-07-01 18:17:46

-->

# 一个不够懒惰的 map：[ezyang 博客](http://blog.ezyang.com/2011/05/an-insufficiently-lazy-map/)

> 来源：[`blog.ezyang.com/2011/05/an-insufficiently-lazy-map/`](http://blog.ezyang.com/2011/05/an-insufficiently-lazy-map/)

另一个常见的 thunk 泄漏源于在容器上映射函数时，并没有严格执行它们的组合函数。通常的修复方法是改用函数的严格版本，比如 `foldl'` 或 `insertWith'`，或者完全使用一个严格版本的结构。在今天的文章中，我们将更仔细地看待这种情况。特别是，我想回答以下几个问题：

### 示例

我们的例子是一个非常简单的数据结构，即 spine-strict 链表：

```
data SpineStrictList a = Nil | Cons a !(SpineStrictList a)
ssFromList [] l = l
ssFromList (x:xs) l = ssFromList xs (Cons x l)
ssMap _ Nil l = l
ssMap f (Cons x xs) l = ssMap f xs (Cons (f x) l)

main = do
    let l = ssFromList ([1..1000000] :: [Int]) Nil
        f x = ssMap permute x Nil
    evaluate (f (f (f (f (f (f (f (f l))))))))

permute y = y * 2 + 1

```

我们首先使用 `ssFromList` 创建数据结构的一个实例，然后使用 `ssMap` 对其所有元素进行映射。我们假设列表的结构在语义上并不重要（毕竟，对于用户来说，不透明数据结构中树的分布可能仅仅出于性能原因是没有兴趣的。实际上，每当调用 `ssFromList` 和 `ssMap` 时，它们都会反转结构，以避免堆栈溢出）。这里的空间泄漏典型地展示了“非严格容器函数”问题，即像 `map` 这样的函数看起来无害，实际上会导致问题。

如果你仔细看这个实现，这并不奇怪，基于对 `SpineStrictList` 的粗略查看：当然会积累 thunk，因为它不严格于值，只对结构本身严格。让我们看看一些解决方法。

### 修复

*Bang-pattern permute.* 这个修复方法很诱人，特别是如果你在考虑[我们上一个例子](http://blog.ezyang.com/2011/05/anatomy-of-a-thunk-leak/)：

```
permute !y = y * 2 + 1

```

但这是错误的。为什么错呢？首先，我们实际上并没有改变这个函数的语义：`y` 已经是严格的了！导致的 `seq` 嵌入表达式太深；我们需要更早地调用 `permute y`，而不是 `y`。还要记住，上次修复组合函数仅起作用是因为我们成功启用了 GHC 优化，它使元组变成非分配，从而完全避免了它们的分配。然而，在这里行不通，因为我们有一个 GHC 不知道能否摆脱的严格数据结构，所以所有分配总是会发生。

*在每次迭代中强制求值结构。* 这种方法虽然有效，但相当不优雅且效率低下。实质上，你每次都要遍历一遍，导致最终的运行时间是二次的，仅仅是为了确保所有东西都被评估了。`rnf`就像是一个重锤，通常最好避免使用它。

*使用 ssMap 的严格版本。* 这是一个相当普通的反应，任何改过 `foo` 函数为 `foo'` 版本的人都已经尝试过：

```
ssMap' _ Nil l = l
ssMap' f (Cons x xs) l = ssMap' f xs ((Cons $! f x) l)

```

剩余的空间使用仅仅是严格的数据结构在内存中的存在。为了修复这个问题，我们必须进入并调整我们`SpineStrictList`的内部表示，以引入这种严格性。这是第一个问题的答案：我们无法通过修改组合函数来修复这个空间泄漏，因为我们需要的额外严格性需要“附加”（使用`seq`）到数据结构本身的外部构造函数上：这是只有当你能够操作数据结构的内部结构时才能访问到的东西。

这样做的一个好处是，当你喜欢的容器库无法提供你需要的函数的严格版本时，这是相当令人恼火的。事实上，历史上容器包一直存在这个问题，尽管我最近已经提交了一个提案来帮助解决这个问题。

*使结构体的值严格。*这是将`ssMap`转换为其严格版本的“更好”方法，因为惰性模式将为您完成所有的序列化工作：

```
data StrictList a = Nil | Cons !a !(SpineStrictList a)

```

当然，如果你真的想要一个脊柱严格但值惰性的列表，这并不是最好的选择。然而，从灵活性的角度来看，完全严格的数据结构确实更加灵活。这是因为你总是可以通过增加额外的间接性来模拟值惰性的版本：

```
data Lazy a = Lazy a
type SpineStrictList a = StrictList (Lazy a)

```

现在构造函数`Lazy`被强制执行，但其内部未必会。你不能利用延迟数据结构来完成这一技巧，因为你需要所有函数的合作，以便在所有情况下评估容器的内部。然而，这种方法有一个缺点，即额外的包装器在内存和指针间接方面确实会造成成本。

*使结构体变得懒惰。*有趣的是，如果我们*添加*了惰性，空间泄漏就消失了：

```
data SpineStrictList a = Nil | Cons a (SpineStrictList a)

instance NFData a => NFData (SpineStrictList a) where
    rnf Nil = ()
    rnf (Cons x xs) = rnf x `seq` rnf xs

main = do
    let l = ssFromListL ([1..1000000] :: [Int])
        f x = ssMapL permute x
    evaluate (rnf (f (f (f (f (f (f (f (f l)))))))))

ssFromListL [] = Nil
ssFromListL (x:xs) = Cons x (ssFromListL xs)
ssMapL _ Nil = Nil
ssMapL f (Cons x xs) = Cons (f x) (ssMapL f xs)

```

我们添加了一个`rnf`来确保所有东西实际上都被评估了。事实上，空间使用显著改善了！

发生了什么？技巧在于，因为数据结构是惰性的，我们实际上并没有一次性创建 1000000 个 thunk；相反，我们只在任何给定时间创建表示列表头部和尾部的 thunk。两者远小于一百万，因此内存使用量相应减少。此外，因为在评估完元素后，`rnf`不需要保留列表的元素，所以我们能够立即进行垃圾回收。

*融合。* 如果你移除我们类似列表的数据构造器包装器，并使用内置的列表数据类型，你会发现 GHC 能够将所有的映射合并为一个极快的非装箱操作：

```
main = do
    let l = [1..1000000] :: [Int]
        f x = map permute x
    evaluate (rnf (f (f (f (f (f (f (f (f l)))))))))

```

这并不完全公平：我们可以用我们严格的代码做同样的技巧；然而，我们不能使用简单的 foldr/build 融合，因为它对于 foldl（带有累积参数的递归）是无效的。我们也不能将我们的函数转换为 foldr，否则在大输入时会有堆栈溢出的风险（尽管在树状数据结构中可以施加对其脊柱大小的对数界限，这可能是可以接受的）。对我来说，也不清楚脊柱严格性是否会为融合带来任何好处，尽管它在值严格性存在时肯定可以更好地运作。
