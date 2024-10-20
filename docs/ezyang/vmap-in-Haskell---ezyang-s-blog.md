<!--yml

category: 未分类

date: 2024-07-01 18:16:52

-->

# vmap in Haskell : ezyang’s blog

> 来源：[`blog.ezyang.com/2020/01/vmap-in-haskell/`](http://blog.ezyang.com/2020/01/vmap-in-haskell/)

[vmap](https://github.com/google/jax#auto-vectorization-with-vmap) 是 JAX 推广的一种接口，为您提供向量化映射。从语义上讲，vmap 与 Haskell 中的 map 完全等效；关键区别在于，在 vmap 下运行的操作是向量化的。如果对卷积和矩阵乘法进行映射，您将得到一个大循环，它会重复调用每个批次条目的卷积和矩阵乘法。如果 *vmap* 一个卷积和矩阵乘法，您将调用这些操作的批量实现一次。除非您有一个融合器，在大多数现代深度学习框架上，调用这些操作的批处理实现会更快。

JAX 实现 vmap 的方式略显复杂；它们有一个“批量解释器”，将原始操作转换为它们的批量版本，并且必须跟踪有关哪些张量是批量化的以及以何种方式批量化的元数据，以便能够插入适当的广播和展开操作。我向 Simon Peyton Jones 提到了这一点，他立即问道，Haskell 的类型检查器不能自动处理这个吗？答案是可以！JAX 需要进行的所有簿记实际上是在运行时进行类型推断；如果您有一个可以在编译时为您完成这项工作的编译器，那么几乎没有什么需要实现的了。

揭示结论，我们将实现一个 vmap 函数族，用于运行以下两个示例：

```
example1 :: [Float] -> [Float] -> [Float]
example1 a0 b0 =
  vmap0_2 (\a b -> add a b) a0 b0

example2 :: [Float] -> [Float] -> [[Float]]
example2 a0 b0 =
  vmap0 (\a -> vmap1 (\b -> add a b) b0) a0

```

在解释器中运行时，我们将看到：

```
*Test> example1 [1,2,3] [4,6,8]
[5.0,8.0,11.0]
*Test> example2 [1,2,3] [4,6,8]
[[5.0,7.0,9.0],[6.0,8.0,10.0],[7.0,9.0,11.0]]

```

这些结果与您使用普通的 `map` 得到的结果相等；然而，在 vmap 的实现中没有循环。（无法编写一个普适的 vmap 的事实是 Haskell 的一个限制；我们稍后会更详细地讨论这一点。）

* * *

我们需要一些语言扩展，所以让我们先把这个问题解决掉：

```
{-# LANGUAGE RankNTypes, GADTs, MultiParamTypeClasses,
             KindSignatures, TypeApplications, FunctionalDependencies,
             FlexibleContexts, FlexibleInstances, UndecidableInstances,
             IncoherentInstances #-}

```

我们的攻击计划是，我们希望编写 `vmap` 的定义，以便推断出 `add` 的类型，从而清晰地显示出必要的广播。 `vmap` 的一个微不足道的实现将具有签名 `([a] -> [b]) -> [a] -> [b]`（也就是恒等函数），但标准列表类型并不允许我们区分应一起广播的维度和不应一起广播的维度（这就是为什么 `example1` 和 `example2` 得到不同结果的原因：在 `example2` 中，我们沿着每个维度分别广播，因此最终得到一个笛卡尔积；在 `example1` 中，我们将维度一起广播并获得了“zip”的行为）。每个不同的 `vmap` 调用应该给我们一个新的维度，这些维度不应与其他 `vmap` 调用混淆。当你在 Haskell 中听到这些时，你的第一反应应该是，“我知道了，让我们使用一个二阶类型！” `vmap` 将我们从普通列表 `[Float]` 的非类型品牌世界移动到带有大小索引向量 `Vec s Float` 的类型品牌世界，其中 `s` 变量都是由我们的二阶类型约束的 skolem 变量：

```
data Vec s a = Vec { unVec :: [a] }
instance Functor (Vec s) where
  fmap f (Vec xs) = Vec (map f xs)

vmap0 :: (forall s. Vec s a -> Vec s b) -> [a] -> [b]
vmap0 f = unVec . f . Vec

```

`vmap0` 的实现什么也不做：我们只是将列表包装成它们的类型品牌等效向量。我们还可以提供 `vmap0` 的二元版本，它一次接受两个列表并分配它们相同的类型品牌：

```
vmap0_2 :: (forall s. Vec s a -> Vec s b -> Vec s c) -> [a] -> [b] -> [c]
vmap0_2 f a b = unVec (f (Vec a) (Vec b))

```

（原则上，一些类似 applicative 的东西应该使得我们可以仅写一个 `vap`（类似于 `ap`），然后免费获取所有 n-ary 版本，但在我简短的调查中，我没有看到一个好的方法来实现这一点。）

当我们嵌套 `vmap` 时，函数可能并不直接返回 `Vec s b`，而是包含 `Vec s b` 的函子。 `vmap1` 处理这种情况（我们稍后将更详细地讨论这一点）：

```
vmap1 :: Functor f => (forall s. Vec s a -> f (Vec s b)) -> [a] -> f [b]
vmap1 f = fmap unVec . f . Vec

```

有了我们手头的 `vmap` 实现，我们可以查看我们的示例，并询问 Haskell 如果我们没有它的实现，`add` 的类型应该是什么：

```
example1 :: [Float] -> [Float] -> [Float]
example1 a0 b0 =
  vmap0_2 (\a b -> _add a b) a0 b0

```

得到：

```
• Found hole: _add :: Vec s Float -> Vec s Float -> Vec s Float
  Where: ‘s’ is a rigid type variable bound by
           a type expected by the context:
             forall s. Vec s Float -> Vec s Float -> Vec s Float

```

然而：

```
example2 :: [Float] -> [Float] -> [[Float]]
example2 a0 b0 =
  vmap0 (\a -> vmap1 (\b -> _add a b) b0) a0

```

得到：

```
• Found hole:
    _add :: Vec s Float -> Vec s1 Float -> Vec s (Vec s1 Float)
  Where: ‘s1’ is a rigid type variable bound by
           a type expected by the context:
             forall s1\. Vec s1 Float -> Vec s (Vec s1 Float)
           at test.hs:41:20-44
         ‘s’ is a rigid type variable bound by
           a type expected by the context:
             forall s. Vec s Float -> Vec s [Float]
           at test.hs:41:7-48

```

注意，这两种情况下 `_add` 的推断类型是不同的：在第一个示例中，我们推断出两个张量以相同方式进行批处理，并且我们想要将它们“zip”在一起。在第二个示例中，我们看到每个张量具有不同的批处理维度，最终得到一个二维结果！

到此为止，`vmap` 的工作已经完成：我们的洞有了我们可以用来确定必要行为的类型。你可以使用这些类型来选择执行矢量化加法的适当内核。但我承诺提供可运行的代码，所以让我们使用传统的 `map` 实现一个简单版本的 `add`。

在 Haskell 中进行类型级计算的传统方式当然是使用类型类！让我们为函数 `add` 定义一个多参数类型类；与 `Num` 中的 `(+)` 定义不同，我们允许输入和输出都具有不同的类型：

```
class Add a b c | a b -> c where
  add :: a -> b -> c

```

我们可以轻松地对普通浮点数进行加法实现：

```
instance Add Float Float Float where
  add = (+)

```

如果我传入两个参数，它们最外层的向量类型一致（也就是它们来自同一个 vmap），我应该像我在`example1`中所做的那样将它们一起压缩。我可以编写另一个实例来表达这个逻辑：

```
instance Add a b r  => Add (Vec s a) (Vec s b) (Vec s r) where
  add (Vec a) (Vec b) = Vec (zipWith add a b)

```

否则，我应该广播一个维度，然后在内部进行加法。这个选择不能在本地轻易完成，所以我必须定义这两个不一致的实例：

```
instance Add a b r => Add (Vec s a) b (Vec s r) where
  add (Vec a) b = Vec (map (\x -> add x b) a)

instance Add a b r => Add a (Vec s b) (Vec s r) where
  add a (Vec b) = Vec (map (\x -> add a x) b)

```

（GHC 的类型类解析引擎不会回溯，所以我不确定它是如何成功选择要使用的正确实例的，但在我的测试中，无论我如何指定 add 的参数顺序，我都得到了正确的实例。）

就这样！运行这两个示例：

```
example1 :: [Float] -> [Float] -> [Float]
example1 a0 b0 =
  vmap0_2 (\a b -> add a b) a0 b0

example2 :: [Float] -> [Float] -> [[Float]]
example2 a0 b0 =
  vmap0 (\a -> vmap1 (\b -> add a b) b0) a0

```

我得到：

```
*Test> example1 [1,2,3] [4,6,8]
[5.0,8.0,11.0]
*Test> example2 [1,2,3] [4,6,8]
[[5.0,7.0,9.0],[6.0,8.0,10.0],[7.0,9.0,11.0]]

```

* * *

所以这就是它！在不到十行的 Haskell 代码中使用 vmap。关于这种实现令人不满意的一点是必须定义`vmap0`、`vmap1`等。我们不能只定义一个通用的`vmapG :: (forall s. Vec s a -> f (Vec s b)) -> [a] -> f [b]`，然后在需要时将`f`统一为恒等类型 lambda `/\a. a`吗？遗憾的是，带类型 lambda 的类型推断是不可判定的（即所谓的高阶一致性问题），所以在这里似乎我们必须帮助 GHC，即使在我们的特定情况下，我们可以在这里进行的统一非常受限制。
