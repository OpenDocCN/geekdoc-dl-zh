<!--yml

category: 未分类

date: 2024-07-01 18:18:26

-->

# Comonads and Convolutions : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/02/comonads-and-convolutions/`](http://blog.ezyang.com/2010/02/comonads-and-convolutions/)

```
> {-# LANGUAGE GeneralizedNewtypeDeriving #-}
> import Control.Comonad
> import Data.List

```

来自 `category-extras` 的那个可怕的 `Control.Comonad` 导入将成为今天文章的主题。我们将看看一个可能的非空列表的余单子实现，它模拟因果时不变系统，这些系统的输出仅依赖于过去的输入。我们将看到这些系统中的计算遵循余单子结构，并且该结构的一个实例强烈强制执行因果性和弱化时不变性。

我们的因果列表简单来说就是一个带有额外约束的 `newtype` 列表，即它们不为空；`causal` 是一个“智能构造器”，用来强制执行这个约束。我们使用 `GeneralizedNewtypeDeriving` 来自动获得 `Functor` 实例。

```
> newtype Causal a = Causal [a]
>    deriving (Functor, Show)
>
> causal :: a -> [a] -> Causal a
> causal x xs = Causal (x:xs)
>
> unCausal :: Causal a -> [a]
> unCausal (Causal xs) = xs
>
> type Voltage = Float

```

*背景.*（如果您已经熟悉信号处理，请随意跳过此部分。）这样的系统模拟了跨不完美电线通道的电压样本的点对点通信。在理想世界中，我们非常希望能够假装我将任何电压输入到这个通道中，它将立即完美地将这个电压传输到通道的另一端。实际上，我们会看到各种不完美，包括上升和下降的时间，延迟，振铃和噪声。噪声是个扫兴的东西，所以在本文中我们将忽略它。

初步的近似条件可以对我们的系统施加以下重要的条件：

+   *因果性.* 我们的电线不能窥视未来并在甚至获得电压之前传输一些电压。

+   *时不变性.* 任何信号，无论现在发送还是延后发送，都会得到相同的响应。

+   *线性.* 对于电线来说是一个简单且有用的近似，它陈述了这个数学属性：如果输入 `x1` 得到输出 `y1`，输入 `x2` 得到输出 `y2`，那么输入 `Ax1 + Bx2` 将得到输出 `Ay1 + By2`。这也意味着我们得到了*叠加*，这是一个我们很快会使用的重要技术。

当你看到一个线性时不变系统时，这意味着我们可以使用一个喜欢的数学工具，即卷积。

*离散卷积.* 通道执行的离散化计算的总体结构是 `[Voltage] -> [Voltage]`；也就是说，我们输入一系列输入电压样本，得到另一系列输出电压样本。另一方面，离散卷积是由以下函数计算的（变量名称具有启发性）：

```
(u ∗ f)[n] = sum from m = -∞ to ∞ of f[m]u[n-m]

```

这里并不完全明显为什么卷积是我们在这里寻找的数学抽象，因此我们将简要推导一下。

我们计算的一个特殊情况是当输入对应于`[1, 0, 0, 0 ...]`，称为*单位样本*。实际上，由于线性性和时不变性，当我们的系统给定单位样本时，单位样本响应*精确地*指定了系统对所有输入的行为：任何可能的输入序列都可以由延迟和缩放的单位样本组成，并且线性性质告诉我们可以将所有结果加在一起得到一个结果。

实际上，一个列表实际上是一个函数`ℕ → a`，如果我们假设约定`f[n] = 0`对于`n < 0`。假设`f[n]`代表我们随时间变化的输入样本，`δ[n]`代表一个单位样本（`δ[0] = 1`，对所有其他`n`，`δ[n] = 0`；你通常会看到`δ[n-t]`，这是时间`t`的单位样本），而`u[n]`代表我们的单位样本响应。然后，我们将`f[n]`分解为一系列单位样本：

```
f[n] = f[0]δ[n] + f[1]δ[n-1] + ...

```

然后使用线性性质来检索我们的响应`g[n]`：

```
g[n] = f[0]u[n] + f[1]u[n-1] + ...
     = sum from m = 0 to ∞ of f[m]u[n-m]

```

看起来就像离散卷积，只是没有-∞的边界。请记住，我们定义了对于`m < 0`，`f[m] = 0`，因此这两者实际上是等价的。

在写出等价的 Haskell 之前，我想再谈一下最后的数学定义。我们最初声明输入-响应计算的类型是`[Voltage] -> [Voltage]`；然而，在我们的数学中，我们实际上定义了一个关系`[Voltage] -> Voltage`，一个特定通道的函数，它接受直到时间`n`的所有输入，即`f[0]..f[n]`，并返回单个输出`g[n]`。我用一种具有暗示性的柯里化形式写了以下定义，以反映这一点：

```
> ltiChannel :: [Voltage] -> Causal Voltage -> Voltage
> ltiChannel u = \(Causal f) -> sum $ zipWith (*) (reverse f) u

```

单位样本响应可以是有限或无限列表，出于效率考虑，建议使用有限列表：

```
> usr :: [Voltage]
> usr = [1,2,5,2,1]

```

*共函子*。现在，我们应该清楚我们一直在努力达成的目标：我们有`ltiChannel usr :: Causal Voltage -> Voltage`，而我们想要：`Causal Voltage -> Causal Voltage`。这正是共函子引起的计算形式！为了方便起见，这里是`Copointed`和`Comonad`类型类的定义：

```
class Functor f => Copointed f where
    extract :: f a -> a

class Copointed w => Comonad w where
    duplicate :: w a -> w (w a)
    extend :: (w a -> b) -> w a -> w b

```

`Copointed`实例非常直接，但说明了为什么`Causal`必须包含*非空*列表：

```
> instance Copointed Causal where
>    extract (Causal xs) = head xs

```

`Comonad`实例可以使用`duplicate`或`extend`定义；两者在彼此的默认实现中已定义。推导这些默认实现留给读者作为练习；我们将在这里定义两者：

```
> instance Comonad Causal where
>    extend f  = Causal . map (f . Causal) . tail . inits . unCausal
>    duplicate = Causal . map      Causal  . tail . inits . unCausal

```

代码的意图有些被`Causal`的解包和封装所遮蔽；对于一个纯列表，实例看起来像这样：

```
instance Comonad [] where
    extend f  = map f . tail . inits
    duplicate = tail . inits

```

函数 `duplicate` 真正深入了解到这个共单子实例所做的事情：我们将输入列表转换为历史记录列表，每一步都比上一步进一步。`tail` 跟随以丢弃 `inits` 的第一个值，这是一个空列表。`duplicate` 构建起 `w (w a)`，然后用户提供的函数将其拆解为 `w b`（如果你考虑到单子，提升的用户函数会构建起 `m (m b)`，然后 `join` 将其拆解为 `m b`。）

一个快速测试来确保它工作：

```
> unitStep :: Causal Voltage
> unitStep = Causal (repeat 1)
>
> result :: Causal Voltage
> result = unitStep =>> ltiChannel usr

```

而事实上，`result` 是：

```
Causal [1.0, 3.0, 8.0, 10.0, 11.0, 11.0, ...]

```

`=>>` 是一个翻转的 `extend`，是单子 `>>=` 的共单子等效物。

*强制不变性。* 我们以这种形式结构化我们的计算（而不是明确地写出该死的卷积）在我们的代码中产生了一些有趣的强制不变性。我们的通道不必是线性的；我可以在与单位样本响应卷积之前对所有输入进行平方处理，这显然不是线性的。然而，我们写的任何通道 *必须* 是因果的，并且通常是时不变的：它必须是因果的，因为我们从未将任何未来的值传递给用户函数，并且它是弱时不变的，因为我们不显式地让用户知道输入流的进度。在我们的实现中，他们可以通过 `length` 推测这些信息；我们可以使用一个将列表反转并附加 `repeat 0` 的组合器获得更强的保证：

```
> tiChannel :: ([Voltage] -> Voltage) -> Causal Voltage -> Voltage
> tiChannel f (Causal xs) = f (reverse xs ++ repeat 0)
>
> ltiChannel' :: [Voltage] -> Causal Voltage -> Voltage
> ltiChannel' u = tiChannel (\xs -> sum $ zipWith (*) u xs)

```

在这种情况下，`u` 必须是有限的，如果它是无限的，可以在某个点截断它，以指定我们的计算应该多精确。

*未解之谜。* 单位样本响应在我们的示例代码中被表达为 `[Voltage]`，但它实际上是 `因果电压`。不幸的是，共单子似乎没有指定结合共单子值的机制，就像列表单子自动结合列表每个值的计算结果一样。我有点好奇类似于这样的东西可能如何工作。
