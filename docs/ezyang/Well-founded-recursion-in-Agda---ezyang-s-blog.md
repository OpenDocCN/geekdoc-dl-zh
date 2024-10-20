<!--yml

category: 未分类

date: 2024-07-01 18:18:16

-->

# Agda 中的良好递归：ezyang’s 博客

> 来源：[`blog.ezyang.com/2010/06/well-founded-recursion-in-agda/`](http://blog.ezyang.com/2010/06/well-founded-recursion-in-agda/)

上周二，Eric Mertens 在 Galois 的技术讲座上发表了 [Introducing Well-Founded Recursion](http://www.galois.com/blog/2010/06/11/tech-talk-introducing-well-founded-recursion/)。我得承认，第一次听到时大部分内容都超出了我的理解范围。以下是我重新阅读代码时写下的一些笔记。建议先阅读 [slides](http://code.galois.com/talk/2010/10-06-mertens.pdf) 以对演示有所了解。这些笔记是针对一个对类型系统感到舒适但不完全理解柯里-霍华德同构的 Haskell 程序员。

```
> module Quicksort where
>
> open import Data.Nat public using (ℕ; suc; zero)
> open import Data.List public using (List; _∷_; []; _++_; [_]; length; partition)
> open import Data.Bool public using (Bool; true; false)
> open import Data.Product public using (_×_; _,_; proj₁; proj₂)

```

Agda 是基于直觉主义类型论的证明辅助工具；也就是说，柯里-霍华德同构定理。*柯里-霍华德同构*表明看起来像类型和数据的东西也可以视为命题和证明，并且在理解 Agda 中的良好递归的关键之一是自由地在这两者之间交换，因为我们将使用类型系统来对我们的代码进行命题，而 Agda 在检查时会使用这些命题。我们将尝试呈现类型和命题的两种视角。

Types : Data :: Propositions : Proofs

Agda 需要确信你的证明是有效的：特别是，Agda 想知道你是否涵盖了所有情况（穷举模式匹配，完全性），并且你是否会推迟回答（终止性）。在情况检查方面，Agda 非常聪明：如果它知道某种情况在实践中无法实现，因为其类型代表一个虚假，它不会要求你填写该情况。然而，在终止性检查方面经常需要帮助，这就是良好递归的用武之地。

*热身。*

今天我们的第一个数据类型是 top：仅有一个值 unit 的类型，即 () 在 Haskell 中。数据居住在类型中，就像命题存在于命题中的证明一样；你可以把类型想象成“房子”，里面居住着任意数量的居民，即数据类型。经常会看到 Set 弹出：严格来说，它是“小”类型的类型，Set₁ 更大，Set₂ 更大，依此类推……

```
> data ⊤ : Set where unit : ⊤

```

Bottom 是一种根本没有任何东西的类型。如果没有命题的证明存在，那么它是假的！同样，在值级别上，这是 Haskell 中的未定义或错误“foobar”；在类型级别上，它被称为 Void，尽管在实际代码中没有人真正使用它。在 Agda 中，它们是同一种东西。

```
> data ⊥ : Set where

```

我们从 Data.Nat 中引入了自然数，但这里是最小定义的样子：

```
data ℕ : Set where
  zero : ℕ
  suc : ℕ → ℕ

```

值得注意的是，Agda 中的数值常量如 0 或 2 是零和 suc (suc zero) 的语法糖。它们也可能出现在类型中，因为 Agda 是依赖类型的。 （在 Haskell 中，你必须将自然数的定义推入类型系统；在这里，我们可以写一个正常的数据定义，然后自动提升它们。[力量给工人阶级！](http://strictlypositive.org/winging-jpgs/)）

这个函数做了一些非常奇怪的事情：

```
> Rel : Set → Set₁
> Rel A = A → A → Set

```

实际上，它等价于这个扩展版本：

```
Rel A = (_ : A) → (_ : A) → (_ : Set)

```

因此，结果类型不是 A → A → Set，而是某些 *其* 类型为 A 的东西，另一些 *其* 类型也是 A 的东西，结果是某些其类型为 Set 的东西。在 Haskell 的术语中，这不是类型函数的类型 `* → *`；这更像是一个非法的 `* -> (a -> a -> *)`。

这里是一个简单关系的例子：自然数的小于关系。

```
> data _<_ (m : ℕ) : ℕ → Set where
>   <-base : m < suc m
>   <-step : {n : ℕ} → m < n → m < suc n

```

Agda 语法并不那么简单：

+   (m : ℕ) 表示 _<_ 是由 m 参数化的，使得 m，一个类型为 ℕ 的值，在我们的数据构造函数中可用。参数化意味着它也是 _<_ 的第一个参数；此时，您应该检查所有构造函数的类型签名，确保它们确实是形式为 m<_ 的。

+   {n : ℕ} 表示一个“隐含”参数，这意味着当我们调用 <-step 时，我们不需要传递它；Agda 将自动从后面的参数中找到它，在这种情况下是 m < n。

+   记住，“对于所有 x : A，y : B”，等同于提供一个全函数 f(x : A) : B。因此有一个便捷的缩写 ∀ x →，等同于 (x : _) →（下划线表示任何类型都可以）。

语法已经解释清楚了，这个表达式的数学意图应该是清楚的：对于任意的数，我们自动得到证明 m<m+1；并且有了 m<n → m<n+1，我们可以归纳地得到其余的证明。如果你眯起眼睛看，你也可以理解数据的含义：<-base 是一个零元构造子，而 <-step 是一个递归构造子。

让我们证明 3 < 5。我们从 <-base 开始：3 < 4（我们怎么知道我们应该从这里开始，而不是从 4 < 5 开始？注意到 m，我们的参数，是 3：这是一个提示，我们所有的类型都将被参数化为 3。）应用一次 step：3 < suc 4 = 3 < 5，证毕。

```
> example₁ : 3 < 5
> example₁ = <-step <-base

```

记住，真命题由数据类型居住，而假命题则不然。我们如何反转它们呢？在逻辑中，我们可以说，“假设命题成立；推导出矛盾。”在类型理论中，我们使用空函数：这是一个没有定义域的函数，因此虽然存在，却不能接受任何输入。一个函数只有在其输入类型不居住时才没有定义域，所以我们能够避免给出矛盾的唯一方法是……一开始就不让它们提出这个问题！

```
> _≮_ : Rel ℕ
> a ≮ b = a < b → ⊥

```

（）表示假，比如（）：5 < 0，这显然永远不可能成立，因为<-base 不匹配它（suc m != 0）。值得一提的是，Agda 要求你的程序是完备的，但不要求你对荒谬情况进行模式匹配。

```
> example₂ : 5 ≮ 2
> example₂ (<-step (<-step ()))

```

*良好基础性。*

我们引入一些 Agda 符号；模块让我们能够在扩展块上对某个变量进行参数化，然后只需‘data’声明的构造函数。模块的成员可以像 WF.Well-founded A（其余参数）那样访问。这非常方便和惯用，虽然不是绝对必要；我们也可以只根据成员参数化。我们还碰巧在一个类型上进行参数化。

```
> module WF {A : Set} (_<_ : Rel A) where

```

从逻辑上讲，一个元素被认为是可访问的意思是对于所有 y，如 y < x，y 是可访问的。从数据和逻辑的角度看，它陈述如果你想让我给你 Acc x，你想要的数据/证明，你必须给我一个证明，对于所有 y，如果你给我一个证明 y < x，我可以确定 Acc y。现在我们正试图证明关于我们类型和函数的属性，严格将我们的数据类型视为纯粹数据的做法变得越来越不合理。

```
>   data Acc (x : A) : Set where
>     acc : (∀ y → y < x → Acc y) → Acc x

```

如果它内部的所有元素都是可访问的，整个类型 A 就是良好基础的。或者，如果给定它内部的一个元素，我能为该元素产生一个可访问性证明，整个类型 A 也是良好基础的。请注意，它的类型是 Set；这是我想要证明的命题！

```
>   Well-founded : Set
>   Well-founded = ∀ x → Acc x

```

关于自然数的良好基础性证明。

```
> <-ℕ-wf : WF.Well-founded _<_
> <-ℕ-wf x = WF.acc (aux x)
>   where
>     aux : ∀ x y → y < x → WF.Acc _<_ y
>     --  : (x : _) → (∀ y → y < x → WF.Acc _<_ y)
>     aux .(suc y) y <-base = <-ℕ-wf y

```

基本情况，（例如 x=5，y=4）。方便的是，这触发了对ℕ上的良好基于结构的递归，通过检查现在是否良好基于 y。

```
>     aux .(suc x) y (<-step {x} y<x) = aux x y y<x

```

这里的结构递归是在 _<_ 上进行的；我们在剥离<-step 的层级，直到 y<x = <-base，就像 3<4 的情况一样（但不是 3<6）。我们基本上是在诉诸一个较弱的证明，它仍然足以证明我们感兴趣的内容。注意，我们也在 x 上递归；实际上，无论我们了解 x 的多少，我们都是从 y<x 中了解的（信息内容较少！），所以我们用一个点来指示这一点。最终，x 会足够小，以至于 y 不会比 x 小得多（<-base）。

我们在哪里处理零？考虑 aux zero：∀ y -> y < zero → WF.Acc _<_ y。这是一个空函数，因为 y < zero = ⊥（没有自然数小于零！）事实上，这就是我们摆脱不编写 yx（上三角形）情况的方式：它等同于 y≮x，这些都是底部，免费提供给我们空函数。

实际上，在这里有一个双结构递归，一个是 x，另一个是 y<x。对 x 的结构递归只是在 aux 上，但一旦我们得出<-base，我们就对 y 进行不同的结构递归，使用<-ℕ-wf。这填补了由 y=x-1 分割的 xy 平面的右下三角形；上左三角形不太有趣，因为它只是废土的荒原。

标准数学技巧：如果你能将问题简化为你已经解决过的另一个问题，你就解决了你的问题！

```
> module Inverse-image-Well-founded { A B }
>   -- Should actually used ≺, but I decided it looked to similar to < for comfort.
>   (_<_ : Rel B)(f : A → B) where
>   _⊰_ : Rel A
>   x ⊰ y = f x < f y
>
>   ii-acc : ∀ {x} → WF.Acc _<_ (f x) → WF.Acc _⊰_ x
>   ii-acc (WF.acc g) = WF.acc (λ y fy<fx → ii-acc (g (f y) fy<fx))

```

类型必须正确，因此我们将旧证明 g 解包并包装成一个新的 lambda，通过 f 推动到我们的证明中（即 WF.acc 数据构造器）。

```
>   ii-wf : WF.Well-founded _<_ → WF.Well-founded _⊰_
>   ii-wf wf x = ii-acc (wf (f x))
>   --    wf = λ x → ii-acc (wf (f x))
>   -- I.e. of course the construction ii-acc will work for any x.

```

在这里，我们最终使用我们的机制证明列表与它们的长度相比是良基的。

```
> module <-on-length-Well-founded { A } where
>   open Inverse-image-Well-founded { List A } _<_ length public
>   wf : WF.Well-founded _⊰_
>   wf = ii-wf <-ℕ-wf

```

一点点支架代码实际上并没有“改变”证明，而是改变了命题。我们需要这个分区引理。

```
> s<s : ∀ {a b} → a < b → suc a < suc b
> s<s <-base = <-base
> s<s (<-step y) = <-step (s<s y)

```

显示分区列表不会增加其大小。

```
> module PartitionLemma { A } where
>   _≼_ : Rel (List A)
>   x ≼ y = length x < suc (length y) -- succ to let us reuse <

```

对于所有谓词和列表，每个分区的长度都小于或等于列表的原始长度。proj₁和 proj₂是 Haskell 中的 fst 和 snd。

```
>   partition-size : (p : A → Bool) (xs : List A)
>     → proj₁ (partition p xs) ≼ xs
>     × proj₂ (partition p xs) ≼ xs

```

虽然我们用≼表达了我们的命题，但我们仍然使用原始的<构造器。<-base 实际上意味着在这个上下文中是相等的！

```
>   partition-size p [] = <-base , <-base
>   partition-size p (x ∷ xs)
>     with p x | partition p xs | partition-size p xs
>   ... | true | as , bs | as-size , bs-size = s<s as-size , <-step bs-size
>   ... | false | as , bs | as-size , bs-size = <-step as-size , s<s bs-size

```

最后，快速排序。

```
> module Quick {A} (p : A → A → Bool) where

```

打开礼物（证明）。

```
>   open <-on-length-Well-founded
>   open PartitionLemma
>   quicksort' : (xs : List A) → WF.Acc _⊰_ xs → List A
>   quicksort' [] _ = []
>   quicksort' (x ∷ xs) (WF.acc g) ::

```

根据分区引理，我们得到了小于或等于 xs 和大于或等于 xs 的小结。通过使长度良基化，我们现在能够“粘合”间接性的层：x ∷ xs 最初严格较小且结构递归，而分区引理让我们能够告诉终止检查器小、大和 xs 本质上是相同的。

```
>     with partition (p x) xs | partition-size (p x) xs
>   ... | small , big | small-size , big-size = small' ++ [ x ] ++ big'
>     where
>       small' = quicksort' small (g small small-size)
>       big' = quicksort' big (g big big-size)
>   quicksort : List A → List A
>   quicksort xs = quicksort' xs (wf xs)

```
