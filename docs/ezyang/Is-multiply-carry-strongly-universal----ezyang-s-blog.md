<!--yml

category: 未分类

date: 2024-07-01 18:18:04

-->

# *multiply-carry*强通用吗？ : ezyang’s blog

> 来源：[`blog.ezyang.com/2010/11/is-multiply-carry-strongly-universal/`](http://blog.ezyang.com/2010/11/is-multiply-carry-strongly-universal/)

我一直想要实现一个[count-min sketch](http://www.eecs.harvard.edu/~michaelm/CS222/countmin.pdf)；这个结构比布隆过滤器稍微不那么广为人知，它是一种相关的*sketch*数据结构（即，一种概率数据结构，用于近似回答某些查询），但它看起来是一个相当实用的结构，并且已经在[一些有趣的方式](http://research.microsoft.com/pubs/132859/popularityISeverything.pdf)中被使用。

可惜的是，当你想要实现一个不到十年前提出的数据结构，而且还没有进入教科书时，会遇到很多理论上的模糊点。在这个特定情况下，理论上的模糊点是选择*通用哈希族*。因为我还没有修过研究生级别的算法课程，所以必须去查书。

通过我对课程笔记、论文和教科书的调查，我注意到两件事情。

首先，通用哈希族可能具有许多不同的独立性保证，每一种保证可能会有许多不同的名称。假设我们的哈希族`H`是由函数`h：M → N`组成，其中`M = {0, 1, ..., m-1}`，`N = {0, 1, ..., n-1}`，且`m >= n`。这里，M 对应我们的“全集”，即将被哈希的可能值，而 N 则是哈希函数的范围。

+   *弱通用哈希族*，也称为*弱 2-universal 哈希族*，有时候会简称为*weak*，是一种哈希族，对于从`H`中随机选择的哈希函数`h`：

    ```
    ∀ x,y ∈ M, x != y. Pr[h(x) = h(y)] ≤ 1/n

    ```

+   *强 2-universal 哈希族*，也称为*(强) 2-independent 通用哈希族*，有时候会简称为*2-universal*，是满足以下条件的哈希族：

    ```
    ∀ x,y ∈ M, a,b ∈ N.
             Pr[h(x) = a ∧ h(y) = b] ≤ 1/n²

    ```

+   *(强) k-independent 通用哈希族*将上述概念推广到以下条件：

    ```
    ∀ x₁,x₂...x_k ∈ M, a₁,a₂...a_k ∈ N.
             Pr[h(x₁) = a₁ ∧ h(x₂) = a₂ ∧ ...] ≤ 1/n^k

    ```

其次，“弱”通常在“弱哈希函数”中省略，是因为 2-universal 哈希族往往也是 2-independent。《随机化算法》指出：“大多数已知的 2-universal 哈希族的构造实际上产生了一个强 2-universal 哈希族。因此，这两个定义通常没有区别。” 并要求学生证明，如果`n = m = p`是一个素数，那么卡特和韦格曼的哈希族是强 2-universal 的。（我马上会说明这是什么。） 因此，[维基百科](http://en.wikipedia.org/wiki/Universal_hashing) 愉快地采纳了弱标准，并在最后一节中简要提到了 2-independence。（我没有编辑文章，因为我不确定是否需要做任何更改。）

那么，卡特和韦格曼的通用哈希族是什么？非常简单：

鉴于 *p ≥ m* 是质数且 ![a,b \in {0, 1, \cdots, p-1}](img/cdots, p-1}")。除此之外，呃，实际上没有人在实践中使用模数。这里有一个来自 [Cormode 的实现](http://www.cs.rutgers.edu/~muthu/massdal-code-index.html) 的例子：

```
#define MOD 2147483647
#define HL 31
long hash31(long long a, long long b, long long x)
{

  long long result;
  long lresult;

  // return a hash of x using a and b mod (2³¹ - 1)
// may need to do another mod afterwards, or drop high bits
// depending on d, number of bad guys
// 2³¹ - 1 = 2147483647

  result=(a * x) + b;
  result = ((result >> HL) + result) & MOD;
  lresult=(long) result;

  return(lresult);
}

```

这个实现显然是正确的：

1.  乘法和加法不能使 `long long` 结果溢出，并且

1.  第二行利用了我们利用 Mersenne 质数进行快速取模的能力，结合了几种替代的位运算。当然，为了做到这一点，我们需要非常小心地选择质数。嗯，神奇的数字。

好的，那很好。有一个小小的疏忽，我们没有明确确保 `n = m = p`，所以我不能百分之百确定我们保留了强一般化。但我还没有做完 *Randomized Algorithms* 练习，所以我不知道这个属性在实践中有多重要。

顺便说一下，[这个函数](http://www2.research.att.com/~marioh/sketches/index.html)也声称是这种非常通用的哈希，但我很难相信它：

```
Tools::UniversalHash::value_type Tools::UniversalHash::hash(
        UniversalHash::value_type x
) const
{
        uint64_t r = m_a[0];
        uint64_t xd = 1;

        for (uint16_t i = 1; i < m_k; i++)
        {
                xd = (xd * x) % m_P;
                r += (m_a[i] * xd) % m_P;
                        // FIXME: multiplications here might overflow.
        }

        r = (r % m_P) & 0xFFFFFFFF;
                // this is the same as x % 2³²\. The modulo operation with powers
                // of 2 (2^n) is a simple bitwise AND with 2^n - 1.

        return static_cast<value_type>(r);
}

```

现在我们把注意力转向 multiply-carry，维基百科声称它是 *目前已知整数最快的通用哈希族*。它设计成在计算机上易于实现：`(unsigned) (a*x) >> (w-M)`（其中 `a` 是奇数）就是你所需要的全部。嗯，准确地说，它是目前已知的最快 *2-一般化* 哈希族：相关论文仅就弱一般化给出了证明，详见 [相关论文](http://www.diku.dk/~jyrki/Paper/CP-11.4.1997.pdf)。

所以，我的问题是：*multiply-carry 是否强一般化*？Motwani 和 Raghavan 暗示它可能是，但我找不到证明。

*Postscript.* 幸运的是，对于 count-min-sketch，我们实际上并不需要强一般化。我向 Graham Cormode 确认过，他们在论文中只使用了 2-一般化。但原始问题仍然存在……在严格的理论基础上，无论如何。

*Non sequitur.* 这里有一个有趣的组合器，用于组合在折叠中使用的函数：

```
f1 <&> f2 = \(r1, r2) a -> (f1 r1 a, f2 r2 a)

```

它允许你将两个组合函数捆绑在一起，这样你可以一次性将它们应用到列表中：

```
(foldl xs f1 z1, foldl xs f2 z2) == foldl xs (f1 <&> f2) (z1, z2)

```

翻转组合器可以使其适用于右折叠。这使我们得到了 `average` 函数的以下可爱实现：

```
average = uncurry (/) . foldl' ((+) <&> (flip (const (+1)))) (0,0)

```

或许我们可以写一条重写规则来为我们做这件事。
