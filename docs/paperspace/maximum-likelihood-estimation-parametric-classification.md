# 如何对参数分类方法使用最大似然估计

> 原文：<https://blog.paperspace.com/maximum-likelihood-estimation-parametric-classification/>

在之前讨论[贝叶斯法则如何工作](https://blog.paperspace.com/bayesian-decision-theory/)的一些教程中，决策是基于一些概率做出的(例如，可能性和先验)。这些概率要么是明确给出的，要么是根据一些给定的信息计算出来的。在本教程中，概率将根据训练数据进行估计。

本教程考虑参数分类方法，其中数据样本的分布遵循已知的分布(例如，高斯分布)。已知分布由一组参数定义。对于高斯分布，参数是均值$\mu$和方差$\sigma^2$.如果样本分布的参数被估计，那么样本的分布就可以形成。因此，我们可以对遵循相同分布的新实例进行预测。

给定一个新的未知数据样本，可以根据它是否遵循旧样本的分布来做出决定。如果它遵循旧的分布，则新样本被类似于旧样本地对待(例如，根据旧样本的类别分类)。

在本教程中，使用**最大似然估计(MLE)** 来估计参数。

教程的大纲如下:

*   估计样本分布的步骤
*   最大似然估计
*   二项分布
*   多项分布
*   高斯(正态)分布

让我们开始吧。

## 估计样本分布的步骤

基于贝叶斯法则，后验概率按下式计算:
$ $
P(C _ I | x)= \ frac { P(x | C _ I)P(C _ I)} { P(x)}
$ $
分母中的证据是归一化项，可以排除。因此，后验概率是根据以下等式计算的:
$ $
P(C _ I | x)= P(x | C _ I)P(C _ I)
$ $
要计算后验概率$P(C_i|x)$首先必须估计似然概率$P(x|C_i)$和先验概率$P(C_i)$。

在参数方法中，这些概率是基于遵循已知分布的概率函数计算的。本教程中讨论的分布是伯努利，多项式和高斯。

每种分布都有其参数。例如，高斯分布有两个参数:平均值$\mu$和方差$\sigma^2$.如果这些参数被估计，那么分布将被估计。因此，可以估计可能性和先验概率。基于这些估计的概率，计算后验概率，因此我们可以对新的未知样本进行预测。

以下是本教程中根据给定样本估计分布参数的步骤总结:

1.  第一步，声称样本遵循一定的分布。根据这个分布的公式，找出它的参数。
2.  使用最大似然估计(MLE)来估计分布的参数。
3.  估计的参数被插入到声称的分布中，这导致估计的样本分布。
4.  最后，估计样本的分布用于决策。

下一节讨论最大似然估计(MLE)的工作原理。

## 最大似然估计

MLE 是一种估计已知分布参数的方法。请注意，还有其他方法来进行估计，如贝叶斯估计。

首先，有两个假设需要考虑:

1.  第一个假设是有一个训练样本$\mathcal{X}={{x^t}_{t=1}^N}$，其中实例$x^t$是**独立的**和**同分布的** (iid)。
2.  第二个假设是实例$x^t$取自先前已知的**概率密度函数(pdf)**$ p(\ mathcal { x } | \ theta)$其中$\theta$是定义分布的参数集。换句话说，实例$x^t$遵循分布$p(\mathcal{x}|\theta)$，假定该分布由参数集$\theta$定义。

请注意，$p(\mathcal{x}|\theta)$表示实例`x`存在于由参数集$\theta$定义的分布中的概率。通过找到一组合适的参数$\theta$，我们可以对与$x^t$.实例遵循相同分布的新实例进行采样我们如何找到 find $\theta$？这就是 MLE 出现的原因。

根据维基百科:

> 对于任意一组**独立的**随机变量，其联合分布的概率密度函数是其密度函数的**乘积。**

因为样本是 **iid** ( *独立*和*同分布*)，所以样本$\mathcal{X}$遵循由参数$\theta$集合定义的分布的**可能性**等于各个实例$x^t$.的可能性的**乘积**

$ $
l(\ theta | \ mathcal { x })\ equiv p(\mathcal{x}|\theta)=\prod_{t=1}^n{p(x^t|\theta)}
$ $

目标是找到使似然估计$L(\theta|\mathcal{X})$最大化的参数集$\theta$。换句话说，找到一组参数$\theta$使从$\theta$定义的分布中获取$x^t$样本的机会最大化。这被称为**最大似然估计(MLE)** 。这是制定如下:

$ $
\theta^* \ space arg \ space max _ \ theta \ space l {(\ theta | \ mathcal { x })}
$ $

可能性$L(\theta|\mathcal{X})$的表示可以简化。目前，它计算的产品之间的可能性个别样本$p(x^t|\theta)$.**对数似然**不是计算似然性，而是简化计算，因为它将乘积转换为**总和**。

$ $
\ mathcal { l } {(\ theta | \ mathcal { x })} \ equiv log \ space l(\ theta | \ mathcal { x } | \ theta)= log \ space \prod_{t=1}^n{p(x^t|\theta)} \
\ mathcal { l } {(\ theta | \ mathcal { x })} \ equiv log \ space l(\ theta | \ mathcal { x })\ equiv log \ space l(\ theta | \ mathcal { x })\ equiv log \ space p(\mathcal{x}|\theta)=\sum_{t=1}^n{log \ space p(x^t|\theta)}
$ $

最大似然估计的目标是找到使对数似然最大化的参数集。这是制定如下:

$ $
\theta^* \ space arg \ space max _ \ theta \ space \ mathcal { l } {(\ theta | \ mathcal { x })}
$ $

例如，在高斯分布中，参数组＄\θ＄只是均值和方差$\theta={{\mu,\sigma^2}}$.这组参数$\theta$有助于选择接近原始样本$\mathcal{X}$的新样本。

前面的讨论准备了一个估计参数集的通用公式。接下来将讨论这如何适用于以下发行版:

1.  伯努利分布
2.  多项分布
3.  高斯(正态)分布

每个发行版要遵循的步骤是:

1.  **概率函数**:求做出预测的概率函数。
2.  **似然**:根据概率函数，推导出分布的似然。
3.  **对数似然**:基于似然，导出对数似然。
4.  **最大似然估计**:求构成分布的参数的最大似然估计。
5.  **估计分布**:将估计的参数代入分布的概率函数。

## 二项分布

**伯努利分布**适用于二进制结果 1 和 0。它假设结果 1 以概率$p$出现。因为 2 个结果的概率必须等于 1 美元，所以结果 0 发生的概率是 1-p 美元。

$$
(p)+(1-p)=1
$$

假设$x$是伯努利随机变量，可能的结果是 0 和 1。

使用伯努利分布可以解决的问题是扔硬币，因为只有两种结果。

### 概率函数

伯努利分布的数学公式如下:

$$
p(x)=p^x(1-p)^{1-x}，\space where \space x={0，1}
$$

根据上面的等式，只有一个参数，即$p$。为了导出数据样本$x$的伯努利分布，必须估计参数$p$的值。

### 可能性

还记得下面给出的一般似然估计公式吗？

$ $
l(\theta|\mathcal{x})=\prod_{t=1}^n{p(x^t|\theta)}
$ $

对于伯努利分布，只有一个参数$p$。因此，$\theta$应该替换为$p$。因此，概率函数看起来像这样，其中$p_0$是参数:

$ $
p(x^t|\theta)=p(x^t|p_0)
$ $

基于该概率函数，伯努利分布的可能性为:

$ $
l(p_0|\mathcal{x})=\prod_{t=1}^n{p(x^t|p_0)}
$ $

概率函数可以分解如下:

$ $
p(x^t|p_0)=p_0^{x^t}(1-p_0)^{1-x^t}
$ $

因此，可能性如下:

$ $
l(p_0|\mathcal{x})=\prod_{t=1}^n{p_0^{x^t}(1-p_0)^{1-x^t}}
$ $

### 对数似然

推导出概率分布的公式后，接下来是计算对数似然性。这是通过将$log$引入到前面的等式中来实现的。

$ $
\ mathcal { l }(p _ 0 | \ mathcal { x })\ equiv log \ space l(p _ 0 | \ mathcal { x })= log \ space \prod_{t=1}^n{p_0^{x^t}(1-p_0)^{1-x^t}}
$ $

当引入$log$时，乘法被转换成求和。

由于$log$运算符，$p_0^{x^t}$和$(1-p_0)^{1-x^t}$之间的乘法被转换为求和，如下所示:

$ $
\ mathcal { l }(p _ 0 | \ mathcal { x })\ equiv log \ space l(p _ 0 | \ mathcal { x })= \sum_{t=1}^n{log \ space p_0^{x^t}}+\sum_{t=1}^n{log \ space(1-p_0)^{1-x^t}}
$ $

使用对数幂规则，对数似然性为:

$ $
\ mathcal { l }(p _ 0 | \ mathcal { x })\ equiv log \ space p_0\sum_{t=1}^n{x^t}+log \ space(1-p _ 0)\sum_{t=1}^n{({1-x^t})}
$ $

最后一个求和项可以简化如下:

$ $
\sum_{t=1}^n{({1-x^t})}=\sum_{t=1}^n{1}-\sum_{t=1}^n{x^t}=n-\sum_{t=1}^n{x^t}
$ $

回到对数似然函数，这是它的最后一种形式:

$ $
\mathcal{l}(p_0|\mathcal{x})=log(p_0)\sum_{t=1}^n{x^t}+圆木(1-P0)(n-\sum_{t=1}^n{x^t})
$ $

导出对数似然后，接下来我们将考虑**最大似然估计**。我们如何找到前一个方程的最大值？

### 最大似然估计

当一个函数的导数等于 0 时，这意味着它有一个特殊的行为；不增不减。这种特殊的行为可能被称为函数的最大值点。因此，通过将它相对于$p_0$的导数设置为 0，可以得到先前对数似然的最大值。

$ $
\ frac { d \ space \ mathcal { L }(p _ 0 | \ mathcal { X })} { d \ space p _ 0 } = 0
$ $

记住$log(x)$的导数计算如下:

$ $
\ frac { d \ space log(x)} { dx } = \ frac { 1 } { x ln(10)}
$ $

对于前面的对数似然方程，下面是它的导数:

$ $
\ frac { d \ space \ mathcal { l }(p _ 0 | \ mathcal { x })} { d \ space p_0}=\frac{\sum_{t=1}^n{x^t}}{p_0 ln(10)}-\frac{(n-\sum_{t=1}^n{x^t})}{(1-p_0)ln(10)} = 0
$ $

注意$log(p_0) log(1-p_0) ln(10)$可以作为统一分母。结果，导数变成如下所示:

$ $
\ frac { d \ space \ mathcal { l }(p _ 0 | \ mathcal { x })} { d \ space p_0}=\frac{(1-p_0)\sum_{t=1}^n{x^t}-p_0(n-\sum_{t=1}^n{x^t})}{p_0(1-p _ 0)ln(10)} = 0
$ $

因为导数等于 0，所以不需要分母。导数现在如下:

$ $
\ frac { d \ space \ mathcal { l }(p _ 0 | \ mathcal { x })} { d \ space p_0}=(1-p_0)\sum_{t=1}^n{x^t}-p_0(n-\sum_{t=1}^n{x^t})=0
$ $

经过一些简化后，结果如下:

$ $
\ frac { d \ space \ mathcal { l }(p _ 0 | \ mathcal { x })} { d \ space p_0}=\sum_{t=1}^n{x^t}-p_0\sum_{t=1}^n{x^t}-p_0n+p_0\sum_{t=1}^n{x^t}=0
$ $

提名者中的下两个术语相互抵消:

$ $
-p_0\sum_{t=1}^n{x^t}+p_0\sum_{t=1}^n{x^t}
$ $

因此，导数变成:

$ $
\ frac { d \ space \ mathcal { l }(p _ 0 | \ mathcal { x })} { d \ space p_0}=\sum_{t=1}^n{x^t}-p_0n=0
$ $

负项可以移到另一边变成:

$ $
\ frac { d \ space \ mathcal { l }(p _ 0 | \ mathcal { x })} { d \ space p_0}=\sum_{t=1}^n{x^t}=p_0n
$ $

将两边除以$N$，计算参数$p_0$的等式为:

$ $
p_0=\frac{\sum_{t=1}^n{x^t}}{n}
$ $

简单地说，参数$p_0$计算为所有样本的平均值。因此，伯努利分布的估计参数$p$是$p_0$。

请记住$x^t \in {0，1}$，这意味着所有样本的总和就是具有$x^t=1$.的样本数因此，如果有 10 个样本，其中有 6 个，那么$p_0=0.6$。通过最大化似然性(或对数似然性)，将得到代表数据的最佳伯努利分布。

### 估计分布

记住，伯努利分布的概率函数是:

$$
p(x)=p^x(1-p)^{1-x}，\space where \space x={0，1}
$$

一旦伯努利分布的参数＄p＄被估计为＄p _ 0 ＄,它就被插入到伯努利分布的通用公式中，以返回样本$\mathcal{X}=x^t$:的**估计分布**

$ $
p(x^t)=p_0^{x^t}(1-p_0)^{1-x^t}，\太空何处\太空 x^t={0,1}
$$

可以使用样本$\mathcal{X}=x^t$.的估计分布进行预测

## 多项分布

伯努利分布只适用于两种结果/状态。为了处理两个以上的结果，使用了**多项分布**，其中结果是互斥的，因此没有一个会影响另一个。多项式分布是伯努利分布的**推广**。

一个可以分布为多项式分布的问题是掷骰子。有两个以上的结果，其中每个结果都是相互独立的。

结果的概率是$p_i$，其中$i$是结果的指数(即类)。因为所有结果的概率之和必须为 1，所以适用以下公式:

$ $
\sum_{i=1}^n{p_i}=1
$ $

对于每个结果$i$来说，都有一个指标变量$x_i$。所有变量的集合是:

$ $
\mathcal{x}=\{x_i\}_{i=1}^k
$ $

变量$x_i$可以是 1 也可以是 0。如果结果是$i$则为 1，否则为 0。
记住每个实验只有一个结果**$ t $**。因此，对于所有的类$i，i=1:K$,所有变量$x^t$的总和必须为 1。

$ $
\sum_{i=1}^k{x_i^t}=1
$ $

### 概率函数

概率函数可以表述如下，其中$K$是结果的数量。它是所有结果的所有概率的乘积。

$ $
p(x1，x2，x3，...x_k)=\prod_{i=1}^k{p_i^{x_i}}
$ $

注意所有$p_i$的总和是 1。

$ $
\sum_{i=1}^k{p_i}=1
$ $

### 可能性

一般似然估计公式如下:

$ $
l(\ theta | \ mathcal { x })\ equiv p(x | \ theta)=\prod_{t=1}^n{p(x^t|\theta)}
$ $

对于多项分布，这里是它的可能性，其中$K$是结果的数量，$N$是样本的数量。

$ $
l(p _ I | \ mathcal { x })\ equiv p(x|\theta)=\prod_{t=1}^n\prod_{i=1}^k{p_i^{x_i^t}}
$ $

### 对数似然

多项式分布的对数似然如下:

$ $
\ mathcal { l }(p _ I | \ mathcal { x })\ equiv log \ space l(p _ I | \ mathcal { x })\ equiv log \ space p(x | \ theta)= log \ space \prod_{t=1}^n\prod_{i=1}^k{p_i^{x_i^t}}
$ $

$log$将乘积转换成总和:

$ $
\mathcal{l}(p_i|\mathcal{x})=\sum_{t=1}^n\sum_{i=1}^k{log \太空 p_i^{x_i^t}}
$$

根据对数幂规则，对数似然性为:

$ $
\mathcal{l}(p_i|\mathcal{x})=\sum_{t=1}^n\sum_{i=1}^k{[{x_i^t} \空间日志\空间 p_i}]
$$

注意，所有类的所有$x$之和等于 1。换句话说，以下成立:

$ $
\sum_{i=1}^kx_i^t=1
$ $

然后，对数似然变成:

$ $
\mathcal{l}(p_i|\mathcal{x})=\sum_{t=1}^n{x_i^t}\sum_{i=1}^k{log \太空 p_i}
$$

下一节使用 MLE 来估计参数$p_i$。

### 最大似然估计

基于多项式分布的对数似然性$\mathcal{L}(p_i|\mathcal{X})$，通过根据下式将对数似然性的导数设置为 0 来估计参数$p_i$。

$ $
\ frac { d \ space \ mathcal { l }(p _ I | \ mathcal { x })} { d \ space p _ I } = \ frac { d \ space \sum_{t=1}^n{x_i^t}\sum_{i=1}^k{log \ space p _ I } } { d \ space p _ I } = 0
$ $

根据导数乘积规则，$\sum_{t=1}^N{x_i^t}$和$\sum_{i=1}^K{log \space p_i}$项乘积的导数计算如下:

$ $
\ frac { d \ space \sum_{t=1}^n{x_i^t}\sum_{i=1}^k{log \ space p _ I } } { d \ space p_i}=\sum_{i=1}^k{log \ space p _ I }。\sum_{t=1}^n{x_i^t}}{d+\sum_{t=1}^n{x_i^t}.\ frac { d \ space \sum_{i=1}^k{log \ space p _ I } } { d \ space p _ I }
$ $

$log(p_i)$的导数是:

$ $
\ frac { d \ space log(p _ I)} { DP _ I } = \ frac { 1 } { p _ I ln(10)}
$ $

对数似然的导数为:

$ $
\ frac { d \ space \sum_{t=1}^n{x_i^t}\sum_{i=1}^k{log \ space p _ I } } { d \ space p _ I } = \frac{\sum_{t=1}^n{x_i^t}}{p_i ln(10)}
$ $

基于[拉格朗日乘数](https://ocw.mit.edu/ans7870/18/18.443/s15/projects/Rproject3_rmd_multinomial_theory.html)并将对数似然的导数设置为零，多项式分布的 MLE 为:

$ $
p_i=\frac{\sum_{t=1}^n{x_i^t}}{n}
$ $

注意，多项式分布只是伯努利分布的推广。他们的 mle 是相似的，除了多项式分布认为有多个结果，而伯努利分布只有两个结果。

计算每个结果的最大似然。它计算结果$i$在结果总数中出现的次数。例如，如果骰子上编号为 2 的面在总共 20 次投掷中出现了 10 次，则它的 MLE 为 10 / 20 = 0.5 美元。

多项式实验可以看作是做$K$伯努利实验。对于每个实验，计算单个类$i$的概率。

### 估计分布

一旦估计了多项式分布的参数＄p _ I ＄,就将其插入到多项式分布的概率函数中，以返回样本$\mathcal{X}=x^t$.的估计分布

$$
p(x_1，x_2，x_3，...x_k)=\prod_{i=1}^k{p_i^{x_i}}
$ $

## 高斯(正态)分布

伯努利分布和多项式分布都将其输入设置为 0 或 1。输入$x$不可能是任何实数。

$$
x^t \in {0，1 } \
t \ in 1:n \
n:\空间编号\空间样本的空间。
$$

在高斯分布中，输入$x$的值介于$-\infty$到$\infty$之间。

$ $
-\ infty<x<\ infty
$ $

### 概率函数

高斯(正态)分布是基于两个参数定义的:均值$\mu$和方差$\sigma^2$.

$$
p(x)=\mathcal{N}(\mu，\sigma^2)
$$

给定这两个参数，下面是高斯分布的概率密度函数。

$ $
\ mathcal { n }(\穆，\sigma^2)=p(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp[-\frac{(x-\mu)^2}{2\sigma^2}]=\frac{1}{\sqrt{2\pi}\sigma}e^{[-\frac{(x-\mu)^2}{2\sigma^2}]} \ $ $

$ $
-\ infty<x<\ infty
$ $

如果一个随机变量$X$的密度函数是根据前面的函数计算的，则称它遵循高斯(正态)分布。在这种情况下，其均值为$E[X]\equiv \mu$且方差为$VAR(X) \equiv \sigma^2$.这表示为$\mathcal{N}(\mu，\sigma^2)$.

### 可能性

让我们先回顾一下计算似然估计的等式。

$ $
l(\theta|\mathcal{x})=\prod_{t=1}^n{p(x^t|\theta)} \太空何处\太空\mathcal{x}=\{x^t\}_{t=1}^n
$ $

对于高斯概率函数，下面是计算可能性的方法。

$ $
l(\mu,\sigma^2|\mathcal{x})\ equiv \prod_{t=1}^n{\mathcal{n}(\mu，\sigma^2)}=\prod_{t=1}^n{\frac{1}{\sqrt{2\pi}\sigma}\exp[-\frac{(x^t-\mu)^2}{2\sigma^2}]}
$ $

### 对数似然

对数被引入高斯分布的似然性，如下所示:

$ $
\mathcal{l}(\mu,\sigma^2|\mathcal{x})\ equiv 日志\太空 l(\mu,\sigma^2|\mathcal{x})\ equiv log\prod_{t=1}^n{\mathcal{n}(\mu，\sigma^2)}=log\prod_{t=1}^n{\frac{1}{\sqrt{2\pi}\sigma}\exp[-\frac{(x^t-\mu)^2}{2\sigma^2}]}
$ $

对数将乘积转换为总和，如下所示:

$ $
\mathcal{l}(\mu,\sigma^2|\mathcal{x})\ equiv log \ space l(\mu,\sigma^2|\mathcal{x})\ equiv \sum_{t=1}^n{log \ space \ mathcal { n }(\ mu，\sigma^2)}=\sum_{t=1}^n{log \ space(\frac{1}{\sqrt{2\pi}\sigma}\exp[-\frac{(x^t-\mu)^2}{2\sigma^2}])}
$ $

使用**对数乘积规则**，对数似然为:

$ $
\mathcal{l}(\mu,\sigma^2|\mathcal{x})\ equiv log \ space l(\mu,\sigma^2|\mathcal{x})\ equiv \sum_{t=1}^n{log \ space \ mathcal { n }(\ mu，\sigma^2)}=\sum_{t=1}^n{(log \ space(\ frac { 1 } { 2 \ pi } \ sigma })+log \ space(\exp[-\frac{(x^t-\mu)^2}{2\sigma^2}]))}
$ $

求和运算符可以分布在两项中:

$ $
\mathcal{l}(\mu,\sigma^2|\mathcal{x})=\sum_{t=1}^n{log \太空公司{ 1 }+\sum_{t=1}^nlog \太空公司\exp[-\frac{(x^t-\mu)^2}{2\sigma^2}]}
$ $

这个等式有两个独立的项。现在让我们分别研究每一项，然后再把结果结合起来。

对于第一项，可以应用**对数商规则**。结果如下:

$ $
\sum_{t=1}^nlog \太空(\frac{1}{\sqrt{2\pi}\sigma})=\sum_{t=1}^n(log(1)-log(\sqrt{2\pi}\sigma))
$ $

假设$log(1)=0$，则结果如下:

$ $
\sum_{t=1}^nlog \太空(\frac{1}{\sqrt{2\pi}\sigma})=-\sum_{t=1}^nlog(\sqrt{2\pi}\sigma)
$ $

根据**对数乘积规则**，第一项的对数为:

$ $
\sum_{t=1}^nlog \太空(\frac{1}{\sqrt{2\pi}\sigma})=-\sum_{t=1}^n[log{\sqrt{2\pi}+log \太空\西格玛}]
$$

注意，第一项不依赖于求和变量$t$，因此它是一个固定项。因此，求和的结果就是将这一项乘以$N$。

$ $
\sum_{t=1}^nlog \ space(\ frac { 1 } { \ sqrt { 2 \ pi } \ sigma })=-\ frac { n } { 2 } log({ \ sqrt { 2 \ pi } })-n \ space log \ space \ sigma
$ $

现在让我们转到下面给出的第二项。

$ $
\sum_{t=1}^nlog \太空(\exp[-\frac{(x^t-\mu)^2}{2\sigma^2})
$ $

**对数幂规则**可用于简化该术语，如下所示:

$ $
\sum_{t=1}^nlog \太空(\exp[-\frac{(x^t-\mu)^2}{2\sigma^2}])=\sum_{t=1}^nlog \太空 e^{[-\frac{(x^t-\mu)^2}{2\sigma^2}]}=\sum_{t=1}^n[-\frac{(x^t-\mu)^2}{2\sigma^2}]\太空日志(e)
$$

假设$log$ base 是$e$，则$log(e)=1$。因此，第二项现在是:

$ $
\sum_{t=1}^nlog \太空(\exp[-\frac{(x^t-\mu)^2}{2\sigma^2}])=-\sum_{t=1}^n\frac{(x^t-\mu)^2}{2\sigma^2}
$ $

分母不依赖于求和变量$t$，因此等式可以写成如下形式:

$ $
\sum_{t=1}^nlog \太空(\exp[-\frac{(x^t-\mu)^2}{2\sigma^2}])=-\frac{1}{2\sigma^2}\sum_{t=1}^n(x^t-\mu)^2
$ $

简化这两项后，下面是高斯分布的对数似然性:

$ $
\mathcal{l}(\mu,\sigma^2|\mathcal{x})=-\frac{n}{2}log({\sqrt{2\pi}})-n \太空日志\太空\sigma-\frac{1}{2\sigma^2}\sum_{t=1}^n(x^t-\mu)^2
$ $

### 最大似然估计

本节讨论如何找到高斯分布中两个参数＄\ mu＄和$\sigma^2$.的最大似然估计

最大似然估计可以通过计算每个参数的对数似然导数得到。通过将该导数设置为 0，可以计算 MLE。下一小节从第一个参数$\mu$开始。

#### 平均值的最大似然估计值

从$\mu$开始，让我们计算对数似然的导数:

$ $
\ frac { d \ space \mathcal{l}(\mu,\sigma^2|\mathcal{x})}{d \ space \ mu } = \ frac { d } { d \ mu }[-\ frac { n } { 2 } log({ \ sqrt { 2 \ pi })-n \ space log \ space \sigma-\frac{1}{2\sigma^2}\sum_{t=1}^n(x^t-\mu)^2]=0
$ $

前两项不依赖于$\mu$，因此它们的导数为 0。

$ $
\ frac { d \ space \mathcal{l}(\mu,\sigma^2|\mathcal{x})}{d \ mu } = { \ frac { d } { d \mu}\sum_{t=1}^n(x^t-\mu)^2}=0
$ $

前一项可以写成如下:

$ $
\ frac { d \ space \mathcal{l}(\mu,\sigma^2|\mathcal{x})}{d \ mu } = { \ frac { d } { d \mu}\sum_{t=1}^n((x^t)^2-2x^t\mu+\mu^2)}=0
$ $

因为$(x^t)^2$不依赖于$\mu$，所以它的导数是 0，可以忽略不计。

$ $
\ frac { d \ space \mathcal{l}(\mu,\sigma^2|\mathcal{x})}{d \ mu } = { \ frac { d } { d \mu}\sum_{t=1}^n(-2x^t\mu+\mu^2)}=0
$ $

求和可以分布在其余两项上:

$ $
\ frac { d \ space \mathcal{l}(\mu,\sigma^2|\mathcal{x})}{d \ mu } = { \ frac { d } { d \mu}[-\sum_{t=1}^n2x^t\mu+\sum_{t=1}^n\mu^2}]=0
$ $

对数似然的导数变成:

$ $
\ frac { d \ space \mathcal{l}(\mu,\sigma^2|\mathcal{x})}{d \mu}=-\sum_{t=1}^n2x^t+2\sum_{t=1}^n\mu=0
$ $

第二项$2\sum_{t=1}^N\mu$不依赖于$t$，因此它是一个固定项，等于$2N\mu$。因此，对数似然的导数如下:

$ $
\ frac { d \ space \mathcal{l}(\mu,\sigma^2|\mathcal{x})}{d \mu}=-\sum_{t=1}^n2x^t+2n\mu=0
$ $

通过求解前面的方程，最终，均值的 MLE 为:

$ $
m=\frac{\sum_{t=1}^nx^t}{n}
$ $

#### $\sigma^2$方差极大似然估计

与计算均值的 MLE 的步骤类似，方差的 MLE 为:

$ $
s^2=\frac{\sum_{t=1}^n(x^t-m)^2}{n}
$ $

## 结论

本教程介绍了最大似然估计(MLE)方法的数学原理，该方法基于训练数据$x^t$.来估计已知分布的参数讨论的三种分布是伯努利分布、多项式分布和高斯分布。

本教程总结了 MLE 用于估计参数的步骤:

1.  要求训练数据的分布。
2.  使用**对数似然**估计分布的参数。
3.  将估计的参数代入分布的概率函数。
4.  最后，估计训练数据的分布。

一旦**对数似然**被计算，它的导数相对于分布中的每个参数被计算。估计参数是使对数似然最大化的参数，对数似然是通过将对数似然导数设置为 0 而得到的。

本教程讨论了 MLE 如何解决分类问题。在后面的教程中，MLE 将用于估计回归问题的参数。敬请关注。