# 一些神经切向核背后的数学知识

> 原文：[`lilianweng.github.io/posts/2022-09-08-ntk/`](https://lilianweng.github.io/posts/2022-09-08-ntk/)

众所周知，神经网络是过度参数化的，通常可以轻松拟合具有接近零训练损失的数据，并在测试数据集上具有良好的泛化性能。尽管所有这些参数都是随机初始化的，但优化过程可以始终导致类似的良好结果。即使模型参数的数量超过训练数据点的数量，这也是正确的。

**神经切向核（NTK）**（[Jacot et al. 2018](https://arxiv.org/abs/1806.07572)）是一个核，用于通过梯度下降解释神经网络在训练过程中的演变。它深入探讨了为什么具有足够宽度的神经网络在被训练以最小化经验损失时可以始终收敛到全局最小值。在本文中，我们将深入探讨 NTK 的动机和定义，以及在不同初始化条件下对具有无限宽度的神经网络的确定性收敛的证明，通过在这种设置中对 NTK 进行表征。

> 🤓 与我之前的文章不同，这篇主要关注少量核心论文，而不是广泛涵盖该领域的文献综述。NTK 之后有许多有趣的工作，对理解神经网络学习动态进行了修改或扩展，但它们不会在这里涵盖。目标是以清晰易懂的格式展示 NTK 背后的所有数学知识，因此本文具有相当高的数学密度。如果您发现任何错误，请告诉我，我将很乐意快速更正。提前感谢！

# 基础知识

本节包含对几个非常基本概念的回顾，这些概念是理解神经切向核的核心。随意跳过。

## 向量对向量的导数

给定输入向量 $\mathbf{x} \in \mathbb{R}^n$（作为列向量）和函数 $f: \mathbb{R}^n \to \mathbb{R}^m$，关于 $\mathbf{x}$ 的导数是一个 $m\times n$ 矩阵，也称为[*雅可比矩阵*](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)：

$$ J = \frac{\partial f}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \dots &\frac{\partial f_1}{\partial x_n} \\ \vdots & & \\ \frac{\partial f_m}{\partial x_1} & \dots &\frac{\partial f_m}{\partial x_n} \\ \end{bmatrix} \in \mathbb{R}^{m \times n} $$

在本文中，我使用整数下标来指代向量或矩阵值中的单个条目；即 $x_i$ 表示向量 $\mathbf{x}$ 中的第 $i$ 个值，$f_i(.)$ 是函数输出中的第 $i$ 个条目。

对于向量关于向量的梯度定义为 $\nabla_\mathbf{x} f = J^\top \in \mathbb{R}^{n \times m}$，当 $m=1$（即，标量输出）时，这种形式也是有效的。

## 微分方程

微分方程描述一个或多个函数及其导数之间的关系。有两种主要类型的微分方程。

+   (1) *ODE（常微分方程）*只包含一个未知函数的一个随机变量。ODEs 是本文中使用的微分方程的主要形式。ODE 的一般形式如$(x, y, \frac{dy}{dx}, \dots, \frac{d^ny}{dx^n}) = 0$。

+   (2) *PDE（偏微分方程）*包含未知的多变量函数及其偏导数。

让我们回顾一下微分方程及其解的最简单情况。*变量分离*（傅立叶方法）可用于当所有包含一个变量的项都移到一边时，而其他项都移到另一边。例如，

$$ \begin{aligned} \text{给定}a\text{是一个常数标量：}\quad\frac{dy}{dx} &= ay \\ \text{将相同变量移到同一侧：}\quad\frac{dy}{y} &= adx \\ \text{两侧加上积分：}\quad\int \frac{dy}{y} &= \int adx \\ \ln (y) &= ax + C' \\ \text{最终}\quad y &= e^{ax + C'} = C e^{ax} \end{aligned} $$

## 中心极限定理

给定一组独立同分布的随机变量，$x_1, \dots, x_N$，均值为$\mu$，方差为$\sigma²$，*中心极限定理（CTL）*表明当$N$变得非常大时，期望值将呈高斯分布。

$$ \bar{x} = \frac{1}{N}\sum_{i=1}^N x_i \sim \mathcal{N}(\mu, \frac{\sigma²}{n})\quad\text{当}N \to \infty $$

CTL 也可以应用于多维向量，然后我们需要计算随机变量$\Sigma$的协方差矩阵，而不是单一尺度$\sigma²$。

## 泰勒展开

[*泰勒展开*](https://en.wikipedia.org/wiki/Taylor_series)是将一个函数表示为无限项的组成部分之和，每个部分都用该函数的导数表示。函数$f(x)$在$x=a$处的泰勒展开可以写成：$$ f(x) = f(a) + \sum_{k=1}^\infty \frac{1}{k!} (x - a)^k\nabla^k_xf(x)\vert_{x=a} $$其中$\nabla^k$表示第$k$阶导数。

第一阶泰勒展开通常用作函数值的线性近似：

$$ f(x) \approx f(a) + (x - a)\nabla_x f(x)\vert_{x=a} $$

## 核函数和核方法

一个[*核函数*](https://en.wikipedia.org/wiki/Kernel_method)本质上是两个数据点之间的相似性函数，$K: \mathcal{X} \times \mathcal{X} \to \mathbb{R$。它描述了对一个数据样本的预测对另一个数据样本的预测的敏感程度；或者换句话说，两个数据点有多相似。核函数应该是对称的，$K(x, x’) = K(x’, x)$。

根据问题结构，一些核函数可以分解为两个特征映射，一个对应一个数据点，核值是这两个特征的内积：$K(x, x’) = \langle \varphi(x), \varphi(x’) \rangle$。

*核方法*是一种非参数、基于实例的机器学习算法。假设我们已知所有训练样本$\{x^{(i)}, y^{(i)}\}$的标签，那么新输入$x$的标签通过加权和$\sum_{i} K(x^{(i)}, x)y^{(i)}$来预测。

## 高斯过程

*高斯过程（GP）*是一种通过对一组随机变量建模多元高斯概率分布的非参数方法。GP 假设函数的先验，然后根据观察到的数据点更新函数的后验。

给定数据点集合$\{x^{(1)}, \dots, x^{(N)}\}$，高斯过程假设它们遵循一个联合多元高斯分布，由均值$\mu(x)$和协方差矩阵$\Sigma(x)$定义。协方差矩阵$\Sigma(x)$中位置$(i,j)$处的每个条目由一个核$\Sigma_{i,j} = K(x^{(i)}, x^{(j)})$定义，也称为*协方差函数*。核心思想是 - 如果两个数据点被核视为相似，那么函数输出也应该接近。使用高斯过程对未知数据点进行预测等同于从该分布中抽取样本，通过给定观察到的数据点的未知数据点的条件分布。

查看[这篇文章](https://distill.pub/2019/visual-exploration-gaussian-processes/)，了解高质量且高度可视化的关于高斯过程的教程。

# 符号

让我们考虑一个具有参数$\theta$的全连接神经网络，$f(.;\theta): \mathbb{R}^{n_0} \to \mathbb{R}^{n_L}$。层从 0（输入）到$L$（输出）进行索引，每一层包含$n_0, \dots, n_L$个神经元，包括大小为$n_0$的输入和大小为$n_L$的输出。总共有$P = \sum_{l=0}^{L-1} (n_l + 1) n_{l+1}$个参数，因此我们有$\theta \in \mathbb{R}^P$。

训练数据集包含$N$个数据点，$\mathcal{D}=\{\mathbf{x}^{(i)}, y^{(i)}\}_{i=1}^N$。所有输入被表示为$\mathcal{X}=\{\mathbf{x}^{(i)}\}_{i=1}^N$，所有标签被表示为$\mathcal{Y}=\{y^{(i)}\}_{i=1}^N$。

现在让我们详细看一下每一层中的前向传播计算。对于$l=0, \dots, L-1$，每一层$l$定义一个带有权重矩阵$\mathbf{w}^{(l)} \in \mathbb{R}^{n_{l} \times n_{l+1}}$和偏置项$\mathbf{b}^{(l)} \in \mathbb{R}^{n_{l+1}}$的仿射变换$A^{(l)}$，以及一个逐点非线性函数$\sigma(.)$，它是[Lipschitz 连续的](https://en.wikipedia.org/wiki/Lipschitz_continuity)。

$$ \begin{aligned} A^{(0)} &= \mathbf{x} \\ \tilde{A}^{(l+1)}(\mathbf{x}) &= \frac{1}{\sqrt{n_l}} {\mathbf{w}^{(l)}}^\top A^{(l)} + \beta\mathbf{b}^{(l)}\quad\in\mathbb{R}^{n_{l+1}} & \text{; 预激活}\\ A^{(l+1)}(\mathbf{x}) &= \sigma(\tilde{A}^{(l+1)}(\mathbf{x}))\quad\in\mathbb{R}^{n_{l+1}} & \text{; 后激活} \end{aligned} $$

注意，*NTK 参数化* 在转换上应用了一个重新缩放权重 $1/\sqrt{n_l}$，以避免与无限宽度网络的发散。常数标量 $\beta \geq 0$ 控制偏置项的影响程度。

所有网络参数在以下分析中都初始化为独立同分布的高斯分布 $\mathcal{N}(0, 1)$。

# 神经切向核

**神经切向核 (NTK)** ([Jacot 等人，2018](https://arxiv.org/abs/1806.07572)) 是通过梯度下降理解神经网络训练的重要概念。在其核心，它解释了更新模型参数对一个数据样本的预测如何影响其他样本。

让我们逐步了解 NTK 背后的直觉。

要在训练期间最小化的经验损失函数 $\mathcal{L}: \mathbb{R}^P \to \mathbb{R}_+$ 定义如下，使用每个样本的成本函数 $\ell: \mathbb{R}^{n_0} \times \mathbb{R}^{n_L} \to \mathbb{R}_+$：

$$ \mathcal{L}(\theta) =\frac{1}{N} \sum_{i=1}^N \ell(f(\mathbf{x}^{(i)}; \theta), y^{(i)}) $$

根据链式法则，损失的梯度是：

$$ \nabla_\theta \mathcal{L}(\theta)= \frac{1}{N} \sum_{i=1}^N \underbrace{\nabla_\theta f(\mathbf{x}^{(i)}; \theta)}_{\text{大小为 }P \times n_L} \underbrace{\nabla_f \ell(f, y^{(i)})}_{\text{大小为 } n_L \times 1} $$

当跟踪网络参数 $\theta$ 在时间上的演变时，每次梯度下降更新都引入了一个微小步长的微小增量变化。由于更新步长足够小，可以近似看作是时间维度上的导数：

$$ \frac{d\theta}{d t} = - \nabla_\theta\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^N \nabla_\theta f(\mathbf{x}^{(i)}; \theta) \nabla_f \ell(f, y^{(i)}) $$

再次，根据链式法则，网络输出根据导数的演变如下：

$$ \frac{df(\mathbf{x};\theta)}{dt} = \frac{df(\mathbf{x};\theta)}{d\theta}\frac{d\theta}{dt} = -\frac{1}{N} \sum_{i=1}^N \color{blue}{\underbrace{\nabla_\theta f(\mathbf{x};\theta)^\top \nabla_\theta f(\mathbf{x}^{(i)}; \theta)}_\text{神经切向核}} \color{black}{\nabla_f \ell(f, y^{(i)})} $$

在这里，我们找到了上述公式中蓝色部分定义的 **神经切向核 (NTK)**，$K: \mathbb{R}^{n_0}\times\mathbb{R}^{n_0} \to \mathbb{R}^{n_L \times n_L}$：

$$ K(\mathbf{x}, \mathbf{x}'; \theta) = \nabla_\theta f(\mathbf{x};\theta)^\top \nabla_\theta f(\mathbf{x}'; \theta) $$

输出矩阵中每个位置 $(m, n), 1 \leq m, n \leq n_L$ 的每个条目是：

$$ K_{m,n}(\mathbf{x}, \mathbf{x}'; \theta) = \sum_{p=1}^P \frac{\partial f_m(\mathbf{x};\theta)}{\partial \theta_p} \frac{\partial f_n(\mathbf{x}';\theta)}{\partial \theta_p} $$

一个输入 $\mathbf{x}$ 的“特征映射”形式是 $\varphi(\mathbf{x}) = \nabla_\theta f(\mathbf{x};\theta)$。

# 无限宽度网络

要理解为什么一个梯度下降的效果对于网络参数的不同初始化如此相似，一些开创性的理论工作从无限宽度的网络开始。我们将通过使用 NTK 来详细证明，无限宽度的网络在训练以最小化经验损失时可以收敛到全局最小值。

## 与高斯过程的连接

深度神经网络与高斯过程有深刻的联系（[Neal 1994](https://www.cs.toronto.edu/~radford/ftp/pin.pdf)）。$L$ 层网络的输出函数 $f_i(\mathbf{x}; \theta)$ 对于 $i=1, \dots, n_L$，是具有协方差 $\Sigma^{(L)}$ 的独立同分布的中心化高斯过程，递归定义如下：

$$ \begin{aligned} \Sigma^{(1)}(\mathbf{x}, \mathbf{x}') &= \frac{1}{n_0}\mathbf{x}^\top{\mathbf{x}'} + \beta² \\ \lambda^{(l+1)}(\mathbf{x}, \mathbf{x}') &= \begin{bmatrix} \Sigma^{(l)}(\mathbf{x}, \mathbf{x}) & \Sigma^{(l)}(\mathbf{x}, \mathbf{x}') \\ \Sigma^{(l)}(\mathbf{x}', \mathbf{x}) & \Sigma^{(l)}(\mathbf{x}', \mathbf{x}') \end{bmatrix} \\ \Sigma^{(l+1)}(\mathbf{x}, \mathbf{x}') &= \mathbb{E}_{f \sim \mathcal{N}(0, \lambda^{(l)})}[\sigma(f(\mathbf{x})) \sigma(f(\mathbf{x}'))] + \beta² \end{aligned} $$

[Lee & Bahri 等人 (2018)](https://arxiv.org/abs/1711.00165) 通过数学归纳法展示了一个证明：

(1) 让我们从 $L=1$ 开始，当没有非线性函数且输入仅通过简单的仿射变换处理时：

$$ \begin{aligned} f(\mathbf{x};\theta) = \tilde{A}^{(1)}(\mathbf{x}) &= \frac{1}{\sqrt{n_0}}{\mathbf{w}^{(0)}}^\top\mathbf{x} + \beta\mathbf{b}^{(0)} \\ \text{其中 }\tilde{A}_m^{(1)}(\mathbf{x}) &= \frac{1}{\sqrt{n_0}}\sum_{i=1}^{n_0} w^{(0)}_{im}x_i + \beta b^{(0)}_m\quad \text{对于 }1 \leq m \leq n_1 \end{aligned} $$

由于权重和偏置是独立同分布初始化的，这个网络的所有输出维度 ${\tilde{A}^{(1)}_1(\mathbf{x}), \dots, \tilde{A}^{(1)}_{n_1}(\mathbf{x})}$ 也是独立同分布的。给定不同的输入，第 $m$ 个网络输出 $\tilde{A}^{(1)}_m(.)$ 具有联合多元高斯分布，相当于具有协方差函数的高斯过程（我们知道均值 $\mu_w=\mu_b=0$ 和方差 $\sigma²_w = \sigma²_b=1$）

$$ \begin{aligned} \Sigma^{(1)}(\mathbf{x}, \mathbf{x}') &= \mathbb{E}[\tilde{A}_m^{(1)}(\mathbf{x})\tilde{A}_m^{(1)}(\mathbf{x}')] \\ &= \mathbb{E}\Big[\Big( \frac{1}{\sqrt{n_0}}\sum_{i=1}^{n_0} w^{(0)}_{i,m}x_i + \beta b^{(0)}_m \Big) \Big( \frac{1}{\sqrt{n_0}}\sum_{i=1}^{n_0} w^{(0)}_{i,m}x'_i + \beta b^{(0)}_m \Big)\Big] \\ &= \frac{1}{n_0} \sigma²_w \sum_{i=1}^{n_0} \sum_{j=1}^{n_0} x_i{x'}_j + \frac{\beta \mu_b}{\sqrt{n_0}} \sum_{i=1}^{n_0} w_{im}(x_i + x'_i) + \sigma²_b \beta² \\ &= \frac{1}{n_0}\mathbf{x}^\top{\mathbf{x}'} + \beta² \end{aligned} $$

(2) 使用归纳法，我们首先假设命题对于 $L=l$，一个 $l$ 层网络成立，因此 $\tilde{A}^{(l)}_m(.)$ 是一个具有协方差 $\Sigma^{(l)}$ 的高斯过程，且 $\{\tilde{A}^{(l)}_i\}_{i=1}^{n_l}$ 是独立同分布的。

然后我们需要证明对于 $L=l+1$ 时命题也成立。我们通过计算输出来：

$$ \begin{aligned} f(\mathbf{x};\theta) = \tilde{A}^{(l+1)}(\mathbf{x}) &= \frac{1}{\sqrt{n_l}}{\mathbf{w}^{(l)}}^\top \sigma(\tilde{A}^{(l)}(\mathbf{x})) + \beta\mathbf{b}^{(l)} \\ \text{其中 }\tilde{A}^{(l+1)}_m(\mathbf{x}) &= \frac{1}{\sqrt{n_l}}\sum_{i=1}^{n_l} w^{(l)}_{im}\sigma(\tilde{A}^{(l)}_i(\mathbf{x})) + \beta b^{(l)}_m \quad \text{对于 }1 \leq m \leq n_{l+1} \end{aligned} $$

我们可以推断前几个隐藏层贡献的期望为零：

$$ \begin{aligned} \mathbb{E}[w^{(l)}_{im}\sigma(\tilde{A}^{(l)}_i(\mathbf{x}))] &= \mathbb{E}[w^{(l)}_{im}]\mathbb{E}[\sigma(\tilde{A}^{(l)}_i(\mathbf{x}))] = \mu_w \mathbb{E}[\sigma(\tilde{A}^{(l)}_i(\mathbf{x}))] = 0 \\ \mathbb{E}[\big(w^{(l)}_{im}\sigma(\tilde{A}^{(l)}_i(\mathbf{x}))\big)²] &= \mathbb{E}[{w^{(l)}_{im}}²]\mathbb{E}[\sigma(\tilde{A}^{(l)}_i(\mathbf{x}))²] = \sigma_w² \Sigma^{(l)}(\mathbf{x}, \mathbf{x}) = \Sigma^{(l)}(\mathbf{x}, \mathbf{x}) \end{aligned} $$

由于 $\{\tilde{A}^{(l)}_i(\mathbf{x})\}_{i=1}^{n_l}$ 是独立同分布的，根据中心极限定理，当隐藏层变得无限宽时 $n_l \to \infty$，$\tilde{A}^{(l+1)}_m(\mathbf{x})$ 服从高斯分布，方差为 $\beta² + \text{Var}(\tilde{A}_i^{(l)}(\mathbf{x}))$。注意 ${\tilde{A}^{(l+1)}_1(\mathbf{x}), \dots, \tilde{A}^{(l+1)}_{n_{l+1}}(\mathbf{x})}$ 仍然是独立同分布的。

$\tilde{A}^{(l+1)}_m(.)$ 等价于具有协方差函数的高斯过程：

$$ \begin{aligned} \Sigma^{(l+1)}(\mathbf{x}, \mathbf{x}') &= \mathbb{E}[\tilde{A}^{(l+1)}_m(\mathbf{x})\tilde{A}^{(l+1)}_m(\mathbf{x}')] \\ &= \frac{1}{n_l} \sigma\big(\tilde{A}^{(l)}_i(\mathbf{x})\big)^\top \sigma\big(\tilde{A}^{(l)}_i(\mathbf{x}')\big) + \beta² \quad\text{；类似于我们得到的 }\Sigma^{(1)} \end{aligned} $$

当 $n_l \to \infty$ 时，根据中心极限定理，

$$ \Sigma^{(l+1)}(\mathbf{x}, \mathbf{x}') \to \mathbb{E}_{f \sim \mathcal{N}(0, \Lambda^{(l)})}[\sigma(f(\mathbf{x}))^\top \sigma(f(\mathbf{x}'))] + \beta² $$

上述过程中的高斯过程形式被称为*神经网络高斯过程（NNGP）*（[Lee & Bahri et al. (2018)](https://arxiv.org/abs/1711.00165)）。

## 确定性神经切向核

最后，我们现在准备好深入研究 NTK 论文中最关键的命题：

**当 $n_1, \dots, n_L \to \infty$（无限宽度的网络）时，NTK 收敛为：**

+   **(1) 在初始化时是确定性的，意味着核与初始化值无关，仅由模型架构决定；以及**

+   **(2) 在训练过程中保持不变。**

证明依赖于数学归纳法：

(1) 首先，我们总是有 $K^{(0)} = 0$。当 $L=1$ 时，我们可以直接得到 NTK 的表示。它是确定性的，不依赖于网络初始化。没有隐藏层，因此没有无限宽度可取。

$$ \begin{aligned} f(\mathbf{x};\theta) &= \tilde{A}^{(1)}(\mathbf{x}) = \frac{1}{\sqrt{n_0}} {\mathbf{w}^{(0)}}^\top\mathbf{x} + \beta\mathbf{b}^{(0)} \\ K^{(1)}(\mathbf{x}, \mathbf{x}';\theta) &= \Big(\frac{\partial f(\mathbf{x}';\theta)}{\partial \mathbf{w}^{(0)}}\Big)^\top \frac{\partial f(\mathbf{x};\theta)}{\partial \mathbf{w}^{(0)}} + \Big(\frac{\partial f(\mathbf{x}';\theta)}{\partial \mathbf{b}^{(0)}}\Big)^\top \frac{\partial f(\mathbf{x};\theta)}{\partial \mathbf{b}^{(0)}} \\ &= \frac{1}{n_0} \mathbf{x}^\top{\mathbf{x}'} + \beta² = \Sigma^{(1)}(\mathbf{x}, \mathbf{x}') \end{aligned} $$

(2) 现在当 $L=l$ 时，我们假设一个总共有 $\tilde{P}$ 个参数的 $l$ 层网络，$\tilde{\theta} = (\mathbf{w}^{(0)}, \dots, \mathbf{w}^{(l-1)}, \mathbf{b}^{(0)}, \dots, \mathbf{b}^{(l-1)}) \in \mathbb{R}^\tilde{P}$，在 $n_1, \dots, n_{l-1} \to \infty$ 时收敛到确定性极限。

$$ K^{(l)}(\mathbf{x}, \mathbf{x}';\tilde{\theta}) = \nabla_{\tilde{\theta}} \tilde{A}^{(l)}(\mathbf{x})^\top \nabla_{\tilde{\theta}} \tilde{A}^{(l)}(\mathbf{x}') \to K^{(l)}_{\infty}(\mathbf{x}, \mathbf{x}') $$

注意 $K_\infty^{(l)}$ 不依赖于 $\theta$。

接下来让我们来看看 $L=l+1$ 的情况。与 $l$ 层网络相比，一个 $(l+1)$ 层网络有额外的权重矩阵 $\mathbf{w}^{(l)}$ 和偏置 $\mathbf{b}^{(l)}$，因此总参数包含 $\theta = (\tilde{\theta}, \mathbf{w}^{(l)}, \mathbf{b}^{(l)})$。

这个 $(l+1)$ 层网络的输出函数是：

$$ f(\mathbf{x};\theta) = \tilde{A}^{(l+1)}(\mathbf{x};\theta) = \frac{1}{\sqrt{n_l}} {\mathbf{w}^{(l)}}^\top \sigma\big(\tilde{A}^{(l)}(\mathbf{x})\big) + \beta \mathbf{b}^{(l)} $$

我们知道它对不同参数集的导数；为了简便起见，在以下方程中用 $\tilde{A}^{(l)} = \tilde{A}^{(l)}(\mathbf{x})$ 表示：

$$ \begin{aligned} \nabla_{\color{blue}{\mathbf{w}^{(l)}}} f(\mathbf{x};\theta) &= \color{blue}{ \frac{1}{\sqrt{n_l}} \sigma\big(\tilde{A}^{(l)}\big)^\top } \color{black}{\quad \in \mathbb{R}^{1 \times n_l}} \\ \nabla_{\color{green}{\mathbf{b}^{(l)}}} f(\mathbf{x};\theta) &= \color{green}{ \beta } \\ \nabla_{\color{red}{\tilde{\theta}}} f(\mathbf{x};\theta) &= \frac{1}{\sqrt{n_l}} \nabla_\tilde{\theta}\sigma(\tilde{A}^{(l)}) \mathbf{w}^{(l)} \\ &= \color{red}{ \frac{1}{\sqrt{n_l}} \begin{bmatrix} \dot{\sigma}(\tilde{A}_1^{(l)})\frac{\partial \tilde{A}_1^{(l)}}{\partial \tilde{\theta}_1} & \dots & \dot{\sigma}(\tilde{A}_{n_l}^{(l)})\frac{\partial \tilde{A}_{n_l}^{(l)}}{\partial \tilde{\theta}_1} \\ \vdots \\ \dot{\sigma}(\tilde{A}_1^{(l)})\frac{\partial \tilde{A}_1^{(l)}}{\partial \tilde{\theta}_\tilde{P}} & \dots & \dot{\sigma}(\tilde{A}_{n_l}^{(l)})\frac{\partial \tilde{A}_{n_l}^{(l)}}{\partial \tilde{\theta}_\tilde{P}}\\ \end{bmatrix} \mathbf{w}^{(l)} \color{black}{\quad \in \mathbb{R}^{\tilde{P} \times n_{l+1}}} } \end{aligned} $$

其中$\dot{\sigma}$是$\sigma$的导数，矩阵$\nabla_{\tilde{\theta}} f(\mathbf{x};\theta)$中位置$(p, m), 1 \leq p \leq \tilde{P}, 1 \leq m \leq n_{l+1}$的每个条目可以写成

$$ \frac{\partial f_m(\mathbf{x};\theta)}{\partial \tilde{\theta}_p} = \sum_{i=1}^{n_l} w^{(l)}_{im} \dot{\sigma}\big(\tilde{A}_i^{(l)} \big) \nabla_{\tilde{\theta}_p} \tilde{A}_i^{(l)} $$

这个$(l+1)$层网络的 NTK 可以相应地定义为：

$$ \begin{aligned} & K^{(l+1)}(\mathbf{x}, \mathbf{x}'; \theta) \\ =& \nabla_{\theta} f(\mathbf{x};\theta)^\top \nabla_{\theta} f(\mathbf{x};\theta) \\ =& \color{blue}{\nabla_{\mathbf{w}^{(l)}} f(\mathbf{x};\theta)^\top \nabla_{\mathbf{w}^{(l)}} f(\mathbf{x};\theta)} + \color{green}{\nabla_{\mathbf{b}^{(l)}} f(\mathbf{x};\theta)^\top \nabla_{\mathbf{b}^{(l)}} f(\mathbf{x};\theta)} + \color{red}{\nabla_{\tilde{\theta}} f(\mathbf{x};\theta)^\top \nabla_{\tilde{\theta}} f(\mathbf{x};\theta)} \\ =& \frac{1}{n_l} \Big[ \color{blue}{\sigma(\tilde{A}^{(l)})\sigma(\tilde{A}^{(l)})^\top} + \color{green}{\beta²} \\ &+ \color{red}{ {\mathbf{w}^{(l)}}^\top \begin{bmatrix} \dot{\sigma}(\tilde{A}_1^{(l)})\dot{\sigma}(\tilde{A}_1^{(l)})\sum_{p=1}^\tilde{P} \frac{\partial \tilde{A}_1^{(l)}}{\partial \tilde{\theta}_p}\frac{\partial \tilde{A}_1^{(l)}}{\partial \tilde{\theta}_p} & \dots & \dot{\sigma}(\tilde{A}_1^{(l)})\dot{\sigma}(\tilde{A}_{n_l}^{(l)})\sum_{p=1}^\tilde{P} \frac{\partial \tilde{A}_1^{(l)}...

矩阵 $K^{(l+1)}$ 中位置 $(m, n), 1 \leq m, n \leq n_{l+1}$ 处的每个单独条目可写为：

$$ \begin{aligned} K^{(l+1)}_{mn} =& \frac{1}{n_l}\Big[ \color{blue}{\sigma(\tilde{A}_m^{(l)})\sigma(\tilde{A}_n^{(l)})} + \color{green}{\beta²} + \color{red}{ \sum_{i=1}^{n_l} \sum_{j=1}^{n_l} w^{(l)}_{im} w^{(l)}_{in} \dot{\sigma}(\tilde{A}_i^{(l)}) \dot{\sigma}(\tilde{A}_{j}^{(l)}) K_{ij}^{(l)} } \Big] \end{aligned} $$

当 $n_l \to \infty$ 时，蓝色和绿色部分的极限为（请参见前一节中的证明）：

$$ \frac{1}{n_l}\sigma(\tilde{A}^{(l)})\sigma(\tilde{A}^{(l)}) + \beta²\to \Sigma^{(l+1)} $$

红色部分的极限为：

$$ \sum_{i=1}^{n_l} \sum_{j=1}^{n_l} w^{(l)}_{im} w^{(l)}_{in} \dot{\sigma}(\tilde{A}_i^{(l)}) \dot{\sigma}(\tilde{A}_{j}^{(l)}) K_{ij}^{(l)} \to \sum_{i=1}^{n_l} \sum_{j=1}^{n_l} w^{(l)}_{im} w^{(l)}_{in} \dot{\sigma}(\tilde{A}_i^{(l)}) \dot{\sigma}(\tilde{A}_{j}^{(l)}) K_{\infty,ij}^{(l)} $$

后来，[Arora 等人（2019）](https://arxiv.org/abs/1904.11955)提供了一个证明，具有更弱的极限，不需要所有隐藏层都是无限宽的，只需要最小宽度足够大。

## 线性化模型

根据前一节，根据导数链规则，我们已经知道宽度无限网络输出的梯度更新如下；为简洁起见，我们在以下分析中省略输入：

$$ \begin{aligned} \frac{df(\theta)}{dt} &= -\eta\nabla_\theta f(\theta)^\top \nabla_\theta f(\theta) \nabla_f \mathcal{L} & \\ &= -\eta\nabla_\theta f(\theta)^\top \nabla_\theta f(\theta) \nabla_f \mathcal{L} & \\ &= -\eta K(\theta) \nabla_f \mathcal{L} \\ &= \color{cyan}{-\eta K_\infty \nabla_f \mathcal{L}} & \text{；对于宽度无限的网络}\\ \end{aligned} $$

为了追踪$\theta$随时间的演变，让我们将其视为时间步长$t$的函数。通过泰勒展开，网络学习动态可以简化为：

$$ f(\theta(t)) \approx f^\text{lin}(\theta(t)) = f(\theta(0)) + \underbrace{\nabla_\theta f(\theta(0))}_{\text{形式上 }\nabla_\theta f(\mathbf{x}; \theta) \vert_{\theta=\theta(0)}} (\theta(t) - \theta(0)) $$

这种形式通常被称为*线性化*模型，假设$\theta(0)$，$f(\theta(0))$和$\nabla_\theta f(\theta(0))$都是常数。假设增量时间步$t$非常小，参数通过梯度下降更新：

$$ \begin{aligned} \theta(t) - \theta(0) &= - \eta \nabla_\theta \mathcal{L}(\theta) = - \eta \nabla_\theta f(\theta)^\top \nabla_f \mathcal{L} \\ f^\text{lin}(\theta(t)) - f(\theta(0)) &= - \eta\nabla_\theta f(\theta(0))^\top \nabla_\theta f(\mathcal{X};\theta(0)) \nabla_f \mathcal{L} \\ \frac{df(\theta(t))}{dt} &= - \eta K(\theta(0)) \nabla_f \mathcal{L} \\ \frac{df(\theta(t))}{dt} &= \color{cyan}{- \eta K_\infty \nabla_f \mathcal{L}} & \text{；对于宽度无限的网络}\\ \end{aligned} $$

最终我们得到了相同的学习动态，这意味着一个宽度无限的神经网络可以被大大简化为上述线性化模型（[Lee & Xiao, et al. 2019](https://arxiv.org/abs/1902.06720)）所控制。

在一个简单的情况下，当经验损失是均方误差损失时，$\nabla_\theta \mathcal{L}(\theta) = f(\mathcal{X}; \theta) - \mathcal{Y}$，网络的动态变为简单的线性 ODE，并且可以以封闭形式解决：

$$ \begin{aligned} \frac{df(\theta)}{dt} =& -\eta K_\infty (f(\theta) - \mathcal{Y}) & \\ \frac{dg(\theta)}{dt} =& -\eta K_\infty g(\theta) & \text{；让}g(\theta)=f(\theta) - \mathcal{Y} \\ \int \frac{dg(\theta)}{g(\theta)} =& -\eta \int K_\infty dt & \\ g(\theta) &= C e^{-\eta K_\infty t} & \end{aligned} $$

当$t=0$时，我们有$C=f(\theta(0)) - \mathcal{Y}$，因此，

$$ f(\theta) = (f(\theta(0)) - \mathcal{Y})e^{-\eta K_\infty t} + \mathcal{Y} \\ = f(\theta(0))e^{-K_\infty t} + (I - e^{-\eta K_\infty t})\mathcal{Y} $$

## 懒惰训练

人们观察到，当神经网络过度参数化时，模型能够快速收敛到零的训练损失，但网络参数几乎不会改变。*懒惰训练*指的就是这种现象。换句话说，当损失$\mathcal{L}$有相当大的减少时，网络$f$的微分（也称为雅可比矩阵）的变化仍然非常小。

让$\theta(0)$为初始网络参数，$\theta(T)$为损失最小化为零时的最终网络参数。参数空间的变化可以用一阶泰勒展开来近似：

$$ \begin{aligned} \hat{y} = f(\theta(T)) &\approx f(\theta(0)) + \nabla_\theta f(\theta(0)) (\theta(T) - \theta(0)) \\ \text{因此 }\Delta \theta &= \theta(T) - \theta(0) \approx \frac{\|\hat{y} - f(\theta(0))\|}{\| \nabla_\theta f(\theta(0)) \|} \end{aligned} $$

仍然遵循一阶泰勒展开，我们可以跟踪$f$的微分的变化：

$$ \begin{aligned} \nabla_\theta f(\theta(T)) &\approx \nabla_\theta f(\theta(0)) + \nabla²_\theta f(\theta(0)) \Delta\theta \\ &= \nabla_\theta f(\theta(0)) + \nabla²_\theta f(\theta(0)) \frac{\|\hat{y} - f(\mathbf{x};\theta(0))\|}{\| \nabla_\theta f(\theta(0)) \|} \\ \text{因此 }\Delta\big(\nabla_\theta f\big) &= \nabla_\theta f(\theta(T)) - \nabla_\theta f(\theta(0)) = \|\hat{y} - f(\mathbf{x};\theta(0))\| \frac{\nabla²_\theta f(\theta(0))}{\| \nabla_\theta f(\theta(0)) \|} \end{aligned} $$

让$\kappa(\theta)$表示$f$的微分相对于参数空间变化的相对变化：

$$ \kappa(\theta) = \frac{\Delta\big(\nabla_\theta f\big)}{\| \nabla_\theta f(\theta(0)) \|} = \|\hat{y} - f(\theta(0))\| \frac{\nabla²_\theta f(\theta(0))}{\| \nabla_\theta f(\theta(0)) \|²} $$

[Chizat et al. (2019)](https://arxiv.org/abs/1812.07956)证明了对于一个两层神经网络，当隐藏神经元的数量$\to \infty$时，$\mathbb{E}[\kappa(\theta_0)] \to 0$（进入懒惰状态）。此外，推荐阅读[这篇文章](https://rajatvd.github.io/NTK/)以获取更多关于线性化模型和懒惰训练的讨论。

# 引用

引用为：

> Weng, Lilian. (Sep 2022). Some math behind neural tangent kernel. Lil’Log. https://lilianweng.github.io/posts/2022-09-08-ntk/.

或者

```py
@article{weng2022ntk,
  title   = "Some Math behind Neural Tangent Kernel",
  author  = "Weng, Lilian",
  journal = "Lil'Log",
  year    = "2022",
  month   = "Sep",
  url     = "https://lilianweng.github.io/posts/2022-09-08-ntk/"
} 
```

# 参考文献

[1] Jacot 等人 [“神经切向核：神经网络中的收敛和泛化。”](https://arxiv.org/abs/1806.07572) NeuriPS 2018.

[2] Radford M. Neal. “无限网络的先验。” 神经网络的贝叶斯学习。Springer, 纽约, 纽约, 1996. 29-53.

[3] 李和巴里等人 [“深度神经网络作为高斯过程。”](https://arxiv.org/abs/1711.00165) ICLR 2018.

[4] Chizat 等人 [“关于可微编程中的懒惰训练”](https://arxiv.org/abs/1812.07956) NeuriPS 2019.

[5] 李和肖等人 [“任意深度的宽神经网络在梯度下降下演变为线性模型。”](https://arxiv.org/abs/1902.06720) NeuriPS 2019.

[6] Arora 等人 [“关于无限宽神经网络的精确计算。”](https://arxiv.org/abs/1904.11955) NeurIPS 2019.

[7] (YouTube 视频) [“神经切向核：神经网络中的收敛和泛化”](https://www.youtube.com/watch?v=raT2ECrvbag) 由 Arthur Jacot, 2018 年 11 月.

[8] (YouTube 视频) [“讲座 7 - 深度学习基础：神经切向核”](https://www.youtube.com/watch?v=DObobAnELkU) 由 Soheil Feizi, 2020 年 9 月.

[9] [“理解神经切向核。”](https://rajatvd.github.io/NTK/) Rajat 的博客.

[10] [“神经切向核。”](https://appliedprobability.blog/2021/03/10/neural-tangent-kernel/) 应用概率笔记, 2021 年 3 月.

[11] [“关于神经切向核的一些直觉。”](https://www.inference.vc/neural-tangent-kernels-some-intuition-for-kernel-gradient-descent/) inFERENCe, 2020 年 11 月.
