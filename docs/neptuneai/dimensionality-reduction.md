# 机器学习的降维

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/dimensionality-reduction>

数据构成了任何机器学习算法的基础，没有它，数据科学就不可能发生。有时，它可能包含大量的特性，有些甚至不是必需的。这种冗余信息使得建模变得复杂。此外，由于高维度，通过可视化来解释和理解数据变得困难。这就是降维发挥作用的地方。

在本文中，您将了解到:

1.  什么是降维？
2.  什么是维度的诅咒？
3.  用于降维的工具和库
4.  用于降维的算法
5.  应用程序
6.  优点和缺点

## 什么是降维？

降维是减少数据集中要素数量的任务。在像[回归](/web/20221219034305/https://neptune.ai/blog/random-forest-regression-when-does-it-fail-and-why)或[分类](/web/20221219034305/https://neptune.ai/blog/image-classification-tips-and-tricks-from-13-kaggle-competitions)这样的机器学习任务中，经常有太多的变量需要处理。这些变量也被称为**特性**。特征数量越多，建模就越困难，这就是所谓的**维度诅咒**。这将在下一节详细讨论。

此外，这些特征中的一些可能非常多余，会向数据集添加噪声，并且在训练数据中包含它们是没有意义的。这就是需要减少特征空间的地方。

降维的过程实质上是将数据从高维特征空间转换到低维特征空间。同时，在转换过程中不丢失数据中存在的有意义的属性也很重要。

降维通常用于数据可视化以理解和解释数据，以及用于机器学习或深度学习技术以简化手头的任务。

### 维度的诅咒

众所周知，ML/DL 算法需要大量的数据来学习不变性、模式和表示。如果该数据包含大量特征，这可能会导致维数灾难。维数灾难首先由 [Bellman](https://web.archive.org/web/20221219034305/https://zbmath.org/0103.12901) 提出，描述了为了以一定的精度估计任意函数，估计所需的特征或维数呈指数增长。对于产生更多**稀疏性**的大数据来说尤其如此。

数据的稀疏性通常指的是具有零值的特征；这并不意味着缺少值。如果数据具有大量稀疏特征，则空间和计算复杂性会增加。Oliver[Kuss【2002】](https://web.archive.org/web/20221219034305/https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.1421)表明在稀疏数据上训练的模型在测试数据集中表现不佳。换句话说，训练期间的模型学习噪声，并且它们不能很好地概括。因此他们过度适应。

当数据稀疏时，训练数据集中的观察值或样本很难聚类，因为高维数据会导致数据集中的每个观察值看起来彼此等距。如果数据是有意义的和非冗余的，那么将存在相似数据点聚集在一起的区域，此外，它们必须具有统计显著性。

高维数据带来的问题有:

1.  冒着过度适应机器学习模型的风险。
2.  相似特征聚类困难。
3.  增加了空间和计算时间的复杂性。

另一方面，非稀疏数据或密集数据是具有非零特征的数据。除了包含非零特征之外，它们还包含有意义且非冗余的信息。

为了解决维数灾难，使用了降维等方法。降维技术对于将稀疏特征转换为密集特征非常有用。此外，降维还用于清洗数据和特征提取。

最流行的降维库是 **scikit-learn** (sklearn)。该库由三个主要的降维算法模块组成:

1.  分解算法
    *   主成分分析
    *   核主成分分析
    *   非负矩阵分解
    *   奇异值分解
2.  流形学习算法
    *   t 分布随机邻居嵌入
    *   光谱嵌入
    *   局部线性嵌入
3.  判别分析
    *   线性判别分析

当谈到深度学习时，可以构建像自动编码器这样的算法来降低维度，并学习特征和表示。Pytorch、Pytorch Lightning、Keras 和 TensorFlow 等框架用于创建自动编码器。

## 降维算法

先说第一类算法。

### 分解算法

scikit-learn 中的分解算法涉及维数约减算法。我们可以使用以下命令调用各种技术:

```py
from sklearn.decomposition import PCA, KernelPCA, NMF
```

#### 主成分分析

主成分分析(PCA)是一种降维方法，通过保留在高维输入空间中测量的**方差**来寻找低维空间。这是一种无监督的降维方法。

PCA 变换是线性变换。它包括寻找主成分的过程，即把特征矩阵分解成特征向量。这意味着当数据集的分布是非线性时，PCA 将是无效的。

让我们用 python 代码来理解 PCA。

```py
def pca(X=np.array([]), no_dims=50):

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    Mean = np.tile(np.mean(X, 0), (n, 1))
    X = X - Mean
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y
```

PCA 的实现非常简单。我们可以将整个过程定义为四个步骤:

1.  **标准化**:通过取原始数据集与整个数据集的平均值之间的差值，将数据转换到一个通用的标度。这将使分布 0 居中。
2.  **求协方差**:协方差会帮助我们理解均值和原始数据之间的关系。
3.  **确定主成分**:主成分可以通过计算特征向量和特征值来确定。**特征向量**是一组特殊的向量，帮助我们理解数据的结构和属性，这些数据将成为主要成分。另一方面，**特征值**帮助我们确定主成分。最高的特征值及其对应的特征向量构成了最重要的主分量。
4.  **最终输出**:是标准化矩阵和特征向量的点积。请注意，列或特征的数量将会改变。

减少数据变量的数量不仅降低了复杂性，还降低了机器学习模型的准确性。然而，由于特征数量较少，因此易于探索、可视化和分析，这也使得机器学习算法的计算成本较低。简而言之，主成分分析的思想是减少数据集的变量数量，同时尽可能多地保留信息。

我们也来看看 sklearn 为 PCA 提供的模块和函数。

我们可以从加载最多的数据集开始:

```py
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape
```

(1797, 64)

数据由 8×8 像素图像组成，这意味着它们是 64 维的。为了了解这些点之间的关系，我们可以使用 PCA 将它们投影到更低的维度，如 2-D:

```py
from sklearn.decomposition import PCA

pca = PCA(2)  
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)
```

(1797, 64)

(1797, 2)

现在，让我们画出前两个主要成分。

```py
plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();
```

![Principal Component Analysis (PCA)](img/a88ccf09dc4c306696fb5636b373601c.png)

*Dimensionality reduction technique: PCA | Source: Author*

我们可以看到，在大多数情况下，PCA 最优地找到了可以非常有效地聚类相似分布的主成分。

#### 内核 PCA (KPCA)

我们之前描述的 PCA 变换是线性变换，对于非线性分布无效。要处理非线性分布，基本思想是使用核技巧。

内核技巧只是一种将非线性数据投影到更高维度空间并分离不同数据分布的方法。一旦分布被分离，我们可以使用主成分分析来线性分离它们。

![Kernel PCA](img/1f6ddbde18f5fac09b7adcf51a7a9e50.png)

*Dimensionality reduction technique: KPCA | Source: Author*

核 PCA 使用核函数ϕ来计算非线性映射的数据的点积。换句话说，函数ϕ通过创建原始特征的非线性组合，将原始 d 维特征映射到更大的 k 维特征空间。

假设数据集 x 包含两个特征 x1 和 x2:

应用内核技巧后，我们得到:

为了更直观地理解内核 PCA，让我们定义一个不能线性分离的特征空间。

```py
​​from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
np.random.seed(0)
X, y = make_circles(n_samples=400, factor=.3, noise=.05)
```

现在，让我们绘制并查看我们的数据集。

```py
plt.figure(figsize=(15,10))
plt.subplot(1, 2, 1, aspect='equal')
plt.title("Original space")
reds = y == 0
blues = y == 1

plt.scatter(X[reds, 0], X[reds, 1], c="red",
           s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
           s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
```

![Kernel PCA](img/f2431ee7e8c38ee95dae6aae00b7173e.png)

*Dimensionality reduction technique: KPCA | Source: Author*

正如您在该数据集中看到的，这两个类不能线性分离。现在，让我们定义内核 PCA，看看它是如何分离这个特征空间的。

```py
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10, )
X_kpca = kpca.fit_transform(X)
plt.subplot(1, 2, 2, aspect='equal')
plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red",
           s=20, edgecolor='k')
plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue",
           s=20, edgecolor='k')
plt.title("Projection by KPCA")
plt.xlabel(r"1st principal component in space induced by $phi$")
plt.ylabel("2nd component")
```

![Kernel PCA](img/223ae8a4b1791e12609786fdfc57ba32.png)

*Dimensionality reduction technique: KPCA | Source: Author*

应用 KPCA 后，它能够线性分离数据集中的两个类。

#### 奇异值分解

奇异值分解或 SVD 是实矩阵或复矩阵的因式分解方法。当处理稀疏数据集时，它是有效的；有许多零条目的数据集。这种类型的数据集通常出现在推荐系统、评级和评论数据集中，等等。

SVD 的思想是将形状 nXp 的每个矩阵分解成 A = USV ^T ，其中 U 是正交矩阵，S 是对角矩阵，^(V^T也是正交矩阵。)

SVD 的优点是正交矩阵捕获了原始矩阵 A 的结构，这意味着当乘以其他数时，它们的属性不会改变。这可以帮助我们近似 a。

现在让我们用代码来理解 SVD。为了更好地理解该算法，我们将使用 scikit-learn 提供的人脸数据集。

```py
from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
```

绘制图像以了解我们正在处理的内容。

```py
X = lfw_people.images.reshape(img_count, img_width * img_height)
X0_img = X[0].reshape(img_height, img_width)

plt.imshow(X0_img, cmap=plt.cm.gray)
```

创建一个函数，以便于图像的可视化。

```py
def draw_img(img_vector, h=img_height, w=img_width):
   plt.imshow( img_vector.reshape((h,w)), cmap=plt.cm.gray)
   plt.xticks(())
   plt.yticks(())
draw_img(X[49])
```

在应用 SVD 之前，最好将数据标准化。

```py
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_std=False)
Xstd = scaler.fit_transform(X)
```

标准化后，这就是图像的外观。

值得注意的是，我们总是可以通过执行逆变换来恢复原始图像。

```py
Xorig = scaler.inverse_transform(Xstd)
draw_img(Xorig[49])
```

现在，我们可以应用 NumPy 中的 SVD 函数，并将矩阵分解为三个矩阵。

```py
from numpy.linalg import svd

U, S, VT = svd(Xstd)
```

为了检查这个函数是否有效，我们总是可以执行三个矩阵的矩阵乘法。

```py
US = U*S
Xhat = US @ VT[0:1288,:]

Xhat_orig = scaler.inverse_transform(Xhat)
draw_img(Xhat_orig[49])
```

现在，让我们进行降维。为此，我们只需减少正交矩阵的特征数量。

```py
Xhat_500 = US[:, 0:500] @ VT[0:500, :]

Xhat_500_orig = scaler.inverse_transform(Xhat_500)

draw_img(Xhat_500_orig[49])
```

我们可以进一步减少更多的功能，看看结果。

```py
Xhat_100 = US[:, 0:100] @ VT[0:100, :]

Xhat_100_orig = scaler.inverse_transform(Xhat_100)

draw_img(Xhat_100_orig[49])
```

现在，让我们创建一个函数，允许我们减少图像的尺寸。

```py
def dim_reduce(US_, VT_, dim=100):

   Xhat_ = US_[:, 0:dim] @ VT_[0:dim, :]

   return scaler.inverse_transform(Xhat_)
```

用不同数量的特征绘制图像。

```py
dim_vec = [50, 100, 200, 400, 800]

plt.figure(figsize=(1.8 * len(dim_vec), 2.4))

for i, d in enumerate(dim_vec):
   plt.subplot(1, len(dim_vec), i + 1)
   draw_img(dim_reduce(US, VT, d)[49])
```

![Singular Value Decomposition](img/d6890cb5280847894d7187692621d866.png)

*Dimensionality reduction technique: SVD | Source: Author*

如你所见，第一幅图像包含最少数量的特征，但它仍然可以构建图像的抽象版本，随着我们增加特征，我们最终获得原始图像。这证明了 SVD 可以保留数据的基本结构。

#### 非负矩阵分解(NMF)

NMF 是一种无监督的机器学习算法。当一个维数为 mXn 的非负输入矩阵 X 被提供给算法时，它被分解成两个非负矩阵 W 和 H 的乘积。W 的维数为 mXp，而 H 的维数为 pXn。

**其中 Y = W.H**

从上面的等式可以看出，要分解矩阵，我们需要最小化距离。最广泛使用的距离函数是平方 Frobenius 范数；这是欧几里德范数在矩阵上的扩展。

同样值得注意的是，这个问题通常是不可解的，这就是为什么它是近似的。事实证明，NMF 有利于数据集的基于部件的表示，即 NMF 提供了一种**高效的**、**分布式表示、**，并且可以帮助发现数据中感兴趣的结构。

让我们用代码来理解 NMF。我们将使用在 SVD 中使用的相同数据。

首先，我们将使模型符合数据。

```py
from sklearn.decomposition import NMF
model = NMF(n_components=200, init='nndsvd', random_state=0)
W = model.fit_transform(X)
V = model.components_
```

NMF 需要一点时间来分解数据。一旦数据被分解，我们就可以可视化分解的组件。

```py
num_faces = 20
plt.figure(figsize=(1.8 * 5, 2.4 * 4))

for i in range(0, num_faces):
   plt.subplot(4, 5, i + 1)
   draw_img(V[i])
```

![Non-negative Matrix Factorization](img/f2b91986a9a5eca8f519a8e4477b2836.png)

*Dimensionality reduction technique: NMF | Source: Author*

从上图我们可以看出，NMF 捕捉数据底层结构的效率非常高。同样值得一提的是，NMF 只获取了线性属性。

**NMF 的优势**:

1.  数据压缩和可视化
2.  对噪声的鲁棒性
3.  更容易理解

### 流形学习

到目前为止，我们已经看到了只涉及线性变换的方法。但是，当我们有一个非线性数据集时，我们该怎么办呢？

流形学习是一种无监督学习，旨在对非线性数据集进行降维。同样，scikit-learn 提供了一个由各种非线性降维技术组成的模块。我们可以通过这个命令调用这些类或技术:

```py
from sklearn.manifold import TSNE, LocallyLinearEmbedding, SpectralEmbedding
```

#### t 分布随机邻居嵌入(t-SNE)

t-分布式随机邻居嵌入或 t-SNE 是一种非常适合数据可视化的降维技术。与简单地最大化方差的 PCA 不同，t-SNE 最小化两个分布之间的差异。本质上，它在低维空间中重建了高维空间的分布，而不是最大化方差，甚至没有使用核技巧。

我们可以通过三个简单的步骤对 SNE 霸王龙有一个高层次的了解:

1.  它首先为高维样本创建一个概率分布。
2.  然后，它为低维嵌入中的点定义了类似的分布。
3.  最后，它试图最小化两个分布之间的 KL-散度。

现在我们用代码来理解一下。对于 SNE 霸王龙，我们将再次使用 MNIST 数据集。首先，我们导入 TSNE，然后导入数据。

```py
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape)

for i in range(0,5):
   plt.figure(figsize=(5,5))
   plt.imshow(digits.images[i])
   plt.show()
```

![t-Distributed Stochastic Neighbor Embedding](img/70be75d61df30ba735443957a65240ef.png)

*Dimensionality reduction technique: t-SNE | Source: Author*

然后我们将使用 np.vstack 按顺序存储这些数字。

```py
X = np.vstack([digits.data[digits.target==i] for i in range(10)])
Y = np.hstack([digits.target[digits.target==i] for i in range(10)])
```

我们将对数据集应用 t-SNE。

```py
digits_final = TSNE(perplexity=30).fit_transform(X)
```

我们现在将创建一个函数来可视化数据。

```py
def plot(x, colors):
    palette = np.array(sb.color_palette("hls", 10))  

   f = plt.figure(figsize=(8, 8))
   ax = plt.subplot(aspect='equal')
   sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,c=palette[colors.astype(np.int)])

   txts = []
   for i in range(10):

       xtext, ytext = np.median(x[colors == i, :], axis=0)
       txt = ax.text(xtext, ytext, str(i), fontsize=24)
       txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"),
                             pe.Normal()])
       txts.append(txt)
   return f, ax, txts
```

现在，我们对转换后的数据集执行数据可视化。

```py
plot(digits_final,Y)
```

![t-Distributed Stochastic Neighbor Embedding](img/a7297afe5a7593de21afc3d145f3b835.png)

*Dimensionality reduction technique: t-SNE | Source: Author*

可以看出，SNE 霸王龙对数据进行了完美的聚类。与 PCA 相比，t-SNE 在非线性数据上表现良好。t-SNE 的缺点是，当数据很大时，它会消耗大量的时间。所以最好先进行主成分分析，然后再进行 t-SNE。

#### 局部线性嵌入(LLE)

局部线性嵌入或 LLE 是一种非线性和无监督的降维机器学习方法。LLE 利用数据的局部结构或拓扑，并将其保存在低维特征空间中。

LLE 优化速度更快，但在噪音数据上失败。

让我们将整个过程分成三个简单的步骤:

1.  找到数据点的最近邻。
2.  通过将每个数据点近似为其 k 个最近邻点的加权线性组合并最小化它们与其线性表示之间的平方距离来构建权重矩阵。
3.  通过使用基于**特征向量的优化**技术将权重映射到低维空间。

![Locally Linear Embedding](img/395f6ddd9b242eb0c5e6ad4dc56d0163.png)

*Dimensionality reduction technique: LLE | Source: S. T. Roweis and L. K. Saul, Nonlinear dimensionality reduction by locally linear embedding*

![Locally Linear Embedding](img/ed04bb41bd926f3933e2fb442d03815c.png)

*Dimensionality reduction technique: LLE | Source: [Scikit Learn](https://web.archive.org/web/20221219034305/https://scikit-learn.org/stable/modules/manifold.html#manifold)*

#### 光谱嵌入

谱嵌入是另一种非线性降维技术，也是一种无监督的机器学习算法。谱嵌入旨在基于低维表示找到不同类别的聚类。

我们可以再次将整个过程分成三个简单的步骤:

1.  **预处理**:构建数据或图形的拉普拉斯矩阵表示。
2.  **分解**:计算构造好的矩阵的特征值和特征向量，然后将每个点映射到一个更低维的表示上。谱嵌入利用了第二小特征值及其对应的特征向量。
3.  **聚类**:根据表示法，将点分配给两个或多个聚类。聚类通常使用 k-means 聚类来完成。

**应用**:光谱嵌入在图像**分割中得到应用**。

### 判别分析

判别分析是 scikit-learn 提供的另一个模块。可以使用以下命令调用它:

```py
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

#### 线性判别分析(LDA)

LDA 是一种算法，用于查找数据集中要素的线性组合。像 PCA 一样，LDA 也是一种基于线性变换的技术。但与 PCA 不同，它是一种监督学习算法。

LDA 计算方向，即可以创建决策边界并最大化多个类之间的分离的线性判别式。对于多类分类任务也非常有效。

为了对 LDA 有一个更直观的理解，考虑绘制两个类的关系，如下图所示。

解决这个问题的一种方法是将所有数据点投影到 x 轴上。

这种方法会导致信息丢失，而且是多余的。

更好的方法是计算数据中所有点之间的距离，并拟合一条穿过这些点的新直线。这条新线现在可以用来投影所有的点。

这条新线最小化了方差，并通过最大化两个类之间的距离来有效地对它们进行分类。

LDA 也可以用于多元数据。它使数据推断变得非常简单。可以使用以下 5 个步骤计算 LDA:

1.  计算数据集中不同类别的 d 维均值向量。
2.  计算散布矩阵(类间和类内散布矩阵)。散布矩阵用于估计协方差矩阵。当协方差矩阵难以计算或者两个随机变量的联合可变性难以计算时，就要这样做。
3.  计算散布矩阵的特征向量(e1，e2，e3…ed)和相应的特征值(λ1，λ2，…，λd)。
4.  将特征向量按特征值递减排序，选择 k 个特征值最大的特征向量，形成 d×k 维矩阵 W(其中每列代表一个特征向量)。
5.  使用这个 d×k 特征向量矩阵将样本变换到新的子空间上。这可以通过矩阵乘法来概括:Y=X×W(其中 X 是表示 n 个样本的 n×d 维矩阵，Y 是新子空间中的变换后的 n×k 维样本)。

要了解 LDA，你可以看看这篇文章。

## 降维的应用

降维在许多实际应用中都有应用，其中包括:

*   客户关系管理
*   文本分类
*   图像检索
*   入侵检测
*   医学图像分割

## 降维的利与弊

**降维的优势**:

*   它通过减少特征来帮助数据压缩。
*   它减少了存储。
*   它使机器学习算法在计算上高效。
*   它还有助于移除多余的特征和噪声。
*   它解决了维度的诅咒

**降维的缺点**:

*   这可能会导致一些数据丢失。
*   准确性大打折扣。

## 最后的想法

在本文中，我们学习了降维以及维数灾难。我们通过数学细节和代码讨论了降维中使用的不同算法。

值得一提的是，这些算法应该根据手头的任务来使用。例如，如果你的数据是线性的，那么使用分解方法，否则使用流形学习技术。

首先将数据可视化，然后决定使用哪种方法，这被认为是一种良好的做法。此外，不要把自己局限在一种方法上，而是探索不同的方法，看看哪一种是最合适的。

我希望你从这篇文章中学到了一些东西。快乐学习。

### 参考

1.  [降维介绍–GeeksforGeeks](https://web.archive.org/web/20221219034305/https://www.geeksforgeeks.org/dimensionality-reduction/)
2.  [机器学习降维介绍](https://web.archive.org/web/20221219034305/https://machinelearningmastery.com/dimensionality-reduction-for-machine-learning/)
3.  [主成分分析:回顾与最新进展](https://web.archive.org/web/20221219034305/https://royalsocietypublishing.org/doi/10.1098/rsta.2015.0202)
4.  [Python 中的线性判别分析|作者 Cory Maklin](https://web.archive.org/web/20221219034305/https://towardsdatascience.com/linear-discriminant-analysis-in-python-76b8b17817c2)
5.  [在机器学习模型中使用稀疏特征](https://web.archive.org/web/20221219034305/https://www.kdnuggets.com/2021/01/sparse-features-machine-learning-models.html)
6.  [必知:什么是维度的诅咒？](https://web.archive.org/web/20221219034305/https://www.kdnuggets.com/2017/04/must-know-curse-dimensionality.html)
7.  [维数灾难和初学者应该如何克服它](https://web.archive.org/web/20221219034305/https://analyticsindiamag.com/curse-of-dimensionality-and-what-beginners-should-do-to-overcome-it/)
8.  [Python 的 6 种降维算法](https://web.archive.org/web/20221219034305/https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/)
9.  硬化 API 参考
10.  [t 分布随机邻居嵌入](https://web.archive.org/web/20221219034305/https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
11.  [特征选择和提取](https://web.archive.org/web/20221219034305/https://www.sciencedirect.com/science/article/pii/B9780124095458000029)