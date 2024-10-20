# 利用独立分量分析(ICA)更深入地降低维数

> 原文：<https://blog.paperspace.com/dimension-reduction-with-independent-components-analysis/>

###### 本教程是关于降维的 7 部分系列的一部分:

1.  [理解主成分分析(PCA)的降维](https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/)

3.  [多维标度(MDS)](https://blog.paperspace.com/dimension-reduction-with-multi-dimension-scaling/)
4.  LLE *(即将推出！)*
5.  t-SNE *(即将推出！)*
6.  IsoMap *(即将推出！)*
7.  自动编码器*(即将推出！)*

(在 [github](https://github.com/asdspal/dimRed) 上有一个带数学和代码的 IPython 笔记本。)

今天，我们将学习另一种叫做 ICA 的降维方法。ICA 是一种线性降维方法，它将数据集转换成独立分量的列。盲源分离和“鸡尾酒会问题”是它的其他名称。ICA 是神经成像、fMRI 和 EEG 分析中的重要工具，有助于将正常信号与异常信号分开。那么，ICA 到底是什么？

ICA 代表独立成分分析。它假设每个数据样本都是独立成分的混合物，并且它的目标是找到这些独立成分。ICA 的核心是“独立”。我们应该先试着理解这一点。

独立在 ICA 的上下文中意味着什么？什么时候我们可以有把握地说两个变量是独立的？与‘关联’有什么不同？最后，你如何衡量独立的程度？

假设 **x** 、 **y** 是两个随机变量，它们的分布函数分别由**P[x]T7、**P[y]给出。如果我们收到一些关于 x 的信息，但这并没有改变我们对 y 的任何知识，那么我们可以有把握地说 x，y 是独立变量。现在你会说“打住，这就是你说的关联缺失”。是的，你是对的，但只是部分对。相关性不是衡量两个变量之间依赖关系的唯一手段。事实上，相关性捕捉的是线性相关性。如果两个变量是独立的，那么线性和非线性相关性都为零。没有线性相关性并不意味着独立，因为可能存在非线性关系。我们举个小例子来理解这一点。****

假设 **x** (-5，-4，-3，-2，-1，0，1，2，3，4，5) &
y = x² 这就给了我们 **y** (25，16，9，4，1，0，1，4，9，16，25)。现在，计算这两个变量之间的相关性。

```py
import numpy as np

x = np.array([-5,-4,-2,-1,0,1,2,3,4,5])
y = np.array([25,16,9,4,1,0,1,4,9,16,25])

np.correlate(x,y)

0.0 
```

正如你所看到的，在上面的例子中，相关性是 0，尽管他们有一个非线性的关系。因此，两个变量之间的独立性意味着零相关，但反之则不然。

让我们回到今天的话题。如前所述，ICA 试图找出构成数据的独立来源。我们将从一个经典的例子开始解释 ICA 及其工作原理。
![ica_toy](img/19f82b5380db59a133e52ab3f6085830.png)
上图所示的爱丽丝和鲍勃，两人同时在说话。两个话筒分别从爱丽丝和鲍勃接收输入 S1 & S2。ICA 假设混合过程是线性的，即它可以表示为矩阵乘法。每个话筒根据矩阵 a 给出的位置和设置混合 S1 & S2。矩阵运算产生矢量 M 作为输出。现在，你想把 S1 的 S2 和 M1 的 M2 分开。这被称为**酒会问题**或**盲源分离**。

如果矩阵 A 已知，这个问题的解是简单的。一个简单的矩阵求逆，然后乘以 M，就会给出答案。但是在现实世界中，矩阵 A 通常是未知的。我们仅有的信息是混合过程的输出。

ICA 解决这个问题的方法基于三个假设。这些是:

1.  混合过程是线性的。
2.  所有源信号都是相互独立的。
3.  所有源信号都具有非高斯分布。

我们已经讨论了前两个假设。让我们来谈谈 ICA 的第三个假设:**源信号的非高斯性**。

这个假设的基础来自于[中心极限定理](https://en.wikipedia.org/wiki/Central_limit_theorem)。根据中心极限定理，独立随机变量之和比独立变量更高斯。所以要推断源变量，我们必须远离高斯。在高斯分布的情况下，不相关的高斯变量也是独立的，这是与高斯分布相关的独特性质。

让我们举一个简单的例子来理解这个概念。首先，创建四个数据集——两个来自高斯分布，两个来自均匀分布。

```py
np.random.seed(100)
U1 = np.random.uniform(-1, 1, 1000)
U2 = np.random.uniform(-1, 1, 1000)

G1 = np.random.randn(1000)
G2 = np.random.randn(1000)

%matplotlib inline
# let's plot our signals

from matplotlib import pyplot as plt

fig = plt.figure()

ax1 = fig.add_subplot(121, aspect = "equal")
ax1.scatter(U1, U2, marker = ".")
ax1.set_title("Uniform")

ax2 = fig.add_subplot(122, aspect = "equal")
ax2.scatter(G1, G2, marker = ".")
ax2.set_title("Gaussian")

plt.show() 
```

![](img/4fa8d1e6b2b1ac0977bffb32353e00fe.png)
现在，混合 U1 & U2 和 G1 & G2 以创建输出 U_mix 和 G_mix。

```py
# now comes the mixing part. we can choose a random matrix for the mixing

A = np.array([[1, 0], [1, 2]])

U_source = np.array([U1,U2])
U_mix = U_source.T.dot(A)

G_source = np.array([G1, G2])
G_mix = G_source.T.dot(A)

# plot of our dataset

fig  = plt.figure()

ax1 = fig.add_subplot(121)
ax1.set_title("Mixed Uniform ")
ax1.scatter(U_mix[:, 0], U_mix[:,1], marker = ".")

ax2 = fig.add_subplot(122)
ax2.set_title("Mixed Gaussian ")
ax2.scatter(G_mix[:, 0], G_mix[:, 1], marker = ".")

plt.show() 
```

![](img/fe38feb307d339a9a384c315c4297d97.png)

U_mix 和 G_mix 是我们在现实世界场景中拥有的。从两种混合物中去除线性相关性。

```py
# PCA and whitening the dataset
from sklearn.decomposition import PCA 
U_pca = PCA(whiten=True).fit_transform(U_mix)
G_pca = PCA(whiten=True).fit_transform(G_mix)

# let's plot the uncorrelated columns from the datasets
fig  = plt.figure()

ax1 = fig.add_subplot(121)
ax1.set_title("PCA Uniform ")
ax1.scatter(U_pca[:, 0], U_pca[:,1], marker = ".")

ax2 = fig.add_subplot(122)
ax2.set_title("PCA Gaussian ")
ax2.scatter(G_pca[:, 0], G_pca[:, 1], marker = ".") 
```

![](img/bb1ef75d518c04f2cd3940a576ed2f47.png)

注意不相关(PCA 均匀，PCA 高斯 2)和源图(均匀，高斯)之间的差异。在高斯的情况下，它们看起来很相似，而不相关的均匀需要旋转才能到达那里。通过去除高斯情况下的相关性，我们实现了变量之间的独立性。如果源变量是高斯型的，则不需要 ICA，PCA 就足够了。

我们如何衡量和消除变量之间的非线性相关性？

变量之间的非线性相关性可以通过变量之间的**互信息**来度量。互信息越高，依赖性就越高。`Mutual information = sum of entropies of marginal distribution - entropy of the joint distribution`
**熵**是分布中不确定性的度量。变量 *x* 的熵由`H(x) = sum(log(P(x))*P(x)) for every possible of value of x`给出。
高斯分布的熵最高。与熵密切相关的一个术语是**负熵**，表述为
`negentropy(x) = H(x_gaussian) - H(x)`。这里 *x_gaussian* 是与 x 具有相同协方差的高斯随机向量，因此，如果 x 是高斯随机变量，则负熵总是非零且等于零。
还有，`mutual information(y1,y2) = constant - sum(negentropy(yi))`

负熵和互信息的计算需要熵的知识。熵计算需要未知的概率分布函数。我们可以用一些合适的函数来近似负熵。一些常见的例子有 tanh(ay)、-exp(-y² )和-y*exp(-y² )。

**伪码 ICA**
*G*&*G*分别为逼近函数及其导数。x 是数据集。

1.  初始化
2.  X =五氯苯甲醚(X)
3.  而 W 变化:
    W = average(X * G(WX))-average(G(W^TX))W
    W =正交化(W)
4.  返回 S = WX

**正交化**是使矩阵的列正交的过程。

应该选择多少个独立元件？应该选择哪些独立组件？
ICA 输出一个列为独立源的源矩阵。它从来没有告诉我们一个组件是重要的还是不相关的。如果列数较少，建议检查每个组件。对于大量组件，应在 PCA 阶段进行选择(2^和步骤)。如果您不熟悉 PCA，请查看本系列的后 1 部分。

让我们在 PySpark 中实现这个算法。我们将创建几个信号，然后将它们混合起来，以获得适合 ICA 分析的数据集。

```py
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
np.random.seed(0)
num_rows = 3000
t = np.linaspace(0,10, n_samples)
# create signals sources
s1 = np.sin(3*t) # a sine wave
s2 = np.sign(np.cos(6*time)) # a square wave
s3 = signal.sawtooth(2 *t) # a sawtooth wave
# combine single sources to create a numpy matrix
S = np.c_[s1,s2,s3]

# add a bit of random noise to each value
S += 0.2 no.random.normal(size = S.shape)

# create a mixing matrix A
A = np.array([[1, 1.5, 0.5], [2.5, 1.0, 2.0], [1.0, 0.5, 4.0]])
X = S.dot(A.T)

#plot the single sources and mixed signals
plt.figure(figsize =(26,12) )
colors = ['red', 'blue', 'orange']

plt.subplot(2,1,1)
plt.title('True Sources')
for color, series in zip(colors, S.T):
    plt.plot(series, color)
plt.subplot(2,1,2)
plt.title('Observations(mixed signal)')
for color, series in zip(colors, X.T):
    plt.plot(series, color) 
```

![](img/74401b68e1a92c79400b065d61249782.png)
对数据集进行 PCA 和白化编码。

```py
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow, BlockMatrix
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.linalg import Vectors, DenseMatrix, Matrix
from sklearn import datasets
# create the standardizer model for standardizing the dataset

X_rdd = sc.parallelize(X).map(lambda x:Vectors.dense(x) )
scaler = StandardScaler(withMean = True, withStd = False).fit(iris_rdd)

X_sc = scaler.transform(X_rdd)

#create the IndexedRowMatrix from rdd
X_rm = IndexedRowMatrix(X_sc.zipWithIndex().map(lambda x: (x[1], x[0])))

# compute the svd factorization of the matrix. First the number of columns and second a boolean stating whether 
# to compute U or not. 
svd_o = X_rm.computeSVD(X_rm.numCols(), True)

# svd_o.V is of shape n * k not k * n(as in sklearn)

P_comps = svd_o.V.toArray().copy()
num_rows = X_rm.numRows()
# U is whitened and projected onto principal components subspace.

S = svd_o.s.toArray()
eig_vals = S**2
# change the ncomp to 3 for this tutorial
#n_comp  = np.argmax(np.cumsum(eig_vals)/eig_vals.sum() > 0.95)+1
n_comp = 3
U = svd_o.U.rows.map(lambda x:(x.index, (np.sqrt(num_rows-1)*x.vector).tolist()[0:n_comp]))
# K is our transformation matrix to obtain projection on PC's subspace
K = (U/S).T[:n_comp] 
```

现在，计算独立分量的代码。

```py
import pyspark.sql.functions as f
import pyspark.sql.types as t

df = spark.createDataFrame(U).toDF("id", "features")

# Approximating function g(y) = x*exp(-x**2/2) and its derivative
def g(X):
    x = np.array(X)
    return(x * np.exp(-x**2/2.0))

def gprime(Y):
    y = np.array(Y) 
    return((1-y**2)*np.exp(-y**2/2.0))

# function for calculating step 2 of the ICA algorithm
def calc(df):

    function to calculate the appoximating function and its derivative
    def foo(x,y):

        y_arr = np.array(y)
        gy = g(y_arr)
        gp = gprime(y_arr)
        x_arr = np.array(x)
        res = np.outer(gy,x_arr)
        return([res.flatten().tolist(), gp.tolist()])

    udf_foo = f.udf(foo, t.ArrayType(t.ArrayType(t.DoubleType())))

    df2 = df.withColumn("vals", udf_foo("features","Y"))

    df2 = df2.select("id", f.col("vals").getItem(0).alias("gy"), f.col("vals").getItem(1).alias("gy_"))
    GY_ = np.array(df2.agg(f.array([f.sum(f.col("gy")[i]) 
                                for i in range(n_comp**2)])).collect()[0][0]).reshape(n_comp,n_comp)/num_rows

    GY_AVG_V  = np.array(df2.agg(f.array([f.avg(f.col("gy_")[i]) 
                                  for i in range(n_comp)])).collect()[0][0]).reshape(n_comp,1)*V

    return(GY_, GY_AVG_V)

np.random.seed(101)
# Initialization
V = np.random.rand(n_comp, n_comp)

# symmetric decorelation function 
def sym_decorrelation(V):

    U,D,VT = np.linalg.svd(V)
    Y = np.dot(np.dot(U,np.diag(1.0/D)),U.T)
    return np.dot(Y,V)

numIters = 10
V = sym_decorrelation(v_init)
tol =1e-3
V_bc = sc.broadcast(V)

for i in range(numIters):

    # Y = V*X
    udf_mult = f.udf(lambda x: V_bc.value.dot(np.array(x)).tolist(), t.ArrayType(t.DoubleType()))
    df = df.withColumn("Y", udf_mult("features"))

    gy_x_mean, g_y_mean_V = calc(df)

    V_new = gy_x_mean - g_y_mean_V

    V_new = sym_decorrelation( V_new )

    #condition for convergence
    lim = max(abs(abs(np.diag(V_new.dot(V.T)))-1))

    V = V_new
    # V needs to be broadcasted after every change
    V_bc = sc.broadcast(V)

    print("i= ",i," lim = ",lim)

    if lim < tol:
        break
    elif i== numIters:
        print("Lower the tolerance or increase the number of iterations")

#calculate the unmixing matrix for dataset         
W = V.dot(K)

#now multiply U with V to get source signals
S_ = df.withColumn("Y", udf_mult("features")) 
```

绘制结果 S_

```py
plt.title('Recovered source Signals')
for color, series in zip(colors, S_.T):
    plt.plot(series, color) 
```

![](img/7e52d11fee86936c44b7cc14523428fd.png)
**ICA 的缺点** : ICA 不能揭示数据集的非线性关系。ICA 没有告诉我们任何关于独立组件的顺序或者它们中有多少是相关的。

**结论**:在这篇文章中，我们学习了独立成分分析的实用方面。我们触及了一些与理解 ICA 相关的重要话题，如高斯性和独立性。之后，在 pyspark 上实现了 ICA 算法，并在玩具数据集上进行了实验。
如果你想了解更多关于 ICA 及其应用的知识，试试[关于 fMRI 和 EEG 数据的 ICA 论文](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2693483/)。

* * *

*本系列的下一篇文章将讨论[多维缩放](https://blog.paperspace.com/dimension-reduction-with-multi-dimension-scaling/)*