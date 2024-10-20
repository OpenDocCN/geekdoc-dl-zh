# 多维标度(MDS)

> 原文：<https://blog.paperspace.com/dimension-reduction-with-multi-dimension-scaling/>

### **本教程来自关于降维的 7 部分系列:**

1.  [理解主成分分析(PCA)的降维](https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/)
2.  [利用独立成分分析(ICA)深入降低维度](https://blog.paperspace.com/dimension-reduction-with-independent-components-analysis/)
3.  **[【MDS】](https://blog.paperspace.com/dimension-reduction-with-multi-dimension-scaling/)**
4.  LLE *(即将推出！)*
5.  t-SNE *(即将推出！)*
6.  IsoMap *(即将推出！)*
7.  自动编码器*(即将推出！)*

(在 [github](https://github.com/asdspal/dimRed) 上有一个 Jupyter 笔记本，里面有数学和代码(python 和 pyspark)。)

**多维标度**是一种距离保持的流形学习方法。所有流形学习算法都假设数据集位于低维的平滑、非线性流形上，并且可以通过保留高维空间的一个或多个属性来找到映射***f:R^D->R^D***(D>>D)。距离保持方法假设流形可以由它的点的成对距离来定义。在距离保持方法中，低维嵌入是以这样的方式从较高维度获得的，即点之间的成对距离保持相同。一些距离保持方法保持空间距离(MDS ),而一些保持图形距离。

MDS 不是单一的方法，而是一系列方法。MDS 采用相异度矩阵 *D* ，其中 D[ij] 表示点 *i* 和 *j* 之间的相异度，并在较低维度上产生映射，尽可能接近地保留相异度。相异矩阵可以从给定的数据集中观察或计算。MDS 在社会学、人类学等人文科学领域，尤其是心理测量学领域广受欢迎和发展。

让我们用 MDS 的书中的一个例子来更好地理解它。下表显示了 1970 年美国不同类型犯罪率之间的相互关系，图片显示的是 MDS 嵌入。随着变量数量的增加，人类的大脑越来越难识别变量之间的关系。
T3


图中各点的相对位置取决于它们在相关表中的不同之处，即具有高相关性的犯罪率彼此靠近，而不具有高相关性的犯罪率彼此远离。从图中我们可以看出，横向犯罪分布可以解释为“暴力犯罪对财产犯罪”，而纵向犯罪分布可以解释为“街头犯罪对隐蔽犯罪”。

MDS 可以分为两类:

*   **度量 MDS** -度量 MDS 用于定量数据，并试图保留原始的相异度量。给定一个相异矩阵 *D* ，一个单调函数 *f* ，p(子空间维数)度量 MDS 试图找到一个最优配置 x⊂r^ps . t . f(d[ij])≈d[ij]=(x[I]—x[j])²。公制 MDS 的另一个版本是经典的 MDS(原 MDS)公式，它提供了封闭形式的解决方案。它不试图在低维中逼近相异度，而是在解中使用特征值分解。

*   **非公制 MDS** -非公制 MDS 用于序数数据。它试图保持不相似性度量的顺序不变。例如，如果 P[ij] 在 i[th] & j[th] 和 P[32] > P[89] 之间不同，那么非公制 mds 创建一个映射 s . t . d[32]>d[89]。

我们将使用 SMACOF(通过复杂函数的优化来缩放)算法来实现度量 MDS。在深入研究度量 MDS 的实现之前，我们需要了解一些关于优化的 MM(优化-少数化)算法。

**求函数最优的 MM**

MM 算法是一种求复杂函数最优的迭代算法。假设我们有一个函数 *f(x)* ，我们需要找到它的最小值。MM 不是直接优化 *f(x)* 而是使用一个近似函数 *g(x，x[m] )* 来寻找一个最优值。如果问题是寻找 *f(x)* 的最小值，那么 *g(x，x[m] )* 称为优化函数，否则称为优化函数，而 *x[m]* 称为支撑点。
如果 *g(x，x[m] )* 是 *f(x)* 的优化函数，则必须满足以下条件:

1.  优化 *g(x，x[m] )* 应该比 *f(x)* 容易。
2.  对于任意 *x* ， *f(x)* ≤ *g(x，x[m] )*
3.  f(x[m] )=g(x[m] ，x[m]

**MM 算法的步骤**:

1.  选择一个随机支撑点 x[m]
2.  求 x[min]= arg min[x]g(x，x[m]
3.  如果 f(x[min])—f(x[m])≈*e*break(其中 *e* 是一个很小的数)否则转到步骤 4
4.  设 x[m] =x[min] 进入第二步
    我们用一个例子来理解 MM 算法。
    ![](img/c389e66b0d0fda5b18b14117b9350530.png)
    绿色的图是我们需要求最小值的函数。每幅图像代表算法的一次迭代，第一幅图像作为初始设置。我们的初始支点是 x[m] =9.00，g(x，9)的最小值在 x[min] =6.49。现在，如果我们移动到下一幅图像，我们会看到 x[m] 现在是 6.49，并且我们基于 g(x，6.49)得到了新的 x[min] =6.20。如果我们移动到下一次迭代，x[m] 变为 6.20，x[min] 值变为 5.84。继续后续迭代，我们看到最小值向绿色图的最小值移动(如下图所示)。MM 算法最重要的部分是找到一个好的逼近函数。
    ![](img/e7297f476d341772813fd64f3f740182.png)
    现在，让我们转到度量 MDS 的 SMACOF 算法。如前所述，度量 MDS 试图逼近相异矩阵并最小化由
    σ(X)=σ[ij]w[ij](δ[ij]d[ij](X))²给出的应力函数， 其中
    w[ij] 是分配给 *i* 和*j*δ[ij]是给定相异度矩阵
    d[ij] 的元素(X)是我们需要找到的 *X* 的相异度
    我们不去深究应力函数的支配函数的推导。 如果你想了解，请查阅这本优秀的书(主题-优化压力，第 187 页)。

**MDS 的脚步**

1.  创建一个不相似矩阵。
2.  选择任意一点 X[m] 作为支点。
3.  求应力优化函数的最小值。
4.  如果σ(X[m])—σ(X[min])<*e*break else 设置 X[m] =X[min] 并转到步骤 2

**第一步——相异度矩阵** :
我们需要一个距离度量来计算数据集中两点之间的距离。让我们快速浏览几个流行的距离度量。
*欧几里德度量*:这是距离测量中最常用的度量。它等于两点之间的直线距离。
*曼哈顿公制*:两点之间的距离是沿轴线的直角方向测量的。它也被称为直线距离、出租车公制或城市街区距离。
*余弦度量*:余弦距离测量两个向量之间角度的余弦。余弦值越小，点越近。
*Mahalanobis 度量* : Mahalanobis 度量用于测量点 p 到分布 d 的距离，在检测异常值时特别有用。
*汉明度量*:汉明度量统计两个向量中条目不同的地方的数量。它是信息论中的一个重要度量。
我们将使用欧几里德度量作为相异度度量。

```py
from sklearn import datasets
import math as ma
import numpy as np
from pyspark.sql import types as t
from pyspark.sql import functions as f

digits = datasets.load_digits(n_class=6)

data = digits.data
# repartitioning the dataframe by id column will speed up the join operation 

df = spark.createDataFrame(sc.parallelize(data.tolist()).zipWithIndex()).toDF("features",
                   "id").repartition("id")
df.cache()

euclidean = lambda x,y:ma.sqrt(np.sum((np.array(x)-np.array(y))**2))
data_bc = sc.broadcast(df.sort("id").select("features").rdd.collect())

# create the distance metric
def pairwise_metric1(y):
    dist = []
    for x in data_bc.value:
        dist += [ma.sqrt(np.sum((np.array(x)-np.array(y))**2))]

    return(dist)

udf_dist1 = f.udf(pairwise_metric1, t.ArrayType(t.DoubleType()))

df = df.withColumn("D", udf_dist1("features")) 
```

**第二步:SCAMOF 算法**:

```py
n,p = data.shape
dim = 2
X = np.random.rand(n,dim)

# randomly initialize a solution for the pivot point.
dfrand = spark.createDataFrame(sc.parallelize(X.tolist()).zipWithIndex()).toDF("X", 
                     "id2").repartition("id2")
df = df.join(dfrand, df.id==dfrand.id2, "inner").drop("id1")

def pairwise_metric2(y):
    dist = []
    for x in X_bc.value:
        dist += [ma.sqrt(np.sum((np.array(x)-np.array(y))**2))]
    return(dist)

# create the matrix B
def B(id,x,y):

    y,x = np.array(y), np.array(x) 
    y[y==0.0] = np.inf
    z = -x/y

    z[id] = -(np.sum(z)-z[id])
    return(z.tolist())

# function for matrix multiplication using outer multiplication
def df_mult(df, col1, col2, n1, n2, matrix=True):

    udf_mult = f.udf(lambda x,y:np.outer(np.array(x), 
                  np.array(y)).flatten().tolist(),
                   t.ArrayType(t.DoubleType()))

    df = df.withColumn("mult", udf_mult(col1, col2))
    df = df.agg(f.array([f.sum(f.col("mult")[i]) 
             for i in range(n1*n2)])).toDF("mult")
    if not matrix:
        return(df)
    st = t.ArrayType(t.StructType(
                [t.StructField("id",t.LongType()),
                 t.StructField("row",t.ArrayType(
                 t.DoubleType()))]))
    udf_arange = (f.udf(lambda x:[(i,j.tolist()) 
                  for i,j in enumerate(np.array(x).
                       reshape(n1,n2)/n1)], st))

    df = (df.withColumn("mult", 
               udf_arange("mult")).select(
               f.explode("mult").alias("mult")))

    df = (df.select(f.col("mult.id").alias("id2"),
                      f.col("mult.row").
                      alias("X_min")).
                      repartition("id2"))
    return(df)

udf_B = f.udf(B, t.ArrayType(t.DoubleType()))
udf_sigma = (f.udf(lambda x,y: float(np.sum((
                 np.array(x)-np.array(y))**2)), 
                 t.DoubleType()))
sigma_old = np.inf
tol = 1e-4
max_iter = 1000

for i in range(max_iter):
    X_bc = sc.broadcast(df.sort("id").select("X").rdd.collect())
    def pairwise_metric2(y):
        dist = []
        for x in X_bc.value:
            dist += [ma.sqrt(np.sum((np.array(x)-np.array(y))**2))]
        return(dist)
    udf_dist2 = f.udf(pairwise_metric2, t.ArrayType(t.DoubleType()))
    df = df.withColumn("di", udf_dist2("X"))

    df = df.withColumn("sigma", udf_sigma("D","di"))
    sigma_new = df.agg({"sigma":"sum"}).collect()[0][0]
    print(sigma_old, sigma_new)
    sigma_old = sigma_new
    df = df.withColumn("B", udf_B("id","D","di")).drop("di")

    X_min = df_mult(df, "B", "X", n, dim)

    df = df.join(X_min, df.id==X_min.id2).select("id", "D", f.col("X_min").alias("X"))
    # cache action will prevent recreation of dataframe from base
    df.cache() 
```

MDS 嵌入的情节。
![](img/fe9c86ebd6845cf9c46c1e7ca0108382.png)
虽然这些簇不很明显，但还是很容易辨认出来。
**MDS 的缺点** :
MDS 每次迭代计算相异度矩阵都需要很大的计算能力。很难将新数据嵌入 MDS。

**结论**:和 PCA 一样，MDS 是一种古老的方法。它已经被很好地研究过了。它几乎没有像 sammon mapping 那样的扩展。通过这篇文章，我们试图加深对 MDS 及其运作的理解。我们复习了一些与 MDS 相关的领域，并在 pyspark 中实现了一个基本的 MDS。
本系列的下一篇文章将在*[先睹为快！](https://blog.paperspace.com/p/a6ee6e43-8af7-4de4-85fc-5bc8d90c789e/)*