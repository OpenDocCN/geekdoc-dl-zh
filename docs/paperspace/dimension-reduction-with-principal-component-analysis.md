# 用主成分分析(PCA)理解降维

> 原文：<https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/>

###### 本教程是关于降维的 7 部分系列的一部分:

1.  **[用主成分分析理解降维](https://blog.paperspace.com/pca/)**
2.  [利用独立成分分析(ICA)深入降低维度](https://blog.paperspace.com/ica/)
3.  [多维标度(MDS)](https://blog.paperspace.com/dimension-reduction-with-multi-dimension-scaling/)
4.  LLE *(即将推出！)*
5.  t-SNE *(即将推出！)*
6.  IsoMap *(即将推出！)*
7.  自动编码器*(即将推出！)*

#### 维度的诅咒

**大数据分析**是如今的流行语。每个人都在谈论它。大数据分析已经在许多领域得到应用，如医学、政治、约会。虽然大数据分析被用于改善人类生活的许多方面，但它也有自己的问题。其中之一就是“**维度诅咒**”。维度诅咒是指大量维度导致的数据规模的指数级增长。随着数据维数的增加，处理数据变得越来越困难。降维是解决维数灾难的一种方法。通俗地说，降维方法是通过提取相关信息，将其余数据作为噪声处理，从而降低数据的大小。
通过一系列的帖子，我们将学习并实现使用大数据框架 pyspark 的降维算法。
本系列的第一篇文章将在 PCA 上发表。

## 主成分分析

(有更数学化的笔记本有 python 和 pyspark 代码可用 [github repo](https://github.com/asdspal/dimRed) )
主成分分析(PCA)是最流行的线性降维之一。有时，它单独使用，有时作为其他降维方法的起始解决方案。PCA 是一种基于投影的方法，它通过将数据投影到一组正交轴上来变换数据。

让我们对 PCA 有一个直观的了解。假设，您希望根据不同食物的营养成分来区分它们。哪一个变量将是区分食物项目的好选择？如果你选择一个变量，这个变量从一种食物到另一种食物变化很大，你将能够正确地分离它们。如果在食品中选择的变量几乎相同，你的工作将会困难得多。如果数据没有一个适当分离食物项目的变量会怎样？我们可以通过
`artVar1 = 2 X orgVar1 - 3 X orgVar2 + 5 X orgVar3`这样的原始变量的线性组合来创建一个人工变量。
这就是主成分分析的本质，它寻找原始变量的最佳线性组合，从而使新变量的方差或分布最大化。

现在，让我们通过一个动画来了解 PCA 是如何达到上述目的的。
![](img/b3765d6e8851a2c81051ccb30cbff853.png)
<video controls="" style="position: relative; left: 15%"><source src="https://s3-us-west-2.amazonaws.com/articles-dimred/pca/animation.webm"> 
图上的每个蓝点代表一个由 x & y 坐标给出的数据点。从数据集的中心，即从 x & y 的平均值，画出一条线 **P** (红线)。图上的每个点都投影在这条线上，由两组红色&绿色的点表示。沿着线 p 的数据分布或方差由两个大红点之间的距离给出。当线 p 旋转时，两个红点之间的距离根据线 p 与 x 轴形成的角度而变化。连接一个点和它的投影的紫线代表当我们用它的投影近似一个点时产生的误差。PCA 从旧变量中创造新变量。如果新变量非常接近旧变量，那么近似误差应该很小。所有紫色线长度的平方和给出了近似值的总误差。最小化误差平方和的角度也最大化红点之间的距离。最大展开方向称为**主轴**。一旦我们知道了主轴，我们减去沿主轴的方差，得到剩余的方差。我们应用同样的程序从剩余方差中寻找下一个主轴。除了是方差最大的方向外，下一个主轴必须与其他主轴正交。
一旦我们得到所有主轴，数据集就被投影到这些轴上。投影或转换的数据集中的列被称为**主成分**。

幸运的是，多亏了线性代数，我们不必为 PCA 费太多力气。线性代数中的特征值分解和奇异值分解是 PCA 中的两个主要步骤。

## 特征值分解，特征向量，特征值

特征值分解是一种适用于半正定矩阵的矩阵分解算法。在 PCA 的上下文中，特征向量表示方向或轴，相应的特征值表示沿该特征向量的方差。特征值越高，沿特征向量的方差就越大。
![](img/96588fd5bd880209965da1fb94f6891a.png)
上图为一个正定矩阵的特征值分解**一个**。 **Q** 是列为特征向量的正交矩阵**λ**是以特征值为对角元素的对角矩阵。

## 奇异值分解

SVD 是一种矩阵分解方法，它将矩阵表示为秩为 1 的矩阵的线性组合。奇异值分解比主成分分析更稳定，并且不需要正定矩阵。
![](img/41e3a29f07ffcd49369fcf2fb183b12a.png)
如图 SVD 产生三个矩阵 U，S & V. U 和 V 正交矩阵，它们的列分别代表 AA^T 和 A^T A 的特征向量。矩阵 S 是对角矩阵，对角值称为奇异值。每个奇异值是相应特征值的平方根。

降维如何适应这一切？一旦你计算出特征值和特征向量，选择重要的特征向量来组成一组主轴。

## 特征向量的选择

我们如何选择要保留的主轴数？应该选择哪些主轴？
一个特征向量的重要性由相应特征值所解释的总方差的百分比来衡量。假设`V1` & `V2`为两个特征向量，`40%` & `10%`分别为沿其方向的总方差。如果让我们从这两个特征向量中选择一个，我们会选择`V1`,因为它给了我们更多关于数据的信息。
所有特征向量按照特征值降序排列。现在，我们必须决定保留多少特征向量。为此，我们将讨论两种方法**总方差解释**和**碎石图**。

**总差异解释**

假设，我们有一个向量的`n`个特征值(e[0] ，...，e[n] )按降序排序。取每个指数的特征值的累积和，直到总和大于总方差的`95%`。拒绝该索引之后的所有特征值和特征向量。

**碎石图**

同样，我们必须按降序排列特征值。绘制特征值与指数的关系图。您将得到如下图所示的图表。
![](img/c41a507beffe1ec49e65601a5696629b.png)
理想的平地是一条陡峭的曲线，其后是一个急转弯和一条直线。舍弃急转弯后的所有特征值及其对应的特征向量。例如，在上面显示的图像中，急弯位于 4。所以，主轴数应该是 4。

**py spark 中的 PCA**

让我们在 pyspark 中实现 PCA 算法。
熟悉数据。

```py
#read the dataset and plot a scatter graph between 1st and 2nd variable
import matplotlib.pyplot as plt
iris = datasets.load_iris()
data = iris.data
target = iris.target
setosa = data[target==0]
versicolor = data[target==1]
verginica = data[target==2]
plt.scatter(setosa[:,0], setosa[:,1], c="b",label="setosa")
plt.scatter(versicolor[:,0], versicolor[:,1], c="g",label="versicolor")
plt.scatter(verginica[:,0], verginica[:,1], c="r",label="verginica") 
```

![](img/71ea818e966593d510f2300ca505d50e.png)

将 numpy 数组转换成 spark 数据帧。

```py
# necesary imports

from pyspark.mllib.linalg.distributed import 
IndexedRowMatrix, IndexedRow
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as f

# numpy array -> rdd -> dataframe

rdd = sc.parallelize(iris_data.tolist()).zipWithIndex()
iris_df = 
spark.createDataFrame(rdd).toDF("features","id")
n = rdd.count()
p = len(rdd.take(1)[0][0])

# change the data type of features to vectorUDT from array[double]

udf_change = f.udf(lambda x: Vectors.dense(x), VectorUDT())

iris_df = iris_df.withColumn("features", udf_change("features")) 
```

**对数据进行预处理**——标准化把所有变量都带到了同一水平。

```py
# create the standard scaler model
stdScaler = StandardScaler(withMean = True, withStd = True, inputCol="features", outputCol="scaled_features")
#fit the model on the dataset    
model = stdScaler.fit(iris_df)
# transform the dataset   
iris_std_df = model.transform(iris_df).drop("features").withColumnRenamed("scaled_features","features") 
```

IndexedRowMatrix 是一个按行索引的分布式矩阵。将数据帧转换为 IndexedRowMatrix，并计算主成分。我们将使用奇异值分解进行矩阵分解。

```py
# now create the indexed row matrix 

iris_irm = IndexedRowMatrix(iris_std_df.rdd.map(lambda x: IndexedRow(x[0], x[1].tolist()))) 
```

computesvd 函数接受两个参数，一个整数和一个布尔值。整数参数给出了要保留的奇异值的数量。因为我们不知道它的值，所以它等于维数。布尔参数声明是否计算 u。

```py
SVD = iris_irm.computeSVD(p, True) 

U = SVD.U
S = SVD.s.toArray()

# compute the eigenvalues and number of components to retain
eigvals = S**2/(n-1)
eigvals = np.flipud(np.sort(eigvals))
cumsum = eigvals.cumsum() 
total_variance_explained = cumsum/eigvals.sum()
K = np.argmax(total_variance_explained>0.95)+1

# compute the principal components
V = SVD.V
U = U.rows.map(lambda x: (x.index, x.vector[0:K]*S[0:K]))
princ_comps = np.array(list(map(lambda x:x[1], sorted(U.collect(), key = lambda x:x[0])))) 
```

画出合成的主成分

```py
setosa = princ_comps[iris_target==0]
versicolor = princ_comps[iris_target==1]
verginica = princ_comps[iris_target==2]
plt.scatter(setosa[:,0], setosa[:,1], c="b",label="setosa")
plt.scatter(versicolor[:,0], versicolor[:,1], c="g",label="versicolor")
plt.scatter(verginica[:,0], verginica[:,1], c="r",label="verginica") 
```

![](img/b23d747f15591d2615bf0811b4448a13.png)

PCA 清楚地呈现了数据集的更好的图像。使用主成分分析对 mnist 数据集的子集进行可视化。
![](img/70438cd4f14b638687473baa1cfbc44a.png)
PCA 能够更准确地区分数字。零、一和四被清楚地分组，而 PCA 发现很难区分二、三和五。

**PCA 的缺点** -如果变量的数量很大，解释主成分变得很困难。当变量之间具有线性关系时，PCA 是最合适的。此外，PCA 容易受到大的异常值的影响。

**结论** : PCA 是一种古老的方法，已经得到了很好的研究。基本主成分分析有许多扩展，解决了它的缺点，如鲁棒主成分分析，核主成分分析，增量主成分分析。
通过这篇文章，我们对 PCA 有了一个基本而直观的了解。我们讨论了与实现 PCA 相关的几个重要概念。后来，我们使用 pyspark 在一些真实数据集上实现了 PCA。如果你想更深入，尝试在更大的数据集上实现 PCA 的一些扩展。

我们将在本系列的下一篇文章中讨论另一种线性降维方法 [ICA](https://blog.paperspace.com/p/dd5e5c1d-b93d-4779-ac40-9b4d9e1383dd/) 。