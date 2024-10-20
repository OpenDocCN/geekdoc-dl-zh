# 降维- LLE

> 原文：<https://blog.paperspace.com/dimension-reduction-with-lle/>

###### 本教程是关于降维的 7 部分系列的一部分:

1.  [理解主成分分析(PCA)的降维](https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/)
2.  [利用独立成分分析(ICA)深入降低维度](https://blog.paperspace.com/dimension-reduction-with-independent-components-analysis/)
3.  [多维标度(MDS)](https://blog.paperspace.com/dimension-reduction-with-multi-dimension-scaling/)
4.  **对[对](https://blog.paperspace.com/dimension-reduction-with-lle)**
5.  t-SNE
6.  [IsoMap](https://blog.paperspace.com/dimension-reduction-with-isomap)
7.  [Autoencoders](https://blog.paperspace.com/dimension-reduction-with-autoencoders)

(带有数学和代码(python 和 pyspark)的 jupyter 笔记本可在 [github repo](https://github.com/asdspal/dimRed) 上获得)

LLE 是一种拓扑保持的流形学习方法。所有流形学习算法都假设数据集位于低维的光滑非线性流形上，并且映射***f:R^D->R^D***(D>>D)可以通过保留高维空间的一个或多个属性来找到。拓扑保留意味着邻域结构是完整的。不像，像 MDS 拓扑保持这样的距离保持方法更直观。高维空间中相邻的点在低维空间中应该是靠近的。《LLE》背后的灵感来源于原创作者所说的“放眼全球，立足本地”。像 SOM(自组织映射)这样的方法也是拓扑保持的，但是它们为下流形假设了预定义的格。LLE 根据数据集中包含的信息创建格网。上图显示了一个模拟数据集及其 LLE 嵌入。模拟 S 曲线的几个点被圈起来。他们是同一点的邻居。我们可以看到，在 LLE 嵌入中，它们被映射得彼此靠近(圆中的点)。这显示了 LLE 在保存邻里结构方面的能力。LLE 几乎没有做出什么重要的假设。这些是:

1.  **数据采样良好，即数据集密度高**。
2.  **数据集位于光滑流形上**。
    数据的良好采样意味着对于每个点，在其邻域内至少有 2d 个点。如果流形是光滑的，我们可以假设该点及其邻域位于局部线性流形上。如果有急弯或洞，这个属性就不成立。

## LLE 的实施

必要的进口。

```py
#necessary imports

from sklearn import datasets
from pyspark.sql import SQLContext as SQC
from pyspark.mllib.linalg import Vectors as mllibVs, VectorUDT as mllibVUDT
from pyspark.ml.linalg import Vectors as mlVs, VectorUDT as mlVUDT
from pyspark.sql.types import *
from pyspark.mllib.linalg import *
from pyspark.sql.functions import *
from pyspark.ml.feature import StandardScaler
from pyspark.mllib.linalg.distributed import IndexedRowMatrix
import math as m
import numpy as np 
```

###### LLE 的脚步:

1.  创建邻域图
2.  对于每个点，计算局部权重矩阵 W
3.  对于每个点，使用 W 从它的邻居创建点。
    ![](img/47813a2dd38f43f230d77a6dcb703bcd.png)

**LLE 的第一步:邻居图**:
LLE 的第一步是创建邻居图。需要距离度量来测量两点之间的距离并将它们归类为邻居。例如欧几里德、马哈拉诺比斯、汉明和余弦。

**邻域矩阵**:可以使用***e*-邻域**、**K-最近邻居**、**位置敏感哈希**创建邻域矩阵。
***e*-邻域**:如果节点[i] &节点[j] 之间的距离小于一个固定参数 ***e*** ，则在它们之间创建一条边。 ***e*** 既不能小，也不能大。如果 ***e*** 大，则每个节点都有一条边与其他节点相连，而它的小很多节点都没有邻居。
**K-最近邻**:使用这种方法，对于每个节点，我们选择 K 个最近的数据点作为它的邻居。这种方法确保每个节点都有 K 个邻居。如果数据点的密度变化很大，这种方法将产生不对称的邻域图。例如，节点[i] 可能被远处的点选择作为其邻居，因为该点的密度较低。
**局部敏感散列**:与上述两种方法不同，局部敏感散列是一种选择邻居的近似方法。通俗地说，位置敏感散列使用一组散列函数来散列每个点，并且被散列到相同桶的所有点被分类为邻居。如果你想更多地了解 LSH，网上有很多好的教程。

```py
# for every point create an id sorted vector of distance from every other point

num_samples = 3000
X, color = datasets.make_s_curve(num_samples)

df = (spark.createDataFrame(sc.parallelize(X.tolist()).zipWithIndex().
                          map(lambda x: (x[1], mlVs.dense(x[0]))), ["id", "features"]))    

udf_dist = udf(lambda x, y:  float(x.squared_distance(y)), DoubleType())

df_2 = df

df = df.crossJoin(df ).toDF('x_id', 'x_feature', 'y_id', 'y_feature')
df = df.withColumn("sim", udf_dist(df.x_feature, df.y_feature))

df = df.drop("x_feature")

st = struct([name for name in ["y_id", "sim","y_feature"]]).alias("map")
df = df.select("x_id", st)
df  = df.groupby("x_id").agg(collect_list("map").alias("map"))
df = df.join(df_2, df_2.id == df.x_id, "inner").drop("x_id") 
```

**LLE 第二步** : *利用其邻居对一个点进行线性加权重建* :
数据集的每个点都被重建为其邻居的线性加权和。因为只有邻居参与重建，所以重建是地方性的。重构是通过权重的线性系数实现的，因此是线性的。这就是为什么这种方法被称为局部线性嵌入。
点 P[i] 和 P[j] 的权重相互独立。
一个点及其邻居的旋转、重缩放和平移都不会影响该点的局部 W 矩阵。
重要的是要注意，在求解 W 时，如果邻居的数量大于原始维度 D，一些权重系数可能为零，从而导致多个解。这个问题可以通过向惩罚大权重的最小二乘问题添加正则化来处理。
对于每个点 Y[i] :
创建一个矩阵 *Z* 与 Y[i]
的所有邻居从 *Z*
中减去 Y[i] 创建局部协方差矩阵 *C* = ZZ^T
添加一个正则项以避免 C 奇异， C = C+reg * I
solve*CW*= 1 for*W*
set*W[ij]*= 0 如果 j 不是 I
set*W*=*W*/sum(*W【T47)$*

```py
# calculate local neighborhood matrix

def get_weights(map_list, k, features, reg):

    sorted_map = sorted(map_list, key = lambda x: x[1])[1:(k+1)]

    neighbors = np.array([s[2] for s in sorted_map])
    ind = [s[0] for s in sorted_map]

    nbors, n_features = neighbors.shape

    neighbors_mat = neighbors - features.toArray().reshape(1,n_features)

    cov_neighbors = np.dot(neighbors_mat, neighbors_mat.T)

    # add regularization term
    trace = np.trace(cov_neighbors)
    if trace > 0:
        R = reg * trace
    else:
        R = reg

    cov_neighbors.flat[::nbors + 1] += R

    weights = linalg.solve(cov_neighbors, np.ones(k).T, sym_pos=True)
    weights = weights/weights.sum()

    full_weights = np.zeros(len(map_list))
    full_weights[ind] = weights

    return(Vectors.dense(full_weights))

udf_sort = udf(get_weights, mllibVUDT())

df = df.withColumn("weights", udf_sort("map", lit(10), "features", lit(0.001))) 
```

**LLE 第三步:重建低维点** :
在这一步，我们不需要数据集。现在，我们必须使用它的邻居和局部 W 矩阵在低维中创建每个点。邻域图和局部权重矩阵捕捉流形的拓扑。
低维再造误差被约束，使其适定。等式 1(在 LLE 的图像*步骤中)约束输出以原点为中心，使其平移不变。需要等式 2 来使输出旋转和缩放不变。*

```py
# udf for creating a row of identity matrix 
def I(ind):
    i = [0]*150
    i[ind]=1.0
    return(mllibVs.dense(i))

# convert dataframe to indexedrowmatrix
weights_irm = IndexedRowMatrix(df.select(["id","weights"]).rdd.map(lambda x:(x[0],  I(x[0])-x[1])))
M = weights_irm.toBlockMatrix().transpose().multiply( weights_irm.toBlockMatrix() )

SVD = M.toIndexedRowMatrix().computeSVD(150, True)

# select the vectors for low dimensional embedding
lle_embbeding = np.fliplr(SVD.V.toArray()[:,-(d+1):-1]) 
```

绘制结果嵌入
![](img/09204a3a50b1bfb56e054239484d3b76.png)

LLE 之后 MNIST 数据集子集的可视化。
![](img/b9e7e66cff06804011c27b294682f53d.png)

**LLE 的缺点** :
LLE 对离群值和噪声很敏感。数据集具有变化的密度，并且不可能总是具有平滑的流形。在这些情况下，LLE 给出了一个糟糕的结果。

**结论**:在本文中，我们讨论了与理解 LLE 及其实施相关的基本和重要概念。后来，我们在 pyspark 上实现了一个标准的 LLE 算法。黑森 LLE，修改 LLE，和 LTSA 地址 LLE 的一些缺点。

我们将在下一篇文章中讨论 t-SNE。