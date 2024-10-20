# 降维- t-SNE

> 原文：<https://blog.paperspace.com/dimension-reduction-with-t-sne/>

###### 本教程是关于降维的 7 部分系列的一部分:

1.  [理解主成分分析(PCA)的降维](https://blog.paperspace.com/dimension-reduction-with-principal-component-analysis/)
2.  [利用独立成分分析(ICA)深入降低维度](https://blog.paperspace.com/dimension-reduction-with-independent-components-analysis/)
3.  [多维标度(MDS)](https://blog.paperspace.com/dimension-reduction-with-multi-dimension-scaling/)
4.  【T0 向】T1
5.  **[t-SNE](https://blog.paperspace.com/dimension-reduction-with-t-sne)**
6.  [IsoMap](https://blog.paperspace.com/dimension-reduction-with-isomap)
7.  [Autoencoders](https://blog.paperspace.com/dimension-reduction-with-autoencoders)

(一个更数学化的带代码的笔记本是可用的 [github repo](https://github.com/asdspal/dimRed) )

t-SNE is a new award-winning technique for dimension reduction and data visualization. t-SNE not only captures the local structure of the higher dimension but also preserves the global structures of the data like clusters. It has stunning ability to produce well-defined segregated clusters. t-SNE is based on stochastic neighbor embedding(SNE). t-SNE was developed to address some of the problems in SNE. So let's have a basic understanding of SNE.
**SNE**: stochastic neighbor embedding uses a probabilistic approach to embed a high dimension dataset into lower dimension by preserving the neighborhood structure of the dataset. A Gaussian probability distribution centered on each point is defined over all the potential neighbors of this point. SNE aims to minimize the difference in probability distribution in the higher dimension and lower dimension.
For each object, *i* and it's neighbor *j* , we compute a P[i|j] which reflects the probability that *j* is neighbor of *i*
P[i|j] = exp(−d²[ij])Σ[k≠i]exp(−d²[ij])) where d²[ij] is the dissimilarity between element *i* and *j* given as input or calculated from the dataset provided.
The dissimilarity between x[i] and x[j] can be calculated using the following formula
d²[ij] = ||x[i]−x[j]||² / (2σ²[i]), where σ[i] generally calculated through a binary search by equating the entropy of the distribution centered at x[i] to perplexity which is chosen by hand. This method generates a probability matrix which is asymmetric.
Now, a random solution is chosen as the starting point for the low dimensional embedding. A probability distribution is defined on it in the same way as done above but with a constant σ=0.5 for all points.
SNE tries to minimize the difference between these two distributions. We can calculate the difference between two distributions using Kullback-Liebler divergence. For two discrete distirbution *P* and *Q* KL divergence is given by DKL(P||Q)=Σ[i]P[i](P[i]/Q[i]).
SNE defines a cost function based of the difference between p[ij] and q[ij] which is given by
C=Σ[i]Σ[j](P[ij]log(P[ij]/q[ij]))
While embedding the dataset in lower dimension, two kinds of error can occur, first neighbors are mapped as faraway points( p[ij] is large and q[ij] is small) and points which are far away mapped as neighbors( p[ij] is small while q[ij] is large).Look closely at the cost function, the cost of the first kind of error i.e. mapping large P[ij] with small q[ij] is smaller than the cost while mapping small p[ij] as large q[ij] . SNE heavily penalizes if the neighbors are mapped faraway from each other.
Some of the shortcomings of SNE approach are asymmetric probability matrix *P*, crowding problem. As pointed out earlier the probability matrix *P* is asymmetric. Suppose a point X[i] is far away from other points, it's P[ij] will be very small for all *j*. So, It will have little effect on the cost function and embedding it correctly in the lower dimension will be hard.
Any n-dimensional Euclidean space can have an object with n+1 or less equidistant vertices not more than that. Now, when the intrinsic dimension of a dataset is high say 20, and we are reducing its dimensions from 100 to 2 or 3 our solution will be affected by crowding problem. The amount of space available to map close points in 10 or 15 dimensions will always be greater than the space available in 2 or 3 dimensions. In order to map close points properly, moderately distant points will be pushed too far. This will eat the gaps in original clusters and it will look like a single giant cluster.
We need to brush up few more topics before we move to t-SNE.
**Student-t distribution** -- Student-t distribution is a continuous symmetric probability distribution function with heavy tails. It has only one parameter degree of freedom. As the degree of freedom increases, it approaches the normal distribution function. When degree of freedom =1, it takes the form of Cauchy distribution function and its probability density function is given by
f(t)=1/π(1+t2)
**Entropy**: Entrophy is measure of the average information contained in a data. For a variable *x* with pdf p(x), it is given by
*H(x) = −Σ[i](p(x[i]) × log[2](p(x[i])))*
**Perpexility**: In information theory, perplexity measures how good a probability distribution predicts a sample. A low perplexity indicates that distribution function is good at predicting sample. It is given by
*Perpx(x)=2^H(x)*, where *H(x)* is the entropy of the distribution.

t-SNE
t-SNE 在两个方面不同于 SNE，首先，它使用学生 t 分布来衡量低维中 Y[i] 和 Y[j] 之间的相似性，其次，对于高维，它使用对称概率分布，使得 P[ji] =P[ij] 。

**t-SNE 算法步骤**:

1.  为每个 *i* 和 *j* 计算成对相似度 P[ij] 。
2.  使 P[ij] 对称。
3.  选择一个随机解 Y[0]
4.  未完成时:
    计算 Y 的两两相似度[I]计算梯度
    如果 i > max_iter break
    否则
    i = i+1

**计算概率分布** :
为了计算成对相似性，我们需要知道以 x[i] 为中心的高斯分布的方差σ[i] 。有人可能会想，为什么不为每个 x[i] 设置一个σ[i] 的单一值。数据的密度可能会有所不同，对于密度较高的地方，我们需要较小的σ[i] ，而对于点距离较远的地方，我们需要较大的σ[i] 。以 x[i] 为中心的高斯分布的熵随着σ[i] 的增加而增加。为了获得σ[i] ，我们需要执行二分搜索法，使得以 x[i] 为中心的高斯分布的困惑度等于用户指定的困惑度。现在，如果你在思考困惑是如何融入这一切的。你可以把困惑看作是邻居数量的一种度量。

**参数对嵌入的影响** :
为了使 t-SNE 有意义，我们必须选择正确的困惑值。困惑平衡了数据集的局部和全局方面。一个非常高的值将导致集群合并成一个大集群，而低的值将产生许多没有意义的小集群。下图显示了虹膜数据集上 t-SNE 的困惑效果。
![](img/1d98b28636d9d348202fc6986e9f9ca5.png)
当 K(邻居数)= 5 t-SNE 产生许多小簇。当班级人数很多时，这会产生问题。随着邻居数量的增加，来自相同类的聚类合并。在 K=25 和 K=50 时，我们有几类定义明确的聚类。此外，随着 K 的增加，团簇变得更加密集。
t-SNE 嵌入后 MNIST 数据集子集的图。
![](img/ebbf0a1a6ec1c27fe6769e88013ec3c5.png)
t-SNE 为每一个数字产生一个定义明确且独立的聚类。
**t-SNE 的缺点**
t-SNE 的问题出现在固有维数较高时，即大于 2-3 维。像其他基于梯度下降的算法一样，t-SNE 有陷入局部最优的趋势。由于最近邻搜索查询，基本的 t-SNE 算法很慢。

**结论**:我们通过这篇帖子谈到了另一种降维可视化方法 **t-SNE** 。一开始，我们讨论了与 SNE 霸王龙相关的重要话题。之后，使用 pyspark 实现了一个基本的 t-SNE。基本 t-SNE 的扩展很少，这提高了算法的时间复杂度。希望深入研究这个话题的人[巴恩斯-胡特 t-SNE](https://arxiv.org/pdf/1301.3342) 将是一个很好的起点。

在下一篇文章中，我们将了解 [Isomap](https://blog.paperspace.com/p/16586ad9-bb4e-4711-9650-b1f3a71e11a4/)