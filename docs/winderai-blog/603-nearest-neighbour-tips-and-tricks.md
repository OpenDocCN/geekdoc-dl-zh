# 603:最近邻提示和技巧

> 原文：<https://winder.ai/603-nearest-neighbour-tips-and-tricks/>

### 维度和领域知识

*   对所有要素使用相同的距离度量是正确的吗？

例如身高和性别？CPU 和磁盘空间？

*   由于其规模的原因，一些特征将比其他特征具有更大的影响。

？？？

在这个版本的算法中，所有特征都用于距离计算。

这将所有功能视为相同。所以测量身高和测量性别的效果是一样的。

结果是一些变量比其他变量更容易影响结果。

解决方法是充分利用特征选择和/或工程(见下文)。即移除或改变不提供任何描述能力的特征。

这就要求你在其他算法(比如深度学习)领域里更像一个专家。

* * *

### 计算效率

*   k-NN 没有训练期。只有预测。
*   性能与特征和观察的数量成正比。

这意味着在生产中寻找邻居的成本会很高。

*你可以找到 k-NN 的优化来克服这个问题。如`KDTree`*

？？？

k-NN 特别有趣，因为大多数算法的训练时间比预测时间长。

k-NN 是不同的，因为它不需要训练，只需要测量距离的预测步骤。

缺点是当特征数量或观察数量很大时，预测时间变得越来越长。

对于需要快速预测的任务(例如在线广告推荐)，这可能不是最佳选择。

尽管您当然可以将数据集限制在某个大小以保证性能。或者使用一些随机抽样来提高性能。

* * *

## 距离测量

*   还有其他距离度量(几百！)

？？？

我们已经使用了欧几里德距离，但是根据您的应用，还有许多其他的距离(实际上有数百个)可能更有意义。

原因是距离度量往往是特定于领域的。

例如，可能有一种很好的方法来计算城市中出租车起点和终点之间的距离(由于建筑物的阻挡，它们通常以直角行驶)。

概括一下&mldr;

* * *

### 欧几里得(L2 范数)

$$ d_{Euclidean}(\mathbf{x}, \mathbf{y}) = ||\mathbf{x} - \mathbf{y}||=\sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + &mldr; } $$

？？？

和以前一样，L2 范数更容易受到异常值的影响。

同样，它假设高斯分布(对于复杂的多维数据，这可能不成立)。

这是最流行的距离测量，但可能不是正确的。

* * *

### 曼哈顿(L1 标准)

$$ d_{Manhatten}(\mathbf{x}, \mathbf{y}) = |\mathbf{x} - \mathbf{y}|=|x_1 - y_1| + |x_2 - y_2| + &mldr; $$

这是 L1 的标准，被命名为曼哈顿，因为这是你在一个城市中到达一个目的地必须走过的全部街道距离。

？？？

与之前一样，L1 范数受异常值的影响也较小，因此可能是含噪数据集的更好选择。

* * *

### 杰克卡

$$ d_{Jaccard}(\mathbf{x}, \mathbf{y}) = 1 - \frac{ \mathbf{x} \cap \mathbf{y} }{ \mathbf{x} \cup \mathbf{y} } $$

| 交集 | 联盟 |
| --- | --- |
| ![:scale 66%](img/04634bcfd9fc4e540cab7a71571d4a97.png) | ![:scale 66%](img/c4000e1c24f3e3fcca872d00cebb4ac4.png) |

*   共享项目/所有项目

例如，如果两杯威士忌是泥炭味的，那就意义重大。如果它们不辣，那很可能不是。

？？？

它是两个集合的并集(整个面积)除以两个集合的交集(分摊面积)。当您拥有高度分类的数据时，Jaccard 距离非常有用。这对于类别相似但类别不相似的情况很有用。

* * *

此处提供更多距离测量:

[http://sci kit-learn . org/stable/modules/generated/sk learn . neighbors . distance metric . html](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html)

这里整理了超过 100 个:

德扎和德扎的《距离词典》

* * *

## 组合函数

*   多数票
*   基于距离对类别进行加权
*   将权重除以所有观察值以生成概率估计值
*   使用中间值，而不是平均值
*   以上都是为了回归。

有关实现示例，请参见 sklearn k-NN 文档中的“weights”参数:[http://sci kit-learn . org/stable/modules/generated/sk learn . neighbors . kneighborsclassifier . html](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

？？？

记住，在我们选择了 k 个最近的邻居之后，我们需要选择最好的类或值或其他。

我们需要做的是把其他的观察结合成一个结果。

正常的 k-NN 算法使用简单多数表决。但是其他的包括:

*   基于距离对类别进行加权
*   将权重除以所有观察值以生成概率估计值
*   使用中间值，而不是平均值
*   以上都是为了回归。

有关实现示例，请参见 sklearn k-NN 文档中的“weights”参数:[http://sci kit-learn . org/stable/modules/generated/sk learn . neighbors . kneighborsclassifier . html](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)