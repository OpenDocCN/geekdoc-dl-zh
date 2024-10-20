# k-均值聚类解释

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/k-means-clustering>

聚类是 1932 年由 H.E. Driver 和 A.L.Kroeber 在他们关于“[文化关系的量化表达](https://web.archive.org/web/20221206120911/https://digitalassets.lib.berkeley.edu/anthpubs/ucb/text/ucp031-005.pdf)”的论文中引入的。从那时起，这项技术有了一个很大的飞跃，并已被用于发现未知的应用领域，如医疗保健。

聚类是一种无监督学习，其中需要从未标记的数据集中提取参考。通常，它用于捕获数据集中固有的有意义的结构、底层过程和分组。在聚类分析中，任务是以这样一种方式将群体分成几个组，即同一组中的数据点比其他组中的数据点彼此更相似。简而言之，它是基于相似性和不相似性的对象的集合。

通过聚类，数据科学家可以在未标记的数据中发现内在的分组。虽然对于一个好的集群没有特定的标准，它完全取决于用户，他们想如何使用它来满足他们的特定需求。它可用于查找数据中的异常数据点/异常值，或识别未知属性，以在数据集中找到合适的分组。

让我们举一个例子，假设你在沃尔玛担任经理，希望更好地了解你的客户，通过使用新的和改进的营销策略来扩大你的业务。手动细分客户很困难。您有一些包含客户年龄和购买历史的数据，这里的聚类可以帮助根据他们的消费对客户进行分组。一旦客户细分完成，你就可以根据目标受众为每个群体定义不同的营销策略。

![What does clustering mean?](img/26e690d85950e1f52453d7a16d27bb6f.png)

*What does clustering mean? | Source: Author*

有许多聚类算法被分成不同的聚类模型。在为用例选择任何算法之前，熟悉集群模型以及它是否适合用例是很重要的。在选择任何聚类算法时，还有一点需要考虑，那就是数据集的大小。

数据集可以包含数百万条记录，并且不是所有算法都能有效扩展。K-Means 是最流行的算法之一，它也是规模有效的，因为它具有复杂度为 O(n) 的 T2。在本文中，我们将深入讨论 K-Means 以及它的流行原因。

## k 均值聚类

K-means 是一种基于质心的聚类算法，其中我们计算每个数据点与质心之间的距离，以将其分配给一个聚类。目标是识别数据集中的 K 个组。

> *“K-means 聚类是一种矢量量化的方法，最初来自信号处理，旨在将 n 个观测值划分为 K 个簇，其中每个观测值属于具有最近均值的簇，作为该簇的原型。”*–[*来源*](https://web.archive.org/web/20221206120911/https://en.wikipedia.org/wiki/K-means_clustering)

这是一个将每个数据点分配到组中的迭代过程，慢慢地，数据点会基于相似的特征进行聚类。目标是最小化数据点和聚类质心之间的距离之和，以识别每个数据点应该属于的正确组。

这里，我们将数据空间分成 K 个簇，并为每个簇分配一个平均值。数据点被放置在最接近该聚类平均值的聚类中。有几种距离度量可用于计算距离。

### K-means 是如何工作的？

我们举个例子来了解一下 K-means 是如何一步步工作的。该算法可以分为 4-5 个步骤。

1.  **选择集群的数量**

第一步是定义 K 个聚类，我们将在这些聚类中对数据进行分组。让我们选择 K=3。

2.  **初始化质心**

质心是聚类的中心，但是最初，数据点的精确中心是未知的，因此，我们选择随机数据点，并将它们定义为每个聚类的质心。我们将初始化数据集中的 3 个质心。

![K-means clustering - centroid](img/8352e23bb51974c41d50068b78d0460f.png)

*K-means clustering – centroid | Source: Author*

3.  **将数据点分配给最近的聚类**

既然已经初始化了质心，下一步就是将数据点 *X* [n] 分配给它们最近的聚类质心*C*k

![K-means clustering - assign data points](img/1de4366a4299adfd228a6a52eab5cf0c.png)

*K-means clustering – assign data points | Source: Author*

在这一步中，我们将首先使用欧几里德距离度量来计算数据点 X 和质心 C 之间的距离。

然后为数据点和质心之间的距离最小的数据点选择聚类。

![K-means clustering](img/1466b898cb33b5aa27b28bff1c980bf9.png)

*K-means clustering | Source: Author*

4.  **重新初始化质心**

接下来，我们将通过计算该聚类的所有数据点的平均值来重新初始化质心。

![K-means clustering](img/09dfa2bb4bd240e9e9d1096caa17e1a4.png)

*K-means clustering | Source: Author*

5.  **重复步骤 3 和 4**

我们将不断重复第 3 步和第 4 步，直到我们有了最佳的质心，并且数据点到正确聚类的分配不再改变。

![K-means clustering](img/817836855e172e42a6f8ac1dfe06e677.png)

*K-means clustering | Source: Author*

这个迭代过程是不是听起来很熟悉？嗯，K-means 遵循与期望最大化(EM)相同的方法。EM 是一种迭代方法，用于寻找参数的最大似然，其中机器学习模型依赖于未观察到的特征。这种方法包括两步期望(E)和最大化(M ),并在这两步之间迭代。

对于 K-means，期望(E)步骤是将每个数据点分配给最可能的聚类，最大化(M)步骤是使用最小平方优化技术重新计算质心。

### 质心初始化方法

定位初始质心具有挑战性，目标是初始化质心，使其尽可能接近实际质心的最佳值。建议使用一些策略来定义初始质心，因为它会直接影响整个运行时间。传统的方法是随机选择质心，但也有其他方法，我们将在本节中介绍。

这是初始化质心的传统方法，其中 K 个随机数据点被选择并定义为质心。正如我们在上面的例子中看到的，在这个方法中，数据集中的每个数据实例都必须被枚举，并且必须保存每个属性的最小/最大值的记录。这是一个耗时的过程；随着数据集复杂性的增加，实现正确质心或正确聚类的步骤数量也将增加。

**分片质心初始化算法主要依赖于数据集中特定实例或行的所有属性的合成总和值。其思想是计算复合值，然后用它对数据的实例进行排序。一旦数据集被排序，它就被水平分成 k 个碎片。**

**![Sorting by composite value and sharding](img/fd40069d7704cc4c4a6545621889651b.png)

*Sorting by composite value and sharding | [Source](https://web.archive.org/web/20221206120911/https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html)*

最后，将每个碎片的所有属性相加，并计算它们的平均值。shard 属性意味着值集合将被标识为可用于初始化的质心集。

![Centroid attribute values](img/3b5364848dc75a0e90f44375443bc1e0.png)

*Centroid attribute values | [Source](https://web.archive.org/web/20221206120911/https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html)*

使用分片的质心初始化在线性时间内发生，并且所得的执行时间比随机质心初始化好得多。

K-means++是 K-mean 算法的智能质心初始化方法。目标是通过随机分配第一个质心来展开初始质心，然后根据最大平方距离选择其余的质心。这个想法是将质心彼此推得尽可能远。下面是使用 K-means++初始化质心的简单步骤:

1.  随机选择第一个质心(C1)
2.  计算所有数据点和所选质心之间的距离

这表示数据点*x[I]到最远质心*C[j]的距离**

3.  将数据点*x[I]初始化为新的质心*
4.  重复步骤 3 和 4，直到找到所有定义的 K 个聚类

> **"** *通过 k-means++初始化，保证算法找到与最优 k-means 解相比具有 O(log k)竞争力的解。"*–[*来源*](https://web.archive.org/web/20221206120911/https://en.wikipedia.org/wiki/K-means%2B%2B)

## 在 Python 中实现 K-Means 聚类

既然您已经熟悉了聚类和 K-means 算法，那么是时候使用 Python 实现 K-means，并看看它如何在真实数据上工作了。

我们将致力于商场访客数据集，以创建客户细分，从而确定营销策略。购物中心访客样本数据集可以在 [Kaggle](https://web.archive.org/web/20221206120911/https://www.kaggle.com/shrutimechlearn/step-by-step-kmeans-explained-in-detail/data) 上找到，它总结了大约 2000 名购物中心访客的消费情况。

让我们为下一阶段细分客户清理、探索和准备数据。

加载数据并检查是否有任何缺失值:

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

credit_data = pd.read_csv("/content/Mall_Customers.csv")

credit_data.head()

credit_data.isna().sum()
```

![Implementing K-Means clustering in Python.](img/f22c5dca0539b6d4db7faf57b97cde97.png)

*Mall visitors dataset | [Source](https://web.archive.org/web/20221206120911/https://www.kaggle.com/shrutimechlearn/step-by-step-kmeans-explained-in-detail/data)*

![Implementing K-Means clustering in Python.](img/c12bc0eb9c90d3b75e6b968c5140447d.png)

*Mall visitors dataset | Source: Author*

我们将使用**年收入**和**支出分数**来寻找数据中的聚类。支出分数从 1 到 100，根据客户行为和支出性质进行分配。

### 从头开始实现 K 均值

有一些开源库为不同类型的聚类算法提供了函数，但是在通过调用函数来使用这些开源代码之前，理解这些函数是如何工作的非常重要。在本节中，我们将使用随机质心初始化方法从头开始构建 K-means 聚类算法。

我们来看看数据，看看是怎么分布的:

```py
plt.scatter(customer_data['Annual Income (k$)'],customer_data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

![Implementing K-Means from scratch](img/390f997b59f899a9b06480be74588aee.png)

*Implementing K-Means from scratch | Source: Author*

从上面的散点图中，很难判断数据集中是否有任何模式。这就是集群将有所帮助的地方。

首先，我们将随机初始化质心:

```py
K=3
centroids = customer_data.sample(n=K)
plt.scatter(customer_data['Annual Income (k$)'],customer_data['Spending Score (1-100)'])
plt.scatter(centroids['Annual Income (k$)'],centroids['Spending Score (1-100)'],c='black')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

![Implementing K-Means from scratch](img/d4783114944e4e0a8ecd52f7bc92cedf.png)

*Implementing K-Means from scratch | Source: Author*

接下来，我们将遍历每个质心和数据点，计算它们之间的距离，找到 K 个聚类并将数据点分配给一个重要的聚类。这个过程将继续，直到先前定义的质心和当前质心之间的差为零:

```py
diff = 1
j=0
XD=X
while(diff!=0):

   i=1
   for index1,row_c in centroids.iterrows():
       ED=[]
       for index2,row_d in XD.iterrows():
           d1=(row_c["Annual Income (k$)"]-row_d["Annual Income (k$)"])**2
           d2=(row_c["Spending Score (1-100)"]-row_d["Spending Score (1-100)"])**2
           d=np.sqrt(d1+d2)
           ED.append(d)
       X[i]=ED
       i=i+1

   C=[]
   for index,row in X.iterrows():
       min_dist=row[1]
       pos=1
       for i in range(K):
           if row[i+1] < min_dist:
               min_dist = row[i+1]
               pos=i+1
       C.append(pos)
   X["Cluster"]=C
   centroids_new = X.groupby(["Cluster"]).mean()[["Spending Score (1-100)","Annual Income (k$)"]]
   if j == 0:
       diff=1
       j=j+1
   else:
       diff = (centroids_new['Spending Score (1-100)'] - centroids['Spending Score (1-100)']).sum() + (centroids_new['Annual Income (k$)'] - centroids['Annual Income (k$)']).sum()
       print(diff.sum())
   centroids = X.groupby(["Cluster"]).mean()[["Spending Score (1-100)","Annual Income (k$)"]]
```

```py
-27.971469513699734
-22.726172762723042
1.7620315282408434
1.2618819139010924
0.0
```

现在，如果我们要查看数据集，所有数据点都应相应地进行聚类:

```py
color=['grey','blue','orange']
for k in range(K):
   data=X[X["Cluster"]==k+1]
   plt.scatter(data["Annual Income (k$)"],data["Spending Score (1-100)"],c=color[k])
plt.scatter(centroids["Annual Income (k$)"],centroids["Spending Score (1-100)"],c='black')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
```

![Implementing K-Means from scratch](img/864ef0aeb480cb81565e9290a04c870d.png)

*Implementing K-Means from scratch | Source: Author*

### 实现 K-意味着使用 Scikit-Learn

这是相当多的代码，算法可能还需要一些优化，以改善其运行时间。为了避免增强算法并重新编写，我们可以使用开源库函数。Scikit Learn 有一个聚类包，其中包括 K-means 函数，该函数经过优化，在研究人员中非常受欢迎。

首先，我们将导入 K-Means 函数，然后通过将聚类数作为参数传递来调用该函数:

```py
from sklearn.cluster import KMeans
km_sample = KMeans(n_clusters=3)
km_sample.fit(customer_data)
```

就这样，您的集群数据就准备好了。我们再来看看数据:

```py
labels_sample = km_sample.labels_
customer_data['label'] = labels_sample
sns.scatterplot(customer_data['Annual Income (k$)'],customer_data['Spending Score (1-100)'],hue=customer_data['label'],palette='Set1')
```

![Implementation using Scikit-Learn](img/178a6ee2f3173a44b058018327704baa.png)

*Implementation using Scikit-Learn | Source: Author* 

我们能够使用 Scikit-Learn 只用几行代码创建客户细分。尽管对于这个特定的数据集，您可以看到最终的聚类数据在两个实现中是相同的。但是，通过这种细分，我们对商场顾客了解了什么:

**Label 0:** 储蓄者，avg 到高收入但花钱明智

标签 1: 无忧无虑，低收入但挥霍无度

**标签 2:** 支出者，平均高收入和支出者

当我们第一次绘制数据时，这很难理解，但现在我们知道我们有这 3 个类别，商场管理人员可以相应地应用营销策略，例如，他们可能为标签 0:储蓄者群体提供更多的储蓄优惠，并为标签 2:挥金如土者开设更有利可图的商店。

## 怎么选 K？

一些因素会挑战 K 均值聚类算法的最终输出的功效，其中之一是最终确定聚类的数量(K)。选择较低数量的分类会导致欠拟合，而指定较高数量的分类会导致过拟合。不幸的是，没有确定的方法来找到最佳数量。

聚类的最佳数量取决于相似性度量和用于聚类的参数。因此，要找到数据中的聚类数，我们需要对一系列值运行 k-means 聚类，并比较结果。[目前](https://web.archive.org/web/20221206120911/https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/)，我们没有任何方法来确定聚类 K 的准确值，但我们可以使用一些技术来估计该值，包括交叉验证、肘方法、信息标准、剪影方法和 G-means 算法。

### 肘法

距离度量是比较不同 K 值的结果的常用度量之一。当聚类的数量 K 增加时，从质心到数据点的距离将减少，并且将达到 K 与数据点的数量相同的点。这就是我们一直使用到质心的距离平均值的原因。在肘点法中，我们绘制平均距离，并寻找下降率移动的[肘点](https://web.archive.org/web/20221206120911/https://blogs.oracle.com/ai-and-datascience/post/introduction-to-k-means-clustering)。这个肘点可以用来确定 k。

> ****肘法*** *是一种用于确定数据集中聚类数量的启发式方法。该方法包括将解释的变化绘制为聚类数的函数，并选取曲线的拐点作为要使用的聚类数。”*–[*来源*](https://web.archive.org/web/20221206120911/https://en.wikipedia.org/wiki/Elbow_method_(clustering))*

 *肘点在数学优化中用作分界点，以决定在哪个点上收益递减不再值得额外的成本。类似地，在聚类中，当添加另一个聚类不能改善建模结果时，这用于选择多个聚类。这是一个迭代过程，其中 K 均值聚类将在数据集上针对如下 K 值范围进行。

1.  使用所有 K 值执行 K 均值聚类。对于每个 K 值，我们计算所有数据点到质心的平均距离:

```py
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

distortions = []
K = range(1,10)
for k in K:
   kmeanModel = KMeans(n_clusters=k).fit(X)
   kmeanModel.fit(X)
   distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
```

2.  标绘这些点，并找出平均距离突然下降的点(肘部):

```py
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

![Elbow method](img/0322b4779c8de683c1791083e9fb62ad.png)

*Elbow method | Source” Author*

这可能是确定最佳聚类数的最流行的方法。虽然找到肘点可能是一个挑战，因为在实践中可能没有一个尖肘。

### 剪影法

找到一个拐点在实践中是具有挑战性的，但是有其他技术来确定 K 的最佳值，其中之一是轮廓评分法。

> *“剪影”是指一种解释和验证数据簇内一致性的方法。这项技术提供了一个简洁的图形表示，显示每个对象的分类情况。*–[*来源*](https://web.archive.org/web/20221206120911/https://en.wikipedia.org/wiki/Silhouette_(clustering))

轮廓系数用于通过检查一个聚类中的数据点与其他聚类相比有多相似来衡量聚类的质量。[剪影分析](https://web.archive.org/web/20221206120911/https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)可用于研究生成的聚类之间的距离。这个离散的度量值范围在-1 和 1 之间:

**+1** 表示数据点远离相邻的聚类，因此处于最佳位置。

**0** 表示它位于或非常接近两个相邻聚类之间的判定边界。

**-1** 表示数据点被分配到错误的簇。

为了找到聚类数 K 的最佳值，我们使用轮廓图来显示一个聚类中的每个点与相邻聚类中的一个点有多接近的度量，从而提供了一种视觉评估聚类数等参数的方法。让我们看看它是如何工作的。

1.  计算一系列值的 K 均值聚类算法。
2.  对于每个 K 值，找出数据点的平均轮廓分数:

```py
from sklearn.metrics import silhouette_score

sil_avg = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for k in range_n_clusters:
 kmeans = KMeans(n_clusters = k).fit(X)
 labels = kmeans.labels_
 sil_avg.append(silhouette_score(X, labels, metric = 'euclidean'))
```

3.  绘制每个 K 值的轮廓分数集合
4.  选择轮廓分数最大时的聚类数:

```py
plt.plot(range_n_clusters,sil_avg,'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis For Optimal k')
plt.show()
```

![Elbow method](img/3a9d1460d680177f9aecfa0ac06db244.png)

*Elbow method | Source” Author*

使用上面的轮廓分析，我们可以选择 K 的最佳值为 3，因为平均轮廓得分较高，表明数据点的位置最佳。

## 聚类评估指标

在聚类中，我们没有任何标记数据，只有一组特征，目标是为这些特征获得高的类内相似性和低的类间相似性。评估任何聚类算法的性能都不像在监督学习中计算错误数或寻找精确度或召回率那样容易。在这里，我们根据数据点之间的相似性或差异性来评估结果。

在上一节中，我们看到了距离度量和轮廓分数如何帮助找到 k 的最佳值。因此，这些评估度量中的许多也可以用于为参数聚类算法找到最佳聚类点。聚类算法只和你的[相似性度量](https://web.archive.org/web/20221206120911/https://developers.google.com/machine-learning/clustering/interpret)一样好。因此，我们需要确保使用适当的相似性度量。一种方法是试验您的测量，并确定哪种算法可以提供更准确的相似性。

> *“理论上，聚类研究人员已经获得了对聚类评估的直觉，但是在实践中，一方面是大量的数据，另一方面是数据表示和聚类算法的微妙细节，使得直觉判断是不可能的。”*–[*博士论文，斯图加特大学*](https://web.archive.org/web/20221206120911/https://www.ims.uni-stuttgart.de/documents/team/schulte/theses/phd/algorithm.pdf)

有几个聚类评估指标可以用来帮助研究人员进行聚类，并且这些指标还在不断发展。在这一部分，我们将讨论一些最常见和最流行的指标。

### 邓恩指数

[邓恩指数](https://web.archive.org/web/20221206120911/https://www.analyticsvidhya.com/blog/2020/10/quick-guide-to-evaluation-metrics-for-supervised-and-unsupervised-machine-learning/#h2_4)用于识别密集且分离良好的群体。它是最小类间距离和最大类内距离的比值。邓恩指数可以计算如下:

这里 *d(i，j)* 是簇 *i* 和 *j* 之间的距离，是所有簇间距离的最小值， *d(k)* 是簇 *k* 的簇内距离，是所有簇内距离的最大值。创建具有高 Dunn 指数的聚类的算法是更理想的，因为这样，聚类将更紧凑并且彼此不同。

### 剪影分数

平均[轮廓分数](https://web.archive.org/web/20221206120911/https://en.wikipedia.org/wiki/Silhouette_(clustering))也用作聚类中的评估度量。最好的剪影评分是 1，最差的是-1。接近零的值表示数据点位于边界上，即与聚类重叠。

### f-测度

[F-measure](https://web.archive.org/web/20221206120911/https://en.wikipedia.org/wiki/F-score) 应用于对的精度和召回率，并用于通过加权召回率来平衡假阴性。

在聚类中，常用的方法是将 F-测度应用于对的精度和召回率，这被称为对计数 F-测度。我们可以使用下面的公式计算 F 值。

这里的选择使得回忆被认为是与精确同等重要的。当我们设置为 1 时，它将是精度和召回率的调和平均值。我们为每个给定的类计算聚类的召回率和精确度，即应该为对象给定一组类。

### 兰德指数

[Rand index](https://web.archive.org/web/20221206120911/https://en.wikipedia.org/wiki/Rand_index) 可用于计算集群与基准的相似程度。Rand 指数的值可以使用以下公式计算。

这里 TP 是真阳性的数量，TN 是真阴性的数量，FP 是假阳性的数量，FN 是假阴性的数量。使用这个评估标准，我们可以计算正确的成对分配的数量。

TP 是在预测分区和基本事实分区中聚集在一起的数据点对的数量，FP 是在预测分区中聚集在一起但不在基本事实分区中的数据点对的数量。

要了解更多评估指标，您可以查看 scikit learn [集群性能评估](https://web.archive.org/web/20221206120911/https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)指标页面。

## GMM(高斯混合模型)与 K-均值聚类算法

确定总体中聚类概率的概率模型被认为是混合模型。K-means 是一种快速而简单的聚类方法，但有时它不能捕捉固有的异质性。

[高斯混合模型](https://web.archive.org/web/20221206120911/https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050920X00093/1-s2.0-S1877050920309820/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEM%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIAtAUSizTDN6gTTrh94GQ3u4ICGwlXcVZ%2Bi0GqABgtp%2FAiEAjU0UqxXcjQYqjn4nYG96XcGpFwSmeOcdgpdL2j%2FMFaMq%2BgMIVxAEGgwwNTkwMDM1NDY4NjUiDF97BQcKlxV2ZqJHYSrXAxofjp0QreapAcdUHkRCQvvmlKW%2FRXR1J5oYS%2BJEpn%2BSc20a3YpU4bFveol9ygx7uiV58M%2FBNbtI0l3LkgnNNjoziGaNlRvFkGBjYpK4hJmOxYxZceCG%2Bsu2THNsQ9Onrw2Df74SX0Qyus2RxW7L5O0d4MybJjmw1yE1VxuLPQjUFT46ITtfhQrG2YHw5EADrKJLxTVqInD%2FgjpxXq2xf5rXm%2F%2FJkxQiAMLBnjbpujRRPKm7KoaltqO4DVGrTTsHNtvhzXdZxNM31L7dIbmlLZa7RhtL%2FF8CP%2B2MfH77IjlIr1WIEt1EDQcWhHVTieE4IIf3I45tOuuVWyvYW03AMcQabWbKkJShzVrXSXGNDEtRzefQu6C%2FE%2BMu%2BHYAThAUfaruwG2tlUW%2BgDzSBEQ7OQZuqYNPyLVa829juGLrKu9T2LtMfKdScb5ktYD0GjG%2FDd8DGXOjzcr0LFJHeWIn0YsWiDkJ2c%2By7vGGmEWRDSqzVGUyDg460rQ9%2BGBRBT1RQP1%2BloYA0nh8%2BmQDwMRoAMz%2B5NW%2BfC8hKaJ6Pr08B8gUL%2BAVeeTtqQaM%2BaCs7iUF7ZRNyCxfULkXprftfWNF5prYrFDbuXTPs7lGkD1VeLi4cGVp2w04kzDKuKSLBjqlAZdX52Xa1WqEDPUQTGjHiZD2yuBmkA1e5FJ4WovbD%2BNz1RJJUtsfHMunjtLkV35z9UowFunn%2Buw80VWZQ4hIM%2F6mb7OP0E%2BcDtt7vAYbi0GB1iyjQLwHhIYheRcursxE8%2FajrVdJZ%2FXae236Xgl%2BUy9r1ff8AKMQIXcZOZOO7d0ZQx4x%2FDnfGwSHswApq4bSTv19L7%2FCKY24OanyF7qN7h5Unb7mWQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20211015T073657Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYYXJ2EAYM%2F20211015%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=6edde39c451e08245746142832ee22199de4d39a88e5346ee25a5ce1abd146b8&hash=25fbc1eae88e3ff7aba6ac8565b384092b287495e6372c3fbf24f505f49ca68b&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050920309820&tid=spdf-64f59060-d678-4cf8-8816-7ed28cf9e3a9&sid=eb5ffcef3fdf614e698a5fa993b5eb623831gxrqb&type=client) (GMM)可以识别复杂的模式并将它们组合在一起，这是数据集中真实模式的近似表示。

### GMM 简介

实际上，无监督数据可能包含高度分散的数据点，并且很难将这些数据点管理到不同的聚类中。高斯混合模型(GMM)初始化一定数量的高斯分布，每个高斯分布代表一个聚类。在 GMM，我们倾向于将相似的数据点分组到一个分布中。

在高斯分布中，我们试图拟合钟形曲线下的数据点，钟形曲线下的所有数据点高度相关，可以形成一个簇。钟形曲线的峰值表示数据点的平均值。GMM 是基于 K 个独立的高斯分布，用于建模 K 个集群。


![Clustering using GMM](img/6a8f109961be49da6179a0ff4099b29b.png)

*Clustering using GMM | [Soure](https://web.archive.org/web/20221206120911/https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95)*

比方说，在一个数据集中，有三组不同的数据点遵循高斯分布，这意味着将有三个钟形曲线。GMM 将确定数据点出现在这些分布中的概率。GMM 的概率函数可以定义为:

是 d 维均值向量
是高斯的 dxd 协方差矩阵

**d** 是特征数量
X 是数据点数量

GMM 模型符合生成模型，该模型给出了数据集的概率分布。为了避免过度拟合或欠拟合，我们必须通过使用交叉验证方法或阿卡克信息标准(AIC)和贝叶斯信息标准(BIC)方法评估模型似然性来找到最佳分布数。

### GMM vs K-means 聚类算法及 K-means 为何如此受欢迎

*   GMM 使用概率分布，K-means 使用距离度量来计算数据点之间的差异，以将数据分成不同的聚类。

*   GMM 是一种软聚类算法，在某种意义上，每个数据点都被分配到一个具有一定程度不确定性的聚类中，例如，您可以在上面的图像中看到，一些数据点很有可能属于一个特定的高斯，而一些数据点则位于两个高斯之间。这意味着一个数据点可以属于一个以上的簇，即以 70/30 的比例分布在两个簇中。

*   在寻找不同大小、形状和密度的聚类时，K-means 可能表现不佳。例如，使用 K-means 对椭圆形分布的数据进行聚类可能会失败，因为它更适合圆形分布。在这种情况下，继续与 GMM 合作是有意义的。

但是，K-means 发现数据点组所需的时间比 GMM 发现高斯分量所需的时间少得多。K-means 之所以受欢迎，是因为它易于应用，收敛速度快。它可以在没有任何关于数据工程过程的假设的情况下使用。由于 K-means 的简单性和效率，它也被用于图像分割，并且它给出了比更复杂的深度神经网络算法更有竞争力的结果。

> “K-means 是一个合理的默认选择，至少在您发现集群步骤是您整体性能的瓶颈之前是如此。”–[*分析 Dimag*](https://web.archive.org/web/20221206120911/https://analyticsindiamag.com/is-k-means-clustering-really-the-best-unsupervised-learning-algorithm/)

## 应用和使用案例

K-means 可以应用于维数较少的数据集、数值型和连续型数据。它适用于希望将随机分布的数据点组合在一起的情况。下面是一些有趣的使用案例，其中 K-means 很容易使用:

> ***“****客户细分是将一家公司的客户分成反映每组客户相似性的组的做法。”*–[*Optimove*](https://web.archive.org/web/20221206120911/https://www.optimove.com/resources/learning-center/customer-segmentation#:~:text=Customer%20segmentation%20is%20the%20practice,each%20customer%20to%20the%20business.)

满足客户的需求是关系营销的起点，通过理解所有的客户并不相同，相同的优惠可能并不适用于所有人，可以提高关系营销的起点。根据客户的需求和行为对客户进行细分，可以帮助公司更好地向正确的客户推销产品。例如，[电信公司](https://web.archive.org/web/20221206120911/https://www.researchgate.net/publication/268445170_Prepaid_Telecom_Customer_Segmentation_Using_the_K-Mean_Algorithm)拥有大量用户，利用市场或客户细分，公司可以个性化活动和激励措施等。

互联网和在线服务的不断发展引起了人们对安全性的关注。目前，对这些安全威胁或欺诈活动(例如，从一个不寻常的城市登录 Instagram 账户或隐藏任何类型的金融犯罪)进行会计处理非常普遍。使用 K-means 聚类等技术，人们可以很容易地识别任何异常活动的模式。[检测到异常值](https://web.archive.org/web/20221206120911/https://link.springer.com/chapter/10.1007/978-981-33-4859-2_17)将意味着欺诈事件已经发生。

众所周知，K-Means 在处理大型数据集时非常有效，这也是它成为文档分类的最佳选择之一的原因。根据主题、内容和标签(如果有)将文档聚类成多个类别。文档将被转换成矢量格式。然后，我们使用术语频率来识别公共术语，并且基于此我们可以识别文档组中的相似性。

> 使用地理空间特征作为输入，可以通过机器学习来预测室外周围声学环境。然而，收集足够的训练数据是一个昂贵的过程，特别是当试图在大范围、地理空间多样化的区域内提高基于监督学习方法的模型的准确性时。”–[*地理空间模型*](https://web.archive.org/web/20221206120911/https://asa.scitation.org/doi/10.1121/2.0001299)

由于这些监督算法的限制，我们需要使用无监督算法，如 K-均值聚类，我们可以通过聚类数据轻松地比较地理多样性。

使用 K-means，我们可以找到图像像素中的模式，这将允许以更快的速度和更有效的方式进行处理。在计算图像的每个像素和质心之间的差异后，将其映射到最近的聚类。在最终输出中，聚类将具有相似的像素组合在一起。

你可以在这里找到更多关于 K-means 应用和用例[的信息。](https://web.archive.org/web/20221206120911/https://dzone.com/articles/10-interesting-use-cases-for-the-k-means-algorithm)

## K-means 聚类算法的优势

*   相对容易理解和实现。
*   可扩展到大型数据集。
*   更好的计算成本。
*   轻松热启动质心的分配和位置

## K-means 聚类算法的缺点

*   手动选择 K 并依赖于初始值
*   对于不同的 K 值缺乏一致的结果
*   总是试图找到圆形集群
*   由于数据集中的异常值，质心会被拖动
*   维数灾难，K 在维数增加时无效

## 最后的想法

在本文中，我们讨论了最流行的聚类算法之一。我们首先对 k-means 及其工作原理进行了概述，随后我们按照相同的步骤从头开始并通过 sklearn 实现了它。我们还研究了与 it 及其替代品相关的各种指标和挑战。

我们也看到 K-means 真的很好理解，可以很快交付训练结果。但是，数据的微小变化会影响其性能。聚类被假定为球形且大小均匀，这可能会降低 K-means 聚类的准确性。如果你想了解更多关于 K 均值聚类的知识，我建议你查看这些文章[Stanford cs 221–K 均值](https://web.archive.org/web/20221206120911/https://stanford.edu/~cpiech/cs221/handouts/kmeans.html)、[数据科学手册](https://web.archive.org/web/20221206120911/https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html)并访问 [scikit 学习网站](https://web.archive.org/web/20221206120911/https://scikit-learn.org/stable/modules/clustering.html#k-means)。***