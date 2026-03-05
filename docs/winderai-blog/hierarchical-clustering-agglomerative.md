# 层次聚类-凝聚的

> 原文：<https://winder.ai/hierarchical-clustering-agglomerative/>

# 层次聚类-凝聚聚类

欢迎光临！本车间来自 [Winder.ai](https://Winder.ai/?utm_source=winderresearch&utm_medium=notebook&utm_campaign=workshop&utm_term=individual) 。注册以获得更多免费的研讨会、培训和视频。

聚类是一项无人监督的任务。换句话说，我们没有任何标签或目标。当您收到类似“我们可以用这些数据做什么？”这样的问题时，这是很常见的或者“你能告诉我这个数据的特点吗？”。

有相当多不同的方式来执行集群，但是一种方式是分层地形成集群。您可以通过两种方式形成层次结构:从顶部开始并拆分，或者从底部开始并合并。

在本次研讨会中，我们将关注后者，这被称为*凝聚聚类*。

当你可以断言在你的领域中有建立等级制度的自然趋势时，等级技术是非常有用的。例如，如果你对使用你的应用程序的人进行剖析，那么你可能会发现人们倾向于形成等级制度。

```py
# Usual imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display 
```

## 数据

为此，我们将再次使用我们现在熟悉的`iris`数据集，因为它有一个自然的层次结构。

```py
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
feat = iris.feature_names
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target
y_name = ['Setosa', 'Versicolour', 'Virginica'] 
```

## 凝聚聚类

请记住，聚集聚类是自下而上形成集群的行为。

我们从单个观察值作为聚类开始，然后迭代地将它们分配到最近的聚类。

最终我们会得到许多集群(需要提前指定)。

让我们坚持“ward”链接来定义如何合并集群；它通常工作得很好。

```py
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(linkage="ward", n_clusters=3)
clustering.fit(X); 
```

让我们再次绘制数据。

```py
# MinMax scale the data so that it fits nicely onto the 0.0->1.0 axes of the plot.
from sklearn import preprocessing
X_plot = preprocessing.MinMaxScaler().fit_transform(X)

colours = 'rbg'
for i in range(X.shape[0]):
    plt.text(X_plot[i, 0], X_plot[i, 1], str(clustering.labels_[i]),
             color=colours[y[i]],
             fontdict={'weight': 'bold', 'size': 9}
        )

plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.show() 
```

![png](img/f0cce46c4602c476ec00bd5edb2a973d.png)

在该图中，数字表示每个观察值被分配到哪个组。

颜色表示原始类别。

我想你会同意集群已经做了相当不错的工作，有一些异常值。

###任务

*   尝试将簇的数量更改为 1、3，其他的&mldr;。

# 系统树图

树状图是聚类的层次图，其中条形的长度表示到下一个聚类中心的距离。

我们可以依靠我们的其他通用数据科学库`scipy`为我们提供绘制树状图的方法。不幸的是，由于一些预期的参数，我们也必须使用`scipy`的连接方法，而不是`sklearn`的。

```py
from scipy.cluster.hierarchy import dendrogram, linkage

linkage_matrix = linkage(X, 'ward')
figure = plt.figure(figsize=(7.5, 5))
dendrogram(
    linkage_matrix,
    color_threshold=0,
)
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.tight_layout()
plt.show() 
```

![png](img/0217fbfa0e647b7947af934f2c49b2de.png)

哇哦，好多条腿啊。让我们切掉一些，以便更好地查看数据&mldr;

```py
figure = plt.figure(figsize=(7.5, 5))
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=24,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.title('Hierarchical Clustering Dendrogram (Ward, aggrogated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
plt.show() 
```

![png](img/b0b8984802163e430754602c0c580f18.png)

好多了。

## 任务:

*   尝试使用树状图中的不同参数。查阅[汇总文件](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)。
    *   p
    *   截断模式

好了，现在我们已经磨练了我们的艺术技巧，让我们在威士忌数据集中测试它们。

## 再看一下威士忌数据集

我们再装一次。

```py
whiskey = pd.read_csv('https://s3.eu-west-2.amazonaws.com/assets.winderresearch.com/data/whiskies.csv')
cols = ['Body', 'Sweetness', 'Smoky', 'Medicinal', 'Tobacco',
       'Honey', 'Spicy', 'Winey', 'Nutty', 'Malty', 'Fruity', 'Floral']
X = whiskey[cols]
y = whiskey['Distillery']
display(X.head())
display(y.head()) 
```

|  | 身体 | 芳香 | 烟雾弥漫 | 药用的 | 烟草 | 蜂蜜 | 香的 | 有酒味的 | 古怪的 | 马耳他之鹰 | 圆润的 | 植物的 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | Two | Two | Two | Zero | Zero | Two | one | Two | Two | Two | Two | Two |
| one | three | three | one | Zero | Zero | four | three | Two | Two | three | three | Two |
| Two | one | three | Two | Zero | Zero | Two | Zero | Zero | Two | Two | three | Two |
| three | four | one | four | four | Zero | Zero | Two | Zero | one | Two | one | Zero |
| four | Two | Two | Two | Zero | Zero | one | one | one | Two | three | one | one |

```py
0    Aberfeldy
1     Aberlour
2       AnCnoc
3       Ardbeg
4      Ardmore
Name: Distillery, dtype: object 
```

## 任务:

*   绘制威士忌数据的树状图
*   初学者:从每个主要类别中选择一种威士忌(手动)
*   高级:编写一个算法，仅使用数据从每个主要威士忌组中选择一种威士忌(截断数据，然后从`AgglomerativeClustering.children_`字段中选择观察值)