# 用于分类的 K-NN

> 原文：<https://winder.ai/k-nn-for-classification/>

## 用于分类的 K-NN

欢迎光临！本车间来自 [Winder.ai](https://Winder.ai/?utm_source=winderresearch&utm_medium=notebook&utm_campaign=workshop&utm_term=individual) 。注册以获得更多免费的研讨会、培训和视频。

在之前的研讨会中，我们研究了最近邻算法如何使用距离概念作为相似性度量。

我们也可以使用相似性的概念作为分类标准。即新的观测值将被分类为与其邻居相同。

这是通过找到最相似的观察值并将预测分类设置为 k 个最近邻的某种组合来实现的。(例如最常见的)

这一次，让我们使用类似于`iris`数据集的分类数据集，并使用`sklearn`中内置的 k-NN 实现。

```py
# Usual imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from sklearn import datasets
from sklearn import neighbors

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target 
```

```py
# Quick plot to see what we're dealing with
plt.scatter(X[:,0], X[:,1], c=y)
plt.show() 
```

![png](img/c0c73f6a7f62764279d84f8749f74178.png)

现在让我们编写一个奇特的方法来绘制决策边界&mldr;

```py
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold) 
```

```py
figure = plt.figure(figsize=(12, 6))
n_neighbours = [1, 20]
i = 1
for n in n_neighbours:
    clf = neighbors.KNeighborsClassifier(n, weights='distance')
    clf.fit(X, y)
    ax = plt.subplot(1, len(n_neighbours), i)
    i += 1

    plot_decision_regions(X=X, y=y, classifier=clf)
    ax.set_xticks(())
    ax.set_yticks(())
    plt.title(str(n) + '-NN')

plt.show() 
```

![png](img/848a77919e4034b652ee60187ba07fee.png)

正如我们可以看到的，当我们只选择单个邻居进行分类时，我们最终会有一个非常复杂的决策边界。如果我们对这个分类器进行验证，我们可能会发现它的性能很差；我们吃得太多了。

一般来说，随着`k`值的增加，决策边界变得更加平滑。

但是要小心。您必须选择一个值`k`以避免平局(即奇数)，并且它必须是互质的(允许所有职业都有机会投票)。有关更多信息，请参见培训视频。

## 任务

*   尝试在此数据集上绘制不同 k 值的验证曲线