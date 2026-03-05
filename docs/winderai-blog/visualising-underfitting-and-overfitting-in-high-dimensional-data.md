# 可视化高维数据中的欠拟合和过拟合

> 原文：<https://winder.ai/visualising-underfitting-and-overfitting-in-high-dimensional-data/>

# 可视化高维数据中的欠拟合和过拟合

欢迎光临！本车间来自 [Winder.ai](https://Winder.ai/?utm_source=winderresearch&utm_medium=notebook&utm_campaign=workshop&utm_term=individual) 。注册以获得更多免费的研讨会、培训和视频。

在[之前的研讨会](https://winder.ai/501-over-and-underfitting/)中，我们绘制了欠拟合和过拟合分类器的决策边界。这很好，但是通常不可能可视化数据，通常是因为数据集中有太多的维度。

在这种情况下，我们需要用另一种方式来想象表演。做到这一点的一个方法是产生一个*验证曲线*。这是一种强力方法，针对您指定的每个参数，根据维持数据对模型的性能进行重复评分。

但是 sklearns 实现最酷的地方是它执行*交叉验证*。这意味着这种方法是相当可靠的，我们得到了一些关于我们分数的统计上的信心。

简而言之，我们在各种超参数值上执行*网格搜索*来控制模型的拟合。

```py
# Usual imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.svm import SVC 
```

```py
# Synthetic data
X, y = make_circles(noise=0.2, factor=0.5, random_state=1, n_samples=200) 
```

```py
# Use sklearns inbuilt validation_curve method
# Although it wouldn't be too hard to code ourselves.
from sklearn.model_selection import validation_curve

param_range = np.logspace(-2, 3, 10)
train_scores, test_scores = validation_curve(
    SVC(), X, y, param_name="gamma", param_range=param_range,
    cv=10, scoring="accuracy", n_jobs=1)

# Generate average scores and standard deviations for more interesting plots
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1) 
```

```py
# Plot the results
plt.figure()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)

plt.title("Validation Curve (SVM)")
plt.xlabel("$\gamma$")
plt.ylabel("Score")
plt.ylim(0.01, 1.1)
plt.grid()
plt.legend(loc="best")
plt.show() 
```

![png](img/2f8332fd6b06fa62cda93351e154ebf3.png)

现在很清楚，伽马的最佳值大约是值`1-10`。美妙之处在于，它可以处理任意维数的数据。

但是当您有多个参数需要优化时，事情就变得棘手了。为此，我们必须执行*网格搜索*。网格搜索是对许多不同超参数的重复训练和拟合过程。

## 学习曲线

另一种可视化表现的方式是学习曲线。该图使用不同大小的样本进行训练。

如果训练分数和验证分数之间有很大的差距，那么我们就是过度拟合了。如果训练分数低，我们就不适合。

如果我们可以看到学习曲线随着更多的样本继续上升，那么如果我们收集更多的样本，我们可能会获得更好的性能。

```py
# Use sklearns inbuilt validation_curve method
# Although it wouldn't be too hard to code ourselves.
from sklearn.model_selection import learning_curve
param_range = np.logspace(-1.3, -0.1, 10)
train_sizes, train_scores, test_scores = learning_curve(
    SVC(), X, y,
    train_sizes=param_range, cv=5)

# Generate average scores and standard deviations for more interesting plots
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1) 
```

```py
# Plot the results
plt.figure()
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)

plt.title("Validation Curve (SVM)")
plt.xlabel("$\gamma$")
plt.ylabel("Score")
plt.ylim(0.01, 1.1)
plt.grid()
plt.legend(loc="best")
plt.show() 
```

![png](img/6bbf921db7ae6be90b46eb0a44e48f65.png)

请注意，在此示例中，当使用所有训练数据时，训练分数和交叉验证分数大致相等。

这表明添加更多的数据不会提高性能，我们已经尽我们所能使用这个模型。此时，我们可能需要考虑其他模型来提高训练分数。

```py
from sklearn.datasets import make_moons

# Synthetic data
X, y = make_moons(noise=0.2, random_state=42)

# This is how we split our data, using the `train_test_split` method
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42) 
```

## 任务

*   用支持向量机拟合上述“月亮”数据。