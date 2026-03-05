# 信息和熵

> 原文：<https://winder.ai/information-and-entropy/>

# 信息和熵

欢迎光临！本车间来自 [Winder.ai](https://Winder.ai/?utm_source=winderresearch&utm_medium=notebook&utm_campaign=workshop&utm_term=individual) 。注册以获得更多免费的研讨会、培训和视频。

记住数据科学的目标。目标是根据一些数据做出决定。这个决定的质量取决于我们的信息。如果我们有好的、清晰的信息，那么我们就能做出明智的决定。如果我们有糟糕、混乱的数据，那么我们的决策将是糟糕的。

## 分类

在分类的背景下，也就是试图预测一个观察值属于哪一个*类*，如果我们的算法能够干净利落地将类*分开*，我们就可以更加确定一个结果。

衡量一组类有多干净或多纯净的一个标准是熵。

在本次研讨会中，我们将从数学上定义熵，它是对有限位数中可以存储的信息量的一种度量。

```py
import numpy as np # Numpy is a general purpose mathematical library for Python.
                   # Most higher level data science libraries use Numpy under the bonnet. 
```

```py
X = np.array([0, 0, 1, 1, -1, -1, 100]) # Create an array. All numpy funcitons expect the data in a Numpy array.
print(np.mean(X))
print(np.var(X)) 
```

```py
14.2857142857
1225.06122449 
```

# 熵

记住 entrpopy 的定义是:

$$H=-\sum(p_i \log_2 (p_i))$$

其中\(p_i\)是观察值属于类\(i\)的概率。(即$p(y==c)/n$，其中 y 是目标，c 是感兴趣的类别，n 是样本总数)

例如，如果我们有两个类:

$ $ H =-P1 \ log _ 2(P1)-p2 \ log _ 2(p2)$ $

### 工作

*   通读这段代码，了解发生了什么。
*   尝试计算另一个值数组的熵。当你增加更多的值时会发生什么？改变价值观？

```py
X = np.array([[4.2, 92], [6.4, 102], [3.5, 3], [4.7, 10]])  # Numpy arrays are general purpose mathematical arrays
y = np.array([0, 0, 1, 1])                                  # They implement all kinds of useful operators, like the == operator. 
```

```py
def entropy(y):
    probs = [] # Probabilities of each class label
    for c in set(y): # Set gets a unique set of values. We're iterating over each value
        num_same_class = sum(y == c)  # Remember that true == 1, so we can sum.
        p = num_same_class / len(y) # Probability of this class label
        probs.append(p)
    return np.sum(-p * np.log2(p) for p in probs) 
```

```py
print(entropy(y)) # Should be 1.0 
```

```py
1.0 
```

# 信息增益

假设我们有一些类似上面的`X`和`y`的数据，其中`X`是特征，`y`是类别标签。

我们可以提出一个阈值或规则来分割`X`中的数据，以区分类别。我们如何量化哪一个是最好的分割？

我们所能做的是比较父项在分裂前的熵和分裂后熵的加权组合。即，如果三个观察结果在左桶中结束，一个在右桶中结束，则左桶将占孩子熵的四分之三。

如果我们从加权的子熵中减去父熵，我们就剩下一个衡量改进的指标*。这被称为*信息增益*。*

 *信息增益被定义为父熵减去子组的加权熵。

$$ \begin{align} IG(parent，children)= & entropy(parent)-\ non number \ \
&\ left(p(C1)熵(C1)+p(C2)熵(C2)+&mldr；\right) \end{align} $$

### 任务:

*   给定下面的`information_gain`函数(理解它),选择一些分裂并计算信息增益。哪个更好？

```py
def information_gain(parent, left_split, right_split):
    return entropy(parent) - (len(left_split) / len(parent)) * entropy(left_split) - (len(right_split) / len(parent)) * entropy(right_split) 
```

```py
# Make a split around the first column, < 5.0:
split1 = information_gain(y, y[X[:, 0] < 5.0], y[X[:, 0] > 5.0])
print("%0.2f" % split1)   # Should be 0.31 
```

```py
0.31 
```

```py
# Make a split around the second column, < 50.0:
split2 = information_gain(y, y[X[:, 1] < 50], y[X[:, 1] > 50])
print(split2)   # Should be 1.0
print("Split %d is better" % ((split1 < split2) + 1))     # Split 2 should be better, higher information gain 
```

```py
1.0
Split 2 is better 
```*