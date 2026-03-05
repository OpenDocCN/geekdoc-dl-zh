# 最近邻算法

> 原文：<https://winder.ai/nearest-neighbour-algorithms/>

# 最近邻算法

欢迎光临！本车间来自 [Winder.ai](https://Winder.ai/?utm_source=winderresearch&utm_medium=notebook&utm_campaign=workshop&utm_term=individual) 。注册以获得更多免费的研讨会、培训和视频。

最近邻算法是一类使用某种*相似度*的算法。它们依赖于这样的前提，即彼此接近的观测值(当比较所有特征时)彼此相似。

做出这个假设，我们可以做一些有趣的事情，比如:

*   推荐
*   找到类似的东西

但更重要的是，它们提供了对数据的*特征*的洞察。

当你得到一些数据，但不知道目标是什么时，这真的很有用。

## 加载数据

这部分的数据都是关于威士忌的。目标是能够根据你喜欢的威士忌推荐相似的威士忌。所以对于威士忌爱好者来说，这是你理想的工作间！

它也类似于严肃的数据集；你可以想象，不是关于威士忌的特征，而是关于竞争对手产品或顾客的特征。

```py
# Usual imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display 
```

```py
whiskey = pd.read_csv('https://s3.eu-west-2.amazonaws.com/assets.winderresearch.com/data/whiskies.csv')
display(whiskey.head())
print(whiskey.columns) 
```

|  | 酿酒厂 | 身体 | 芳香 | 烟雾弥漫 | 药用的 | 烟草 | 蜂蜜 | 香的 | 有酒味的 | 古怪的 | 马耳他之鹰 | 圆润的 | 植物的 | 邮政编码 | 纬度 | 经度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zero | 阿伯费尔德 | Two | Two | Two | Zero | Zero | Two | one | Two | Two | Two | Two | Two | PH15 2EB | Two hundred and eighty-six thousand five hundred and eighty | Seven hundred and forty-nine thousand six hundred and eighty |
| one | 阿伯鲁尔 | three | three | one | Zero | Zero | four | three | Two | Two | three | three | Two | AB38 9PJ | Three hundred and twenty-six thousand three hundred and forty | Eight hundred and forty-two thousand five hundred and seventy |
| Two | AnCnoc | one | three | Two | Zero | Zero | Two | Zero | Zero | Two | Two | three | Two | AB5 5LI | Three hundred and fifty-two thousand nine hundred and sixty | Eight hundred and thirty-nine thousand three hundred and twenty |
| three | 阿德贝格 | four | one | four | four | Zero | Zero | Two | Zero | one | Two | one | Zero | PA42 7EB | One hundred and forty-one thousand five hundred and sixty | Six hundred and forty-six thousand two hundred and twenty |
| four | 阿德莫尔 | Two | Two | Two | Zero | Zero | one | one | one | Two | three | one | one | AB54 4NH | Three hundred and fifty-five thousand three hundred and fifty | Eight hundred and twenty-nine thousand one hundred and forty |

```py
Index(['Distillery', 'Body', 'Sweetness', 'Smoky', 'Medicinal', 'Tobacco',
       'Honey', 'Spicy', 'Winey', 'Nutty', 'Malty', 'Fruity', 'Floral',
       'Postcode', ' Latitude', ' Longitude'],
      dtype='object') 
```

我们可以看到，这里有几个我们不想泄露到数据集中的特征。

我们感兴趣的是威士忌的味道有多接近，而不是地理上有多接近(尽管我们认为这将是结果！)

```py
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

## 距离测量

首先让我们定义一个距离度量。

我们在演示中定义了很多，但是最好从最简单的开始。所以让我们写一个 euclieanDistance 方法&mldr;

$$ d_{Euclidean}(\mathbf{x}, \mathbf{y}) = ||\mathbf{x} - \mathbf{y}||=\sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + &mldr; } $$

```py
import math
def euclideanDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance) 
```

有一件事我们还没怎么谈过，那就是测试驱动开发(TDD)。

(如果你不是所有的软件工程师)TDD 是这样一种思想，所有的代码都有一个测试来证明它做了它想要做的事情。

TDD 声明应该首先编写测试。

在整个笔记本中散布这些方法是非常方便的，以确保你的自定义方法做了它们应该做的事情

```py
from unittest import *
class TestDistance(TestCase):
    def testSimple(self):
        self.assertAlmostEqual(euclideanDistance([0], [1]), 1.0, places=1)
    def test2D(self):
        self.assertAlmostEqual(euclideanDistance([0, 0], [1, 1]), 1.4, places=1)

suite = TestLoader().loadTestsFromModule(TestDistance())
TextTestRunner(verbosity=2).run(suite) ; 
```

```py
test2D (__main__.TestDistance) ... ok
testSimple (__main__.TestDistance) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.006s

OK 
```

## 最近的邻居

现在我们有了距离的度量，让我们写一个算法来得到最近的邻居..

```py
import operator
def getNeighbors(trainingSet, instance, k):
    """Return the first k locations of the nearest neighbours to an instance"""
    distances = []
    for x in range(len(trainingSet)):
        dist = euclideanDistance(instance, trainingSet[x])
        distances.append(dist)
    locs = np.argsort(distances)
    return locs[:k] 
```

为了测试这是否可行，让我们定义一个我们想要比较的测试实例。

我们期望测试实例是最接近的匹配，然后是更多的匹配。

```py
testInstance = X.loc[y == 'Laphroig']
display(testInstance) 
```

|  | 身体 | 芳香 | 烟雾弥漫 | 药用的 | 烟草 | 蜂蜜 | 香的 | 有酒味的 | 古怪的 | 马耳他之鹰 | 圆润的 | 植物的 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fifty-eight | four | Two | four | four | one | Zero | Zero | one | one | one | Zero | Zero |

```py
neighbors = getNeighbors(X.as_matrix(), testInstance.as_matrix()[0], 5)
print(y[neighbors]) 
```

```py
58     Laphroig
57    Lagavulin
3        Ardbeg
23    Clynelish
21     Caol Ila
Name: Distillery, dtype: object 
```

太好了。如果你了解你的领域(我知道一点，至少是关于 smokey 威士忌:-)，那么我们可以看到这些都是非常合理的建议。

### 任务:

*   根据您喜欢的威士忌，为您推荐一些威士忌！