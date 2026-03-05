# Scikit 向熊猫学习:数据类型不应该这么难

> 原文：<https://winder.ai/scikit-learn-to-pandas-data-types-shouldnt-be-this-hard/>

几乎每个使用 Python 进行数据科学研究的人都已经或者正在使用 [Pandas](https://pandas.pydata.org/) 数据分析/预处理库。它和 Scikit-Learn 一样重要。尽管如此，一个持续的问题是它们各自使用不同的核心数据类型:`pandas.DataFrame`和`np.array`。如果我们不用担心将`DataFrame`转换成`numpy`类型，然后再转换回来，这不是很好吗？是的，会的。向前一步 Scikit 熊猫。

## Sklearn 熊猫

作为 [Scikit Contrib](https://github.com/scikit-learn-contrib/scikit-learn-contrib/blob/master/README.md) 包的一部分，Sklearn Pandas 添加了一些语法糖，以便在 Sklearn 管道中使用数据帧，然后再返回。

让我们以自述文件中的[为例。这为我们提供了一些包含分类和数字数据的简单数据:](https://github.com/scikit-learn-contrib/sklearn-pandas#load-some-data)

```py
data = pd.DataFrame({'pet':      ['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'],
                     'children': [4., 6, 3, 3, 2, 3, 5, 4],
                     'salary':   [90., 24, 44, 27, 32, 59, 36, 27]})
data['pet'] = data['pet'].astype("category") 
```

现在我们可以使用这个库来创建一个地图，允许我们使用我们的熊猫数组和 sklearn:

```py
mapper = DataFrameMapper([
    ('pet', preprocessing.LabelBinarizer()),
    (['children'], preprocessing.StandardScaler())
])
mapper.fit_transform(data.copy()) 
```

我们正在使用新的类`DataFrameMapper`，我们将用它将输入`data`映射到数组中声明的 sklearn 函数的输出。注意这个类符合标准的 sklearn `fit` / `transform` api。当我们运行这个程序时，我们得到:

```py
array([[ 1.        ,  0.        ,  0.        ,  0.20851441],
       [ 0.        ,  1.        ,  0.        ,  1.87662973],
       [ 0.        ,  1.        ,  0.        , -0.62554324],
       [ 0.        ,  0.        ,  1.        , -0.62554324],
       [ 1.        ,  0.        ,  0.        , -1.4596009 ],
       [ 0.        ,  1.        ,  0.        , -0.62554324],
       [ 1.        ,  0.        ,  0.        ,  1.04257207],
       [ 0.        ,  0.        ,  1.        ,  0.20851441]]) 
```

首先要注意的是输出是一个`numpy`1。这有点令人惊讶，因为它应该是一个可以从熊猫来回映射的库。

第二个需要注意的是，新的`DataFrameMapper`看起来和 sklearn 的`pipeline.Pipeline`类非常相似。事实上，我甚至可以说这是复制了`Pipeline`类的功能。

另外，这也是对`Pipeline`类的不满，但是我不喜欢使用命名元组。这将是更干净的对待这一点，因为它是真的；功能管道。通过 sklearn 类/函数传入 lambda 来映射数据会使它更干净，更容易重用。

## Scikit-learn 的`Pipeline`就是你所需要的

这些想法不仅仅是我的。John Ramey 提出了一个简单的*适配器*类，为操作选择正确的数据类型(Ramey，2018)。汤姆·德·鲁伊特也提出了同样的观点(鲁伊特，2017)。

本质上，他们所做的是创建一个类来过滤特定的特性(看看我们在这里是如何使用函数式语言的)。在下面的例子中，我们过滤了一个数据`type`，但是我们可以很容易地过滤不同的参数，比如特性的名称。

```py
from sklearn.base import BaseEstimator, TransformerMixin
class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype]) 
```

我们可以在映射器前面使用这个过滤器来确保我们有正确的类型。例如，对于一个分类特征，我们现在可以创建一个标准的 sklearn 管道，如下所示:

```py
pipeline.make_pipeline(
    TypeSelector("category"),
    preprocessing.OneHotEncoder()
) 
```

我们现在需要做的就是对每个数据`type`或特征重复这个模式，然后将它们再次合并在一起。这就是它的作用:

```py
pipe = pipeline.make_union(
    pipeline.make_pipeline(
        TypeSelector("category"),
        preprocessing.OneHotEncoder()
    ),
    pipeline.make_pipeline(
        TypeSelector(np.number),
        preprocessing.StandardScaler()
    )
)
pipe.fit_transform(data.copy()).toarray() 
```

```py
array([[ 1.        ,  0.        ,  0.        ,  0.20851441,  2.27500192],
       [ 0.        ,  1.        ,  0.        ,  1.87662973, -0.87775665],
       [ 0.        ,  1.        ,  0.        , -0.62554324,  0.07762474],
       [ 0.        ,  0.        ,  1.        , -0.62554324, -0.73444944],
       [ 1.        ,  0.        ,  0.        , -1.4596009 , -0.49560409],
       [ 0.        ,  1.        ,  0.        , -0.62554324,  0.79416078],
       [ 1.        ,  0.        ,  0.        ,  1.04257207, -0.30452782],
       [ 0.        ,  0.        ,  1.        ,  0.20851441, -0.73444944]]) 
```

我们找到了。与库几乎相同的功能，使用标准方法的代码行更少。我们唯一没有做的是维护管道末端的特性元数据。上面代码的结果是一个`numpy`数组。

## 结论:你不需要额外的复杂性

scikit pandas 库也有一些覆盖了`sklearn`实现的帮助包装器方法，比如用于交叉验证的包装器和矢量化函数映射器。还是那句话，我觉得这些都是多余的。你可以用标准的`numpy`方法或者一点 python 函数很容易地做到这一点。

考虑到应该有多简单，我还担心库的圈复杂度。`_transform`方法的复杂度为 18 (21 被认为是高的——苏本德里和萨尔诺，2017)。

我不推荐使用目前这个库。我认为利用`sklearn` s `Pipeline`或带有一些包装方法/类的函数库是明智的。

但这让我想到了一个问题，考虑到这两个库是世界上最受欢迎的数据科学库，为什么集成性这么差？

## 参考

*   汤姆·德·鲁伊特。"整合熊猫和 sci kit-通过管道学习."Bigdatarepublic(博客)，2017 年 11 月 21 日。[https://medium . com/bigdata Republic/integrating-pandas-and-scikit-learn-with-pipelines-f 70 EB 6183696](https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696)。
*   苏本德里，穆罕默德·阿塞普和里亚纳托·萨尔诺。"确定 COCOMO II 中产品复杂程度的圈复杂度."Procedia 计算机科学，2017 年第四届信息系统国际会议，ISICO 2017，2017 年 11 月 6-8 日，印度尼西亚巴厘岛，124(2017 年 1 月 1 日):478–86。[https://doi.org/10.1016/j.procs.2017.12.180](https://doi.org/10.1016/j.procs.2017.12.180)。