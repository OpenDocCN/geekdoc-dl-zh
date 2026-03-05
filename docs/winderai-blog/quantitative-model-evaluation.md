# 定量模型评估

> 原文：<https://winder.ai/quantitative-model-evaluation/>

# 定量模型评估

欢迎光临！本车间来自 [Winder.ai](https://Winder.ai/?utm_source=winderresearch&utm_medium=notebook&utm_campaign=workshop&utm_term=individual) 。注册以获得更多免费的研讨会、培训和视频。

我们需要能够比较一系列任务的模型。最常见的用例是确定对模型的更改是否会提高性能。通常，我们希望看到这一点，我们将在另一个研讨会上这样做，但首先我们需要建立一些性能的量化指标。

```py
from sklearn import metrics
import numpy as np 
```

## 准确(性)

首先，让我们深入了解一下准确性。根据下面的混淆矩阵，计算准确度。

我用‘a’表示正，用‘n’表示负。这是因为自然排序会使混淆矩阵变得混乱。(也就是说，如果我们使用 y 和 n，y 会在底部！)

```py
y_true = ['a'] * 15 + ['n'] * 35
y_pred = ['a'] * 12 + ['n'] * 3 + ['n'] * 30 + ['a'] * 5
cm = metrics.confusion_matrix(y_true, y_pred)
print(cm)
print(metrics.accuracy_score(y_true, y_pred)) 
```

```py
[[12  3]
 [ 5 30]]
0.84 
```

请注意，这里也有相当多的偏斜。

## 预期值

想象一下，我们有一个营销示例，就像培训中一样。这个想法是，我们想花一些钱在营销上，但我们只想针对有意义的人。我们得到了以下信息:

*   每笔销售利润:50 英镑
*   营销成本:9

我们可以生成如下成本/收益矩阵:

```py
profit = 50
cost   = -9
cost_benefit = np.array([[profit+cost, cost],[0   , 0]])
print(cost_benefit) 
```

```py
[[41 -9]
 [ 0  0]] 
```

给定前面混淆矩阵中的结果，期望值是多少？

```py
def expected_value(confusion_matrix, cost_benefit_matrix):
    return sum(sum(confusion_matrix * cost_benefit_matrix)) / sum(sum(confusion_matrix))

print(expected_value(cm, cost_benefit)) 
```

```py
9.3 
```

## 不平衡阵列

让我们来看看在培训中看到的两个困惑矩阵。

```py
model_a = np.array([[25, 30], [0, 45]])
model_b = np.array([[30, 0], [20, 50]])
print("model a:\n", model_a)
print("model b:\n", model_b) 
```

```py
model a:
 [[25 30]
 [ 0 45]]
model b:
 [[30  0]
 [20 50]] 
```

让我们根据之前的成本/收益矩阵来计算这些模型的预期价值:

```py
print("model a:\n", expected_value(model_a, cost_benefit))
print("model b:\n", expected_value(model_b, cost_benefit)) 
```

```py
model a:
 7.55
model b:
 12.3 
```

但是看看每个模型测试集的大小，有一个很大的偏差。

```py
print("model a sample size:\n", sum(model_a))
print("model b sample size:\n", sum(model_b)) 
```

```py
model a sample size:
 [25 75]
model b sample size:
 [50 50] 
```

## 分解样本量

我们可以用概率恒等式分解出样本量:

$p(\mathbf{Y},\mathbf{n}) = p(\mathbf{n})\cdotp(\mathbf{Y} \vert \mathbf{n})$

也就是说，我们可以通过以下因素来确定样本量:

\ begin { align } \ \
\ text { expected profit } =&p(\ mathbf { p })\ cdot \ left[p(\ mathbf { Y } \ vert \ mathbf { p })\ cdot b(\ mathbf { Y }，\ mathbf { p })+p(\ mathbf { N } \ vert \ mathbf { p })\ cdot b(\ mathbf { N }，\ mathbf { p } \ right]+\ \
【T4(\ p

让我们创建一个实现上述等式的新方法。我们将首先使用与上面提供的数据中相同的类偏斜，并检查它是否产生与以前相同的值(健全性检查！).

```py
def factored_expected_value(m, cb, p_p=0.5, p_n=0.5):
    t_p = sum(m[:,0])
    t_n = sum(m[:,1])
    return p_p * (m[0,0]/t_p) * cb[0,0] + (m[1,0]/t_p) * cb[1,0] + \
           p_n * (m[0,1]/t_n) * cb[0,1] + (m[1,1]/t_n) * cb[1,1]

print("Should be equal:", expected_value(model_a, cost_benefit), factored_expected_value(model_a, cost_benefit, 0.25, 0.75))
print("Should be equal:", expected_value(model_b, cost_benefit), factored_expected_value(model_b, cost_benefit)) 
```

```py
Should be equal: 7.55 7.55
Should be equal: 12.3 12.3 
```

现在，让我们看看当我们排除类偏差时期望值是什么(通过改变上面等式中每个类的概率):

```py
print("Results after factoring out training sample skew")
print("================================================")
print("Model A expected value:", factored_expected_value(model_a, cost_benefit, 0.5, 0.5))
print("Model B expected value:", factored_expected_value(model_b, cost_benefit, 0.5, 0.5)) 
```

```py
Results after factoring out training sample skew
================================================
Model A expected value: 18.7
Model B expected value: 12.3 
```

请注意，在考虑了类偏差之后，期望值是多么的不同！

这表明当你比较模型时，确保它们有相同的分类概率。它们不一定要有相等的类别偏差，但是您需要比较具有相同偏差的模型。

还要注意，我们可以通过改变底层数据来修复偏差；即平衡测试数据中的类别。

## 其他评估指标

让我们快速看一下我们在培训中定义的其他一些评估指标。我们可以手动计算或使用`sklearn.metrics`模块中的方法。

这是技术指标。选择最适合您的问题的技术指标是明智的。比如你是不是更担心误报？然后，确保您使用的指标考虑了误报(如误报率)。

但是请记住，这些很难被非数据科学家解释。也就是说，你不应该向企业展示它们。总是试图找到业务友好的汇总统计数据，如上面定义的期望值。

$ $ \ text { accuracy } = \ frac { \ text { correct predictions } } { \ text { all instances } } = \ frac { TP+TN } = \ frac { TP+TN } { TP+FP+TN+FN } $ $

$ $ \ text { precision } = \ frac { \ text { true positives } } { \ text { all predicted yes } } = \ frac { TP } { TP+FP } $ $

$ $ \ text { recall } = \ frac { \ text { true positives } } { \ text { all positives } } = \ frac { TP } { TP+FN } $ $

$ $ \ text {假阳性率} = \ frac { \ text {假阳性} } { \ text {所有阴性} } = \ frac { FP } { N } = \ frac { FP } { FP+TN } $ $

```py
print("model a:\n", model_a, "\n", np.array([['TP', 'FN'],['FP', 'TN']]), "\n")

TP = model_a[0,0]; FP = model_a[1,0];
TN = model_a[1,1]; FN = model_a[0,1]

print("Accuracy:", (TP+TN)/(TP+FP+TN+FN) )
print("Precision:", (TP)/(TP+FP) )
print("Recall:", (TP)/(TP+FN) )
print("FPR:", (FP)/(FP+TN) ) 
```

```py
model a:
 [[25 30]
 [ 0 45]]
 [['TP' 'FN']
 ['FP' 'TN']]

Accuracy: 0.7
Precision: 1.0
Recall: 0.454545454545
FPR: 0.0 
```

```py
y_true = ['a'] * 25 + ['n'] * 75
y_pred = ['a'] * 20 + ['n'] * 30 + ['n'] * 15 + ['a'] * 35 
```

## 任务

*   为上述数据创建一个混淆矩阵
*   计算准确率、预见率、召回率和 FPR

## 奖金

如果你提前结束了，去喝杯咖啡，或者试试这些任务:

*   再次调出数字数据集
*   尝试生成所有这些指标(准确度、精确度、召回率等。)用于该数据集。
*   调查 sklearn 还能提供哪些指标，并尝试一下。