# 回归:处理异常值

> 原文：<https://winder.ai/regression-dealing-with-outliers/>

# 回归:处理异常值

欢迎光临！本车间来自 [Winder.ai](https://Winder.ai/?utm_source=winderresearch&utm_medium=notebook&utm_campaign=workshop&utm_term=individual) 。注册以获得更多免费的研讨会、培训和视频。

异常值是虚假的观察值。您通常可以直观地发现异常值；它们常常远离其余的观测。

有时它们是由测量误差引起的，有时是噪声，有时是感兴趣的观察结果(例如欺诈检测)。

但是异常值会扭曲平均值和标准偏差的估计值，因此会影响使用假设正态性的误差度量(例如最小平方误差)的线性模型。

```py
# Usual imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from sklearn.linear_model import LinearRegression 
```

```py
np.random.seed(42)  # To ensure we get the same data every time.
X = 2 * np.random.rand(50, 1)
y = 8 + 6 * X + np.random.randn(50, 1)
y[10:20] = y[10:20] + 10 
```

```py
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_ 
```

```py
(array([ 10.64908656]), array([[ 5.26889785]])) 
```

```py
X_new = np.array([[0], [2]])
y_predict = np.array([lin_reg.intercept_, X_new[1]*lin_reg.coef_ + lin_reg.intercept_])
print(X_new.shape)
print(y_predict.shape)

plt.scatter(X, y, color='red', marker='o')
plt.scatter(X[10:20], y[10:20], color='k', marker='x')
plt.plot(X_new, y_predict, "b-")
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.axis([0, 2, 8, 22])
plt.show() 
```

```py
(2, 1)
(2, 1) 
```

![png](img/b30b441a2a5102ea6f245d859a397a31.png)

我们可以看到它影响了回归。但我们也可以看到，最佳拟合线仍然拥抱原始数据。这是一堂有趣的课。更多的数据减轻了离群值的影响。

尽管如此，我们可以做得更好。

RANSAC 是一种算法，它重复对数据进行子采样，然后在该子集上训练模型。在这已经被执行多次之后，参数以某种方式被组合以做出更稳健的估计。这种方法可以用于任何算法。

注意，SKLearn 中的实现工作方式稍有不同，它试图选择不是异常值的观察值，但前提是相似的。

```py
from sklearn.linear_model import RANSACRegressor

# Robustly fit linear model with RANSAC algorithm
model_ransac = RANSACRegressor(LinearRegression())
model_ransac.fit(X, y) 
```

```py
RANSACRegressor(base_estimator=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False),
        is_data_valid=None, is_model_valid=None, loss='absolute_loss',
        max_skips=inf, max_trials=100, min_samples=None, random_state=None,
        residual_metric=None, residual_threshold=None, stop_n_inliers=inf,
        stop_probability=0.99, stop_score=inf) 
```

```py
plt.scatter(X, y, color='red', marker='o', label='raw')
plt.plot(X_new, y_predict, "b-", label='MSE')
plt.plot(X_new, model_ransac.predict(X_new), "g-", label='RANSAC')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.tight_layout()
plt.axis([0, 2, 8, 22])
plt.show() 
```

![png](img/4929f360c7ab74ff90c72128c7830ae2.png)

请注意更好的拟合。RANSAC 工作得如何取决于您有多少离群值，它们是如何分布的，以及您的内联程序有多接近正态分布。