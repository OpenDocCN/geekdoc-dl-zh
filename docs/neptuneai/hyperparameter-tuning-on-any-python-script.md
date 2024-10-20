# 如何通过 3 个简单的步骤在任何 Python 脚本上进行超参数调优

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/hyperparameter-tuning-on-any-python-script>

你写了一个 Python 脚本，用来训练和评估你的机器学习模型。现在，您想自动调整超参数以提高其性能吗？

我抓住你了！

在本文中，我将向您展示如何将您的脚本转换成可以用任何超参数优化库进行优化的目标函数。

只需 3 个步骤，你就可以像没有明天一样调整模型参数。

准备好了吗？

我们走吧！

我想你的`main.py`脚本应该是这样的:

```py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/train.csv', nrows=10000)
X = data.drop(['ID_code', 'target'], axis=1)
y = data['target']
(X_train, X_valid, 
y_train, y_valid )= train_test_split(X, y, test_size=0.2, random_state=1234)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

params = {'objective': 'binary',
          'metric': 'auc',
          'learning_rate': 0.4,
          'max_depth': 15,
          'num_leaves': 20,
          'feature_fraction': 0.8,
          'subsample': 0.2}

model = lgb.train(params, train_data,
                  num_boost_round=300,
                  early_stopping_rounds=30,
                  valid_sets=[valid_data],
                  valid_names=['valid'])

score = model.best_score['valid']['auc']
print('validation AUC:', score)
```

## 步骤 1:从代码中分离搜索参数

获取您想要调整的参数，并将它们放在脚本顶部的字典中。这样做可以有效地将搜索参数从代码的其余部分中分离出来。

```py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

SEARCH_PARAMS = {'learning_rate': 0.4,
                 'max_depth': 15,
                 'num_leaves': 20,
                 'feature_fraction': 0.8,
                 'subsample': 0.2}

data = pd.read_csv('../data/train.csv', nrows=10000)
X = data.drop(['ID_code', 'target'], axis=1)
y = data['target']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1234)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

params = {'objective': 'binary',
          'metric': 'auc',
          **SEARCH_PARAMS}

model = lgb.train(params, train_data,
                  num_boost_round=300,
                  early_stopping_rounds=30,
                  valid_sets=[valid_data],
                  valid_names=['valid'])

score = model.best_score['valid']['auc']
print('validation AUC:', score)
```

## 步骤 2:将培训和评估打包成一个功能

现在，您可以将整个训练和评估逻辑放在一个`train_evaluate`函数中。该函数将参数作为输入，并输出验证分数。

```py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

SEARCH_PARAMS = {'learning_rate': 0.4,
                 'max_depth': 15,
                 'num_leaves': 20,
                 'feature_fraction': 0.8,
                 'subsample': 0.2}

def train_evaluate(search_params):
    data = pd.read_csv('../data/train.csv', nrows=10000)
    X = data.drop(['ID_code', 'target'], axis=1)
    y = data['target']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1234)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    params = {'objective': 'binary',
              'metric': 'auc',
              **search_params}

    model = lgb.train(params, train_data,
                      num_boost_round=300,
                      early_stopping_rounds=30,
                      valid_sets=[valid_data],
                      valid_names=['valid'])

    score = model.best_score['valid']['auc']
    return score

if __name__ == '__main__':
    score = train_evaluate(SEARCH_PARAMS)
    print('validation AUC:', score)
```

## 步骤 3:运行 hypeparameter 调整脚本

我们快到了。

你现在需要做的就是使用这个`train_evaluate`函数作为你选择的黑盒优化库的目标。

我将使用 [Scikit Optimize](https://web.archive.org/web/20221007001227/https://scikit-optimize.github.io/) ，我已经在[的另一篇文章](/web/20221007001227/https://neptune.ai/blog/scikit-optimize)中详细描述了它，但是你可以使用任何超参数优化库。

### 了解更多信息

💡使用 [Scikit-Optimize + Neptune 集成，检查如何在运行时可视化运行，记录每次运行时尝试的参数，等等。](https://web.archive.org/web/20221007001227/https://docs.neptune.ai/integrations-and-supported-tools/hyperparameter-optimization/scikit-optimize)

简单地说，我

*   定义搜索 ***空间*** ，
*   创建将被最小化的`objective`函数，
*   通过`skopt.forest_minimize`功能运行优化。

在这个例子中，我将从 **10** 随机选择的参数集开始，尝试 **100** 种不同的配置。

```py
import skopt

from script_step2 import train_evaluate

SPACE = [
    skopt.space.Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
    skopt.space.Integer(1, 30, name='max_depth'),
    skopt.space.Integer(2, 100, name='num_leaves'),
    skopt.space.Real(0.1, 1.0, name='feature_fraction', prior='uniform'),
    skopt.space.Real(0.1, 1.0, name='subsample', prior='uniform')]

@skopt.utils.use_named_args(SPACE)
def objective(**params):
    return -1.0 * train_evaluate(params)

results = skopt.forest_minimize(objective, SPACE, n_calls=30, n_random_starts=10)
best_auc = -1.0 * results.fun
best_params = results.x

print('best result: ', best_auc)
print('best parameters: ', best_params)
```

这就是了。

***结果*** 对象包含关于产生它的**最佳分数** **和参数**的信息。

* * *

请注意，由于最近的 [API 更新](/web/20221007001227/https://neptune.ai/blog/neptune-new)，这篇文章也需要一些改变——我们正在努力！与此同时，请检查[海王星文档](https://web.archive.org/web/20221007001227/https://docs.neptune.ai/)，那里的一切都是最新的！🥳

* * *

### 注意:

如果您想**可视化您的训练并在训练结束后保存诊断图表**，您可以添加一个回调和一个函数调用来**将每个超参数搜索**记录到 Neptune。只需使用 neptune-contrib 库中的这个[助手函数。](https://web.archive.org/web/20221007001227/https://neptune-contrib.readthedocs.io/_modules/neptunecontrib/monitoring/skopt.html#NeptuneMonitor)

```py
import neptune
import neptunecontrib.monitoring.skopt as sk_utils
import skopt

from script_step2 import train_evaluate

neptune.init('jakub-czakon/blog-hpo')
neptune.create_experiment('hpo-on-any-script', upload_source_files=['*.py'])

SPACE = [
    skopt.space.Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
    skopt.space.Integer(1, 30, name='max_depth'),
    skopt.space.Integer(2, 100, name='num_leaves'),
    skopt.space.Real(0.1, 1.0, name='feature_fraction', prior='uniform'),
    skopt.space.Real(0.1, 1.0, name='subsample', prior='uniform')]

@skopt.utils.use_named_args(SPACE)
def objective(**params):
    return -1.0 * train_evaluate(params)

monitor = sk_utils.NeptuneMonitor()
results = skopt.forest_minimize(objective, SPACE, n_calls=100, n_random_starts=10, callback=[monitor])
sk_utils.log_results(results)

neptune.stop()
```

现在，当您运行参数扫描时，您将看到以下内容:

## 最后的想法

在本文中，您已经学习了如何通过 3 个步骤优化几乎所有 Python 脚本的超参数。

希望有了这些知识，你会用更少的努力建立更好的机器学习模型。

快乐训练！

### 雅各布·查孔

大部分是 ML 的人。构建 MLOps 工具，编写技术资料，在 Neptune 进行想法实验。

* * *

**阅读下一篇**

## 如何跟踪机器学习模型的超参数？

卡米尔·卡什马雷克|发布于 2020 年 7 月 1 日

**机器学习算法可通过称为超参数**的多个量规进行调整。最近的深度学习模型可以通过数十个超参数进行调整，这些超参数与数据扩充参数和训练程序参数一起创建了非常复杂的空间。在强化学习领域，您还应该计算环境参数。

数据科学家要**控制好** **超参数** **空间**，才能**使** **进步**。

在这里，我们将向您展示**最近的** **实践**，**提示&技巧，**和**工具**以最小的开销高效地跟踪超参数。你会发现自己掌控了最复杂的深度学习实验！

## 为什么我应该跟踪我的超参数？也就是为什么这很重要？

几乎每一个深度学习实验指南，像[这本深度学习书籍](https://web.archive.org/web/20221007001227/https://www.deeplearningbook.org/contents/guidelines.html)，都建议你如何调整超参数，使模型按预期工作。在**实验-分析-学习循环**中，数据科学家必须控制正在进行的更改，以便循环的“学习”部分正常工作。

哦，忘了说**随机种子也是一个超参数**(特别是在 RL 领域:例如检查[这个 Reddit](https://web.archive.org/web/20221007001227/https://www.reddit.com/r/MachineLearning/comments/76th74/d_why_random_seeds_sometimes_have_quite_large/) )。

## 超参数跟踪的当前实践是什么？

让我们逐一回顾一下管理超参数的常见做法。我们关注于如何构建、保存和传递超参数给你的 ML 脚本。

[Continue reading ->](/web/20221007001227/https://neptune.ai/blog/how-to-track-hyperparameters)

* * *