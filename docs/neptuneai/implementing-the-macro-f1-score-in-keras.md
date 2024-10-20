# 在 Keras 中实现宏观 F1 分数:注意事项

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras>

作为 [TensorFlow 2.0](https://web.archive.org/web/20230224202511/https://www.tensorflow.org/guide/effective_tf2) 生态系统的一部分， [Keras](https://web.archive.org/web/20230224202511/https://keras.io/) 是用于训练和评估神经网络模型的最强大、但易于使用的深度学习框架之一。

当我们[构建神经网络模型](https://web.archive.org/web/20230224202511/https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)时，我们遵循与任何其他机器学习模型相同的模型生命周期步骤:

*   用超参数构造和编译网络，
*   适合网络，
*   评估网络，
*   使用最佳调整的模型进行预测。

具体来说，在网络评估步骤中，选择和定义适当的性能指标至关重要——本质上是判断您的模型性能的函数，包括[宏 F1 分数](https://web.archive.org/web/20230224202511/https://peltarion.com/knowledge-center/documentation/evaluation-view/classification-loss-metrics/macro-f1-score)。

## 模型性能评估指标与损失函数

预测模型的建立过程只不过是连续的反馈循环。我们构建一个初始模型，接收来自性能指标的反馈，调整模型以进行改进，并进行迭代，直到获得我们想要的预测结果。

数据科学家，尤其是机器学习/预测建模实践的新手，经常混淆**性能指标**的概念和**损失函数**的概念。为什么在训练过程中，我们试图最大化给定的评估指标，如准确性，而算法本身试图最小化完全不同的损失函数，如交叉熵？对我来说，这是一个完全有效的问题！

在我看来，答案有两部分:

1.  损失函数(如交叉熵)与评估度量(如准确性)相比，通常更容易优化，因为损失函数相对于模型参数是可微分的，而评估度量则不是；
2.  评估指标主要依赖于我们试图解决的特定业务问题陈述，对于非技术利益相关者来说更容易理解。例如，当向 C 级主管展示我们的分类模型时，解释什么是熵是没有意义的，相反，我们要展示准确度或精确度。

这两点结合起来解释了为什么损失函数和性能度量通常在相反的方向上被优化。损失函数最小化，性能指标最大化。

尽管如此，我仍然认为我们试图优化的损失函数应该与我们最关心的评估指标相对应。你能想出一个损失函数等于性能指标的场景吗？回归模型的某些指标，如 MSE(均方误差),既可以作为损失函数，也可以作为性能指标！

## 不平衡分类问题的性能度量

对于分类问题，最基本的衡量标准是准确性——正确预测与数据中样本总数的比率。开发预测模型是为了实现高准确性，就好像它是判断分类模型性能的最终权威。

毫无疑问，对于具有平衡类分布的数据集(在二进制分类中大约为 50%)，准确性是一个有效的度量。然而，当我们的数据集变得不平衡时，这是大多数现实世界业务问题的情况，准确性无法提供全貌。更糟糕的是，它可能会产生误导。

**高精度并不表示少数类**的预测能力高，少数类最有可能是感兴趣的类。如果这个概念听起来很陌生，你可以在关于[准确性悖论](https://web.archive.org/web/20230224202511/https://en.wikipedia.org/wiki/Accuracy_paradox)和[精确回忆曲线](https://web.archive.org/web/20230224202511/http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf)的论文中找到很好的解释。

现在，不平衡数据集的理想性能指标是什么？由于正确识别少数类通常是我们的目标，[召回/灵敏度、精确度、F 测量分数](https://web.archive.org/web/20230224202511/https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)将是有用的，其中:

Keras metrics

清楚地了解了评估指标、它们与损失函数的区别以及哪些指标可用于不平衡数据集后，让我们简要回顾一下 Keras 中的指标规范。对于 Keras 中可用的指标，最简单的方法是在 **model.compile()** 方法中指定 **"metrics"** 参数:

## 自 [Keras 2.0](https://web.archive.org/web/20230224202511/https://github.com/keras-team/keras/wiki/Keras-2.0-release-notes) 以来，传统评估指标——F 分数、精确度和召回率——已经从即用列表中删除。用户必须自己定义这些指标。因此，作为处理神经网络中不平衡数据集的基础，我们将重点关注在 Keras 中实现 F1 分数度量，并讨论您应该做什么，以及不应该做什么。

用海王星跟踪神经网络模型实验

```py
from keras import metrics
model.compile(loss='binary_crossentropy', optimizer='adam',
metrics=[metrics.categorical_accuracy])

```

在模型训练过程中，许多数据科学家(包括我自己)从 excel 电子表格或包含日志信息的文本文件开始，来跟踪我们的实验。这样我们可以看到什么有效，什么无效。这种方法没有错，尤其是考虑到它对我们繁琐的模型构建是多么的方便。然而，问题是这些笔记的结构并不有序。因此，当我们几年后试图回到它们时，我们不知道它们是什么意思。

## 幸运的是，海王星来救援。它跟踪并记录了我们模型训练过程中的几乎所有内容，从超参数规范到最佳模型保存，再到结果图等等。使用 Neptune 跟踪的[实验的酷之处在于，它会自动生成性能图表，用于比较不同的运行，并选择最佳的运行。这是与您的团队共享模型和结果的一个很好的方式。](/web/20230224202511/https://neptune.ai/experiment-tracking)

关于如何配置你的海王星环境和设置你的实验的更详细的解释，请查看[这个完整的指南](https://web.archive.org/web/20230224202511/https://docs.neptune.ai/getting-started/installation)。它非常简单，所以我没有必要在这里介绍海王星初始化。

我将演示如何在 Keras F1 指标实现过程中利用 Neptune，并向您展示模型训练过程变得多么简单和直观。

你兴奋吗？我们开始吧！

创造海王星实验

首先，我们需要导入所有的包和函数:

### 现在，让我们在 Neptune 中专门为这个练习创建一个项目:

接下来，我们将创建一个连接到我们的 **KerasMetricNeptune** 项目的 Neptune 实验，以便我们可以在 Neptune 上记录和监控模型训练信息:

```py
import neptune as neptune.new

import os
import pandas as pd
import numpy as np
from random import sample, seed
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout, Reshape, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

pd.options.display.max_columns = 100
np.set_printoptions(suppress=True)

os.chdir('PATH_TO_YOUR_WORK_DIRECTORY')
```

这里有两点需要注意:

**neptune.init()** 中的 *api_token* arg 取您在配置步骤中生成的 Neptune API

```py
myProject = "YourUserName/YourProjectName"
project = neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'),
                       project=myProject)
project.stop()

npt_exp = neptune.init(
        api_token=os.getenv('NEPTUNE_API_TOKEN'),
        project=myProject,
        name='step-by-step-implement-fscores',
        tags=['keras', 'classification', 'macro f-scores','neptune'])

```

**neptune.init()** 中的*标签* arg 是可选的，但是最好为给定的项目指定标签，以便于共享和跟踪。

*   随着 Neptune 项目—**KerasMetricNeptune**在我的演示中——以及成功创建的初始实验，我们可以继续进行建模部分。
*   第一次尝试:自定义 F1 得分指标

根据 [Keras 文档](https://web.archive.org/web/20230224202511/https://keras.io/api/metrics/#custom-metrics)，用户可以在神经网络编译步骤传递自定义指标。听起来很容易，不是吗？我实现了一个度量函数 ***custom_f1*** 。它接受真实结果和预测结果作为参数:

### 数据集:信用卡欺诈检测

为了展示这个自定义指标函数是如何工作的，我将使用信用卡欺诈检测数据集作为例子。这是最受欢迎的不平衡数据集之一(更多细节[在这里](https://web.archive.org/web/20230224202511/https://www.kaggle.com/mlg-ulb/creditcardfraud))。

```py

def custom_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives+K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

```

### 基本的探索性数据分析表明，0 级(99.83%)和 1 级(0.17%)存在极端的阶级不平衡:

出于演示目的，我将在我的神经网络模型中包括所有输入特征，并将 20%的数据保存为保留测试集:

使用神经网络的模型结构

```py
credit_dat = pd.read_csv('creditcard.csv')

counts = credit_dat.Class.value_counts()
class0, class1 = round(counts[0]/sum(counts)*100, 2), round(counts[1]/sum(counts)*100, 2)
print(f'Class 0 = {class0}% and Class 1 = {class1}%')

sns.set(style="whitegrid")
ax = sns.countplot(x="Class", data=credit_dat)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(p.get_height()/len(credit_dat)*100), (p.get_x()+0.15, p.get_height()+1000))
ax.set(ylabel='Count',
       title='Credit Card Fraud Class Distribution')

npt_exp['Distribution'].upload(neptune.types.File.as_image(ax.get_figure()))

dat = credit_dat

```

在对数据进行预处理之后，我们现在可以进入建模部分。在这篇文章中，我将构建一个具有两个隐藏层的神经网络用于二进制分类(使用 sigmoid 作为输出层的激活函数):

```py
def myformat(value, decimal=4):
    return str(round(value, decimal))

def Pre_proc(dat, current_test_size=0.2, current_seed=42):
    x_train, x_test, y_train, y_test = train_test_split(dat.iloc[:, 0:dat.shape[1]-1],
                                                        dat['Class'],
                                                        test_size=current_test_size,
                                                        random_state=current_seed)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    y_train, y_test = np.array(y_train), np.array(y_test)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = Pre_proc(dat)
```

### 使用自定义 F1 指标建模

接下来，我们使用交叉验证(CV)来训练模型。由于构建准确的模型超出了本文的范围，我设置了一个 5 重 CV，每个 CV 只有 20 个时期，以展示 F1 度量函数是如何工作的:

```py
def runModel(x_tr, y_tr, x_val, y_val, epos=20, my_batch_size=112):

    inp = Input(shape = (x_tr.shape[1],))

    x = Dense(1024, activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)

    return model

```

### **几个注意事项:**

预定义函数 *custom_f1* 在 model.compile 步骤中指定；

```py
f1_cv, precision_cv, recall_cv = [], [], []

current_folds = 5
current_epochs = 20
current_batch_size = 112

kfold = StratifiedKFold(current_folds, random_state=42, shuffle=True)

for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(X=x_train, y=y_train)):
    print('---- Starting fold %d ----'%(k_fold+1))

    x_tr, y_tr = x_train[tr_inds], y_train[tr_inds]
    x_val, y_val = x_train[val_inds], y_train[val_inds]

    model = runModel(x_tr, y_tr, x_val, y_val, epos=current_epochs)

    model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=[custom_f1, 'accuracy'])

    for val in history.history['custom_f1']:
            npt_exp['Custom F1 metric'].log(val)

    model.fit(x_tr,
              y_tr,
              epochs=current_epochs,
              batch_size=current_batch_size,
              verbose=1)

    y_val_pred = model.predict(x_val)
    y_val_pred_cat = (np.asarray(y_val_pred)).round()

    f1, precision, recall = f1_score(y_val, y_val_pred_cat), precision_score(y_val, y_val_pred_cat), recall_score(y_val, y_val_pred_cat)

    metric_text = f'Fold {k_fold+1} f1 score = '

    npt_exp[metric_text] = myformat(f1)

    f1_cv.append(round(f1, 6))
    precision_cv.append(round(precision, 6))
    recall_cv.append(round(recall, 6))

metric_text_final = 'Mean f1 score through CV = '
npt_exp[metric_text_final] = myformat(np.mean(f1_cv))
```

我们从我们的训练实验中提取 f1 值，并使用 *send_metric()* 函数在 Neptune 上跟踪这些 f1 值；

*   在每次折叠之后，计算性能指标，即 f1、精度和召回率，并因此使用 *send_text()* 函数发送到 Neptune
*   当整个交叉验证完成时，通过取每个 CV 的 f1 分数的平均值来计算最终的 f1 分数。同样，这个值被发送到 Neptune 进行跟踪。
*   在您启动模型后，您将立即看到 Neptune 开始跟踪训练过程，如下所示。因为还没有要记录的指标，所以在此阶段只显示 CPU 和内存信息:
*   随着模型训练的进行，会记录更多的性能指标值。单击项目 ID 旁边的小眼睛图标，我们将启用交互式跟踪图表，显示每个训练迭代期间的 f1 值:

训练过程结束后，我们可以点击项目 ID 来查看 Neptune 自动存储的所有元数据。正如您在下面的视频中所看到的，该元数据包括每个折叠的 f1 分数，以及 5 个折叠 CV 的 f1 分数的平均值。在元数据之上，图表选项显示由我们的自定义度量函数为每个时期计算的 f1 值，即 5 倍* 20 个时期= 100 个 f1 值:

目前为止一切正常！然而，当我们检查 Neptune 上的详细日志记录时，我们注意到一些意想不到的事情。**在训练期间计算的 F1 分数(例如，0.137)与为每个验证集计算的分数(例如，0.824)显著不同**。这种趋势在图表中更加明显(在右下方)，其中最大 F1 值约为 0.14。

为什么会这样？

使用回调来指定指标

深入研究这个问题，我们意识到 Keras 是通过批量创建自定义度量函数来进行计算的。每个度量在每个批次后应用，然后平均以获得特定时期的全局近似值。这个信息是误导性的，因为我们所监控的应该是每个时期的宏观训练表现。这就是为什么 Keras 2.0 版本中删除了这些指标。综上所述，实施宏观 F1 指标的正确方法是什么？答案是回调功能:

### 这里，我们定义了一个回调类 **NeptuneMetrics** 来计算和跟踪每个时期结束时的模型性能指标，也就是宏分数。

然后我们这样编译和拟合我们的模型:

```py
class NeptuneMetrics(Callback):
    def __init__(self, neptune_experiment, validation, current_fold):
        super(NeptuneMetrics, self).__init__()
        self.exp = neptune_experiment
        self.validation = validation
        self.curFold = current_fold

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_targ = self.validation[1]
        val_predict = (np.asarray(self.model.predict(self.validation[0]))).round()

        val_f1 = round(f1_score(val_targ, val_predict), 4)
        val_recall = round(recall_score(val_targ, val_predict), 4)
        val_precision = round(precision_score(val_targ, val_predict), 4)

        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precision)

        print(f' — val_f1: {val_f1} — val_precision: {val_precision}, — val_recall: {val_recall}')

	    self.exp['Epoch End Loss'].log(logs['loss'])
        self.exp['Epoch End F1-score'].log(val_f1)
        self.exp['Epoch End Precision'].log(val_precision)
        self.exp['Epoch End Recall'].log(val_recall)

        if self.curFold == 4:

            msg = f' End of epoch {epoch} val_f1: {val_f1} — val_precision: {val_precision}, — val_recall: {val_recall}'

            self.exp[f'Epoch End Metrics (each step) for fold {self.curFold}'] = msg

```

现在，如果我们重新运行 CV 训练，Neptune 将自动创建一个新的模型跟踪–在我们的示例中为 KER1-9–以便于比较(不同实验之间):

与之前一样，在训练发生时检查由新回调方法生成的详细日志记录，我们观察到我们的 NeptuneMetrics 对象为训练过程和验证生成一致的 F1 分数(大约 0.7-0.9)，如 Neptune 视频剪辑所示:

```py
model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=[])
model.fit(x_tr,
          y_tr,
          callbacks=[NeptuneMetrics(npt_exp, validation=(x_val, y_val), current_fold=k_fold)],  
          epochs=current_epochs,
          batch_size=current_batch_size,
          verbose=1)

```

完成模型训练后，让我们检查并确认在最后一个 CV 文件夹的每个(时期)步骤中记录的性能指标符合预期:

太好了！一切看起来都在合理的范围内。

让我们比较一下我们刚刚试验的这两种方法之间的区别，即**自定义 F1 指标与 NeptuneMetrics 回调**:

我们可以清楚地看到，**自定义 F1 指标(左边)**实现是不正确的，而 **NeptuneMetrics 回调**实现是理想的方法！

现在，最后一次检查。使用回调方法预测测试集给我们提供了 F1 分数= 0.8125，这与训练相当接近:

最后的想法

你有它！在您的神经网络模型中计算和监控 F1 分数的正确和不正确方法。如果你感兴趣的话，相似的过程可以应用于召回和精确。我希望这篇博客对你有所帮助。完整的代码可以在[这个 Github repo](https://web.archive.org/web/20230224202511/https://github.com/YiLi225/NeptuneBlogs/blob/main/Implement_F1score_neptune_git_NewVersion.py) 中找到，整个 Neptune 模型可以在[这里](https://web.archive.org/web/20230224202511/https://app.neptune.ai/katyl/KerasMetricNeptuneNewVersion/experiments?split=bth&dash=charts&viewId=standard-view)找到。

```py
def predict(x_test):
    model_num = len(models)
    for k, m in enumerate(models):
        if k==0:
            y_pred = m.predict(x_test, batch_size=current_batch_size)
        else:
            y_pred += m.predict(x_test, batch_size=current_batch_size)

    y_pred = y_pred / model_num

    return y_pred

y_test_pred_cat = predict(x_test).round()

cm = confusion_matrix(y_test, y_test_pred_cat)
f1_final = round(f1_score(y_test, y_test_pred_cat), 4)

npt_exp['TestSet F1 score'] = myformat(f1_final)

from scikitplot.metrics import plot_confusion_matrix
fig_confmat, ax = plt.subplots(figsize=(12, 10))
plot_confusion_matrix(y_test, y_test_pred_cat.astype(int).flatten(), ax=ax)

npt_exp['Confusion Matrix'].upload(neptune.types.File.as_image(fig_confmat))
npt_exp.stop()

```

## 在我让你走之前，这个 NeptuneMetrics 回调计算 F1 分数，但这并不意味着模型是在 F1 分数上训练的。为了基于优化 F1 分数进行“训练”,这有时是处理不平衡分类的首选，我们需要额外的模型/回调配置。请继续关注我的下一篇文章，在那里我将讨论 F1 分数调整和阈值移动。感谢阅读！

There you have it! The correct and incorrect ways to calculate and monitor the F1 score in your neural network models. Similar procedures can be applied for recall and precision if it’s your measure of interest. I hope that you find this blog helpful. The full code is available in [this Github repo](https://web.archive.org/web/20230224202511/https://github.com/YiLi225/NeptuneBlogs/blob/main/Implement_F1score_neptune_git_NewVersion.py), and the entire Neptune model can be found [here](https://web.archive.org/web/20230224202511/https://app.neptune.ai/katyl/KerasMetricNeptuneNewVersion/experiments?split=bth&dash=charts&viewId=standard-view).

Before I let you go, this NeptuneMetrics callback calculates the F1 score, but it doesn’t mean that the model is trained on the F1 score. In order to ‘train’ based on optimizing the F1 score, which sometimes is preferred for handling imbalanced classification, we need additional model/callback configurations. Stay tuned for my next article, where I will be discussing F1 score tuning and threshold-moving. Thanks for reading!