# 销售中的数据科学:客户情绪分析

> 原文：<https://web.archive.org/web/20221129050302/https://www.datacamp.com/blog/data-science-in-sales-customer-sentiment-analysis>

![Artificial Intelligence Concept Illustration](img/b2822b6fc98f37bf51bf2bcc2209c322.png)

*   [销售中的数据科学用例:分析客户情绪](#data-science-use-case-in-sales:-analyzing-customer-sentiment)
*   [准备数据集](#preparing-the-dataset)
*   [应用弓法](#applying-the-bow-method)
*   [使用监督机器学习模型预测情绪](#using-a-supervised-machine-learning-model-to-predict-sentiment)
*   [结论](#conclusion)

数据科学用例几乎可以与任何积累了大量数据的行业相关。商店和电子商务网站是被营销活动吸引的人们的真实客户体验发生的地方，也是收集特定品牌或公司的有价值的购买数据的地方。在这里，人们做出最终决定，他们是否真的想购买某种产品，他们是否有兴趣购买他们之前没有计划的其他东西，他们准备支付多少钱，他们是否会回到这家商店，以及他们会留下什么关于他们的客户体验的评论。

事实上，客户评论构成了分析和理解在整个销售过程中可以改变或加强的数据的坚实来源。以这种方式分析数据可以降低成本、提高运营效率、改善客户体验、发现新机会、发展业务并最终增加收入。让我们仔细看看如何使用数据科学算法对这些宝贵的信息进行分析和建模，以获得隐藏的见解，并从每个客户那里捕捉整体信息。

## 销售中的数据科学用例:分析客户情绪

客户情感分析是在客户使用某家公司的服务或产品时，识别客户情感的自动化过程。这通常是从在线调查、社交媒体、支持票、反馈表、产品评论、论坛、电话、电子邮件和聊天机器人中收集的非结构化文本数据。在机器学习中，客户情绪分析是通过自然语言处理(NLP)进行的，NLP 应用统计和语言方法直接从文本数据中提取积极、消极和中性的情绪。本质上，它输出两个参数:

*   极性:表明一种情绪是积极的还是消极的。
*   数量:表明该情绪的强度。

客户情绪分析是任何现代企业的一个关键工具，因为它有助于获得可操作的见解，发现并解决让客户感到不愉快的关键重复问题，强化导致客户积极情绪的产品或服务功能，并在整体上做出更有效的数据驱动型决策。在更精细的层面上，客户情绪分析使我们能够:

*   改善客户服务，从而改善客户体验，
*   提高客户忠诚度，
*   降低流失率，
*   及时升级产品和服务，
*   优化营销活动，
*   预测新的趋势和市场，
*   维护我们公司的高声誉，
*   增加利润。

与任何文本分析任务一样，在进行客户情感分析时会遇到一些陷阱。例如，NLP 算法没有捕捉到一些评论中的讽刺，并将它们错误地分类。它有时也无法破译非常具体的缩写或很少使用的俚语。

## 准备数据集

让我们使用 IMDB 电影评论数据集来探索客户情绪分析在实践中是如何工作的:

```py
import pandas as pd

movies = pd.read_csv('movies.csv', index_col=0).reset_index(drop=True)
print(f'Number of reviews: {movies.shape[0]:,}\n')
print(movies.head())
```

```py
Number of reviews: 7,501

                                              review  label
0  This short spoof can be found on Elite's Mille...      0
1  A singularly unfunny musical comedy that artif...      0
2  An excellent series, masterfully acted and dir...      1
3  The master of movie spectacle Cecil B. De Mill...      1
4  I was gifted with this movie as it had such a ...      0
```

我们有两列:一列是每个评论的文本，另一列是对整体情绪的估计:正面(1)或负面(0)。

让我们来计算正面和负面评论的百分比:

```py
round(movies['label'].value_counts()*100/len(movies['label'])).convert_dtypes()
```

```py
0    50
1    50
Name: label, dtype: Int64
```

因此，我们有几乎相等比例的正面和负面评论。

## 运用弓法

我们的下一步将是将文本数据转换为数字形式，因为机器学习模型只能处理数字特征。特别是，我们将创建计算每个单词在各自的评论中出现的次数的功能。用于此目的的最基本和最直接的方法被称为词袋(BOW ),它建立文档中出现的所有词的词汇表，并统计每个词在每个评论中的频率。因此，我们将获得新的特征，每个单词一个，具有相应的频率。

让我们将 BOW 方法应用于我们的数据集:

```py
from sklearn.feature_extraction.text import CountVectorizer

# Creating new features
vect = CountVectorizer(max_features=200)
vect.fit(movies.review)
X_review = vect.transform(movies.review)
X_df = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())

# Combining the new features with the label
movies_bow = pd.concat([movies['label'], X_df], axis=1)
print(movies_bow.head())
```

```py
 label  10  about  acting  action  actors  actually  after  again  all  ...  \
0      0   0      0       0       0       0         0      0      0    0  ...  
1      0   1      0       1       0       1         0      0      0    3  ...  
2      1   0      0       0       0       0         0      1      0    0  ...  
3      1   0      0       0       1       0         0      0      0    0  ...  
4      0   1      0       0       1       0         0      0      0    3  ...  

  will  with  without  work  world  would  years  you  young  your 
0     0     1        0     0      0      1      0    0      0     0 
1     2     7        1     0      0      2      0    3      0     2 
2     0     2        0     0      0      0      0    0      1     0 
3     0     0        0     0      0      0      0    1      1     0 
4     0     2        0     1      0      0      0    0      0     0 

[5 rows x 201 columns]
```

上面，我们应用了一个可选参数 max_features，只考虑 200 个最常用的单词，避免潜在的模型过拟合。

## 使用监督机器学习模型来预测情感

现在，我们将使用监督机器学习模型来预测情绪。因为我们想要基于已经标记的评论来估计来自新评论的情感是属于积极的还是消极的类别，所以我们不得不再次处理分类问题。同样，让我们使用逻辑回归算法并测量模型准确性:

```py
Accuracy score: 0.754

Confusion matrix:
[[37.62772101 13.23856064]
[11.32829853 37.80541981]]
```

我们看到，该模型将所有积极的评论中的 11%标记为消极的，将 13%标记为积极的，尽管它们是消极的。作为提高模型准确性的可能方法，我们可以考虑排除停用词(即，出现太频繁的低信息量词，例如“about”、“will”、“you”等)。)和增加词汇量。

当我们应用 BOW 方法时，我们可能会在我们的数据框架中有数百甚至数千个新特征。这会导致创建一个过于复杂的模型:过度拟合，有太多不必要的特征和参数。解决这个问题的一个方法是使用正则化，正则化会限制模型的功能。这里要调整的参数是 C，代表正则化的强度。让我们测试这个参数的两个值:100 和 0.1，看看哪一个在测试数据上给我们最好的模型性能:

```py
lr_1 = LogisticRegression(C=100)
lr_1.fit(X_train, y_train)
predictions_1 = lr_1.predict(X_test)

lr_2 = LogisticRegression(C=0.1)
lr_2.fit(X_train, y_train)
predictions_2 = lr_2.predict(X_test)

print(f'Accuracy score, lr_1 model: {round(accuracy_score(y_test, predictions_1), 3)}\n'
      f'Accuracy score, lr_2 model: {round(accuracy_score(y_test, predictions_2), 3)}\n\n'
      f'Confusion matrix for lr_1 model, %:\n{confusion_matrix(y_test, predictions_1)/len(y_test)*100}\n\n'
      f'Confusion matrix for lr_2 model, %:\n{confusion_matrix(y_test, predictions_2)/len(y_test)*100}')
```

```py
Accuracy score, lr_1 model: 0.753
Accuracy score, lr_2 model: 0.756

Confusion matrix for lr_1 model, %:
[[37.53887161 13.32741004]
[11.32829853 37.80541981]]

Confusion matrix for lr_2 model, %:
[[37.67214571 13.19413594]
[11.19502443 37.93869391]]
```

当使用参数 C 的选定值时，模型精度的差异是不明显的。通过进一步试验这个参数的更多值，我们可能会找到一个更能提高模型性能的参数。然而，这里我们只有 200 个新特征，所以我们的模型并不复杂，正则化步骤在我们的例子中并不真正需要。

可以使用 predict_proba 来预测情感概率，而不是使用 predict 函数来预测标签 0 或 1。这里，我们必须记住，我们不能直接将准确度分数或混淆矩阵应用于预测的概率，因为这些度量标准只对类有效。因此，我们需要将它们进一步编码为类。默认情况下，大于或等于 0.5 的概率转换为 1 类，否则转换为 0 类。

```py
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predicting the probability of the 0 class
predictions_prob_0 = lr.predict_proba(X_test)[:, 0]

# Predicting the probability of the 1 class
predictions_prob_1 = lr.predict_proba(X_test)[:, 1]

print(f'First 10 predicted probabilities of class 0: {predictions_prob_0[:10].round(3)}\n'
      f'First 10 predicted probabilities of class 1: {predictions_prob_1[:10].round(3)}')
```

```py
First 10 predicted probabilities of class 0: [0.246 0.143 0.123 0.708 0.001 0.828 0.204 0.531 0.121 0.515]
First 10 predicted probabilities of class 1: [0.754 0.857 0.877 0.292 0.999 0.172 0.796 0.469 0.879 0.485]
```

## 结论

还有许多其他有用的方法可以应用于我们的数据集，以进行更精细的情感分析:

*   使用 n 元语法(单词的组合)而不仅仅是单个单词来保留上下文，
*   排除停用词，
*   基于较高或较低频率值限制词汇的大小，
*   创建描述每个评论的长度或标点符号数量的数字额外特征(后者有时可以与情感的大小相关联)，
*   排除数字，某些字符，一定长度的单词，或者考虑更复杂的单词模式，
*   应用词干化和词汇化，即将单词简化到它们的词根，
*   使用更复杂的方法而不是 BOW 来创建词汇表，例如 TfIdf(术语频率逆文档频率),其说明一个单词相对于其余评论在评论中出现的频率。
*   使用一些专门为情感分析设计的库，如 TextBlob、SentiWordNet、VADER(价感知词典和情感推理器)。

如果你有兴趣探索这些和其他有用的技术来进行深刻的情感分析，请查看 Python 中的[情感分析课程。](https://web.archive.org/web/20220523130336/https://app.datacamp.com/learn/courses/sentiment-analysis-in-python)