# 测量机器学习中的偏差:统计偏差测试

> 原文：<https://web.archive.org/web/20230101103301/https://www.datacamp.com/blog/measuring-bias-in-machine-learning-the-statistical-bias-test>

[![](img/8bf5b52bf0807ae1d7ae4e30dc28bd20.png)](https://web.archive.org/web/20220529050733/https://www.datacamp.com/discover/enterprise)

本文由 DataRobot 的数据科学家 Sarah Khatry 和 Haniyeh Mahmoudian 撰写。

近年来，机器学习模型中的偏差问题一直是许多关注的主题。模型出问题的故事成为头条新闻，人道主义律师、政治家和记者都参与了关于我们希望在我们建立的模型中反映出什么样的伦理和价值观的讨论。

虽然人类的偏见是一个棘手的问题，并不总是容易定义，但机器学习中的偏见归根结底是数学上的。您可以对模型执行许多不同类型的测试，以识别其预测中不同类型的偏差。执行哪种测试主要取决于您所关心的内容以及使用模型的环境。

最广泛适用的测试之一是统计奇偶性，本实践教程将介绍这一点。现在，偏见总是相对于由数据中受保护的属性识别的不同人群进行评估，例如，种族、性别、年龄、性别、国籍等。

使用统计奇偶校验，您的目标是衡量不同的组是否有相同的概率获得有利的结果。一个典型的例子是招聘模型，在这个模型中，你希望确保男性和女性申请人有同等的被录用的可能性。在一个有偏见的模型中，你会发现一个群体享有特权，被雇佣的可能性更高，而另一个群体则处于劣势。

为了演示这在实践中是如何工作的，我们将首先用我们预先定义的偏差构建合成数据，然后通过分析确认数据反映了我们想要的情况，最后应用统计奇偶检验。

## 生成合成数据

在本教程中，我们将使用 Python 中的 pandas 包，但是这个过程中的每一步也可以在 r。

```py
import pandas as pd 
```

要生成具有一个受保护属性和模型预测的合成数据，我们首先需要指定一些输入:记录总数、受保护属性本身(这里是两个通用值 A 和 B)以及与有利结果相关联的模型预测，在本例中是值 1。

```py
num_row = 1000 # number of rows in the data
prot_att = 'prot_att' # column name for protected attribute
model_pred_bin = 'prediction' # column name for predictions
pos_label = 1 # prediction value associated to positive/favorable outcome
prot_group = ['A','B'] # two groups in our protected attribute 
```

正如在现实生活中，A 组和 B 组在我们的数据中可能不是均匀分布的。在下面的代码中，我们决定数据中 60%的人口来自特权群体 B，他们有 30%的机会获得有利的结果。非特权组 A 将构成剩余的 40%的数据，并且只有 15%的概率获得有利的结果。

```py
priv_g = 'B'  # privileged group
priv_g_rate = 60 # 60% of the population is from group B
priv_p_rate = 30 # 30% of the predictions for group B was for favorable outcome 1
unpriv_p_rate = 15 # 15% of the predictions for group A was for favorable outcome 1
biased_list_priv = [prot_group[0]] * (100 - priv_g_rate) + [prot_group[1]] * priv_g_rate
biased_list_pos_priv = [0] * (100 - priv_p_rate) + [1] * priv_p_rate
biased_list_pos_unpriv = [0] * (100 - unpriv_p_rate) + [1] * unpriv_p_rate 
```

对于数据的每个记录，我们随机分配一个保护组和一个预测，使用我们之前指定的偏差作为权重，然后从记录列表中创建一个数据帧。

```py
list_df = [] # empty list to store the synthetic records
for i in range(0, num_row):
   # generating random value representing protected groups with bias towards B
   prot_rand = random.choices(biased_list_priv)[0]
   if prot_rand == priv_g:
       prot_g = priv_g
       # generating random binary value representing prediction with bias towards 0
       pred = random.choices(biased_list_pos_priv)[0]
       # adding the new record to the list
       list_df.append([prot_g,pred])
   else:
       prot_g = prot_group[0]
       pred = random.choices(biased_list_pos_unpriv)[0]
       list_df.append([prot_g,pred])
# create a dataframe from the list
df = pd.DataFrame(list_df,columns=['prot_att','prediction']) 
```

## 解读数据

现在我们已经有了合成数据，让我们来分析我们已经建立了什么。对于每一组，A 对 B，他们实现有利或不利结果的类概率是多少？

```py
df_group = (df.groupby([prot_att])[model_pred_bin].value_counts() / df.groupby([prot_att])[
   model_pred_bin].count()).reset_index(name='probability')
print(df_group)
  prot_att  prediction  probability
0        A           0     0.849490
1        A           1     0.150510
2        B           0     0.713816
3        B           1     0.286184 
```

检查表格，我们不难看出，B 组实现有利结果的可能性几乎是两倍，概率为 28.6%。我们的合成数据被设计成有 30%的概率，所以我们接近目标了。然后，我们将概率保存在字典中。

因为它是随机生成的，所以您的代码可能会给出稍微不同的结果。

```py
prot_att_dic = {}
for att in prot_group:
   temp = df_group[(df_group[prot_att] == att) & (df_group[model_pred_bin] == pos_label)]
   prob = float(temp['probability'])
   prot_att_dic[att] = prob 
```

## 统计奇偶检验

对于每个组，统计奇偶校验输出他们实现有利结果的概率与特权组实现有利结果的概率的比率。我们迭代每个受保护组的字典，用它们有利结果的概率来构建比率。

```py
prot_test_res = {}
for k in prot_att_dic.keys():
   res = float(prot_att_dic[k] / prot_att_dic[priv_g])
   prot_test_res[k] = res
for key in prot_test_res.keys():
   value = prot_test_res[key]
   print(key, ' : ', value)
A  :  0.5259207131128314
B  :  1.0 
```

对于特权组 B 来说，统计奇偶分数是 1，这是应该的。对于另一组 A，他们的得分是 0.526，这表明他们获得有利结果的可能性大约是 b 组的一半。

统计偏差测试提供了一种简单的评估方法，用于评估数据中特定组的预测结果可能存在的差异。测量偏差的目的有两个。一方面，这个测试产生了一个透明的度量，使得交流更加容易和具体。但是理想情况下，识别偏差是在你的模型中开始减轻偏差的第一步。这是机器学习的一个热门研究领域，许多技术正在开发，以适应不同类型的偏见和建模方法。

有了测试和缓解技术的正确组合，就有可能迭代地改进您的模型，减少偏差，并保持准确性。你可以设计机器学习系统，不仅从历史结果中学习，而且在未来的决策中反映你的价值。

了解如何使用 [DataRobot](https://web.archive.org/web/20220529050733/https://www.datarobot.com/) 构建你可以信任的人工智能。

[![](img/74a8ee96bad9f336f2463d0ec30afaf0.png)](https://web.archive.org/web/20220529050733/https://www.datacamp.com/business/demo)