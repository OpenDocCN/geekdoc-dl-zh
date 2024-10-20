# 如何使用 Keras 对患者问题进行分类(1 分钟培训)

> 原文：<https://www.dlology.com/blog/tutorial-medical-triage-with-patient-query/>

###### 发布者:[程维](/blog/author/Chengwei/) 5 年 2 个月前

([评论](/blog/tutorial-medical-triage-with-patient-query/#disqus_thread))

在本教程中，我们将构建一个基于患者查询文本数据的分诊模型。

例如

| **查询(输入)** | **分流(输出)** |
| 皮肤很痒。 | 皮肤病学 |
| 喉咙痛发烧疲劳。 | 嘴脸 |
| 后腰疼，好疼。 | 背部 |

我们将使用 Keras 和 Tensorflow(版本 1.3.0)后端来构建模型

对于本教程中使用的源代码和数据集，请查看我的 [GitHub repo](https://github.com/Tony607/Medical_Triage) 。

## 依赖性

Python 3.5, numpy, pickle, keras, tensorflow, nltk, pandas

## 关于日期

1261 个患者查询，**phrases _ embed . CSV**来自  [巴比伦博客《聊天机器人如何理解句子》](https://blog.babylonhealth.com/how-the-chatbot-understands-sentences-fe6c5deb6e81)。

查看数据可视化  [这里](https://s3-eu-west-1.amazonaws.com/nils-demo/phrases.html)。

## 准备数据

我们将执行以下步骤来准备用于训练模型的数据。

1.从 CSV 文件读取数据到 Pandas 数据框，只保留 2 列“疾病”和“类别”

2.将 Pandas 数据帧转换为 numpy 数组对

"疾病"栏== >文档

" class" columns ==> body_positions

3.清理数据

对于每个句子，我们将所有字母转换为小写，只保留英文字母和数字，删除停用词，如下所示。

4.Input **tokenizer** 将输入单词转换为 id，如果输入序列较短，则将每个输入序列填充到最大输入长度。

保存输入标记器，因为我们需要在预测过程中使用同一个标记器来标记任何新的输入数据。

5.将输出单词转换为 id，然后转换为类别(一键向量)

7.make**target _ reverse _ word _ index**将谓词类 id 转换为文本。

## 建立模型

模型结构将如下所示

```py
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 18, 256)           232960    
_________________________________________________________________
gru_1 (GRU)                  (None, 18, 256)           393984    
_________________________________________________________________
gru_2 (GRU)                  (None, 256)               393984    
_________________________________________________________________
dense_1 (Dense)              (None, 19)                4883      
=================================================================
```

**嵌入**层将单词 id 转换成它们对应的单词嵌入，来自嵌入层的每个输出将具有(18×256)的大小，这是**最大输入序列填充长度**乘以**嵌入维度**。

数据被传递到一个递归层来处理输入序列，我们在这里使用 **GRU** ，你也可以试试 LSTM。

所有的中间输出被收集，然后传递到第二 GRU 层。

然后输出被发送到一个完全连接的层，这将给出我们最终的预测类别。我们使用“**soft max**”**activation**来给出每个职业的概率。

使用标准的**‘分类 _ 交叉熵’**损失函数进行多类分类。

使用 **"** 亚当 **"** 优化器，因为它适应学习率。

然后，我们将训练该模型，并保存它以供以后预测。

## 预测新数据

1.加载我们之前保存的模型。

2.加载输入标记器，标记新的患者查询文本，将序列填充到最大长度

3.将序列输入模型，模型将输出类别 id 和概率，我们使用“target_reverse_word_index”将类别 id 转换为实际的分类结果文本。

以下是一些预测结果

## 摘要

Keras 训练 40 个纪元，用 GPU (GTX 1070)不到 1 分钟最终 acc:0.9146

训练数据的大小相对较小，拥有较大的数据集可能会提高最终的准确性。

查看我的 [GitHub repo](https://github.com/Tony607/Medical_Triage) 获取 Jupyter 笔记本源代码和数据集。

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/tutorial-medical-triage-with-patient-query/&text=How%20to%20triage%20patient%20queries%20with%20Keras%20%281%20minute%20training%29) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/tutorial-medical-triage-with-patient-query/)

*   [←利用酒店点评数据进行中文情感分析的简易指南](/blog/tutorial-chinese-sentiment-analysis-with-hotel-review-data/)
*   [如何用 Tensorflow 总结亚马逊评论→](/blog/tutorial-summarizing-text-with-amazon-reviews/)