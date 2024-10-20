# Keras +通用语句编码器=文本数据的迁移学习

> 原文：<https://www.dlology.com/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/>

###### 发帖人:[程维](/blog/author/Chengwei/)四年零六个月前

([评论](/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/#disqus_thread))

![tf-hub-meets-keras](img/2658e9dbd87b25d93c55b85952fc4398.png)

我们将构建一个 Keras 模型，利用预先训练的“通用句子编码器”将给定的问题文本分类到六个类别之一。

TensorFlow Hub 模块可应用于各种迁移学习任务和数据集，无论是图像还是文本。 “通用句子编码器”是众多新发布的 TensorFlow Hub 可重用模块之一，TensorFlow 图形的一个独立部分，包含预先训练好的权重值。

一个可运行的 [Colab 笔记本](https://colab.research.google.com/drive/1Odry08Jm0f_YALhAt4vp9qa5w8prUzDY)已经上市，你可以边读边试验代码。

## 什么是通用语句编码器，它是如何被训练的

虽然您可以选择将所有 TensorFlow Hub 模块视为黑盒，不知道内部发生了什么，但仍然能够构建功能迁移学习模型。这将有助于加深理解，让你对每个模块的能力、限制以及迁移学习结果的潜力有一个新的认识。

### 通用语句编码器 VS 文字嵌入

如果你还记得我们[之前的教程](https://www.dlology.com/blog/simple-stock-sentiment-analysis-with-news-data-in-keras/)中的手套单词嵌入向量，它将单词转换成 50 维向量，通用句子编码器要强大得多，它不仅能够嵌入单词，还能够嵌入短语和句子。即它以变长英文文本为输入，输出一个 512 维向量。处理可变长度的文本输入听起来很棒，但问题是随着句子越来越长，嵌入结果可能会越来越模糊。由于模型是在单词级别训练的，它可能会发现错别字和难以处理的单词。更多关于世界级和角色级语言模型的区别，可以看我的[以前的教程](https://www.dlology.com/blog/how-to-train-a-keras-model-to-generate-colors/)。

有两种通用语句编码器可供选择，它们具有不同的编码器架构，以实现不同的设计目标，一种基于 transformer 架构，目标是以更大的模型复杂性和资源消耗为代价获得高精度。另一个目标是通过深度平均网络(DAN)以稍微降低的准确度进行有效的推断。 

并列式变压器和单句编码器的模型架构比较。 

![dan-and-transformer](img/8e6f6fd897aff5e5b35d2f8f2bdcb497.png)

原来的[变压器](https://arxiv.org/pdf/1706.03762.pdf)模型构成了编码器和解码器，但这里我们只用到了它的编码器部分。

编码器由 N = 6 个相同层的堆叠组成。每层有两个子层。第一种是多头自关注机制，第二种是简单的、位置式全连接前馈网络。他们还在两个子层中的每一个周围采用了剩余连接，然后进行层归一化。由于模型不包含递归和卷积，为了使模型利用序列的顺序，它必须注入一些关于序列中 记号的相对或绝对位置的信息，这就是“位置编码所做的。T 基于变压器的编码器实现了最佳的整体传输任务性能。然而，这是以计算时间和内存使用随句子长度急剧增加为代价的。 

深度平均网络(DAN)要简单得多，其中单词和二元模型的输入嵌入首先一起平均，然后通过前馈深度神经网络(DNN)产生句子嵌入。DAN 编码器的主要优点是计算时间与输入序列的长度成线性关系。

根据培训数据的类型和所选的培训指标，它会对迁移学习结果产生重大影响。 

这两个模型都是用斯坦福自然语言推理(SNLI)语料库训练的。 [SNLI 语料库](https://nlp.stanford.edu/pubs/snli_paper.pdf)是 570，000 个人类书写的英语句子对的集合，这些句子对被手动标记为平衡分类，具有标签蕴涵、矛盾和中性，支持自然语言推理(NLI)的任务，也称为识别文本蕴涵(RTE)。本质上，模型被训练来学习句子对之间的语义相似性。

考虑到这一点， 句子嵌入可以简单地用于计算句子级语义相似度得分。

![semantic-similarity](img/bd43751bf17ddfe90a6ad414b4b60151.png)

生成相似性热图的源代码在我的 Colab 笔记本和 GitHub repo 中都有。基于任意两个句子的编码内积进行着色。这意味着两个句子越相似，颜色就越深。

加载通用语句编码器并计算某些文本的嵌入可以像下面这样简单。

首次加载模块可能需要一段时间，因为它会下载重量文件。

`message_embeddings`的值是 两个数组对应两个句子的嵌入，每个都是 512 个浮点数的数组。

```py
array([[ 0.06587551, 0.02066354, -0.01454356, ..., 0.06447642, 0.01654527, -0.04688655], [ 0.06909196, 0.01529877, 0.03278331, ..., 0.01220771, 0.03000253, -0.01277521]], dtype=float32)
```

## 问题分类任务和数据预处理

在作为检索任务的问答中，为了正确回答给定的大量文本中的问题，将问题分类成细粒度的类别是至关重要的。我们的目标是将问题分为不同的语义类别，这些类别对潜在答案施加约束，以便它们可以在问答过程的后期阶段使用。比如在考虑问题**Q**:*加拿大哪个城市人口最多？* 希望将这个问题归类为有答案类型 **地点** ，暗示只有是地点的候选答案需要考虑。 

我们使用的数据集是 [TREC 问题分类数据集](http://cogcomp.org/Data/QA/QC/)，总共有 5452 个训练样本和 500 个测试样本，即 5452 + 500 个问题，每个问题被归入六个标签中的一个。

1.  **【ABBR———‘缩写’**:表达缩写等。
2.  **DESC -【描述和抽象概念】**:动作的方式，对某事物的描述。等等。
3.  **ENTY -【实体】**:动物、颜色、事件、食物等。
4.  **HUM -【人类】**:一群人或一个组织，个人等。
5.  **【地点】**:城市、国家等。
6.  **NUM-‘数值’**:邮编、日期、速度、温度等

我们希望我们的模型是一个多类分类模型，将字符串作为 6 个类标签中每一个的输入和输出概率。记住这一点，你就知道如何为它准备训练和测试数据。

第一步是将原始文本文件转换成 pandas DataFrame，并将“label”列设置为 categorical 列，这样我们就可以进一步以数值形式访问标签。

前 5 个训练样本如下所示。

![df_train_head](img/7f86368de8af4b212dcac8af6ddaf644.png)

下一步，我们将为模型准备输入/输出数据，输入为一列问题字符串，输出为一列一次性编码标签。如果你对一键编码还不熟悉，我在之前的[文章](https://www.dlology.com/blog/how-to-train-a-keras-model-to-generate-colors/)中已经介绍过了。

如果你瞥一眼`train_label`的值，你会看到它以一键编码的形式出现。

```py
array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0] ...], dtype=int8)
```

现在我们已经准备好构建模型了。

## Keras 遇上通用句子编码器

我们之前已经将通用语句编码器加载为变量“`embed`”，为了让它与 Keras 很好地配合，有必要将其包装在 Keras Lambda 层中，并将其输入显式转换为字符串。

然后我们在它的标准[功能 API](https://keras.io/getting-started/functional-api-guide/) 、中构建 Keras 模型

我们可以查看模型摘要，并意识到只有 Keras 层是可训练的，这就是迁移学习任务如何通过确保通用句子编码器权重不变来工作的。

```py
_________________________________________________________________
Layer (type) Output Shape Param # 
=================================================================
input_1 (InputLayer) (None, 1) 0 
_________________________________________________________________
lambda_1 (Lambda) (None, 512) 0 
_________________________________________________________________
dense_1 (Dense) (None, 256) 131328 
_________________________________________________________________
dense_2 (Dense) (None, 6) 1542 
=================================================================
Total params: 132,870
Trainable params: 132,870
Non-trainable params: 0
_________________________________________________________________
```

在下一步中，我们使用训练数据集训练模型，并在每个训练时期结束时使用测试数据集验证其性能。

最终验证结果显示，经过 10 个时期的训练，最高准确率达到 97%左右。

在我们对模型进行训练并将其权重保存到文件中之后，实际上是对新问题进行预测。

在这里，我们提出了 3 个新问题供模型进行分类。

分类结果看起来不错。

```py
['NUM', 'LOC', 'HUM']
```

## 结论和进一步阅读

恭喜你！您已经建立了一个由通用句子编码器支持的 Keras 文本迁移学习模型，并在问题分类任务中取得了很好的结果。通用句子编码器可以嵌入更长的段落，因此可以随意试验其他数据集，如新闻主题分类、情感分析等。

一些你可能会觉得有用的相关资源。

[TensorFlow Hub](https://www.tensorflow.org/hub/)

[TensorFlow Hub 示例笔记本](https://github.com/tensorflow/hub/tree/master/examples/colab)

关于使用 Google Colab notebook 的介绍，你可以免费阅读我的文章的第一部分- [如何在视频上快速运行对象检测和分割](https://www.dlology.com/blog/how-to-run-object-detection-and-segmentation-on-video-fast-for-free/)。

[my GitHub](https://github.com/Tony607/Keras-Text-Transfer-Learning) 中的源代码和一个可运行的 [Colab 笔记本](https://colab.research.google.com/drive/1Odry08Jm0f_YALhAt4vp9qa5w8prUzDY)。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/&text=Keras%20%2B%20Universal%20Sentence%20Encoder%20%3D%20Transfer%20Learning%20for%20text%20data) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/)

*   [←如何训练 Keras 模型生成颜色](/blog/how-to-train-a-keras-model-to-generate-colors/)
*   [使用 TensorFlow Hub DELF 模块轻松识别地标图像→](/blog/easy-landmark-image-recognition-with-tensorflow-hub-delf-module/)