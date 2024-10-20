# 用预先训练好的变形金刚构建搜索引擎:一步一步指南

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/building-search-engine-with-pre-trained-transformers-guide>

我们都用[搜索引擎](https://web.archive.org/web/20221206052553/https://serpact.com/8-machine-learning-principles-and-models-used-by-search-engines/)。我们搜索关于最好的商品、一个好去处的信息，或者回答我们想知道的任何问题。

我们也非常依赖搜索来检查电子邮件、文件和金融交易。很多这样的搜索交互都是通过文本或语音转换成语音输入来实现的。这意味着大量的语言处理发生在搜索引擎上，所以 NLP 在现代搜索引擎中扮演着非常重要的角色

让我们快速看一下搜索时会发生什么。当您使用查询进行搜索时，搜索引擎会收集与该查询匹配的文档的排序列表。要做到这一点，首先应该为文档和其中使用的词汇构建一个“**索引**，然后用于搜索和排列结果。对文本数据进行索引和对搜索结果进行排名的一种流行形式是 [TF-IDF](https://web.archive.org/web/20221206052553/https://en.wikipedia.org/wiki/Tf%E2%80%93idf) 。

NLP 的深度学习模型的最新发展可以用于此。例如，谷歌最近开始使用 [BERT](https://web.archive.org/web/20221206052553/https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) 模型对搜索结果进行排名并显示片段。他们声称这提高了搜索结果的质量和相关性。

有两种类型的搜索引擎:

*   通用搜索引擎，如 Google 和 Bing，抓取网页，并通过不断寻找新网页来覆盖尽可能多的网页。
*   **企业搜索引擎**，我们的搜索空间被限制在一个组织内已经存在的一个较小的文档集。

第二种形式的搜索是你在任何工作场所都会遇到的最常见的用例。看下图就清楚了。

您可以在 transformers 中使用最先进的句子嵌入，并在下游任务中使用它们来实现语义文本的相似性。

在本文中，我们将探索如何构建一个基于向量的搜索引擎。

## 为什么你需要一个基于矢量的搜索引擎？

基于关键字搜索引擎面临以下问题:

*   具有双重含义的复杂查询或单词。
*   长搜索查询。
*   不熟悉重要关键字的用户检索最佳结果。

基于向量(也称为语义搜索)的搜索通过使用 SOTA 语言模型找到文本查询的数字表示来解决这些问题。然后，它在高维向量空间中对它们进行索引，并测量查询向量与索引文档的相似程度。

让我们看看预训练模型能提供什么:

*   他们生产**高质量的嵌入**，因为他们在大量的文本数据上被训练。
*   他们不会强迫你创建一个自定义的记号赋予器，因为变形金刚有它们自己的方法。
*   它们**非常简单方便**来微调模型以适应你的下游任务。

这些模型为文档中的每个标记生成一个固定大小的向量。

现在，让我们看看如何使用预训练的 BERT 模型来构建搜索引擎的特征提取器。

## 步骤 1:加载预先训练的模型

```py
!wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip uncased_L-12_H-768_A-12.zip
!pip install bert-serving-server --no-deps
```

对于这个实现，我将使用 BERT uncased。还有其他可用的 bert 变体——BERT-as-a-service 使用 BERT 作为句子编码器，并通过 [ZeroMQ](https://web.archive.org/web/20221206052553/https://zeromq.org/) 将其作为服务托管，让您只需两行代码就可以将句子映射为固定长度的表示。如果您希望避免客户端-服务器架构引入的额外延迟和潜在模式，这将非常有用。

## 步骤 2:优化推理图

为了修改模型图，我们需要一些低级的张量流编程。因为我们使用的是 bert-as-a-service，所以我们可以使用一个简单的 CLI 界面来配置推断图。

(用于此实现的 tensorflow 版本是 tensorflow==1.15.2)

```py
import os
import tensorflow as tf
import tensorflow.compat.v1 as tfc

sess = tfc.InteractiveSession()

from bert_serving.server.graph import optimize_graph
from bert_serving.server.helper import get_args_parser

MODEL_DIR = '/content/uncased_L-12_H-768_A-12' 

GRAPH_DIR = '/content/graph/' 

GRAPH_OUT = 'extractor.pbtxt' 

POOL_STRAT = 'REDUCE_MEAN' 
POOL_LAYER = '-2' 
SEQ_LEN = '256' 

tf.io.gfile.mkdir(GRAPH_DIR)

carg = get_args_parser().parse_args(args=['-model_dir', MODEL_DIR,
                              '-graph_tmp_dir', GRAPH_DIR,
                              '-max_seq_len', str(SEQ_LEN),
                              '-pooling_layer', str(POOL_LAYER),
                              '-pooling_strategy', POOL_STRAT])

tmp_name, config = optimize_graph(carg)
graph_fout = os.path.join(GRAPH_DIR, GRAPH_OUT)

tf.gfile.Rename(
   tmp_name,
   graph_fout,
   overwrite=True
)
print("nSerialized graph to {}".format(graph_fout))
```

看看上面代码片段中的几个参数。

对于每个文本样本，基于 BERT 的模型编码层输出形状为[ ***sequence_len，encoder_dim*** ]的张量，每个输入令牌一个向量。为了获得固定的表示，我们需要应用某种类型的池。

**POOL_STRAT** 参数定义了应用于编码器层数 **POOL_LAYER** 的池策略。默认值**‘REDUCE _ MEAN’**对序列中所有记号的向量进行平均。当模型没有微调时，这种特殊的策略最适合大多数句子级的任务。另一个选项是 **NONE** ，在这种情况下不应用池。

**SEQ _ 莱恩**对模型处理的序列的最大长度有影响。如果想让模型推理速度几乎线性提升，可以给更小的值。

运行上面的代码片段会将模型图和权重放入一个 **GraphDef** 对象中，该对象将在 **GRAPH_OUT** 序列化为一个 ***pbtxt*** 文件。该文件通常比预训练模型小，因为训练所需的节点和变量将被移除。

让我们使用序列化的图来构建一个使用 [tf 的特征提取器。估计器](https://web.archive.org/web/20221206052553/https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) API。我们需要定义 2 件事情:**输入 _fn** 和**模型 _fn** 。

**input_fn** 将数据获取到模型中。这包括执行整个文本预处理管道，并为 BERT 准备一个 feed_dict。
每个文本样本被转换成一个 tf。示例实例，具有在**输入名称**中列出的必要特征。bert_tokenizer 对象包含了*单词表*并执行文本处理。之后，示例在 *feed_dict* 中按照特性名称重新分组。

```py
import logging
import numpy as np

from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.keras.utils import Progbar

from bert_serving.server.bert.tokenization import FullTokenizer
from bert_serving.server.bert.extract_features import convert_lst_to_features

log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
log.handlers = []
```

```py
GRAPH_PATH = "/content/graph/extractor.pbtxt" 
VOCAB_PATH = "/content/uncased_L-12_H-768_A-12/vocab.txt" 

SEQ_LEN = 256 
```

```py
INPUT_NAMES = ['input_ids', 'input_mask', 'input_type_ids']
bert_tokenizer = FullTokenizer(VOCAB_PATH)

def build_feed_dict(texts):

   text_features = list(convert_lst_to_features(
       texts, SEQ_LEN, SEQ_LEN,
       bert_tokenizer, log, False, False))

   target_shape = (len(texts), -1)

   feed_dict = {}
   for iname in INPUT_NAMES:
       features_i = np.array([getattr(f, iname) for f in text_features])
       features_i = features_i.reshape(target_shape).astype("int32")
       feed_dict[iname] = features_i

   return feed_dict
```

tf。估算器有一个特性，使它们在每次调用 predict 函数时重建和重新初始化整个计算图。

因此，为了避免开销，我们将**将生成器传递给预测函数**，生成器将在一个永无止境的循环中为模型生成特征。

```py
def build_input_fn(container):

   def gen():
       while True:
         try:
           yield build_feed_dict(container.get())
         except:
           yield build_feed_dict(container.get())

   def input_fn():
       return tf.data.Dataset.from_generator(
           gen,
           output_types={iname: tf.int32 for iname in INPUT_NAMES},
           output_shapes={iname: (None, None) for iname in INPUT_NAMES})
   return input_fn

class DataContainer:
 def __init__(self):
   self._texts = None
  def set(self, texts):
   if type(texts) is str:
     texts = [texts]
   self._texts = texts

 def get(self):
   return self._texts
```

**model_fn** 包含模型的规格。在我们的例子中，它是从我们在上一步中保存的 *pbtxt* 文件中加载的。这些特征通过**输入映射**明确映射到相应的输入节点。

```py
def model_fn(features, mode):
   with tf.gfile.GFile(GRAPH_PATH, 'rb') as f:
       graph_def = tf.GraphDef()
       graph_def.ParseFromString(f.read())

   output = tf.import_graph_def(graph_def,
                                input_map={k + ':0': features[k] for k in INPUT_NAMES},
                                return_elements=['final_encodes:0'])

   return EstimatorSpec(mode=mode, predictions={'output': output[0]})

estimator = Estimator(model_fn=model_fn)

```

现在我们已经准备好了，我们需要进行推理。

```py
def batch(iterable, n=1):
   l = len(iterable)
   for ndx in range(0, l, n):
       yield iterable[ndx:min(ndx + n, l)]

def build_vectorizer(_estimator, _input_fn_builder, batch_size=128):
 container = DataContainer()
 predict_fn = _estimator.predict(_input_fn_builder(container), yield_single_examples=False)
  def vectorize(text, verbose=False):
   x = []
   bar = Progbar(len(text))
   for text_batch in batch(text, batch_size):
     container.set(text_batch)
     x.append(next(predict_fn)['output'])
     if verbose:
       bar.add(len(text_batch))

   r = np.vstack(x)
   return r
  return vectorize
bert_vectorizer = build_vectorizer(estimator, build_input_fn)
```

```py
bert_vectorizer(64*['sample text']).shape
o/p: (64, 768)

```

## 第四步:用投影仪探索向量空间

使用矢量器，我们将为来自 [Reuters-221578](https://web.archive.org/web/20221206052553/https://paperswithcode.com/dataset/reuters-21578) 基准语料库的文章生成嵌入。

为了探索和可视化三维嵌入向量空间，我们将使用一种称为 [T-SNE 的降维技术。](https://web.archive.org/web/20221206052553/https://distill.pub/2016/misread-tsne/)

首先让我们得到文章嵌入。

```py
from nltk.corpus import reuters

import nltk
nltk.download("reuters")
nltk.download("punkt")

max_samples = 256
categories = ['wheat', 'tea', 'strategic-metal',
             'housing', 'money-supply', 'fuel']

S, X, Y = [], [], []

for category in categories:
 print(category)
  sents = reuters.sents(categories=category)
 sents = [' '.join(sent) for sent in sents][:max_samples]
 X.append(bert_vectorizer(sents, verbose=True))
 Y += [category] * len(sents)
 S += sents
 X = np.vstack(X)
X.shape

```

运行上述代码后，如果你在 collab 中遇到任何问题，比如:" **Resource reuters not found。请使用 NLTK 下载程序获取资源。**

…然后运行以下命令，其中-d 后面的相对路径将给出文件解压缩的位置:

```py
!unzip /root/nltk_data/corpora/reuters.zip -d /root/nltk_data/corpora

```

生成的嵌入的交互式可视化可在[嵌入投影仪](https://web.archive.org/web/20221206052553/https://projector.tensorflow.org/)上获得。

通过链接，你可以自己运行 *t-SNE* ，或者使用右下角的书签加载一个检查点(加载在 Chrome 上有效)。

要重现用于该可视化的输入文件，请运行下面的代码片段。然后将文件下载到您的机器上，并上传到投影仪。

```py
with open("embeddings.tsv", "w") as fo:
 for x in X.astype('float'):
   line = "t".join([str(v) for v in x])
   fo.write(line+'n')

with open('metadata.tsv', 'w') as fo:
 fo.write("LabeltSentencen")
 for y, s in zip(Y, S):
   fo.write("{}t{}n".format(y, s))

```

这是我用投影仪捕捉到的。

```py
from IPython.display import HTML

HTML("""
<video width="900" height="632" controls>
 <source src="https://storage.googleapis.com/bert_resourses/reuters_tsne_hd.mp4" type="video/mp4">
</video>
""")

```

使用生成的特征构建监督模型非常简单:

```py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
Xtr, Xts, Ytr, Yts = train_test_split(X, Y, random_state=34)

mlp = LogisticRegression()
mlp.fit(Xtr, Ytr)

print(classification_report(Yts, mlp.predict(Xts)))

```

|  | 精确 | 回忆 | f1-分数 | 支持 |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

## 步骤 5:构建搜索引擎

假设我们有一个 50，000 个文本样本的知识库，我们需要根据这些数据快速回答查询。如何从文本数据库中检索出与查询最相似的结果？答案之一可以是最近邻搜索。

我们在这里解决的搜索问题可以定义如下:

给定向量空间中的一组点**S**M**和一个查询点**Q**∈*****M，*** 求距离 **S** 到 **Q** 最近的点。在向量空间中有多种方法来定义 ***【最接近】***——我们将使用[欧几里德距离](https://web.archive.org/web/20221206052553/https://en.wikipedia.org/wiki/Euclidean_distance#:~:text=In%20mathematics%2C%20the%20Euclidean%20distance,being%20called%20the%20Pythagorean%20distance.)。

要构建文本搜索引擎，我们将遵循以下步骤:

1.  对知识库中的所有样本进行矢量化——这就给出了 **S** 。
2.  向量化查询——这给出了 **Q** 。
3.  计算 **Q** 和 **S** 之间的欧几里德距离 **D** 。
4.  按升序排序**D**-提供最相似样本的索引。
5.  从知识库中检索所述样本的标签。

我们可以为 **Q** 和 **S:** 创建占位符

```py
graph = tf.Graph()

sess = tf.InteractiveSession(graph=graph)

dim = X.shape[1]

Q = tf.placeholder("float", [dim])
S = tf.placeholder("float", [None, dim])
```

定义欧几里德距离计算:

```py
squared_distance = tf.reduce_sum(tf.pow(Q - S, 2), reduction_indices=1)
distance = tf.sqrt(squared_distance)

```

获取最相似的指数:

```py
top_k = 10

top_neg_dists, top_indices = tf.math.top_k(tf.negative(distance), k=top_k)
top_dists = tf.negative(top_neg_dists)

```

```py
from sklearn.metrics.pairwise import euclidean_distances

top_indices.eval({Q:X[0], S:X})

np.argsort(euclidean_distances(X[:1], X)[0])[:10]

```

## 第六步:用数学加速搜索

在 tensorflow 中，这可以按如下方式完成:

```py
Q = tf.placeholder("float", [dim])
S = tf.placeholder("float", [None, dim])

Qr = tf.reshape(Q, (1, -1))

PP = tf.keras.backend.batch_dot(S, S, axes=1)
QQ = tf.matmul(Qr, tf.transpose(Qr))
PQ = tf.matmul(S, tf.transpose(Qr))

distance = PP - 2 * PQ + QQ
distance = tf.sqrt(tf.reshape(distance, (-1,)))

top_neg_dists, top_indices = tf.math.top_k(tf.negative(distance), k=top_k)

```

上式**中 PP** 和 **QQ** 实际上是各自向量的平方 [L2 范数](https://web.archive.org/web/20221206052553/https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm)。如果两个向量都是 L2 归一化的，则:

***PP = QQ = 1***

进行 L2 归一化会丢弃关于矢量幅度的信息，这在很多情况下是你不想做的。

相反，我们可能会注意到，只要知识库保持不变——PP——其平方向量范数也保持不变。因此，我们可以只做一次，然后使用预先计算的结果，而不是每次都重新计算，从而进一步加快距离计算。

让我们一起努力。

```py
class L2Retriever:
 def __init__(self, dim, top_k=3, use_norm=False, use_gpu=True):
   self.dim = dim
   self.top_k = top_k
   self.use_norm = use_norm
   config = tf.ConfigProto(
       device_count = {'GPU': (1 if use_gpu else 0)}
   )
   self.session = tf.Session(config=config)

   self.norm = None
   self.query = tf.placeholder("float", [self.dim])
   self.kbase = tf.placeholder("float", [None, self.dim])

   self.build_graph()

 def build_graph():
   if self.use_norm:
     self.norm = tf.placeholder("float", [None, 1])

   distance = dot_l2_distances(self.kbase, self.query, self.norm)
   top_neg_dists, top_indices = tf.math.top_k(tf.negative(distance), k=self.top_k)
   top_dists = tf.negative(top_neg_dists)

   self.top_distances = top_dists
   self.top_indices = top_indices

 def predict(self, kbase, query, norm=None):
   query = np.squeeze(query)
   feed_dict = {self.query: query, self.kbase: kbase}
   if self.use_norm:
     feed_dict[self.norm] = norm

   I, D = self.session.run([self.top_indices, self.top_distances],
                           feed_dict=feed_dict)
   return I, D

def dot_l2_distances(kbase, query, norm=None):
 query = tf.reshape(query, (1, 1))

 if norm is None:
   XX = tf.keras.backend.batch_dot(kbase, kbase, axes=1)
 else:
   XX = norm
 YY = tf.matmul(query, tf.transpose(query))
 XY = tf.matmul(kbase, tf.transpose(query))

 distance = XX - 2 * XY + YY
 distance = tf.sqrt(tf.reshape(distance, (-1, 1)))

 return distance
```

我们可以将这种实现用于任何矢量器模型，而不仅仅是 BERT。它在最近邻检索方面非常有效，能够在双核 colab CPU 上每秒处理几十个请求。

**在构建机器学习应用时，你需要考虑一些额外的方面:**

*   您如何确保解决方案的可扩展性？

*   选择正确的框架/语言。
*   使用正确的处理器。
*   收集和存储数据。
*   输入管道。
*   模特培训。
*   分布式系统。
*   其他优化。
*   资源利用和监控。
*   展开。

*   你如何训练、测试和部署你的产品模型？

*   创建一个可用于下载和处理数据的笔记本实例。
*   准备数据/预处理它，你需要训练你的 ML 模型，然后上传数据(例如:亚马逊 S3)。
*   使用您的训练数据集来训练您的机器学习模型。
*   将模型部署到端点，重新格式化并加载 csv 数据，然后运行模型以创建预测。
*   评估 ML 模型的性能和准确性。

## 旁注–通过实验跟踪简化 ML

一个工具可以照顾你所有的[实验跟踪](/web/20221206052553/https://neptune.ai/experiment-tracking)和协作需求——[Neptune . ai](/web/20221206052553/https://neptune.ai/)

Neptune 记录您的整个实验过程——探索笔记本、模型训练运行、代码、超参数、指标、数据版本、结果、探索可视化等等。

这是 MLOps 的元数据存储，为进行大量实验的研究和生产团队而构建。专注于 ML，把元数据管理留给 Neptune。要开始使用 Neptune，请访问他们广泛的[指南](https://web.archive.org/web/20221206052553/https://docs.neptune.ai/getting-started/hello-world)。

像 Neptune 这样的 ML 元数据存储是 MLOps 栈的重要组成部分。在您构建模型时，它负责元数据管理。

它记录、存储、显示、组织、比较和查询 ML 模型生命周期中生成的所有元数据。

您可以使用 ML metastore 来跟踪、组织和比较您在 ML 实验中关心的一切。

Neptune 集成了所有你喜欢的框架和工具——最流行的集成之一是直接通过 [TensorBoard](https://web.archive.org/web/20221206052553/https://docs.neptune.ai/integrations-and-supported-tools/experiment-tracking/tensorboard) 完成的 [Tensorflow/Keras](https://web.archive.org/web/20221206052553/https://docs.neptune.ai/integrations-and-supported-tools/model-training/tensorflow-keras) 。

## 结论

使用 BERT 搜索的主要探索领域是**相似性**。文档的相似性、推荐的相似性以及查询和文档之间的相似性，用于返回和排列搜索结果。

如果你能使用相似性来解决这个问题，并得到非常精确的结果，那么你就能很好地搜索到你的产品或应用。

我希望你在这里学到了新东西。感谢阅读。不断学习。