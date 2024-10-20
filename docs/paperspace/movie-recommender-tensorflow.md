# 基于 TensorFlow 的电影推荐系统

> 原文：<https://blog.paperspace.com/movie-recommender-tensorflow/>

我们在 [YouTube](https://www.youtube.com/) (或[网飞](netflix.com))上观看一个视频(或电影)，然后马上就会看到一个建议接下来观看的视频(或电影)列表，这种情况并不少见。同样的事情也经常发生在数字音乐流媒体服务上。一个人在 [Spotify](https://open.spotify.com/) 上听一首歌，马上就会找到一个类似歌曲的列表，可能是同一流派或同一艺术家的作品。

该列表由推荐机器学习模型构建，该模型通常被称为推荐引擎/系统。推荐系统不仅仅是简单的机器学习。需要建立一个数据管道来收集模型所需的输入数据(例如，像用户观看的最后五个视频这样的输入)。推荐系统满足了这一需求。

一个主要的误解是推荐系统只是向用户推荐产品。这与事实相去甚远。推荐系统不仅可以向用户推荐产品，还可以向用户推荐产品。例如，在营销应用程序中，当有新的促销活动时，推荐系统可以找到前一千个最相关的当前客户。这叫做瞄准。此外，与谷歌地图通过推荐系统建议避开收费公路的路线一样，Gmail 中的智能回复也是通过推荐系统来建议对刚收到的电子邮件的可能回复。搜索引擎是推荐引擎如何提供个性化的另一个很好的例子。您的搜索查询会考虑您的位置、您的用户历史、帐户偏好和以前的搜索，以确保您获得的内容与用户最相关。

例如，在搜索栏中输入“*giants”*可能会产生不同的结果，这取决于用户所在的位置。如果用户在纽约，很可能会得到很多纽约巨人队的结果。但是，在旧金山进行相同的搜索可能会返回关于旧金山棒球队的信息。本质上，从用户的角度来看，推荐系统可以帮助找到相关的内容，探索新的项目，并改善用户的决策。从生产者的角度来看，它有助于提高用户参与度，了解更多的用户，并监测用户行为的变化。总之，推荐系统都是关于个性化的。这意味着要有一个适合所有人的产品，并为个人用户进行个性化定制。

# 推荐系统的类型

## 基于内容的过滤:

在这种类型的推荐框架中，我们利用系统中可用的产品元数据。假设一个用户观看并评价了几部电影。他们给了一些人一个大拇指，给了一些人一个大拇指，我们想知道数据库中的哪部电影是下一部推荐的。

由于我们有关于电影的元数据，也许我们知道这个特定用户更喜欢科幻而不是情景喜剧。因此，使用这种系统，我们可以利用该数据向该客户推荐受欢迎的科幻节目。其他时候，我们没有每个用户的偏好。要创建一个基于内容的推荐系统，我们可能需要的只是一个市场细分，显示世界不同地区的用户喜欢哪些电影。有观点认为这里不涉及机器学习。这是一个简单的规则，它依赖于推荐系统的创建者来适当地标记人和对象。这种方法的主要缺点是，为了使这个系统正确工作，它需要领域知识。虽然存在解决这种“冷启动”问题的方法，但是没有什么方法可以完全克服缺乏培训信息的影响。此外，由于其性质，该系统仅提供安全建议。

## 协作过滤:

在这种方法中，我们没有任何关于产品的元数据；相反，我们可以从评级数据中推断出关于项目和用户相似性的信息。例如，我们可能需要将用户的电影数据保存在一个带有复选标记的矩阵中，以指示用户是否观看了整部电影，是否留下了评论，是否给了它一个星级，或者您用来确定某个用户是否喜欢某部电影的任何内容。正如你所料，这个矩阵的规模是巨大的。一个人只能看到这些电影中的一小部分，因为可能有数百万或数十亿人和数百或数百万部电影可用。因此，大多数矩阵既庞大又稀疏。

为了逼近这个庞大的用户-项目矩阵，协同过滤结合了两个更小的矩阵，称为用户因素和项目因素。然后，如果我们想确定一个特定的用户是否会喜欢某部电影，我们所要做的就是获取与该电影相对应的行，并将它们相乘以获得预测的评级。然后，我们选择我们认为收视率最高的电影，然后推荐给消费者。

协同过滤最大的好处是我们不需要熟悉任何项目的元数据。此外，只要我们有一个互动矩阵，我们就可以做得很好，不需要对你的用户进行市场细分。也就是说，问题可能源于特性的稀疏性和无上下文特性。

## 基于知识的建议:

在这种类型的推荐系统中，数据取自用户调查或用户输入的显示其偏好的设置。这通常通过询问用户的偏好来实现。基于知识的推荐的一个很大的好处是它不需要用户-项目交互数据。相反，它可以简单地依靠以用户为中心的数据将用户与其他用户联系起来，并推荐那些用户喜欢的类似东西。此外，基于知识的推荐最终使用高保真数据，因为感兴趣的用户已经自我报告了他们的信息和偏好。因此，假设这些都是真的是公平的。然而，另一方面，当用户不愿意分享他们的偏好时，可能会出现一个重大的挑战。由于隐私问题，缺少用户数据可能是一个问题。由于这些隐私问题，尝试基于知识以外的推荐方法可能更容易。

# 演示

例如，在本教程中，您将使用 TensorFlow 创建一个实际操作的电影推荐系统。TensorFlow 的核心是允许您使用 Python(或 JavaScript)开发和训练模型，并且无论您使用何种编程语言，都可以轻松地在云中、本地、浏览器或设备上部署。我们将使用 Papersapce Gradient 的免费 GPU 笔记本来进行这次演示。

在我们继续之前，需要注意的是现实世界的推荐系统通常由两个阶段组成:

*   **检索阶段**:该阶段用于从所有可能的电影候选者中选择电影候选者的初始集合。该模型的主要目的是有效地驱逐用户不感兴趣的所有候选人。检索阶段通常使用协同过滤。
*   排名阶段:这一阶段从检索模型中获取输出，并对其进行微调，以选择尽可能多的最佳电影推荐。它的任务是将用户可能感兴趣的电影集缩小到可能的候选名单中。

**检索模型**

与所有使用协同过滤的推荐系统一样，这些模型通常由两个子模型组成:

1.  使用特征计算查询表示(通常是固定维度的嵌入向量)的查询模型。
2.  一种候选模型，使用影片的特征来计算影片候选表示(大小相等的向量)。

然后，将两个模型的输出相乘，以给出查询候选相似性分数，较高的分数表示电影候选和查询之间的较好匹配。在本教程中，我们将使用带有 TensorFlow 的 Movielens 数据集来构建和训练一个推荐系统。 [Movielens 数据集](https://grouplens.org/datasets/movielens/)是来自 [GroupLens](https://grouplens.org/datasets/movielens/) 研究小组的数据集。它包含一组用户对电影的评级，这些用户是在不同的时间段收集的，具体取决于集合的大小。这是推荐系统研究中的一个热点。

这个数据可以从两个方面来看。可以解释为用户看了哪些电影(以及评级)，没有看哪些电影。它也可以被看作是用户有多喜欢他们观看的电影。第一种观点将数据集视为一种隐性反馈，用户的观看历史告诉我们他们更喜欢看什么，不喜欢看什么。后一种观点可以将数据集转化为一种明确的反馈形式，通过查看用户给出的评级，可以大致了解观看电影的用户喜欢这部电影的程度。

对于检索系统，模型从用户可能观看的目录中预测一组电影，隐式数据在传统上更有用。因此，我们将电影镜头视为一个隐含系统。本质上，用户看过的每一部电影都是正面例子，没看过的每一部电影都是隐含的反面例子。

让我们开始吧。

第一步:导入必要的库。

```py
!pip install -q tensorflow-recommenders
!pip install -q --upgrade tensorflow-datasets
!pip install -q scann

import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs 
```

第二步:获取你的数据，并将其分成训练集和测试集。

```py
# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train") 
```

变量`ratings`包含分级数据，而变量`movies`包含所有可用电影的特征。分级数据集返回电影 id、用户 id、分配的分级、时间戳、电影信息和用户信息的字典，如下所示。而电影数据集包含电影 id、电影标题和关于其所属类型的数据。这些类型用整数标签编码。值得注意的是，由于 Movielens 数据集没有预定义的分割，因此它的所有数据都在`train`分割下。

```py
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
})
movies = movies.map(lambda x: x["movie_title"]) 
```

在本教程中，您将关注评级数据本身，您将只保留评级数据集中的`user_id`和`movie_title`字段。

```py
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000) 
```

为了拟合和评估模型，我们将把它分成一个训练集和一个评估集。我们将使用随机分割，将 80%的评级放在训练集中，20%放在测试集中。

此时，我们希望知道数据中出现的唯一用户 id 和电影标题。这很重要，因为我们需要能够将分类特征的原始值映射到模型中的嵌入向量。为了实现这一点，我们需要一个词汇表，将原始特征值映射到一个连续范围内的整数:这允许我们在嵌入表中查找相应的嵌入。

```py
movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids))) 
```

第三步:实现一个检索模型。

```py
embedding_dimension = 32
user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
      vocabulary=unique_user_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
]) 
```

假设嵌入维度的较高值将对应于可能更准确的模型，但是也将更慢地拟合并且更易于过拟合，32 被挑选为查询和候选表示的维度。为了定义模型本身，`keras`预处理层将用于将用户 id 转换成整数，然后使用`Embedding`层将它们转换成用户嵌入。

我们将对电影《候选人之塔》做同样的事情。

```py
movie_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_movie_titles, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
]) 
```

在训练数据中，我们注意到我们有积极的用户和电影对。为了评估我们的模型以及它有多好，我们将比较模型为这一对计算的亲和力分数和所有其他可能的候选人的分数。这意味着，如果阳性对的得分高于所有其他候选项，则您的模型是高度准确的。为了检查这一点，我们可以使用 [`tfrs.metrics.FactorizedTopK`](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/metrics/FactorizedTopK) 度量。这个指标有一个必需的参数:候选人的数据集，您将其用作评估的隐式否定。这意味着您将通过电影模型转换成嵌入的`movies`数据集。

```py
metrics = tfrs.metrics.FactorizedTopK(
  candidates=movies.batch(128).map(movie_model)
) 
```

此外，我们必须检查用于训练模型的损失。好东西`tfrs`为此有几个损耗层和任务。我们可以使用`Retrieval`任务对象，它是一个方便的包装器，将损失函数和度量计算与下面几行代码捆绑在一起。

```py
task = tfrs.tasks.Retrieval(
  metrics=metrics
) 
```

完成所有设置后，我们现在可以将它们放入一个模型中。`tfrs.models.Model`，`tfrs`的基础模型类将用于简化建筑模型。 [`tfrs.Model`](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/models/Model) 基类的存在使得我们可以使用相同的方法计算训练和测试损失。我们需要做的就是在`__init__`方法中设置组件，然后使用原始特性实现`compute_loss`方法并返回一个损失值。此后，我们将使用基础模型创建适合您的模型的适当培训循环。

```py
class MovielensModel(tfrs.Model):

  def __init__(self, user_model, movie_model):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    user_embeddings = self.user_model(features["user_id"])
    positive_movie_embeddings = self.movie_model(features["movie_title"])
    return self.task(user_embeddings, positive_movie_embeddings) 
```

`compute_loss`方法从挑选出用户特征开始，然后将它们传递到用户模型中。此后，它挑选出电影特征，并把它们传递到电影模型中，从而取回嵌入内容。

第四步:拟合和评估。

定义模型后，我们将使用标准的 Keras 拟合和评估例程来拟合和评估模型。

```py
model = MovielensModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()
model.fit(cached_train, epochs=3)
```

我们的模型将在三个时期内被训练。我们可以看到，随着模型的训练，损失下降，一组 top-k 检索指标正在更新。这些度量让我们知道真正的肯定是否在整个候选集的前 k 个检索项目中。请注意，在本教程中，我们将在培训和评估过程中评估指标。因为对于大型候选集，这可能会非常慢，所以谨慎的做法是在培训中关闭度量计算，仅在评估中运行它。

最后，我们可以在测试集上评估我们的模型:

```py
model.evaluate(cached_test, return_dict=True) 
```

我们应该注意到，测试集性能不如训练集性能好。理由并不牵强。我们的模型在处理之前看到的数据时会表现得更好。此外，该模型只是重新推荐用户已经看过的一些电影。

**第五步:做预测**

既然我们已经有了一个运行的模型，我们就可以开始做预测了。为此我们将使用 [`tfrs.layers.factorized_top_k.BruteForce`](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/layers/factorized_top_k/BruteForce) 图层。我们将使用它来获取原始查询特性，然后从整个电影数据集中推荐电影。最后，我们得到我们的建议。

```py
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
  tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
)

_, titles = index(tf.constant(["46"]))
print(f"Recommendations for user 46: {titles[0, :3]}")
```

在上面的代码块中，我们将获得对用户 46 的推荐。

**步骤 6:通过建立近似最近邻(ANN)索引，导出它以进行有效的服务。**

直观地说，`BruteForce`层太慢了，不能服务于有许多候选对象的模型。使用近似检索索引可以加快这个过程。虽然检索模型中的服务有两个组件(即服务查询模型和服务候选模型)，但使用`tfrs`，这两个组件可以打包成我们可以导出的单个模型。该模型获取原始用户 id，并为该用户返回热门电影的标题。为此，我们将把模型导出到一个`SavedModel`格式，这样就可以使用 [TensorFlow 服务](https://www.tensorflow.org/tfx/guide/serving)了。

```py
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")
  tf.saved_model.save(index, path)
  loaded = tf.saved_model.load(path)
  scores, titles = loaded(["42"])
  print(f"Recommendations: {titles[0][:3]}") 
```

为了有效地从数以百万计的候选电影中获得推荐，我们将使用一个可选的 TFRS 依赖项，称为 TFRS `scann`层。这个包是在教程开始时通过调用`!pip install -q scann`单独安装的。该层可以执行 **近似** 查找，这将使检索稍微不太准确，同时在大型候选集上保持数量级的速度。

```py
scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
scann_index.index_from_dataset(
  tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
)

_, titles = scann_index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")
```

最后，我们将导出查询模型，保存索引，将其加载回来，然后传入一个用户 id 以获取热门预测电影标题。

```py
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")

  tf.saved_model.save(
      index,
      path,
      options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
  )
  loaded = tf.saved_model.load(path)

  scores, titles = loaded(["42"])

  print(f"Recommendations: {titles[0][:3]}") 
```

\

### 排名模型

使用排序模型，前两步(即导入必要的库并将数据分成训练集和测试集)与检索模型完全相同。

**第三步:实现排名模型**

使用排序模型，所面临的效率约束与检索模型完全不同。因此，我们在选择架构时有更多的自由。由多个堆叠的密集层组成的模型通常用于分级任务。我们现在将按如下方式实现它:

注意:该模型将用户 id 和电影名称作为输入，然后输出预测的评级。

```py
class RankingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    embedding_dimension = 32

    # Compute embeddings for users
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # Compute embeddings for movies
    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])

    # Compute predictions
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
  ])

  def call(self, inputs):

    user_id, movie_title = inputs

    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_title)

    return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))
```

为了评估用于训练我们的模型的损失，我们将使用将损失函数与度量计算相结合的`Ranking`任务对象。我们将它与`MeanSquaredError` Keras 损失一起使用，以预测收视率。

```py
task = tfrs.tasks.Ranking(
  loss = tf.keras.losses.MeanSquaredError(),
  metrics=[tf.keras.metrics.RootMeanSquaredError()]
) 
```

将所有这些放入一个完整的排名模型中，我们有:

```py
class MovielensModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model(
        (features["user_id"], features["movie_title"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("user_rating")

    rating_predictions = self(features)
    return self.task(labels=labels, predictions=rating_predictions) 
```

**第四步:拟合和评估排名模型**

定义模型后，我们将使用标准的 Keras 拟合和评估例程来拟合和评估您的排名模型。

```py
model = MovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()
model.fit(cached_train, epochs=3)
model.evaluate(cached_test, return_dict=True) 
```

利用在三个时期上训练的模型，我们将通过计算对一组电影的预测来测试排名模型，然后基于预测对这些电影进行排名:

```py
test_ratings = {}
test_movie_titles = ["M*A*S*H (1970)", "Dances with Wolves (1990)", "Speed (1994)"]
for movie_title in test_movie_titles:
  test_ratings[movie_title] = model({
      "user_id": np.array(["42"]),
      "movie_title": np.array([movie_title])
  })

print("Ratings:")
for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
  print(f"{title}: {score}") 
```

**第五步:导出服务，将模型转换为 TensorFlow Lite**

如果一个推荐系统不能被用户使用，那么它是没有用的。因此，我们必须导出模型来提供服务。此后，我们可以加载它并执行预测。

```py
tf.saved_model.save(model, "export")
loaded = tf.saved_model.load("export")

loaded({"user_id": np.array(["42"]), "movie_title": ["Speed (1994)"]}).numpy() 
```

为了更好地保护用户隐私和降低延迟，我们将使用 TensorFlow Lite 在设备上运行经过训练的排名模型，尽管 TensorFlow 推荐器主要用于执行服务器端推荐。

```py
converter = tf.lite.TFLiteConverter.from_saved_model("export")
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model) 
```

**结论**

我们现在应该知道推荐器是什么，它是如何工作的，隐式和显式反馈之间的区别，以及如何使用协同过滤算法来构建推荐系统。在我们自己，我们可以调整网络设置，如隐藏层的尺寸，以查看相应的变化。根据经验，这些维度取决于您想要近似的函数的复杂性。如果隐藏层太大，我们的模型会有过度拟合的风险，从而失去在测试集上很好地概括的能力。另一方面，如果隐藏层太小，神经网络将缺少参数来很好地拟合数据。