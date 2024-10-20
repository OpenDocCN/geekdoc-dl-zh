# 流式数据训练模型[实用指南]

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/training-models-on-streaming-data>

当你听到流式数据时，你会想到什么？可能是通过 YouTube 等视频流平台生成的数据，但这不是唯一有资格作为流数据的东西。有许多平台和来源可以生成这种数据。

在本文中，我们将介绍流数据的基础知识，它是什么，以及它与传统数据有何不同。我们还将熟悉有助于记录这些数据并对其进行进一步分析的工具。在本文的后面部分，我们将讨论它的重要性，以及我们如何在一个动手示例的帮助下使用机器学习进行流数据分析。

## 什么是流数据？

> *“流数据是连续的信息流，是事件驱动架构软件模型的基础”——*[*red hat*](https://web.archive.org/web/20230212113518/https://www.redhat.com/en/topics/integration/what-is-streaming-data)

世界各地的企业比以往任何时候都更加依赖数据。一些行业不仅依赖传统数据，还需要来自安全日志、物联网传感器和 web 应用程序等来源的数据来提供最佳客户体验。例如，在任何视频流媒体服务之前，用户必须等待视频或音频下载。如今，当你在听一首歌或一个视频时，如果你打开了自动播放，平台会根据你的实时流媒体数据为你创建一个播放列表。

### 批处理与流式处理

> *“随着当今现代需求的复杂性，传统的数据处理方法对于大多数用例来说已经过时，因为它们只能将数据作为随时间收集的事务组来处理。现代组织需要在数据变得陈旧之前对最新的数据采取行动。这种连续的数据提供了许多优势，正在改变企业的运营方式。”–*[*上升溶剂*](https://web.archive.org/web/20230212113518/https://www.upsolver.com/blog/streaming-data-architecture-key-components)

在开发应用程序或系统时，了解应用程序或用户等待数据可用的时间很重要，这是您必须在批处理和流数据处理之间做出选择的地方。在本文中，我们的重点是流数据，但是在我们处理它之前，理解它与批量数据处理的不同是很重要的。这也有助于我们观察流数据的重要性。

|  | 成批处理 | 流式处理 |
| --- | --- | --- |
|  | 

用于对不同数据集运行任意查询

 | 

最适合事件驱动的系统

 |
|  |  | 

最适合事件驱动的系统

 |
|  | 

一次性处理大量数据集

 | 

实时处理数据

 |
|  | 

输入流是静态的，通常大小有限

 | 

输入流是动态的，大小未知

 |
|  | 

用于复杂分析

 | 

用于简单和滚动的度量

 |
|  | 

批处理作业完成后才收到响应

 | 

数据一到就收到响应

 |
|  |  |  |

## 流式数据处理体系结构

> *“传统上，需要实时响应事件的应用依赖于数据库和消息处理系统。这样的系统跟不上今天产生的数据洪流。”–*[*红帽*](https://web.archive.org/web/20230212113518/https://www.redhat.com/en/topics/integration/what-is-streaming-data)

![Illustration of basic I/O flow in Streaming Data Processing](img/605bfec37e5f78f700249cc125067cf4.png)

*Basic I/O flow in streaming data processing | [Source](https://web.archive.org/web/20230212113518/https://memgraph.com/blog/batch-processing-vs-stream-processing)*

流式处理引擎不仅仅是将数据从一个地方传送到另一个地方，它还会在数据经过时对其进行转换。这一管道促进了信息的顺畅、自动化流动，防止了企业面临的许多问题，如数据损坏、冲突和数据条目重复。流式数据管道是一个增强版本，能够大规模实时处理数百万个事件。因此，可以收集、分析和存储大量信息。有了这一功能，应用程序、分析和报告可以实时完成。

机器学习模型是流处理引擎的一部分，它提供了帮助流数据管道在流中以及潜在地在历史数据存储中展示特征的逻辑。

有许多工具可以帮助收集和处理流数据，其中一些流行的工具包括:

*   Apache Kafka :一个开源的分布式事件流平台，每秒可以处理数百万个事件。它可以用来实时收集、存储和处理流数据。

*   Apache Flink :一个开源的分布式流处理框架，可以处理批处理和流数据。它可用于执行复杂的数据处理任务，如窗口聚合、连接和事件时处理。

*   [Apache Spark](https://web.archive.org/web/20230212113518/https://spark.apache.org/) :开源的分布式计算系统，可以处理大数据处理任务。它可以用于处理批处理和流数据，并内置了对机器学习和图形处理的支持。

*   Apache NiFi :一个开源工具，可以用来自动收集、处理和分发数据。它为构建数据管道提供了一个基于 web 的界面，可用于处理批处理和流数据。

*   [Azure Stream Analytics](https://web.archive.org/web/20230212113518/https://azure.microsoft.com/en-us/products/stream-analytics) :一种基于云的服务，可用于实时处理流媒体数据。它提供了各种各样的特性，比如数据接收、数据转换和实时处理。

这些只是可用于流数据收集和处理的许多工具中的几个例子。工具的选择将取决于应用程序的具体要求，例如数据的量和速度、数据处理的复杂性以及可伸缩性和容错需求。

## 流式数据的机器学习:实践指南

现在，我们已经对什么是流数据、它在哪里被使用以及它与批处理数据有什么不同有了一个相当好的理解，让我们动手来学习如何用几行代码设置流处理。

在本练习中，我们将使用 Tensorflow、Keras、Scikit-learn 和 Pandas 来预处理数据并创建机器学习模型。为了建立数据流/连续数据流，我们将使用[卡夫卡](https://web.archive.org/web/20230212113518/https://kafka.apache.org/)和[动物园管理员](https://web.archive.org/web/20230212113518/https://zookeeper.apache.org/)。

首先，让我们安装必要的库:

```py
 ```
!pip install tensorflow==2.7.1
!pip install tensorflow_io==0.23.1
!pip install kafka-python
```py 
```

导入所有函数和各自的库:

```py
 ```
import os
from datetime import datetime
import time
import threading
import json
from kafka import KafkaProducer
from kafka.errors import KafkaError

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
```py 
```

我们使用 Kafka 进行流处理，因为 Kafka 流提供了真正的一次记录处理能力。它是一个消息代理，从中可以轻松地使用消息(数据)。像 Spark 这样的工具也使用 Kafka 来读取消息，然后将它们分成小批来进一步处理。这取决于用例以及我们想要的工具。在这个练习中，我们使用 Kafka，因为它是最流行的工具之一。此外，python Kafka 库易于使用和理解。

让我们在本地安装和设置 Kafka，这样我们就可以轻松地模拟流数据环境:

```py
 ```
!curl -sSOL https://downloads.apache.org/kafka/3.3.2/kafka_2.13-3.3.2.tgz
!tar -xzf kafka_2.13-3.3.2.tgz

!./kafka_2.13-3.3.2/bin/zookeeper-server-start.sh -daemon ./kafka_2.13-3.3.2/config/zookeeper.properties
!./kafka_2.13-3.3.2/bin/kafka-server-start.sh -daemon
./kafka_2.13-3.3.2/config/server.properties
!echo "Waiting for 10 secs until kafka and zookeeper services are up and running"
!sleep 10
```py 
```

为训练和测试数据创建 Kafka 主题:

```py
 ```
!./kafka_2.13-3.3.2/bin/kafka-topics.sh --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 1 --topic cancer-train
!./kafka_2.13-3.3.2/bin/kafka-topics.sh --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 2 --topic cancer-test
```py 
```

![Created topic cancer-train. 
Created topic cancer-test. ](img/b7161ada48139780c8ba64c0b5997d99.png)

出于本练习的目的，我们将使用乳腺癌数据集，并在接下来的几个步骤中将其提供给 Kafka 主题。这个数据集是一个批处理数据集，但是通过将它存储在 Kafka 中，我们模拟了一个为训练和推理提供连续数据检索的环境。

```py
 ```
cancer_df = pd.read_csv('breast-cancer-wisconsin.data.csv')
cancer_df.head()
```py 
```

用 0 和 1 替换列“Class”值

```py
 ```
cancer_df['Class'] = cancer_df['Class'].replace(2,0)
cancer_df['Class'] = cancer_df['Class'].replace(4,1)
```py 
```

创建训练和测试子集:

```py
 ```
train_df, test_df = train_test_split(cancer_df,                                     test_size=0.4,                                     shuffle=True)

print("Number of training samples: ",len(train_df))
print("Number of testing sample: ",len(test_df))

x_train_df = train_df.drop(["Class"], axis=1)
y_train_df = train_df["Class"]

x_test_df = test_df.drop(["Class"], axis=1)
y_test_df = test_df["Class"]
```py 
```

![Number of training samples: 419 
Number of testing samples: 280 ](img/8bcfc83ee291bcef7efab4df7ef6f0a2.png)

标签，即类标签，被设置为存储在多个分区中的 Kafka 消息的密钥。这使得使用消费者组进行高效的数据检索成为可能。

```py
 ```
x_train = list(filter(None,                       x_train_df.to_csv(index=False).split("\n")[1:]))
                    y_train = list(filter(None,                       y_train_df.to_csv(index=False).split("\n")[1:]))
x_test = list(filter(None,                      x_test_df.to_csv(index=False).split("\n")[1:]))                     y_test = list(filter(None,                      y_test_df.to_csv(index=False).split("\n")[1:]))
```py 
```

是时候将数据推送到我们之前创建的 Kafka 主题了。

```py
def error_callback(exc):
      raise Exception('Error while sending data to kafka: {0}'.format(str(exc)))

def write_to_kafka(topic_name, items):
      count=0
      producer = KafkaProducer(bootstrap_servers=['127.0.0.1:9092'])
      for message, key in items:
        print(message.encode('utf-8'))
        producer.send(topic_name,
                      key=key.encode('utf-8'),
                      value=message.encode('utf-8')).add_errback(error_callback)
        count+=1
      producer.flush()
      print("Wrote {0} messages into topic: {1}".format(count, topic_name))

write_to_kafka("cancer-train", zip(x_train, y_train))
write_to_kafka("cancer-test", zip(x_test, y_test))
```

![Wrote 419 messages into topic: cancer-train
Wrote 280 messages into topic: cancer-test](img/d6ecb119c2262876eecbf05af401599a.png)

为了从 Kafka 主题中读取数据，我们需要对数据进行解码，并创建一个可用于模型训练的数据集。

```py
 ```
def decode_kafka_item(item):
      message = tf.io.decode_csv(item.message,
                                [[0.0] for i in range(NUM_COLUMNS)])
      key = tf.strings.to_number(item.key)
      return (message, key)

BATCH_SIZE=64
SHUFFLE_BUFFER_SIZE=64

train_ds = tfio.IODataset.from_kafka('cancer-train', partition=0, offset=0)
train_ds = train_ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
train_ds = train_ds.map(decode_kafka_item)
train_ds = train_ds.batch(BATCH_SIZE)
```py 
```

让我们准备模型创建、设置优化器、损失和指标:

```py
 ```
OPTIMIZER = "adam"
LOSS = tf.keras.losses.BinaryCrossentropy(from_logits=True)
METRICS = ['accuracy']
EPOCHS = 10
```py 
```

设计和构建模型:

现在，我们将编译该模型:

```py
 ```
model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(9,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())
```py 
```

是时候用卡夫卡的主题来训练模型了。在线或增量学习不同于训练模型的传统方式，在传统方式中，您提供一批数据值，并让模型在相同的基础上进行训练。然而，对于流数据，当新的数据点到达管道时，模型应该继续递增地更新超参数。在在线学习/培训中，数据点一旦用于培训(或消息被阅读)就可能不可用。

```py
 ```
model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
```py 
```

我们将逐步训练我们的模型，也可以定期保存，以后，我们可以利用它来推断测试数据。

```py
 ```
online_train_ds = tfio.experimental.streaming.KafkaBatchIODataset(
    topics=["cancer-train"],
    group_id="cgonline",
    servers="127.0.0.1:9092",
    stream_timeout=10000, 
    configuration=[
        "session.timeout.ms=7000",
        "max.poll.interval.ms=8000",
        "auto.offset.reset=earliest"
    ],
)
```py 
```

这就是我们如何保持数据流入，并不断训练我们的模型。要了解更多关于 tensorflow streaming api 的信息，请查看[这个](https://web.archive.org/web/20230212113518/https://www.tensorflow.org/io/api_docs/python/tfio/experimental/streaming)页面。

```py
 ```
def decode_kafka_online_item(raw_message, raw_key):
    message = tf.io.decode_csv(raw_message, [[0.0] for i in range(NUM_COLUMNS)])
    key = tf.strings.to_number(raw_key)
    return (message, key)

for mini_ds in online_train_ds:
    mini_ds = mini_ds.shuffle(buffer_size=32)
    mini_ds = mini_ds.map(decode_kafka_online_item)
    mini_ds = mini_ds.batch(32)
    if len(mini_ds) > 0:
      model.fit(mini_ds, epochs=3)
```py 
```

流式数据的重要性和含义

根据不同的用例，还需要收集和处理更多的数据。批量数据处理对于今天的企业业务来说不再可行。实时数据流的使用无处不在，从欺诈检测和股市平台到乘车共享应用和电子商务网站。虽然它引起了一些关于隐私和安全的担忧，但好处远不止这些。

## 重要

随着[流数据](https://web.archive.org/web/20230212113518/https://www.confluent.io/learn/data-streaming/)变得越来越普遍，应用程序可以在收到流数据时实时处理、过滤、分析并做出反应。因此，各种新的机会变得可用，例如实时欺诈检测、网飞推荐和跨多种设备的无缝购物。处理大数据的行业正受益于持续的实时数据。

### 实时决策:流式数据使组织能够实时处理和分析数据，从而做出快速、明智的决策。这对于金融、医疗和运输等行业尤其有用，因为在这些行业，时间是至关重要的。

[改善客户体验](https://web.archive.org/web/20230212113518/https://www.ttec.com/articles/streaming-data-opens-door-new-customer-insights):流式数据可用于实时监控和分析客户互动，从而改善客户服务并提供个性化建议。

*   预测分析:流数据可用于实时训练机器学习模型，这些模型可用于预测分析和预测。

*   [运营效率](https://web.archive.org/web/20230212113518/https://www.teradata.com/Trends/Data-Management/Streaming-Data):流式数据可用于监控和分析工业设备的性能，从而提高运营效率，减少停机时间。

*   [欺诈检测](https://web.archive.org/web/20230212113518/https://www.oreilly.com/library/view/streaming-architecture/9781491953914/ch06.html):流式数据可用于实时检测和预防欺诈活动，这有助于组织将财务损失降至最低。

*   [物联网](https://web.archive.org/web/20230212113518/https://solace.com/blog/real-time-data-streaming-in-iot/):流式数据对于物联网设备通信和数据收集非常重要，它允许设备实时[发送和接收数据](https://web.archive.org/web/20230212113518/https://www.confluent.io/blog/stream-processing-iot-data-best-practices-and-techniques/)，并有助于更准确和高效地做出决策。
*   含义
*   根据使用的环境，流数据可以有多种含义。

### 实时处理:流数据允许实时处理和分析，这对于各种应用程序都很有用，例如监控系统、金融交易和在线客户交互。

可扩展性:流数据系统旨在处理大量数据，非常适合社交媒体分析和物联网数据处理等大数据应用。

*   延迟:流数据系统通常延迟较低，这意味着数据生成和处理之间的时间很短。这对于需要快速响应时间的应用来说非常重要，例如金融交易或自动驾驶汽车。

*   复杂性:流数据系统的设计、实现和维护可能很复杂，尤其是在处理大量数据、多个数据源和实时需求时。

*   安全性:流数据也可能意味着安全风险，因为它增加了攻击面和暴露的数据量，因此拥有强大的安全基础设施非常重要。

*   隐私:流数据系统也可能引起隐私问题，因为它们经常收集和处理大量的个人信息。务必确保数据的收集和使用符合相关法律法规，并采取适当措施保护用户隐私。

*   实时处理和分析数据的能力可以为组织提供显著的竞争优势，提高客户满意度并做出更明智的决策。

结论

流式数据处理及其架构可以消除运行可扩展数据工程功能的需求。它也很灵活，可以适应任何使用情况。随着时间的推移，流数据变得越来越流行，我们需要建立一个基于 ML 的系统，它可以使用这些实时数据，并有助于更复杂的数据分析。在本文中，我们学习了流式数据以及如何处理它。我们还看到了它与批量数据处理的不同之处。

## 我们还熟悉了一些可以帮助我们收集流数据的工具，后来在动手练习中，我们使用了其中的一个工具——Kafka。在动手练习中，我们看到了如何设置卡夫卡主题，以及如何将数据输入其中。一旦 Kafka 主题的数据可用，我们就可以解码并利用它来逐步训练我们的机器学习模型。

对于未来的工作，我们可以利用 [Twitter API](https://web.archive.org/web/20230212113518/https://developer.twitter.com/en/docs/twitter-api) 并创建一个用于情感分析的[机器学习模型](https://web.archive.org/web/20230212113518/https://datascienceplus.com/brexit-tweets-sentiment-analysis-in-python/)，而不是使用 csv 文件。

快乐学习！

参考

[机器学习&流式数据管道架构](https://web.archive.org/web/20230212113518/https://www.eckerson.com/articles/machine-learning-and-streaming-data-pipelines-part-i-definitions-and-architecture)

### [流首个实时 ML](https://web.archive.org/web/20230212113518/https://www.infoq.com/articles/streaming-first-real-time-ml/)

1.  [流式数据的机器学习:最新技术](https://web.archive.org/web/20230212113518/https://kdd.org/exploration_files/3._CR_7._Machine_learning_for_streaming_data_state_of_the_art-Final.pdf)
2.  [用 Creme 对数据流进行机器学习](https://web.archive.org/web/20230212113518/https://towardsdatascience.com/machine-learning-for-streaming-data-with-creme-dacf5fb469df)
3.  [流式数据的 ML 预测](https://web.archive.org/web/20230212113518/https://towardsdatascience.com/ml-prediction-on-streaming-data-using-kafka-streams-1e4ebd21008)
4.  [对流式数据的连续机器学习](https://web.archive.org/web/20230212113518/https://conferences.oreilly.com/strata/strata-ny-2018/public/schedule/detail/68992.html)
5.  [为高频流数据建立人工智能模型](https://web.archive.org/web/20230212113518/https://www.kdnuggets.com/2020/12/mathworks-pt2-ai-models-streaming-data.html)
6.  [数据流的机器学习:真实例子](https://web.archive.org/web/20230212113518/https://direct.mit.edu/books/book/4475/Machine-Learning-for-Data-Streamswith-Practical)
7.  [Building AI models for High Frequency Streaming Data](https://web.archive.org/web/20230212113518/https://www.kdnuggets.com/2020/12/mathworks-pt2-ai-models-streaming-data.html)
8.  [Machine Learning for Data Streams:Real Examples](https://web.archive.org/web/20230212113518/https://direct.mit.edu/books/book/4475/Machine-Learning-for-Data-Streamswith-Practical)