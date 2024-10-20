# 为 NLP 构建 MLOps 管道:机器翻译任务[教程]

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/mlops-pipeline-for-nlp-machine-translation>

机器学习操作通常被称为 **[MLOps](https://web.archive.org/web/20221203090558/https://ml-ops.org/)** 使我们能够创建一个端到端的机器学习管道，从设计实验、构建 ML 模型、训练和测试，到部署和监控，换句话说就是 **[机器学习生命周期](/web/20221203090558/https://neptune.ai/blog/life-cycle-of-a-machine-learning-project)** 。MLOps 的这个领域类似于 DevOps，但专门为机器学习项目量身定制。

作为一个相对较新的领域，像机器学习这样的 MLOps 已经获得了很大的吸引力，正因为如此，人工智能驱动的软件正在所有行业中流行。我们必须有专门的操作人员来完成这一过程。MLOps 使我们能够利用其两个主要组件来构建人工智能驱动的软件:持续集成和持续部署。我们可以创建从开始到部署的无缝管道，并随时修改构建。

在本文中，我们将详细讨论如何使用各种技术为机器翻译构建 MLOps 管道。我们将使用的一些关键技术是:

*   唐斯多夫，
*   海王星啊！
*   GitHub 股份公司，
*   码头工，
*   Kubernetes，
*   和谷歌云构建。

本教程旨在为您提供如何为您自己的机器学习或数据科学项目逻辑地实现 MLOps 的完整理解。

## 什么是 MLOps 管道？

MLOps 可以被描述为机器学习或数据科学项目的生命周期。生命周期本身由三个主要部分组成:

## 

*   1 设计
*   2 模型开发
*   3 操作

通过结合这三个部分，我们可以建立一个集成的系统，可以利用机器学习和软件应用程序的力量。这个管道自动化了数据收集、数据预处理、培训和测试、部署和监控的过程。除此之外，它还能够检测构建中的任何新变化，并同时更新全局的新变化。

## 为机器翻译构建 MLOps 管道:从哪里开始？

为了打造流畅的 MLOps 生命周期管道，必须考虑以下步骤。

### 设计

设计基本上是理解通常与目标受众打交道的**业务问题**的过程。

设计实验还包括研究可用的资源，如知识收集、可以使用的数据类型、合适的体系结构、财务资源、计算资源等等。

通常，在这个过程中，数据科学家和机器学习工程师为了节省时间，会尝试寻找可用的解决方案，并根据要求进行修改。

本质上，设计通过预测的解决方案设定目标。

#### 问题陈述

作为本文的一部分，让我们考虑我们需要构建一个应用程序，将葡萄牙语翻译成英语。这个问题属于自然语言处理的范畴，更具体地说是机器翻译。现在，作为一名数据科学家或 ML 工程师，您需要考虑以下几点:

## 

*   1 应该使用什么计算语言和相关库？
*   我们可以从哪里获得数据？
*   3 模型的核心架构必须是什么？
*   4 培训目标和输出应该是什么，以及准确性和损失指标、优化技术等等。
*   截止日期和预算是什么？

#### 研究

研究是我们探索每一个可能的解决方案来制造产品的一部分。例如，在选择构建深度学习模型的语言时，python 是最佳选择。但如果你是一名 iOS 开发者，那么 Swift 是首选语言，一些公司如 Tesla 确实考虑 C 和 C++以及 Python。对于本文，让我们坚持使用 python，因为它是构建深度学习和机器学习模型最广泛使用的语言。

现在，为了构建深度学习模型，可以使用两个 python 库中的一个:Tensorflow 和 Pytorch。两者都非常受欢迎，多才多艺，并拥有大量的社区支持。在这个阶段，这一切都归结为偏好和一方相对于另一方的优势。在我们的例子中，我们将使用 Tensorflow，因为它有一个非常结构化的 API，即 Keras，并且与 Pytorch 相比，实现它需要更少的代码行。

当语言和核心库设置好后，我们就可以研究可以用来实现机器翻译的架构了。目前我们知道大多数 SOTA 语言模型大量使用**变形金刚**，因为它的**自我关注机制**。所以我们也会这样做。

说到数据，我们可以很容易地从几乎任何地方下载语言翻译数据，但最佳实践是从合法资源下载精选数据，如 Kaggle。在我们的例子中，我们将使用 **TensorFlow-dataset** API 来下载数据。

现在让我们了解一下目录结构。

#### 目录结构

在所有项目中，最关键的是**目录结构**。一个结构良好的项目有利于读者高效地跟踪和协作。就 MLOps 而言，它扮演着重要的角色，因为从构建到部署，我们将使用不同的技术来访问端点。

MLOps 项目的一般结构如下所示:

```py
Machine-translation
├── kube
├── metadata
├── notebook
├── requirements.txt
├── README.md
└── source
```

这是主目录结构及其子目录 requirements.txt 和 README.md 文件。随着我们继续前进，我们将不断向目录中添加更多的文件。

## 机器翻译的 MLOps 流水线:模型开发

为了这篇文章，我们将使用 Tensorflow 网站上提供的笔记本。该笔记本信息量很大，并给出了如何编写和训练机器翻译模型的透彻想法。

我们将对笔记本进行一些修改，并在培训期间集成 Neptune 客户端来监控模型。现在我们来简单的探讨一下，修改一下笔记本。

### 设置

首先，我们必须安装三个库: **Tensorflow-datasets** 用于下载数据， **Tensorflow** 用于深度学习，以及 **Neptune-client** 用于监控和保存元数据。

```py
!pip install tensorflow_datasets
!pip install -U 'tensorflow-text==2.8.*'

!pip install neptune-client
```

一旦安装了库，我们就可以将它们全部导入到笔记本中。

### 下载数据集

我们将使用的数据集，即将葡萄牙语翻译成英语，可以直接从 TensorFlow-datasets 库中下载。一旦数据集被下载，我们就可以把它分成训练数据集和验证数据集。

```py
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']
```

### 创建 requirements.txt

Requirements.txt 是一个重要的文件，因为它包含了所有的库。这允许新的贡献者在他们的工作环境中快速安装所有的库或依赖项。

要创建一个 **requirement.txt** 文件，我们需要做的就是运行:

```py
!pip freeze > requirements.txt 
```

这可以在我们安装并导入所有文件之后，或者在您完成了对模型的训练和执行推断之后完成。优选地，实践后者。

requirements.txt 文件应该是这样的:

```py
matplotlib==3.2.2
neptune-client==0.16.1
numpy==1.21.6
tensorflow-datasets==4.0.1
tensorflow==2.8.0
```

### 跟踪模型元数据

[登录 neptune.ai](https://web.archive.org/web/20221203090558/https://docs.neptune.ai/you-should-know/logging-metadata) 仪表盘相当简单。首先，我们创建一个类，[存储所有的超参数](https://web.archive.org/web/20221203090558/https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#parameters-and-model-configuration)。这种方法在创建单独的 python 模块时非常方便(我们将在后面看到)。

```py
class config():
 BUFFER_SIZE = 20000
 BATCH_SIZE = 64
 MAX_TOKENS = 128
 MAX_EPOCHS = 5
 TRAIN_LOSS = 'train_loss'
 TRAIN_ACCURACY = 'train_accuracy'
 OPTIMIZER = 'Adam'
 BETA_1 = 0.9
 BETA_2 = 0.98
 EPSILON = 1e-9
 NUM_LAYER = 4
 D_MODEL = 128
 DFF = 512
 NUM_HEAD = 8
 DROP_OUT = 0.1
```

然后我们可以创建一个存储所有超参数的字典。

```py
params = {
   'BUFFER_SIZE': config.BUFFER_SIZE,
   'BATCH_SIZE' : config.BATCH_SIZE,
   "MAX_TOKENS" : config.MAX_TOKENS,
   "MAX_EPOCHS" : config.MAX_EPOCHS,
   "TRAIN_LOSS" : config.TRAIN_LOSS,
   "TRAIN_ACCURACY" : config.TRAIN_ACCURACY,
   "OPTIMIZER" : config.OPTIMIZER,
   "BETA_1" : config.BETA_1,
   "BETA_2" : config.BETA_2,
   "EPSILON" : config.EPSILON,
   "NUM_LAYER" : config.NUM_LAYER,
   "D_MODEL" : config.D_MODEL,
   "DFF" : config.DFF,
   "NUM_HEAD" : config.NUM_HEAD,
   "DROP_OUT" : config.DROP_OUT,
}
```

一旦创建了字典，我们就可以使用 [API 令牌](https://web.archive.org/web/20221203090558/https://docs.neptune.ai/getting-started/installation#authentication-neptune-api-token)初始化 Neptune 客户端，并将参数作为字典传递。

```py
run = neptune.init(
   project="nielspace/machine-translation",
   api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkYjRhYzI0Ny0zZjBmLTQ3YjYtOTY0Yi05ZTQ4ODM3YzE0YWEifQ==",
)

run["parameters"] = params
```

一旦执行，这将是它在 neptune.ai 仪表板中的外观。

[![Loging into Neptune](img/96d3a6073cc3d6f86288697c84e8056e.png)](https://web.archive.org/web/20221203090558/https://neptune.ai/building-mlops-pipeline-for-machine-translation-task-step-by-step-tutorial6)

*Parameters logged in Neptune.ai | [Source](https://web.archive.org/web/20221203090558/https://app.neptune.ai/nielspace/machine-translation/e/MAC-25/all?path=parameters%2F)*

如您所见，所有的超参数都被记录下来。

### 模特培训

一旦模型的所有组件如编码器、解码器、自我关注机制等都准备好了，就可以训练模型了。但是，我们必须再次确保在训练期间集成 Neptune-client 来监控模型，以查看它的表现如何。

要做到这一点，我们只需要定义精度和损失函数，并将它们传递到训练循环中。

```py
train_loss = tf.keras.metrics.Mean(name=config.TRAIN_LOSS)
train_accuracy = tf.keras.metrics.Mean(name=config.TRAIN_ACCURACY)
```

在训练循环中，我们将使用与之前相同的方法来记录准确性和损失。

```py
run['Training Accuracy'].log(train_accuracy.result())
run['Training Loss'].log(train_loss.result())
```

让我们将它们整合到培训循环中。

```py
for epoch in range(config.MAX_EPOCHS):
 start = time.time()

 train_loss.reset_states()
 train_accuracy.reset_states()

 for (batch, (inp, tar)) in enumerate(train_batches):
   train_step(inp, tar)
   run['Training Accuracy'].log(train_accuracy.result())
   run['Training Loss'].log(train_loss.result())

   if batch % 50 == 0:
     print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

 if (epoch + 1) % 5 == 0:
   ckpt_save_path = ckpt_manager.save()
   print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

 print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

 print(f'Time taken for 1 epoch: {time.time() - start:.2f} secsn')

run.stop()
```

这是培训期间仪表板的外观。

[![Training in Neptune](img/088b6bea32c7b516f8ad5afb5379f46e.png)](https://web.archive.org/web/20221203090558/https://neptune.ai/building-mlops-pipeline-for-machine-translation-task-step-by-step-tutorial21)

*Training accuracy and loss logged in Neptune.ai | [Source](https://web.archive.org/web/20221203090558/https://app.neptune.ai/nielspace/machine-translation/e/MAC-25/charts)*

要记住的一个关键点是，在实验完成或训练循环完全执行后停止运行。

Neptune-client API 的一个好处是，您可以记录几乎任何事情。

### 验证和测试模型

一旦训练完成，我们就可以在创建应用程序之前对模型进行推理测试。当创建一个用于推理的类对象时，你必须记住，所有的预处理步骤都必须包含在内，因为这个相同的类将在 app.py 中用来创建端点。

这里有一个例子:

```py
class Translator(tf.Module):
 def __init__(self, tokenizers, transformer):
   self.tokenizers = tokenizers
   self.transformer = transformer

 def __call__(self, sentence, max_length=config.MAX_TOKENS):

   assert isinstance(sentence, tf.Tensor)
   if len(sentence.shape) == 0:
     sentence = sentence[tf.newaxis]

   sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

   encoder_input = sentence

   start_end = self.tokenizers.en.tokenize([''])[0]
   start = start_end[0][tf.newaxis]
   end = start_end[1][tf.newaxis]

   output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
   output_array = output_array.write(0, start)

   for i in tf.range(max_length):
     output = tf.transpose(output_array.stack())
     predictions, _ = self.transformer([encoder_input, output], training=False)

     predictions = predictions[:, -1:, :]  

     predicted_id = tf.argmax(predictions, axis=-1)

     output_array = output_array.write(i+1, predicted_id[0])

     if predicted_id == end:
       break

   output = tf.transpose(output_array.stack())

   text = tokenizers.en.detokenize(output)[0]  

   tokens = tokenizers.en.lookup(output)[0]

   _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)

   return text, tokens, attention_weights
```

如您所见，预处理和预测所需的步骤包含在同一个类对象中。现在我们测试我们的模型在看不见的数据上的表现。

```py
def print_translation(sentence, tokens, ground_truth):
 print(f'{"Input:":15s}: {sentence}')
 print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
 print(f'{"Ground truth":15s}: {ground_truth}')

sentence = 'este é um problema que temos que resolver.'
ground_truth = 'this is a problem we have to solve .'

translator = Translator(tokenizers, transformer)
translated_text, translated_tokens, attention_weights = translator(
   tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)
```

输出:

```py
Input:         : este é um problema que temos que resolver.
Prediction     : this is a problem that we have to solve .
Ground truth   : this is a problem we have to solve .
```

### 下载元数据和笔记本

一旦训练和推理完成，我们就可以从 Google Colab 下载元数据和笔记本到我们的本地目录，然后为每个类对象创建单独的 python 模块。

例如，你可以看到所有的目录已经完全被它们各自的文件和元数据填满了。

```py
machine-translation
├── metadata
│   ├── checkpoints
│   │   └── train
│   │       ├── checkpoint
│   │       ├── ckpt-1.data-00000-of-00001
│   │       └── ckpt-1.index
│   ├── ted_hrlr_translate_pt_en_converter
│   │   ├── assets
│   │   │   ├── en_vocab.txt
│   │   │   └── pt_vocab.txt
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── ted_hrlr_translate_pt_en_converter.zip
│   └── translator
│       ├── assets
│       │   ├── en_vocab.txt
│       │   └── pt_vocab.txt
│       ├── saved_model.pb
│       └── variables
│           ├── variables.data-00000-of-00001
│           └── variables.index
├── notebook
├── requirements.txt
└── source
    ├── attention.py
    ├── config.py
    ├── decoder.py
    ├── encoder.py
    ├── inference.py
    ├── preprocessing.py
    ├── train.py
    └── transformer.py

```

在上面的例子中，源目录由 python 模块组成，这些模块包含直接取自笔记本的函数和类对象。

### app.py

下一步是创建一个 app.py，它将利用 flask API 为模型服务。为了服务于该模型，我们需要:

## 

*   1 从预处理和转换模块导入所有函数。
*   2 加载保存在翻译目录中的重量。
*   3 定义获得预测的端点。

```py
import flask
from flask import Flask
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text

import source.config as config
from source.preprocessing import *
from source.transformer import *

app = Flask(__name__)

@app.route("/predict")
def predict():
   sentence = request.args.get("sentence")
   response = translator(sentence).numpy()
   return flask.jsonify(response)

if __name__ == "__main__":
   translator = tf.saved_model.load('../translator')
   app.run(host="0.0.0.0", port="9999")
```

## 机器翻译的 MLOps 管道:操作(CI/CD)

一旦代码库准备就绪，我们就可以继续我们的第三个也是最后一个阶段，即运营，这是我们将实施持续部署和持续集成的地方。这就是事情变得有点棘手的地方。但是不要担心，我已经把整个操作分成了不同的部分，这样每个模块都清晰易懂。这些是我们将遵循的步骤:

## 

*   1 创建 Github repo。
*   使用 Docker 文件创建图像。
*   3 将图像推送到 Google Cloud Build。
*   4 展开。

### 创建 GitHub repo

第一步是将您的所有代码从本地目录推到您的 GitHub 帐户。这将有助于我们将回购目录连接到谷歌云开发。在那里，我们可以创建 Kubernetes pods 并在其中部署应用程序。

### Dockerfile

dockerfile 使我们能够创建一个容器*，这是一种包装应用程序的软件，包括它的库和依赖关系*。它还创建了一个静态环境，使应用程序能够在任何环境中运行。

另一方面，Docker 是构建和管理容器的软件。我们可以创建任意多的容器。

为了给我们的 app.py 创建一个容器，我们需要创建一个 **Dockerfile** 。该文件即*映像*必须存储构建**容器**所需遵循的所有指令。一些一般说明如下:

## 

*   安装编程语言，在我们的例子中是 python。
*   2 创造环境。
*   3 安装库和依赖项。
*   4 复制模型、API 和其他实用程序文件，以便正确执行。

在下面的例子中，你会看到我是如何构建 Docker 配置的。相当简约极简。

```py
FROM python:3.7-slim
RUN apt-get update

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN ls -la $APP_HOME/

RUN pip install -r requirements.txt

CMD ["python3","app.py" ]
```

### 谷歌云开发

配置 Dockerfile 后，我们将使用 Google Cloud Development 或 GCD 来自动化 CI/CD 管道。当然，你可以使用任何其他服务，比如 Azure、AWS 等等，但是我发现 GCD 简单易用。我将详细解释这一部分，以便您可以轻松掌握完整的概念并理解该过程。

第一步是登录你的 GCD 账户，如果你是新用户，你将获得 300 美元的免费积分，可以在 90 天内使用。

#### 谷歌云开发:云壳

一旦你登录到你的 GCD，创建一个**新项目。**之后点击屏幕右上角的云壳图标激活云壳。

需要注意的一点是，在大多数情况下，我们将使用内置的**云外壳**来执行流程。第一步是从 Github 克隆存储库。您可以在给定的[链接](https://web.archive.org/web/20221203090558/https://github.com/Nielspace/tensorflow-machine-translation)中使用相同的代码。

![Google Cloud development: Cloud Shell ](img/c44491ed768f67e65d30822bcdfd3657.png)

*Google Cloud development: Cloud Shell | Source: Author*

一旦回购被克隆到 cd 中。

#### 谷歌云开发:Kubernetes

现在我们将**启用**，**设置，**和**启动**Kubernetes 引擎。Kubernetes 引擎将允许我们管理我们的 dockerized 应用程序。要配置 Kubernetes，我们需要遵循 4 个步骤:

## 

*   1 创建 deployment.yaml 文件。
*   2 创建一个 service.yaml 文件。
*   3 在 GCD 中启用 Kubernetes 引擎。
*   4 通过云壳启动 Kubernetes 引擎。

**创建一个 deployment.yaml**

deployment.yaml 的目的是配置部署设置。它由两部分组成:

*   API 版本和操作种类

```py
apiVersion: apps/v1
kind: Deployment
```

```py
metadata:
 name: translation

spec:
 replicas: 2 
 selector:
   matchLabels:
     app: translation-app 
 template:
   metadata:
     labels:
       app: translation-app 
   spec:
     containers:
     - name: translation-app
       image: gcr.io/tensor-machine-translation/translation:v1
       ports:
       - containerPort: 9999
```

下面是我在元数据中指定的配置内容:

*   在本例中，我将使用的复制副本或 pod 的数量是 2。
*   集装箱位置。该位置也可以分为四个步骤:
    *   URL: " **gcr.io** "。这是常见的容器地址。
    *   项目 ID: " **张量机器翻译**"
    *   App 名称:**翻译**。您可以给出任何名称，但它必须在所有配置文件中相同。
    *   版本: **v1** 。这是一个标签。

**创建一个 service.yaml**

service.yaml 将整个应用程序暴露给网络。它类似于 deployment.yaml，但主要区别在于我们想要实现的操作类型，它也由两部分组成:

*   API 版本和操作种类

```py
apiVersion: v1
kind: Service
```

```py
metadata:
 name: machinetranslation
spec:
 type: LoadBalancer
 selector:
   app: machinetranslation
 ports:
 - port: 80
   targetPort: 9999
```

配置完部署和服务文件后，让我们转到 Kubernetes。

**在 GCD 中启用 Kubernetes 引擎**

一旦我们的配置文件准备好了，我们就可以开始启用 Kubernetes API 了。这是一个用于管理 Dockerized 应用程序的开源系统。要启用 Kubernetes，只需在 GCD 搜索栏中输入*‘GKE’*。只要点击**‘启用**，你就会被导航到 Kubernetes 引擎 API 页面。

![Building MLOPS pipeline with Kubernets](img/e534dc2d5d85ea658cc968ea48f1e85a.png)

*Building MLOPS pipeline with Kubernets | Source: Author* 

启用 API 后，您需要创建 Kubernetes 集群。有两种方法可以创建集群:

## 

*   1 只需点击屏幕上的“创建”按钮。
*   使用 Google-Cloud shell。

![Kubernets clusters](img/2f3027acaeae4e84c724a741e5f3ab8f.png)

*Kubernets clusters | Source: Author*

在我们的例子中，我们将使用 Google-Cloud shell，因为您可以更具体地了解您想要什么类型的集群。

**启动 Kubernetes 引擎**

要启动 Kubernetes 引擎，请在您的云 shell 中编写以下代码。

```py
!gcloud config set project tensor-machine-translation
!gcloud config set compute/zone us-central1
!gcloud container clusters create tensorflow-machine-translation --num-nodes=2
```

![Launching the Kubernetes engine](img/1a8e204884931f7220b25a7da2ccde5e.png)

*Launching the Kubernetes engine | Source: Author*

完成后，您可以在仪表板上查看 K8 集群。

![Launching the Kubernetes engine](img/c156438e95e1b9426ad2cf35157f6a46.png)

*Launching the Kubernetes engine | Source: Author*

现在，我们的 Kubernetes 配置文件和引擎已经准备好了，我们可以开始配置我们的 *clouldbuild.yaml* 文件，然后启动整个应用程序。

#### 谷歌云开发:云构建

cloudbuild.yaml 文件将所有进程同步在一起。这很容易理解。配置文件通常包含以下步骤:

## 

*   1 从 Dockerfile 文件构建容器映像。
*   2 将容器映像推送到 Google Cloud Registry (GCR)。
*   3 配置入口点。
*   4 在 Kubernetes 引擎中部署整个应用程序。

```py
steps:
- name: 'gcr.io/cloud-builders/docker'
 args: ['build', '-t', 'gcr.io/tensor-machine-translation/translation', '.']
 timeout: 180s
- name: 'gcr.io/cloud-builders/docker'
 args: ['push', 'gcr.io/tensor-machine-translation/translation']
- name: 'gcr.io/cloud-builders/gcloud'
 entrypoint: "bash"
 args:
 - "-c"
 - |
   echo "Docker Container Built"
   ls -la
   ls -al metadata/
- name: "gcr.io/cloud-builders/gke-deploy"
 args:
 - run
 - --filename=kube/
 - --location=us-west1-b
 - --cluster=tensor-machine-translation
```

配置完 cloudbuild.yaml 文件后，您可以返回到 Google-Cloud Shell 并运行以下命令:

```py
!gcloud builds submit --config cloudbuild.yaml

```

![Google Cloud development: Cloudbuild](img/019f70cb8e2d9e2db6f6dd3f7a81d440.png)

*Google Cloud development: Cloud Build | Source: Author*

部署完成后，您将获得应用程序的链接。

### GitHub 操作

现在，我们已经完成了所有重要的过程，我们的应用程序已经启动并运行了。但现在我们将看到如何使用 Github 动作创建触发器，这将使我们能够在修改和推送 Github repo 中的代码时自动更新构建。让我们看看我们能做些什么。

去 Github 市场搜索 Google Cloud Build。

![Github marketplace](img/b8933b9d31dd58bed4f7e7fb0ffc692c.png)

*Github marketplace | Source: Author*

点击谷歌云构建。

![Selecting Google Cloud build](img/0627340a2f3589f4eca309b6c6bb5f03.png)

*Selecting Google Cloud Build | Source: Author*

点击建立一个计划。

![Setting up a plan with Google Cloud build](img/6fb3256b5e1a5559ac7f059bfb54fd2a.png)

*Setting up a plan with Google Cloud Build | Source: Author*

![Setting up a plan with Google Cloud build](img/ae06843680899547a961a82a05b05c6a.png)

*Setting up a plan with Google Cloud Build | Source: Author*

点击配置。

![Google Cloud Build configuration](img/ff49615c26d916e3940cc2bafd13641d.png)

*Google Cloud Build configuration | Source: Author*

选择存储库。

![Selecting Google Cloud Build repository](img/e1cb418e496d860ad9482fe4f8335281.png)

*Selecting Google Cloud Build repository | Source: Author*

选择项目在我们的情况下，它是张量机器翻译。

![Selecting "Tensor-machine-translation"](img/bacb86a818fd0b1975f4bbae97c7c322.png)

*Selecting “Tensor-machine-translation” | Source: Author*

点击创建触发器。

![Creating a trigger](img/e23cacc23b03bbd94bc5cd02a7da69e1.png)

*Creating a trigger | Source: Author*

为触发器提供一个“名称”,并保留所有设置。

![Naming the trigger](img/82d0bb8d27d2e4ae6863aa41447c14e2.png)

*Naming the trigger | Source: Author*

点击**创建**。

![Creating a trigger](img/302211dc7bfdecd39c26d8b93c5e4e52.png)

*Creating a trigger | Source: Author*

创建后，您将被引导至以下页面。

![Trigger's view in Google Cloud Build](img/6bd87c0768731e8a8b0d1eb3f3c37670.png)

*Trigger’s view in Google Cloud Build | Source: Author*

现在，有趣的部分是，每当您对 Github repo 进行任何更改时，云构建都会自动检测到它，并在部署的构建中进行更改。

### 生产中的监控模型

到目前为止，我们已经了解了如何构建自动 MLOps 管道，现在我们将探讨最后一步，即如何监控部署在云上的应用。这个过程称为云部署。GCD 为我们提供了一个仪表板，使我们能够从任何设备监控应用程序:手机、平板电脑或电脑。

下面是 iOS 中给出的谷歌云控制台的图像。

![Google cloud console](img/6290b9f2839a220cbcce68312e65da8c.png)

*Google Cloud console | Source: Author*

为了监控该应用程序，您只需在 GCD 搜索栏中键入 monitor，它就会将您导航到相应的页面。

![Monitoring in Google Cloud](img/d6e0d3b2688096e0d1bf672ad187c079.png)

*Monitoring in Google Cloud | Source: Author*

你会看到一堆选项可供选择。我建议您首先选择一个概述，然后探索所有可能的选项。例如，如果您选择 GKE，那么您将看到关于 Kubernetes 的所有信息:各个项目的 pod、节点、集群等等。

![Monitoring in Google Cloud](img/1cbe6051f49af08322cff38abfd996f7.png)

*Monitoring in Google Cloud | Source: Author*

同样，您也可以创建警报策略。和许多其他事情。

![Monitoring in Google Cloud](img/4a534b680cc0847a70c9d10aecfbd4ba.png)

*Monitoring in Google Cloud | Source: Author*

在监控阶段，您必须熟悉的一个重要概念是**模型漂移**。当模型的预测能力随着时间的推移而下降时，就会发生这种情况。

模型漂移可分为两种类型:

#### 数据漂移

数据漂移通常是生产数据与训练/测试数据之间的差异。它通常发生在训练和部署之间有时间间隔的时候。一个这样的例子是任何时间序列数据，如新冠肺炎数据或股票市场数据。在这种情况下，像可变性这样的因素每小时都会被引入，换句话说，数据会随着时间的推移而不断发展。数据的这种演变会在生产阶段产生预测误差。

在我们的例子中，我们不必担心，因为语言数据在大多数情况下保持稳定，因此我们可以在更长的时间内使用相同的模型。

您可以在下面的博客中找到更多关于数据漂移检测的信息:[为什么数据漂移检测如此重要，以及如何通过 5 个简单的步骤实现自动化](https://web.archive.org/web/20221203090558/https://towardsdatascience.com/why-data-drift-detection-is-important-and-how-do-you-automate-it-in-5-simple-steps-96d611095d93)。

#### 概念漂移

当预测变量随时间变化时，或者换句话说，当输出的统计属性随时间变化时，就会发生概念漂移。在概念漂移中，模型无法利用它在训练期间提取的模式。例如，让我们说垃圾邮件自从被定义以来已经随着时间的推移而发展。现在，该模型将发现很难使用 3 周前在训练中提取的模式来检测垃圾邮件。为了解决这些问题，必须调整模型参数。

如果公司的商业模式正在发生变化，或者所使用的数据集不能代表整个人口，那么概念漂移就可能发生。

#### 如何监控一个模型？

*   像顺序分析和时间分布方法这样的方法有助于识别数据漂移。
*   另一方面，持续监控输入数据并观察其统计特性有助于克服概念漂移。
*   除了像 ADWIN、卡方检验、直方图交叉、kolmogorov smirnov 统计之类技术之外，这些技术也是有用的。

#### 如何克服模型漂移？

模型再训练是解决模型漂移(包括数据漂移、概念漂移和模型退化问题)的最好方法。再培训可以参考这些策略:

*   **定期再培训**–定期:每周、每月等等。
*   **数据/事件驱动**–每当新数据可用时。
*   **模型/度量驱动**–当精度低于阈值时。

## 结论

在本教程中，我们探讨了如何使用各种技术来无缝部署机器翻译应用程序。这就是 MLOps 的力量。当然，你可以像 UI/UX 一样在 app.py 文件中添加很多东西，创建一个全新的网页。

总而言之，我们看到:

## 1 如何设计一个实验？

*   2 训练和测试模型。
*   3 保存模型的权重和元数据。
*   4 构建目录。
*   5 分离功能，创建 python 模块。
*   6 创建 Flask app。
*   7 将应用程序归档。
*   创建和配置 Google Kubernetes 引擎。
*   9 在 Google Kubernetes 引擎中部署应用程序。
*   最后，使用 Github Actions 自动化整个过程。
*   如果你正在创建任何利用机器学习或深度学习算法的项目，那么你应该超越并创建一个 MLOps 管道，而不是停留在模型训练上。我希望本教程能够帮助您理解如何在自己的项目中实现 MLOps。

请务必尝试一下，因为这是部署您的 ML 和 DL 应用程序的最佳方式。

密码

### 你可以在这里找到完整的库。另外，值得一提的是，机器翻译的笔记本取自[官网](https://web.archive.org/web/20221203090558/https://www.tensorflow.org/text/tutorials/nmt_with_attention)。

参考

### [注意力神经机器翻译](https://web.archive.org/web/20221203090558/https://www.tensorflow.org/text/tutorials/nmt_with_attention)

1.  [机器学习操作](https://web.archive.org/web/20221203090558/https://ml-ops.org/)
2.  [如何使用 GitHub 操作构建 MLOps 管道【分步指南】](/web/20221203090558/https://neptune.ai/blog/build-mlops-pipelines-with-github-actions-guide)
3.  [为 MLOps 使用 GitHub 动作](https://web.archive.org/web/20221203090558/https://github.blog/2020-06-17-using-github-actions-for-mlops-data-science/)
4.  [在 Google Kubernetes 引擎上部署机器学习管道](https://web.archive.org/web/20221203090558/https://towardsdatascience.com/deploy-machine-learning-model-on-google-kubernetes-engine-94daac85108b)
5.  [为什么数据漂移检测如此重要，您如何通过 5 个简单的步骤实现自动化](https://web.archive.org/web/20221203090558/https://towardsdatascience.com/why-data-drift-detection-is-important-and-how-do-you-automate-it-in-5-simple-steps-96d611095d93)
6.  [Why data drift detection is important and how do you automate it in 5 simple steps](https://web.archive.org/web/20221203090558/https://towardsdatascience.com/why-data-drift-detection-is-important-and-how-do-you-automate-it-in-5-simple-steps-96d611095d93)