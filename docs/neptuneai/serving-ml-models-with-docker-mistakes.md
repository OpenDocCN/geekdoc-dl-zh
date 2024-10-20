# 用 Docker 服务机器学习模型:你应该避免的 5 个错误

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/serving-ml-models-with-docker-mistakes>

正如您已经知道的， [Docker](https://web.archive.org/web/20221201155629/https://www.docker.com/) 是一个工具，它允许您使用容器来创建和部署隔离的环境，以便运行您的应用程序及其依赖项。既然这样，在进入主题之前，让我们简单回顾一下 Docker 的一些基本概念。

## 数据科学家为什么要容器化 ML 模型？

你是否曾经训练过一个机器学习模型，然后决定与同事分享你的代码，但后来发现你的代码不断出错，尽管它在你的笔记本电脑上工作得很好。大多数情况下，这可能是包兼容性问题或环境问题。解决这个问题的好办法是使用 **[容器](/web/20221201155629/https://neptune.ai/blog/data-science-machine-learning-in-containers)** 。

![Why should data scientists containerize ML models?](img/fcc904ac814a1162451969358e9a81a3.png)

*Source: Author*

集装箱优惠:

*   **再现性***——通过将你的机器学习模型容器化，你可以将你的代码运送到任何其他安装了 Docker 的系统，并期望你的应用程序能给你类似于你在本地测试时的结果。*

 **   **协作开发*–***容器化的机器学习模型允许团队成员协作，这也使得版本控制更加容易。

## 使用 Docker 服务于您的机器学习模型

既然你知道为什么你需要容器化你的机器学习模型，接下来的事情就是理解你如何容器化你的模型。

一些您可能已经知道并在本文中遇到的与 Docker 相关的术语:

*   Dockerfile :你可以把 Dockerfile 想象成一个描述你想要如何设置你想要运行的系统的操作系统安装的文件。它包含了设置 Docker 容器所需的所有代码，从下载 Docker 映像到设置环境。

*   **Docker image** :它是一个只读模板，包含创建 Docker 容器的指令列表。

*   **Docker 容器**:容器是 Docker 映像的一个可运行实例。

![Basic Docker commands](img/0b119b7748c38a1c3139c5a938b24111.png)

*Basic Docker commands | Source: Author*

创建 Docker 文件时，可以考虑一些最佳实践，比如在构建 Docker 映像时避免安装不必要的库或包，减少 Docker 文件的层数等等。查看以下文章，了解使用 Docker 的最佳实践。

## 如何为机器学习模型服务？

模型服务的重要概念是托管机器学习模型(内部或云中)，并通过 API 提供其功能，以便公司可以将人工智能集成到他们的系统中。

通常有两种模型服务:批处理和在线。

**批量预测**表示模型的输入是大量的数据，通常是预定的操作，预测可以以表格的形式发布。

**在线部署**需要部署带有端点的模型，以便应用程序可以向模型提交请求，并以最小的延迟获得快速响应。

### 服务 ML 模型时需要考虑的重要要求

#### 交通管理

根据目标服务的不同，端点上的请求会采用不同的路径。为了同时处理请求，流量管理还可以部署负载平衡功能。

#### 监视

监控在生产中部署的机器学习模型是很重要的。通过监控最大似然模型，我们可以检测模型的性能何时恶化以及何时重新训练模型。没有模型监控，机器学习生命周期是不完整的。

#### 数据预处理

对于实时服务，机器学习模型要求模型的输入具有合适的格式。应该有一个专用的转换服务用于数据预处理。

您可以使用不同的工具来为生产中的机器学习模型提供服务。你可以查看[这篇](/web/20221201155629/https://neptune.ai/blog/ml-model-serving-best-tools)文章，获得关于你可以用于模型服务的不同机器学习工具/平台的全面指导。

## 使用 Docker 服务机器学习模型时应该避免的错误

现在您已经理解了模型服务的含义以及如何使用 Docker 来服务您的模型。在使用 Docker 为您的机器学习模型提供服务时，知道做什么和不做什么是很重要的。

操作错误是数据科学家在使用 Docker 部署他们的机器学习模型时最常见的错误。这种错误通常会导致应用程序的 ML 服务性能很差。一个 ML 应用程序是通过它的整体服务性能来衡量的——它应该具有低推理延迟、低服务延迟和良好的监控架构。

### 错误一:用 TensorFlow Serving 和 Docker 服务机器学习模型时使用 REST API 而不是 gRPC

TensorFlow 服务是由 Google 开发人员开发的，它提供了一种更简单的方法来部署您的算法和运行实验。

要了解更多关于如何使用 TensorFlow 服务 Docker 来服务您的 ML 模型，请查看这篇[帖子](https://web.archive.org/web/20221201155629/https://neptune.ai/blog/how-to-serve-machine-learning-models-with-tensorflow-serving-and-docker)。

当使用 TensorFlow 服务为机器学习模型提供服务时，您需要了解 Tensorflow 服务提供的不同类型的端点以及何时使用它们。

#### gRPC 和 REST API 端点

**gRPC**

是由谷歌发明的一种通讯协议。它使用一个协议缓冲区作为它的消息格式，它是高度打包的，对于序列化结构化数据是高效的。借助对负载平衡、跟踪、运行状况检查和身份验证的可插拔支持，它可以高效地连接数据中心内部和数据中心之间的服务。

**休息**

大多数 web 应用程序使用 REST 作为通信协议。它说明了客户端如何与 web 服务通信。尽管 REST 仍然是客户机和服务器之间交换数据的好方法，但它也有缺点，那就是速度和可伸缩性。

#### gRPC 和 REST API 的区别

gRPC 和 REST API 在操作方式上有不同的特点。下表比较了两种 API 的不同特征

| 特性 | gRPC | 休息 |
| --- | --- | --- |
|  |  |  |
|  | 

【协议缓冲区】

 |  |
|  |  |  |

如下图所示，大多数服务 API 请求都是使用 REST 到达的。在使用[RESTful API](https://web.archive.org/web/20221201155629/https://www.tensorflow.org/tfx/serving/api_rest)或[gRPC API](https://web.archive.org/web/20221201155629/https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto)进行预测，将预处理数据发送到 Tensorflow 服务器之前，预处理和后处理步骤在 API 内部进行。

![How to use gRPC for model serving ](img/7974752a343d83fc3f22a3fda4a37a6e.png)

*How to use gRPC for model serving | Source: Author*

大多数数据科学家经常利用 **REST API** 进行模型服务，然而，它也有缺点。主要是速度和可伸缩性。你的模型在被输入后做出预测所花费的时间被称为 ML 推理延迟。为了改善应用程序的用户体验，ML 服务快速返回预测是非常重要的。

对于较小的有效载荷，这两种 API 都可以产生类似的性能，同时 [AWS Sagemaker](https://web.archive.org/web/20221201155629/https://aws.amazon.com/blogs/machine-learning/reduce-compuer-vision-inference-latency-using-grpc-with-tensorflow-serving-on-amazon-sagemaker/) 证明，对于图像分类和对象检测等计算机视觉任务，在 Docker 端点中使用 gRPC 可以将整体延迟减少 75%或更多。

#### 使用 gRPC API 和 Docker 部署您的机器学习模型

**步骤 1:** 确保您的电脑上安装了 Docker

**步骤 2:** 要使用 Tensorflow 服务，您需要从容器存储库中提取 Tensorflow 服务图像。

```py
docker pull tensorflow/serving

```

第三步:建立并训练一个简单的模型

```py
import matplotlib.pyplot as plt
import time
from numpy import asarray
from numpy import unique
from numpy import argmax
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

(x_train, y_train), (x_test, y_test) = load_data()
print(f'Train: X={x_train.shape}, y={y_train.shape}')
print(f'Test: X={x_test.shape}, y={y_test.shape}')

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

input_shape = x_train.shape[1:]

n_classes = len(unique(y_train))

model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=input_shape))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy: %.3f' % acc)
```

**第四步:**保存模型

保存 TensorFlow 模型时，可以将其保存为协议缓冲文件，通过在 save_format 参数中传递“tf”将模型保存到协议缓冲文件中。

```py
file_path = f"./img_classifier/{ts}/"
model.save(filepath=file_path, save_format='tf')
```

可以使用 ***saved_model_cli*** 命令对保存的模型进行调查。

```py
!saved_model_cli show --dir {export_path} --all
```

**步骤 5** :使用 gRPC 服务模型

您需要安装 gRPC 库。

```py
Import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorboard.compat.proto import types_pb2
```

您需要使用端口 8500 在客户端和服务器之间建立一个通道。

```py
channel = grpc.insecure_channel('127.0.0.1:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
```

服务器的请求有效负载需要通过指定模型的名称、存储模型的路径、预期的数据类型以及数据中的记录数来设置为协议缓冲区。

```py
request = predict_pb2.PredictRequest()
request.model_spec.name = 'mnist-model'
request.inputs['flatten_input'].CopyFrom(tf.make_tensor_proto(X_test[0],dtype=types_pb2.DT_FLOAT,  shape=[28,28,1]))
```

最后，要用 Docker 部署您的模型，您需要运行 Docker 容器。

```py
docker run -p 8500:8500 --mount type=bind,source=<absolute_path>,target=/models/mnist-model/ -e MODEL_NAME=mnist -t tensorflow/serving
```

现在服务器可以接受客户端请求了。从存根调用 Predict 方法来预测请求的结果。

```py
stub.Predict(request, 10.0)
```

按照上述步骤，您将能够使用 gRPC API 为 TensorFlow 服务模型提供服务。

### 错误 2:在使用 Docker 为机器学习模型提供服务时，对数据进行预处理

开发人员在使用 Docker 服务于他们的机器学习模型时犯的另一个错误是在做出预测之前实时预处理他们的数据。在 ML 模型提供预测之前，它期望数据点必须包括在训练算法时使用的所有输入特征。

例如，如果您训练一个线性回归算法来根据房子的大小、位置、年龄、房间数量和朝向来估计房子的价格，则训练好的模型将需要这些要素的值作为推断过程中的输入，以便提供估计的价格。

在大多数情况下，需要对输入数据进行预处理和清理，甚至需要对某些要素进行工程设计。现在想象一下，每次触发模型端点时都要实时地做这件事，这意味着对一些特性进行重复的预处理，尤其是静态特性和高 ML 模型延迟。在这种情况下，**特性存储库**被证明是一个无价的资源。

#### 什么是功能商店？

功能存储与存储相关，用于跨多个管道分支存储和服务功能，从而实现共享计算和优化。

#### 在 Docker 中为 ml 模型提供服务时使用特征库的重要性

*   数据科学家可以使用要素存储来简化要素的维护方式，为更高效的流程铺平道路，同时确保要素得到正确存储、记录和测试。

*   在整个公司的许多项目和研究任务中都使用了相同的功能。数据科学家可以使用要素存储来快速访问他们需要的要素，并避免重复工作。

为机器学习模型提供服务时，为了调用模型进行预测，会实时获取两种类型的输入要素:

1.  **静态参考**:这些特征值是需要预测的实体的静态或渐变属性。这包括描述性属性，如客户人口统计信息。它还包括客户的购买行为，如他们花了多少钱，多久消费一次等。

2.  **实时动态特性:**这些特性值是基于实时事件动态捕获和计算的。这些特征通常是在事件流处理管道中实时计算的。

要素服务 API 使要素数据可用于生产中的模型。创建服务 API 时考虑到了对最新特性值的低延迟访问。要更好地理解特性存储，了解可用的不同特性存储，请查看本文:[特性存储:数据科学工厂的组件](/web/20221201155629/https://neptune.ai/blog/feature-stores-components-of-a-data-science-factory-guide)。

### 错误 3:使用 IP 地址在 Docker 容器之间通信

最后，您已经使用 Docker 部署了您的机器学习模型，并且您的应用程序正在生产环境中返回预测，但是由于某些原因，您需要对容器进行更新。在进行必要的更改并重启容器化的应用程序后，您会不断地得到 ***“错误:连接失败”。***

您的应用程序无法建立到数据库的连接，即使它以前工作得非常好。每个容器都有自己的内部 IP 地址，该地址在容器重新启动时会发生变化。数据科学家犯的错误是使用 Docker 的默认网络驱动程序 bridge 在容器之间进行通信。同一桥接网络中的所有容器可以通过 IP 地址相互通信。因为 IP 地址会波动，这显然不是最好的方法。

#### 不使用 IP 地址，如何在 Docker 容器之间进行通信？

为了与容器通信，您应该使用环境变量来传递主机名，而不是 IP 地址。您可以通过创建用户定义的桥接网络来实现这一点。

![How to create a user-defined bridge network](img/03c19540138fbd1213e6be77953ec989.png)

*How to create a user-defined bridge network | [Source](https://web.archive.org/web/20221201155629/https://www.tutorialworks.com/container-networking/)*

1.  您需要创建自己的自定义桥接网络。您可以通过运行 Docker network create 命令来实现这一点。这里我们创建一个名为“虚拟网络”的网络。

```py
Docker network create dummy-network
```

2.  用 ***docker run*** 命令正常运行你的容器。使用***—网络选项*** 将其添加到您自定义的桥接网络中。您还可以使用–name 选项添加别名。

```py
docker run --rm --net dummy-network --name tulipnginx -d nginx
```

3.  将另一个容器连接到您创建的自定义桥接网络。

```py
docker run --net dummy-network -it busybox 
```

4.  现在，您可以使用容器主机名连接到任何容器，只要它们在同一个自定义桥接网络上，而不用担心重启。

### 错误 4:作为根用户运行您的流程

许多数据科学家在作为根用户运行他们的流程时犯了这样的错误，我将解释为什么这是错误的，并推荐解决方案。在设计系统时，坚持最小特权原则是很重要的。这意味着应用程序应该只能访问完成任务所需的资源。授予进程执行所需的最少特权是保护自己免受任何意外入侵的最佳策略之一。

因为大多数容器化的流程是应用程序服务，所以它们不需要 root 访问。容器不需要 root 才能运行，但是 Docker 需要。编写良好、安全且可重用的 Docker 映像不应该以 root 用户身份运行，而应该提供一种可预测且简单的方法来限制访问。

默认情况下当你运行你的容器时，它假定 ***根*** 用户。我也犯过这样的错误，总是以 root 用户身份运行我的进程，或者总是使用 sudo 来完成工作。但是我了解到，拥有不必要的权限会导致灾难性的问题。

让我通过一个例子来说明这一点。这是我过去用于一个项目的 docker 文件样本。

```py
FROM tiangolo/uvicorn-gunicorn:python3.9

RUN mkdir /fastapi

WORKDIR /fastapi

COPY requirements.txt /fastapi

RUN pip install -r /fastapi/requirements.txt

COPY . /fastapi

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

第一件事是构建一个 Docker 映像并运行 Docker 容器，您可以用这个命令来完成

```py
docker build -t getting-started .
```

```py
docker run -d p 8000:8000 getting-started
```

接下来是获取 containerID，你可以通过用 ***docker ps*** 检查你的 Docker 容器进程来做到这一点，然后你可以运行 ***whoami*** 命令来查看哪个用户可以访问容器。

![Running your processes as root users](img/41104fe3949d2ad17b66292a7d2ae124.png)

*Source: Author*

如果应用程序存在漏洞，攻击者就可以获得容器的超级用户访问权限。用户在容器中拥有 root 权限，可以做任何他们想做的事情。攻击者不仅可以利用这一点来干扰程序，还可以安装额外的工具来转到其他设备或容器。

#### 如何以非根用户身份运行 Docker

使用 dockerfile 文件:

```py

FROM debian:stretch

RUN useradd -u 1099 user-tesla

USER user-tesla
```

作为一个容器用户，对改变用户的支持程度取决于容器维护者。使用-user 参数，Docker 允许您更改用户(或 docker-compose.yml 中的用户密钥)。应将进程更改到的用户的用户 id 作为参数提供。这限制了任何不必要的访问。

### 错误 5:用 Docker 服务 ML 模型时没有监控模型版本

数据科学家犯的一个操作错误是，在将 ML 系统部署到生产环境之前，没有跟踪对其进行的更改或更新。模型版本化帮助 ML 工程师了解模型中发生了什么变化，研究人员更新了哪些特性，以及特性是如何变化的。了解进行了哪些更改，以及在集成多个功能时，这些更改如何影响部署的速度和简易性。

#### 模型版本化的优势

模型版本控制有助于跟踪您先前已经部署到生产环境中的不同模型文件，通过这样做，您可以实现:

1.  **模型谱系可追溯性:**如果最近部署的模型在生产中表现不佳，您可以重新部署表现更好的模型的先前版本。

2.  **模型注册表:**像 [Neptune AI](/web/20221201155629/https://neptune.ai/) 和 [MLFlow](https://web.archive.org/web/20221201155629/https://mlflow.org/) 这样的工具可以作为模型注册表，方便你记录它们的模型文件。每当您需要服务的模型时，您可以获取模型和特定的版本。

#### 使用 Neptune.ai 进行模型版本控制并使用 Docker 进行部署

Neptune.ai 允许您[跟踪您的实验](https://web.archive.org/web/20221201155629/https://docs.neptune.ai/how-to-guides/experiment-tracking)，超参数值，用于特定实验运行的数据集，以及模型工件。Neptune.ai 提供了一个 python SDK，您可以在构建机器学习模型时使用它。

第一步是确保您已经安装了 [neptune python 客户端](https://web.archive.org/web/20221201155629/https://docs.neptune.ai/integrations-and-supported-tools/languages/neptune-client-python)。根据您的操作系统，打开您的终端并运行以下命令:

```py
pip install neptune-client

```

在训练好你的模型之后，[你可以在 Neptune](https://web.archive.org/web/20221201155629/https://docs.neptune.ai/how-to-guides/model-registry/registering-a-model) 中注册它来追踪任何相关的元数据。首先，需要初始化一个 Neptune 模型对象。模型对象适用于保存在训练过程中由所有模型版本共享的通用元数据。

```py
import neptune.new as neptune
model = neptune.init_model(project='<project name>’',
    name="<MODEL_NAME>",
    key="<MODEL>",
    api_token="<token>"
)
```

这将生成一个到 Neptune 仪表板的 URL，在这里您可以看到您已经创建的不同模型。查看[工作区](https://web.archive.org/web/20221201155629/https://app.neptune.ai/akinwande/docker-demo/models)。

[![How to create a model version in neptune.ai](img/b7be3f3e4e593e7d4abcc173b6426621.png)](https://web.archive.org/web/20221201155629/https://neptune.ai/serving-machine-learning-models-in-docker-5-mistakes-you-should-avoid7)

*ML model logged in Neptune.ai | [Source](https://web.archive.org/web/20221201155629/https://app.neptune.ai/akinwande/docker-demo/models)*

为了[在 Neptune](https://web.archive.org/web/20221201155629/https://docs.neptune.ai/how-to-guides/model-registry/creating-model-versions) 中创建模型版本，您需要在同一个 Neptune 项目中注册您的模型，并且您可以在仪表板上的 models 选项卡下找到您的模型。

要在 Neptune 上创建模型版本，您需要运行以下命令:

```py
import neptune.new as neptune
model_version = neptune.init_model_version(
    model="MODEL_ID",
)

```

下一件事是存储任何相关的模型元数据和工件，您可以通过将它们分配给您创建的模型对象来完成。要了解如何记录模型元数据，请查看这个[文档页面](https://web.archive.org/web/20221201155629/https://docs.neptune.ai/how-to-guides/experiment-tracking/log-model-building-metadata)。

[![How to create a model version in neptune.ai](img/2965ca9147c4793881fbb09bd12adb01.png)](https://web.archive.org/web/20221201155629/https://neptune.ai/serving-machine-learning-models-in-docker-5-mistakes-you-should-avoid1)

*Different versions of the model are visible in the Neptune’s UI | [Source](https://web.archive.org/web/20221201155629/https://app.neptune.ai/akinwande/docker-demo/m/DOC-MODEL/versions)*

现在您可以看到您已经创建的模型的不同版本，每个模型版本的相关元数据，以及模型度量。您还可以管理每个模型版本的模型阶段。从上图来看， **DOC-MODEL-1** 已经部署到生产中。这样，您可以看到当前部署到生产环境中的模型版本以及该模型的相关元数据。

在构建您的机器学习模型时，您不应该将关联的元数据(如超参数、注释和配置数据)作为文件存储在 Docker 容器中。当容器被停止、销毁和替换时，您可能会丢失容器中的所有相关数据。使用 Neptune-client，您可以记录和存储每次运行的所有相关元数据。

#### 在 Docker 与 Neptune 一起服务时如何监控模型版本

因为 Neptune 通过创建模型、创建模型版本和管理模型阶段转换来管理您的数据，所以您可以使用 Neptune 作为[模型注册表](https://web.archive.org/web/20221201155629/https://docs.neptune.ai/how-to-guides/model-registry)来查询和下载您存储的模型。

创建一个新脚本来提供和导入必要的依赖项。您所需要做的就是指定您需要在生产中使用的模型版本。您可以通过将您的 [NEPTUNE_API_TOKEN](https://web.archive.org/web/20221201155629/https://docs.neptune.ai/api-reference/environment-variables#neptune_api_token) 和您的 [MODEL_VERSION](https://web.archive.org/web/20221201155629/https://app.neptune.ai/common/showcase-model-registry/m/SHOW2-MOD21/v/SHOW2-MOD21-13/metadata) 作为 Docker 环境变量来运行您的 Docker 容器:

```py
import neptune.new as neptune
import pickle,requests

api_token = os.environ['NEPTUNE_API_TOKEN']
model_version = os.environ['MODEL_VERSION']

def load_pickle(fp):

   """
   Load pickle file(data, model or pipeline object).
   Parameters:
       fp: the file path of the pickle files.

   Returns:
       Loaded pickle file
   """
   with open(fp, 'rb') as f:
       return pickle.load(f)

def predict(data):

   input_data = requests.get(data)

   model_version = neptune.init_model_version(project='docker-demo',
   version=model_version,
   api_token=api_token
   )
   model_version['classifier']['pickled_model'].download()
   model = load_pickle('xgb-model.pkl')
   predictions = model.predict(data)
   return predictions
```

通过创建 Docker 文件并提供 requirements.txt 文件上的依赖项列表，可以使用 Docker 将机器学习模型服务容器化。

```py
neptune-client
sklearn==1.0.2
```

```py
FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get -y install gcc

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
CMD [ "python3", "-W ignore" ,"src/serving.py"]
```

要从上面的 Docker 文件构建 Docker 映像，您需要运行以下命令:

```py
docker build --tag <image-name> . 

docker run -e NEPTUNE_API_TOKEN="<YOUR_API_TOKEN>"  -e MODEL_VERSION =”<YOUR_MODEL_VERSION>” <image-name>

```

在 Docker 容器上管理数据有几种替代方法，您可以在开发期间[绑定挂载目录](https://web.archive.org/web/20221201155629/https://docs.docker.com/storage/bind-mounts/)。这是调试代码的一个很好的选择。您可以通过运行以下命令来实现这一点:

```py
docker run -it <image-name>:<image-version> -v /home/<user>/my_code:/code

```

现在，您可以同时调试和执行容器中的代码，所做的更改将反映在主机上。[这让我们回到了在容器中使用相同的主机用户 ID 和组 ID 的优势](https://web.archive.org/web/20221201155629/https://faun.pub/set-current-host-user-for-docker-container-4e521cef9ffc)。您所做的所有修改都将显示为来自主机用户。

要启动 Docker 容器，您需要运行以下命令:

```py
docker run -d -e NEPTUNE_API_TOKEN="<YOUR_API_TOKEN>"  -e MODEL_VERSION =”<YOUR_MODEL_VERSION>” <image-name>

```

***-d*** 选项指定容器应该以[守护模式](https://web.archive.org/web/20221201155629/https://docs.docker.com/get-started/overview/#the-docker-daemon)启动。

## 最后的想法

可再现性和协作开发是数据科学家应该用 Docker 容器部署他们的模型的最重要的原因。 [TensorFlow serving](https://web.archive.org/web/20221201155629/https://www.tensorflow.org/tfx/guide/serving) 是流行的模型服务工具之一，您可以扩展它来服务其他类型的模型和数据。此外，当使用 TensorFlow 服务机器学习模型时，您需要了解不同的[客户端 API](https://web.archive.org/web/20221201155629/https://www.tensorflow.org/tfx/serving/api_rest)，并选择最适合您的用例。

Docker 是在生产中部署和服务模型的好工具。尽管如此，找出许多数据科学家犯的错误并避免犯类似的错误是至关重要的。

数据科学家在用 Docker 服务机器学习模型时犯的错误围绕着模型延迟、应用程序安全和监控。模型延迟和[模型管理](/web/20221201155629/https://neptune.ai/blog/machine-learning-model-management)是你的 ML 系统的重要部分。一个好的 ML 应用程序应该在收到请求时返回预测。通过避免这些错误，您应该能够使用 Docker 有效地部署一个工作的 ML 系统。

### 参考

1.  [Docker 数据科学家最佳实践](https://web.archive.org/web/20221201155629/https://towardsdatascience.com/docker-best-practices-for-data-scientists-2ed7f6876dff)
2.  [使用 gRPC 和亚马逊 SageMaker 上的 TensorFlow 服务减少计算机视觉推理延迟](https://web.archive.org/web/20221201155629/https://aws.amazon.com/blogs/machine-learning/reduce-compuer-vision-inference-latency-using-grpc-with-tensorflow-serving-on-amazon-sagemaker/)
3.  [使用 docker 编写部署 ml flow](https://web.archive.org/web/20221201155629/https://towardsdatascience.com/deploy-mlflow-with-docker-compose-8059f16b6039)
4.  [ML 服务水平性能监控的两大要素](https://web.archive.org/web/20221201155629/https://towardsdatascience.com/two-essentials-for-ml-service-level-performance-monitoring-2637bdabc0d2)
5.  [最小化机器学习中的实时预测服务延迟](https://web.archive.org/web/20221201155629/https://cloud.google.com/architecture/minimizing-predictive-serving-latency-in-machine-learning#precomputing_and_caching_predictions)
6.  [如何使用 gRPC API 服务于深度学习模型？](https://web.archive.org/web/20221201155629/https://towardsdatascience.com/serving-deep-learning-model-in-production-using-fast-and-efficient-grpc-6dfe94bf9234)
7.  [将机器学习模型构建到 Docker 图像中](https://web.archive.org/web/20221201155629/https://docs.google.com/presentation/d/1yPxocBvpAM2dqdILfZtEwd13znC7l7ymx_y6BAD3Pg0/edit#slide=id.ge87238424d_0_47)*