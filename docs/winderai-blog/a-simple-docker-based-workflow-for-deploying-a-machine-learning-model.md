# 用于部署机器学习模型的简单的基于 Docker 的工作流

> 原文：<https://winder.ai/a-simple-docker-based-workflow-for-deploying-a-machine-learning-model/>

在软件工程中，Phil Karlton 的名言，由 Martin Fowler 引申为:“在计算机科学中有两件困难的事情:缓存失效、命名和一个接一个的错误。”在[数据科学](https://winder.ai/services/data-science/)中，有一个硬东西凌驾于所有其他硬东西之上:部署。

新冠肺炎研究

以下是本系列中其他新冠肺炎文章的链接列表。

*   [新冠肺炎回应:雅典娜项目和贝叶斯分析简介](https://winder.ai/covid-19-response-athena-project-and-an-introduction-bayesian-analysis/)
*   [新冠肺炎指数贝叶斯模型](https://winder.ai/covid-19-exponential-bayesian-model/)
*   [新冠肺炎指数贝叶斯模型回测](https://winder.ai/covid-19-exponential-bayesian-model-backtesting/)
*   [新冠肺炎逻辑斯谛贝叶斯模型](https://winder.ai/covid-19-logistic-bayesian-model/)
*   [新冠肺炎分层贝叶斯逻辑模型与 pymc3](https://winder.ai/covid-19-hierarchical-bayesian-logistic-model-with-pymc3/)

## 为什么部署数据科学模型如此困难？

首先，模型的开发与实现是如此的分离，以至于你可以用另一种语言来编写它。我指的是字面意思。我见过许多公司用一种语言开发，比如 Matlab，用另一种语言部署，比如 Java。这增加了如此多的障碍，以至于我可以写一整篇博文。

第二个问题是，如果你要从头开始创建一个管道来构建、训练和部署一个模型，这实际上是相当复杂的。数据的领域和技术的选择就像洗衣机里的一滴食用色素，导致碎片化，增加维护负担和粉色衬衫。

将此归纳为三重抱怨:意见。我不喜欢观点，除非是我的。我当然是在开玩笑。但是每种语言，每种框架，所有的编排者都有一个关于如何部署的观点。他们都有正确的目标，但是在这个过程中他们增加了一个过多的限制。就像它必须 24/7 连接到某个主服务，或者我必须使用一个特殊的库或容器。不要！我有我需要的所有抽象，非常感谢。

我见过并使用了一系列专有和开源的“解决方案”，所有这些*都告诉了*我应该如何部署。多么粗鲁！事不宜迟，让我*告诉你*你应该如何部署*。🙃*

## 指导原则

在我深入研究之前，我想先阐明我的目标。我也有看法，欢迎你忽略，但我觉得有道理。目标是产生一个简单的构建-训练-部署工作流，它可以在任何 CI/CD 工具中工作。此外，我希望结果是:

*   使用 docker 容器来构建/服务容器可以运行的任何地方
*   公开一个 REST API，以便其他人可以使用它
*   分离训练和服务，以最小化操作容器的复杂性和攻击面

结果是一个基于 Docker 的工作流，任何软件工程师都会很熟悉。

## 码头集装箱内的培训

我大部分数据科学工作(T2)都是用 Python 完成的。所以第一项工作是把我在 Jupyter 笔记本上所做的一切记录下来，并创建一个`train.py`文件。是的，我知道现在有很多直接从 Jupyter Notebook 训练模型的选择。但我认为它们增加了不必要的复杂性。一旦我完成了开发，我通常很长时间都不会回到笔记本上。Python 文件成为事实的来源。这样，我可以更像对待一个“正常的”软件工程项目。

我不会用大量的代码清单来浪费你的时间。但是如果你确实想看一些实际的代码，[看看我为](https://gitlab.com/WinderAI/covid/model_deaths)[雅典娜项目](https://project-athena-covid-19.web.app)开发的[新冠肺炎模型](https://gitlab.com/WinderAI/covid/)中的一个。总之，看起来[像这样](https://gitlab.com/WinderAI/covid/model_deaths/-/blob/master/app/train.py):

1.  加载和解析数据
2.  清理和准备数据
3.  火车模型
4.  将模型参数或结果保存到文件中

简单吧？

接下来，您需要一个`Dockerfile`来运行训练脚本。可以想象，这非常简单，就像:

```py
FROM base AS train  COPY requirements.txt .  RUN pip install -r requirements.txt  COPY ./app/train.py /app/train.py  RUN python3 /app/train.py 
```

其中`base`是适合您项目的容器。

为了训练你的模型，你需要做的就是运行`docker build -t myimage .`并等待它完成，然后你就有了一个包含训练好的参数/结果的容器。

## 创建服务容器

下一个任务是构建一个容器来服务您的模型。首先，创建一个`main.py`文件，它负责([这里的例子是](https://gitlab.com/WinderAI/covid/model_deaths/-/blob/master/app/main.py)):

1.  加载保存的模型参数/结果
2.  实例化 REST API
3.  定义路线和服务模型

步骤 1 是你保存模型的相反步骤。第 2 步和第 3 步取决于您想用什么来服务您 API。最近我发现了 [FastAPI](https://fastapi.tiangolo.com/) 这是一个真正干净的、包含电池的 REST 框架。它包括 OpenAPI 和 Swagger docs 开箱即用，没有额外的工作，包括健壮但简单的数据规范工具，它的语法非常紧凑，我不必考虑 web 服务，因为它与 gunicorn 一起提供。最棒的是，有[一个码头集装箱](https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker)整装待发！

你可以把这个换成[瓶或者别的什么](https://github.com/tiangolo/meinheld-gunicorn-flask-docker)。我不会介意的。

接下来，我在我们的`Dockerfile`中定义了另一个容器，我用它来提供服务([同样的例子还有](https://gitlab.com/WinderAI/covid/model_deaths/-/blob/master/Dockerfile)):

```py
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7 as base    FROM base AS train  # ... Rest of training dockerfile    # Now for the serving container  FROM base  COPY requirements.txt .  RUN pip install -r requirements.txt  COPY --from=train predictions.pkl .  COPY main.py . 
```

这是一个[多级 Dockerfile](https://docs.docker.com/develop/develop-images/multistage-build/) ，它包含了我们的培训和服务容器。服务容器从训练构建中复制训练工件，并将它们粘贴到`main.py`文件所期望的位置。

## 构建和部署您的容器

构建就像做一个`docker build -t myimage .`一样简单。这应该可以启动培训，将结果复制到您的服务容器中，并生成一个没有培训所需的所有无用内容的服务容器。你可以把它添加到你的 CI/CD 管道， [Gitlab 的 AutoDevOps](https://docs.gitlab.com/ee/topics/autodevops/) 管道，或者，我敢说，在本地构建它，然后手动把它推送到一个容器注册表。

部署你的容器完全依赖于你的技术。它可以像一个`docker run -d -p 8080:80 myimage`一样简单。它就在那里，在`http://localhost:8080/docs`上提供您的 API 文档。或者你可以在 Kubernetes[kna tive manifest](https://knative.dev/)中使用这个容器来完成无服务器的机器学习部署。

不管怎样，我希望您同意这极大地简化了部署。你没有理由不公布你的模型！

## 扩展和/或未来想法

我的目标是简单，所以我忽略了一些你需要注意的事情:

*   如果你对训练有很高的计算需求，例如，如果你正在训练一个深度学习模型，那么你需要考虑你在哪里进行训练。因为一个免费的 CI/CD 服务要么 a)不够强大，要么 b)你会很快用完免费的运行时间。
*   当自动化机器学习构建时，测试尤其重要。为了简化构建过程，我跳过了对它的任何提及。但是请记住。如果您正在构建一个供一个人每月使用一次的模型，那么简单的随叫随到模型可能更容易。
*   创建一个弹性部署并不比我在这里所做的难多少，因为容器是不可变的。请随意跨区域/可用性区域扩展容器，以符合您的 SLA。
*   添加监控/跟踪/警报并不难，但需要根据您的堆栈进行相应的工作。我建议构建一个新的基本容器，其中包含一些自动中间件来添加所有这些东西，这样其他人就可以免费获得它。或者你可以建立一个自动添加它的库！
*   注意你的 REST APIs。仔细阅读构建优秀的 API，这样你就不会淹没在自己的 URL 中。