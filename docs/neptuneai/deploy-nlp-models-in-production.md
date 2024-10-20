# 如何在生产中部署 NLP 模型

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/deploy-nlp-models-in-production>

自然语言处理目前是最令人兴奋的领域之一，因为变形金刚和大型语言模型的出现，如 GPT 和伯特已经重新定义了该领域的可能性。然而，博客和大众媒体的大部分关注点是模型本身，而不是非常重要的实际细节，比如如何在生产中部署这些模型。本文试图弥合这一差距，并解释 NLP 模型部署的一些最佳实践。

我们将讨论模型部署过程的许多关键方面，例如:

*   选择模型框架，
*   决定一个 API 后端，
*   使用 Flask 创建微服务，
*   使用 Docker 等工具将模型容器化，
*   监控部署，
*   以及使用 Kubernetes 等工具和 AWS Lambda 等服务来扩展云基础设施。

这样，我们将从头到尾浏览一个部署文本分类模型的小例子，并提供一些关于模型部署最佳实践的想法。

## 模型培训框架与模型部署

您对 NLP 框架的选择将对模型的部署方式产生影响。Sklearn 是支持向量机、朴素贝叶斯或逻辑回归等简单分类模型的流行选择，它与 Python 后端集成得很好。Spacy 还因其一体化的语言处理功能而备受认可，如句子解析、词性标注和命名实体识别。它也是一个 Python 包。

基于深度学习的模型通常是在 Python 的 PyTorch 库中编写的，因为其通过运行定义的自动签名接口是构建模型的理想选择，这些模型可能会创建响应动态输入的计算图，如解析变长句子。许多流行的库也是建立在 PyTorch 之上的，比如 HuggingFace Transformers，这是一个受人尊敬的使用预先训练好的 transformer 模型的工具。显然，Python 生态系统对于 ML 和 NLP 来说是极其流行的；然而，还有其他选择。

预先训练的单词嵌入，如 FastText、GloVe 和 Word2Vec，可以简单地从文本文件中读取，并可以用于任何后端语言和框架。Tensorflow.js 是 Tensorflow 的扩展，允许直接用 Javascript 编写深度学习模型，并使用 Node.js 部署在后端，微软 CNTK 框架可以很容易地集成在。基于. NET 和 C#的后端。类似的机器学习包可以在许多其他语言和框架中找到，尽管它们的质量各不相同。尽管如此，Python 仍然是创建和部署机器学习和 NLP 模型的事实上的标准。

## 后端框架与模型部署

您对后端框架的选择对于成功的模型部署至关重要。虽然语言和框架的任何组合在技术上都可以工作，但是能够使用与您的模型相同的语言开发的后端通常是很好的。这使得将您的模型导入到您的后端系统变得很容易，而不必在不同的交互后端服务或不同系统之间的端口之间服务请求。它还减少了引入错误的机会，并保持后端代码干净，没有混乱和不必要的库。

Python 生态系统中的两个主要后端解决方案是 [Django](https://web.archive.org/web/20221203090307/https://www.djangoproject.com/) 和[Flask](https://web.archive.org/web/20221203090307/https://flask.palletsprojects.com/en/2.1.x/)。推荐使用 Flask 来快速原型化模型微服务，因为它可以很容易地用几行代码建立并运行一个简单的服务器。但是，如果您要构建一个生产系统，Django 的功能更加全面，并且集成了流行的 Django REST 框架，用于构建复杂的 API 驱动的后端。

流行的 NLP 库 **[HuggingFace](https://web.archive.org/web/20221203090307/https://huggingface.co/)** ，也提供了一种通过推理 API 部署模型的简单方法。当您使用 HuggingFace 库构建模型时，您可以训练它并将其上传到他们的模型中心。从那里，他们提供一个可扩展的计算后端，服务于中心托管的模型。只需几行代码，每天花费几美元，任何人都可以部署用 HuggingFace 库构建的安全、可伸缩的 NLP 模型。

另一个伟大的 NLP 专用部署解决方案是 **[Novetta 的 AdaptNLP](https://web.archive.org/web/20221203090307/https://github.com/Novetta/adaptnlp)** :

*   它们为快速原型化和部署 NLP 模型提供了各种易于使用的集成。例如，他们有一系列方法，使用 [FastAI](https://web.archive.org/web/20221203090307/https://www.fast.ai/) 回调和功能集成不同类型 HuggingFace NLP 模型的训练，从而加快部署中的训练和推理。
*   他们还提供现成的 [REST API 微服务](https://web.archive.org/web/20221203090307/https://novetta.github.io/adaptnlp/rest)，打包为 Docker 容器，围绕各种 HuggingFace 模型类型，如问题回答、令牌标记和序列分类。这些 API 拥有成熟的 Swagger UIs，为测试模型提供了一个清晰的界面。

## NLP 模型的实际部署

现在，让我们看看如何使用 Flask 部署逻辑回归文本分类器。我们将训练分类器来预测电子邮件是“垃圾邮件”还是“火腿”。

你可以访问这个 [Kaggle 页面](https://web.archive.org/web/20221203090307/https://www.kaggle.com/datasets/team-ai/spam-text-message-classification?resource=download)并下载数据集。然后，运行下面的命令创建一个 conda 环境来托管本教程的 Python 和库安装。

```py
conda create -n model-deploy python=3.9.7

```

安装完成后，通过运行以下命令激活环境:

```py
conda activate model-deploy

```

然后，通过运行以下命令安装我们需要的库:

```py
pip install Flask scikit-learn

```

在您等待的时候，请继续查看您下载的 csv 数据集。它有一个标题，指定了两个字段,“类别”(这将是我们的标签)和“消息”(这将是我们的模型输入)。

现在，打开代码编辑器，开始输入。首先，我们将建立分类模型。因为这篇文章是关于部署的教程，我们不会遍历所有的模型细节，但是我们在下面提供了它的代码。

进行所需的进口。

```py
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

```

创建所需的功能。

```py
def load_data(fpath):

	cat_map = {
		"ham": 0,
		"spam": 1
	}
	tfidf = TfidfVectorizer()
	msgs, y = [], []
	filein = open(fpath, "r")
	reader = csv.reader(filein)
	for i, line in enumerate(reader):
		if i == 0:

			continue
		cat, msg = line
		y.append(cat_map[cat])
		msg = msg.strip() 
		msgs.append(msg)
	X = tfidf.fit_transform(msgs)
	return X, y, tfidf

def featurize(text, tfidf):
	features = tfidf.transform(text)
	return features

def train(X, y, model):
	model.fit(X, y)
	return model

def predict(X, model):
	return model.predict(X)

clf = LogisticRegression()
X, y, tfidf = load_data('spamorham.csv')
train(X, y, clf)

```

现在，让我们设置 Flask，并为服务微服务的模型创建端点。我们首先要导入 Flask 并创建一个简单的应用程序。

```py
import model
import json

from flask import (
	Flask,
	request
)

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/predict', methods=['POST'])
def predict():
	args = request.json
	X = model.featurize([args['text']], model.tfidf)
	labels = model.predict(X, model.clf).tolist()
	return json.dumps({'predictions': labels})

app.run()

```

正如你所看到的，我们构建了一个 Flask 应用程序，并以“调试”模式运行，这样如果出现任何错误，它都会提醒我们。我们的应用程序有一条用端点“/predict”定义的“路线”。这是一个 POST 端点，它接收一个文本字符串，并将其分类为“垃圾邮件”或“垃圾邮件”。

我们以“request.json”的形式访问 post 参数。只有一个参数“text ”,它指定了我们想要分类的电子邮件消息的文本。为了提高效率，我们也可以重新编写这个程序，一次对多段文本进行分类。如果您愿意，可以尝试添加此功能:)。

预测函数很简单。它接收一封电子邮件，将其转换为 TF-IDF 特征向量，然后运行经过训练的逻辑回归分类器来预测它是垃圾邮件还是 ham。

现在让我们测试一下这个应用程序，以确保它能正常工作！为此，请在命令行中运行以下命令:

```py
python deploy.py

```

这将启动位于 [http://localhost:5000](https://web.archive.org/web/20221203090307/http://localhost:5000/) 上的 Flask 服务器。现在，打开一个单独的 Python 提示符并运行以下代码。

```py
res = requests.post('http://127.0.0.1:5000/predict', json={"text": "You are a winner U have been specially selected 2 receive ¬£1000 or a 4* holiday (flights inc) speak to a live operator 2 claim 0871277810910p/min (18+)"})
```

您可以看到，我们正在向`/predict '端点发出一个 POST 请求，该请求带有一个 json 字段，该字段在参数“text”下指定了电子邮件消息。这显然是一条垃圾短信。让我们看看我们的模型返回什么。要从 API 获得响应，只需运行` res.json()`。您应该会看到以下结果:

**{ '预测':[1]}**

您也可以通过在 POSTMAN 中发送请求来测试您的请求，如下所示:

[![NLP deployment POSTMAN](img/791ddb9254223e936a49617b470a2610.png)](https://web.archive.org/web/20221203090307/https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/NLP-deployment-POSTMAN.png?ssl=1)

*Testing request by sending it to POSTMAN*

您需要做的就是输入您的 URL，将请求类型设置为 POST，并将您的请求的 JSON 放在请求的“Body”字段中。然后，您应该会看到您的预测返回在较低的窗口。

如您所见，该模型返回的预测值为 1，这意味着它将该邮件归类为垃圾邮件。万岁。这就是用 Flask 部署 NLP 模型的基础。

在接下来的小节中，我们将讨论更高级的概念，比如如何扩展部署来处理更大的请求负载。

## 模型部署环境中的容器化

任何模型部署的关键部分是**集装箱化**。像 [Docker](https://web.archive.org/web/20221203090307/https://www.docker.com/) 这样的工具允许你将你的代码打包到一个容器中，这个容器基本上是一个虚拟的运行时，包含系统工具、程序安装、库以及运行你的代码所需的任何东西。

将您的服务容器化使它们更加模块化，并允许它们在任何安装了 Docker 的系统上运行。有了容器，你的代码应该总是不需要任何预配置或者混乱的安装步骤就可以工作。容器还使得使用编排工具(如 Docker-Compose 和 Kubernetes)在许多机器上处理服务的大规模部署变得容易，我们将在本教程的后面部分介绍这些工具。

在这里，我们为我们的文本分类微服务遍历一个 docker 文件。要让这个容器工作，您需要创建一个“requirements.txt”文件，指定运行我们的微服务所需的包。

您可以在终端的该目录下运行该命令来创建它。

```py
pip freeze > requirements.txt

```

我们还需要对我们的 Flask 脚本做一个修改，让它在 Docker 中工作。只需将“app.run()”改为“app.run(host='0.0.0.0 ')”。

现在来看文档。

```py
FROM python:3.9.7-slim

COPY requirements.txt /app/requirements.txt

RUN cd /app &&
	pip install -r requirements.txt

ADD . /app

WORKDIR /app

ENTRYPOINT [“python”, “deploy.py”]

```

让我们试着理解这几行是什么意思。

*   第一行“来自 python:3.9.7-slim ”,指定了容器的*基础映像*。您可以想象我们的映像继承了库、系统配置和其他元素。我们使用的基本映像提供了 Python v3.9.7 的最小安装。

*   下一行将我们的“requirements.txt”文件复制到“/app”目录下的 Docker 映像中。`/app `将存放我们的应用程序文件和相关资源。

*   在下面一行中，我们通过运行命令“pip install -r requirements.txt ”,进入/app 并安装我们需要的 python 库。

*   现在，我们使用“add”将当前构建目录的内容添加到/app 文件夹中。/app `。这将复制我们所有的烧瓶和模型脚本。

*   最后，我们通过运行“WORKDIR /app”将容器的工作目录设置为/app。然后我们指定入口点，这是容器启动时将运行的命令。我们将其设置为运行“python deploy.py ”,这将启动我们的 Flask 服务器。

要构建 docker 映像，请从包含 Docker 文件的目录中运行“Docker build-t spam-or-ham-deploy”。假设一切都正常工作，您应该得到一个构建过程的读数，如下所示:

我们现在还可以在 Docker 桌面中看到我们的容器图像:

接下来，要运行包含 Flask 部署脚本的 Docker 容器，请键入:

```py
docker run -p 5000:5000 -t spam-or-ham-deploy
```

`-p 5000:5000 '标志将容器中的端口 5000 发布到主机中的端口 5000。这使得容器的服务可以从机器上的端口访问。现在容器正在运行，我们可以在 Docker 桌面中查看它的一些统计信息:

我们还可以像以前一样，在 POSTMAN 中再次尝试运行相同的请求:

![NLP deployment POSTMAN ](img/927542174a99becfcc7ad2417f3ad21d.png)

*Testing request by sending it to POSTMAN*

## 云部署

到目前为止，我们的 API 只是被设计来处理适度的请求负载。如果您正在为数百万客户部署大规模服务，您将需要对如何部署模型进行许多调整。

### 库伯内特斯

Kubernetes 是一个跨大型部署编排容器的工具。使用 Kubernetes，您可以毫不费力地在许多机器上部署多个容器，并监控所有这些部署。学习使用 Kubernetes 是扩展到更大规模部署的一项基本技能。

要在本地运行 Kubernetes，你必须[安装 minikube](https://web.archive.org/web/20221203090307/https://minikube.sigs.k8s.io/docs/start/) 。

完成后，在你的终端上运行“minikube start”。下载 Kubernetes 和基本映像需要几分钟时间。您将得到如下所示的读数:

接下来，我们希望通过运行以下命令来创建部署:

```py
kubectl create deployment hello-minikube --image=spam-or-ham-deploy
```

然后，我们希望使用以下方式公开我们的部署:

```py
kubectl expose deployment hello-minikube --type=NodePort --port=8080
```

如果我们运行“ku bectl get services hello-mini kube ”,它将显示一些关于我们服务的有用信息:

然后，我们可以通过运行“minikube service hello-minikube”在浏览器中启动该服务

您还可以通过运行“minikube dashboard”在仪表盘中查看您的服务。

欲了解更多信息，请查看 [Kubernetes 入门文档](https://web.archive.org/web/20221203090307/https://minikube.sigs.k8s.io/docs/start/)。

### 自动气象站λ

如果你喜欢自动化程度更高的解决方案，像 AWS Lambda 这样的弹性推理服务会非常有用。这些是事件驱动的服务，这意味着它们将自动启动和管理计算资源，以响应它们所经历的请求负载。你所需要做的就是定义运行你的模型推理代码的 Lambda 函数，AWS Lambda 会为你处理部署和伸缩过程。

您可以在这里了解更多关于在 AWS [上部署模型的信息。](https://web.archive.org/web/20221203090307/https://pages.awscloud.com/Deploying-Machine-Learning-Models-in-Production_2019_0418-MCL_OD.html)

### 火炬服务

如果您正在使用深度学习 NLP 模型，如 Transformers，PyTorch 的 [TorchServe](https://web.archive.org/web/20221203090307/https://pytorch.org/serve/) 库是扩展和管理 PyTorch 部署的绝佳资源。它有一个 REST API 和一个 gRPC API 来定义远程过程调用。它还包括用于处理日志记录、跟踪指标和监控部署的有用工具。

## NLP 模型部署中的挑战

1.NLP 模型部署的一个关键方面是**确保适当的 MLOps 工作流**。MLOps 工具允许您通过跟踪模型的训练和推理中涉及的步骤来确保模型的可重复性。这包括版本数据、代码、超参数和验证指标。

[MLOps 工具](/web/20221203090307/https://neptune.ai/blog/best-mlops-tools)如 [Neptune.ai](/web/20221203090307/https://neptune.ai/) 、 [MLFlow](https://web.archive.org/web/20221203090307/https://mlflow.org/) 等。为跟踪和记录参数(比如注意系数)和度量(比如 NLP 模型困惑)、代码版本(实际上，任何通过 Git 管理的东西)、模型工件和训练运行提供 API。使用此类工具监控 NLP 模型的训练和部署对于防止模型漂移和确保模型继续准确反映系统中的全部数据至关重要。

2.另一个挑战是 **NLP** **模型可能需要定期重新训练**。例如，考虑在生产中部署的翻译模型的用例。随着企业在不同国家增加更多的客户，它可能希望向模型中添加更多的语言翻译对。在这种情况下，确保添加新的训练数据和重新训练不会降低现有模型的质量非常重要。因此，如上所述，对各种 NLP 指标的持续模型监控非常重要。

3. **NLP 模型可能还需要在生产中进行增量和在线训练**。例如，如果从原始文本中部署一个情感检测模型，您可能会突然获得一些与“讽刺”情感相对应的新数据。

通常，在这种情况下，您不会想要从头重新训练整个模型，尤其是当模型很大的时候。幸运的是，有许多算法和库可以用来在生产中部署流式 NLP 模型。例如，scikit-multiflow 实现了分类算法，如 [Hoeffding Trees](https://web.archive.org/web/20221203090307/https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.trees.HoeffdingTreeClassifier.html) ，这些算法被设计为在次线性时间内进行增量训练。

## 结论

在部署 NLP 模型时，必须考虑许多因素，如部署的规模、部署的 NLP 模型的类型、推理延迟和服务器负载等。

部署 NLP 模型的最佳实践包括使用 Django 或 Flask 等 Python 后端，使用 Docker 进行容器化，使用 MLFlow 或 Kubeflow 进行 MLOps 管理，以及使用 AWS Lambda 或 Kubernetes 等服务进行扩展。

对于那些不想自己处理大规模部署的人来说，有一些易于使用的付费服务，如 HuggingFace 的推理 API，可以为您处理部署。虽然需要一些时间来加快如何优化部署 NLP 模型，但这是一项非常值得的投资，因为它确保您可以将您的模型提供给世界其他地方！

### 参考