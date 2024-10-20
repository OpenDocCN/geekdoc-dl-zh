# 构建 ML 管道:6 个问题和解决方案[来自数据科学家的经验]

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/building-ml-pipeline-problems-solutions>

ML 工作以 jupyter 笔记本开始和结束的时代已经过去很久了。

由于所有公司都希望将他们的模型部署到生产中，因此拥有一个高效和严格的 MLOps 管道来实现这一点是当今 ML 工程师必须面对的真正挑战。

但是考虑到 MLOps 工具是多么新，创建这样一个管道并不是一件容易的事情。事实上，对于绝大多数中型公司来说，这个领域本身也不过几年的历史。因此，创建这样的管道只能通过反复试验来完成，并且需要掌握大量的工具/库。

在本文中，我将向您介绍

*   我在之前工作过的公司里看到的常见陷阱，
*   以及我是如何解决这些问题的。

然而，这绝不是故事的结尾，我相信从现在起两年后，MLOps 领域将会更加成熟。但是通过向你展示我所面临的挑战，我希望你能在这个过程中学到一些东西。我确实做了！

所以我们开始吧！

**作者简介**

在继续之前，了解一下我的背景可能会对你有所启发。

我是一名法国工程师，在离开研究生态系统加入行业生态系统之前，我攻读了粒子物理学的硕士和博士学位，因为我想对社会产生更直接的影响。当时(2015 年)，我只为自己和 1-2 个合作者开发代码，因此你可以猜测我的产品兼容编码能力(如果你没有:没有:)。

但从那以后，我用不同的语言(主要是 C#和 Python)贡献了不同的代码库，即使我不是编队的开发人员，我也不止一次地看到了什么可行，什么不可行:)。

为了在开始旅程之前不摧毁我所有的可信度，让我赶紧补充一下，我确实对深度学习有非零的了解(2017 年在 [github](https://web.archive.org/web/20230106144228/https://github.com/tomepel/Technical_Book_DL) 上向社区提供的这本[白皮书](https://web.archive.org/web/20230106144228/https://arxiv.org/abs/1709.01412)有望证明这一事实:)。

## 构建 MLOps 管道:我遇到的最常见的问题

以下是我在过去 6 年的 ML 活动中遇到的 6 个最常见的陷阱。

在整篇文章中，我将深入探讨每一个问题，首先提出问题，然后提供可能的解决方案。

### 问题 1: POC 风格的代码

我经常遇到以概念验证(POC)风格开发的代码库。

例如，要将一个模型发布到产品中，可能需要链接 5 到 10 个 [click](https://web.archive.org/web/20230106144228/https://github.com/pallets/click) 命令(或者更糟，argparse！)为了能够:

*   预处理数据
*   特征化数据
*   训练一个 ML 模型
*   将 ML 模型导出到生产中
*   制作关于模型性能的 CSV 报告

此外，需要编辑两个命令之间的代码以使整个过程工作是非常常见的。

这在创业公司中很正常，他们想打造创新产品，而且想快速打造。但是根据我的经验，将代码库留在 POC 级别是一个长期的灾难。

事实上，随着维护成本越来越高，以这种方式添加新功能变得越来越昂贵。另一个值得考虑的因素是，在人员流动均匀的公司中，每次带着这种代码库离开都会对结构速度产生实际影响。

### 问题 2:没有高层次的关注点分离

ML 代码库中的关注点分离在高层次上经常缺失。这意味着，所谓的 ML 代码通常也在进行功能转换，比如与 ML 无关的操作——比如物理文档接收、管理数据转换等。

此外，这些模块之间的依赖关系往往没有经过深思熟虑。看看由我编写的一个小包装器创建的幻想图(我的目标是有一天在 PyPI 上发布:)，它基于优秀的 [pydeps](https://web.archive.org/web/20230106144228/https://github.com/thebjorn/pydeps) ，给出了模块级重组的代码基础依赖关系(这更接近您可能认为的真实生活情况:):

对我来说，这个图中最令人担忧的方面是低级包和高级包之间存在的循环依赖的数量。

另一个我个人认为不太好的架构是一个大的 utils 文件夹，在 ML 代码库中经常看到 utils 文件夹中有几十个模块。

### 问题 3:没有关注点的低层次分离

不幸的是，代码中的关注点分离在低层次上也经常缺失。当这种情况发生时，您最终会有 2000 多个 line 类来处理几乎所有的事情:特征化、预处理、构建模型图、训练、预测、导出……只要您说得出，这些 master 类都涵盖了您的基础(只有 coffee 不在，有时您永远不知道……:)。但正如你所知，这并不是[固体](https://web.archive.org/web/20230106144228/https://en.wikipedia.org/wiki/SOLID)的 S 会推荐的。

### 问题 4:没有配置数据模型

用于处理 ML 配置的数据模型经常缺失。例如，这是一个幻想模型超参数声明的样子(同样，比您想象的更接近真实情况)。

![MLOps pipeline hyperparameters](img/93ff7a0b91841532b7cfbb1eeeadd1bf.png)

*350+ lines dictionaries :’(*

更有问题的是(但是可以理解的)，这允许动态修改模型配置(从许多真实情况中得到灵感的幻想片段):

从上面的代码片段中可以看出,“params”属性被修改了。当这种情况在代码中的几个地方发生时(相信我，当您开始走这条路时，确实如此)，您最终得到的代码是一个真正的调试噩梦，因为您放入配置中的内容不一定是后续 ML 管道步骤中到达的内容。

### 问题 5:处理遗留模型

由于训练一个 ML 模型的过程经常涉及手工操作(见问题 1 ),这样做可能需要很长时间。它也容易出现一些错误(当人在循环中时，错误也是:)。在这种情况下，您最终会得到(幻想代码片段)这样的东西:

*提示:查看文档字符串日期🙂*

### 问题 6:代码质量:类型提示、文档、复杂性、死代码

正如上面的代码片段可以证明的那样，类型提示很少在最需要的时候出现。我猜 *n_model_to_keep* 是一个 int，但是在问题 5 的代码片段中命名 *graph_configuration* 的类型会很困难。

此外，我遇到的 ML 代码库通常只有有限数量的 docstring，并且代码质量的现代概念，如循环/认知复杂性或工作记忆(参见[这篇](https://web.archive.org/web/20230106144228/https://sourcery.ai/blog/working-memory/)帖子以了解更多信息)并不被尊重。

最后，大家都不知道的是，解决方案中经常会出现大量死代码。在这种情况下，您可能会在添加新功能的几天中绞尽脑汁，才意识到您没有设法让它与这个新功能一起工作的代码甚至没有被调用(再次，真实的故事)！

## 构建 MLOps 管道:我如何解决这些问题

现在，让我们来看看我为上述 6 个紧迫问题找到的解决方案(当然是在多年来我的合作者的帮助下),并向您概述如果我现在必须开发一个新的 MLOPS 管道，我会在哪里。

### 解决方案 1:从 poco 到 prod

多亏了 [Typer](https://web.archive.org/web/20230106144228/https://typer.tiangolo.com/) ，大量的 click/argpase 样板代码可以从命令行中取消。

我是几个咒语的忠实粉丝:

1.  最好的代码是你不需要写的代码(有趣的民间传说)。
2.  当一个观察开始被用作度量时(在这种情况下，是用来证明所有工作完成的行数)，它就不再是一个好的观察了。

在我看来，这是启动端到端 ML 模型培训的一个很好的高级命令签名:

TL DR:对所有的命令行工具使用 Typer。

### 解决方案 2:处理高层次的关注点分离——从 ML 整体服务到 ML 微服务

这是一个大问题，我花了很长时间来改进。正如我猜我的大多数读者今天一样，在微服务/monolith 之战中，我站在微服务一边(尽管我知道微服务不是弹指一挥间解决所有开发问题的奇迹)。使用 docker 和 docker-compose 来包含不同的服务，您可以增量地改进您的架构的功能，并且与其余已经实现的功能隔离。不幸的是，ML docker 架构通常是这样的:

![MLOps pipeline ML container](img/398b73ac510db7322860b56386f1636a.png)

*A typical docker ML container architecture*

现在，我想提倡一些更像这样的东西(数据处理部分也得到承认):

与 ML 无关的数据接收和存储功能现在被委托给一个专用的特性存储容器。它将接收到的数据存储到一个 [MongoDB](https://web.archive.org/web/20230106144228/https://hub.docker.com/_/mongo) (我习惯处理非结构化文档，当然如果你也/只处理结构化数据，使用一个 [Postgresql](https://web.archive.org/web/20230106144228/https://hub.docker.com/_/postgres) 容器)容器，在处理完文档后，它通过调用一个 [gotenberg](https://web.archive.org/web/20230106144228/https://gotenberg.dev/) 容器(一个非常有用的现成容器，用于处理文档)来接收数据。

ML 在这里被分成三个部分:

*   **计算机视觉部分:文档识别容器，**将计算机视觉技术应用于文档时，可以想到常见的疑点:open-cv、Pillow…我有在[标签工具](https://web.archive.org/web/20230106144228/https://github.com/Slava/label-tool)容器的帮助下做标签的经验，但是那里有很多替代品。
*   一个 NLP 部件:NLP ，带有一个将 NLP 技术应用于从文档中提取的文本的容器。想想常见的嫌疑人:nltk，Spacy，DL/BERT…我有在 [doccano](https://web.archive.org/web/20230106144228/https://github.com/doccano/doccano) 容器的帮助下做标签的经验，在我看来没有更好的替代方案了:)
*   **核心 DL 部件:pytorch_dl 容器**。在我所有的 DL 活动中，我从 TensorFlow 迁移到 PyTorch，因为与 TensorFlow 互动是我沮丧的来源。我面临的一些问题:
    *   在我的开发过程中，它很慢而且容易出错，
    *   缺乏官方 github 的支持(有些问题已经存在很多年了！),
    *   调试难度(即使 tensorflow2 的急切模式在一定程度上缓解了这一点)。

您一定听说过代码库和功能应该只进行增量更改。根据我的经验，这在 95%的情况下都是正确的好建议。但是有 5%的时间事情是如此错综复杂，并且通过做增量式的改变来悄悄破坏的危险是如此之高(低测试覆盖率，我正在看着你)，以至于我建议在一个新的包中从头重写所有的东西，确保新的包具有与旧的包相同的特性，并且此后，一举拔掉错误的代码以插入新的包。

在我以前的经历中，我曾经处理过 TensorFlow 到 PyTorch 的迁移。

为了实现 PyTorch 网络，我推荐使用 Pytorch Lightning T1，这是一个非常简洁易用的 py torch 高级库。为了衡量这种差异，我的旧 TensorFlow 代码库中的代码行大约是数千行，而 Pytorch Lightning 可以用十分之一的代码完成更多的工作。我通常在这些不同的模块中处理 DL 概念:

![MLOps pipeline Pytorch lightning](img/db4766b096adb16e3e17355b59754e67.png)

*The PyTorch-dependent code*

感谢 PyTorch Lightning，每个模块不到 50 行长(网络除外:)。

训练器是一个奇迹，你可以在弹指间使用你选择的实验记录器。我从来自 TensorFlow 生态系统的优秀的旧 TensorBoard logger 开始了我的旅程。但正如你在上面的屏幕上看到的，我最近开始使用它的一个替代品:是的，你猜对了， [neptune.ai](/web/20230106144228/https://neptune.ai/) ，到目前为止我很喜欢它。使用[和上面代码片段中看到的一样少的代码，](https://web.archive.org/web/20230106144228/https://docs.neptune.ai/getting-started/how-to-add-neptune-to-your-code)您最终将所有模型以非常用户友好的方式存储在 Neptune 仪表盘上。

[![MLOps pipeline neptune dashboard](img/3e48389fd2a30ba014260c935a9df1b4.png)](https://web.archive.org/web/20230106144228/https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/MLOps-pipeline-neptune-dashboard.png?ssl=1)

*Metrics and losses tracked in the Neptune UI*

为了进行超参数优化，在这篇深入的[博文](/web/20230106144228/https://neptune.ai/blog/optuna-vs-hyperopt)之后，这些年我从[hyperpt](https://web.archive.org/web/20230106144228/http://hyperopt.github.io/hyperopt/)切换到了 [Optuna](https://web.archive.org/web/20230106144228/https://optuna.org/) 。这种转变的原因很多。除其他外:

*   较差的远视记录
*   易于与 optuna 的 PyTorch Lightning 集成
*   超参数搜索的可视化

将为您节省大量时间的技巧:在 pytorch_dl 容器由于任何原因(服务器重启、服务器资源不足等)崩溃后，允许优雅的模型重启。)，我用相同的随机种子重放已完成运行的整个 [TPEsamplings](https://web.archive.org/web/20230106144228/https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html) ，并从最后保存的检查点开始未完成的试验。这使得我不必每次在服务器上发生意外的事情时都在未完成的运行上浪费时间。

对于我的 R&D 实验，我使用[屏幕](https://web.archive.org/web/20230106144228/https://doc.ubuntu-fr.org/screen)和越来越多的[tmux](https://web.archive.org/web/20230106144228/https://doc.ubuntu-fr.org/tmux)(tmux 上的一个[好参考](https://web.archive.org/web/20230106144228/https://books.google.fr/books/about/Tmux_2.html?id=ugsMvgAACAAJ&redir_esc=y))脚本来启动超参数优化运行。

由于 [plotly 平行坐标](https://web.archive.org/web/20230106144228/https://plotly.com/python/parallel-coordinates-plot/)图，超参数比较非常容易。

最后，我使用一个定制的 reporter 容器将一个 tex 模板编译成一个 beamer pdf。想象一下 jinja2 like [tex](https://web.archive.org/web/20230106144228/https://en.wikipedia.org/wiki/LaTeX) 模板，你用特定于每次运行的 png 和 CSV 填充该模板，以生成一个 PDF，当企业/客户开始理解机器学习模型性能(主要混淆、标签重新分配、性能等)时，该 PDF 是与他们进行对话的完美开端。).

这些架构模式极大地简化了新功能的编码。如果你熟悉 [Accelerate](https://web.archive.org/web/20230106144228/https://en.wikipedia.org/wiki/Accelerate_(book)) ，那么你就会知道，拥有一个好的代码库可以将实现一个新特性所需的时间减少 10 到 50 倍，这是千真万确的，我可以证明这一点:)。

如果您需要将消息代理添加到您的微服务架构中，我可以推荐 [rabbit MQ](https://web.archive.org/web/20230106144228/https://www.rabbitmq.com/) ,因为多亏了 pika 库，插入 python 代码非常容易。但是在这里，我对备选方案没什么可说的(除了阅读:[、卡夫卡](https://web.archive.org/web/20230106144228/https://www-inf.telecom-sudparis.eu/COURS/CILS-IAAIO/Articles/debs_kafka_versus_rabbitmq.pdf)、雷迪斯……)，因为我迄今为止从未与他们合作过。

### 解决方案 3:处理低层次的关注点分离——良好的代码架构

容器之间关注点的清晰分离允许拥有一个非常干净的容器级架构。看看这个幻想(但是我提倡的那个！:))pytorch_dl 容器的依赖关系图:

以及不同模块动作的时间顺序:

我提倡的不同模块重组的高级视图:

*   适配器将原始 CSV 转换为专用于特定预测任务的 CSV。
*   如果通过的 CSV 行未能通过给定的过滤标准(太罕见的标签等)，过滤器将删除这些行。对于过滤器和适配器，我通常使用通用类来实现所有的适配和过滤逻辑，并继承覆盖每个给定过滤器/适配器的特定适配/过滤逻辑的类([ABC/protocols 上的资源)](https://web.archive.org/web/20230106144228/https://www.youtube.com/watch?v=xvb5hGLoK0A)。
*   特性化器总是基于 sklearn，本质上是将 CSV 转换成特性名称字典(字符串)和 NumPy 数组。在这里，我将常见的疑点( [TfidfVectorizer](https://web.archive.org/web/20230106144228/https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) ， [StandardScaler](https://web.archive.org/web/20230106144228/https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) )包装到我自己的类中，本质上是因为(出于我不知道的原因)，sklearn 没有为其特征提供记忆化。我不想使用 [pickle](https://web.archive.org/web/20230106144228/https://pypi.org/project/pickle5/) ，因为它不是一个[安全兼容的](https://web.archive.org/web/20230106144228/https://medium.com/ochrona/python-pickle-is-notoriously-insecure-d6651f1974c9)库，并且不提供任何针对 [sklearn 版本变化的保护](https://web.archive.org/web/20230106144228/https://github.com/scikit-learn/scikit-learn/issues/16033#issuecomment-571656393)。因此，我总是使用自制的改进产品。
*   PyTorch 包含数据集、数据加载器和训练器逻辑。
*   模型报告生成上面已经讨论过的 pdf beamer 报告
*   例如，标记者重组确定性技术来预测(想想专家规则)稀有数据。根据我的经验，DL 模型的性能可以通过人的知识来提高，如果可行的话，您应该始终考虑这样做的可能性。
*   MLConfiguration 包含 ML 数据模型:不包含任何处理方法的枚举和类。想想超参数类，PredictionType 枚举等。旁注:在所有有意义的地方使用枚举代替字符串(封闭列表)
*   管道将所有的基本砖块连接在一起。
*   路由包含允许其他容器请求对新数据进行预测的 [FastAPI](https://web.archive.org/web/20230106144228/https://fastapi.tiangolo.com/) 路由。事实上，我把[烧瓶](https://web.archive.org/web/20230106144228/https://flask.palletsprojects.com/en/2.1.x/)放在一边的原因和我左键单击放在一边的原因是一样的——更少的模板、易用性和可维护性，甚至更多的功能。[天鬼](https://web.archive.org/web/20230106144228/https://github.com/tiangolo)是神:)。我瞥了一眼为模特服务的火炬，但考虑到我职业生涯中参与的项目规模，我还不觉得有必要投入其中。加上 TorchServe(截至 2022 年 7 月)还处于起步阶段。

我现在总是用一个定制的[预提交](https://web.archive.org/web/20230106144228/https://pre-commit.com/)钩子来执行不同代码库的模块依赖重组。这意味着每当有人试图添加新的代码来增加新的依赖关系时，协作者之间就会引发一场讨论来评估这种新的依赖关系的相关性。例如，根据我给出的架构，我看不出有什么理由要依赖 pytorch 的模型报告。并且会根据任何情况投票反对 ml_configuration。

### 解决方案 4:由于 Pydantic，简单的配置数据模型

为了避免代码中的 config 成为无类型的大字典，我对所有配置/数据模型类强制使用了 [Pydantic](https://web.archive.org/web/20230106144228/https://pydantic-docs.helpmanual.io/) 。我甚至从最好的迪斯尼电影中获得灵感🙂(参见代码片段)

这使得配置只能在一个地方定义，最好是在代码之外的 JSON 文件中，并且由于 Pydantic 一行程序可以序列化和反序列化配置。我一直关注着[九头蛇](https://web.archive.org/web/20230106144228/https://hydra.cc/docs/intro/)，但是就像这里解释的(非常好的渠道)举例来说，这个框架可能太年轻了，大概在几个月/几年后会更加成熟和自然。

为了用 optuna 试用版更新冻结的配置，我通常只定义一个静音动作字典(optuna 试用版中出现的每个超参数键的静音动作值)。

### 解决方案 5:通过频繁的自动再培训来处理遗留模型

因为训练一个模型的入口点是一个唯一的 Typer 命令(如果您遵循解决方案 1 到 4:)，所以很容易 [cron](https://web.archive.org/web/20230106144228/https://doc.ubuntu-fr.org/cron) 它定期自动地重新训练模型。由于报告和它包含的指标，您有两个层次来决定是否将新模型投入生产。

*   **自动，高级:**如果新车型的宏观性能比老款好，就把新车型投入生产。
*   **手动，细粒度:**专家可以详细比较两个模型，并得出结论，即使某个模型在整体性能方面比另一个模型稍差，但如果它的预测在出错时更有意义，它可能会更好。例如(这里有一个完全虚假的视觉例子来清楚地说明 ImageNet 上的观点)，第二个模型错了，它把老虎和狮子混为一谈，而第一个模型预测蜜蜂。

我所说的将模型导出到生产中是什么意思？在上面描述的框架中，它本质上只是将一个模型文件夹从一个位置复制到另一个位置。然后，其中一个高级配置类可以将所有这些加载到一个中，以便通过 FastApi 和 PyTorch 进行新的预测。根据我的经验，PyTorch 简化了这个过程。使用 TensorFlow，当我将模型从一个文件夹移动到另一个文件夹时，我必须手动调整模型检查点。

### 解决方案 6:提高代码质量，在我的工具的帮助下，这是一场持久战

在代码质量和附属方面，我有几匹战马:

*   正如已经提到的，我实现的所有数据模型类都基于 [Pydantic](https://web.archive.org/web/20230106144228/https://pydantic-docs.helpmanual.io/) (另一个 python 神: [Samuel Covin](https://web.archive.org/web/20230106144228/https://github.com/samuelcolvin) )。
*   我 docstring 每一个方法(但是试图禁止方法中的注释，在我看来，这是迫切需要应用优秀的老提取方法[重构](https://web.archive.org/web/20230106144228/https://martinfowler.com/books/refactoring.html)模式:)。[谷歌风格指南](https://web.archive.org/web/20230106144228/https://google.github.io/styleguide/pyguide.html)是必读之作(即使你不完全遵守它，也要知道你为什么不遵守:)。
*   我使用 [sourcery](https://web.archive.org/web/20230106144228/https://sourcery.ai/) 来自动追踪糟糕的设计并应用建议的重构模式(你可以在这里找到当前的列表[，他们定期添加新的)。这个工具是一个省时的工具——糟糕的代码不会存在很久，你的同事不必阅读它，也不必在痛苦的代码审查中指出它。事实上，我建议每个人在 pycharm 上使用的唯一扩展是 sourcery 和](https://web.archive.org/web/20230106144228/https://docs.sourcery.ai/refactorings/) [tabnine](https://web.archive.org/web/20230106144228/https://www.tabnine.com/)
*   在其他预提交钩子中(还记得我已经谈到的关于高级依赖关系的自制钩子)，我使用了 [autopep8](https://web.archive.org/web/20230106144228/https://github.com/pre-commit/mirrors-autopep8) 、 [flake](https://web.archive.org/web/20230106144228/https://github.com/pre-commit/pre-commit-hooks) 和 [mypy](https://web.archive.org/web/20230106144228/https://github.com/pre-commit/mirrors-mypy) 。
*   我使用 [pylint](https://web.archive.org/web/20230106144228/https://pypi.org/project/pylint/) 对我的代码库进行 lint 处理，目标是 9-9.5。这完全是武断的，但正如理查德·塞勒所说——“我确信这有一个进化的解释，如果你给他们(男人)一个目标，他们就会瞄准。”
*   我使用 [unittest](https://web.archive.org/web/20230106144228/https://docs.python.org/3/library/unittest.html) (这是我曾经使用过的一个，我觉得没有必要切换到 pytest。即使这意味着一些样板文件，只要测试存在，我在测试方面会更加宽容！).和上一点提到的原因一样，我的目标是 95%的覆盖率。
*   对于导入，我采用了 [sklearn 模式](https://web.archive.org/web/20230106144228/https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/covariance/__init__.py)，这意味着导入到模块重组文件夹之外的所有内容，其中 __init__。py stands 必须在这个非常 __init__.py 中列出，这里列出的每个类/方法都是“包”的接口，必须经过测试(单元的和/或功能的)。
*   我经常试图实现跨平台的确定性测试(读[这个](https://web.archive.org/web/20230106144228/https://pytorch.org/docs/stable/notes/randomness.html)和[这个](https://web.archive.org/web/20230106144228/https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#reproducibility))但是失败了(虽然我确实在固定平台上成功了)。由于 GitLab 跑步者经常变化，这通常会导致很多痛苦。我决定在端到端测试中有一个“足够高”的性能。
*   为了避免几个容器之间的代码重复，我提倡在每个容器中安装一个低级的自制库(通过各自 docker 文件中的命令行)。
*   关于 [CI](https://web.archive.org/web/20230106144228/https://en.wikipedia.org/wiki/Continuous_integration) ，在各自的 GitLab 管道中构建你的 docker 镜像。
*   尽量不要在生产中挂载代码(但是在本地挂载以便于开发。非常好的[参考](https://web.archive.org/web/20230106144228/https://pythonspeed.com/)docker+python 上的博客)。
*   我不在产品中发布测试，也不发布运行测试所需的库(因此你应该瞄准两个需求文件，一个 requirement-dev.txt 不在产品中使用)。
*   我经常有一个定制的 python dev docker-compose 文件来简化我的生活(和新成员的加入),它不同于生产文件。
*   我主张(广泛地)使用你的 GitLab repos 的 wiki 部分:)，因为口述传统在人类历史的某些阶段是好的，但肯定不适合 IT 公司:)。
*   我试图最小化我的容器上装载的卷的数量，最好的数量是 0，但是对于一些数据源(比如模型检查点)来说，这可能很复杂。
*   处理死代码有一个简单的解决方法:[秃鹫](https://web.archive.org/web/20230106144228/https://pypi.org/project/vulture/)。运行它，检查(密切关注，因为他们是一些假阳性)它的输出，拔掉死代码，清洗和重复。

## 结论

很多时候，你会看到沾沾自喜的文章掩盖了 ML 领域的真实生活。我希望你离开这篇文章时知道这不是其中的一篇。这是我在过去六年中开发 MLOPS 管道的真实旅程，当我回顾我在 2006 年开始编码时的情况时，我会更加自豪(C 代码中超过 400 个字符的一行方法:)。

根据我的经验，有些切换决策很容易做出和实现(flask 到 FastAPI)，有些很容易做出但不容易实现(比如 Hyperopt 到 Optuna)，有些很难做出也很难实现(比如 TensorFlow 到 PyTorch)，但最终都值得努力避免我提出的 6 个陷阱。

这种心态有望让你从类似 POC 的 ML 环境过渡到符合 [Accelerate](https://web.archive.org/web/20230106144228/https://en.wikipedia.org/wiki/Accelerate_(book)) 的环境，在这种环境中，实现新特性不到一个小时，将它们添加到代码库也不到一个小时。

就我个人而言，我学到了很多东西，为此我深深感谢我以前的雇主和同事！