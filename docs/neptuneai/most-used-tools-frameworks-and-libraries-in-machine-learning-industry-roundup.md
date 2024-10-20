# 机器学习行业中最常用的工具、框架和库(综述)

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/most-used-tools-frameworks-and-libraries-in-machine-learning-industry-roundup>

机器学习越来越受欢迎，越来越多的公司正在利用这项新技术的力量。然而，关于团队本身的知识仍然有限——他们使用什么？他们喜欢什么？他们是谁？

Neptune 是由数据科学家为数据科学家打造的——中间没有晦涩难懂的东西。因此，当面临知识缺失的挑战时，我们会尽最大努力收集数据。受益于我们的联系人和读者网络，其中许多人本身就是机器学习专家，我们发起了一项投票，以获得关于最流行的技术、团队的性质及其日常工作的答案。

结果？看下面！

你在研究机器学习的哪个领域？

## 第一个问题——团队在做什么？炒作其他技术的墙后面是不是有某种隐藏的王者？一些现代机器学习的和平领导者，在服务器机房的无菌光线下稳步增长，而媒体供应商继续追逐过度兴奋的幽灵？

我们将类别分为:

计算机视觉

1.  强化学习
2.  自然语言处理
3.  预测
4.  扁平的
5.  其他的
6.  我们发现团队中最突出的领域是 **NLP(自然语言处理)**，其次是**计算机视觉、预测、表格、其他和强化学习**。

根据 2019 年的一份报告，[在未来 5 年](https://web.archive.org/web/20221207095620/https://solutionsreview.com/data-management/80-percent-of-your-data-will-be-unstructured-in-five-years/)内，您 80%的数据将是非结构化的，我们可以使用这些数据来为情感分析、命名实体识别、主题分割、文本摘要、关系提取等任务建立机器学习模型——只要是您能想到的。

考虑到 NLP 技术的进步，更多的用例将很快出现。

当您连接到公司的服务中心时，您可能会在电话中听到“此电话可能会出于质量和培训目的而被录音”，或者在您从某个组织购买服务或产品后被要求填写调查问卷。

这些信息可以用来了解受众对某个组织的产品或服务的情感倾向。

在进行的调查中，31 个团队中的 26 个团队确认他们都与 NLP 一起处理文本数据。

几乎所有的团队都在几乎所有的领域工作过，除了强化学习。

“强化学习是一种基于代理的学习，它使代理能够在交互式环境中通过试错来学习。”

但考虑到强化学习仍然是更学术和研究的领域，而不是商业领域，这并不奇怪。一个严重的原因是——强化学习可能不适合初创公司，因为强化学习所需的[数据](https://web.archive.org/web/20221207095620/https://www.youtube.com/watch?v=NQK4ZY_gwKI)是海量的，[面临众多技术挑战](https://web.archive.org/web/20221207095620/https://www.alexirpan.com/2018/02/14/rl-hard.html)。

[OpenAIFive](https://web.archive.org/web/20221207095620/https://openai.com/five/) 系统的估计批量超过 100 万次观测。这意味着我们有超过一百万的国家行动奖励来更新每个模拟。这使得强化学习样本有效，因为需要大量训练数据来达到期望的性能。

例如， [AlphaGo](https://web.archive.org/web/20221207095620/https://deepmind.com/research/case-studies/alphago-the-story-so-far) ，第一个在[围棋](https://web.archive.org/web/20221207095620/https://en.wikipedia.org/wiki/Go_(board_game))中击败世界冠军 [Lee Sedol](https://web.archive.org/web/20221207095620/https://en.wikipedia.org/wiki/Lee_Sedol) 的 AI 智能体，通过玩数百万场游戏不间断地接受了几天的训练，在这些模拟中积累了数千年的知识，估计仅计算能力就花费了近[300 万美元](https://web.archive.org/web/20221207095620/https://www.yuzeh.com/data/agz-cost.html)。

AlphaGo Zero 向所有人展示了建立甚至可以击败世界冠军的系统是可能的，但由于模型的昂贵，大多数初创公司仍然无法开发这种系统。

关键要点——在有钱的地方工作。无论是在面向商业的领域，还是对拥有大量现金用于强化学习的科技巨头来说。

你和什么类型的模特一起工作？

## 我们向所有团队询问了他们用来构建模型的方法，并将它们分为四类。

深度学习

1.  增强树(LightGBM、XGBoost、Catboost)
2.  线性回归
3.  其他的
4.  深度学习模型在这里被证明是赢家，被 14 个团队经常使用。

31 个团队中至少有 29 个在某个时候使用了深度学习模型。

我们在这里可以再次看到，深度学习是赢家。由于在大量数据集上训练时的预测准确性，深度学习越来越受欢迎。

深度学习往往最适合非结构化数据，根据 [Gartner](https://web.archive.org/web/20221207095620/https://www.gartner.com/en) 、[的数据，80%的企业数据是以文本、图像、pdf 等形式存在的非结构化数据](https://web.archive.org/web/20221207095620/https://www.dataversity.net/putting-structure-around-unstructured-data/#:~:text=In%20fact%2C%20analysts%20at%20Gartner,of%20enterprise%20data%20is%20unstructured.&text=Finding%20ways%20to%20integrate%20new,the%20data%20as%20it%20grows.)。

深度学习方法应用于社交媒体和非结构化数据，以更好地了解客户并对其进行细分。金融行业正在快速采用深度学习方法来构建用于检测欺诈交易、交易、股票预测等的系统。

医疗保健行业正在利用深度学习的力量进行药物发明、疾病诊断、虚拟医疗保健，这样的例子不胜枚举。

因此，团队在保持其通用性的同时，坚持使用交付结果的技术。因此，深度学习是一项领先的技术，似乎没有什么可以改变这一进程。

你现在用什么来训练你的 ML 模型？

## 由于计算能力有限，训练机器学习或深度学习模型有时可能会令人生畏。另一方面，聪明的数据科学家可以调整模型，减少对数据的渴求，使用更少的资源。因此，我们向所有 31 个团队询问了他们用于训练或建立预测模型的基础设施。

我们可以将它们分为四类。

本地机器(笔记本电脑或个人电脑)

1.  本地集群
2.  大云(AWS、GCP、Azure)
3.  专用 ML 云(Sagemaker、Floydhub)
4.  我们在这里看到的使用最少的基础设施是**专用的 ML 云(SageMaker，Floydhub)**，使用最多的是这里的**本地 pc 和大云(AWS，GCP，Azure)** 。

也许是因为在专门的 ML 云服务上训练你的模型有时会在你的口袋里烧一个洞。对于现金充裕的团队来说，它们很酷——确实有这样的团队，但不是很多。

除了专用的 ML cloud (SageMaker，Floydhub)之外，我们可以看到所有平台上的模型训练，无论是本地机器，大云还是本地集群，都在某个时候被使用过。这并不奇怪——在本地以较低的成本运行 prototype 以节省云中的试错过程是很常见的——最终，每一秒都要花费实实在在的现金。

我们可以看到，训练预测模型最常用的基础设施是大云(Azure、AWS、GCP)。

有时候你可能只有有限的 RAM 和 GPU 来训练一个预测模型，你只是不能随心所欲的升级系统来满足你的系统需求；这就是云服务发挥作用的地方。或者你希望在一个小时内完成训练，而不是一周——毕竟钱不是唯一的价值。

数据科学团队倾向于让每个数据科学家的机器学习任务成为他们框架武器库中的武器，以解决任何数据科学问题。

**数据科学家最常用的库**

### Scikit-learn 是一个开源库，适用于任何公司或初创公司的每个数据科学家的机器学习任务。它建立在几个现有的 python 包之上，如 Numpy、Scipy 和 Matplotlib。

我们可以执行各种各样的机器学习算法，如回归、分类、聚类等。具有各种性能度量，如 MSE(均方误差)、AUC(曲线下面积)、ROC(接收机工作特性)等。

像[摩根大通](https://web.archive.org/web/20221207095620/https://www.jpmorgan.com/)、 [Spotify](https://web.archive.org/web/20221207095620/https://www.spotify.com/) 、 [Inria](https://web.archive.org/web/20221207095620/https://inria.fr/) 、[抱脸](https://web.archive.org/web/20221207095620/https://huggingface.co/)、[等](https://web.archive.org/web/20221207095620/https://scikit-learn.org/stable/testimonials/testimonials.html)的公司。正在将 [scikit-learn](https://web.archive.org/web/20221207095620/https://scikit-learn.org/) 用于他们的机器学习工作流程。

TensorFlow 是一个开源库，由 Google 开发，用于进行端到端的机器学习项目。

TensorFlow 使用数据流图，其中数据(张量)可以由一系列处理。TensorFlow 提供了简单的模型构建、ML 工具如 [TensorBoard](https://web.archive.org/web/20221207095620/https://www.tensorflow.org/tensorboard) 和 ML 产品。

像[英伟达](https://web.archive.org/web/20221207095620/https://www.nvidia.com/en-us/)、 [Snapchat](https://web.archive.org/web/20221207095620/https://www.snapchat.com/) 、[高通](https://web.archive.org/web/20221207095620/https://www.qualcomm.com/)、[苹果公司](https://web.archive.org/web/20221207095620/https://www.apple.com/sitemap/)这样的公司正在将 TensorFlow 用于他们基于人工智能的应用。

[PyTorch](https://web.archive.org/web/20221207095620/https://pytorch.org/) 是由[脸书](https://web.archive.org/web/20221207095620/https://www.facebook.com/)创建的一个开源机器学习库，使用户能够使用 GPU 加速进行张量计算，并建立深度学习模型。

像[微软](https://web.archive.org/web/20221207095620/https://www.microsoft.com/)、 [Airbnb](https://web.archive.org/web/20221207095620/https://www.airbnb.com/) 和 [OpenAI](https://web.archive.org/web/20221207095620/https://openai.com/) 这样的公司似乎利用 PyTorch 来利用人工智能的力量。OpenAI 甚至终止了它对强化学习教育研究的 TensorFlow 使用。

**数据科学家最常用的工具**

### 除了硬件和技术方面，还有日常工作使用常见的贸易工具。铁匠有他的锤子，木匠有他的凿子，数据科学家有…

是的，每个人的朋友都来了。Jupyter Notebook 是一个基于网络的交互式环境，用于构建机器学习模型和数据可视化。您可以执行其他几项任务，如数据清理、数据转换、统计建模等。

Weka 引用“无需编程的机器学习”。Weka 可以建立机器学习管道，训练分类器，而无需编写一行代码。

Weka 还提供了一个用于深度学习的包，名为 [WekaDeepLearning4j](https://web.archive.org/web/20221207095620/https://deeplearning.cms.waikato.ac.nz/) ，它有一个 GUI，可以直接构建神经网络、卷积网络和递归网络。

Apache spark 提供了一个名为 MLlib 的机器学习 API，用于分类、聚类、频繁模式匹配、推荐系统、回归等机器学习任务。

最后的想法

参加调查让我们深入了解了最突出的领域，用于 ML 工作流的机器学习基础设施，最常用的机器学习方法，为什么深度学习压倒了传统的机器学习，以及强化学习的问题。

## 但与流行的方法相反，这是来自数据科学家之间的数据科学家，没有旨在出售任何东西的夸大其词。答案的真诚和准确性为我们提供了对数据科学界的乐观看法——当涉及到我们使用的技术堆栈或工具时，我们有自知之明，充满常识，并抵制胡说八道。

在今天这个臃肿而充满活力的世界里，这很酷。

But contrary to the popular approach, this has come from data scientists to data scientists with no overhyped babble aimed to sell anything. The sincerity and accuracy of the answers provided us with an optimistic view of the data science community – we are self aware, full of the common sense and bullshit-resistant when it comes to tech stack or tools we use. 

And that’s cool in today’s bloated and pumped-up word.