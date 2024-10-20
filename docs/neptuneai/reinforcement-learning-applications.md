# 强化学习的 10 个现实应用

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/reinforcement-learning-applications>

在**强化学习(RL)** 中，智能体按照**奖励**和**惩罚**机制接受训练。代理人因正确的移动而得到奖励，因错误的移动而受到惩罚。在这样做的时候，代理人试图最小化错误的移动，最大化正确的移动。

在本文中，我们将看看强化学习在现实世界中的一些应用。

## 自动驾驶汽车中的应用

各种论文提出了[深度强化学习](https://web.archive.org/web/20230228161547/https://arxiv.org/pdf/2002.00444.pdf)用于**自动驾驶**。在自动驾驶汽车中，有各种各样的方面需要考虑，例如各个地方的速度限制，可驾驶区，避免碰撞——仅举几例。

一些可以应用强化学习的自动驾驶任务包括轨迹优化、运动规划、动态路径、控制器优化和基于场景的高速公路学习策略。

比如，可以通过学习自动泊车政策来实现泊车。使用 [Q-Learning](https://web.archive.org/web/20230228161547/https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56) 可以实现变道，而超车可以通过学习超车策略来实现，同时避免碰撞并在其后保持稳定的速度。

AWS DeepRacer 是一款自主赛车，旨在在物理赛道上测试 RL。它使用摄像头来可视化跑道，并使用强化学习模型来控制油门和方向。

Wayve.ai 已经成功地将强化学习应用于训练汽车如何在一天内**驾驶。**他们使用深度强化学习算法来处理车道跟随任务。他们的网络架构是一个具有 4 个卷积层和 3 个全连接层的深度网络。以下示例显示了车道跟踪任务。中间的图像代表驾驶员的视角。

<https://web.archive.org/web/20230228161547im_/https://neptune.ai/wp-content/uploads/2022/11/10-Real-Life-Applications-of-Reinforcement-Learning.mp4>

[**Source**](https://web.archive.org/web/20230228161547/https://wayve.ai/blog/learning-to-drive-in-a-day-with-reinforcement-learning/) 

## 具有强化学习的工业自动化

在工业强化中，基于学习的**机器人**被用来执行各种任务。除了这些机器人比人更有效率这一事实之外，它们还能完成对人来说很危险的任务。

一个很好的例子是 [Deepmind 使用人工智能代理来冷却谷歌数据中心](https://web.archive.org/web/20230228161547/https://deepmind.com/blog/article/safety-first-ai-autonomous-data-centre-cooling-and-industrial-control)。这导致**能源支出**减少了 40%。这些中心现在完全由人工智能系统控制，无需人工干预。显然仍然有来自数据中心专家的监督。该系统的工作方式如下:

*   每五分钟从数据中心获取数据快照，并将其输入深度神经网络
*   然后，它预测不同的组合将如何影响未来的能源消耗
*   确定在维持安全标准的同时将导致最小功耗的行动
*   在数据中心发送和实施这些操作

这些动作由本地控制系统验证。

## 强化学习在交易和金融中的应用

有监督的[**时间序列**](https://web.archive.org/web/20230228161547/https://arxiv.org/ftp/arxiv/papers/1803/1803.03916.pdf) 模型可以用于预测未来的销售以及预测**股票价格**。然而，这些模型并不能决定在特定的股票价格下应该采取的行动。进入强化学习(RL)。RL 代理可以决定这样的任务；是持有、买入还是卖出。RL 模型使用市场基准标准进行评估，以确保其性能最佳。

这种自动化为流程带来了一致性，不像以前的方法，分析师必须做出每一个决策。例如，IBM 有一个复杂的基于强化学习的平台，能够进行**金融交易**。它根据每笔金融交易的盈亏来计算回报函数。

## 自然语言处理中的强化学习

在 NLP 中，RL 可以用于**文本摘要**、**问答、**和**机器翻译**等等。

这篇[论文的作者](https://web.archive.org/web/20230228161547/https://homes.cs.washington.edu/~eunsol/papers/acl17eunsol.pdf) Eunsol Choi、Daniel Hewlett 和 Jakob Uszkoreit 提出了一种基于强化学习的方法来回答给定的长文本问题。他们的方法是首先从文档中选择一些与回答问题相关的句子。然后用慢速 RNN 来回答所选的句子。

在本文中，监督学习和强化学习的组合被用于[抽象文本摘要。该报由罗曼·保卢斯、蔡明·熊&理查德·索歇署名。他们的目标是在较长的文档中使用注意力的、基于 RNN 的编码器-解码器模型时，解决在**摘要**中面临的问题。本文的作者提出了一种具有新型内部注意力的神经网络，该网络关注输入并分别连续产生输出。他们的训练方法是标准监督单词预测和强化学习的结合。](https://web.archive.org/web/20230228161547/https://arxiv.org/pdf/1705.04304.pdf)

在机器翻译方面，来自科罗拉多大学和马里兰大学的作者提出了一种基于强化学习的同步 T2 机器翻译方法。这项工作有趣的地方在于，它有能力学习何时信任预测的单词，并使用 RL 来确定何时等待更多输入。

来自斯坦福大学、俄亥俄州立大学和微软研究院的研究人员已经将 Deep RL 用于对话生成。深度 RL 可以用于在**聊天机器人对话中模拟未来的奖励。**使用两个虚拟代理模拟对话。策略梯度方法用于奖励包含重要对话属性的序列，如连贯性、信息性和易于回答。

更多 NLP 应用可以在[这里](https://web.archive.org/web/20230228161547/https://github.com/adityathakker/awesome-rl-nlp)或者[这里](https://web.archive.org/web/20230228161547/https://www.future-processing.com/blog/the-future-of-natural-language-processing/)找到。

## 强化学习在医疗保健中的应用

在医疗保健领域，患者可以**根据从 RL 系统学习到的政策**接受治疗。RL 能够使用先前的经验找到最优策略，而不需要关于生物系统的数学模型的先前信息。这使得这种方法比医疗保健中其他基于控制的系统更适用。

医疗保健中的 RL 被归类为慢性病或重症护理、自动医疗诊断和其他一般领域中的[动态治疗方案(DTRs)](https://web.archive.org/web/20230228161547/https://arxiv.org/pdf/1908.08796.pdf) 。

在 DTRs 中，输入是一组对患者的临床观察和评估。输出是每个阶段的治疗方案。这些类似于 RL 中的状态。在 DTRs 中应用 RL 是有利的，因为它能够确定依赖于时间的决定，以便在特定时间对患者进行最佳治疗。

[RL 在医疗保健中的使用](https://web.archive.org/web/20230228161547/https://arxiv.org/pdf/1908.08796.pdf)也通过考虑治疗的延迟效应而改善了长期结果。

RL 还被用于发现和产生用于慢性疾病的最佳 DTR。

您可以通过浏览这篇[文章](https://web.archive.org/web/20230228161547/https://arxiv.org/pdf/1908.08796.pdf)来深入了解医疗保健领域的 RL 应用。

## 强化学习在工程中的应用

在工程前沿，脸书开发了**开源强化学习平台**——[地平线](https://web.archive.org/web/20230228161547/https://engineering.fb.com/ml-applications/horizon/)。该平台使用强化学习来优化大规模生产系统。脸书在内部使用了 Horizon:

*   个性化建议
*   向用户发送更有意义的通知
*   优化视频流质量。

Horizon 还包含以下工作流:

*   模拟环境
*   分布式数据预处理平台
*   在生产中培训和导出模型。

视频显示中强化学习的一个经典示例是根据视频缓冲区的状态和来自其他机器学习系统的估计，为用户提供低比特率或高比特率的视频。

Horizon 能够处理类似生产的问题，例如:

*   大规模部署
*   特征标准化
*   分布学习
*   提供和处理包含高维数据和数千种要素类型的数据集。

## 新闻推荐中的强化学习

用户偏好可能经常改变，因此基于评论和喜欢向用户推荐新闻的[](https://web.archive.org/web/20230228161547/http://www.personal.psu.edu/~gjz5038/paper/www2018_reinforceRec/www2018_reinforceRec.pdf)**可能会很快过时。通过强化学习，RL 系统可以跟踪读者的返回行为。**

 **构建这样的系统将涉及获取新闻特征、读者特征、上下文特征和读者新闻特征。新闻特写包括但不限于内容、标题和出版商。阅读器功能指的是阅读器如何与内容互动，例如点击和分享。上下文特征包括新闻方面，例如新闻的时间和新鲜度。然后根据这些用户行为定义奖励。

## 游戏中的强化学习

让我们来看看**游戏**前沿的一个应用，具体来说就是 **AlphaGo Zero** 。使用强化学习，AlphaGo Zero 能够从零开始学习围棋。它通过与自己对抗来学习。经过 40 天的自我训练，Alpha Go Zero 能够超越被称为*大师*的 Alpha Go 版本，该版本击败了[世界排名第一的柯洁](https://web.archive.org/web/20230228161547/https://deepmind.com/alphago-china)。它只使用棋盘上的黑白棋子作为输入特征和一个单一的神经网络。依赖于单个神经网络的简单树搜索用于评估位置移动和样本移动，而不使用任何[蒙特卡罗](https://web.archive.org/web/20230228161547/https://en.wikipedia.org/wiki/Monte_Carlo_method)展开。

## 实时竞价——强化学习在营销和广告中的应用

在这篇[论文](https://web.archive.org/web/20230228161547/https://arxiv.org/pdf/1802.09756.pdf)中，作者提出了具有多智能体强化学习的**实时竞价**。使用聚类方法和给每个聚类分配一个策略投标代理来处理大量广告客户的处理。为了平衡广告商之间的竞争与合作，提出了一种分布式协调多代理竞价(DCMAB)。

在市场营销中，准确定位个人的能力至关重要。这是因为正确的目标显然会带来高投资回报。本文的研究基于中国最大的电子商务平台淘宝。所提出的方法优于最先进的单代理强化学习方法。

## 机器人操作中的强化学习

深度学习和强化学习[的使用可以训练机器人](https://web.archive.org/web/20230228161547/https://ai.googleblog.com/2018/06/scalable-deep-reinforcement-learning.html)，这些机器人有能力抓住各种物体——甚至是那些在训练中看不见的物体。例如，这可以用于在装配线上制造产品。

这是通过结合大规模分布式优化和一种叫做 [QT-Opt](https://web.archive.org/web/20230228161547/https://arxiv.org/abs/1806.10293) 的[深度 Q 学习](https://web.archive.org/web/20230228161547/https://en.wikipedia.org/wiki/Q-learning)来实现的。QT-Opt 对连续动作空间的支持使其适用于机器人问题。首先离线训练一个模型，然后在真实的机器人上进行部署和微调。

谷歌人工智能将这种方法应用于**机器人抓取**，其中 7 个真实世界的机器人在 4 个月的时间内运行了 800 个机器人小时。

在[这个实验](https://web.archive.org/web/20230228161547/https://www.youtube.com/watch?v=W4joe3zzglU)中，QT-Opt 方法在 700 次抓取之前看不到的物体的尝试中，有 96%的抓取成功。Google AI 之前的方法有 78%的成功率。

## 最后的想法

虽然强化学习仍然是一个非常活跃的研究领域，但在推进该领域并将其应用于现实生活方面已经取得了重大进展。

在本文中，就强化学习的应用领域而言，我们仅仅触及了皮毛。希望这能激发你的好奇心，让你更深入地探索这个领域。如果你想了解更多，看看这个[棒极了的回购](https://web.archive.org/web/20230228161547/https://github.com/aikorea/awesome-rl)——没有双关语，还有[这个](https://web.archive.org/web/20230228161547/https://github.com/dennybritz/reinforcement-learning)。**