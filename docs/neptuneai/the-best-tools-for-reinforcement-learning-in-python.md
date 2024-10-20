# Python 中强化学习的最佳工具是你真正想尝试的

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/the-best-tools-for-reinforcement-learning-in-python>

如今，[深度强化学习](https://web.archive.org/web/20230306225449/https://medium.com/ai%C2%B3-theory-practice-business/reinforcement-learning-part-1-a-brief-introduction-a53a849771cf) (RL)是数据科学界最热门的话题之一。快速开发的快速发展导致了对易于理解和方便使用的快速开发工具的需求不断增长。

近年来，大量的 RL 库被开发出来。这些库被设计成拥有实现和测试**强化学习**模型的所有必要工具。

尽管如此，他们还是有很大的不同。这就是为什么选择一个快速、可靠、与你的 RL 任务相关的库是很重要的。

在本文中，我们将涵盖:

*   选择**深度强化学习**库的标准，
*   RL 库: **Pyqlearning** ， **KerasRL** ， **Tensorforce** ，**RL _ 蔻驰**， **TFAgents** ， **MAME RL** ， **MushroomRL** 。

## 用于强化学习的 Python 库

有很多 RL 库，因此为您的情况选择正确的库可能是一项复杂的任务。我们需要形成标准来评估每个库。

### **标准**

本文中的每个 RL 库都将基于以下标准进行分析:

1.  实施的**最先进的**(**【SOTA】)**RL 算法的数量——我认为最重要的一个
2.  官方**文档、简单教程**和示例的可用性
3.  **易于定制的可读代码**
4.  支持环境的**数量——这是**强化学习**库的关键决定因素**
5.  **记录和跟踪工具**支持——例如 Neptune 或 TensorBoard
6.  **矢量化环境** ( **VE** )特征——进行多进程训练的方法。使用并行环境，你的代理将会经历比单一环境更多的情况
7.  **定期更新**–RL 发展非常迅速，您希望使用最新技术

我们将讨论以下库:

## 角膜的

[**KerasRL**](https://web.archive.org/web/20230306225449/https://github.com/keras-rl/keras-rl) 是一个**深度强化学习** Python 库。它实现了一些最先进的 RL 算法，并与**深度学习**库 **[Keras](/web/20230306225449/https://neptune.ai/integrations/keras)** 无缝集成。

此外， **KerasRL** 与 [OpenAI Gym](https://web.archive.org/web/20230306225449/https://gym.openai.com/) 开箱即用。这意味着你可以很容易地评估和使用不同的算法。

要安装 **KerasRL** ，只需使用一个 pip 命令:

```py
pip install keras-rl
```

让我们看看 **KerasRL** 是否符合标准:

1.  实施的 **SOTA** RL 算法的数量

到今天**为止，KerasRL** 已经实现了以下算法:

*   **深度 Q-Learning** ( **DQN** )及其改进(**双**和**对决**)
*   **深度确定性政策梯度** ( **DDPG** )
*   **连续 DQN** ( **CDQN** 或 **NAF** )
*   **交叉熵方法** ( **CEM** )
*   **深 SARS**

你可能已经注意到了， **KerasRL** 忽略了两个重要因素:行动者批评方法和邻近政策优化(PPO)。

2.  官方文档、教程和示例的可用性

代码很容易阅读，并且充满了注释，这非常有用。尽管如此，文档似乎不完整，因为它错过了参数和教程的解释。此外，实际例子也有许多不足之处。

3.  易于定制的可读代码

非常容易。您需要做的就是按照示例创建一个新的代理，然后将其添加到 **rl.agents** 。

4.  支持的环境数量

KerasRL 被要求只与**开放健身馆**合作。因此，如果要使用任何其他环境，您需要修改代理。

5.  日志和跟踪工具支持

未实现日志和跟踪工具支持。不过，你可以用 [**海王星**来追踪你的实验](https://web.archive.org/web/20230306225449/https://docs.neptune.ai/how-to-guides/experiment-tracking)。

6.  矢量化环境特征

包括矢量化的环境特征。

7.  定期更新

该库似乎不再维护，因为上次更新是在一年多以前。

综上所述， **KerasRL** 有一套很好的实现。不幸的是，它错过了有价值的点，如可视化工具，新的架构和更新。你应该使用另一个图书馆。

## Pyqlearning

[Pyqlearning](https://web.archive.org/web/20230306225449/https://pypi.org/project/pyqlearning/) 是实现 RL 的 Python 库。重点介绍了 **Q-Learning** 和**多智能体深度 Q-Network。**

**Pyqlearning** 为设计师提供组件，而不是为最终用户提供最先进的黑匣子。因此，这个库很难使用。你可以用它来设计信息搜索算法，比如 GameAI 或者 web crawlers。

要安装 **Pyqlearning** ，只需使用一个 pip 命令:

```py
pip install pyqlearning
```

让我们看看 **Pyqlearning** 是否符合标准:

1.  实施的 **SOTA** RL 算法的数量

到今天**为止，Pyqlearning** 已经实现了以下算法:

*   **深度 Q 学习** ( **DQN** )及其改进(**ε贪心**和**玻尔兹曼**)

您可能已经注意到， **Pyqlearning** 只有一个重要的代理。这个图书馆还有许多需要改进的地方。

2.  官方文档、教程和示例的可用性

**Pyqlearning** 有几个不同任务的例子和两个由 **Deep Q-Network** 开发的迷宫解决和追逃游戏教程。你可以在[官方文件](https://web.archive.org/web/20230306225449/https://code.accel-brain.com/Reinforcement-Learning/)中找到它们。文档似乎不完整，因为它关注的是数学，而不是库的描述和使用。

3.  易于定制的可读代码

Pyqlearning 是一个开源库。源代码可以在 [Github](https://web.archive.org/web/20230306225449/https://github.com/chimera0/accel-brain-code/tree/master/Reinforcement-Learning) 上找到。代码缺少注释。定制它可能是一项复杂的任务。不过，教程可能会有所帮助。

4.  支持的环境数量

因为这个库是不可知的，所以添加到任何环境都相对容易。

5.  日志和跟踪工具支持

作者在教程中使用了一个简单的**日志**包。 **Pyqlearning** 不支持其他测井和跟踪工具，例如 **[TensorBoard](/web/20230306225449/https://neptune.ai/vs/tensorboard)** 。

6.  矢量化环境特征

Pyqlearning 不支持矢量化环境功能。

7.  定期更新

图书馆得到了维护。最后一次更新是在两个月前。尽管如此，开发过程似乎是一个缓慢的过程。

总而言之，学习还有许多不足之处。这不是一个你通常会用到的库。因此，您可能应该使用其他东西。

## 张量力

[Tensorforce](https://web.archive.org/web/20230306225449/https://github.com/tensorforce/tensorforce) 是一个基于谷歌 **Tensorflow** 框架构建的开源**深度** RL 库。它的用法很简单，有可能成为最好的强化学习库之一。

**Tensorforce** 拥有与其他 RL 库不同的关键设计选择:

*   基于组件的模块化设计:最重要的是，功能实现往往尽可能地具有通用性和可配置性。
*   RL 算法与应用的分离:算法不知道输入(状态/观察)和输出(动作/决策)的类型和结构，以及与应用环境的交互。

要安装 **Tensorforce** ，只需使用一个 pip 命令:

```py
pip install tensorforce
```

让我们看看 **Tensorforce** 是否符合标准:

1.  实施的 **SOTA** RL 算法的数量

截至今天， **Tensorforce** 已经实施了以下算法:

*   **深度 Q-Learning** ( **DQN** )及其改进(**双**和**对决**)
*   **香草政策梯度** ( **PG**
*   **深度确定性政策梯度** ( **DDPG** )
*   **连续 DQN** ( **CDQN** 或 **NAF** )
*   **演员评论家** ( **A2C 和 A3C** )
*   **信任区域政策优化** ( **TRPO**
*   **近端策略优化** ( **PPO** )

你可能已经注意到了， **Tensorforce** 错过了**软演员评论家** ( **SAC** )实现。此外，它是完美的。

2.  官方文档、教程和示例的可用性

由于有各种简单的例子和教程，开始使用 **Tensorforce** 非常容易。[官方文档](https://web.archive.org/web/20230306225449/https://tensorforce.readthedocs.io/en/latest/index.html)看起来很完整，浏览起来也很方便。

3.  易于定制的可读代码

**Tensorforce** 得益于其模块化设计。架构的每个部分，例如网络、模型、转轮都是不同的。因此，您可以轻松地修改它们。然而，代码缺少注释，这可能是一个问题。

4.  支持的环境数量

**Tensorforce** 与多种环境协同工作，例如 **OpenAI Gym** 、 **OpenAI Retro** 和 **DeepMind Lab** 。它也有帮助您插入其他环境的文档。

5.  日志和跟踪工具支持

该库支持 **TensorBoard** 和其他测井/跟踪工具。

6.  矢量化环境特征

**Tensorforce** 支持矢量化环境特征。

7.  定期更新

**Tensorforce** 定期更新。最近一次更新是在几周前。

综上所述， **Tensorforce** 是一款强大的 RL 工具。它是最新的，并且拥有开始使用它所需的所有文档。

## 蔻驰 RL

英特尔 AI Lab 的[强化学习蔻驰](https://web.archive.org/web/20230306225449/https://github.com/NervanaSystems/coach) ( **蔻驰**)是一个 Python RL 框架，包含许多最先进的算法。

它公开了一组易于使用的 API，用于试验新的 RL 算法。该库的组件，例如算法、环境、神经网络架构是模块化的。因此，扩展和重用现有的组件是相当容易的。

要安装**蔻驰**只需使用一个 pip 命令。

```py
pip install rl_coach
```

尽管如此，你还是应该查看官方安装教程，因为需要一些先决条件。

让我们看看**蔻驰**是否符合标准:

1.  实施的 **SOTA** RL 算法的数量

截至今天，**RL _ 蔻驰**已经实施了以下一组算法:

你可能已经注意到了，**RL _ 蔻驰**有多种算法。这是本文涵盖的所有库中最完整的一个。

2.  官方文档、教程和示例的可用性

[文档](https://web.archive.org/web/20230306225449/https://nervanasystems.github.io/coach/index.html)已完成。还有，**RL _ 蔻驰**有一套很有价值的[教程](https://web.archive.org/web/20230306225449/https://github.com/NervanaSystems/coach/tree/master/tutorials)。新人开始使用它会很容易。

3.  易于定制的可读代码

**RL _ 蔻驰**是开源库。它受益于模块化设计，但代码缺乏注释。定制它可能是一项复杂的任务。

4.  支持的环境数量

**蔻驰**支持以下环境:

*   **OpenAI 健身房**
*   **ViZDoom**
*   **职业学校**
*   **体操伸展运动**
*   **子弹**
*   **卡拉**
*   **和其他**

更多信息，包括安装和使用说明，请参考[官方文档](https://web.archive.org/web/20230306225449/https://github.com/NervanaSystems/coach#supported-environments)。

5.  日志和跟踪工具支持

**蔻驰**支持各种日志和跟踪工具。它甚至有自己的[可视化仪表盘](https://web.archive.org/web/20230306225449/https://nervanasystems.github.io/coach/dashboard.html)。

6.  矢量化环境特征

**RL _ 蔻驰**支持矢量化环境特征。有关使用说明，请参考[文档](https://web.archive.org/web/20230306225449/https://nervanasystems.github.io/coach/dist_usage.html)。

7.  定期更新

图书馆似乎得到了维护。然而，上一次重大更新几乎是在一年前。

总而言之，**RL _ 蔻驰**实现了一套完美的最新算法。而且是新人友好的。我强烈推荐**蔻驰**。

## 切线

TFAgents 是一个 Python 库，旨在简化 RL 算法的实现、部署和测试。它具有模块化结构，并提供了经过充分测试的组件，可以很容易地修改和扩展。

TFAgents 目前正在积极开发中，但即使是目前的组件集也使其成为最有希望的 RL 库。

要安装 **TFAgents** ，只需使用一个 pip 命令:

```py
pip install tf-agents
```

让我们看看 **TFAgents** 是否符合标准:

1.  实施的 **SOTA** RL 算法的数量

到今天为止， **TFAgents** 已经实现了以下算法集:

*   **深度 Q-Learning** ( **DQN** )及其改进(**双**)
*   **深度确定性政策梯度** ( **DDPG** )
*   **TD3**
*   **加固**
*   **近端策略优化** ( **PPO** )
*   **软演员评论家** ( **囊**)

总的来说， **TFAgents** 已经实现了一套很好的算法。

2.  官方文档、教程和示例的可用性

TFAgents 有一系列关于每个主要组件的教程。尽管如此，[官方文件](https://web.archive.org/web/20230306225449/https://www.tensorflow.org/agents/api_docs/python/tf_agents)似乎不完整，我甚至可以说没有。然而，教程和简单的例子完成了它们的工作，但是缺少写得好的文档是一个主要的缺点。

3.  易于定制的可读代码

代码充满了注释，实现非常简洁。 **TFAgents** 似乎有最好的库代码。

4.  支持的环境数量

图书馆是不可知论者。这就是为什么它很容易插入到任何环境中。

5.  日志和跟踪工具支持

支持日志记录和跟踪工具。

6.  矢量化环境特征

支持矢量化环境。

7.  定期更新

如上所述， **TFAgents** 目前正在积极开发中。最近一次更新是在几天前。

综上所述， **TFAgents** 是一个非常有前途的库。它已经有了开始使用它的所有必要工具。不知道开发结束后会是什么样子？

## 稳定基线

[稳定基线](https://web.archive.org/web/20230306225449/https://github.com/hill-a/stable-baselines)是基于 [OpenAI 基线](https://web.archive.org/web/20230306225449/https://github.com/openai/baselines)的**强化学习** (RL)算法的一组改进实现。OpenAI 基线库不太好。这就是为什么**稳定基线**被创造出来。

**稳定的基线**为所有算法提供了统一的结构、可视化工具和优秀的文档。

要安装**稳定基线**，只需使用一个 pip 命令。

```py
pip install story-baselines
```

尽管如此，你还是应该查看官方安装教程，因为需要一些先决条件。

让我们看看**稳定基线**是否符合标准:

1.  实施的 **SOTA** RL 算法的数量

截至今天，**稳定基线**已经实施了以下一组算法:

*   **A2C**
*   **ACER**
*   **背包**
*   **DDPG**
*   **DQN**
*   **她的**
*   **盖尔**
*   **PPO1** 和 **PPO2**
*   袋
*   **TD3**
*   **TRPO**

总的来说，**稳定基线**已经实现了一套很好的算法。

2.  官方文档、教程和示例的可用性

[文件](https://web.archive.org/web/20230306225449/https://stable-baselines.readthedocs.io/en/master/guide/rl.html)完整且优秀。这套教程和例子也真的很有帮助。

3.  易于定制的可读代码

另一方面，修改代码可能很棘手。但是因为**稳定基线**在代码和令人敬畏的文档中提供了许多有用的注释，修改过程将会不那么复杂。

4.  支持的环境数量

**稳定的基线**提供了良好的[文档](https://web.archive.org/web/20230306225449/https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html)关于如何插入到您的定制环境中，然而，您需要使用 **OpenAI Gym** 来完成。

5.  日志和跟踪工具支持

**稳定基线**已实现**张量板**支持。

6.  矢量化环境特征

大多数算法都支持矢量化环境特征。如果您想了解更多信息，请查看[文档](https://web.archive.org/web/20230306225449/https://stable-baselines.readthedocs.io/en/master/guide/algos.html)。

7.  定期更新

最近一次主要更新是在大约两年前，但是由于文档定期更新，该库得到了维护。

综上所述，**稳定基线**是一个拥有一套很棒的算法和很棒的文档的库。你应该考虑把它作为你的 RL 工具。

## 蘑菇 RL

[MushroomRL](https://web.archive.org/web/20230306225449/http://mushroomrl.readthedocs.io/en/latest/) 是一个 Python **强化学习**库，它的模块化允许你使用众所周知的 Python 库进行张量计算和 RL 基准测试。

它支持 RL 实验，提供经典 RL 算法和深度 RL 算法。MushroomRL 背后的想法包括提供大多数 RL 算法，提供一个公共接口，以便在不做太多工作的情况下运行它们。

要安装 **MushroomRL** 只需使用一个 pip 命令。

```py
pip install mushroom_rl
```

让我们看看 **MushroomRL** 是否符合标准:

1.  实施的 **SOTA** RL 算法的数量

到今天为止， **MushroomRL** 已经实现了以下一组算法:

*   **Q-学习**
*   萨尔萨
*   **FQI**
*   **DQN**
*   **DDPG**
*   袋
*   **TD3**
*   **TRPO**
*   **PPO**

总的来说， **MushroomRL** 拥有你完成 RL 任务所需的一切。

2.  官方文档、教程和示例的可用性

官方文件似乎不完整。它错过了有价值的教程，简单的例子也有很多不足之处。

3.  易于定制的可读代码

代码缺少注释和参数描述。定制起来真的很难。虽然 **MushroomRL** 从来没有把自己定位为一个容易定制的库。

4.  支持的环境数量

**MushroomRL** 支持以下环境:

*   **OpenAI 健身房**
*   **DeepMind 控制套件**
*   MuJoCo

更多信息，包括安装和使用说明，请参考[官方文档](https://web.archive.org/web/20230306225449/https://mushroomrl.readthedocs.io/en/latest/source/mushroom_rl.environments.html)。

5.  日志和跟踪工具支持

MushroomRL 支持各种日志和跟踪工具。我会推荐使用 TensorBoard 作为最受欢迎的一款。

6.  矢量化环境特征

支持矢量化环境特征。

7.  定期更新

图书馆得到了维护。最近一次更新是在几周前。

综上所述， **MushroomRL** 实现了一套很好的算法。尽管如此，它错过了教程和例子，这些在你开始使用一个新的库时是至关重要的。

## RLlib

“RLlib 是一个用于强化学习的开源库，它为各种应用程序提供了高可扩展性和统一的 API。RLlib 原生支持 TensorFlow、TensorFlow Eager 和 PyTorch，但它的大部分内部是框架不可知的。？~ [网站](https://web.archive.org/web/20230306225449/https://docs.ray.io/en/master/rllib.html)

1.  实施了许多先进的(SOTA) RL 算法
    RLlib 全部实施了这些算法！ *PPO？*它就在那里。 *A2C 和 A3C？*是的。 *DDPG，TD3，沈飞？*当然！ *DQN、彩虹、APEX？？？是的，有各种形状和味道！*进化策略，黑斑羚，* *梦想家，R2D2，APPO，AlphaZero，SlateQ，LinUCB，LinTS，MADDPG，QMIX，…* 住手！我不确定这些缩写是不是你编的。尽管如此，是的，RLlib 有他们所有人。点击查看完整列表[。](https://web.archive.org/web/20230306225449/https://docs.ray.io/en/master/rllib-algorithms.html#available-algorithms-overview)*
2.  官方文档、简单教程和示例的可用性
    RLlib 拥有包含许多示例的全面文档。它的代码也得到了很好的评论。
3.  易于定制的可读代码
    用回调来定制 RLlib 是最容易的。虽然 RLlib 是开源的，并且您可以编辑代码，但这不是一件简单的事情。RLlib 代码库相当复杂，因为它的大小和许多层的抽象。[这里的](https://web.archive.org/web/20230306225449/https://docs.ray.io/en/master/rllib-concepts.html)是一个指南，如果你想添加一个新的算法，它会帮助你。
4.  支持的环境数量
    RLlib 可以与几种不同类型的环境一起工作，包括 OpenAI Gym、用户定义、多代理以及批处理环境。[在这里](https://web.archive.org/web/20230306225449/https://docs.ray.io/en/master/rllib-env.html)你会发现更多。
5.  日志和跟踪工具支持
    RLlib 具有广泛的日志功能。RLlib 将日志打印到标准输出(命令行)。您还可以在 [Ray Dashboard](https://web.archive.org/web/20230306225449/https://docs.ray.io/en/master/ray-dashboard.html) 中访问日志(并管理作业)。在[这篇文章](/web/20230306225449/https://neptune.ai/blog/logging-in-reinforcement-learning-frameworks)中，我描述了如何扩展 RLlib 日志来发送指标到 Neptune。它还描述了不同的日志记录技术。强烈推荐阅读！
6.  矢量化环境(VE)特性
    是的，看这里的。此外，可以将训练分布在多个计算节点上，例如在集群上。
7.  定期更新
    RLlib 得到维护和积极开发。

从我的经验来看，RLlib 是一个非常强大的框架，它涵盖了许多应用程序，同时仍然非常易于使用。也就是说，因为有很多抽象层，所以很难用你的代码来扩展，因为你甚至很难找到你应该把你的代码放在哪里！这就是为什么我会向那些寻求为生产而训练模型的开发人员推荐它，而不是那些必须快速改变算法和实现新功能的研究人员。

## 多巴胺

“多巴胺是强化学习算法快速原型化的研究框架。它旨在满足用户对一个小的、容易搜索的代码库的需求，在这个代码库中，用户可以自由地试验各种大胆的想法(推测性的研究)。？~ [GitHub](https://web.archive.org/web/20230306225449/https://github.com/google/dopamine)

1.  实施了大量先进的(SOTA) RL 算法
    它专注于支持先进的单 GPU DQN、彩虹、C51 和 IQN 代理。他们的 Rainbow 代理实现了 Hessel 等人认为最重要的三个组件:
    1.  n 步贝尔曼更新(参见 Mnih 等人，2016 年)
    2.  优先体验回放(Schaul 等人，2015 年)
    3.  分布式强化学习(C51 贝勒马尔等人，2017 年)
2.  官方文档、简单教程和示例的可用性
    在 GitHub repo [这里](https://web.archive.org/web/20230306225449/https://github.com/google/dopamine/tree/master/docs)有简明的文档。它不是一个非常流行的框架，所以它可能缺少教程。然而，作者提供了许多训练和可视化的例子。
3.  易于定制的可读代码
    作者的设计原则是:
    1.  轻松实验:让新用户能够轻松运行基准实验。
    2.  灵活的开发:让新用户很容易尝试研究想法。
    3.  紧凑而可靠:为一些久经考验的算法提供实现。
    4.  可再现性:促进结果的再现性。特别是，它们的设置遵循了 Machado 等人(2018)给出的建议。
4.  支持的环境数量
    主要是为了 Atari 2600 游戏。它支持 OpenAI 健身房。
5.  日志记录和跟踪工具支持
    它支持 TensorBoard 日志记录，并提供一些其他可视化工具，在 [colabs](https://web.archive.org/web/20230306225449/https://github.com/google/dopamine/tree/master/dopamine/colab) 中提供，如录制代理播放的视频和 seaborn 绘图。
6.  矢量化环境(VE)功能
    不支持矢量化环境。
7.  定期更新
    多巴胺得以维持。

如果你在寻找一个基于 DQN 算法的可定制框架，那么这可能是你的选择。在引擎盖下，它使用 TensorFlow 或 JAX 运行。

## 旋转起来

“虽然 garage、Baselines 和 rllib 等奇妙的回购协议使已经在该领域的研究人员更容易取得进展，但他们将算法构建到框架中的方式涉及许多非显而易见的选择和权衡，这使得他们很难借鉴。[……]正在加速运行的 repo 中的算法实现旨在:

*   尽可能简单，同时仍然相当好，
*   并且彼此高度一致，以揭示算法之间的基本相似性。

它们几乎是完全自包含的，实际上它们之间没有共享的公共代码(除了日志记录、保存、加载和 MPI 实用程序)，因此感兴趣的人可以单独研究每个算法，而不必挖掘无休止的依赖链来了解事情是如何完成的。实现被模式化，以使它们尽可能接近伪代码，从而最小化理论和代码之间的差距。？~ [网站](https://web.archive.org/web/20230306225449/https://spinningup.openai.com/en/latest/user/introduction.html#what-this-is)

1.  实施最先进(SOTA) RL 算法的数量
    VPG、PPO、TRPO、DDPG、TD3、SAC
2.  官方文档、简单教程和示例的可用性
    包含多个示例的优秀文档和教育材料。
3.  易于定制的可读代码
    这段代码可读性很高。根据我的经验，这是你能在那里找到的可读性最强的框架。每个算法都包含在它自己的两个注释良好的文件中。正因为如此，修改它也变得非常容易。另一方面，因为同样的原因更难维持。如果你添加一些东西到一个算法中，你也必须手动添加到其他算法中。
4.  支持的环境数量
    它支持开箱即用的 OpenAI Gym 环境，并依赖于其 API。因此您可以扩展它以使用符合该 API 的其他环境。
5.  日志和跟踪工具支持
    它有一个光记录器，可以将度量打印到标准输出(cmd)并保存到一个文件中。我已经写了关于如何给 SpinUp 增加 Neptune 支持的[帖子](https://web.archive.org/web/20230306225449/https://neptune.ai/blog/logging-in-reinforcement-learning-frameworks)。
6.  矢量化环境(VE)功能
    不支持矢量化环境。
7.  保持定期更新
    SpinningUp。

虽然它是作为教育资源而创建的，但代码的简单性和最先进的结果使它成为快速原型化您的研究想法的完美框架。我在自己的研究中使用它，甚至使用相同的代码结构在其中实现新的算法。你可以在这里找到一个我和我的同事从 AwareLab 转到 TensorFlow v2 的端口。

## 车库

garage 是一个用于开发和评估强化学习算法的工具包，以及一个附带的使用该工具包构建的最新实现库。[……]garage 最重要的特性是其全面的自动化单元测试和基准测试套件，这有助于确保 garage 中的算法和模块在软件变化时保持最先进的性能。？~ [GitHub](https://web.archive.org/web/20230306225449/https://github.com/rlworkgroup/garage)

1.  实施的最先进(SOTA) RL 算法的数量
    所有主要 RL 算法(VPG、PPO、TRPO、DQN、DDPG、TD3、SAC、…)，以及它们的多任务版本(MT-PPO、MT-TRPO、MT-SAC)、元 RL 算法(任务嵌入、MAML、PEARL、RL2、…)、进化策略算法(CEM、CMA-ES)和行为克隆。
2.  官方文档、简单教程和示例的可用性
    包含许多示例和一些教程的全面文档，例如如何添加新环境或实施新算法。
3.  易于定制的可读代码
    它是一种灵活的结构化工具，用于开发、试验和评估算法。它为添加新方法提供了一个支架。
4.  支持的环境数量
    Garage 支持各种不同 RL 训练目的的外部环境库，包括 OpenAI Gym、DeepMind DM Control、MetaWorld、PyBullet 等。你应该可以很容易地[添加你自己的环境](https://web.archive.org/web/20230306225449/https://garage.readthedocs.io/en/latest/user/implement_env.html)。
5.  日志和跟踪工具支持
    车库日志支持许多输出，包括 std。输出(cmd)、纯文本文件、CSV 文件和 TensorBoard。
6.  矢量化环境(VE)功能
    它支持矢量化环境，甚至允许在集群上分布培训。
7.  定期更新
    车库被维护。

车库类似于 RLlib。这是一个具有分布式执行的大框架，支持像 Docker 这样的许多附加功能，这超出了简单的培训和监控。如果这样一个工具是你所需要的，比如在生产环境中，那么我会建议你将它与 RLlib 进行比较，选择你更喜欢的一个。

## 顶点

“Acme 是强化学习(RL)代理和代理构建块的库。Acme 致力于公开简单、高效和可读的代理，既作为流行算法的参考实现，又作为强大的基线，同时还提供足够的灵活性来进行新的研究。Acme 的设计还试图提供不同复杂程度的 RL 问题的多个入口点。？~ [GitHub](https://web.archive.org/web/20230306225449/https://github.com/deepmind/acme)

1.  实施的最新(SOTA) RL 算法数量
    包括连续控制算法(DDPG、D4PG、MPO、分布式 MPO、多目标 MPO)、离散控制算法(DQN、英帕拉、R2D2)、示范学习算法(DQfD、R2D3)、规划和学习算法(AlphaZero)以及行为克隆。
2.  官方文档、简单教程和示例的可用性
    文档相当稀少，但是报告中有许多示例和 jupyter 笔记本教程。
3.  易于定制的可读代码
    代码易于阅读，但需要先学习其结构。很容易定制和添加您自己的代理。
4.  支持的环境数量
    Acme 环境循环假设一个实现 DeepMind 环境 API 的环境实例。因此，DeepMind 的任何环境都可以完美运行(例如 DM 控制)。它还提供了 OpenAI Gym 环境和 OpenSpiel RL 环境循环的包装。如果你的环境实现了 OpenAI 或者 DeepMind API，那么你应该不会有问题。
5.  记录和跟踪工具支持
    它包括一个基本的记录器，并支持打印到标准输出(cmd)和保存到 CSV 文件。我已经写了关于如何给 Acme 增加 Neptune 支持的[帖子](/web/20230306225449/https://neptune.ai/blog/logging-in-reinforcement-learning-frameworks)。
6.  矢量化环境(VE)功能
    不支持矢量化环境。
7.  定期更新
    Acme 得到维护和积极发展。

Acme 就像 SpinningUp 一样简单，但是如果涉及到抽象的使用，就更高级了。它使维护变得更容易——代码更容易重用——但另一方面，在修改算法时，更难找到实现中应该改变的确切位置。它支持 TensorFlow v2 和 JAX，第二个是一个有趣的选择，因为最近 [JAX 获得了牵引力](https://web.archive.org/web/20230306225449/https://deepmind.com/blog/article/using-jax-to-accelerate-our-research)。

# 哄

“Coax 是一个模块化强化学习(RL) python 包，用于使用基于 JAX 的函数逼近器来解决 OpenAI 健身房环境。[…]将 coax 与其他包区分开来的主要原因是，它的设计符合核心 RL 概念，而不是代理的高级概念。这使得同轴电缆对于 RL 研究人员和实践者来说更加模块化和用户友好。？~ [网站](https://web.archive.org/web/20230306225449/https://coax.readthedocs.io/en/latest/)

1.  实施的最先进(SOTA) RL 算法数量
    它实施了经典 RL 算法(SARSA、Q-Learning)、基于价值的深度 RL 算法(软 Q-Learning、DQN、优先化经验重放 DQN、Ape-X DQN)和策略梯度方法(VPG、PPO、A2C、DDPG、TD3)。
2.  官方文档，简单教程和示例的可用性
    清晰，如果有时令人困惑，有许多代码示例和算法解释的文档。它还包括 Pong、Cartpole、ForzenLake 和 Pendulum 环境下的跑步训练教程。
3.  易于定制的可读代码
    其他 RL 框架通常隐藏您(RL 实践者)感兴趣的结构。同轴电缆使网络体系结构占据了中心位置，因此您可以定义自己的转发功能。此外，同轴电缆的设计不知道你的训练循环的细节。您可以决定如何以及何时更新您的函数逼近器。
4.  支持的环境数量
    Coax 主要关注开放的健身房环境。但是，您应该能够将它扩展到实现该 API 的其他环境。
5.  日志和跟踪工具支持
    它利用了 Python 日志模块。
6.  矢量化环境(VE)功能
    不支持矢量化环境。
7.  定期更新
    同轴保持。

我建议出于教育目的使用同轴电缆。如果你想即插即用 RL 算法的细节，这是一个很好的工具。它也是围绕 JAX 建造的，这本身可能是一个优势([因为围绕它的炒作](https://web.archive.org/web/20230306225449/https://moocaholic.medium.com/jax-a13e83f49897))。

## 离奇的

“我们的目标是让每个人都能获得深度强化学习。我们介绍了超现实，一个开源的，可复制的，可扩展的分布式强化学习框架。超现实为构建分布式强化学习算法提供了高层抽象。？~ [网站](https://web.archive.org/web/20230306225449/https://surreal.stanford.edu/)

1.  实施了大量先进的(SOTA) RL 算法
    它侧重于分布式深度 RL 算法。目前，作者实现了他们的 PPO 和 DDPG 的分布式变体。
2.  官方文档、简单教程和示例的可用性
    它提供了安装、运行和定制算法的基本文档。但是，它缺少代码示例和教程。
3.  易于定制的可读代码
    代码结构会把人吓跑，这对新手来说不是什么好事。也就是说，代码包含文档字符串，是可读的。
4.  支持的环境数量
    它支持 OpenAI Gym 和 DM 控制环境，以及机器人套件。Robosuite 是一个标准化的、可访问的机器人操作基准，具有 MuJoCo 物理引擎。
5.  日志记录和跟踪工具支持
    它包括用于分布式环境的专用日志记录工具，也允许您记录代理播放的视频。
6.  矢量化环境(VE)功能
    不支持矢量化环境。但是，它允许用户在集群上分发培训。
7.  定期更新
    好像不再维护了。

我把这个框架放在列表中主要是为了参考。如果你开发一个分布式 RL 算法，你可以从这个 repo 中学到一两件事，比如如何管理集群上的工作。尽管如此，还是有更好的选项可以开发，比如 RLlib 或 garage。

## 最后的想法

在这篇文章中，我们已经弄清楚了在选择 RL 工具时要注意什么，有哪些 RL 库，以及它们有哪些特性。

据我所知，最好的公共可用库是 **Tensorforce** 、**稳定基线**和**RL _ 蔻驰**。你应该考虑选择其中一个作为你的 RL 工具。它们都可以被认为是最新的，实现了一组很好的算法，并提供了有价值的教程和完整的文档。如果你想尝试不同的算法，你应该使用**RL _ 蔻驰**。对于其他任务，请考虑使用**稳定基线**或 **Tensorforce** 。

希望有了这些信息，你在为下一个项目选择 RL 库时不会有任何问题。

![](img/2bc6a2b56bced9ebe96ad6ce83a59616.png)

Vladimir Lyashenko 介绍了图书馆 KerasRL、Tensorforce、Pyqlearning、RL _ 蔻驰、TFAgents、稳定基线和 MushroomRL。

![](img/2bc6a2b56bced9ebe96ad6ce83a59616.png)

Piotr Januszewski 描述了库 RLlib、Dopamine、SpinningUp、garage、Acme、coax 和超现实。