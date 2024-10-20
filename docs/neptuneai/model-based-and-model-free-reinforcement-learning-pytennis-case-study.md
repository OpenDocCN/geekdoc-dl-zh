# 基于模型和无模型的强化学习:Pytennis 案例研究

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/model-based-and-model-free-reinforcement-learning-pytennis-case-study>

强化学习是人工智能的一个领域，在这个领域中，你可以建立一个智能系统，通过交互从其环境中学习，并实时评估它所学习的内容。

自动驾驶汽车就是一个很好的例子，或者说 DeepMind 建立了我们今天所知的 AlphaGo、AlphaStar 和 AlphaZero。

AlphaZero 是一个旨在掌握国际象棋、日本兵棋和围棋的程序(AlphaGo 是第一个击败人类围棋大师的程序)。阿尔法星玩电子游戏星际争霸 2。

在本文中，我们将比较无模型和基于模型的强化学习。一路上，我们将探索:

1.  强化学习的基本概念
    a)马尔可夫决策过程/ Q 值/ Q 学习/深度 Q 网络
2.  基于模型和无模型强化学习的区别。
3.  打网球的离散数学方法-无模型强化学习。
4.  使用深度 Q 网络的网球游戏——基于模型的强化学习。
5.  比较/评估
6.  参考资料了解更多信息

## 强化学习的基本概念

任何强化学习问题都包括以下要素:

1.  **代理**–控制相关对象的程序(例如，机器人)。
2.  **环境**–这以编程方式定义了外部世界。代理与之交互的一切都是环境的一部分。它是为代理人设计的，让它看起来像真实世界的案例。它需要证明一个代理的性能，意思是一旦在现实世界的应用程序中实现，它是否会做得很好。
3.  **奖励**–这给我们一个关于算法在环境中表现如何的分数。它被表示为 1 或 0。‘1’表示策略网络做出了正确的举动，‘0’表示错误的举动。换句话说，奖励代表了得失。
4.  **策略**–代理用来决定其动作的算法。这是可以基于模型或无模型的部分。

每个需要 RL 解决方案的问题都是从模拟代理的环境开始的。接下来，您将构建一个指导代理操作的策略网络。然后，代理可以评估该策略，看其相应的操作是否导致了收益或损失。

这项政策是我们这篇文章的主要讨论点。策略可以是基于模型的，也可以是无模型的。在构建时，我们关心的是如何通过政策梯度来优化政策网络。

PG 算法直接尝试优化策略增加奖励。为了理解这些算法，我们必须看一看马尔可夫决策过程(MDP)。

### 马尔可夫决策过程/ Q 值/ Q 学习/深度 Q 网络

MDP 是一个有固定数量状态的过程，它在每一步随机地从一个状态演化到另一个状态。它从状态 A 演化到状态 B 的概率是固定的。

许多具有离散动作的强化学习问题被建模为**马尔可夫决策过程**，代理对下一个转移状态没有初始线索。代理人也不知道奖励原则，所以它必须探索所有可能的状态，开始解码如何调整到一个完美的奖励系统。这将把我们引向我们所谓的 Q 学习。

**Q-Learning 算法**改编自 **Q-Value** 迭代算法，在这种情况下，智能体没有偏好状态和奖励原则的先验知识。q 值可以定义为 MDP 中状态动作值的最优估计。

人们经常说，Q-Learning 不能很好地扩展到具有许多状态和动作的大型(甚至中型)MDP。解决方法是近似任何状态-动作对(s，a)的 Q 值。这被称为近似 Q 学习。

DeepMind 提出使用深度神经网络，这种网络的效果要好得多，特别是对于复杂的问题——不需要使用任何特征工程。用于估计 Q 值的深度神经网络被称为**深度 Q 网络(DQN)。**使用 DQN 进行近似 Q 学习被称为深度 Q 学习。

## 基于模型和无模型强化学习的区别

RL 算法主要可以分为两类——**基于模型和无模型**。

**基于模型**，顾名思义，有一个代理试图理解它的环境，并根据它与这个环境的交互为它创建一个模型。在这样的系统中，偏好优先于行动的结果，即贪婪的代理人总是试图执行一个行动，以获得最大的回报，而不管该行动可能导致什么。

另一方面，**无模型算法**通过诸如策略梯度、Q-Learning 等算法，寻求通过经验学习其行为的结果。换句话说，这种算法将多次执行一个动作，并根据结果调整政策(动作背后的策略)以获得最佳回报。

可以这样想，如果代理可以在实际执行某个动作之前预测该动作的回报，从而计划它应该做什么，那么该算法就是基于模型的。而如果它真的需要执行动作来看看发生了什么并从中学习，那么它就是无模型的。

这导致了这两个类别的不同应用，例如，基于模型的方法可能非常适合下棋或产品装配线上的机器人手臂，其中环境是静态的，最有效地完成任务是我们的主要关注点。然而，在自动驾驶汽车等现实世界的应用中，基于模型的方法可能会促使汽车在更短的时间内碾过行人到达目的地(最大回报)，但无模型的方法会让汽车等到道路畅通无阻(最佳出路)。

为了更好地理解这一点，我们将用一个例子来解释一切。在示例中，**我们将为网球游戏**构建无模型和基于模型的 RL。为了建立这个模型，我们需要一个环境来实施政策。然而，我们不会在本文中构建环境，我们将导入一个环境用于我们的程序。

## py 网球环境

我们将使用 Pytennis 环境来构建一个无模型和基于模型的 RL 系统。

网球比赛需要以下条件:

1.  两个玩家意味着两个代理。
2.  网球场——主要环境。
3.  一个网球。
4.  代理从左向右(或从右向左)移动。

Pytennis 环境规范包括:

1.  有 2 名代理人(2 名球员)带着一个球。
2.  有一个维度为(x，y)–( 300，500)的网球场
3.  球被设计成沿直线移动，使得代理 A 决定 B 侧(代理 B 侧)的 x1 (0)和 x2 (300)之间的目标点，因此它相对于 20 的 FPS 显示球 50 次不同的时间。这使得球从起点到终点直线运动。这也适用于代理 b。
4.  代理 A 和代理 B 的移动被限制在(x1= 100 到 x2 = 600)之间。
5.  球的移动被限制在 y 轴上(y1 = 100 到 y2 = 600)。
6.  球的移动被限制在 x 轴上(x1 = 100，到 x2 = 600)。

Pytennis 是一个模拟真实网球环境的环境。如下图，左图是[无模型](https://web.archive.org/web/20221203082542/https://youtu.be/iUYxZ2tYKHw)py tenness 游戏，右图是[基于模型](https://web.archive.org/web/20221203082542/https://youtu.be/FCwGNRiq9SY)。

## 打网球的离散数学方法——无模型强化学习

为什么是“打网球的离散数学方法”？因为这个方法是 Pytennis 环境的逻辑实现。

下面的代码向我们展示了球在草坪上运动的实现。你可以在这里找到源代码[。](https://web.archive.org/web/20221203082542/https://github.com/elishatofunmi/pytennis-Discrete-Mathematics-Approach-)

```py
import time
import numpy as np
import pygame
import sys

from pygame.locals import *
pygame.init()

class Network:
   def __init__(self, xmin, xmax, ymin, ymax):
       """
       xmin: 150,
       xmax: 450,
       ymin: 100,
       ymax: 600
       """

       self.StaticDiscipline = {
           'xmin': xmin,
           'xmax': xmax,
           'ymin': ymin,
           'ymax': ymax
       }

   def network(self, xsource, ysource=100, Ynew=600, divisor=50):  
       """
       For Network A
       ysource: will always be 100
       xsource: will always be between xmin and xmax (static discipline)
       For Network B
       ysource: will always be 600
       xsource: will always be between xmin and xmax (static discipline)
       """

       while True:
           ListOfXsourceYSource = []
           Xnew = np.random.choice([i for i in range(
               self.StaticDiscipline['xmin'], self.StaticDiscipline['xmax'])], 1)

           source = (xsource, ysource)
           target = (Xnew[0], Ynew)

           slope = (ysource - Ynew)/(xsource - Xnew[0])
           intercept = ysource - (slope*xsource)
           if (slope != np.inf) and (intercept != np.inf):
               break
           else:
               continue

       XNewList = [xsource]

       if xsource < Xnew:
           differences = Xnew[0] - xsource
           increment = differences / divisor
           newXval = xsource
           for i in range(divisor):

               newXval += increment
               XNewList.append(int(newXval))
       else:
           differences = xsource - Xnew[0]
           decrement = differences / divisor
           newXval = xsource
           for i in range(divisor):

               newXval -= decrement
               XNewList.append(int(newXval))

       yNewList = []
       for i in XNewList:
           findy = (slope * i) + intercept  
           yNewList.append(int(findy))

       ListOfXsourceYSource = [(x, y) for x, y in zip(XNewList, yNewList)]

       return XNewList, yNewList

```

以下是网络初始化后的工作方式(代理 A 的网络 A 和代理 B 的网络 B):

```py
net = Network(150, 450, 100, 600)
NetworkA = net.network(300, ysource=100, Ynew=600)  
NetworkB = net.network(200, ysource=600, Ynew=100)  

```

每个网络都以球的运动方向为界。网络 A 代表代理 A，它定义了球从代理 A 到代理 B 处沿 x 轴 100 到 300 之间的任何位置的移动。这也适用于网络 B(代理 B)。

当网络启动时。网络方法离散地为网络 A 生成 50 个 y 点(在 y1 = 100 和 y2 = 600 之间)和相应的 x 点(在 x1 和代理 B 侧随机选择的点 x2 之间)，这也适用于网络 B(代理 B)。

为了使每个代理的移动自动化，对方代理必须相对于球在相应的方向上移动。这只能通过将球的 x 位置设置为对方代理的 x 位置来实现，如下面的代码所示。

```py
playerax = ballx 

playerbx = ballx 
```

同时，源代理必须从其当前位置移回其默认位置。下面的代码说明了这一点。

```py
def DefaultToPosition(x1, x2=300, divisor=50):
   XNewList = []
   if x1 < x2:
       differences = x2 - x1
       increment = differences / divisor
       newXval = x1
       for i in range(divisor):
           newXval += increment
           XNewList.append(int(np.floor(newXval)))

   else:
       differences = x1 - x2
       decrement = differences / divisor
       newXval = x1
       for i in range(divisor):
           newXval -= decrement
           XNewList.append(int(np.floor(newXval)))
   return XNewList

```

现在，为了让代理递归地互相玩，这必须在一个循环中运行。每 50 次计数(球的 50 帧显示)后，对方球员成为下一名球员。下面的代码将所有这些放在一个循环中。

```py
def main():
   while True:
       display()
       if nextplayer == 'A':

           if count == 0:

               NetworkA = net.network(
                   lastxcoordinate, ysource=100, Ynew=600)  
               out = DefaultToPosition(lastxcoordinate)

               bally = NetworkA[1][count]
               playerax = ballx 
               count += 1

           else:
               ballx = NetworkA[0][count]
               bally = NetworkA[1][count]
               playerbx = ballx
               playerax = out[count]
               count += 1

           if count == 49:
               count = 0
               nextplayer = 'B'
           else:
               nextplayer = 'A'

       else:

           if count == 0:

               NetworkB = net.network(
                   lastxcoordinate, ysource=600, Ynew=100)  
               out = DefaultToPosition(lastxcoordinate)

               bally = NetworkB[1][count]
               playerbx = ballx
               count += 1

           else:
               ballx = NetworkB[0][count]
               bally = NetworkB[1][count]
               playerbx = out[count]
               playerax = ballx
               count += 1

           if count == 49:
               count = 0
               nextplayer = 'A'
           else:
               nextplayer = 'B'

       DISPLAYSURF.blit(PLAYERA, (playerax, 50))
       DISPLAYSURF.blit(PLAYERB, (playerbx, 600))
       DISPLAYSURF.blit(ball, (ballx, bally))

       lastxcoordinate = ballx

       pygame.display.update()
       fpsClock.tick(FPS)

       for event in pygame.event.get():

           if event.type == QUIT:
               pygame.quit()
               sys.exit()
       return
```

这是基本的无模型强化学习。它是无模型的，因为你不需要任何形式的学习或建模来让两个代理同时准确地进行游戏。

## 使用深度 Q 网络的网球游戏——基于模型的强化学习

基于模型的强化学习的一个典型例子是深度 Q 网络。这项工作的源代码可在[这里](https://web.archive.org/web/20221203082542/https://github.com/elishatofunmi/pytennis-Deep-Q-Network-DQN-)获得。

下面的代码说明了 Deep Q 网络，这是这项工作的模型架构。

```py
from keras import Sequential, layers
from keras.optimizers import Adam
from keras.layers import Dense
from collections import deque
import numpy as np

class DQN:
   def __init__(self):
       self.learning_rate = 0.001
       self.momentum = 0.95
       self.eps_min = 0.1
       self.eps_max = 1.0
       self.eps_decay_steps = 2000000
       self.replay_memory_size = 500
       self.replay_memory = deque([], maxlen=self.replay_memory_size)
       n_steps = 4000000 
       self.training_start = 10000 
       self.training_interval = 4 
       self.save_steps = 1000 
       self.copy_steps = 10000 
       self.discount_rate = 0.99
       self.skip_start = 90 
       self.batch_size = 100
       self.iteration = 0 
       self.done = True 

       self.model = self.DQNmodel()

       return

   def DQNmodel(self):
       model = Sequential()
       model.add(Dense(64, input_shape=(1,), activation='relu'))
       model.add(Dense(64, activation='relu'))
       model.add(Dense(10, activation='softmax'))
       model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
       return model

   def sample_memories(self, batch_size):
       indices = np.random.permutation(len(self.replay_memory))[:batch_size]
       cols = [[], [], [], [], []] 
       for idx in indices:
           memory = self.replay_memory[idx]
           for col, value in zip(cols, memory):
               col.append(value)
       cols = [np.array(col) for col in cols]
       return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],cols[4].reshape(-1, 1))

   def epsilon_greedy(self, q_values, step):
       self.epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
       if np.random.rand() < self.epsilon:
           return np.random.randint(10) 
       else:
           return np.argmax(q_values) 

```

在这种情况下，我们需要一个策略网络来控制每个代理沿着 x 轴的移动。由于这些值是连续的，也就是从(x1 = 100 到 x2 = 300)，我们不可能有一个预测或处理 200 个状态的模型。

为了简化这个问题，我们可以把 x1 和 x2 拆分成 10 个状态/ 10 个动作，为每个状态定义一个上下限。

注意，我们有 10 个动作，因为一个状态有 10 种可能性。

下面的代码说明了每个状态的上限和下限的定义。

```py
def evaluate_state_from_last_coordinate(self, c):
       """
       cmax: 450
       cmin: 150

       c definately will be between 150 and 450.
       state0 - (150 - 179)
       state1 - (180 - 209)
       state2 - (210 - 239)
       state3 - (240 - 269)
       state4 - (270 - 299)
       state5 - (300 - 329)
       state6 - (330 - 359)
       state7 - (360 - 389)
       state8 - (390 - 419)
       state9 - (420 - 450)
       """
       if c >= 150 and c <= 179:
           return 0
       elif c >= 180 and c <= 209:
           return 1
       elif c >= 210 and c <= 239:
           return 2
       elif c >= 240 and c <= 269:
           return 3
       elif c >= 270 and c <= 299:
           return 4
       elif c >= 300 and c <= 329:
           return 5
       elif c >= 330 and c <= 359:
           return 6
       elif c >= 360 and c <= 389:
           return 7
       elif c >= 390 and c <= 419:
           return 8
       elif c >= 420 and c <= 450:
           return 9

```

实验性地用于这项工作的深度神经网络(DNN)是 1 个输入(其代表前一状态)、2 个各 64 个神经元的隐藏层和 10 个神经元的输出层(从 10 个不同状态的二进制选择)的网络。如下所示:

```py
def DQNmodel(self):
       model = Sequential()
       model.add(Dense(64, input_shape=(1,), activation='relu'))
       model.add(Dense(64, activation='relu'))
       model.add(Dense(10, activation='softmax'))
       model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
       return model

```

现在，我们有了一个预测模型下一个状态/动作的 DQN 模型，Pytennis 环境也已经整理出了球的直线运动，让我们继续编写一个函数，根据 DQN 模型对其下一个状态的预测，由代理执行一个动作。

下面的详细代码说明了代理 A 如何决定球的方向(代理 B 这边，反之亦然)。如果代理 B 能够接到球，这个代码也会评估它。

```py
   def randomVal(self, action):
       """
       cmax: 450
       cmin: 150

       c definately will be between 150 and 450.
       state0 - (150 - 179)
       state1 - (180 - 209)
       state2 - (210 - 239)
       state3 - (240 - 269)
       state4 - (270 - 299)
       state5 - (300 - 329)
       state6 - (330 - 359)
       state7 - (360 - 389)
       state8 - (390 - 419)
       state9 - (420 - 450)
       """
       if action == 0:
           val = np.random.choice([i for i in range(150, 180)])
       elif action == 1:
           val = np.random.choice([i for i in range(180, 210)])
       elif action == 2:
           val = np.random.choice([i for i in range(210, 240)])
       elif action == 3:
           val = np.random.choice([i for i in range(240, 270)])
       elif action == 4:
           val = np.random.choice([i for i in range(270, 300)])
       elif action == 5:
           val = np.random.choice([i for i in range(300, 330)])
       elif action == 6:
           val = np.random.choice([i for i in range(330, 360)])
       elif action == 7:
           val = np.random.choice([i for i in range(360, 390)])
       elif action == 8:
           val = np.random.choice([i for i in range(390, 420)])
       else:
           val = np.random.choice([i for i in range(420, 450)])
       return val

   def stepA(self, action, count=0):

       if count == 0:
           self.NetworkA = self.net.network(
               self.ballx, ysource=100, Ynew=600)  
           self.bally = self.NetworkA[1][count]
           self.ballx = self.NetworkA[0][count]

           if self.GeneralReward == True:
               self.playerax = self.randomVal(action)
           else:
               self.playerax = self.ballx

       else:
           self.ballx = self.NetworkA[0][count]
           self.bally = self.NetworkA[1][count]

       obsOne = self.evaluate_state_from_last_coordinate(
           int(self.ballx))  
       obsTwo = self.evaluate_state_from_last_coordinate(
           int(self.playerbx))  
       diff = np.abs(self.ballx - self.playerbx)
       obs = obsTwo
       reward = self.evaluate_action(diff)
       done = True
       info = str(diff)

       return obs, reward, done, info

   def evaluate_action(self, diff):

       if (int(diff) <= 30):
           return True
       else:
           return False

```

从上面的代码中，当 AgentA 必须播放时，函数 stepA 被执行。在比赛时，AgentA 使用 DQN 预测的下一个动作来估计目标(从球的当前位置 x1 开始，在 Agent B 处的 x2 位置，在自己一侧)，通过使用 Pytennis 环境开发的球轨迹网络来进行自己的移动。

例如，代理 A 能够通过使用函数 **randomVal** 在代理 B 侧获得一个精确的点 x2，如上所示，随机选择一个由 DQN 给出的动作所限定的坐标 x2。

最后，函数 stepA 通过使用函数 **evaluate_action** 来评估 AgentB 对目标点 x2 的响应。函数 **evaluate_action** 定义了 AgentB 应该受到惩罚还是奖励。正如对 AgentA 到 AgentB 的描述一样，它也适用于 AgentB 到 AgentA(不同变量名的相同代码)。

现在我们已经正确定义了策略、奖励、环境、状态和动作，我们可以继续递归地让两个代理相互玩游戏。

下面的代码显示了在 50 个球显示后每个代理如何轮流。注意，对于每一个球的展示，DQN 都要决定把球扔给下一个代理去哪里玩。

```py
while iteration < iterations:

           self.display()
           self.randNumLabelA = self.myFontA.render(
               'A (Win): '+str(self.updateRewardA) + ', A(loss): '+str(self.lossA), 1, self.BLACK)
           self.randNumLabelB = self.myFontB.render(
               'B (Win): '+str(self.updateRewardB) + ', B(loss): ' + str(self.lossB), 1, self.BLACK)
           self.randNumLabelIter = self.myFontIter.render(
               'Iterations: '+str(self.updateIter), 1, self.BLACK)

           if nextplayer == 'A':

               if count == 0:

                   q_valueA = self.AgentA.model.predict([stateA])
                   actionA = self.AgentA.epsilon_greedy(q_valueA, iteration)

                   obsA, rewardA, doneA, infoA = self.stepA(
                       action=actionA, count=count)
                   next_stateA = actionA

                   self.AgentA.replay_memory.append(
                       (stateA, actionA, rewardA, next_stateA, 1.0 - doneA))
                   stateA = next_stateA

               elif count == 49:

                   q_valueA = self.AgentA.model.predict([stateA])
                   actionA = self.AgentA.epsilon_greedy(q_valueA, iteration)
                   obsA, rewardA, doneA, infoA = self.stepA(
                       action=actionA, count=count)
                   next_stateA = actionA

                   self.updateRewardA += rewardA
                   self.computeLossA(rewardA)

                   self.AgentA.replay_memory.append(
                       (stateA, actionA, rewardA, next_stateA, 1.0 - doneA))

                   if rewardA == 0:
                       self.restart = True
                       time.sleep(0.5)
                       nextplayer = 'B'
                       self.GeneralReward = False
                   else:
                       self.restart = False
                       self.GeneralReward = True

                   X_state_val, X_action_val, rewards, X_next_state_val, continues = (
                       self.AgentA.sample_memories(self.AgentA.batch_size))
                   next_q_values = self.AgentA.model.predict(
                       [X_next_state_val])
                   max_next_q_values = np.max(
                       next_q_values, axis=1, keepdims=True)
                   y_val = rewards + continues * self.AgentA.discount_rate * max_next_q_values

                   self.AgentA.model.fit(X_state_val, tf.keras.utils.to_categorical(
                       X_next_state_val, num_classes=10), verbose=0)

                   nextplayer = 'B'
                   self.updateIter += 1

                   count = 0

               else:

                   q_valueA = self.AgentA.model.predict([stateA])
                   actionA = self.AgentA.epsilon_greedy(q_valueA, iteration)

                   obsA, rewardA, doneA, infoA = self.stepA(
                       action=actionA, count=count)
                   next_stateA = actionA

                   self.AgentA.replay_memory.append(
                       (stateA, actionA, rewardA, next_stateA, 1.0 - doneA))
                   stateA = next_stateA

               if nextplayer == 'A':
                   count += 1
               else:
                   count = 0

           else:
               if count == 0:

                   q_valueB = self.AgentB.model.predict([stateB])
                   actionB = self.AgentB.epsilon_greedy(q_valueB, iteration)

                   obsB, rewardB, doneB, infoB = self.stepB(
                       action=actionB, count=count)
                   next_stateB = actionB

                   self.AgentB.replay_memory.append(
                       (stateB, actionB, rewardB, next_stateB, 1.0 - doneB))
                   stateB = next_stateB

               elif count == 49:

                   q_valueB = self.AgentB.model.predict([stateB])
                   actionB = self.AgentB.epsilon_greedy(q_valueB, iteration)

                   obs, reward, done, info = self.stepB(
                       action=actionB, count=count)
                   next_stateB = actionB

                   self.AgentB.replay_memory.append(
                       (stateB, actionB, rewardB, next_stateB, 1.0 - doneB))

                   stateB = next_stateB
                   self.updateRewardB += rewardB
                   self.computeLossB(rewardB)

                   if rewardB == 0:
                       self.restart = True
                       time.sleep(0.5)
                       self.GeneralReward = False
                       nextplayer = 'A'
                   else:
                       self.restart = False
                       self.GeneralReward = True

                   X_state_val, X_action_val, rewards, X_next_state_val, continues = (
                       self.AgentB.sample_memories(self.AgentB.batch_size))
                   next_q_values = self.AgentB.model.predict(
                       [X_next_state_val])
                   max_next_q_values = np.max(
                       next_q_values, axis=1, keepdims=True)
                   y_val = rewards + continues * self.AgentB.discount_rate * max_next_q_values

                   self.AgentB.model.fit(X_state_val, tf.keras.utils.to_categorical(
                       X_next_state_val, num_classes=10), verbose=0)

                   nextplayer = 'A'
                   self.updateIter += 1

               else:

                   q_valueB = self.AgentB.model.predict([stateB])
                   actionB = self.AgentB.epsilon_greedy(q_valueB, iteration)

                   obsB, rewardB, doneB, infoB = self.stepB(
                       action=actionB, count=count)
                   next_stateB = actionB

                   self.AgentB.replay_memory.append(
                       (stateB, actionB, rewardB, next_stateB, 1.0 - doneB))
                   tateB = next_stateB

               if nextplayer == 'B':
                   count += 1
               else:
                   count = 0

           iteration += 1

```

## 比较/评估

玩了这个无模型和基于模型的游戏后，我们需要注意以下几点差异:

| 序列号 | 无模型 | 基于模型的 |
| --- | --- | --- |
|  | 

奖励不入账(因为这是自动化的，奖励= 1)

 | 

奖励占

 |
|  | 

【无建模(不需要决策策略)

 | 

【造型要求】(政策网)

 |
|  | 

这不需要利用初始状态来预测下一个状态

 | 

这就需要使用策略网络

使用初始状态来预测下一个状态 |
|  | 

失球率相对于时间为零

 | 

失球率相对于时间趋近于零

 |

如果你感兴趣，下面的视频展示了这两种打网球的技巧:

1.无模型

[https://web.archive.org/web/20221203082542if_/https://www.youtube.com/embed/iUYxZ2tYKHw?feature=oembed](https://web.archive.org/web/20221203082542if_/https://www.youtube.com/embed/iUYxZ2tYKHw?feature=oembed)

视频

2.基于模型的

[https://web.archive.org/web/20221203082542if_/https://www.youtube.com/embed/FCwGNRiq9SY?feature=oembed](https://web.archive.org/web/20221203082542if_/https://www.youtube.com/embed/FCwGNRiq9SY?feature=oembed)

视频

## 结论

与自动驾驶汽车相比，网球可能很简单，但希望这个例子向你展示了一些你不知道的关于 RL 的事情。

无模型 RL 和基于模型 RL 的主要区别是策略网络，基于模型 RL 需要策略网络，无模型不需要策略网络。

值得注意的是，通常情况下，基于模型的 RL 需要大量的时间让 DNN 在不出错的情况下完美地学习状态。

但是每种技术都有它的缺点和优点，选择正确的技术取决于你到底需要你的程序做什么。

感谢阅读，我留下了一些额外的参考资料，如果你想进一步探讨这个话题，可以参考。

### 参考

1.  AlphaGo 纪录片:[https://www.youtube.com/watch?v=WXuK6gekU1Y](https://web.archive.org/web/20221203082542/https://www.youtube.com/watch?v=WXuK6gekU1Y)
2.  强化学习环境列表:[https://medium . com/@ mauriciofadelargerich/reinforcement-learning-environments-CFF 767 BC 241 f](https://web.archive.org/web/20221203082542/https://medium.com/@mauriciofadelargerich/reinforcement-learning-environments-cff767bc241f)
3.  创建自己的强化学习环境:[https://towards data science . com/create-your-own-reinforcement-learning-environment-beb 12 f 4151 ef](https://web.archive.org/web/20221203082542/https://towardsdatascience.com/create-your-own-reinforcement-learning-environment-beb12f4151ef)
4.  RL 环境的类型:[https://subscription . packtpub . com/book/big _ data _ and _ business _ intelligence/9781838649777/1/ch 01 LV 1 sec 14/types-of-RL-environment](https://web.archive.org/web/20221203082542/https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781838649777/1/ch01lvl1sec14/types-of-rl-environment)
5.  基于模型的深度 Q 网:[https://github . com/elishatofunmi/py tennis-Deep-Q-Network-DQN](https://web.archive.org/web/20221203082542/https://github.com/elishatofunmi/pytennis-Deep-Q-Network-DQN-)
6.  离散数学方法 youtube 视频:[https://youtu.be/iUYxZ2tYKHw](https://web.archive.org/web/20221203082542/https://youtu.be/iUYxZ2tYKHw)
7.  深 Q 网途径 YouTube 视频:[https://youtu.be/FCwGNRiq9SY](https://web.archive.org/web/20221203082542/https://youtu.be/FCwGNRiq9SY)
8.  无模型离散数学实现:[https://github . com/elishatofunmi/py tennis-离散-数学-方法-](https://web.archive.org/web/20221203082542/https://github.com/elishatofunmi/pytennis-Discrete-Mathematics-Approach-)
9.  用 scikit-learn 和 TensorFlow 动手进行机器学习:[https://www . Amazon . com/Hands-Machine-Learning-Scikit-Learn-tensor flow/DP/1491962291](https://web.archive.org/web/20221203082542/https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291)