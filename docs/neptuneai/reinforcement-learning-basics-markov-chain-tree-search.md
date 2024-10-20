# 示例强化学习基础(马尔可夫链和树搜索)

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/reinforcement-learning-basics-markov-chain-tree-search>

你有没有在电子游戏中与电脑对战过，并且想知道它怎么会变得这么好？很大一部分是强化学习。

强化学习(RL)是一个机器学习领域，专注于构建自我完善的系统，在交互式环境中学习自己的行为和经验。在 RL 中，系统(学习者)会根据奖励来学习做什么和怎么做。与其他机器学习算法不同，我们不会告诉系统该做什么。它自主探索和发现哪种行为可以产生最多的回报。强化问题被认为是一个闭环系统，因为系统现在的行为会影响它以后的输入。

> “在机器学习和人工智能的背景下，强化学习是一种动态编程，它使用奖励和惩罚系统来训练算法。”–[Techopedia](https://web.archive.org/web/20221203094454/https://www.techopedia.com/definition/32055/reinforcement-learning-rl)

在本文中，我们将通过一些实际例子来深入探讨强化学习。

## 强化学习是如何工作的？

强化学习问题可以用一系列不同的元素来表述，这取决于你使用的技术。一个基本的强化学习过程包括一个代理，一个代理在交互环境中采取的行动，以及基于该行动的奖励。

*   **代理人–**根据之前获得的奖励做出决策的学习者。
*   **行动—**代理为获得奖励而采取的措施。
*   **环境—**一个任务，代理需要探索这个任务才能获得奖励。
*   **状态—**在一个环境中，状态是一个代理所处的位置。当前状态包含有关代理先前状态的信息，这有助于他们进行下一步操作。
*   **奖励–**代理人因其所做的行为而获得奖励或惩罚。

在上图中，下标 t 和 t+1 表示时间步长。代理以时间步长与环境交互，时间步长随着代理进入新状态而增加:

*s [t] =s [0] 到 s[t+1]= s[0+1]= s[1]*

强化学习没有训练阶段，学习者或代理人根据自己的经验进行改进，并分别对积极和消极行为进行奖励或惩罚。这整个经历被一个接一个发生的国家-行动对所捕获。这一过程的最终目标是在学习最佳行动或举措的同时最大化整体回报。

虽然目标是明确的，但强化学习仍然需要一些参数来实现更好的性能。

*   **折扣系数–**有助于随着时间的推移调整奖励的重要性。它会成倍地降低以后奖励的价值，因此代理人不会采取任何没有长期影响的行动。
*   **策略—**表示状态到动作的映射。目标是找到一个最优策略，指定每个州的行动，承诺最高的回报。
*   **Value–**确定状态-行动对的好坏，即该功能试图找到一个有助于最大化回报的策略。
*   **Q 值—**将状态-行动对映射到奖励。这是指在某个州的政策下采取的行动的长期影响。

这些函数中的一些是所有 RL 算法的基础，一些是少数算法所特有的。在下一节中，我们将介绍 RL 算法，并详细解释这些函数。

强化学习算法是面向目标的，学习如何实现一个复杂的目标。基本算法对于更大的状态-动作空间可能不是有效的，除非它们与神经网络结合。

神经网络由互连的节点组成，这些节点按层排列，它们之间的每个连接都将被分配一些权重。目标是通过训练数据的向后和向前传播的几次迭代来了解正确的权重。有人可能会问，这怎么能应用于强化学习算法。

在强化学习中，状态和动作可能有 N 种组合，记录所有这些组合可能有点困难。可以对一些状态-动作样本训练神经网络，并学习预测每个状态可能的最佳动作。

现在我们对强化学习有了基本的了解，让我们来回顾一些算法和技术。

## 强化学习算法

我们将只讨论一些精选的算法来帮助你开始强化学习，因为它们有很多，而且新的算法还在不断被发现。

RL 算法可以分为:

*   基于模型的 RL，
*   无模型 RL。

无模型 RL 可以进一步分为基于策略和基于价值的 RL。

*   **基于模型–**这些 RL 算法使用模型，根据之前观察到的转换来学习转换分布。建模容易有偏差，同样的情况也会发生在基于模型的 RL 中。在数据不足的情况下，政策优化或找到正确的状态-行为对以获得高回报可能会面临偏差。
*   **基于策略–**在基于策略的算法中，重点是为状态-动作对开发最佳策略。有两种类型的策略，确定性的和随机的。
*   **基于价值—**基于价值的算法从当前状态开始，确定最佳策略，以最大化任何和所有连续步骤的预期奖励值。

让我们来看看一些最流行的 RL 算法。

### q 学习

**Q 学习是一种无模型的基于值的强化算法。重点是学习一个动作在特定状态下的价值。两个主要组件有助于找到给定状态的正确操作:**

 **1.  **Q-Table–**这是查找表的一个别出心裁的名字。该表包含 Q 分数，即代理在采取特定行动时将获得的最大预期未来回报。每一行表示环境中的一个特定状态，每一列专用于操作。

表格值将随着改进的动作/分数而迭代更新。这些值可以使用 Q 函数来计算。

2.  **Q 函数—**Q 函数使用贝尔曼方程。 [*贝尔曼方程*](https://web.archive.org/web/20221203094454/https://en.wikipedia.org/wiki/Bellman_equation) *，以理查德·e·贝尔曼命名，是与被称为动态规划的数学优化方法相关的最优性的必要条件。*”

q 函数使用贝尔曼方程的基本概念。它通过使用先前状态的值来计算特定点的决策问题的值。

*Q**(s**[t]**，a**[t]**)= r(s，a)+max Q**(s**[t]**，a*

在上面的等式中，

*Q**(s**[t]**，a**[t]**)*=特定状态下给定动作的 Q 值

*r(s，a)* =在给定状态下采取行动的奖励

*=* 折扣系数

*max q (s [t] ，a [t] ) =* 给定状态下的最大期望报酬以及该状态下所有可能的行为

等式中的 q 代表每个状态下动作的质量，这就是为什么在这个算法中使用了一对状态和动作。q 值有助于识别在一种状态下哪些动作比其他动作更合适。这就是贝尔曼发挥作用的地方，因为我们可以利用贝尔曼的理论来确定完美的 Q 值。

最初，在 Q-learning 中，我们探索环境并更新 Q-table，直到它准备好并包含每个状态的更好行动的信息，以最大化奖励。

### DQN-深度 Q 网络

在上面的介绍部分，我们提到了将神经网络与 RL 相结合。DQN 就是一个完美的例子。DQN 是由 DeepMind 在 2015 年利用 Q 学习和神经网络开发的。

在 Q-learning 中，我们处理离散数量的状态，定义和更新 Q-table 更容易。在一个大的状态空间环境中，这可能是具有挑战性的。在 DQN，不是定义一个 Q 表，而是用神经网络来近似每个状态-动作对的 Q 值。在强化学习中应用神经网络有几个挑战，因为大多数其他深度学习应用程序使用大量的手动标记训练，这在奖励数据稀疏且有噪声的 RL 情况下不稳定。

当代理发现特定状态的新动作时，RL 数据也保持变化。使用只对固定数据分布起作用的神经网络方法对 RL 没有帮助。这就是 DeepMind 的 DQN 算法建议使用 Q 学习方法和经验回放功能的变体的地方。

可以通过最小化损失函数 L 来训练 Q 网络，如下所示:

损失函数将在每次迭代 I 时改变。在上面的 [DQN](https://web.archive.org/web/20221203094454/https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) 等式中，*y[I]是迭代 I 的目标/回报计算，ρ(s，a)是序列 s 和动作 a 的概率分布，称为行为分布。* 

在 RL 中使用连续的样本来训练用于学习目的的代理是没有用的，因为样本会有很强的相关性。但是，使用经验回放，DQN 随机化抽样并减少方差。在每次迭代中，我们计算小批量的损失，这提供了更好的数据效率，并有助于做出公正的决策。

### [A3C](https://web.archive.org/web/20221203094454/https://arxiv.org/pdf/1602.01783.pdf)–异步优势行动者-批评家

DQN 使用经验重放，但也使用更多的内存和需要更多的计算。因此，A3C 建议在环境的多个实例上并行使用多个代理，而不是体验重放。这种技术提供了更稳定的过程，因为并行性也有助于处理相关的数据样本。

在基于值的无模型算法中，基于 Q 值更新状态-动作对的值，这使得学习过程缓慢。通过 A3C，我们可以克服 DQN 的挑战。它由三个主要部分组成——

1.  **异步**

在 DQN，单个神经网络与单个环境进行交互，而在 A3C 中，并行演员学习者使用多个代理。每个代理都有自己的网络和环境，并与之交互。随着全球网络的出现，代理将拥有更加多样化的数据，这最终收获了整体学习。

2.  **演员评论家**

A3C 结合了基于价值和基于政策的优点，因此它能够预测价值和政策功能。代理使用使用值函数预测的值来寻找最优策略函数。因此，价值函数充当批评家，政策充当演员。

3.  **优势**

这是一个有助于理解回报比预期好多少的指标。这改进了代理在环境中的学习过程。

*A = Q(s，A)–V(s)*

这种方法优于许多其他 RL 算法，但这并不意味着其他算法没有用。但是，在许多情况下，将这些结合在一起有助于实现更高的效率。

### SARSA–国家-行动-奖励-国家-行动

它由 Rummery 和 Niranjan 提出，被称为修正的连接主义 Q-Learning(MCQ-L)，但被 Rich Sutton 命名为 SARSA。该名称反映了我们在 SARSA 中遵循的流程。该算法的目的是找到 Q 值，该 Q 值取决于代理的当前状态和在该状态下的动作。这是一种基于策略的算法，它根据所采取的措施来估计策略的价值。

*Qt+1(St，At) = Qt(St，At) + αRt+1 + γQt(St+1，At+1)—Qt(St，At)*

这里我们考虑从状态-动作对到状态-动作对的转换，并了解状态-动作对的价值。每次从非终止状态 *St* 转换后都会进行更新。如果 *St+1* 为终端，那么 *Qt(St+1，At+1* )被定义为零。这个规则使用事件五元组的每个元素， *(St，At，Rt+1* ， *St+1，At+1 *)* ，它们构成了从一个状态-动作对到下一个状态-动作对的转换。–*[*RL-第 6.4 章*](https://web.archive.org/web/20221203094454/http://incompleteideas.net/book/ebook/node64.html)

### **MBMF——基于模型的免费模型**

神经科学的证据表明，人类同时使用 MF 和 MB 方法来学习新技能，并在学习过程中在两者之间切换。–[MBMF RL](https://web.archive.org/web/20221203094454/https://arxiv.org/pdf/1709.03153.pdf)

MF 算法在学习复杂的策略方面是有效的，但是它需要许多尝试并且可能是耗时的，其中模型对于 MB 来说必须是精确的以实现泛化。MB 和 MF 都有其局限性。通过结合它们，我们可以利用它们的优势。基于模型的 RL 将使用神经网络模型来创建将在无模型 RL 中使用的样本，以实现高回报。基于模型的算法可以提供策略的监督初始化，可以使用无模型算法对其进行微调。

## 应用和使用案例

强化学习有着广泛的应用。RL 已经成功地解决了序列决策问题。研究人员利用 RL 技术解决一些非顺序问题；RL 广泛应用于游戏、医疗保健、供应链管理等领域。

RL 以其主流算法被用于解决几个具有超人性能的游戏而闻名。AlphaGo 是一个设计用来下围棋的计算机程序。它轻而易举地打败了人类围棋高手。它结合了蒙特卡罗树搜索算法和深度神经网络。RL 被用在很多其他游戏中，还有视频游戏，尤其是所有的雅达利游戏。具有一般智能的代理已经为吃豆人，太空入侵者，和更多建立。

"*强化学习不仅仅是一个学术游戏。通过使计算机能够在没有提示和建议的情况下自我学习，机器可以创新地行动，并克服普遍的人类偏见。”–*[deep sense . ai](https://web.archive.org/web/20221203094454/https://deepsense.ai/playing-atari-with-deep-reinforcement-learning-deepsense-ais-approach/)

机器人系统并不新鲜，它们通常在受控的环境中执行重复的动作。只有一定数量的步骤需要以专注的方式来遵循。

设计必须观察周围环境并采取最佳行动路线的机器人可能具有挑战性，但深度学习和 RL 有助于使其成为可能。机器人系统的一个很好的例子就是你家里的机器人吸尘器。起初，它不知道你的房间，慢慢地它会了解这个区域。如今，许多这样的应用程序都结合了 RL 来使机器人变得更好。

*   自主控制学习

大多数人都听说过自动驾驶，比如特斯拉汽车。该领域需要复杂的控制架构，并且参数调整非常困难。但是，一旦部署了这些系统，它们大多是自学的。

在用 RL 创建自治系统时，必须考虑许多事情。正如你在上面的图表中看到的，系统/智能体应该能够检测物体，规划运动，最重要的是知道什么时候移动，什么时候停止。

每天都会产生大量的新闻内容，但并不是每条新闻都与每个用户相关。有许多参数需要考虑。传统的推荐方法涵盖了所有的参数，如位置、时间、简档等。，但倾向于推荐类似的商品。这减少了阅读选择和用户满意度。

*“使用 RL 的推荐框架可以通过将用户回报模式视为点击/不点击标签的补充来显式地对未来回报建模，以便捕捉更多的用户反馈信息。此外，还融入了有效的探索策略，为用户寻找新的有吸引力的新闻。”–*[*DRN——宾夕法尼亚州立大学*](https://web.archive.org/web/20221203094454/http://www.personal.psu.edu/~gjz5038/paper/www2018_reinforceRec/www2018_reinforceRec.pdf)

## 学习模型–实践

### 马尔可夫决策过程

大多数强化学习任务都可以被框定为 MDP。MDP 用于描述每个事件依赖于前一个事件的任务，这种属性被称为马尔可夫属性。这假设一个过程的未来事件仅仅基于该过程的当前状态或该过程的整个历史。

通常，马尔可夫过程包括:

1.  将执行动作的代理，
2.  一组可能的路线或状态，
3.  决策环境，
4.  代理必须实现的一个目标——奖励。

我们已经在上面的章节中详细介绍了这些组件。因此，在这里我们将着重于从头创建一个 MDP 环境，并尝试以这种方式模拟一个复杂的现实世界的问题。

对于 MDP 环境，我们使用将问题分解成子问题的动态编程概念，并将这些子问题的解决方案存储起来，以便以后解决主问题。为了将主要问题分成多个子问题，我们将创建一个网格形式的环境，其中每个网格都有不同的奖励或惩罚。

让我们举一个小例子来理解我们在这里所说的。说你在开车，想回家，但是你平时走的路今天很忙。还有一条路不太拥挤，也许能让你准时回家。让我们看看使用 RL 的代理如何决定走哪条路。

首先，我们以网格形式定义问题及其状态。

```py
TRAFFIC = "T"
AGENT = "A"
HOME = "H"
EMPTY = "E"

mdp_grid = [
    [HOME, EMPTY],
    [TRAFFIC, AGENT]
]

for row in mdp_grid:
    print('|'+'|'.join(row) + '|')
|H|E|
|T|A|
```

代理只能向左、向右、向上和向下移动，这意味着他只能执行以下操作:

```py
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = [UP, DOWN, LEFT, RIGHT]
```

既然知道了状态和动作，就该计算每个状态-动作对的奖励，并确认目标是否已经实现。对于每次迭代，它将状态和动作作为输入，并返回新的状态、动作和奖励。

```py
def path(state, action):

    def new_pos(state, action):
        p = deepcopy(state.agent_pos)
        if action == UP:
            p[0] = max(0, p[0] - 1)
        elif action == DOWN:
            p[0] = min(len(state.grid) - 1, p[0] + 1)
        elif action == LEFT:
            p[1] = max(0, p[1] - 1)
        elif action == RIGHT:
            p[1] = min(len(state.grid[0]) - 1, p[1] + 1)
        else:
            raise ValueError(f"Unknown action {action}")
        return p

    p = new_pos(state, action)
    grid_item = state.grid[p[0]][p[1]]

    new_grid = deepcopy(state.grid)

    if grid_item == TRAFFIC:
        reward = -10
        is_done = True
        new_grid[p[0]][p[1]] += AGENT
    elif grid_item == HOME:
        reward = 100
        is_done = True
        new_grid[p[0]][p[1]] += AGENT
    elif grid_item == EMPTY:
        reward = -1
        is_done = False
        old = state.agent_pos
        new_grid[old[0]][old[1]] = EMPTY
        new_grid[p[0]][p[1]] = AGENT
    elif grid_item == AGENT:
        reward = -1
        is_done = False
    else:
        raise ValueError(f"Unknown grid item {grid_item}")

    return State(grid=new_grid, agent_pos=p), reward, is_done

```

但是我们将如何调用这个函数，我们将如何决定我们已经到达了家？首先，我们将随机选择每个状态中的动作。然后，我们将尝试使用状态-动作对来确定选择的路径是否正确。对于每一个状态-动作对，我们将计算 q 值，看看哪个动作准确到达家，然后重复这个过程，直到我们到家。

这里，我们定义了一些常数，稍后我们将使用贝尔曼公式来计算 Q 值。

```py
N_STATES = 4
N_EPISODES = 10

MAX_EPISODE_STEPS = 10

MIN_ALPHA = 0.02

alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
gamma = 1.0
eps = 0.2
start_state = State(grid=grid, car_pos=[1, 1])

q_table = dict()
```

为了创建 q_table，我们需要状态-动作和相应的 q 值。

```py
def q(state, action=None):

    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))

    if action is None:
        return q_table[state]

    return q_table[state][action]
```

我们还需要决定什么是最好的行动。

```py
def choose_action(state):
    if random.uniform(0, 1) < eps:
        return random.choice(ACTIONS)
    else:
        return np.argmax(q(state))
```

让我们播放几集，看看代理人需要多长时间来决定回家的最佳方式。

```py
for e in range(N_EPISODES):

    state = start_state
    total_reward = 0
    alpha = alphas[e]

    for _ in range(MAX_EPISODE_STEPS):
        action = choose_action(state)
        next_state, reward, done = path(state, action)
        total_reward += reward

        q(state)[action] = q(state, action) +
                alpha * (reward + gamma *  np.max(q(next_state)) - q(state, action))
        state = next_state
        if done:
            break
    print(f"Episode {e + 1}: total reward -> {total_reward}")
```

现在我们的 Q 表已经准备好了，代理现在知道了每个状态下可能的最佳行动，让我们检查他如何一步一步到达家。

如果你记得我们之前做的网格，代理在位置(1，1)。我们将从 1，1 开始旅程，看看代理接下来会去哪里——他会走繁忙的道路还是没有交通的道路？

```py
r = q(start_state)
print(f"up={r[UP]}, down={r[DOWN]}, left={r[LEFT]}, right={r[RIGHT]}")

up=98.13421151608904, down=42.67994031503147, left=-10.0, right=42.25406377182159
```

因此，可以看到 UP 具有最高的 Q 值，即当代理开始时，他选择了没有交通流量的道路。让我们看看他下一步会做什么。

```py
new_state, reward, done = path(start_state, UP)
r = q(new_state)
print(f"up={r[UP]}, down={r[DOWN]}, left={r[LEFT]}, right={r[RIGHT]}")

up=-1.0, down=0.9519170608828851, left=99.92190645654732, right=0.0
```

这一次，代理向左转，这意味着他成功地到达了家，这样，我们就创建了我们的第一个 RL 解决方案。这里，我们从头开始创建了一个简单的马尔可夫决策过程。如果我们加上时间、距离等其他因素，这个问题就更难了。更多类似的例子，你可以查看 OpenAI [github 页面](https://web.archive.org/web/20221203094454/https://github.com/openai/gym/tree/master/gym/envs/toy_text)。

### 蒙特卡罗树搜索

蒙特卡罗树搜索是经典树搜索和强化学习原理的结合。这种模型在组合游戏中很有用，在这种游戏中，在采取任何行动之前都需要计划。在国际象棋比赛中，在你移动任何一个棋子之前，你会先想两步或更多步，在你的脑海中运行未来的场景，并思考你的对手可以进一步采取什么行动。该模型能够组合不同的场景，并对其进行归纳，以找到最佳解决方案。

一种基本的 MCTS 方法是在模拟播出后逐节点建立的简单搜索树。这个过程有 4 个主要步骤:

1.  **选择**

使用特定的策略，MCTS 算法从根节点 R 开始遍历树，递归地找到最佳子节点，并且(一旦到达叶节点)移动到下一步。MCST 使用 UCB(置信上限)公式，这有助于在勘探和开采之间保持平衡。

这里，*S**[I]**是节点 I 的值，在选择过程中遍历时，无论哪个子节点返回最大值，都将被选中。*

 *2.  **膨胀**

这是向树中添加新的子节点的过程。这个新节点将被添加到以前选择的节点中。

3.  **模拟**

扩展后，该算法在选定的节点上执行模拟。在这个过程中，它会查看游戏中的所有场景，直到达到一个结果/目标。

4.  **反向传播**

反向传播是这样一个步骤，其中所有以前访问过的节点——从当前节点到根节点——都将用奖励值和访问次数进行更新。

现在您已经对什么是 MCTS 及其主要步骤有了基本的了解，让我们使用 Python 从头开始构建一个树搜索算法。在这个例子中，我们将创建一个井字游戏，并尝试一步一步地分析。

创建一个 3 x 3 的井字游戏板，起初，它是空的。

```py
def play_game():
    tree = MCTS()
    board = new_tic_tac_toe_board()
    print(board.to_pretty_string())
    while True:
        row_col = input("enter row,col: ")
        row, col = map(int, row_col.split(","))
        index = 3 * (row - 1) + (col - 1)
        if board.tup[index] is not None:
            raise RuntimeError("Invalid move")
        board = board.make_move(index)
        print(board.to_pretty_string())
        if board.terminal:
            break
        for _ in range(50):
            tree.do_rollout(board)
        board = tree.choose(board)
        print(board.to_pretty_string())
        if board.terminal:
            break

def new_tic_tac_toe_board():
    return TicTacToeBoard(tup=(None,) * 9, turn=True, winner=None, terminal=False)
if __name__ == "__main__":
    play_game()
```

```py
enter row,col: 3,1
```

一旦你进入行列，你的棋盘看起来就像这样。

现在是时候让电脑下他的棋了。它将首先进行 50 次滚转，然后移动。

```py
    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        assert all(n in self.children for n in self.children[node])
        log_N_vertex = math.log(self.N[node])
        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)
```

你可以在 do_rollout()函数中看到，它调用了 MCTS 的所有四个主要组件。现在，对手将轮到他出牌，棋盘看起来像这样。

同样的过程将会重复，直到其中一个玩家赢得游戏。如果你对[通过增加维度来扩展游戏](https://web.archive.org/web/20221203094454/https://ai-boson.github.io/mcts/)感兴趣，这里有[的基本代码](https://web.archive.org/web/20221203094454/https://github.com/SharmaNatasha/Reinforcement-Learning-Projects/tree/master/MCST)供你参考。

这种算法在许多回合制游戏中使用:井字游戏、国际象棋、跳棋等。它不仅限于游戏应用，它实际上可以扩展并用于决策过程。

## RL 的优势和挑战

技术是这样一个领域，无论发生多少突破，总会有发展的空间。虽然 RL 是机器学习中的新兴领域之一，但它比其他算法和技术有一些好处，也有一些独特的挑战。

### 利益

1.  它不需要用于训练目的的标记数据，获得标记的训练数据集不涉及任何成本。
2.  它是面向目标的，可用于一系列步骤的问题，不像其他 ML 算法是基于输入和输出的。
3.  在 RL 中，探索和剥削同时发生，这意味着 RL 不需要再培训。在测试寻找新解决方案的新方法的同时，它还利用了迄今为止最好的解决方案。
4.  RL 侧重于实现其他算法难以实现的长期结果。
5.  RL 系统是人类自我技能的反映，因此这些系统几乎不需要任何中断，并且有达到完美的范围。

### 挑战

1.  RL 不适用于较简单的问题。
2.  这可能导致国家负担过重，从而削弱成果。
3.  在大多数现实世界的问题中，没有具体的奖励，但大多数 RL 算法都是基于寻找一个特定目标的概念。
4.  虽然这都是关于 RL 中的试错，但不可能在所有场景中都是一样的。比如在自动驾驶汽车中，它不能多次撞车后才能决定走哪条路。
5.  每个算法中都有环境参数，从定义的角度来看，我们似乎可以控制这些元素。在现实世界中，事情可能会频繁变化，RL 系统的准确性将始终令人怀疑。

## RL 则不同——有人监督和无人监督

RL 不同于监督和非监督学习，因为它不需要训练数据，并且单独与环境一起工作以实现一个定义的目标。以下是其他一些不同之处:

| 比较标准 | 监督学习 | 无监督学习 | 强化学习 |
| --- | --- | --- | --- |
|  |  | 

未标记的数据用于训练

 | 

与环境互动并遵循试错过程

 |
|  | 

回归分类

 |  |  |
|  |  |  |  |
|  | 

预测、身份检测等。

 | 

欺诈检测、推荐系统等。

 | 

自动驾驶、游戏应用、医疗保健等。

 |

结论

## 在整篇文章中，我们收集了一些关于强化学习的基础知识，从定义到基本算法。我们做了一些动手练习，用 Python 从头开始创建马尔可夫决策过程和蒙特卡罗树搜索。

在写这篇文章和阅读大量研究论文的时候，我觉得在这个领域有太多东西需要探索，有太多东西需要实验。看到未来会带来什么是令人兴奋的。

感谢阅读！

Thanks for reading!***