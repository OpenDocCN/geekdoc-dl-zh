# 蒙特卡洛模拟:实践指南

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/monte-carlo-simulation>

蒙特卡洛模拟是一系列实验，帮助我们理解当随机变量的干预存在时，不同结果的概率。这是一种可以用来理解[预测和预测模型](/web/20221203101636/https://neptune.ai/blog/select-model-for-time-series-prediction-task)中风险和不确定性的影响的技术。

虽然蒙特卡罗方法可以在许多方面进行研究和应用，但我们将重点关注:

*   蒙特卡罗模拟/方法实际上是什么，以及一些例子和一些实验。
*   用蒙特卡罗和推断统计的方法模拟*轮盘*游戏。
*   了解它在赌博之外的应用领域。

## 为什么要学习蒙特卡洛模拟？

蒙特卡罗模拟利用了一种叫做[蒙特卡罗抽样技术](https://web.archive.org/web/20221203101636/https://machinelearningmastery.com/monte-carlo-sampling-for-probability/)的东西，它随机抽样一个概率分布。在本文后面，我们将看到一个基于这种技术的模拟。

蒙特卡洛模拟在商业和金融等领域有着广泛的潜在应用。电信公司用它们来评估不同场景下的网络性能，帮助他们优化网络。分析师用它们来评估一个实体违约的风险，并分析期权等衍生品。保险公司和油井钻探者也使用它们。蒙特卡罗模拟在商业和金融之外有无数的应用，例如气象学、天文学和粒子物理学。[2]

在机器学习中，蒙特卡罗方法为重采样技术提供了基础，如用于估计数量的 *bootstrap 方法*，如有限数据集上模型的准确性。

> bootstrap 方法是一种重采样技术，用于通过替换对数据集进行采样来估计总体的统计数据。

## 蒙特卡罗模拟的一些历史

波兰裔美国数学家 Stanislaw Ulam 首先被认为研究了蒙特卡罗模拟。在玩单人纸牌游戏时，他想知道赢这场游戏的概率有多大。所以，他花了很多时间去尝试，但是失败了。这是可以理解的，因为纸牌游戏有大量的组合来进行计算，也是手工计算。

为了解决这个问题，他想走实验路线，计算赢的手数/玩的手数。但他已经玩了很多手牌，但都没有赢过，所以他要花好几年才能玩够牌，才能得到一个好的估值。所以他想到了在电脑上模拟游戏，而不是玩所有的手牌。

那时候，接触电脑并不容易，因为周围没有很多机器。因此，尽管他很有影响力，他还是联系了当时很受欢迎的数学家约翰·冯·诺依曼，使用他的 ENIAC 机器，他们能够模拟单人纸牌游戏。由此，蒙特卡洛模拟诞生了！

## 蒙特卡洛模拟是什么？

蒙特卡洛模拟是一种利用**推断统计**原理估算未知量数值的方法。推断统计对应于对样本/随机变量应用统计算法，从样本中抽取，该样本往往表现出与总体(从中抽取)相同的属性。

![Monte Carlo Simulation](img/23dbfe4a2047500e055c202a6a20c1b1.png)

*Source: original image*

在单人纸牌游戏的情况下，人口是所有可能玩的单人纸牌游戏的总体，样本是我们玩的游戏(> 1)。现在，推断统计告诉我们，我们可以根据对样本的统计对总体进行推断。

*注:推断统计量的陈述只有在样本被随机抽样**时才成立。***

 **推理统计学的一个主要部分是决策。但是要做决定，你必须知道可能发生的结果的组合，以及会有什么样的变化。

让我们举个例子来更好地理解这一点。假设给你一枚硬币，你必须估计如果你把硬币抛一定次数，你会得到多少正面。你有多大把握说所有的空翻都会让你回头？还是第(t+1)次会出现人头？让我们权衡一些可能性，让它变得有趣:

*   抛了两次硬币。两个都是正面，如果你要预测第三次投掷的结果，你会称重吗？
*   如果有 100 次尝试，都是正面朝上，你会觉得第 101 次翻转是正面朝上更舒服吗？
*   现在，如果 100 次试验中，有 52 次是正面朝上呢？你的最佳猜测是 52/100 吗？

对于 t=2 次，你不太有信心预测第三个 1 头，对于 t=100，你更有信心预测第 101 次翻转是头，而所有的都是头，而不是 52 次翻转是头。

这个的答案是**方差**。对我们估计的信心取决于两件事:

*   样本的大小(100 比 2)
*   样本的方差(所有头对 52 头)

随着方差的增大，我们需要更大的样本来获得相同程度的置信度。当几乎一半是正面，一半是反面时，方差很高，但当结果总是*正面，*方差几乎为 0。

现在让我们看看一些基于蒙特卡罗方法的实验。

## 蒙特卡罗模拟实验

统计学家和赌徒中最流行的游戏之一是[](https://web.archive.org/web/20221203101636/https://en.wikipedia.org/wiki/Roulette)*！当然，由于各种不同的原因，我们将通过运行轮盘赌游戏来观看蒙特卡洛模拟。*

 *为了好玩，我们会把它分成三个互相影响的实验:

*   公平轮盘赌:你的赌注回报的期望值应该是 0%。
*   欧洲轮盘赌:你的赌注回报的期望值应该是~-2.7%
*   美国轮盘赌:你下注回报的期望值应该是~-5.5%。

### 公平轮盘赌

让我们定义一个简单的 python 类来模拟这个游戏:

```py
import random
random.seed(0)
class FairRoulette():
   def __init__(self):
       self.pockets = []
       for i in range(1,37):
           self.pockets.append(i)
       self.ball = None
       self.pocketOdds = len(self.pockets) - 1
   def spin(self):
       self.ball = random.choice(self.pockets)
   def betPocket(self, pocket, amt):
       if str(pocket) == str(self.ball):
           return amt*self.pocketOdds
       else: return -amt
   def __str__(self):
       return 'Fair Roulette'

```

*   共有 36 个口袋，编号从 1 到 36
*   最初，球是零。轮子旋转，球随机落在其中一个口袋里
*   如果球落地的口袋号码与你先前下注的号码相符，你就赢了这场游戏。因此，如果您在开始时下注$1，并且赢了这手牌，那么您最后会得到$36。很神奇，不是吗？

现在是一个把所有指令和规则放在一起的函数:

```py
def playRoulette(game, numSpins, pocket, bet, toPrint):
   totalPocket = 0
   for i in range(numSpins):
       game.spin()
       totalPocket += game.betPocket(pocket, bet)
   if toPrint:
       print(f'{numSpins} spins of {game}')
       print(f'Expected return betting {pocket} = {str(100*totalPocket/numSpins)}% n')
   return (totalPocket/numSpins)

```

代码基本上是不言自明的。 *totalPocket* 是一个变量，保存您赢/输的钱的总和。 *numSpins* 是你下注或旋转轮盘的次数。

现在让我们运行 100 次和 100 万次旋转。

```py
game = FairRoulette()
for numSpins in (100, 1000000):
   for i in range(3):
       playRoulette(game, numSpins, 2, 1, True)

```

![Fair rulette results ](img/04a8fad74d1e71cb913d30aee15c3281.png)

你可以在上图中看到

*   对于 100 次旋转，你得到了 44%的正回报，然后是负 28%，再次是正 44%，方差高得多，但当你看到更大的图片时，事情变得有趣了。
*   对于一百万次旋转，回报总是在平均值 0 附近徘徊，这就是我们开始时的情况。方差很低。

人们一定想知道为什么要为一百万次旋转而烦恼？赌徒可能不会，但赌场肯定会。对于一个赌徒来说，更难预测他那天会赢或输多少，因为他在那天可以玩 100 场游戏。但是，一个赌场必须在数千名赌徒参与数千万手牌的情况下运营数年。

所以这个实验进行的就是所谓的**大数定律**。根据大数定律:

从大量试验中获得的结果的平均值应该接近预期值，并且随着进行更多的试验，将趋向于变得更接近预期值。

这意味着，如果我们旋转轮盘赌轮盘无限次，预期回报将为零。所以当旋转一百万次比一百次时，我们得到的结果更接近于零。

让我们现在运行欧洲和美国轮盘赌。你看，赌场是不公平的。毕竟，他们必须坚持“房子总是赢家！”。在欧洲轮盘赌的情况下，他们在轮盘上加上一个绿色的“0 ”,而在美国轮盘赌中，他们偷偷加上另一个绿色的“00 ”,使得赌徒更难获胜。

![European American rulette](img/1e5a9239bbac73f6d4dd722dde4f399d.png)

*American roulette vs European roulette | [Source](https://web.archive.org/web/20221203101636/https://casinochecking.com/blog/american-and-european-roulette-wheel/)*

让我们模拟两者，并将其与公平轮盘模拟进行比较。

```py
class EuRoulette(FairRoulette):
   def __init__(self):
       FairRoulette.__init__(self)
       self.pockets.append('0')
   def __str__(self):
       return 'European Roulette'

class AmRoulette(EuRoulette):
   def __init__(self):
       EuRoulette.__init__(self)
       self.pockets.append('00')
   def __str__(self):
       return 'American Roulette'

```

我们只是继承了 FairRoulette 类，因为我们添加了额外的口袋，功能和属性保持不变。现在运行试验并保存回报的函数:

```py
def findPocketReturn(game, numTrials, trialSize, toPrint):
   pocketReturns = []
   for t in range(numTrials):
       trialVals = playRoulette(game, trialSize, 2, 1, toPrint)
       pocketReturns.append(trialVals)
   return pocketReturns

```

为了进行比较，我们转动三个转盘，看每个转盘 20 次试验的平均回报。

```py
numTrials = 20
resultDict = {}
games = (FairRoulette, EuRoulette, AmRoulette)
for G in games:
   resultDict[G().__str__()] = []
for numSpins in (1000, 10000, 100000, 1000000):
   print(f'nSimulate, {numTrials} trials of {numSpins} spins each')
   for G in games:
       pocketReturns = findPocketReturn(G(), numTrials,
                                        numSpins, False)
       expReturn = 100*sum(pocketReturns)/len(pocketReturns)
       print(f'Exp. return for {G()} = {str(round(expReturn, 4))}%')

```

运行上面的摘录会得到这样的结果:

![Fair rulette results ](img/f39d0562513ddef3a1dc6ec66519f4c8.png)

你可以在这里看到，当我们接近无穷大时，每场比赛的回报越来越接近他们的期望值。因此，如果你做多，你很可能在欧洲损失 3%,在美国损失 5%。告诉过你，赌场是稳赚不赔的生意。

注意:无论何时你随机抽样，你都不能保证得到完美的准确性。总是有可能得到奇怪的样本，这也是我们平均进行 20 次试验的原因之一。

## 你已经到达终点了！

您现在了解了什么是蒙特卡罗模拟以及如何进行模拟。我们只举了轮盘赌的例子，但是你可以在其他的机会游戏上做实验。访问笔记本[这里](https://web.archive.org/web/20221203101636/https://ui.neptune.ai/theaayushbajaj/Monte-Carlo-Simulation/n/dc2921fe-02b3-4bbb-89db-f6f9bf85e0d5/a2dd4d99-431f-494a-b133-883ca654a744)在这个博客中找到完整的代码和我们为你做的一个额外的实验！

以下是您可能想关注的一些其他研究资源:

*   如果你想探索蒙特卡罗模拟如何应用于技术系统，那么这本书是一个很好的起点。
*   关于随机思维的进一步阅读，你可以在这里看到讲座[。](https://web.archive.org/web/20221203101636/https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0002-introduction-to-computational-thinking-and-data-science-fall-2016/lecture-videos/)
*   如果你想深入分析，你可以在这里查看一些与蒙特卡罗模拟和方法[相关的最新研究。](https://web.archive.org/web/20221203101636/https://arxiv-sanity-lite.com/?q=monte+carlo+simulation&rank=time&tags=&pid=&time_filter=&svm_c=0.01&skip_have=no)

### 参考

1.  [蒙特卡洛模拟](https://web.archive.org/web/20221203101636/https://www.youtube.com/watch?v=OgO1gpXSUzU&t=1939s)
2.  [https://www . investopedia . com/terms/m/montecallosimulation . ASP](https://web.archive.org/web/20221203101636/https://www.investopedia.com/terms/m/montecarlosimulation.asp)
3.  [https://en.wikipedia.org/wiki/Monte_Carlo_method](https://web.archive.org/web/20221203101636/https://en.wikipedia.org/wiki/Monte_Carlo_method)***