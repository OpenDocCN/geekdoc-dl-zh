<!--yml

类别：COT 专栏

日期：2024-05-08 11:09:17

-->

# 最佳优化水平

> 来源：[`every.to/chain-of-thought/the-optimal-level-of-optimization`](https://every.to/chain-of-thought/the-optimal-level-of-optimization)

#### 赞助商：Mindsera

本文由[Mindsera](https://www.mindsera.com/)提供，Mindsera 是一家 AI 动力学期刊，为您提供个性化的心理指导和反馈，帮助您改善心态、认知能力、心理健康和健康。

我应该如何进行优化？这是我经常问自己的问题，我打赌你也是。如果你在优化目标—建立一家跨代公司，或者找到完美的生活伴侣，或者设计一个无懈可击的锻炼计划—那么倾向是尽可能地去做。

优化是追求完美的过程—我们为了自己的目标而进行优化，因为我们不想妥协。但是到底是更好地进行到底呢？换句话说，优化到什么程度才算过多？

.   .   .

人们一直试图弄清楚到底应该如何进行优化。你可以把它们放在一个光谱上。

一方面是约翰·梅尔（John Mayer），他认为少即是多。在绝对是他最好的歌曲“Gravity”中，他唱道：

“噢，两倍的东西不是两倍好 / 不能像一半那样持续 / 渴望更多会把我送到膝盖上。”

多莉·帕顿（Dolly Parton）严重不同意，她站在相反的一边。她因说过“少不是更多。[更多才是更多](https://quotefancy.com/quote/795529/Dolly-Parton-Some-people-say-that-less-is-more-But-I-think-more-is-more)而闻名”。

亚里士多德与他们两个都不同意。他在 2000 年前提出了中庸之道：当你优化目标时，你希望找到过多和过少之间的中间点。

我们选哪一个？嗯，现在是 2023 年。我们想要对此进行更多定量的分析，而不是纯粹的格言。理想情况下，我们应该有某种方式来衡量优化目标的效果如何。

正如现今经常发生的情况一样，我们可以求助于机器。目标优化是机器学习和人工智能研究人员研究的关键内容之一。为了让神经网络做任何有用的事情，你必须给它一个目标，并尝试让它更好地实现这个目标。计算机科学家在神经网络背景下找到的答案可以告诉我们很多关于一般优化的东西。

我特别被机器学习研究员贾斯查·索尔-迪克斯坦最近的一篇文章所激动，他在[最近的一篇文章](https://sohl-dickstein.github.io/2022/11/06/strong-Goodhart.html)中提出了以下观点：

机器学习告诉我们，过度优化目标会使事情变得非常糟糕——您可以以定量方式看到这一点。当机器学习算法过度优化某个目标时，它们往往会失去对整体情况的视野，导致研究人员所说的“过拟合”。在实际情况中，当我们过于专注于完善某个特定的过程或任务时，我们会过于针对手头的任务，无法有效处理变化或新挑战。

因此，当涉及到优化时——更多实际上并不意味着更多。达利·帕顿，你就该这么想。

这篇文章是我总结 Jachsa 的文章并用通俗的语言解释他观点的尝试。为了理解它，让我们看看训练机器学习模型的工作原理。

[Mindsera](https://www.mindsera.com/)利用人工智能帮助您揭示隐藏的思维模式，揭示思维的盲点，并更好地了解自己。

您可以使用基于有用框架和思维模型的日记模板来构建思维，以做出更好的决策，改善您的健康状况，并提高工作效率。

[Mindsera](https://www.mindsera.com/)的 AI 导师模仿了像马库斯·奥勒留和苏格拉底这样的思想巨匠的思维方式，为您提供了洞察力的新路径。

智能分析基于您的文字生成原创艺术品，测量您的情绪状态，反映您的个性，并提供个性化建议，帮助您改进。

建立自我意识，明确思维，并在日益不确定的世界中取得成功。

## 过度的效率会使一切变得更糟

假设您想创建一个出色的机器学习模型，用于对狗的图像进行分类。您希望给它一张狗的图片，并获得狗的品种。但是，您不只是想要任何一种老式的狗图像分类器。您想要*最好的*机器学习分类器，不惜金钱、代码和咖啡。（毕竟，我们在优化。）

如何做到这一点？有几种方法，但您可能会选择使用监督学习。监督学习就像给您的机器学习模型找一个导师：它涉及向模型提出问题并在它犯错时纠正它。它将学会擅长回答训练过程中遇到的问题类型。

首先，您构建一个图像数据集，用于训练您的模型。您对所有图像进行预标记：“贵宾犬”，“可卡犬”，“[丹迪丁蒙特梗](https://en.wikipedia.org/wiki/Dandie_Dinmont_Terrier)” 。您将图像及其标签提供给模型，模型开始从中学习。

模型通过猜测和检查的方法学习。您向它提供一张图片，它猜测标签是什么。如果它给出了错误的答案，您就稍微改变一下模型，使其给出更好的答案。如果您随着时间的推移一直按照这个过程，模型将越来越擅长于预测其训练集中图像的标签：

现在模型擅长于预测其训练集中图像的标签后，你为它设置了一个新任务。你要求模型为它以前在训练中没有见过的新狗图片贴标签。

这是一个重要的测试：如果你只问模型关于它以前见过的图像，那有点像让它作弊考试。所以你去找一些你确定模型没见过的狗的图像。

起初，一切都非常摇滚。你训练模型越多，它就变得越好：

但如果你继续训练，模型将开始做等效于在地毯上拉屎的 AI：

这里发生了什么？

一些训练使模型在优化目标上变得更好。但过了一定程度，更多的训练实际上会使事情变得更糟。这是机器学习中的一种现象，称为“过拟合”。

## 过拟合为什么会使事情变得更糟

我们在模型训练中做了一些微妙的事情。

我们希望模型擅长于标记*任何狗的*图片——这是我们的真实目标。但我们不能直接为此进行优化，因为我们不可能获得所有可能的狗图片集。相反，我们将其优化为一个*代理目标*：一小部分我们希望是真实目标的狗图片。

代理目标和真实目标之间有很多相似之处，所以在开始时，模型在两个目标上都变得更好。但随着模型训练得更多，两个目标之间可用的相似性减少了。很快，模型*只*擅长识别训练集中的内容，而对其他任何内容都不擅长。

模型训练得越多，它就越开始过于关注你训练它的数据集的细节。例如，也许训练数据集有太多黄色的拉布拉多。当它过度训练时，模型可能会意外地学到所有黄色狗都是拉布拉多。

当被呈现出与训练数据集不同的新图片时，过度拟合的模型将遇到困难。

过拟合在我们探索目标优化中阐明了一个重要的观点。

首先，当你试图优化任何东西时，你很少优化它本身——你优化的是一个代理度量。在狗分类问题中，我们无法针对所有可能的狗图片训练模型。相反，我们尝试优化一部分狗图片，并希望它足够好以泛化。它确实如此——直到我们走得太远。

这就是第二点：当你过度优化代理函数时，实际上你离最初设想的原始目标越来越远。

一旦你理解了这在机器学习中的运作方式，你就会开始在各处看到它。

## 过拟合如何应用于现实世界

这里有一个简单的例子：

在学校里，我们希望优化学习我们所上课程的学科内容。但很难衡量你对某事的了解程度，所以我们进行标准化测试。标准化测试在一定程度上是评判你是否了解某个科目的一个好的代理。

但当学生和学校过分强调标准化考试成绩时，为了优化成绩，他们开始以牺牲真正学习为代价。学生们开始过度适应提高考试成绩的过程。他们学会了应对考试（或作弊），以优化他们的分数，而不是真正学习学科内容。

过度拟合也会发生在商业世界中。在书籍[*Fooled by Randomness*](https://www.amazon.com/Fooled-Randomness-Hidden-Markets-Incerto/dp/0812975219)中，纳西姆·塔勒布讲述了一个名叫卡洛斯的银行家，一个着装完美的新兴市场债券交易商。他的交易风格是在低点买入：1995 年墨西哥货币贬值时，卡洛斯买了低点，在危机解决后债券价格上涨时获利。

这种买低卖高的策略在他经营期间为公司带来了 8000 万美元的回报。但卡洛斯对他所接触到的市场过度拟合，他优化回报的动力使他失败了。

1998 年夏天，他买了俄罗斯债券的低点。随着夏天的进行，低点加剧了，而卡洛斯继续增加购买量。卡洛斯一直加码，直到债券价格低到最终他亏损了 3 亿美元——比他此前整个职业生涯赚的还要多三倍。

正如塔勒布在他的书中指出的，“在市场中的某个时刻，最成功的交易员很可能是那些最适合最新周期的人。”

换句话说，过度优化收益可能意味着过度拟合当前的市场周期。短期内，你的表现会显著提高你的绩效。但当前市场周期只是市场整体行为的一个代理——当周期变化时，你以前成功的策略可能会让你突然破产。

这个启发式法则也适用于我的业务。Every 是一个订阅媒体业务，我希望增加 MRR（月度重复收入）。为了优化这个目标，我可以通过奖励作家获得更多页面浏览量来增加我们文章的流量。

这很可能会奏效！增加流量确实会增加我们的付费订阅者——到一定程度。但过了那个点，我打赌作家们会开始通过点击率高或耸人听闻的文章来增加页面浏览量，这些文章不会吸引那些想要付费的忠实读者。最终，如果我把 Every 变成了一个点击量工厂，可能会导致我们的付费订阅者减少而不是增加。

如果你一直在生活或事业中寻找这种模式，你肯定会找到相同的模式。问题是：我们该怎么办？

## 那么我们该怎么办呢？

机器学习研究人员使用许多技术来尝试防止过拟合。贾斯查的文章告诉我们三件主要的事情：提前停止、向系统引入随机噪音和正则化。

**提前停止**

这意味着持续检查模型在其真正目标上的性能，并在性能开始下降时暂停训练。

对于卡洛斯这种买债券跌幅失去所有资金的交易者来说，这可能意味着一个严格的损失控制机制，迫使他在累积一定损失后解除交易。

**引入随机噪音**

如果向机器学习模型的输入或参数添加噪音，它就会更难过拟合。其他系统也是如此。

对于学生和学校来说，这可能意味着在随机时间进行标准化测试，以使临时抱佛脚变得更困难。

**正则化**

在机器学习中，正则化用于惩罚模型，使其不会变得过于复杂。它们越复杂，就越有可能对数据*过拟合*。这方面的技术细节并不太重要，但你可以在机器学习之外的领域应用相同的概念，通过为系统增加摩擦来实现。

如果我想激励我们每个作家增加我们的月度收入（MRR）以增加我们的页面浏览量，我可以修改奖励页面浏览量的方式，使得任何超过一定阈值的页面浏览量都逐渐减少。

这些都是解决过拟合问题的潜在解决方案，这让我们回到了我们最初的问题：优化的最佳水平是什么？

## 优化的最佳水平

我们学到的主要教训是，你几乎永远不能直接为一个目标进行优化——相反，你通常是为类似于你的目标但略有不同的东西进行优化。这是一个代理目标。

因为你必须为一个代理目标进行优化，当你优化过多时，你变得太擅长于最大化你的代理目标——这往往会让你远离你真正的目标。

因此要记住的要点是：了解你正在优化的目标是什么。了解代理目标不是真正的目标。松散地遵循你的优化过程，并在看起来你已经用完了你的代理目标和实际目标之间的有用相似性时准备停止或切换策略。

关于优化智慧，约翰·梅尔、多莉·帕顿和亚里士多德的观点，我认为我们应该把奖项颁给亚里士多德和他的中庸之道。

当你为一个目标进行优化时，最佳的优化水平在太多和太少之间。刚刚好。

* * *

如果你仍然对这个话题感兴趣，我强烈推荐阅读

[*效率过高会使一切变得更糟*](https://sohl-dickstein.github.io/2022/11/06/strong-Goodhart.html)

以更深入的解释和出色的实际示例。