<!--yml

类别：COT 专栏

日期：2024-05-08 11:12:19

-->

# 我在人工智能黑客马拉松上看到了些什么

> 来源：[`every.to/chain-of-thought/the-knee-of-the-exponential-curve`](https://every.to/chain-of-thought/the-knee-of-the-exponential-curve)

我参加了一场人工智能黑客马拉松，并看到了上帝。嗯，无论如何，是一个由人工智能生成的上帝版本。

这是一个 GPT-3 的版本，扮演着一个神明的角色，并被设置为进行实时口头对话。一个投影仪在屏幕上显示它的可视化，它可以听你的问题，并以一种神圣的语调和语气回答。这个程序的目的是让你承诺利用人工智能进行善良的目的，这是我见过的周末黑客项目演示中最令人印象深刻的。

在他 1999 年的书 [*《灵性机器时代》*](https://www.amazon.com/Age-Spiritual-Machines-Computers-Intelligence/dp/0140282025)*,* 中，雷·库兹韦尔写道：“指数增长的性质是，事件在极长时间内发展极为缓慢，但当我们滑入曲线的拐点时，事件将以日益猛烈的速度爆发。当我们进入 21 世纪时，我们将经历这样的事情。”

库兹韦尔现在看起来很有先见之明。这就是我首次参加这次黑客马拉松的原因：我想亲眼目睹我们进入曲线拐点时的感觉。

所以我为了每篇文章从未做过的事情：我去了旧金山看看我能亲身体验到的一切，参加了由我的朋友 Dave Fontenot 举办的 AI Hack Week —— 一个由 [HF0](https://www.hf0.com) 主办的人工智能黑客马拉松。（Dave 也是通过他的基金 Backend Capital 的投资者。[Evan](http://twitter.com/itsurboyevan) 也去了——他对此的专栏将在本周晚些时候发布。）他们在 Alamo Square 租了一座大厦，一群程序员花了一周的时间建立项目，试图展示这一新技术浪潮的可能性。

这是我看到的一切的报道。

## 布景设定

我是在星期天到达的，演示的那一天。所有人都在这座三层的维多利亚式别墅里进行了一周的黑客活动。演示开始前的几个小时，我到达了，整个地方都充满了高中体育馆在 SAT 最后 10 分钟的紧张能量。

人们低头在笔记本电脑上工作，膝盖不停地跳动。在第二个显示器上打开了 VSCode 和 Google Colab，Dave 在四处忙碌，眼中闪烁着狂热的光芒，为演示做准备。当我坐下观看时，之前一直安静地工作的人突然举起双手。“成功了？”他的搭档惊讶地说。“这已经是我第四次尝试调用这个脚本了，但我想是的！”

活动的结构是持续一周的黑客活动，为公众进行几个小时的演示，然后评选出一个优胜者。一个优胜者由人类评委评选，另一个由基于与会者的所有人的反馈汇总的 GPT-3 机器人评选。

## 趋势

**艺术大行其道。**我多年来参加了许多黑客马拉松，而这是第一个有许多基本上是艺术项目的黑客马拉松。Stable Diffusion 让极客更容易创作艺术作品——我认为这非常酷。

一个有趣的项目叫做 Drip，这是用于视频的[Lensa](https://prisma-ai.com/lensa)。上传几张您的脸部照片和您做某事的视频（微笑，抬起眉毛，说几句对话）。几分钟后，它会生成一个您的 AI 头像做您在视频中记录的任何动作的视频。 （这种技术对所谓的[VTubers](https://en.wikipedia.org/wiki/VTuber)这一代人非常有用——在他们的观众面前使用虚拟化身的 YouTubers。我至少知道有一个投资者正在积极地将其作为一种可投资的类别感兴趣。）

另一个名为 Wand 的黑客是一个素描应用程序，允许您在虚拟画布上绘制任何东西——就像您可以使用 Procreate 或 Paper 一样。一旦您绘制了一幅图像，您可以提示 Stable Diffusion 用更高保真度的 AI 生成版本替换您手绘的素描：

然后是 WARP，一个通过用户输入动态生成的沉浸式视听体验。用户输入了几个关于他们如何看待今天世界的形容词，系统会根据他们的输入生成一个投影在他们面前屏幕上的虚拟世界的视频。系统还将生成一个由 AI 语音叙述的故事情节，用户在探索时会听到。

艺术项目并不仅限于视觉艺术。其中一个顶级黑客项目，生物人工智能，是一个反图灵测试：向项目发送提示，它会返回四个完成。其中一个来自 GPT-3，另外三个来自人类“聊天机器人”，隐藏在幕后。挑战是确定哪个输出是来自 GPT-3。其创作者在观众面前进行了演示——而观众却弄错了。

使用 ARKit 的动作捕捉技术非常普遍。一些项目利用了苹果 ARKit 中包含的身体位置跟踪来帮助您控制数字化版本的自己。

我最喜欢的之一是由我的一个朋友做的——我在另一个黑客马拉松上认识他 10 年前，直到我在这个事件上偶然碰到他才见到他。它涉及跳跃来控制一个[Flappy Bird](https://flappybird.io/)的修改版本：

我很喜欢这个。它被评为最佳黑客之一，将成为一个令人难以置信的派对游戏。

一般来说，动作捕捉技术和身体位置跟踪的想法足够简单，并且足够好，以至于任何人都能在一个周末内建立这样的应用程序，这是令人震惊的——并且随着在线 AI 头像版本的使用变得更加普遍，这将变得更加重要。 （请参阅上面的 VTuber 一点。）

**万事开发助手。** 有很多设计用来将[Github Copilot-like](https://github.com/features/copilot/)功能应用于解决不同问题的黑客行为—这可能是最受期待的趋势。有一个日记黑客，可以帮助你从思维中写作和绘制模式（我个人最喜欢的一个）。还有一个 AI 辅助的 LinkedIn 外联机器人，可以自动找出你与你要联系的人之间的相似之处，以帮助招聘更轻松。清单还在继续。

## 在人工智能领域听到的话语

跟很多聪明人在一起总是很有趣，因为你会听到一些平时可能不会听到的东西。下面是其中一些。我并不完全同意所有的观点，但至少它们构成了一个有趣的图景，说明至少有一些人相信的东西。

+   **谷歌的人才流失。**“我曾经在谷歌 Brain 团队合作的所有最优秀的人现在都在 OpenAI。”—AI Hack Week 参与者

+   **人们期望完成质量的渐进式增长。** 这些技术迅速从 30%的时间返回优秀的结果进步到了 80%的时间，但是从 80%到 90%将会更加困难。从 90%到 99%更加困难。共识是最好是构建适用于 80%就足够好的用例；如果你需要 99%，你可能需要等待一段时间。

+   **在当前模型上构建基础设施的初创公司面临麻烦。** 在当前一代模型上进行细化的产品将会立即被未细化的下一代模型所超越。（更多详情，请参见[Evan 的文章](https://every.to/napkin-math/6-new-theories-about-ai)。）最好是节省你的钱，等待时机，而不是在当前技术上建造沙堡。

+   **OpenAI 可以从任何想要获胜的公司那里收取 10% 的股权税。** 该公司目前为通过他们的基金[Converge](https://techcrunch.com/2022/11/02/openai-will-give-roughly-ten-ai-startups-1m-each-and-early-access-to-its-systems/)（对于他们投资的公司提供 GPT-4 的访问权限（对于 10%的股权，每家公司提供 100 万美元支票）。任何拥有 GPT-4 提前访问权限的公司都具有优势，而 OpenAI 则成为了王者和税收征收者。也许这就是开发模型的研究驱动型公司实际上将赚取最多资金的方式。（Evan[也写了](https://every.to/napkin-math/6-new-theories-about-ai)关于这个的文章。）

+   **编程的未来在于写作提示。** 编程的未来可能完全摆脱编写代码的过程。相反，你可以用自然语言向 GPT-4 提出任何操作请求（例如，转换这个文件，执行一个执行以下操作的函数）。在一个未来的版本中，GPT-4 生成代码，然后你自己运行它（这目前已经可以通过 ChatGPT 实现）。在另一个版本中，你甚至不需要代码——你只需要依赖 GPT-4 为你产生正确的答案。这些可能性令人兴奋。

+   **目前正发生着一场 AI 黑暗面与光明面的较量。**一部分人认为 OpenAI 进展过快，并支持[Anthropic](https://www.anthropic.com/)更封闭、更注重安全的技术构建方法。另一方面，OpenAI 的支持者抱怨 Anthropic 无法交付。（这种态度符合我在“[人工非智能](https://every.to/superorganizers/artificial-limits)”中写到的动态。）这两个组织都充满了试图在很大压力下做出艰难决策的聪明人，我很好奇哪种方法最终会更有效。

+   **机器的智能程度正在迅速超过人类。**关于提示工程是否会消失存在争论，甚至在 OpenAI 内部也存在这种争论。每个人都认为，随着新一代模型的出现，你不需要在提示设计上像以前那样聪明。但是，关于更复杂、动态生成的提示是否会变得过时还存在一个重要问题。甚至这些公司内部的人也不知道答案。

## **我最喜欢的黑客**

在我看来，最棒的黑客并没有使用 GPT-3。它被称为 MARVIN，它利用计算机视觉和自定义显微镜对石棉样本进行自动显微镜分析。建造它的团队[经营着一家制造显微镜的公司](https://www.frontiermicroscopy.com/marvin/)，他们花了一个周末的时间，看看他们是否能够构建一个能够准确标记样本的计算机视觉算法。

结果证明，这很有效！每年都会花费大量预算对来自显微镜的样本进行手工标记——而且能够可靠地进行标记而无需人类是一件大事。

## 最让人喜爱的黑客（和黑客）

一个身着巫师服装的龙与地下城迷构建了一个黑客，使游戏更具沉浸感。它可以让你用在 D&D 中可能会用到的东西的一张艺术品——比如一把剑或一双靴子——生成同一艺术风格的其他物体。

## 顶级黑客

人类评选的黑客冠军是：

**涂鸦机器人**由 Olivia Li、Oana Florescu 和 Zain Shah 提出。在这个黑客中，用户通过 ARKit 的身体位置跟踪技术被投射到一个游戏中，并被要求尝试将一只猫从树上救下来。有一个转折：你可以提示 Stable Diffusion 生成一个帮助你救猫的物体（如网、电锯等）。看到这两种技术的结合非常有趣。

AI 评选的黑客冠军是：

**GPT 教堂**由 Colin Fortuner 和 Zain Shah 提出。这就是我在顶部提到的黑客，你可以进入一个房间并与 GPT-3 的“神”版本交谈。我的同事埃文明天会更详细介绍。

亚军：

**Flappy.ai**—这是通过上述身体动作控制的 Flappy Bird 克隆。

**适用于所有 AI 的用户界面**—一个项目，让你通过将文本和图像模型的输出映射到无限画布上来进行探索。

## 名人侦探

我在这次黑客马拉松上弄清楚了[Roon](https://twitter.com/tszzl)是谁。（如果你在网上不够频繁，可能不知道 Roon 是谁，他是一个在 Twitter 上发帖的化名 AI 研究员。）我当时正坐在演示会上，专心看着自己的事情，浏览着 Twitter，他就发了这条推文：

我听说他会在那里，所以我开始留意起来。果然，我检查了他的时间线的其余部分，看到了这条：

当然，作为一个爱打听八卦的人，我放大了第一张照片，试图看看他是否在房间里，如果是的话，是否有任何关于他位置的线索。我注意到他腿旁边的地板上的木边。然后…我注意到我旁边也有同样的木边：

*我的腿靠在和 Roon 腿相同的木边上。*

我抬头看见一个人坐在我前面，穿着和图片中一样的运动裤。太巧了！短暂的一刻，我坐在那里沐浴在我那神奇的夏洛克·福尔摩斯般的洞察力和享受着知道一个只有我一个人知道的秘密的温暖光辉中——但是任何其他人也可以弄清楚。

几分钟后我在后台见到了他。他很友善。在 Twitter 上更有趣。我没有告诉他我对互联网进行了调查。（对此抱歉，Roon——你的身份我会保密的。）

## 接下来会发生什么

黑客马拉松的活力是令人愉快的。这些活动的目的并不一定是要创造革命性的东西——而是利用最后期限和压缩的时间段来创造一些有趣和新颖的东西。这一代新的 AI 技术为这类项目提供了一种以前无法实现的质量和趣味性水平。

所有的与会者都感受到了身处指数曲线膝盖处的兴奋，但也有些忧虑。没有人确切知道这将如何发展，谁会胜出，或者对未来的后果会有什么影响。

但据我所知，每个人都在努力确保一切顺利进行。我认为这是一个好的开端。
