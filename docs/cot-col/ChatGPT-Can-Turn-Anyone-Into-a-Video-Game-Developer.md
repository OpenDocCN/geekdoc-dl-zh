<!--yml

类别：COT 专栏

日期：2024-05-08 11:04:35

-->

# ChatGPT 能让任何人成为视频游戏开发者

> 来源：[`every.to/chain-of-thought/chatgpt-can-turn-anyone-into-a-video-game-developer`](https://every.to/chain-of-thought/chatgpt-can-turn-anyone-into-a-video-game-developer)

#### 赞助商：Gamma

幻灯片已经过时了。[Gamma](https://gamma.app/) 提出了一种全新的呈现思想的方式——更快速、更灵活，由人工智能驱动。使用 [Gamma](https://gamma.app/)，你可以轻松创建令人惊艳的内容，为任何设备和平台进行优化，而不用浪费时间在设计或格式上。

《你如何使用 ChatGPT？》*正在休息一周——我们下周将回归，带来一集新的节目。在此期间，我们将发布* [*我们系列中的下一篇文章*](https://every.to/chain-of-thought/how-to-use-chatgpt-to-set-ambitious-goals) *，该系列基于播客，我们在其中分享一些技术领域中最聪明的人如何使用 ChatGPT 和其他人工智能工具的可操作的战术性方法。每个贡献者 Rhea Purohit 分解了播客中的对话，并提取了提示和回复——包括截图——供您复制。继续阅读，了解丹·希珀和洛根·基尔帕特里克是如何在不到一个小时的时间内使用 ChatGPT 制作一个视频游戏的，他曾是 OpenAI 的第一位开发者关系倡导者（他在我们录制这一集之后离开了公司）。—凯特·李*

* * *

自从我开始以写作为生，我就一直致力于让更多的人去写作。在超市里，我告诉人们将观点浓缩成文字的乐趣，而他们在沉重的购物袋下面不适地挪动着。他们中的大多数承认，他们成年后从未参与过写作，怀疑自己的能力。写作是一种超能力，尽管许多人感觉到，它不需要令人生畏，特别是因为我们有比以往任何时候都更多的工具来帮助表达我们的思想。

然而，我确实理解对于一个完全陌生的技能感到畏惧的感受。这就是我对软件的感觉。我不知道怎么编程，甚至不知道从哪里开始。

[丹·希珀](https://twitter.com/danshipper) 和 [洛根·基尔帕特里克](https://twitter.com/OfficialLoganK) 认为编写软件是一种超能力。在这次[对话](https://every.to/chain-of-thought/how-to-make-a-video-game-with-chatgpt-in-60-minutes)中，他们谈到了 ChatGPT 如何使每个人都能成为建设者。他们还通过在不到一个小时的时间内使用 ChatGPT 制作了一个名为 [Allocator](https://chat.openai.com/g/g-oooxUbOkj-allocator) 的视频游戏，而且完全不用编写一行代码。

告别基础幻灯片演示。[Gamma](https://gamma.app/)使用尖端人工智能彻底改变了我们分享想法的方式。这不仅仅是一个工具；它是一个创意助手，让您能够快速、无缝地创建视觉吸引人的内容。无论您是向一个小团队还是向一个大型观众展示，[Gamma](https://gamma.app/)都能确保您的想法在所有设备和平台上闪耀。免费体验不同！

洛根是 OpenAI 的第一位开发者关系和倡导聘用，致力于支持使用 ChatGPT、DALL-E 和 OpenAI API 构建的社区的人员。 （自从我们录制了这一集之后，他宣布了[他离职](https://twitter.com/OfficialLoganK/status/1763580712874094693)。）

几个月前，OpenAI 发布了[GPT Builder](https://chat.openai.com/gpts/editor)，这是一个工具，使人们能够定制几乎[他们想要的任何内容](https://openai.com/chatgpt#do-more-with-gpts)的定制 GPT。正是这个工具使得丹和洛根的视频游戏实验成为可能。洛根说，GPT Builder 降低了创新的门槛，特别是对于像我这样不懂编程的人来说。

如果你是一个充满创意的人，一直梦想着将自己的想法变成现实，那就跟着丹和洛根一起实现他们共同的童年梦想，用 GPT Builder 和 ChatGPT 构建一个视频游戏。

丹和洛根构思了他们应该建立什么样的游戏的想法，最终选择了一款文本策略游戏，玩家扮演历史上的美国总统，任务是管理联邦政府的预算。

首先，我们将给出丹和洛根的提示，然后是来自 GPT Builder 和 ChatGPT 的屏幕截图。我们的评论用斜体标注

**丹和洛根：**我们想做一个游戏。核心概念是你可以选择要成为哪位总统，然后 ChatGPT 将搜索美国政府预算和该预算的支出分配，让用户重新分配预算，然后根据更新后的分配玩出这个世界。

*GPT Builder 在游戏开发过程中扮演着一个积极的向导的角色，建议丹和洛根首先为游戏选择一个名称。*

所有截图由丹·希普和[你如何使用 ChatGPT？](https://youtu.be/3TLORk-eZAw?si=RXvkL3ELdAZs8T3s)提供。

*洛根认为一个更短的名字可能比 Budget Commander 更好。*

**丹和洛根：**我喜欢一个好的单词名称，你能为我找到一个吗？

*Allocato 不是一个坏名字，但它促使丹想到一个更好的名字，Allocator，这是对他最近写作的一个主题的致敬：*[*分配经济*](https://every.to/chain-of-thought/the-knowledge-economy-is-over-welcome-to-the-allocation-economy)*。*

**丹和洛根：**我喜欢 Allocator 这个名字，我们就用这个吧。

*GPT Builder 为 Allocator 生成了封面艺术品，但丹和洛根觉得它与历史视频游戏的氛围不太匹配。*

**丹和洛根**：我觉得不太对劲。这个游戏是关于历史事件和总统分配资源的。我们能不能做得更接近一些？

*GPT Builder 似乎从热门的尼古拉斯·凯奇系列电影* [国家宝藏](https://en.wikipedia.org/wiki/National_Treasure_(franchise)) *中获得了创造性的灵感。这样更好，所以丹和洛根决定继续前进。*

**丹和洛根**：很好。接下来是什么？

*丹和洛根希望 Allocator 包括历史事实和沉浸式的假设情景。但由于他们不是游戏开发者，在继续之前，他们指示 GPT Builder 以专家身份发言，并概述游戏开发的框架。*

**丹和洛根**：我觉得它需要两者兼顾。这个游戏需要一些基本的核心机制，但我不太确定应该是什么。我们不是专家。我们需要你作为专家的知识，来构建非常有趣和引人入胜的基于情景的游戏的框架。*文明*和*帝国时代*是我们感到受到启发的一些游戏。它们并不完全和我们正在制作的游戏相同，但是它们的氛围是我们的灵感来源。

*丹和洛根希望与 GPT Builder 一起迭代 Allocator 的游戏机制，但它却在游戏方面大展拳脚。洛根认为这可能是因为 GPT Builder 正在遵循的标准指令，并建议点击配置标签页来调整这些指令。*

**丹和洛根**：请忽略预算分配的沉闷，让它变得令人兴奋起来。

*之后，他们还回答了 GPT Builder 提出的关于 Allocator 要关注哪个历史时期的问题。*

**丹和洛根**：我觉得我们希望吸引尽可能广泛的受众，所以也许保持话题适合所有人玩这个游戏。围绕登月的那段时期会很有趣，但我们希望时间段在定制的初始提示中而不是限制你到任何特定的时间段。

*丹和洛根仍然希望 Allocator 的核心机制能更加关注，但他们决定在 GPT Builder 中构建游戏，然后使用 ChatGPT 进行完善。他们继续与 GPT Builder 互动。*

**丹和洛根**：我觉得一个充满信息并且有帮助的旁白会很有用，就像尼古拉斯·凯奇在*国家宝藏*中一样，谢谢你。（清楚地说，不像尼古拉斯·凯奇在*离开拉斯维加斯*中那样，那是令人沮丧的。）

*Allocator 的第一个版本已经准备好了！丹和洛根切换到 ChatGPT 标签页。他们从 GPT Builder 的配置标签页中复制并粘贴自定义指令到 ChatGPT，并指示其为游戏生成主要和次要的机制。（洛根在为制作棋盘游戏而定制的 GPT 中偶然发现了这个关于游戏机制的术语。）*

了解更多

### 这篇文章是给

付费订阅者

订阅 →

或者，登录。

#### 感谢我们的赞助商：Gamma

再次感谢我们的赞助商 [Gamma](https://gamma.app/)，这款反 PowerPoint 软件。

[Gamma](https://gamma.app/) 是由人工智能驱动的一股清新风，使您可以专注于您的思想而不是格式。是时候让每次演示都有所作为，而不需要额外的努力了。
