<!--yml

分类：COT 专栏

日期：2024-05-08 11:07:42

-->

# 我在 OpenAI 开发者日看到的事情

> 来源：[`every.to/chain-of-thought/what-i-saw-at-openai-s-developer-day`](https://every.to/chain-of-thought/what-i-saw-at-openai-s-developer-day)

#### 由 LiveFlow 赞助

本文由 [LiveFlow](https://bit.ly/45CaNVY) 赞助，北美第一财务分析软件。

计划你的 2024 年预算但不知从何开始？[LiveFlow](https://bit.ly/45CaNVY)已经制作好了财务模型和模板。按部门预算与实际情况对比、13 周现金流预测、合并……你只需插上并开始使用。只需将你的 QuickBooks Online 连接到他们的模型，几分钟内就能完全设置好。告别手动输入财务数据。迎接你的定制模型自动更新和完全流程化的财务管理。使用促销码 EVERY 获得三个月 20% 的优惠。

[LiveFlow](https://bit.ly/45CaNVY)，自动化和效率的理想选择。G2 评选的财务规划与分析软件中最易用的第一名。

我喜欢看人们在认为没人注意时的所作所为。

去 OpenAI Dev Day 这样的活动真是令人难以置信的事情：你可以看到摄像机没有捕捉到的东西，听到舞台上没有被说出的话。

这里全是人群、混凝土、快速的 wifi 和 LED 灯。对像我这样的 AI 迷来说是一场魔术表演。

我在人群中穿行，做着我专利的 FCO：秘密会议打量。我会看到某些可能有名气的人——比如 [Roon](https://twitter.com/tszzl)，或者 [Karpathy,](https://twitter.com/karpathy)，或者 [Kevin Roose](https://twitter.com/kevinroose)——然后迅速地瞥一眼他们的徽章，再抬起头看他们的脸，然后他们还没来得及发出像“喂，伙计，我的眼睛在这里呢！”那样的眼神。

我通常在活动中坐在后排，但在 Dev Day 上，我确保坐在了前排。我想近距离看魔术。

Sam Altman 走上舞台，向人群问好。我能看到他脸上和身上紧绷、充满紧张能量的表情。我能感觉到他在演讲中的演练时间。开场白之后，Sam 展示了一个由创意专业人士、开发者和普通人讨论他们如何使用 ChatGPT 的视频。灯光暗了下来，他走到一边，视频开始播放。每个人都在看视频，但我在看 Sam。

他独自站在舞台角落的阴影中。他穿着深色牛仔裤和原色的阿迪达斯 x 乐高联名运动鞋。他的双手紧握在前面，目不转睛地盯着地板。Sam 是一个充满张力和总是保持状态的人。但在舞台边上，听着视频播放，他显得不熟练也没做过准备。我感觉自己就像抓住了魔术师左手操纵隐藏的硬币，而观众们则注视着他挥舞的右手。

看到魔术师的秘密暂时打破了他们的魔法。但它也创造了一种新的魔法：你看到魔术师是一个人类。吃饭、呼吸、一次穿上一条裤子，但依然做着魔术。

萨姆正朝着成为科技传奇人物的方向迈进。但在舞台上的那一刻，他也是一个人类。他看起来很享受，观察和期待着他所创造的事物，并在世界上最大的舞台上观看它的演出。他正在活着，任何曾经创造过东西并希望世界会喜欢的人都在实现梦想。

观看他在那一刻是值得一票入场费的。我不会快忘记它的。

他告诉我们的如下：

更大、更聪明、更快、更便宜、更容易。

这就是 OpenAI 昨天宣布的变化摘要。我们来逐一讨论更新，并讨论它们为什么重要。

想象一下：无需举手之劳地进行财务报告。[LiveFlow](https://bit.ly/45CaNVY)是一个改变游戏规则的平台，自动化报告，每年节省您超过 5 周的手工工作时间。

有数百种财务模型可供选择，您只需插上并播放。[LiveFlow](https://bit.ly/45CaNVY)将连接到您的 QuickBooks，以更新您在 Google Sheets 中的自定义模型。

加入那些利用 LiveFlow 的力量来节省您宝贵时间的人的行列。想提升您的财务管理水平？现在就通过 LiveFlow 探索更多。使用促销代码 EVERY，连续 3 个月享受 8 折优惠。

## 一个新的模型：GPT-4 Turbo

### 更大

OpenAI 发布了一个新模型，GPT-4 Turbo，它拥有 128K 令牌的上下文窗口。这意味着您发送到 GPT-4 Turbo 的每个提示可以相当于 300 页的文本。以下是几件事情，可以放入 300 页中：

+   《精益创业》全文 100%

+   三本安东尼·德·圣-埃克絮佩里的《小王子》

+   我中学时代的抑郁日记至少有一半。

这是比今天之前最广泛可用的 GPT-4 版本的上下文窗口长度增加了 16 倍。它显著增强了开发人员可以在 GPT-4 上运行的查询的复杂性和功能。以前，开发人员必须花费时间和精力决定将哪些信息放入他们的提示中，我曾经认为这是[LLM 性能的最重要瓶颈之一](https://every.to/chain-of-thought/a-few-things-i-believe-about-ai?sid=27581)。

128K 上下文窗口使得这项任务变得更加容易，但它并不能解决所有问题。长上下文窗口很难管理，LLM 越长，输入上下文就越有可能遗忘或错过上下文。我们还不知道 GPT-4 Turbo 是否会遇到这些问题，但随着我开始用它构建东西，我会告诉你的。

### 更聪明

GPT-4 Turbo 在几个方面比以前的 OpenAI 模型更聪明。

**它可以同时使用多个工具。** GPT-4 的上一个版本引入了工具使用，我曾经报道过。工具使用允许 GPT-4 调用开发者定义的工具——例如网络浏览器、计算器或 API——以完成查询。以前，GPT-4 一次只能使用一个工具。现在它可以同时使用多个工具。

**更新的知识截止时间。** GPT-4 的先前版本只知道 2021 年 9 月之前的事件。这个版本的截至日期是 2023 年 4 月，使其更加可靠。

**GPT-4 支持 JSON 格式。** JSON 是易于非 AI 应用程序阅读的文本。GPT-4 Turbo 可以可靠地以这种格式返回结果，这使得将其集成到其他软件中变得更加容易。以前，开发人员必须劝说 GPT 正确地格式化其输出，例如告诉 GPT 如果它弄错了他们就会被解雇。现在不再需要这样做。

**GPT-4 可以编写和运行代码。**有一段时间了，ChatGPT Plus 用户已经能够使用[Code Interpreter](https://every.to/napkin-math/openai-s-code-interpreter-is-about-to-remake-finance)(后来更名为 Advanced Data Analysis)，这是一个 ChatGPT 插件，可以为您编写和运行 Python 代码。就像你口袋里有一位数据科学家一样——现在开发人员可以通过 GPT-4 API 使用和集成它。

**多模态。** GPT-4 API 可以接受图像作为输入：开发者可以发送任何图像给它，GPT-4 可以告诉他们看到了什么。它还可以进行文本到语音的转换，意味着它可以以人类般的声音回复文本输入。而且它可以使用 DALL-E 进行图像生成。

### 更快

就我所知，目前没有公开可用的速度基准，但 Sam 说它更快。根据我昨晚穿着睡衣玩耍时进行的非常科学的测试，他是对的。它速度非常快。它让 GPT-4 远远甩在后面，看起来至少和 GPT 3.5 Turbo 一样快，如果不是稍微快一点的话——这是之前最快的模型。

### 更便宜

GPT-4 Turbo 比 GPT-4 **便宜 3 倍**。我记不起来有哪家公司能够在大幅降低价格的同时如此激进地提高性能。

我们很幸运，OpenAI 正在执行硅谷的策略，旨在创造大规模采用，而不仅仅是获得丰厚的企业合同。

如果 IBM 发明了 GPT，你认为它会做这样的事吗？我觉得不会。

### 更便宜

OpenAI 还大大简化了开发者和非开发者与 GPT-4 Turbo 的交互。该公司使许多第三方库的功能（以及开发者通常编写的样板代码）变得不再必要。以下是一些方法：

**检索。** 这是一个大问题。[提高 LLM 性能的重要方法之一](https://every.to/chain-of-thought/a-few-things-i-believe-about-ai?sid=27581)是让模型访问私有数据，如公司知识库或个人笔记。以前，这个功能必须手工构建（[就像我为我的 *Huberman Lab* 机器人所做的那样](https://every.to/chain-of-thought/i-trained-a-gpt-3-chatbot-on-every-episode-of-my-favorite-podcast)）或使用 Langchain 或 LlamaIndex 等第三方库（后者我是一位投资者）。OpenAI 将这些库的一些功能合并到其核心 API 中，通过其检索功能，使开发者更容易开始构建 GPT-4 应用。

这将是一个有趣的观察对象。一方面，这减少了对这些第三方库的需求。另一方面，OpenAI 的检索机制目前是一个没有可配置性的黑盒子。检索是一个难题，而且有许多不同的检索机制用于不同的目的。OpenAI 的新发布涵盖了基础知识，但 Langchain 和 LlamaIndex 实现了各种检索类型 *并且* 与 OpenAI 制作的模型不相容——因此仍然需要它们的服务。

**保存状态。** 我之前写过[GPT-4 就像 *50 次初恋* 中的 Drew Barrymore](https://every.to/chain-of-thought/using-chatgpt-custom-instructions-for-fun-and-profit)：每次你与它交互，你都必须一遍又一遍地介绍自己和为什么它爱你。GPT-4 API 可以通过一个名为 Threads 的新功能自动记住对话历史，节省开发者的时间和麻烦，因为他们不再需要自己管理对话历史。

**定制无代码 ChatGPT。** OpenAI 还为任何人都很容易地构建他们自己的定制版本的 ChatGPT，使用私人数据——无需编程。任何人都可以建立一个具有自己个性和访问私人知识的 ChatGPT 版本。这是一件大事。今年早些时候，[我基于 Lenny Rachitsky 的新闻通讯档案构建了一个 Substacker Lenny Rachitsky 的机器人](https://www.lennysnewsletter.com/p/i-built-a-lenny-chatbot-using-gpt)。今天的更新意味着任何人都可以构建一个相当的机器人——无需代码。

**GPT 应用商店。** OpenAI 宣布任何人都可以在公共应用商店中列出他们的 GPT 并收费。我已经争论了近一年，[聊天机器人是一种新的内容格式](https://twitter.com/danshipper/status/1721643301257068870)——这一发展支持了这一论点。

**不再切换模型。** 这是一个巨大的更新。在 ChatGPT 的先前版本中，您必须选择要使用的模型：GPT-3.5、GPT-4、带有 DALL-E 的 GPT、带有 Web 浏览的 GPT 或带有高级数据分析的 GPT。现在不再是这样：现在，您只需向 ChatGPT 输入消息，它将为您选择正确的模型。用户可以更轻松地将 ChatGPT 的所有不同功能融合在一起，而无需来回切换，并且为开发者创造了新的机会（稍后在本文中讨论）。

## 渐进式更新——为未来打下基础

所有这些更新都很棒，但它们大多是渐进式的。它们将开发者自己必须完成的许多任务整合到 API 中，使开发者构建事物更快、更便宜、更强大。

然而，这些功能为一个可能更重大的更新奠定了基础：代理。代理是可以分配复杂的、多步骤任务并且不需要监督就可以完成的模型。这就是 GPT-4 新的 Assistant API 所涉及的内容。

这是使检索、保存状态和使用工具（如上所述）的 API。这些元素一起构成了一个代理提供的开端。从字里行间看，OpenAI 正在预览一个世界，在这个世界中，您将能够指定助理的目标，给他们一组工具，然后让他们自行完成目标。

我们离那个目标还很远，因为 GPT-4 还不足以自主规划和执行任务。但是 OpenAI 现在正在奠定架构和安全基础，并故意逐步推出增量步骤，以使技术准备就绪。

## OpenAI 正试图让一个应用商店成为现实

今年四月，OpenAI [推出了插件](https://openai.com/blog/chatgpt-plugins)，允许用户从 ChatGPT 内访问第三方服务和数据。插件成为新的应用商店的炒作很多，但事实并非如此。OpenAI 从未发布过数字，但据我所知，第三方插件的采用率非常低，尽管由 OpenAI 构建的两个插件：Code Interpreter 和 DALL-E 的采用率很高。

现在，OpenAI 正在通过 GPTs 再次尝试（它允许任何人创建具有私有数据的 ChatGPT 的自定义版本）：

任何用户都可以创建一个 GPT。您可以定义它的个性：它如何回答查询以及使用的语音和语调。您可以赋予它访问工具的能力，例如执行代码或从私有知识库中回答问题的能力。然后，GPT 可以发布供其他用户使用。

我安装了一个名为“谈判者”的新 GPT（由 OpenAI 构建），它可以帮助你在任何谈判中为自己辩护。它会出现在我的 ChatGPT 侧边栏中，如下所示：

如果我点击谈判者，它会将我从普通的 ChatGPT 转移到一个专门设计来帮助我在任何谈判中获得最佳结果的体验中：

我支持这种方法。我喜欢使构建聊天机器人的能力民主化的想法——我可以看到自己在接下来的几周里在这里做很多实验。

话虽如此，我有疑虑。它遭受了与 OpenAI 失败的插件实验相同的问题：没有人想要在不同的使用情景中切换不同版本的 ChatGPT。

更好的方法是让 ChatGPT 自动地学会如何在需要时切换到特定的人格，比如谈判者，而不需要人为干预，在不需要时再切换回来。在这种情况下，我不认为这些机器人会得到大规模的应用。

但如果它*确实*发生了，那将是巨大的。为 ChatGPT 下载一个新的人格将相当于让你的人工智能读一本书或上一门新课程。在这个世界上，将会有一个整个经济体系的人们专门为 LLM 创建内容，而不是为人类。例如，与其买一本关于谈判的书自己阅读，我可能会买一个相当于 ChatGPT 阅读和消化的版本。

出于这个原因，我认为 OpenAI 确实有可能最终建立起一种应用商店的体验。但在他们能够想出让 ChatGPT 自动在一长串人格之间切换的方法之前，这是不会发生的。OpenAI 改变了 ChatGPT，这样你就不必在他们的内部模型之间切换，这意味着对于定制的 GPT 来说，这很可能也会很快发生。

## OpenAI 与开发者的关系

这次开发者大会最引人注目的一点是 OpenAI 发布了许多更适合消费者而不是开发者的更新。例如，定制 GPT 面向的是消费者群体，公司发布的一些 ChatGPT 特定更新也是如此。这反映了一个重要的事实：OpenAI 目前在成为一个消费者公司和一个开发者公司之间存在分歧。

ChatGPT 是罪孽的产物。当 OpenAI 刚开始时，它的目标是为开发者服务——直到它无意中创造出了有史以来最大的消费者应用程序。不幸的是，这将公司与开发者置于对立的位置，因为 ChatGPT 直接与开发者想要构建的许多内容竞争——无论是在消费者层面还是基础设施层面。

如果 OpenAI 必须在 ChatGPT 和其开发者生态系统之间选择，那它*必须*选择 ChatGPT。ChatGPT 是 OpenAI 最有价值的高质量训练数据来源，因此这是提高模型质量的最佳途径。

不仅如此，OpenAI 正在走向使开发工作商品化和消费化的方向。ChatGPT 本身可以将任何人都变成一个半能干的程序员。而它昨天推出的功能使任何人都可以构建一个聊天机器人，而无需编码。

这是公司核心的根本紧张关系。 这是许多平台核心的紧张关系之一——例如，Apple 在 iOS 和 MacOS 上就面临着这个问题。 Apple 因建立与第三方开发者所建立的内部产品竞争而受到批评，这被称为 [“Sherlocking。”](https://thehustle.co/sherlocking-explained/#:~:text=You%20might%20assume%20%E2%80%9Csherlocking%E2%80%9D%20means,a%20third%2Dparty%20tool%20irrelevant.)

但这对于 OpenAI 来说更加棘手，因为其消费产品与其为开发者提供的产品非常相似。 这就像是 Apple 允许开发者发布他们自己的 iOS 版本一样。

我的猜测是，如果你想在 OpenAI 生态系统中玩耍，最好的方法是收集一个私人数据集，这对使用 ChatGPT 的人有用，并将其发布为自定义 GPT。

OpenAI 很可能会投资于使 ChatGPT 界面中的 GPTs 更加易于访问，并且随着时间的推移变得更加强大。 你要为派对带来的优势是私人、精选的数据，以及一套规则，用于如何为特定类型的用户在特定情况下使用这些数据。 这是 OpenAI 不太可能直接竞争的事情——所以这是双赢的。

## 世界上最令人兴奋的公司

目前没有任何公司比 OpenAI 做的工作更有趣、更快。公司的进展速度令人震惊，并且看起来在短时间内不会放慢脚步。在这次会议上，街上流传着 OpenAI 是一个人才横扫机器，并且它在其黄金时期感觉非常像 Stripe。 （事实上，我听说 OpenAI 曾雇佣了许多曾在 Stripe 工作过的人。）

房间里的能量是可以感受到的。 我认为在技术领域没有比这更大或更有趣的故事了。 未来几个月和几年将会是疯狂的。

* * *

## 杂七杂八

**多样性。** 我欣赏这次会议的一件事是其包容性。 公司提供了优质的食物，由本地的女性或少数族裔拥有的企业提供。 展示和小组讨论有许多在 OpenAI 和其他大型科技公司（如 Shopify 和 Salesforce）担任领导角色的演讲者。 所有这一切都很低调，没有表演性质。 在我看来，OpenAI 做得很对，值得赞扬

**OpenAI 和 Microsoft。** 一位与会者对我说，他认为 OpenAI 与微软的关系让他想起了苹果与英特尔的长期关系。 英特尔制造处理器，而苹果做其他的一切。 在 OpenAI 的情况下，微软提供托管基础设施，而 OpenAI 做其他的一切。 这不是一个完美的类比，但它让我产生共鸣，尤其是因为萨提亚·纳德拉在这次会议上出现，与 Sam 一起在后者的主题演讲中站在舞台上。

**有人能弄清楚 OpenAI 的命名吗？** 我简直不敢相信它把它的新定制无代码 ChatGPT 叫做“GPTs”。 有人需要进行干预——这太令人困惑了。

* * *

我将在本周玩弄所有新技术，所以请期待更多更新（也许还有一些演示和代码样本）。现在是一个令人兴奋的时代，可以做一名骚扰者。敬请关注更多信息。
