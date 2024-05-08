<!--yml

分类：COT 专栏

日期：2024-05-08 11:10:17

-->

# 关于人工智能的几点信念

> 来源：[`every.to/chain-of-thought/a-few-things-i-believe-about-ai`](https://every.to/chain-of-thought/a-few-things-i-believe-about-ai)

#### 聊天机器人课程早鸟价即将结束！

我们很高兴地宣布我们将重新推出我们的[如何构建聊天机器人课程](https://www.chatbot-course.com/)！如果您有兴趣学习如何利用人工智能构建，请加入我们即将到来的秋季队伍，由我们自己的 Dan Shipper 教授

在这门课程中，你将学到：

+   如何汇总聊天机器人的数据来源

+   如何管理您的向量数据库

+   如何构建一个 UI 并将您的聊天机器人投入生产

+   如何使用 Langchain 和 LlamaIndex 构建一个可以访问私人数据并使用工具的聊天机器人

+   还有更多！

学习[如何在不到 30 天内构建自己的聊天机器人](https://www.chatbot-course.com/)。课程将于 9 月 5 日开始，每周进行一次，为期五周，提供早鸟价 1300 美元，同时还提供 Every 会员资格。上次我们开设了这门课程，它很快就售罄了，所以如果你有兴趣，现在就抢购你的位置，早鸟价仍然有效。

早鸟价将于 7 月 31 日结束，之后价格将为 2000 美元。

在过去的六个月里，我一直在着迷地摆弄、撰写和投资于人工智能。这是一段旅程。我与一些[最有趣的思想家交谈过](https://every.to/chain-of-thought/linus-lee-is-living-with-ai)，我在深夜熬夜[进行了无数的小实验](https://every.to/chain-of-thought/how-to-build-a-chatbot-with-gpt-3?sid=18166)，我为[人工智能的末日](https://every.to/chain-of-thought/a-primer-on-ai-doom-for-people-who-don-t-yet-wear-fedoras?sid=18167)感到恐慌，我曾经幻想过永远不[再组织任何事情](https://every.to/chain-of-thought/the-end-of-organizing)，我也沉浸在使用这些模型带来的温暖的[好奇心](https://every.to/chain-of-thought/gpt-3-is-the-best-journal-you-ve-ever-used)和[愉悦](https://every.to/chain-of-thought/permission-to-be-excited)之中。

在这个时期从事人工智能工作感觉就像是让一枚 SpaceX 火箭绑在我的屁股上。我想每个人都有这种感觉。你飞得很快，但你总是感觉落后。偶尔你的大脑会因为你面前的可能性而爆炸。很容易陷入对于“世界已经改变”的全大写推文中。

今天，我想写点更加细致和反思性的东西。即使你屁股上绑着火箭，偶尔俯视一下，盘点一下你的位置也很重要。因此，这里是我对人工智能的几点信念的简短列表，这些信念正在影响我在 Every 及以后的工作方式。

## 知识编排是人工智能应用中最重要的瓶颈

智能有两个重要组成部分：推理和知识。[GPT-4 在推理方面非常出色](https://every.to/chain-of-thought/gpt-4-is-a-reasoning-engine)，但它对世界的了解是有限的。因此，它的性能受制于我们在适当的时间给予它正确的知识来推理的能力。

解锁人工智能的力量，并在我们的同步课程中仅需 30 天学会[创建个人人工智能聊天机器人](https://www.chatbot-course.com/)。无需高级编程技能，只需渴望学习。

**你将学到以下内容：**

+   掌握人工智能基础知识，如 GPT-4、ChatGPT、向量数据库和 LLM 库

+   学会构建、编码和发布多功能人工智能聊天机器人

+   通过您的人工智能助手增强写作、决策和构思能力

**包括以下内容：**

+   每周直播会话和专家指导

+   访问我们蓬勃发展的人工智能社区

+   实践项目和深入课程

+   与行业专家的现场问答会议

+   发布您的人工智能助手的逐步路线图

**早鸟价将于 7 月 31 日结束。** [现在报名](https://www.chatbot-course.com/)来把握机会。在短短 30 天内学会利用 AI 构建 AI！

这个问题，我称之为*知识编排*，是 AI 构建者在基础模型进展之外最大的未解决问题。它涉及到如何存储、索引和检索您需要执行有用的大型语言模型任务的知识。有许多人试图在堆栈的不同层次上改进这一点：

OpenAI 和其他参与者正在致力于在基础模型层构建此功能。他们正在构建更大的上下文窗口大小：您可以将更多的知识放入您的提示中，效果就会更好。[GPT-4 的 32,000 个标记上下文窗口](https://help.openai.com/en/articles/7127966-what-is-the-difference-between-the-gpt-4-models)比之前的模型好了 8 倍，因此改进正在迅速发生。

在此之上，[LlamaIndex](https://gpt-index.readthedocs.io/en/latest/) 和 [Langchain](https://python.langchain.com/en/latest/index.html) 正在构建开发者工具 / 基础设施层。它们使开发者只需几行代码就能轻松地将知识从各种不同类型的数据库中分块、存储和检索。

向量数据库提供商也在解决这个问题。Pinecone、Weaviate 和 Chroma 都在这里争夺统治地位，而 Pinecone 处于领先地位。

最后，各种一体化解决方案如 [Metal](https://getmetal.io) 和 [Baseplate](https://www.baseplate.ai) 将所有这些堆栈层捆绑在一起，使开发者能够轻松入门。它们拥有流畅的 Web 界面，使数据可观察，并使开发者能够快速入门。

我怀疑所有这些参与者都将开始泄漏他们当前的堆栈层，并尝试成长到其他领域。正确的解决方案将找出如何以正确的方式集成层，使构建者能够轻松入门并加快迭代速度。

知识编排是围绕知识的*过程*。但是知识本身呢？最有价值的是什么*类型*的知识？

这是我感到兴奋的一种类型：关于任何过程生命周期的端到端交互数据。一个过程可以是公司自己做的事情（比如创建 D2C 产品），也可以是公司使其客户能够做的事情（比如编辑照片）。

端到端交互意味着你可以看到开始的过程，你可以看到中间的所有迭代和编辑步骤，然后你可以看到并测量这个过程在最后产生的结果。

能够看到中间的迭代步骤和最终的结果会为你提供关于人类偏好的关键数据，以及他们是如何*达到*他们偏好解决方案的。你能看到的越多，你就能够更好地通过技术来引导模型，比如通过人类反馈的强化学习（RLHF）、微调和提示来自动重新创建这些过程——关键是，随着时间的推移使它们变得更好。

这种动态导致了我的下一个论点：集成了一个过程的初创企业将占据主导地位。

## 集成了一个过程的初创企业将占据主导地位

越多的过程你能够看到并存储在你的数据库中，你就能够越多地改进它。因此，在 AI 优先的世界中，初创企业有巨大的激励去横向集成和捆绑，以实现更好的性能。这意味着用自己的解决方案替换客户正在支付其他公司的东西，以便他们可以看到更多的过程并将其集成在一起。

我第一次想到这个想法是通过与 Replit 创始人 [Amjad Masad](https://twitter.com/amasad) 的一次对话。 （访谈即将发布！） Replit 是一个很好的例子，他们集成了客户所做的过程。他们构建了一个开发者平台，允许客户从浏览器窗口编写和部署应用程序。因此，在他们的情况下，他们正在集成的过程是将想法转化为软件。

我在其他地方也看到了类似的关于端到端过程集成重要性的观点。几周前我参加了 Sequoia 的一个活动，在那里 Sam Altman 提到，OpenAI 构建 ChatGPT 的主要原因之一是为了能够将终端用户的人类反馈纳入到他们的模型中。换句话说，他们起初只是一个 API，但意识到改善性能的最佳方法是向价值链的更多层次前进集成，以便他们可以直接访问他们客户的数据。

通过价值链早期部分的逆向整合也可能是有用的。你能够访问导致最终输出的编辑过程，你就能更好地学习重新创建或改进该过程：

这意味着什么？

**初创公司应该整合客户执行的更多流程**

如今，Midjourney 还不知道在生成图像后你会做什么。你可以想象一个 Midjourney 的版本，它生成一个初始图像，包括类似 Photoshop 的完整图像编辑功能，然后允许你将生成的图像发布到社交媒体并衡量其性能。像 Playground 这样的初创公司已经在做这个，我预计这种设计模式将随着时间的推移变得更加普遍。Adobe 也对他们的 Firefly 模型做了同样的事情。

**已经整合整个流程的公司将开始自动化，并取得良好的效果**

如今，浏览器能够看到客户所做的许多端到端流程。如果你是一名在 LinkedIn 上研究潜在客户、在 Google Meets 参加会议、在 Roam 记笔记，然后在 Salesforce 记录结果销售的销售人员——你的浏览器会看到这一切。这使得浏览器（和操作系统）具有自动化和改进这些流程的独特能力。它们将从减少手工数据输入开始，但会进入到诸如建议新的潜在客户，这样你甚至都不需要浏览 LinkedIn 的领域。

这种情况已经开始出现，[Adept](https://www.adept.ai/)正在努力自动化计算机上的流程。这种交互类型的潜在价值也是像[AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT)这样的代理范式如此受欢迎的原因。

**这个世界对于现有企业来说很好，但对于初创企业来说更好**

你可以认为，最有可能收集这种数据的公司是现有企业。毕竟，Google 拥有从你的邮件、文档、代码到网络分析的一切。但是，试图在其生态系统中的应用程序之间共享数据将会有重大的隐私问题，并且在这样做方面也会有重大的内部政治阻力。简而言之，现有企业的组织结构不适合这个世界。

从一开始就设计出这种意图的初创公司，他们知道拥有整个端到端的流程并集中存储该流程的数据以便用于改进模型是一项优先任务，他们将具有优势。我认为，他们在安全和隐私问题上也会有更多的自由，因为他们的规模要小得多。

当然，这种流程整合是耗时且昂贵的。初创公司无法一次性建立所有内容。但我认为，从一开始就有愿景最终控制这种流程的初创公司最终是一个不错的选择。

## AI 将在科学遇到困难的地方取得进展

科学的目的在于创造简单、因果关系的解释世界的方法，以便我们可以做出良好的预测。但是，在世界某些地区，简单、因果关系的解释似乎难以捉摸：心理学、社会科学、经济学等领域。尽管经过几个世纪的研究，进展微小，我们仍在这些领域寻找科学解释，因为我们没有更好的选择。

AI 改变了这一方程。它让我们能够对我们目前无法用科学解释的世界进行预测。现在 AI 可以[筛查癌症](https://github.com/jostmey/msm)。它还可以[预测谁可能被诊断出焦虑或抑郁](https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050920X00056/1-s2.0-S1877050920309091/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEK%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDxLfD5QWxhaFx%2BovJxDlRFBa9zmjzGCufFARpHdTqFtQIgfv9BvSt1UdOOu%2B%2B7TRjW%2B26y9quweszoFJvqcr2tCsYquwUIp%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDPg8DQ7WfK2zwca3OyqPBXJpOyqPfySdcKDkTxmn9sjOX0kTW9%2BjMpQfSD5pmiEUpZEwaS2X7ylcKiRVkuthrjxs0mjFQ%2B0ygi50QCTjTRse%2FRyBqWBae9vgHfJZ3yF8ydqO%2FiX9beSm%2F0hv7CD26AYNuNpiaF%2BXVcUPLNHDNuYHgXhoiBZUqyML0d9wI2Ta%2Fr8DoHjCrmxTd%2FbQa%2Bu3tfkg8lgqrwYZzhCEPYZiODCbWLton61wSdFHF66KWsIchxnmUpY5QIZ5sT%2FmlV%2FLD0Ub7%2Fuk%2FDx3jBZZbJNUW6vE8o2uS0TZ%2B9zRHqvEqKoRbxzTpWRDYHvAPfQUbtGX6FjRsyR3sO%2F%2BJnXG2uHz3nZ1F1iZlAo6dWnkmwu2wkUN4I%2BUwWTAKnGSNLVoLv%2Fz7EnM53F9qUmh748RDh6y%2BD2Ve5d38EffLBiBn3JsiUgyowBXuj%2FmpJlKqyNPPjvD%2BqcSkloGjTCOnSQ1zTi3k8cucx95v%2FUqrVNGKJR2PBdNZIkvnX0IjdG1bKDyH2NGuQ79VrUkPCYt9Yc6cDXyIdcnI54Nlpcg7p7jsmrQjbWr3C%2F2znT8LWQCK26T1MKoJjB0zbkwzvYn%2B2uswDTUrXfeTB6PM42TXKI61MLvpj00ige9MPsYxy7Wh24GaC1mBethVTxtr6oJqKc2%2FTt%2B8BxixiiOIurI2fNGkTG4QenbXINSfCSxsb7j3f6YOzHD1FW5H3M18beNK7GUFXdVzdN0lB4gfcjjYU5DeOniLdVutq6K1dr8E1lMUqBVXfRvDbkt3TTWN0AKu6LLXqjt0xmpQoagn1WQxHLKvpADkMlUja0u6X0ZKcEbo6q%2BbCc%2Flw6MwXJ1Gxc2ez8z6yEOHgXzY4aElX%2Bx5HvpqWYGPGkwqrWKogY6sQGTgugykIKrB6QhVTOlZ%2FLgD5p2ETTw%2BCFACT9jphyyTXXXMDu4by7kzQxVRl%2BzIw8H0Zo1nzqoGNps7EXYiEJ%2FEizDb8bJ8LRljAAwN5rhtHIx1y7OWd2gDof4qzB7OEShHkm24dUF1SCY9Vz8vp4BSafXFa1n%2FDHDVGxU33GrWq2fFEDVvGmRYvdHPvHG7hADxg0CBtvsw0mmr4dJirRIdZymmkg%2FvOXuzOog2jmMiSQ%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230421T143456Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYZWDRWUFY%2F20230421%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=48b6256c1821231fbb12b0923034301c35cbefa1d421d05e981147a67a2b3ee0&hash=702f97f07904a2500841e3e07829b419880cae5d61c59e948e0ec4fa5010f01a&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050920309091&tid=spdf-f9e8a19f-d4d4-450e-98f0-cddd616f4a36&sid=9992636595806941e52841e264dbc66c98e1gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f15570401535b5a0c5e00&rr=7bb659c8da740f3a&cc=us)，或者理论上讲，预测哪些干预或从业人员最有可能治愈它。它可以做到这一点，而不需要对焦虑是什么以及它如何产生有任何明确的、普遍适用的科学解释——这是几十年的研究都未能找到的。

有趣的是，如果人工智能能够学会成为类似焦虑这样的东西的良好预测器，那么在某种程度上，它已经在其网络内部编码了至少焦虑解释的一部分。神经网络难以检查，但比大脑容易检查——因此我们可能能够通过首先用人工智能学会预测它们来找到对于难以研究的现象的良好、简单的科学解释。

我们也可能发现，对于像焦虑这样的复杂的生物心理社会现象，最简单的可能解释实际上接近于可以预测它的神经网络的大小和复杂性。如果我们学会解释神经网络，并打印出它们的决策过程，我们可能会发现，对于像焦虑这样的事物的解释大小相当于一部小说，或一本教科书，或十本放在一起。

自然并不保证存在简单的解释——尽管我们作为人类本能地被它们吸引。这可能代表了科学的完整循环旅程。起初，它放弃了人类直觉和人类故事，转而采用方程式和实验。最终，也许直觉和叙事是我们的心智预测和解释那些超出我们更有限理性思维能力的事物的最佳方式。这确实是一件了不起的事情。
