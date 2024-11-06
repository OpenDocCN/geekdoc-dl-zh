<!--yml

category: 未分类

date: 2024-07-01 18:16:59

-->

# Systems ML workshop panel：ezyang's 博客

> 来源：[`blog.ezyang.com/2017/12/systems-ml-workshop-panel/`](http://blog.ezyang.com/2017/12/systems-ml-workshop-panel/)

+   JG: Joseph Gonzalez

+   GG: Garth Gibson (CMU)

+   DS: Dawn Song（加州大学伯克利分校）

+   JL: John Langford（微软纽约）

+   YQ: Yangqing Jia（Facebook）

+   SB: Sarah Bird

+   M: 主持人

+   A: 观众

M: 这个研讨会汇集了 ML 和系统。你如何定位在这个光谱上？你属于哪个社区？

YJ: 就在中间。我想更多地转向系统方向，但伯克利并行实验室把我踢出来了。ML 是我的主基地。

JL: 我来自机器学习领域，也将一直在这里，但我对系统很感兴趣。我的家是 NIPS 和 ICML。

DS: 我的领域是人工智能和安全，过去从事计算机安全，现在转向 AI。

GG: 系统。

JG: 我最初从事 ML，研究概率方法。在博士期间中途，我转向了系统。现在我正转向成为一个做 ML 的系统人员。

M: 我们看到了大量需要大量开发工作、资金和时间投入的深度学习 / 机器学习框架的蓬勃发展。问：学术界在这一领域的研究角色是什么？你可以进行怎样的大规模机器学习学习？

GG: 我喜欢 YJ 上次的回答。

YJ: 令人惊讶的是，学术界是如此多创新的源泉。尽管恭维不尽，我们在谷歌做了很多好工作，但然后 Alex 推出了两个 GPU，彻底改变了这一领域。学术界是我们发现所有新想法的神奇之地，而工业则是扩展它们的地方。

JL: 一些例子。如果你来自学术界，也许你没有在大公司做研究，但这是一个优势，因为你会花时间去寻找解决问题的正确算法。这将在长期内获胜。短期内，他们会用 AutoML 来蛮力解决。长期来看，学习算法将被设计成不需要参数。一个常见的 ML 论文是“我们消除了这个超参数”。当它们变得更自动化、更高效时，伟大的事情将会发生。资源受限有一个优势，因为你将以正确的方式解决问题。

另一个例子是，机器学习的研究告诉我们，在未来，我们将把任何刚学习并部署的模型视为固有破损和有缺陷的，因为数据收集不是训练和部署过程的一部分。它会腐化并变得无关紧要。ML 的整体范式是你与世界互动并学习，这在学术界可以很容易地进行研究，对你如何设计系统有着巨大的影响。

DS: 人们经常谈论在创业公司中，最好不要筹集大量资金；如果资源受限，你会更加专注和创造力。ML 非常广泛，有很多问题。现在我们从大量数据中学习，但在 NIPS 会议上，人类有能力从很少的例子中学习。这些是学术界要解决的问题，考虑到独特的资源限制。

GG: 缺乏足够的数据，很难集中精力在顶级准确性上，而学生可获得的数据如 DAWNbench 之类的东西往往落后。在学术界，我们与行业建立关系，派学生实习，他们有能力处理大数据，同时在大学探索第一原理。这是一个挑战，但开放出版和代码共享使这个世界更容忍。

JG: 我一直在努力专注于人力资源这一问题。我有研究生，优秀的学生，专注于一个关键问题可以取得很大进展。我们在处理大量数据时遇到困难。在 RL 方面的困难确实存在，我们可以建立模拟器以在这个规模上构建。能够使用模拟来获取数据；要有创意，找到新的有趣问题。

M: 跟进流程。我认为你们很多人都试图在自己的社区发布 ML 研究。他们能正确评估工作吗？他们不重视的常见原因是什么？

JG: 发表 ML 方面的研究在系统领域，或反之，都很难。这两个社区都不擅长评估其他领域的工作。在系统中进行 ML 研究，在这里看到的情况令人惊讶。反之，如果以系统的方式进行 ML 研究，可能不会在系统会场上表现良好。我看到的失败模式是，系统社区无法理解极端复杂性。在 ML 中，我有这个非常复杂的东西，但将它们简化为其基本组成部分。ML 试图通过复杂性扩展作为创新。更广泛地说，每个社区在看待研究时都有自己的偏见。我注意到的一件事是，情况有所改善。系统在评估方面做得更好，在这个研讨会上，人们正在推动先进的研究。

GG: 我年纪大了，见证了会议的创立。所以，你从重叠的领域开始。在我之前的生活中，主要是存储作为研究领域的概念，而不是设备的应用。你开始，提交论文。PC 上有两个了解此事的人，他们未分配，评审草率，有一个会议做得稍微好一点，但其他会议不读。我在容错、数据库、操作系统社区都遇到过这个问题，它们不互相阅读。当你有足够的数量时，在中间得到一个聚焦的会议；评审和 PC 已经看到该领域大部分好的工作。这很难，但我们正在 SysML 的边缘做这件事。我们在竞争中做正确的事情，处于技术最前沿。

M: 这是唯一的解决方案吗，还是我们可以混合 PC？

GG: 我见过很多试验来尝试它。你可能最终会导致永久性的分裂社区。

JL: 乔伊和道恩是 ICML 的一个领域主席。我发现机器学习社区对系统类的事物很友好。有一个系统领域的主席。希望论文能够得到适当分配。

M: 我们在系统方面做得不好。

DS: 关于机器学习和安全性，我们有这个问题。在安全领域，我们也有非常小的机器学习百分比，如果你提交机器学习，很难找到能够审查论文的人，因此审查质量变化很大。在机器学习安全方面，类似的问题。思考为什么会发生这种情况以及如何解决这个问题是很有趣的。总的来说，有时候最有趣的工作是跨学科领域。机器学习和系统，安全性，以及我看到的例子，包括在系统中的机器学习……所以，我真正能理解的一件事情是，尽管审查质量不同，从委员会的角度来看，他们真正想要的是对社区更有意义的论文，帮助人们接触到这个新领域，促进新的探索。随着时间的推移，会有更多的交叉污染。

JG: 我们正在启动一个 SysML 会议。我有一点犹豫：机器学习在系统上的进展很大，但现在我必须决定把论文投到哪里。我们看到的许多机器学习论文都涉及系统。

GG: 当你有一个新的会议领域时，并不是所有的工作都会发送到那里。重叠的是，你有一个喜欢的会议，你的英雄，你会把你最激动人心的工作发送到那个根会议。没有问题。

YJ: SysML 很棒，这就是它的出现方式。新的领域，它值得新的会议。

M: 你认为机器学习专家也需要成为系统专家吗？处在这种交叉点的人有不同的看待方式吗？或者你提出了一个好的算法，然后你

JL: 有一个墙是不行的。

有很多方法可以改变学习算法。如果你不理解，把工程师丢掉是有问题的。但如果你能建立桥梁来理解，它们不是艺术品，你可以打开并修改。这可以让你获得更好的解决方案。

GG: 同意，但最初发生的事情是你跨越到另一边，将其放入系统中，而我创新的部分是冗余性导致容错性，即使从另一边来看它相当普通。如果它是一个实质性的改进，值得去做。我们都在成长。

JG: 我们需要一堵墙，但我们会不断地拆除它。研究生阶段的 Matlab，我们开了它的玩笑，而 MKL 社区则使它变得更快。然后他们说我们将为分布式计算算法构建 ML，而 ML 将为系统编写类算法。然后在 pytorch、TF 等的开发中，这种抽象层次升级了。这个堆栈正在重新构建；系统社区正在使其更加高效。嗯，fp 可能会改变，这可能会影响算法。所以我们又开始拆除它了。但系统设计就是关于设计这堵墙。

YJ: 它更像是一个酒吧凳子。这是一个障碍，但我们不必要做到两者兼而有之，但你需要它来使其高效。一个故事：一个训练系统我们看了看，SGD。那个人发现一个非常好的圆整数：100。但人们皱眉，你应该将其圆整到 128。理解和改进 CS 和工程的共同核心，对于人们如何设计 ML 算法非常有帮助。

M: 有很多关于民主化 AI 的讨论，你们所有人都帮助了这个过程。一个真正民主化的 AI 景观是什么样子，我们离这样的世界有多远。

YJ: 我承认参与了框架战争。阅读计算机科学历史时，有一件很自然的事情，当领域刚刚开始时，会有各种标准和协议。FTP，Gopher，最终 HTTP 占据了主导地位，现在所有东西都在 HTTP 上运行。现在有各种不同的抽象层次；归根结底，每个人都在做计算图、优化。我期待着我们有一个非常好的图形表示、优化图形的协议。这不是一个美好的梦想，因为在编译器中我们已经有了这样的解决方案，LLVM。我不知道我们是否会达到那个状态，但我认为总有一天我们会到达那里。

JL: 当任何人都可以使用时，AI/ML 被民主化了。这意味着什么，一个程序员有一个库，或者语言结构，他们可以轻松地和常规地使用；没有数据不匹配、混淆或偏见的问题。所有人们在数据科学中担心的错误都从系统中移除，因为系统设计得正确并且易于使用。超越这一点的是，当有人使用系统时，该系统正在学习适应你。在人们互动方式上有巨大的改进空间。我不知道有多少次重写规则让我抓狂；为什么它不能按照我想要的方式重写。人们可以向学习算法传递信息，当这些信息能够有效地辅助人们时，你就实现了 AI 的民主化。

DS：我对 AI 民主化有着非常不同的看法。我认为真正有趣的是思考这里的民主化究竟意味着什么。对于系统工程师来说，这意味着让人们更容易学习，使用这些库和平台。但这实际上只是为他们提供工具。对我来说，我在讨论 AI 民主化时，我们是从完全不同的角度来看待它的。即使是代码：谁控制 AI，谁就控制世界。那么谁控制 AI？即使你给了每个人工具，按下按钮，但他们没有数据来进行训练。那么今天和明天谁控制 AI？是 Facebook、微软、谷歌...所以对我来说，民主化意味着完全不同的事情。今天，他们收集数据，训练模型，并控制谁可以使用模型，用户可以获得推荐，但不能直接访问模型。我们有一个项目实际上是要民主化 AI，用户可以控制自己的数据。结合区块链和 AI，用户可以将数据捐赠给智能合约，智能合约将规定条款；例如，如果您训练了一个模型，用户可以使用该模型，如果模型产生利润，用户可以获得一部分利润。智能合约可以规定各种激励条款；例如，如果数据比其他人更好，他们可以获得更多的利润，以及其他机制。开发者将提供 ML 训练算法，并在训练良好时获得收益。我们正在去中心化 AI 的力量；用户将能够直接访问模型并使用它们。在这种情况下，我希望有一个替代的未来，大公司可以继续经营业务，但用户通过以去中心化的方式集合他们的数据，将看到 AI 的真正民主化；他们将访问 AI 的力量，而不仅仅是使用工具。

（掌声）

GG：我认为 AI 民主化中很多内容意味着如何从少数创新者转向大量创新者。工具开发和标准化。我们离这一目标已经很近了。过去有一个例子，就是 VSLI 画框。直到某个时刻，只有电子工程师才能真正开发硬件。他们花了很多时间和精力确保可以在每个部件中都能通过，不会有太多串扰。一个团队聚集在一起，想，好吧，有一些设计规则。这让你可以相对容易地构建硬件。我可以画绿色/红色的框，几个月后，硬件就能工作了。虽然它永远不会像那些电子工程师那样快速工作，但它让我们可以构建一个 RISC 计算机，并且把它交付出去。我们参与了这场比赛，我们可以创新，可以做到。我们现在正试图构建的工具可以建立在统计学的基础上。

JG：当我开始博士学位时，我们手动计算积分和导数。自动微分是一个巨大的进步。我认为这是论文爆炸的原因之一。一年级生可以构建比我能做的更复杂的东西。这推动了算法方面的 AI 发展。

数据方面很有趣，这是我在系统中考虑的问题。有很多机会可以思考安全性如何互动，利用硬件保护它，市场从各种来源买卖数据，并在很多地方保护数据。我认为我们在思考算法的方式上取得了实质性的进展。

M：当我考虑普及人工智能时，最近困扰我们思想的问题，如解释性，公平性等。你能分享……任何解释性成为问题、问题的经验，我们是否需要在机器学习或系统-机器学习中更多地担心这些事情。

JG：我的研究生来找我，说模型停止工作了。我不知道如何解决这个问题；这个过程非常实验性。跟踪实验是这个过程的一个重要部分。我们非常关注可解释的模型，这意味着一些非常具体的东西。现在是可以解释的；我们不需要知道它究竟做了什么，但需要与我们所做的有某种联系。可解释，解释计算，它可能与决策相关或无关。这是关于可解释性的两个答案，以及我们如何调试这些系统。

GG：SOSP 刚刚结束，他们有十年的……他们提交的所有东西的好副本。在会议结束时，Peter Chen 拿走了所有的 PDF 文件，做了一个朴素贝叶斯分类器，看看他能多好地预测它会被接受。它预测被接受的东西中，一半确实被接受了。

那么他们做了什么？他们为流行的作者制作了一个检测器。所以你做的是那些成功了的人，他们会跟在后面。我意识到了这个问题。你可能认为你找到了一个好方法，但实际上是尼古拉·泽尔多维奇的论文。

DS：存在一个很大的争议。有些人认为这非常重要，有时只要模型运行良好就可以了。我们的大脑，我们无法真正解释我们如何做出某些决定，但它运行良好。这取决于应用场景。有些应用对可解释性有更强的要求；例如法律和医疗，而在其他领域则不那么重要。作为整个社区，有很多我们不理解的地方。我们可以谈论因果关系，透明度，所有相关的内容。作为整个社区，我们不真正理解可解释性意味着什么。没有一个好的定义。所有这些概念都相关，我们正在努力弄清楚真正的核心。这是一个非常好的开放性问题。

JL：有两种不同的解释。你能向一个人解释吗？这是有限的；没有可以解释的视觉模型。另一个定义是可调试性。如果你想创建复杂系统，它们需要是可调试的。这对于分布式系统来说是非平凡的，对于机器学习来说也是非平凡的。如果你想创建非平凡的机器学习系统，你必须弄清楚为什么它们不按你想要的方式行事。

DS：我们会调试我们的大脑吗？

JL：漫长的进化过程解决了很多问题……很多人的大脑里都有小问题。我知道我也有小问题。有时我会得视觉偏头痛……非常烦人。不，我们不调试我们的大脑，这是一个问题。

YJ：我确信我的大脑里有些小问题；我曾经在我奶奶家里追过鸡；鸡背上有一个地方，你按一下它，它就会弯腰坐下。它因为害怕而停止了。我们人类不会这样做。但是这些问题也存在于我们的大脑中。追求可解释性有助于理解事物的运作方式。旧日的深度梦境；这一行业始于弄清楚梯度的作用，我们向后传播，发现直接梯度不起作用；然后我们增加了 L1 先验，然后我们得到了图片。这种好奇心导致了卷积神经网络（CNNs）用随机权重编码了局部相关性；我们在 CNNs 中硬编码了结构化信息，这是我们以前不知道的。所以也许我们不会实现完全的可解释性，但一定程度的可解释性和创造力会有所帮助。

（听众提问）

A：我真的很想听听杰夫对系统机器学习的看法。作为系统的一部分，我对此很感兴趣，但有人说，你可以通过启发式方法取得很大进展。

JL：我觉得这很令人兴奋。

GG：索引数据库，当我阅读时，我感叹，“哇！这真的可能吗？”我认为这类应用的新颖性开拓了很多人的思路。现在我们认为机器学习工具是昂贵的东西，重复了人类轻而易举但计算机做得不好的事情。但数据库索引并非如此。我们可以执行它，但我们不会更好。但是通过预测器的压缩思路，让它半大小且速度翻倍，这是一个了不起的见解。

JG：我曾试图在这个领域发表文章。有段时间，系统不喜欢在它们的系统中使用复杂的算法。现在，这些天，系统说，“机器学习很酷。”但在哪里更容易取得成功，你的预测改进了系统，但一个糟糕的预测不会破坏系统。因此，调度是好的。在模型可以提高性能但不会损害的地方。解决系统问题的机器学习工作是成功的。

DS: 系统 ML 非常令人兴奋。我个人对这个领域非常兴奋，尤其是对那些从事系统工作并对 AI 感兴趣的人来说。系统 ML 是 ML 的一个令人惊奇的领域。我不会感到惊讶，我希望看到，在五年内，我们的系统更多地受 ML 驱动。许多系统有许多旋钮可以调节，试错设置，ML 可以帮助解决问题。在这些惊人的技术上，RL、bandits，不是用 bandits 来服务广告，我们可以尝试自动调整系统。就像我们看到 AI 正在改变许多应用领域一样，更智能的系统，旧系统，我们建造的那些，应该更智能。这是一个预测：我认为我们将在这个领域看到很多工作。我认为它将改变系统。

M: 我在这方面工作了很多。在某些设置中，我们在 bandits 中取得了一些成功，但是有些设置确实非常困难：有状态、选择、决策影响未来，这使得应用 RL 变得困难，或者 RL 技术需要大量数据。存在挑战，但也有成功案例。有很多论文在缓存、资源分配中应用 RL。真正的问题是为什么它没有在生产中使用？我不知道我们是否有答案，论文中这样做看起来非常好，但它并不那么主流，特别是在各个地方都有 RL。为什么它不普及。我看不到那个。

A: 难道不是因为它不可验证吗？你想要某种验证分析。

GG: 这被称为回归扫描。如果你在许多系统上部署。这涉及很多钱，它必须有效。如果它倒下来，那就是一场诉讼。我雇了一位软件副总裁。好的，现在我负责，事情将放慢速度。每一行代码都是 bug，如果我想要低 bug 率，我会阻止程序员编写代码，通过制定非常高的标准。这就是 JOy 所谈论的事情；他们需要一个真正引人注目的理由，没有任何不利因素，然后他们必须通过测试才能通过。因此，任何随机的事情都有一个高标准。

SB: 另一件事情正在发生，没有那么多人既了解这两个领域。在没有深入系统专业知识的情况下进行系统 ML 是非常困难的。你真的需要理解来解释它。

GG: 不久以前我们还没有托管服务。

M: 护栏，你约束了 ML 系统不建议不好的东西。我们在 MS 有一个场景，机器无响应。等待多久？你可以在 ML 中做到。选择都是合理的，它们从不超过你希望等待的最大时间。

A: 关于民主化。有很多关于优化模型的讨论，以便它们可以承受成本。另一个是去中心化数据...但是系统和模型有两个非常大的限制。它们成本高昂，而且存在很大的方差。因为成本的原因，如果有人涉足编程并进行研究，他将没有资源来做这件事。所以他们不会进入工程领域；他们会在亚马逊实习。所以，如果有一些社区试图降低门槛，民主化，有什么解决方案可以让人们更容易地进入呢？因为经济成本巨大。人们试图赚取巨额利润，创业公司，但没有...系统在去中心化方面存在缺陷...这只是一个大问题与机器学习相冲突。

JG: 我们在伯克利教授数据科学。总结一下，关于深度学习的成本如何？训练模型的成本，GPU，数据，如何让一个大学新生对此感兴趣，Chromebook，他们可以做研究并探索机会。在伯克利，我们正面临这个问题。我教了 200 名学生，其中很多是新生，Chromebook 和 iPad 是他们的主要计算机。我们使用 Azure 构建了工具...我们在 Azure 上运行云，在这些设备上，他们可以实验模型。他们可以使用预训练的模型，并学会如何...有人建造了一个俄罗斯 Twitter 机器人检测器，并在其中看到了价值和机会。然后他们参与了更多资金和工具的研究项目。

JL: 正确的接口可以起到很大作用，因为它们可以防止由于 bug 而无法执行任务。此外，深度学习正在风靡，但问题的框架比你所做的表示更重要。如果你有正确的问题，即使是一个愚蠢的表示，你仍然会做出有趣的事情。否则，它根本不会很好地工作。

YJ   YJ: 作为行业，不要害怕行业并尝试一下。回到伯克利，当伯克利人工智能使用 GPU 时，要求是每个 GPU 一个项目。我们学生，框定了十个不同的项目，我们只要求十个 GPU。英伟达来找我们问，你在干什么。我们就给你四十个 GPU 做研究。现在，FAIR 有实习，Google AI 有实习，所有这些都在行业和学术之间创造了非常好的合作，我想鼓励人们试试看。行业有资金，学术有人才，把它们结合在一起是永恒的主题。

A: 回到关于会议未来的方向，这个研讨会的未来；已经做出了任何决定吗，我们往哪里走？

SB: 这是一个正在进行中的工作。我们对反馈和您的看法感兴趣。我们已经进行了 10 年的工作坊，与 NIPS 和 iCML 一起。然后我们在 SOSP 上做了一个，非常令人兴奋。我们现在将在二月份在斯坦福举办一个单独的会议。我们认为在与 NIPS 和 ICML 共同举办的研讨会中有非常重要的角色要发挥。我们仍然计划继续这一系列的工作坊。在 ICML 和 NIPS 中也有越来越多的系统工作，这是自然扩展来接受这项工作。这个领域正在成长，我们将尝试几个场地，并形成一个社区。如果人们有想法。

JG: 更多的人应该参与进来。

M: 我们计划继续这个；观众很多，参与度也很高。

由于这是一个小组讨论，所以我必须要求你预测未来。告诉我你对未来 50-100 年真正激动的事情。如果你那时还活着，我会找到你，看看你的预测是否成真。或者说出你希望会发生的事情...

YJ: 今天我们用 Python 写作。希望我们能在一行中编写每个 ML 模型。分类器，get a cat。

JL: 现在，人们正处于一个逐渐增加学习曲线的阶段。ML 的核心是减少旋钮。我相信 ML 的视野在减少旋钮。我也相信普及 AI。你不断地转动...周围，开发者可以将学习算法融入系统。这将成为技术的一部分。这是炒作周期的一部分。NIPS 经历了一个阶段性转变。在某些时候，它必须下降。当它变得例行公事时，我们正在普及事物。

DS: 很难做出预测...我猜，现在，我们看到 ML 是一个例子，我们看到了浪潮。不久前，有神经网络的浪潮，图形模型，现在我们回到了神经网络。我认为...我希望我们...有一个稳定期。即使是在今年，我也与许多优秀的 ML 研究人员交谈过，尽管可以说今年写的论文更多，但当你听到人们谈论的里程碑时，许多人提到了过去几年的里程碑。AlexNet，ResNet，...我希望我们能看到超越深度学习的新创新。我确实教授 DL 课程，但我希望我们能看到一些超越 DL 的东西，能带领我们...我们需要更多，才能带领我们走向下一个水平。

GG: 我很想指出 DL 是五年前的事情，互联网泡沫时代也不过五年...我认为，我期待 CS，总体科学的做法变化，从统计 AI 中学到。我最喜欢的是过拟合。我对过拟合的理解很浅显，直到 ML 强调了这一点。我期待有一天，学生告诉我，他们停止写代码，因为他们正在添加参数...他们为测试代码添加了一个体面的随机，iid 过程。我们还远远没有到那一步，但我认为它即将到来。

JG：我期待图形模型的回归……实际上并不期待。当我们使 AI 民主化时，最终发生的是，我们在使技术民主化。我可以走到 Alexa 面前教它。或者我可以教我的特斯拉如何更恰当地停车。技术能够适应我们，因为它能学习；当我能向计算机解释我想要什么时。（就像星际迷航但没有传送装置。）