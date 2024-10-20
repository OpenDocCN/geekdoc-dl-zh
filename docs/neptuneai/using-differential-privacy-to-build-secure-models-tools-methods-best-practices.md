# 使用差异隐私构建安全模型:工具、方法、最佳实践

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/using-differential-privacy-to-build-secure-models-tools-methods-best-practices>

2020 年的新冠肺炎·疫情让我们看到了生活中不同的挑战，有些是痛苦的，有些则暴露了社会的缺陷。这也让我们记住了适当数据的重要性和缺乏。

[世卫组织的这篇](https://web.archive.org/web/20221206013925/https://www.who.int/news/item/19-11-2020-joint-statement-on-data-protection-and-privacy-in-the-covid-19-response)文章引用了*“数据的收集、使用、共享和进一步处理有助于限制病毒的传播，有助于加速恢复，特别是通过* [*数字接触追踪*](https://web.archive.org/web/20221206013925/https://ourworldindata.org/covid-exemplar-south-korea) *。”*

它还引用了*“例如，从人们使用手机、电子邮件、银行、社交媒体、邮政服务中获得的移动数据可以帮助监测病毒的传播，并支持联合国系统各组织授权活动的实施。"*稍后发布[伦理考量，指导使用数字邻近追踪技术进行新冠肺炎接触者追踪](https://web.archive.org/web/20221206013925/https://www.who.int/publications/i/item/WHO-2019-nCoV-Ethics_Contact_tracing_apps-2020.1)。

这听起来像是一个简单的答案，但确实是一个需要解决的复杂问题。医疗数据是最机密的数据之一，与任何其他个人信息不同，它既可以用于个人，也可以用于个人。例如，医疗保健[数据泄露](https://web.archive.org/web/20221206013925/https://healthitsecurity.com/news/the-10-biggest-healthcare-data-breaches-of-2020)可能导致 [COVID 欺诈骗局](https://web.archive.org/web/20221206013925/https://healthitsecurity.com/news/feds-issue-joint-alert-on-covid-19-cares-act-payment-fraud-scams)，我们在过去的一年中已经听说过。

在本文中，我们将继续探讨 ML 隐私，讨论某些问题，并深入研究差分隐私(DP)概念，这是解决隐私问题的方法之一。进一步列出五个您可以使用或贡献的开源差分隐私库或工具。

## 什么是数据，它是如何创建的？

数据是为参考或分析而收集的事实或统计数据。

我们几乎每天都在创造数据。可能是既有[在线又有](https://web.archive.org/web/20221206013925/https://www.acxiom.co.uk/blog/understand-your-online-and-offline-data-to-build-defined-data-strategy/)离线的数据。

比如医院的病人健康记录；学校或学院的学生信息；员工信息和项目绩效的公司内部日志；或者只是简单的记笔记就可以认为是*离线数据*。

然而，从连接到互联网的在线平台或应用程序收集的数据被视为*在线数据*，如发布推文、YouTube 视频或博客帖子，或收集用户表现数据的移动应用程序等。

### 隐私与安全

虽然敏感的个人数据(如癌症患者记录或合同追踪数据)对于数据科学家和分析师来说似乎是一座金矿，但这也引发了人们对收集此类数据的方法的担忧，而且谁来确保这些数据不会被用于恶意目的呢？

术语“隐私”和“安全”经常被混淆，但还是有区别的。安全控制"*、谁"*可以访问数据，而隐私更多的是关于"*、何时"和"什么"类型的*数据可以被访问。 ***“没有安全你不可能有隐私，但没有隐私你可以有安全。***

例如，我们都熟悉术语“登录认证和授权”。在这里，认证是关于谁可以访问数据，所以这是一个安全问题。授权是关于*什么，什么时候，以及如何* *大部分*数据对特定用户是可访问的，所以这是一个隐私问题。

### 私有安全机器学习(ML)

来自数据泄露和数据滥用的风险已经导致许多政府制定数据保护法。为了遵守数据隐私法并最小化风险，ML 研究人员提出了解决这些隐私和安全问题的技术，称为**私有和安全机器学习(ML)。**

正如 PyTorch 的这篇博文所说:

****私有和安全机器学习(ML)*** *深受密码学和隐私研究的启发。它由一组技术组成，这些技术允许在不直接访问数据的情况下训练模型，并防止这些模型无意中存储有关数据的敏感信息。”**

 *同一篇博文列举了一些应对不同隐私问题的常用技巧:

联合学习意味着根据存储在世界各地不同设备或服务器上的数据训练您的 ML 模型，而不必集中收集数据样本。

有时，人工智能模型可以记住他们训练过的数据的细节，并可能在以后“泄露”这些细节。差异隐私是一个衡量这种泄露并降低其发生风险的框架。

同态加密让你的数据不可读，但你仍然可以对它进行计算。

安全多方计算允许多方共同执行一些计算，并接收结果输出，而不会暴露任何一方的敏感输入。

当两方想要测试他们的数据集是否包含匹配值，但不想向对方“展示”他们的数据时，他们可以使用 PSI 来这样做。

虽然联合学习和差分隐私可以用来保护数据所有者免受隐私损失，但它们不足以保护模型免受数据所有者的盗窃或滥用。例如，联合学习要求模型所有者将模型的副本发送给许多数据所有者，从而通过数据中毒将模型置于知识产权盗窃或破坏的风险中。通过允许模型在加密状态下训练，加密计算可用于解决这种风险。最著名的加密计算方法是同态加密、安全多方计算和函数加密。

我们将重点关注差分隐私–让我们看看它是如何工作的，以及您可以使用哪些工具。

## 什么是差分隐私？

> “差分隐私描述了数据持有人或管理者对数据主体(所有者)做出的承诺，该承诺如下: ***通过允许在任何研究或分析中使用您的数据，您将不会受到不利影响或其他影响，无论有什么其他研究、数据集或信息源可用。”***
> 
> *–*[*辛西娅·德沃克，《差分隐私算法基础》*](https://web.archive.org/web/20221206013925/https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) *。*

差分隐私背后的直觉是“ ***我们限制如果改变数据库*** 中单个个体的数据，输出可以改变多少”。

也就是说，如果在同一个查询中从数据库中删除了某人的数据，会改变输出吗？如果是，那么对手能够分析它并找到一些辅助信息的可能性很高。简单来说——***隐私泄露！***

### 例如:

Adam 想知道向他的 XYZ 组织捐款的捐赠者的平均年龄。在这一点上，这可能看起来没问题！但是，这些数据也有可能被用来发现特定捐赠者的年龄。简单地说，1000 人进行了捐赠，结果发现捐赠者的平均年龄为 28 岁。

现在，仅通过从数据库中排除 John Doe 的数据进行相同的查询，让我们假设 999 个捐献者的平均值已经变为 28.007。从这一点，亚当可以很容易地发现，约翰多伊是 21 岁。(1000 * 28–999 * 28.007 = 21.007)同样，Adam 可以对其他捐献者重复该过程，以找到他们的实际年龄。

*注意:如果 Adam 可以进行逆向工程并获得他们的值，即使每个捐献者的年龄都被加密(例如同态加密),结果也是一样的。*

为了避免这种数据泄露，我们添加了受控数量的统计噪声来掩盖数据集中个体的数据贡献。

也就是说，捐献者被要求在提交之前或者在加密他们的年龄之前，给他们的原始年龄加上 100 到 100 之间的任何值。假设 John Doe 在他的原始年龄即 21 岁上加了-30，则加密前登记的年龄将是-9。

这听起来可能很疯狂？！但是，有趣的是，根据概率统计中的**大数定律**，可以看出，当对这些统计收集的数据取平均值时，噪声被抵消，并且获得的平均值接近于**真实平均值**(没有添加噪声的数据平均值(随机数))

现在，即使亚当对无名氏的年龄进行逆向工程，-9 也没有任何意义，因此，在保护无名氏隐私的同时允许亚当找到捐献者的平均年龄。

换句话说，**差分隐私是** ***不是*** **数据库的一个属性，而是查询的一个属性。t 有助于提供输出隐私，也就是说，通过对输出进行逆向工程，某人可以从输入中获得多少洞察力。**

在人工智能模型训练的情况下，添加了噪声，同时确保模型仍然能够洞察总体人口，从而提供足够准确的有用预测——同时让任何人都难以从查询的数据中获得任何意义。

*注:关于差分隐私的更多细节，请查看我的[差分隐私基础系列](https://web.archive.org/web/20221206013925/https://becominghuman.ai/what-is-differential-privacy-1fd7bf507049)。*

## 谁在使用差分隐私？

顶尖的科技公司 FAANGs，IBM，都在使用差别隐私，并且经常发布开源工具和库。

最有趣的例子是:

1.  [***RAPPOR***](https://web.archive.org/web/20221206013925/https://research.google/pubs/pub42852/) ，谷歌在这里使用本地差分隐私收集用户的数据，就像其他正在运行的进程和 Chrome 主页一样。
2.  [***Private Count Mean Sketch***](https://web.archive.org/web/20221206013925/https://machinelearning.apple.com/2017/12/06/learning-with-privacy-at-scale.html)(以及 variances)苹果利用本地差分隐私收集 iPhone 用户(iOS 键盘)表情符号使用数据、单词使用情况等信息的地方。
3.  [***隐私保护的个人健康数据流聚集***](https://web.archive.org/web/20221206013925/https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0207639) 论文提出了一种新的隐私保护的个人健康数据流收集机制，其特征在于利用局部差分隐私(Local DP)以固定间隔收集时态数据
4.  [***人口普查局为 2020 年人口普查采用尖端隐私保护***](https://web.archive.org/web/20221206013925/https://www.census.gov/newsroom/blogs/random-samplings/2019/02/census_bureau_adopts.html) 即美国人口普查局将在公布数据前使用差分隐私匿名。

更多信息，请查看我的[差分隐私基础系列](https://web.archive.org/web/20221206013925/https://becominghuman.ai/what-is-differential-privacy-1fd7bf507049)的[本地与全球 DP](https://web.archive.org/web/20221206013925/https://medium.com/@shaistha24/global-vs-local-differential-privacy-56b45eb22168) 博客。

## 我们真的需要它吗？为什么重要？

从上面 Adam 试图找到捐献者平均年龄的例子可以看出，加密本身不能保护个人的数据隐私，因为去匿名化是可能的。

一个这样的现实世界的例子将是对 Netflix 奖品数据集 的 **的 [**去匿名化，其中对单个订户仅了解一点点的对手可以容易地在数据集中识别该订户的记录。使用互联网电影数据库(IMDb)作为背景知识的来源，有可能成功地确定已知用户的网飞记录，揭示他们明显的政治偏好和其他潜在的敏感信息。**](https://web.archive.org/web/20221206013925/https://arxiv.org/pdf/cs/0610105.pdf)**

有时，由于深度神经网络的过度参数化导致不必要的数据泄漏，机器学习模型也可能无意中记住单个样本。

例如，设计用于发出预测文本(如智能手机上看到的下一个单词建议)[的语言模型可以被探测以发布有关用于训练的单个样本的信息](https://web.archive.org/web/20221206013925/https://www.usenix.org/system/files/sec19-carlini.pdf)(“我的 ID 是……”)。

这一领域的研究让我们可以计算隐私损失的程度，并根据隐私“预算”的概念对其进行评估。**最终，** **差分隐私的使用是隐私保护和模型实用性或准确性之间的谨慎权衡**。

## 差异隐私最佳实践

1.了解差别隐私承诺什么和不承诺什么总是很重要的。(参考: [*辛西娅·德沃克，《差分隐私的算法基础》*](https://web.archive.org/web/20221206013925/https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) **)**

*   差分隐私**承诺**保护个人免受由于他们的数据在私人数据库 *x* 中而可能面临的任何额外伤害，如果他们的数据不是 *x* 的一部分，他们就不会面临这些伤害。
*   差别隐私**并不能保证**一个人认为是自己的**秘密将会保持秘密。**也就是说，它承诺使数据成为不同的私有数据，不会泄露，但不会保护数据免受攻击者的攻击！
    Ex:差分攻击是最常见的隐私攻击形式之一。
*   它**仅仅确保**一个人参与调查**本身不会被披露**，如果保持有差别的私密性，参与调查也不会导致任何个人参与调查的细节被披露。

2.**差分隐私*不是*数据库的属性，而是查询的属性。**(如前所述)

3.添加的噪声量很重要，因为为使数据保密而添加的噪声越高，模型的实用性或准确性就越低。

4.了解局限性，例如:

*   差分隐私一直是学术界和研究界广泛探讨的话题，但由于其强大的隐私保障，在业界较少涉及。
*   如果要在 k 个查询中保持原始保证，则必须注入 k 次噪声。当 k 较大时，输出的效用被破坏。
*   对于一系列的查询，需要添加更多的噪音，这将耗尽隐私预算，并可能最终导致用户死亡。这意味着一旦隐私预算耗尽，用户就不再被允许询问任何更多的问题，如果你开始允许用户之间的勾结，你就开始陷入这个隐私预算对每个用户意味着什么的麻烦中。最终导致用户死亡。
*   单独的差分隐私不能保护用户数据，因为隐私攻击可能会发生！

是的，当然，可以采取各种方法来解决这个问题，但这超出了本博客的范围。

脸书的 Opacus 是一个面向任何人的库，这些人希望以最少的代码更改训练一个具有不同隐私的模型，或者用他们的 PyTorch 代码或纯 Python 代码快速原型化他们的想法。它也有很好的文档，作为一个开源库，如果你感兴趣，你也可以[贡献](https://web.archive.org/web/20221206013925/https://github.com/pytorch/opacus/blob/master/CONTRIBUTING.md)到它的代码库中。

加入 [PyTorch 论坛](https://web.archive.org/web/20221206013925/https://discuss.pytorch.org/)提出任何问题。

资源:

Google 提供了两个开源库(或存储库)wrt 差分隐私。

这是一个包含 3 个*构建块库*的存储库，用于针对适用于研究、实验或生产用例的 [C++](https://web.archive.org/web/20221206013925/https://github.com/google/differential-privacy/blob/main/cc) 、 [Go](https://web.archive.org/web/20221206013925/https://github.com/google/differential-privacy/blob/main/go) 和 [Java](https://web.archive.org/web/20221206013925/https://github.com/google/differential-privacy/blob/main/java) 支持的数据集生成ε-和(ε，δ)-差分私有统计数据。

而提供的其他工具，如 [Privacy on Beam](https://web.archive.org/web/20221206013925/https://github.com/google/differential-privacy/blob/main/privacy-on-beam) 、[random tester](https://web.archive.org/web/20221206013925/https://github.com/google/differential-privacy/blob/main/cc/testing)、[differential Privacy accounting library](https://web.archive.org/web/20221206013925/https://github.com/google/differential-privacy/blob/main/python/dp_accounting)和[命令行接口](https://web.archive.org/web/20221206013925/https://github.com/google/differential-privacy/blob/main/examples/zetasql)，用于运行带有 [ZetaSQL](https://web.archive.org/web/20221206013925/https://github.com/google/zetasql) 的 DP 查询，都是相当实验性的。

就像以前的图书馆一样，这个图书馆对投稿开放，并有一个公共讨论组。

这个库可以称为上面提到的 PyTorch Opacus 库的 TensorFlow 对应物，实现了 TensorFlow 优化器，用于训练具有差分隐私的机器学习模型。这也接受捐款，并有据可查。

资源:

另一个通用库用于[试验](https://web.archive.org/web/20221206013925/https://github.com/IBM/differential-privacy-library/tree/main/notebooks)，调查和开发差分隐私的应用程序，例如使用分类和聚类模型探索差分隐私对机器学习准确性的影响，或者只是开发一个新的应用程序。面向具有不同隐私知识的专家。

资源:

这是 Google 的 Java 差分隐私库的 Python 版本，提供了一组ε差分私有算法，用于生成包含私有或敏感信息的数字数据集的聚合统计数据。现在它被 OpenMined 的 [PySyft 库所支持。](https://web.archive.org/web/20221206013925/https://github.com/openmined/pysyft)

PyDP 团队正在积极招募成员，以进一步开发 Google Java 差分隐私库之外的库。

加入 [OpenMined Slack](https://web.archive.org/web/20221206013925/https://openmined.slack.com/) #lib_pydp 与团队互动，开始[贡献](https://web.archive.org/web/20221206013925/https://github.com/OpenMined/PyDP/blob/dev/contributing.md)。

资源:

这是 SmartNoise Project 和 OpenDP 之间的合作，旨在将学术知识应用到实际的部署中。

它提供了用于发布隐私保护查询和统计数据的不同私有算法和机制，以及用于定义分析的 API 和用于评估这些分析和组成数据集的总隐私损失的验证器。

也开放贡献，所以如果感兴趣，请随时加入。

资源:

虽然这个项目已被否决，没有得到维护，但它可以用于教育目的。它是为查询分析和重写框架而构建的，目的是为通用 SQL 查询实施不同的隐私保护。

## 摘要

如您所见，差分隐私是当今数据科学领域的一个重要话题，也是所有顶级科技巨头都关心的问题。

这不足为奇，因为在一个依靠数据运行的世界里，尽最大努力保护这些数据符合我们的利益。

如果你对差分隐私的更新感兴趣，[在 Twitter 上关注我](https://web.archive.org/web/20221206013925/https://twitter.com/shaistha24)，如果你还没有关注过的话，[也关注一下海王星](https://web.archive.org/web/20221206013925/https://twitter.com/neptune_ai)。

感谢阅读！

资源或额外读取:

1.  [opened](https://web.archive.org/web/20221206013925/https://www.openmined.org/)[隐私会议(pricon 2020)](https://web.archive.org/web/20221206013925/https://pricon.openmined.org/)–[视频](https://web.archive.org/web/20221206013925/https://www.youtube.com/watch?v=oM_cgRkN6MQ&list=PLUNOsx6Az_ZGKQd_p4StdZRFQkCBwnaY6)
2.  [为什么许多国家在 COVID 联系追踪方面失败了——但有些国家成功了](https://web.archive.org/web/20221206013925/https://www.nature.com/articles/d41586-020-03518-4)
3.  [患者数据存在哪些风险？](https://web.archive.org/web/20221206013925/https://understandingpatientdata.org.uk/weighing-up-risks)
4.  [关于公司如何使用大数据的 40 个统计数据和真实例子](https://web.archive.org/web/20221206013925/https://www.scnsoft.com/blog/big-data-use-cases-stats-and-examples)
5.  [了解联合学习的类型](https://web.archive.org/web/20221206013925/https://blog.openmined.org/federated-learning-types/)
6.  [谷歌联合学习漫画](https://web.archive.org/web/20221206013925/https://federated.withgoogle.com/)
7.  什么是安全多方计算？技术+代码实现。
8.  隐私和机器学习:两个意想不到的盟友？
9.  [差分密码分析教程](https://web.archive.org/web/20221206013925/http://theamazingking.com/crypto-diff.php)*