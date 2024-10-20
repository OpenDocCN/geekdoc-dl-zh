# 来自 ECML-PKDD 2020 会议的顶级研究论文

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/ecml-pkdd-2020-research>

上周，我有幸参加了 ECML-PKDD 2020 会议。欧洲机器学习和数据库中知识发现的原理和实践会议是**欧洲最受认可的关于 ML 的学术会议之一**。

完全在线的活动，24 小时不间断运行——让所有时区都可以参加的好主意。会议时间表被整齐地分成许多不同风格的部分，这使得我可以很容易地进入强化学习、对抗学习和元主题中我最喜欢的主题。

**ECML-PKDD** 在 ML 领域带来了大量新的想法和鼓舞人心的发展，所以我想**挑选顶级论文并在这里分享**。

在这篇文章中，我**主要关注研究论文，这些论文分为以下几类**:

尽情享受吧！

## 强化学习

### 1.自我映射:深层强化学习的投射映射和结构化自我中心记忆

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_122.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932251/egomap-projective-mapping-and-structured-egocentric-memory-for-deep-rl)

**论文摘要:**在部分可观测的 3D 环境中涉及定位、记忆和规划的任务是深度强化学习中的一个持续挑战。我们提出了 EgoMap，一种空间结构化的神经记忆体系结构。EgoMap 增强了深度强化学习代理在 3D 环境中执行具有多步目标的挑战性任务的性能。(…)

* * *

### 2.选项编码器:强化学习中发现策略基础的框架

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_584.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932334)

**论文摘要:**选项发现和技能获取框架对于分层组织的强化学习代理的功能是不可或缺的。然而，这样的技术通常会产生大量的选项或技能，可以通过过滤掉任何冗余信息来简洁地表示出来。这种减少可以减少所需的计算，同时还可以提高目标任务的性能。为了压缩一系列选项策略，我们试图找到一个策略基础，准确地捕捉所有选项的集合。在这项工作中，我们提出了 Option Encoder，这是一个基于自动编码器的框架，具有智能约束的权重，有助于发现基本策略的集合。(…)

主要作者:

* * *

### 3.ELSIM:通过内在动机进行可重用技能的端到端学习

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_530.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932323)

**论文摘要:**受发展学习的启发，我们提出了一种新的强化学习架构，它以端到端的方式分层学习和表示自我生成的技能。在这种架构下，智能体只关注任务奖励技能，同时保持技能的学习过程自下而上。这种自下而上的方法允许学习这样的技能:1-可以跨任务转移，2-当回报很少时可以提高探索能力。为此，我们将先前定义的交互信息目标与新颖的课程学习算法相结合，创建了一个无限的、可探索的技能树。(…)

第一作者:

**亚瑟奥伯莱特**

[GitHub](https://web.archive.org/web/20221207145212/https://github.com/Aubret)

* * *

### 4.基于图的运动规划网络

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_257.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932280)

**论文摘要:**可微分规划网络体系结构已经显示出在解决转移规划任务方面的强大功能，同时它具有简单的端到端训练特征。(…)然而，现有的框架只能在具有网格结构的域上进行有效的学习和规划，即嵌入在特定欧几里得空间中的正则图。在本文中，我们提出了一个通用的规划网络称为基于图的运动规划网络(GrMPN)。GrMPN 将能够 I)学习和规划一般的不规则图形，因此 ii)呈现现有规划网络架构的特殊情况。(…)

第一作者:

**泰皇**

## 使聚集

### 1.利用结构丰富的特征来改进聚类

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_605.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932339)

**论文摘要:**为了成功的聚类，算法需要找到聚类之间的边界。虽然如果聚类是紧凑的和非重叠的并且因此边界被清楚地定义，这是相对容易的，但是聚类相互融合的特征阻碍了聚类方法正确地估计这些边界。因此，我们的目标是提取显示清晰聚类边界的特征，从而增强数据中的聚类结构。我们的新技术创建了包含对聚类重要的结构的数据集的压缩版本，但是没有噪声信息。我们证明了数据集的这种转换对于 k-means 以及各种其他算法来说更容易聚类。此外，基于这些结构丰富的特征，我们引入了 k-means 的确定性初始化策略。(…)

第一作者:

本杰明·谢林

[LinkedIn](https://web.archive.org/web/20221207145212/https://www.linkedin.com/in/benjamin-schelling-b5b24b158/)

* * *

### 2.在线二元不完全多视图聚类

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_397.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932302)

**论文摘要:**在过去的几十年中，多视图聚类因其在多模态或来自不同来源的数据上的良好表现而吸引了相当多的关注。在现实应用中，多视图数据经常遭遇实例不完整的问题。在这样的多视图数据上进行聚类被称为不完全多视图聚类(IMC)。大多数现有的 IMC 解决方案是离线的，并且具有高的计算和存储成本，尤其是对于大规模数据集。为了应对这些挑战，本文提出了一个在线二元不完全多视图聚类框架。OBIMC 鲁棒地学习不完全多视图特征的公共紧凑二进制码。(…)

第一作者:

**杨**

* * *

### 3.简单、可扩展且稳定的可变深度聚类

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_785.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932370)

**论文摘要:**深度聚类(DC)已经成为无监督聚类的最新发展。原则上，DC 代表了各种无监督的方法，这些方法联合学习底层聚类和直接来自非结构化数据集的潜在表示。然而，由于高运营成本、低可扩展性和不稳定的结果，DC 方法通常应用不佳。在本文中，我们首先使用八个经验标准在工业适用性的背景下评估几个流行的 DC 变量。然后，我们选择关注变分深度聚类(VDC)方法，因为除了简单性、可伸缩性和稳定性之外，它们大多满足这些标准。(…)

* * *

### 4.高斯偏移:密度吸引子聚类比均值偏移更快

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_343.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932291)

**论文摘要:**均值漂移是一种流行而强大的聚类方法。虽然存在改进其绝对运行时间的技术，但是没有方法能够有效地改进其关于数据集大小的二次时间复杂度。为了能够开发一种可替代的、更快的方法来得到相同的结果，我们首先贡献了正式的聚类定义，这意味着隐式地跟随了 shift。基于这一定义，我们推导并贡献了高斯移位——一种具有线性时间复杂度的方法。我们使用具有已知拓扑的合成数据集来量化高斯偏移的特征。我们使用来自活跃神经科学研究的真实生活数据进一步限定高斯偏移，这是迄今为止对任何亚细胞器最全面的描述。

## 神经网络体系结构

### 1.在分类任务中寻找最佳网络深度

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_1036.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932407)

**论文摘要:**我们开发了一种使用多个分类器头训练轻量级神经网络的快速端到端方法。通过允许模型确定每个头部的重要性并奖励单个浅层分类器的选择，我们能够检测并移除网络中不需要的组件。该操作可以被视为找到模型的最佳深度，显著地减少了参数的数量并加速了跨不同硬件处理单元的推断，这对于许多标准剪枝方法来说并不是这样。(…)

主要作者:

**你把缰绳解开**

**子宫肌瘤**

* * *

### 2.XferNAS:转移神经结构搜索

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_239.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932275)

**论文摘要:**术语神经架构搜索(NAS)是指为一个新的、以前未知的任务自动优化网络架构。由于测试一个架构在计算上是非常昂贵的，许多优化者需要几天甚至几周的时间来找到合适的架构。但是，如果重用以前对不同任务进行搜索的知识，则可以大大减少搜索时间。在这项工作中，我们提出了一个普遍适用的框架，只需对现有的优化器进行微小的修改，就可以利用这一特性。(…)此外，我们在 CIFAR 基准测试中观察到 NAS 优化器分别获得了 1.99 和 14.06 的新纪录。在另一项研究中，我们分析了源数据和目标数据量的影响。(…)

* * *

### 3.稀疏神经网络的拓扑洞察

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_988.pdf)|[Prese](https://web.archive.org/web/20221207145212/https://slideslive.com/38932397)[n](https://web.archive.org/web/20221207145212/https://slideslive.com/38932397)[station](https://web.archive.org/web/20221207145212/https://slideslive.com/38932397)

**论文摘要:**稀疏神经网络是降低深度神经网络部署资源需求的有效途径。最近，出现了自适应稀疏连接的概念，以允许通过在训练期间优化稀疏结构来从头开始训练稀疏神经网络。(…)在这项工作中，我们从图论的角度介绍了一种理解和比较稀疏神经网络拓扑的方法。我们首先提出神经网络稀疏拓扑距离(NNSTD)来度量不同稀疏神经网络之间的距离。此外，我们证明了稀疏神经网络在性能方面可以胜过过参数化模型，即使没有任何进一步的结构优化。(…)

主要作者:

## 迁移和多任务学习

### 1.图形扩散 Wasserstein 距离

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_558.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932329) [t](https://web.archive.org/web/20221207145212/https://slideslive.com/38932329) [离子](https://web.archive.org/web/20221207145212/https://slideslive.com/38932329)

**论文摘要**:结构化数据的最优传输(OT)在机器学习社区中受到了很多关注，尤其是在解决图分类或图转移学习任务方面。在本文中，我们提出了扩散 Wasserstein (DW)距离，作为对无向图和连通图的标准 Wasserstein 距离的推广，其中节点由特征向量描述。DW 基于拉普拉斯指数核，并受益于热扩散来从图中捕捉结构和特征信息。(…)

第一作者:

**改善胡子**

* * *

### 2.使用双层规划实现可解释的多任务学习

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_826.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932376)

**论文摘要:**全局可解释多任务学习可以表示为基于学习模型的预测性能来学习任务关系的稀疏图。我们提出了学习稀疏图的回归多任务问题的双层模型。我们表明，这种稀疏图提高了学习模型的可解释性。

第一作者:

**弗朗切斯科·阿莱西亚尼**

[领英](https://web.archive.org/web/20221207145212/https://www.linkedin.com/in/francesco-alesiani-2b48b74/?originalSubdomain=de)

* * *

### 3.领域转移下基于多样性的无监督文本分类泛化

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_1133.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932419)

**论文摘要:**领域适应方法寻求从源领域学习并将其推广到未知的目标领域。(…)在本文中，我们提出了一种用于单任务文本分类问题的领域适应的新方法，该方法基于基于多样性的泛化的简单而有效的思想，该思想不需要未标记的目标数据。多样性通过迫使模型不依赖于相同的特征来进行预测，起到了促进模型更好地概括和不加选择地进行域转换的作用。我们将这一概念应用于神经网络中最容易解释的部分，即注意力层。(…)

* * *

### 4.深度学习、语法迁移和迁移理论

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_609.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932340)

**论文摘要:**尽管基于深度学习的人工智能技术被广泛采用并取得了成功，但在提供可理解的决策过程方面存在局限性。这使得“智能”部分受到质疑，因为我们希望真正的人工智能不仅能完成给定的任务，还能以人类可以理解的方式执行。为了实现这一点，我们需要在人工智能和人类智能之间建立联系。在这里，我们用语法迁移来展示连接这两种智力的范例。(…)

第一作者:

**张凯旋**

* * *

### 5.具有联合领域对抗重构网络的无监督领域适应

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_249.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932277)

**论文摘要:**无监督领域自适应(UDA)试图将知识从已标记的源领域转移到未标记的目标领域。(…)我们在本文中提出了一种称为联合领域对抗重构网络(JDARN)的新模型，它将领域对抗学习与数据重构相结合，以学习领域不变和领域特定的表示。同时，我们提出了两种新的鉴别器来实现联合比对，并采用了一种新的联合对抗损失来训练它们。(…)

第一作者:

**钱晨**

## 联合学习和聚类

### 1.分散矩阵分解的算法框架

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_1259.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932431)

**论文摘要:**我们提出了一个完全去中心化机器学习的框架，并将其应用于 top-N 推荐的潜在因素模型。分散学习环境中的训练数据分布在多个代理上，这些代理共同优化一个共同的全局目标函数(损失函数)。这里，与联合学习的客户机-服务器体系结构相反，代理直接通信，维护和更新它们自己的模型参数，没有中央集合，也没有共享它们自己的数据。(…)

主要作者:

艾丽卡·杜利亚科娃

**黄伟鹏**

* * *

### 2.用于个性化推荐的联合多视图矩阵分解

[Pap](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_480.pdf)[e](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_480.pdf)[r](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_480.pdf)|[演示文稿](https://web.archive.org/web/20221207145212/https://slideslive.com/38932315)

**论文摘要:**我们介绍了联邦多视图矩阵分解方法，该方法将联邦学习框架扩展到具有多个数据源的矩阵分解。我们的方法能够学习多视图模型，而无需将用户的个人数据传输到中央服务器。据我们所知，这是第一个使用多视图矩阵分解提供推荐的联邦模型。该模型在生产环境的三个数据集上进行了严格评估。(…)

* * *

### 3.FedMAX:减少激活分歧，实现精确和高效的联合学习

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_953.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932391)

**论文摘要:**在本文中，我们发现了一种称为激活-发散的新现象，这种现象在联邦学习中由于数据异构而发生。具体来说，我们认为，当使用联合学习时，激活向量可能会出现分歧，即使一个用户子集与驻留在不同设备上的数据共享一些公共类。为了解决这个问题，我们引入了基于最大熵原理的先验知识；该先验假设关于每个设备的激活向量的信息最少，并且旨在使相同类别的激活向量在多个设备上相似。(…)

第一作者:

**陈为**

* * *

### 4.使用 HDBSCAN*进行基于模型的聚类

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_658.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932349)

**论文摘要:**我们提出了一种有效的基于模型的聚类方法，用于从有限数据集创建高斯混合模型。使用分类可能性和期望最大化算法从 HDBSCAN*层次结构中提取模型。对应于聚类数量的模型组件数量的先验知识不是必需的，并且可以动态确定。由于与以前的方法相比，HDBSCAN*创建的层次相对较小，因此可以高效地完成这项工作。(…)

第一作者:

迈克尔施特罗布尔

[GitHub](https://web.archive.org/web/20221207145212/https://github.com/mjstrobl)

## 网络建模

### 1.节点分类的渐进监督

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_221.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932270)

**论文摘要:**图卷积网络(GCNs)是一种用于节点分类任务的强大方法，其中通过最小化最终层预测的损失来训练 GCNs。然而，这种训练方案的局限性在于，它强制每个节点从感受野的固定和统一大小中被分类，这可能不是最佳的。我们提出了 ProSup(渐进式监督),通过以不同的方式培训 gcn 来提高其有效性。ProSup 逐步监督所有层，引导它们的表现向我们期望的特征发展。(…)

第一作者:

**王义桅**

* * *

### 2.基于时间 RNN 的分层注意动态异构网络链路预测模型

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_458.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932312)

**论文摘要:**网络嵌入旨在学习节点的低维表示，同时捕捉网络的结构信息。(…)在本文中，我们提出了一种新的动态异构网络嵌入方法，称为 DyHATR，它使用分层注意来学习异构信息，并将递归神经网络与时间注意相结合来捕获进化模式。(…)

* * *

### 3.GIKT:基于图的知识追踪交互模型

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_1250.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932429)

**论文摘要:**随着在线教育的快速发展，知识追踪(KT)已经成为追踪学生知识状态和预测他们在新问题上表现的基本问题。在线教育系统中的问题通常很多，并且总是与很少的技能相关联。(…)在本文中，我们提出了一个基于图的知识追踪交互模型(GIKT)来解决上述问题。更具体地说，GIKT 利用图卷积网络(GCN)通过嵌入传播来充分结合问题-技能相关性。(…)

第一作者:

**杨洋**

## 图形神经网络

### 1.格拉姆-SMOT:基于图关注机制和子模块优化的 Top-N 个性化捆绑推荐

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_688.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932355)

**论文摘要:**捆绑推荐——向客户推荐一组产品来代替单个产品，日益受到关注。它提出了两个有趣的挑战——(1)如何有效地向用户推荐现有的捆绑包，以及(2)如何生成针对特定用户的个性化小说捆绑包。(……)在这项工作中，我们提出了 GRAM-SMOT——一种基于图形注意力的框架，以解决上述挑战。此外，我们定义了一个基于度量学习方法的损失函数来有效地学习实体的嵌入。(…)

第一作者:

********维杰库玛米********

* * *

### 2.用于下一个项目推荐的时间异构交互图嵌入

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_678.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932353)

**论文摘要:**在下一个项目推荐的场景中，以前的方法试图通过捕捉顺序交互的演变来建模用户偏好。然而，它们的顺序表达往往是有限的，没有对短期需求往往会受到长期习惯影响的复杂动态进行建模。此外，它们很少考虑用户和项目之间的异构类型的交互。在本文中，我们将这种复杂数据建模为时间异构交互图(THIG ),并学习 THIGs 上的用户和项目嵌入，以解决下一个项目推荐。主要的挑战包括两个方面:复杂的动力学和丰富的交互异构性。(…)

* * *

### 3.一种基于自关注网络的节点嵌入模型

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_231.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932272)

**论文摘要:**尽管最近取得了一些进展，但对归纳设置进行的研究有限，在归纳设置中，新出现的未知节点需要嵌入——这是图网络深度学习的实际应用中常见的设置。(…)为此，我们提出了 SANNE——一种新的无监督嵌入模型——其中心思想是采用一种自我注意机制，后跟一个前馈网络，以便迭代地聚合采样随机行走中节点的矢量表示。(…)

* * *

### 4.图形修正卷积网络

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_626.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932343)

**论文摘要:**图卷积网络(GCNs)在机器学习社区中受到越来越多的关注，因为它在各种应用中有效地利用了节点的内容特征和跨图的链接模式。(…)本文提出了一个称为图修正卷积网络(GRCN)的新框架，它避免了这两个极端。具体而言，引入了基于 GCN 的图修正模块，用于通过联合优化来预测缺失边和修正下游任务的边权重。(…)

* * *

### 5.基于潜在扰动的图卷积网络鲁棒训练

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_554.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932327)

**论文摘要:**尽管最近图卷积网络(GCNs)在对图结构数据建模方面取得了成功，但其对敌对攻击的脆弱性已经暴露出来，针对节点特征和图结构的攻击已经被设计出来。(…)我们建议通过扰动 GCNs 中的潜在表示来解决这个问题，这不仅免除了生成敌对网络，而且通过尊重数据的潜在流形来实现改进的鲁棒性和准确性。这种在图上进行潜在对抗训练的新框架被应用于节点分类、链接预测和推荐系统。(…)

## 自然语言处理

### 1.多源弱社会监督下的假新闻早期发现

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_217.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932268)

**论文摘要:**社交媒体极大地让人们以前所未有的速度参与到网络活动中。然而，这种无限制的访问也加剧了错误信息和虚假新闻在网上的传播，除非及早发现并加以缓解，否则可能会造成混乱和混乱。(…)在这项工作中，我们利用来自用户和内容约定的不同来源的多个弱信号及其互补效用来检测假新闻。我们共同利用有限数量的干净数据以及来自社交活动的弱信号，在元学习框架中训练深度神经网络，以估计不同弱实例的质量。(…)

* * *

### 2.通过多重编辑神经网络从宏观新闻中生成财务报告

[帕佩](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_814.pdf) [r](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_814.pdf) | [演讲](https://web.archive.org/web/20221207145212/https://slideslive.com/38932374)

**论文摘要:**给定一条突发的宏观新闻，自动生成财务报告是相当具有挑战性的任务。本质上，该任务是文本到文本的生成问题，但是要从一条短新闻中学习长文本，即大于 40 个单词。(…)为了解决这个问题，我们提出了新的多重编辑神经网络方法，该方法首先学习给定新闻的大纲，然后根据学习的大纲生成财务报告。具体来说，输入新闻首先通过跳格模型嵌入，然后输入双 LSTM 组件训练上下文表示向量。(…)

第一作者:

**任**

* * *

### 3.面向短文本聚类的归纳文档表示学习

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_353.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932296)

**论文摘要:**短文本聚类(STC)是一项重要的任务，可以在快速增长的社交网络中发现主题或群组，例如 Tweets 和 Google News。(…)受 GNNs 中图结构引导的顶点信息传播机制的启发，我们提出了一个称为 IDRL 的归纳文档表示学习模型，该模型可以将短文本结构映射到图网络中，并递归地聚合看不见的文档中单词的邻居信息。然后，我们可以用以前学习的有限数量的单词嵌入来重建以前看不见的短文本的表示。(…)

第一作者:

**陈**

* * *

### 4.用于文档级情感分析的具有反思机制的分层交互网络

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_210.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932263)

**论文摘要:**由于模糊的语义链接和复杂的情感信息，文档级情感分析(DSA)更具挑战性。最近的工作致力于利用文本摘要，并取得了可喜的成果。然而，这些基于摘要的方法没有充分利用摘要，包括忽略摘要和文档之间的内在交互。(…)在本文中，我们研究了如何有效地为 DSA 生成具有显式主题模式和情感上下文的区别性表示。提出了一种层次交互网络(HIN)来探索摘要和文档在多个粒度上的双向交互，并学习面向主题的文档表示用于情感分类。(…)

第一作者:

**魏**

* * *

### 5.学习一系列情感分类任务

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_1205.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932425)

**论文摘要:**研究终身学习环境下的情感分类，以提高情感分类的准确性。在 LL 设置中，系统在神经网络中递增地学习一系列 SC 任务。这种情况在情感分析应用程序中很常见，因为情感分析公司需要为不同的客户处理大量的任务。(…)本文提出了一种称为 KAN 的新技术来实现这些目标。KAN 通过前向和后向知识转移，可以显著提高新旧任务的 SC 准确性。(…)

第一作者:

**Zixuan Ke**

## 时间序列和递归神经网络

### 1.用于时间序列分类的时间字典集成(TDE)分类器

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_258.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932281)

**论文摘要:**用词袋表示时间序列是一种流行的时间序列分类方法。这些算法包括在一系列上近似和离散化窗口以形成单词，然后在给定的字典上形成单词计数。在得到的单词计数直方图上构建分类器。2017 年对一系列时间序列分类器的评估发现，符号-傅立叶近似符号包(BOSS)集成是基于字典的分类器中最好的。(…)我们提出了这些基于字典的分类器的进一步扩展，它将其他分类器的最佳元素与一种基于参数空间的自适应高斯过程模型构建集成成员的新方法相结合。(…)

* * *

### 2.利用多尺度动态记忆的递归神经网络的增量训练

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_1069.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932409)

**论文摘要:**递归神经网络的有效性很大程度上受到其将从不同频率和时间尺度的输入序列中提取的信息存储到其动态记忆中的能力的影响。(…)在本文中，我们提出了一种新的增量训练递归体系结构，明确针对多尺度学习。首先，我们展示了如何通过将简单 RNN 的隐藏状态分成不同的模块来扩展其架构，每个模块以不同的频率对网络隐藏激活进行子采样。然后，我们讨论一种训练算法，其中新模块被迭代地添加到模型中，以学习逐渐变长的依赖性。(…)

* * *

### 3.柔性递归神经网络

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_1087.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932414)

**论文摘要:**我们介绍了两种方法，使得递归神经网络(RNNs)能够在序列分析过程中以计算成本来权衡准确性。(…)第一种方法对模型的改动很小。因此，它避免了从慢速存储器加载新参数。在第二种方法中，不同的模型可以在序列分析中相互替换。后者适用于更多数据集。(…)

主要作者:

* * *

### 4.z 嵌入:用于有效聚类和分类的事件区间的谱表示

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_649.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932347)

**论文摘要:**事件区间序列出现在多个应用领域，而其固有的复杂性阻碍了聚类和分类等任务的可扩展解决方案。本文提出了一种新的依赖于二部图的事件区间序列的谱嵌入表示。更具体地，每个事件区间序列通过以下三个主要步骤由二分图表示:(1)创建可以将事件区间序列的集合快速转换成二分图表示的哈希表，(2)创建并正则化对应于二分图的双邻接矩阵，(3)定义双邻接矩阵上的谱嵌入映射。(…)

## 维数缩减和自动编码器

### 1.具有一跳线性模型的简单有效的图形自动编码器

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_259.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932282)

**论文摘要:**在过去的几年里，图自动编码器(AE)和变分自动编码器(VAE)作为强大的节点嵌入方法出现，(…)。图形 AE、VAE 及其大多数扩展依赖于多层图形卷积网络(GCN)编码器来学习节点的向量空间表示。在本文中，我们表明，GCN 编码器实际上是不必要的复杂的许多应用。我们建议用图的直接邻域(一跳)邻接矩阵的更简单和更易解释的线性模型来代替它们，包括更少的操作、更少的参数和没有激活函数。(…)

* * *

### 2.稀疏可分非负矩阵分解

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_453.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932310)

**论文摘要:**我们提出了一种新的非负矩阵分解(NMF)变体，结合了可分性和稀疏性假设。可分离性要求第一 NMF 因子的列等于输入矩阵的列，而稀疏性要求第二 NMF 因子的列是稀疏的。我们称这种变体为稀疏可分 NMF (SSNMF)，我们证明它是 NP 难的，而不是可以在多项式时间内求解的可分 NMF。(…)

## 大规模优化和差分隐私

### 1.L1-正则化优化的正交近似随机梯度方法

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_407.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932304)

**论文摘要:**稀疏诱导正则化问题在机器学习应用中普遍存在，范围从特征选择到模型压缩。在本文中，我们提出了一种新的随机方法——基于蚂蚁的近似随机梯度方法(OBProx-SG)——来解决可能是最流行的实例，即 l1 正则化问题。OBProx-SG 方法包含两个步骤:(I)预测解的支撑覆盖的近似随机梯度步骤；以及(ii)另一个步骤，通过另一个面部投影积极地增强清晰度水平。(…)

* * *

### 2.结构化非凸优化的坐标下降法的效率

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_417.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932306)

**论文摘要:**提出了新的坐标下降(CD)方法，用于最小化由三项组成的非凸函数:(I)连续 diﬀerentiable 项，(ii)简单凸项，以及(iii)凹连续项。首先，通过将随机 CD 扩展到非光滑非凸环境，我们发展了一种坐标次梯度方法，该方法通过使用块复合次梯度映射来随机更新块坐标变量。(…)第二，我们开发了一个随机置换 CD 方法，它有两个交替的步骤:线性化凹形部分和循环变量。(…)第三，我们将加速坐标下降法(ACD)推广到非光滑和非凸优化中，发展了一种新的随机近似 DC 算法，从而用 ACD 不精确地求解子问题。(…)

第一作者:

**齐登**

* * *

### 3.基于 DP-信赖域方法的经验风险鞍点的私密可伸缩规避

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_604.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932338)

**论文摘要:**最近有研究表明，机器学习和深度学习中的许多非凸目标/损失函数已知是严格鞍点。这意味着找到二阶驻点({em 即，}近似局部最小值)并因此避开鞍点对于这样的函数来说足以获得具有良好推广性能的分类器。然而，现有的鞍点逃逸算法都没有考虑到设计中的一个关键问题，即训练集中敏感信息的保护。(…)本文研究了非凸损失函数的经验风险的私逃鞍点和寻找二阶驻点问题。(…)

## 对抗性学习

### 1.对抗性学习分子图的推理和生成

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_114.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932249)

**论文摘要:**最近用于产生新分子的方法使用分子的图形表示，并采用各种形式的图形卷积神经网络进行推理。然而，训练需要解决一个昂贵的图同构问题，以前的方法没有解决或只是近似地解决。在这项工作中，我们提出了 ALMGIG，一个用于推理和从头分子生成的无似然对抗性学习框架，它避免了显式计算重建损失。我们的方法扩展了生成对抗网络，通过包含对抗循环一致性损失来隐含地执行重建属性。(…)

* * *

### 2.可解释人工智能的通用模型不可知样本合成框架

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_972.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932394)

**论文摘要:**随着深度学习方法在实际应用中的日益复杂，解释和诠释这种方法的决策的需求越来越迫切。在这项工作中，我们专注于可解释的人工智能，并提出一种新的通用和模型不可知的框架，用于合成输入样本，最大化机器学习模型的期望响应。为此，我们使用一个生成模型，该模型作为生成数据的先验，并使用一种新的具有动量更新的进化策略遍历其潜在空间。(…)

* * *

### 3.通过无监督对抗攻击的自动编码器的质量保证

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_969.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932393)

**论文摘要:**自动编码器是无监督学习中的一个基本概念。目前，自动编码器的质量要么在内部评估(例如，基于均方误差)，要么在外部评估(例如，通过分类性能)。然而，无法证明自动编码器的泛化能力超过了有限的训练数据，因此，对于要求对看不见的数据也有正式保证的安全关键应用来说，它们是不可靠的。为了解决这个问题，我们提出了第一个框架，将自动编码器的最坏情况错误限制在无限值域的安全关键区域内，以及导致这种最坏情况错误的无监督对抗示例的定义。(…)

* * *

### 4.显著图与对抗鲁棒性

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_411.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932305)

**论文摘要:**最近出现了一种趋势，即将可解释性和对抗稳健性的概念结合起来，而不是像早期那样只关注良好的解释或对抗对手的稳健性。(…)在这项工作中，我们提供了一个不同的角度来看待这种耦合，并提供了一种方法，即基于显著性的对抗训练(SAT)，以使用显著性图来提高模型的对抗鲁棒性。特别地，我们表明，使用已经与数据集一起提供的注释(如边界框和分割掩模)作为弱显著图，足以提高对抗鲁棒性，而无需额外的努力来生成扰动本身。(…)

* * *

### 5.神经网络中的可扩展后门检测

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_425.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932308)

**论文摘要:**最近有研究表明，深度学习模型容易受到木马攻击。在特洛伊木马攻击中，攻击者可以在训练期间安装后门，使模型错误识别被小触发补丁污染的样本。当前的后门检测方法不能实现良好的检测性能，并且计算量大。在本文中，我们提出了一种新的基于触发逆向工程的方法，其计算复杂度不随标签数量的增加而增加，并且基于跨不同网络和补丁类型的可解释和通用的度量。(…)

## 深度学习理论

### 1.答:激活异常分析

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_576.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932333)

**论文摘要:**受最近神经网络覆盖引导分析进展的启发，我们提出了一种新的异常检测方法。我们表明，隐藏的激活值包含有助于区分正常和异常样本的信息。我们的方法在一个纯数据驱动的端到端模型中结合了三个神经网络。基于目标网络中的激活值，警报网络决定给定样本是否正常。多亏了异常网络，我们的方法甚至可以在严格的半监督环境下工作。(…)

主要作者:

### 2.卷积神经网络的有效版本空间缩减

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_1075.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932412)

**论文摘要:**在主动学习中，采样偏差会造成严重的不一致性问题，阻碍算法找到最优假设。然而，许多方法是假设空间不可知的，没有解决这个问题。我们通过版本空间缩减的原则镜头检查深度神经网络的主动学习，并检查可实现性假设。基于他们的目标，我们确定了以前的质量减少和直径减少方法之间的核心差异，并提出了一种新的基于直径的查询方法-吉布斯投票分歧。(…)

主要作者:

* * *

### 3.通过记忆驱动的通信提高小规模多智能体深度强化学习的协调性

[论文](https://web.archive.org/web/20221207145212/https://link.springer.com/article/10.1007/s10994-019-05864-5) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932466)

**论文摘要:**深度强化学习算法最近被用于以集中的方式训练多个相互作用的代理，同时保持它们的执行是分散的。当智能体只能获得部分观察值，并且面临需要协调和同步技能的任务时，智能体间的通信起着重要的作用。在这项工作中，我们提出了一个使用深度确定性策略梯度的多代理训练框架，该框架允许通过存储设备对显式通信协议进行并行的端到端学习。在训练期间，代理学习执行读和写操作，使他们能够推断世界的共享表示。(…)

主要作者:

* * *

### 4.神经网络训练的最小作用原理

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_1293.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932433)

**论文摘要:**尽管高度过度参数化，但神经网络已经在许多任务上实现了高泛化性能。由于经典的统计学习理论难以解释这种行为，最近许多努力都集中在揭示其背后的机制上，希望开发一个更合适的理论框架，并对训练好的模型进行更好的控制。在这项工作中，我们采用了另一种观点，将神经网络视为一个动态系统，随着时间的推移取代输入粒子。我们进行了一系列的实验，并通过分析网络的位移行为，我们表明在网络的传输图中存在低动能偏差，并将这种偏差与泛化性能联系起来。(…)

## 计算机视觉/图像处理

### 1.用于人脸识别的同伴引导的软边界

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_529.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932322)

**论文摘要:**人脸识别在基于角度裕度的 softmax 损失的帮助下取得了显著的进步。然而，在训练过程中，间隔通常是手动设置并保持不变的，这忽略了优化难度和不同实例之间的信息相似性结构。(…)在本文中，我们从超球流形结构的角度提出了一种新的样本自适应边缘损失函数，我们称之为同伴引导软边缘(CGSM)。CGSM 在特征空间中引入分布信息，并在每个小批内进行师生优化。(…)

* * *

### 2.用于无监督领域适应的具有区别表示学习的软标签转移

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_839.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932378)

**论文摘要:**领域自适应旨在解决将从标签信息丰富的源领域获得的知识转移到标签信息较少甚至没有标签信息的目标领域的挑战。最近的方法开始通过结合目标样本的硬伪标签来解决这个问题，以更好地减少跨域分布偏移。然而，这些方法易受误差累积的影响，因此无法保持跨领域类别的一致性。(…)为了解决这个问题，我们提出了一个具有判别表示学习的软标签转移(SLDR)框架，以联合优化具有软目标标签的类别式适应，并在一个统一的模型中学习判别域不变特征。(…)

第一作者:

**曹曼良**

* * *

### 3.显著区域发现的信息瓶颈方法

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_717.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932364)

**论文摘要:**基于信息瓶颈原理，我们提出了一种在半监督环境下学习图像注意力掩模的新方法。在提供了一组标记图像的情况下，屏蔽生成模型最小化输入和屏蔽图像之间的互信息，同时最大化同一屏蔽图像和图像标签之间的互信息。与其他方法相比，我们的注意力模型产生一个布尔型而不是连续的掩蔽，完全隐藏了被掩蔽的像素中的信息。(…)

* * *

### 4.FAWA:对光学字符识别(OCR)系统的快速对抗性水印攻击

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_1276.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932432)

**论文摘要:**光学字符识别(OCR)作为一种关键的预处理工具，广泛应用于信息抽取和情感分析等实际应用中。在 OCR 中采用深度神经网络(DNN)会导致针对恶意示例的漏洞，恶意示例是为了误导威胁模型的输出而精心制作的。针对白盒 OCR 模型提出了快速水印对抗攻击(FAWA ),在水印的伪装下产生自然失真，逃避人眼的检测。本文首次尝试将普通的对抗性扰动和水印一起应用于对抗性攻击，生成对抗性水印。(…)

第一作者:

**陈箓**

## 深度学习的优化

### 1.ADMMiRNN:通过有效的 ADMM 方法训练稳定收敛的 RNN

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_224.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932271)

**论文摘要:**由于递归单元中的权值在迭代之间是重复的，因此很难训练出稳定收敛的递归神经网络(RNN)并避免梯度消失和爆炸。此外，RNN 对权重和偏差的初始化很敏感，这给训练阶段带来了困难。与传统的随机梯度算法相比，交替方向乘子法(ADMM)具有无梯度特性和对恶劣条件的免疫能力，是一种很有前途的神经网络训练算法。然而，ADMM 不能直接用于训练 RNN，因为循环单元中的状态在时间步长上重复更新。因此，本文在 RNN 展开形式的基础上构建了一个新的框架 ADMMiRNN，以同时解决上述挑战，并提供了新的更新规则和理论收敛性分析。(…)

第一作者:

**玉堂**

* * *

### 2.网络零和凹对策中梯度方法的指数收敛性

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_719.pdf) | [预](https://web.archive.org/web/20221207145212/https://slideslive.com/38932366) [论文](https://web.archive.org/web/20221207145212/https://slideslive.com/38932366) [论文](https://web.archive.org/web/20221207145212/https://slideslive.com/38932366)

**论文摘要:**本文以生成型竞争网络为研究对象，研究了凹型网络零和博弈(NZGSs)中纳什均衡的计算..推广了蔡等人的结果，我们证明了凹凸两人零和对策的各种博弈论性质在这一推广中是保持的。然后我们推广了以前在两人零和博弈中得到的最后迭代收敛结果。(…)

* * *

### 3.神经网络优化的自适应动量系数

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_1005.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932402)

**论文摘要:**我们提出了一种新颖有效的基于动量项的一阶神经网络优化算法。我们的算法称为{it 自适应动量系数} (AMoC)，它利用梯度的内积和参数的先前更新，根据优化路径中方向的变化，有效地控制动量项的权重。该算法易于实现，并且其计算开销比动量法小得多。(…)

* * *

### 4.用于资源高效深度神经网络的压缩相关神经元

[论文](https://web.archive.org/web/20221207145212/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/RT/sub_1120.pdf) | [简报](https://web.archive.org/web/20221207145212/https://slideslive.com/38932418)

**论文摘要:**由于 dnn 在挑战性问题上的准确性，它们在现实生活中有大量的应用，然而它们对内存和计算成本的要求挑战了它们在资源受限环境中的适用性。迄今为止，驯服计算成本一直集中在一阶技术上，例如通过数值贡献度量优先化来消除数值上无关紧要的神经元/滤波器，从而产生可接受的改进。然而，DNNs 中的冗余远远超出了数值无关紧要的界限。(…)为此，我们采用了实用的数据分析技术，并结合了一种新颖的特征消除算法，以确定一个最小的计算单元集，从而捕获图层的信息内容并压缩其余内容。(…)

## 摘要

就是这样！

我个人建议您也访问活动网站，更深入地探索您最感兴趣的话题。

请注意，还有另一篇文章提供了最好的应用数据科学论文，敬请关注！

如果你觉得少了什么很酷的东西，简单地告诉我，我会延长这个帖子。