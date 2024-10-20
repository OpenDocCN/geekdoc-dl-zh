# 来自 ECML-PKDD 2020 会议的顶级“应用数据科学”论文

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/ecml-pkdd-2020-applied-data-science>

上周，我参加了 ECML-PKDD 2020 会议。欧洲机器学习和数据库中知识发现的原理和实践会议是**欧洲最受认可的关于 ML 的学术会议之一**。

本着传播 ML 发展的精神，我想分享我从会议中挑选的最好的“应用数据科学”论文。这是这个系列的第二篇文章。上一篇关于顶级研究的论文，可以在[这里找到](/web/20221206013611/https://neptune.ai/blog/ecml-pkdd-2020-research)。一定要检查它。

*应用数据科学*论文和演示对纯研究论文是一个很好的补充。由于这一点，会议在研究和行业主题之间取得了平衡。

在本帖**中，论文按照会议计划**进行分类:

尽情享受吧！

## 运动

### 1.停止时钟:超时效应是真实的吗？

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_667.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932351)

**论文摘要:**暂停是游戏过程中的短暂中断，用于传达策略的变化，让玩家休息或停止游戏中的负面流。(…)但是，这些超时在这方面的效果如何呢？暂停前后得分差异的简单平均值被用作存在影响且影响显著的证据。我们声称这些统计平均值不是适当的证据，需要一个更合理的方法。我们应用了一个正式的因果框架，使用了一个大型的 NBA 官方比赛列表数据集，并在因果图中绘制了我们对数据生成过程的假设。(…)

* * *

### 2.基于目标检测和 LSTM 的足球视频流自动传球标注

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_1083.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932413)

**论文摘要:**由于可以获得描述每场比赛中发生的所有时空事件的数据，足球分析正在吸引学术界和工业界越来越多的兴趣。(…)在本文中，我们描述了 PassNet，这是一种从视频流中识别足球中最常见事件(即传球)的方法。我们的模型结合了一组人工神经网络，这些网络从视频流中执行特征提取，对象检测以识别球和球员的位置，并将帧序列分类为传球或不传球。(…)

* * *

### 3.SoccerMix:用混合模型表示足球动作

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_946.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932389)

**论文摘要:**分析比赛风格是足球分析中的一项重复任务，在俱乐部活动中起着至关重要的作用，如球员球探和比赛准备。(…)当前用于分析比赛风格的技术经常受到足球事件流数据的稀疏性的阻碍(即，同一球员很少在同一位置多次执行相同的动作)。本文提出了 SoccerMix，一种基于混合模型的软聚类技术，实现了一种新的足球动作概率表示。(…)

主要作者:

* * *

### 4.SoccerMap:一个用于足球可视化分析的深度学习架构

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_1006.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932403)

**论文摘要:**我们提出了一种全卷积神经网络架构，它能够从高频时空数据中估计足球中潜在传球的全概率表面。该网络接收多层低级输入，并学习在不同采样级别生成预测的要素等级，从而捕捉粗略和精细的空间细节。通过合并这些预测，我们可以为任何比赛情况生成视觉上丰富的概率表面，使教练能够对球员的定位和决策进行精细分析，这是体育领域中迄今为止很少探索的领域。(…)

## 硬件和制造

### 1.学习 I/O 访问模式以提高固态硬盘的预取性能

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_627.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932344)

**论文摘要:**基于闪存的固态硬盘已经成为云计算和移动环境中硬盘的高性能替代产品。然而，固态硬盘仍然是计算机系统的性能瓶颈，因为它们的 I/O 访问延迟很高。(…)在本文中，我们讨论了固态硬盘中预取的挑战，解释了先前方法无法实现高精度的原因，并提出了一种基于神经网络的预取方法，该方法明显优于现有技术。为了实现高性能，我们解决了在非常大的稀疏地址空间中预取的挑战，以及通过提前预测及时预取的挑战。(…)

* * *

### 2.FlowFrontNet:用 CNN 改进碳复合材料制造

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_349.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932293)

**论文摘要:**碳纤维增强聚合物(CFRP)是一种轻质而坚固的复合材料，旨在减轻航空航天或汽车部件的重量，从而减少温室气体排放。树脂传递模塑(RTM)是 CFRP 的一种制造工艺，可以放大到工业规模生产。(…)我们提出了一种深度学习方法 FlowFrontNet，通过学习从传感器到流动前沿“图像”的映射(使用向上扩展层)来增强原位过程视角，以捕捉流动前沿的空间不规则性来预测干点(使用卷积层)。(…)

* * *

### 3.基于 FPGAs 的深度学习现场伽马-强子分离

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_1091.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932415)

**论文摘要:**现代高能天体粒子实验每天在连续的大容量流中产生大量数据。(……)从背景噪音中分离出伽马射线是不可避免地要被记录下来的，这被称为伽马-强子分离问题。当前的解决方案严重依赖手工制作的功能。(…)事件发生后，整个机器学习管道在商用计算机硬件上执行。在本文中，我们提出了一种替代方法，将卷积神经网络(CNN)和二进制神经网络(BNNs)直接应用于望远镜相机的原始特征流。(…)

第一作者:

**************塞巴斯蒂安**************

[网站](https://web.archive.org/web/20221206013611/https://www-ai.cs.tu-dortmund.de/PERSONAL/buschjaeger.html)

* * *

### 4.从电网络传感器提取可解释的尺寸一致特征

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_795.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932372)

**论文摘要:**电力网络是高度监控的系统，需要操作员在了解底层网络状态之前执行复杂的信息合成。我们的研究旨在通过从传感器数据自动创建特征来帮助这一合成步骤。我们提出了一种使用语法引导进化的监督特征提取方法，它输出可解释的和维度一致的特征。(…)

## 运输

### 1.用于推荐具有提前预约的出租车路线的多标准系统

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_718.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932365)

**论文摘要:**随着出租车预约服务需求的增加，如何通过高级服务增加出租车司机收入的策略备受关注。然而，由于利润的不平衡，需求通常得不到满足。本文提出了一个考虑实时时空预测和交通网络信息的多准则路径推荐框架，旨在优化出租车司机提前预约时的利润。(…)

第一作者:

**范德尔林**

* * *

### 2.基于模型字典聚类的自动驾驶验证

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_949.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932390)

**论文摘要:**自动驾驶系统的验证仍然是汽车制造商为了提供安全的无人驾驶汽车而必须解决的最大挑战之一。(…)在本文中，我们提出了一种应用于自动驾驶数值模拟产生的时间序列的新方法。这是一种基于字典的方法，由三个步骤组成:每个时间序列的自动分段、状态字典的构建和产生的分类序列的聚类。我们提出了时间序列的具体结构和建议的方法优势，处理这样的数据，与国家的最先进的参考方法。

* * *

### 3.使用深度学习模型实现租赁车辆回报评估的自动化

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_617.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932342)

**论文摘要:**损害评估包括对损害进行分类和估计其修理费用，是车辆租赁和保险行业的一个基本流程。特别是车辆租赁，租赁结束评估对客户必须支付的实际成本有很大影响。(…)在本文中，我们提出了一种基于机器学习(ML)的租赁车辆回报评估自动化解决方案。此外，我们强调了在处理没有标准过程收集的数据集时，标准的 ML 模型及其训练协议是如何失败的。(…)

* * *

### 4.具有协调强化学习的实时车道配置

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_473.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932314)

**论文摘要:**根据交通模式改变道路的车道配置，是提高交通流量的一种行之有效的解决方案。传统的车道方向配置解决方案假设预先知道交通模式，因此不适合现实世界的应用，因为它们不能适应变化的交通条件。我们提出了一个动态车道配置解决方案来改善交通流使用两层，多代理架构，命名为协调学习为基础的车道分配(CLLA)。(…)

* * *

### 5.学习用于按需交付应用的感兴趣区域的上下文和拓扑表示

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_731.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932368)

**论文摘要:**城市区域的良好表现对于按需交付服务(如 ETA 预测)非常重要。然而，现有的表示方法要么从稀疏的签到历史中学习，要么从拓扑几何中学习，因此要么缺乏覆盖并且违反了地理规律，要么忽略了来自数据的上下文信息。在本文中，我们提出了一种新的表示学习框架，用于从上下文数据(轨迹)和拓扑数据(图形)中获得感兴趣区域的统一表示。(…)

## 异常检测

### 1.使用上下文特征解释端到端 ECG 自动诊断

[P](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_1072.pdf) [a](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_1072.pdf) [按](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_1072.pdf) | [演示](https://web.archive.org/web/20221206013611/https://slideslive.com/38932410)

**论文摘要:**我们提出了一种新的方法来生成对端到端分类模型的解释。解释包括对用户有意义的特征，即上下文特征。我们在自动心电图(ECG)诊断的场景中实例化我们的方法，并分析在可解释性和鲁棒性方面产生的解释。所提出的方法使用噪声插入策略来量化 ECG 信号的间隔和分段对自动分类结果的影响。(…)

* * *

### 2.自我监督的日志解析

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_313.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932287)

**论文摘要:**日志在软件系统的开发和维护过程中被广泛使用。(…)然而，大规模软件系统会产生大量的半结构化日志记录，这给自动化分析带来了重大挑战。将带有自由格式文本日志消息的半结构化记录解析成结构化模板是实现进一步分析的第一步，也是关键的一步。现有的方法依赖于特定于日志的试探法或手动规则提取。(…)我们提出了一种称为 NuLog 的新解析技术，该技术利用自我监督学习模型，并将解析任务公式化为掩蔽语言建模(MLM)。(…)

* * *

### 3.环境智能系统中基于上下文的异常行为检测方法

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_1139.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932420)

**论文摘要:**异常的人类行为可能是健康问题或危险事件发生的征兆。检测这种行为在环境智能(AmI)系统中是必不可少的，以增强人们的安全性。(…)在本文中，提出了一种利用人类行为的上下文信息来检测这种行为的新方法。(…)

* * *

### 4.基于预测误差模式的多元时间序列异常检测

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_1139.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932309)

**论文摘要:**以信息物理系统(CPSs)的发展为部分特征的工业 4.0 的到来，自然需要可靠的安全方案。(…)在这项工作中，我们的目标是对异常检测技术在 CPSs 中的应用文献做出贡献。我们提出了新的功能数据分析(FDA)和基于自动编码器的方法，用于安全水处理(SWaT)数据集中的异常检测，该数据集真实地代表了按比例缩小的工业水处理厂。(…)

* * *

### 5.基于时间因果网络模型的复杂活动识别

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_173.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932256)

**论文摘要:**复杂活动识别具有挑战性，这是由于执行复杂活动的固有多样性和因果性，其每个实例具有其自己的原始事件配置及其时间因果依赖性。(…)我们的方法引入了从优化的网络骨架生成的时间因果网络，以明确地将特定复杂活动的这些独特的时间因果配置表征为可变数量的节点和链接。(…)

第一作者:

**廖俊**

## 广告

### 1.包装之外的思考:推荐电子商务运输的包装类型

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_8.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932233)

**论文摘要:**多种产品属性，如尺寸、重量、易碎性、液体含量等。确定电子商务公司运送产品时使用的包装类型。(…)在这项工作中，我们提出了一种多阶段方法，在每种产品的运输和损坏成本之间进行权衡，并使用一种可扩展的、计算高效的线性时间算法来准确分配最佳包装类型。提出了一种简单的二分搜索法算法来寻找在运输成本和损坏成本之间取得平衡的超参数。(…)

* * *

### 2.为工作推荐 MOOCs 课程:一种自动弱监督方法

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_1173.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932423)

**论文摘要:**大规模开放在线课程(mooc)的激增需要一种有效的课程推荐方式来推荐招聘网站上发布的职位，尤其是那些参加 mooc 寻找新工作的人。(…)本文提出了一个通用的自动化弱监督框架 AutoWeakS 通过强化学习来解决这个问题。一方面，该框架使得能够在由多个非监督排序模型产生的伪标签上训练多个监督排序模型。另一方面，该框架能够自动搜索这些监督和非监督模型的最佳组合。(…)

* * *

### 3.反馈引导的属性图嵌入相关视频推荐

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_121.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932250)

**论文摘要:**图的表示学习作为传统特征工程的替代方法，已经在从电子商务到计算生物学的许多应用领域得到了开发。(…)在本文中，我们提出了一种名为 Equuleus 的视频嵌入方法，该方法从用户交互行为中学习视频嵌入。在 Equuleus 中，我们仔细地将用户行为特征结合到视频图的构造和节点序列的生成中。(…)

第一作者:

**薛**

* * *

### 4.用于好友增强推荐的社会影响力注意力神经网络

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_65.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932243)

**论文摘要:**随着在线社交网络的蓬勃发展，在许多社交应用中出现了一种新的推荐场景，称为好友增强推荐(FER)。在 FER，用户被推荐他们的朋友喜欢/分享的项目(称为朋友推荐圈)。(…)在本文中，我们首先阐述了 FER 问题，并提出了一种新颖的社会影响注意神经网络(SIAN)解决方案。为了融合丰富的异构信息，SIAN 中的 attentive 特征聚集器被设计为在节点和类型级别学习用户和项目表示。(…)

## Web 挖掘

### 1.校准在线广告中的用户响应预测

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_214.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932265)

**论文摘要:**准确预测点击率(CTR)和转化率(CVR)等用户响应概率对在线广告系统至关重要。(…)由于点击和转换等用户响应行为的稀疏性和延迟性，传统的校准方法在现实世界的在线广告系统中可能不太适用。在本文中，我们提出了一个全面的在线广告校准解决方案。更具体地说，我们提出了一个校准算法来利用预测概率的隐含属性，以减少数据稀疏问题的负面影响。(…)

* * *

### 2.6 ve clm:IPv6 目标生成的向量空间语言建模

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_485.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932316)

**论文摘要:**快速 IPv6 扫描在网络测量领域具有挑战性，因为它需要探索整个 IPv6 地址空间，但受到当前计算能力的限制。(…)在本文中，我们介绍了我们的方法 6VecLM 来探索实现这样的目标生成算法。该体系结构可以将地址映射到向量空间以解释语义关系，并使用转换器网络来构建 IPv6 语言模型以预测地址序列。(…)

第一作者:

**天宇崔**

* * *

### 3.有限样本下多分类器的精度估计

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_701.pdf) | [Pres](https://web.archive.org/web/20221206013611/https://slideslive.com/38932362) [e](https://web.archive.org/web/20221206013611/https://slideslive.com/38932362) [论文](https://web.archive.org/web/20221206013611/https://slideslive.com/38932362)

**论文摘要:**机器学习分类器通常需要定期跟踪其性能指标，如精度、召回率等。，用于模型改进和诊断。(…)我们提出了一种采样方法来估计多个二元分类器的精度，该方法利用了它们的预测集之间的重叠。我们从理论上保证我们的估计量是无偏的，并从经验上证明从我们的抽样技术估计的精度度量与从均匀随机样本获得的精度度量一样好(就方差和置信区间而言)。(…)

* * *

### 4.从浏览事件中嵌入神经用户

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_70.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932244)

**论文摘要:**基于在线用户行为数据的深度理解对于向他们提供个性化服务至关重要。然而，现有的学习用户表示的方法通常基于监督框架，例如人口预测和产品推荐。(…).受预训练词嵌入在许多自然语言处理任务中的成功启发，我们提出了一种简单而有效的神经用户嵌入方法，通过使用在线用户的未标记行为数据来学习他们的深层表征。一旦用户被编码成低维密集嵌入向量，这些隐藏的用户向量可以在各种用户参与的任务(例如人口统计预测)中用作附加的用户特征，以丰富用户表示。(…)

## 计算社会科学

### 1.用于需求预测的空间社区信息演化图

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_163.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932254)

**论文摘要:**数量迅速增加的共享自行车极大地方便了人们的日常通勤。然而，由于用户的免费入住和退房，不同车站的可用自行车数量可能会不平衡。(…)为了应对这些挑战，我们提出了一种新的空间社区信息演化图(SCEG)框架来预测站级需求，该框架考虑了两种不同粒度的交互。具体来说，我们使用 EvolveGCN 从进化站网络中的细粒度交互中学习时间进化表示。(…)

第一作者:

**Qianru Wang **

[网站](https://web.archive.org/web/20221206013611/https://scholar.google.com/citations?user=aOxb8dgAAAAJ&hl=en)

* * *

### 2.深入探究多语言仇恨言论分类

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_843.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932379)

**论文摘要:**仇恨言论是目前困扰社会的一个严重问题，并已造成缅甸罗辛亚社区种族灭绝等严重事件。社交媒体让人们可以更快地传播这种仇恨内容。对于缺乏仇恨言论检测系统的国家来说，这尤其令人担忧。在本文中，我们使用来自 16 个不同来源的 9 种语言的仇恨言论数据集，首次对多语言仇恨言论检测进行了广泛的评估。我们分析了不同深度学习模型在各种场景下的性能。(…)

* * *

### 3.基于分层联合分解的半监督多方面误报检测

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_997.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932400)

**论文摘要:**区分误传信息和真实信息是当今互联世界最具挑战性的问题之一。绝大多数检测错误信息的最新技术都是完全监督的，需要大量高质量的人工注释。(…)在这项工作中，我们感兴趣的是探索注释数量有限的情况。在这种情况下，我们研究了如何挖掘表征一篇新闻文章的不同数量的资源，以下称为“方面”，可以弥补标签的缺乏。(…)

* * *

### 4.模型桥接:仿真模型与神经网络的连接

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_238.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932274)

**论文摘要:**机器学习，尤其是深度神经网络的可解释性，对于现实世界应用中的决策至关重要。一种方法是用具有简单解释结构的代理模型代替不可解释的机器学习模型。另一种方法是通过使用由人类知识建模的具有可解释的模拟参数的模拟来理解目标系统。(…)我们的想法是使用模拟模型作为可解释的代理模型。然而，由于仿真模型的复杂性，模拟器校准的计算成本很高。因此，我们提出了一个“模型桥接”框架，通过一系列内核均值嵌入来桥接机器学习模型和仿真模型，以解决这些困难。(…)

## 电子商务和金融

### 1.面向电子商务的时尚服装代

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_824.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932375)

将赠送的衣服组合成一套服装的任务对大多数人来说都很熟悉，但迄今为止证明很难实现自动化。我们提出了一个基于图像和文本描述的服装多模态嵌入模型。在一种新颖的深度神经网络结构中，嵌入和共享风格空间被端到端地训练。该网络是在迄今为止最大最丰富的标签服装数据集上训练的，我们开源了这个数据集。(…)

* * *

### 2.用机器学习测量移民对本地人购物消费的采纳

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_256.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932279)

**论文摘要:**“告诉我你吃什么，我就告诉你你是什么”。Jean Anthelme Brillat-Savarin 是最早认识到身份与食物消费之间关系的人之一。与其他个体行为相比，食物选择受外部判断和社会压力的影响要小得多，而且可以长期观察。这使得它们成为从食物消费角度研究移民融合的有趣基础。事实上，在这项工作中，我们从购物零售数据中分析了移民的食品消费，以了解它是否以及如何向本地人靠拢。(…)

* * *

### 3.我的消费者为什么购物？学习零售商交易数据的有效距离度量

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_1256.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932430)

**论文摘要:**交易分析是旨在了解消费者行为的研究中的一个重要部分。(…)在本文中，我们提出了一种新的距离度量，这种度量在设计上独立于零售商，允许跨零售商和跨国分析。该指标带有一种寻找产品类别重要性的新方法，在无监督学习技术和重要性校准之间交替使用。(…)

* * *

### 4.用于植入式广告的 3D 广告创作系统

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_760.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932369)

**论文摘要:**在过去的十年里，视频分享平台的发展吸引了大量对情境广告的投资。常见的上下文广告平台利用用户提供的信息将 2D 视觉广告整合到视频中。(…)本文提出了一个视频广告植入&集成(Adverts)框架，它能够感知场景的三维几何形状和摄像机运动，以融合视频中的三维虚拟对象，并创建现实的幻觉。(…)

* * *

### 5.发现和预测巴西市场内幕交易的证据

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_207.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932262)

**论文摘要:**众所周知，内幕交易会对市场风险产生负面影响，在许多国家被视为犯罪。然而，执行率差别很大。在巴西，特别是很少的法律案件得到追究，而且据我们所知，以前的案件的数据集是不存在的。在这项工作中，我们考虑巴西市场，并处理两个问题。首先，我们提出了一种建立内幕交易证据数据集的方法。(……)其次，我们使用我们的数据集，试图在相关事件披露之前识别可疑的谈判。(…)

作者:

## 社会公益

### 1.使用堆叠非参数贝叶斯方法的能源消耗预测

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_852.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932381)

**论文摘要:**利用多个短时间序列数据，在非参数高斯过程(GP)的框架内研究家庭能源消费的预测过程。随着我们开始使用智能电表数据来描绘澳大利亚住宅用电的更清晰的画面，越来越明显的是，我们还必须构建一个详细的画面，并理解澳大利亚与天然气消费的复杂关系。(…)考虑到这些事实，我们构建了一个堆叠 GP 模型，其中应用于每个任务的每个 GP 的预测后验概率用于下一级 GP 的先验和似然。我们将我们的模型应用于真实世界的数据集，以预测澳大利亚几个州的家庭能源消耗。(…)

* * *

### 2.基于非参数生存分析的管道长期失效预测

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_660.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932350)

**论文摘要:**澳大利亚的水利基础设施已经有一百多年的历史，因此已经开始通过水管故障来显示其老化。我们的工作涉及横跨澳大利亚主要城市的大约 50 万条管道，这些管道向家庭和企业供水，服务于 500 多万客户。(…)我们应用了机器学习技术，为这些澳大利亚城市的管道故障问题找到了一种经济高效的解决方案，这些城市每年平均发生 1500 起主水管故障。为了实现这一目标，我们构建了一个详细的图像并理解了水管网的行为(…)。

* * *

### 3.约束深度学习的拉格朗日对偶

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_877.pdf)|[Prese](https://web.archive.org/web/20221206013611/https://slideslive.com/38932383)[n](https://web.archive.org/web/20221206013611/https://slideslive.com/38932383)[station](https://web.archive.org/web/20221206013611/https://slideslive.com/38932383)

**论文摘要:**本文探讨了拉格朗日对偶对于具有复杂约束的学习应用的潜力。这种约束出现在许多科学和工程应用中，在这些应用中，任务相当于学习优化问题，这些问题必须重复求解，并且包括硬的物理和操作约束。本文还考虑了学习任务必须对预测器本身施加约束的应用，因为它们是要学习的函数的自然属性，或者因为从社会角度来看施加这些约束是可取的。(…)

* * *

### 4.犯罪预测:通过利用邻居的时空依赖性进行犯罪预测

[论文](https://web.archive.org/web/20221206013611/https://bitbucket.org/ghentdatascience/ecmlpkdd20-papers/raw/master/ADS/sub_713.pdf) | [简报](https://web.archive.org/web/20221206013611/https://slideslive.com/38932363)

**论文摘要:**城市地区的犯罪预测可以改善资源分配(例如，警察巡逻)以实现更安全的社会。最近，研究人员一直在使用深度学习框架进行城市犯罪预测，与以前的工作相比，精确度更高。(…)在本文中，我们设计并实现了一个端到端的时空深度学习框架，称为犯罪预测器，它可以同时捕获区域内和跨区域的时间重现和空间依赖性。(…)

## 摘要

我个人建议您也访问活动网站，更深入地探索您最感兴趣的话题。

请注意，我们还发布了来自会议的**顶级研究论文的帖子。看看这里的。**

我很乐意扩展这个列表，因为这是我的主观选择。欢迎提出更多的论文。如果你觉得少了什么很酷的东西，简单地告诉我，我会延长这个帖子。