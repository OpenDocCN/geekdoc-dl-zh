# 使用 IPTOP 构建可扩展的数据策略:基础设施、人员、工具、组织和流程

> 原文：<https://web.archive.org/web/20221129040116/https://www.datacamp.com/blog/building-a-scalable-data-strategy-with-iptop-infrastructure-people-tools-organization-and-processes>

[![](img/dbe41f2268d05925d10fa22f370cadc7.png)](https://web.archive.org/web/20220630224323/http://www.datacamp.com/groups/business/data-science-for-managers-free-trial?utm_source=internal&utm_medium=blog&utm_campaign=2021_b2b_building_a_scalable_data_strategy)

[![](img/6bc56fb74aa9b5e2517d3211d9756be7.png)](https://web.archive.org/web/20220630224323/https://www.datacamp.com/webinars/scaling-data-science-at-your-organization)

如今，许多组织正在将数据科学实践作为其数字化转型计划的一部分。然而，如果没有数据战略和清晰的组织内扩展数据科学的蓝图，他们中的大多数人不会收获挖掘数据的回报[。](https://web.archive.org/web/20220630224323/https://www.datacamp.com/community/blog/digital-transformation)[麦肯锡](https://web.archive.org/web/20220630224323/https://www.mckinsey.com/business-functions/mckinsey-analytics/our-insights/ten-red-flags-signaling-your-analytics-program-will-fail)发现**1000 家进行数字化转型的公司**中只有 8 家能够将数据科学扩展到少数试点项目之外。

此外，虽然大多数组织了解数据驱动的价值，但许多组织将数据科学视为一个孤立的集中式支持功能，它根据不同团队的请求工作。这与数据科学的本质是不一致的，数据科学是实现业务目标的一种手段。正如 Anaconda 的首席执行官王蒙杰所说，数据科学是在商业世界中导航的“一种调查和探索模式”。

> 正如物理学家使用数学来推理自然世界一样，数据科学家利用数学和计算工具来推理商业世界。—王蒙杰，Anaconda 首席执行官

这种孤岛效应因一个错误的前提而加剧，即数据科学的最终目标应该始终是能够自动化或简化组织内主要垂直生产的人工智能模型。寻求直接投资回报的公司失望地发现，事实往往并非如此。

这种狭隘的数据科学观将其价值归因于相对难以实现的预测分析(*即利用数据预测未来的能力*)。它还忽略了大规模执行机器学习所需的基础设施工作。然而，数据科学的大部分价值来自于实施相对简单的描述性分析(*描述数据并将其置于正确手中的能力*)和规定性分析(*做出数据驱动的决策*)。

[![](img/2c26f7ef5bdd7f870d4160729b8ff472.png)](https://web.archive.org/web/20220630224323/https://www.datacamp.com/groups/business)

完成成功的数字化转型需要[培养数据科学和分析方面的组织能力](https://web.archive.org/web/20220630224323/https://www.datacamp.com/community/blog/digital-transformation)。这需要构建和执行一个智能的、包容的、可扩展的数据策略。

这就是我们的 IPTOP 框架的用武之地。IPTOP 是一个建立五大支柱(***I***n 基础架构、 ***P*** 人、 ***T*** 工具、 ***O*** 组织、 ***P*** 流程)到**可扩展地执行您的数据策略来完成一次成功的数字化转型的框架。[加入我们即将举办的网络研讨会系列](https://web.archive.org/web/20220630224323/https://www.datacamp.com/webinars/scaling-data-science-at-your-organization)了解更多信息。**

 **## 基础设施

任何数据策略的目标都是将原始数据转化为洞察力和决策。这要求组织安全高效地收集、记录和存储数据，以便所有人都能访问。但是数据通常以不同的形式、形状和大小收集。有助于这一过程的各种数据库、数据湖、数据仓库、脚本和仪表板构成了数据基础设施。构建健壮的数据基础架构需要了解最佳实践。

## 人

将数据科学视为实现更好决策这一最终目标的一种手段，可以让组织根据他们所需的技能来构建自己的团队。[基于角色的方法](https://web.archive.org/web/20220630224323/https://www.datacamp.com/community/blog/persona-driven-learning)需要用实现业务目标所需的技能来识别、评估和映射绩效目标，例如预测流失或使用仪表板可视化数据。这导致每个角色都有专门的学习途径。

一个很好的例子是 [Airbnb 的数据大学](https://web.archive.org/web/20220630224323/https://medium.com/airbnb-engineering/how-airbnb-democratizes-data-science-with-data-university-3eccc71e073a)，这是他们的专有培训计划，旨在让每位员工掌握做出数据驱动型决策所需的技能。通过让非数据科学家的员工成为能够做出明智决策的公民数据科学家，数据科学团队可以腾出时间从事更具战略性的项目。

## 工具

虽然基础架构使组织能够从数据中获得洞察力，但工具可以促进和激励整个组织采用通用的数据语言。使用工具进行数据访问、分析、可视化和仪表板可以让组织变得更加高效，从而缩短获得洞察力的时间。这些工具包括从 Python、R 和 SQL 等开源编程语言到 Power BI、Tableau 和 Excel 等基于点击的工具。

在这些工具的基础上构建简化数据访问的特定于组织的框架，可以大大降低扩展数据科学的门槛。在 DataCamp，我们有专有的 Python 和 R 包，它们抽象出连接到数据湖、查询数据以及使用简单的命令聚合数据。任何人都可以回答类似于*“过去 Y 周课程 X 的评分是多少？”*用一两行代码。同样， [Airbnb](https://web.archive.org/web/20220630224323/https://medium.com/airbnb-engineering/using-r-packages-and-education-to-scale-data-science-at-airbnb-906faa58e12d) 有一个 R 包，可以根据他们想要的美感来方便地查询和可视化数据，这消除了猜测，以确保整个组织内一致的数据分析和可视化。

[![](img/167dcca541fcd790263707becf2a2a6c.png)](https://web.archive.org/web/20220630224323/https://www.datacamp.com/webinars/scaling-data-science-at-your-organization)

## 组织

数据战略的一个重要方面是如何组织数据专业人员。鉴于在大多数公司中，报告结构和议程推动工作，组织结构必须为你的公司建立可持续的成功。在数据科学家属于一个数据科学团队的集中式模型和数据科学家嵌入不同部门的分散式模型之间存在权衡。

在集中式模型中，中央数据科学团队对来自其他部门的信息请求进行优先排序和处理。在这个模型中，问题进来，答案期待出来。这使得数据科学团队成为一个卓越的中心，在这里，数据科学家在一个战略方向下协作和共享知识。然而，这种做法孤立了数据科学团队及其工具，使得数据科学家与其他部门的协调和沟通变得复杂。

在分散模型中，数据科学家被嵌入到组织内的不同部门。随着数据科学家获得成功所需的领域知识，这为数据科学提供了影响其部门战略方向的席位。然而，由于它们是分散的，由业务团队经理管理，缺点是它以牺牲数据科学家的成长、学习和发展以及协作能力为代价。

这两个模型应该被看作是一个光谱的两个相反的末端。有许多混合模型结合了集中式和分散式模型的元素，以不同的方式对部门进行分组和捆绑，以最大限度地发挥数据科学的价值。

[![](img/d310b4d066817126088dcc122ebfc4eb.png)](https://web.archive.org/web/20220630224323/https://www.datacamp.com/webinars/scaling-data-science-at-your-organization)

## 处理

最后，构建可扩展的数据策略需要在约定、最佳实践和流程上保持一致。促进一致性对于促进协作和避免孤立的组织至关重要。这允许所有团队无缝地一起工作，并在一种公共的数据语言下进行交流。

开始在流程上建立一致性的一个简单方法是创建一个预定义的项目结构和模板，其中分析项目的不同任务和子任务被提前用它们的需求映射出来。微软已经采用[团队数据科学流程](https://web.archive.org/web/20220630224323/https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview)，该流程允许任何利益相关者清楚地了解项目需求，利用模板进行数据分析和计算能力访问，并确定谁拥有流程的不同阶段。

根据您的组织和行业，预定义的项目模板可能受特定法规要求的约束，并且可能需要复杂的流程。您可以利用[开源工具](https://web.archive.org/web/20220630224323/https://drivendata.github.io/cookiecutter-data-science/)来设置项目结构模板，以增加团队和数据专业人员之间的一致性。

这只是我们在组织内扩展数据科学的 IPTOP 框架的皮毛。如果您想了解更多信息，[请参加我们将于 8 月 20 日、8 月 27 日和 9 月 3 日举办的由三部分组成的网络研讨会系列](https://web.archive.org/web/20220630224323/https://www.datacamp.com/webinars/scaling-data-science-at-your-organization)。


[![](img/6bc56fb74aa9b5e2517d3211d9756be7.png)](https://web.archive.org/web/20220630224323/https://www.datacamp.com/webinars/scaling-data-science-at-your-organization)

[![](img/fdd7e4ab4ea05d095c729356c11ac7a8.png)](https://web.archive.org/web/20220630224323/http://www.datacamp.com/groups/business/data-science-for-managers-free-trial?utm_source=internal&utm_medium=blog&utm_campaign=2021_b2b_building_a_scalable_data_strategy)**