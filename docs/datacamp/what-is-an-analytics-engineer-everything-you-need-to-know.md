# 什么是分析工程师？你需要知道的一切

> 原文：<https://web.archive.org/web/20230101102924/https://www.datacamp.com/blog/what-is-an-analytics-engineer-everything-you-need-to-know>

数据科学在不断发展，任何现代数据团队中的职称和角色也是如此。在数据科学的早期，许多职位都包含“大数据”一词。

在过去的两年里，随着机器学习变得越来越可操作化，MLOps 开始出现在不同的职位名称中。如今，随着组织在数据素养和分析成熟度方面的发展，我们看到了分析工程角色的崛起。

简而言之，分析工程师充当工程和分析师职能之间的桥梁。他们的角色是应用工程最佳实践来提供干净的、经过转换的数据集，以便进行分析。

本文旨在揭开分析工程师是什么和他们做什么的神秘面纱，以及将该角色与其他常见的数据角色进行比较，并提供进入分析工程的资源。

## 什么是分析工程师？

[Dataform](https://web.archive.org/web/20221212135912/https://dataform.co/blog/what-do-analytics-engineers-do) 使用一个熟悉的例子——橱柜，为分析工程师如何融入更广泛的数据团队提供了一个奇妙的类比。当我们考虑为什么分析工程师在增加时，请记住这个类比。

在分析工程兴起之前，数据分析师会使用可视化工具，如 [Tableau](https://web.archive.org/web/20221212135912/https://www.datacamp.com/learn/tableau) 或 [Power BI](https://web.archive.org/web/20221212135912/https://www.datacamp.com/learn/power-bi) 来为利益相关者提供见解。这些工具是呈现数据的绝佳方式，但不是转换和存储数据的最佳方式。

想象一个场景，一个数据工程师部署了一个加载营销数据的数据管道，但是数据质量有所欠缺。只有营销团队中的数据分析师拥有转换和提高数据质量的领域知识。但是，他们不会使用与数据工程师相同的技术堆栈。

同时，数据工程师没有领域知识来快速理解需要什么样的转换，并且很可能被来自整个企业的更高优先级的任务淹没。因此，分析师可能必须通过在关系数据库中构建自己的临时表来转换 Tableau 或 Power BI 中的数据。这导致效率低下，洞察速度变慢。

进入分析工程师。分析工程师位于数据分析师和数据工程师之间。在上面的场景中，他们将拥有优化数据转换的技术技能，以及领域知识。

通过与数据分析师密切合作，分析工程师可以使用适当的工具和技术转换数据，从而为数据分析师提供干净的数据。

## 分析工程师与其他数据角色有何不同？

不同角色之间的界限很模糊，那么分析工程师与其他数据角色有什么不同，他们是否能一起工作？

### 分析工程师 vs 数据分析师

数据分析师负责分析数据并报告他们的分析见解。他们对数据分析工作流有着深刻的理解，并通过结合使用编码和非编码工具来报告他们的见解。数据分析师通常非常擅长于 SQL 和商业智能工具，如 T2 的 Power BI 或 Tableau，但很少使用 T4 的 Python。

分析工程师与数据分析师合作，优化可供使用的数据模型。他们还负责维护围绕数据的文档，这使数据分析师能够更快地找到见解。

### 分析工程师与数据工程师

[数据工程师](https://web.archive.org/web/20221212135912/https://www.datacamp.com/tracks/data-engineer-with-python)负责将正确的数据送到正确的人手中。他们创建并维护基础设施和数据管道，这些管道将来自不同来源的万亿字节的原始数据放入一个集中的位置，为组织提供干净、相关的数据。

不可否认，这个定义与分析工程角色有很大程度的重叠。然而，通过再次引用橱柜示例，数据工程师负责确保分析工程师拥有正确的数据源来组织和建模供数据团队的数据分析师使用。

此外，数据工程师负责设置来自专有来源的定制 API 和 ETL 管道——而数据工程师将更多时间用于改进来自 Google Analytics 等供应商 API 的管道。

### 分析工程师 vs 数据科学家

数据科学家调查、提取并报告对组织数据的有意义的见解。他们将这些见解传达给非技术利益相关者，并对机器学习工作流以及如何将它们与业务应用联系起来有很好的理解。他们几乎专门使用 SQL、Python 和 R 等编码工具，进行分析，并经常使用大数据工具。

数据科学家和分析工程师之间的关系类似于数据分析师和分析工程师之间的关系。分析工程师使数据科学家能够更多地关注洞察力，而不是清理和优化数据集。

### 分析工程师 vs 机器学习工程师

机器学习工程师设计和部署机器学习系统，这些系统根据组织的数据进行预测。他们解决诸如预测客户流失和终身价值等问题，并负责部署模型供组织使用。他们专门使用基于编码的工具，并且比他们的同行更加关注技术。

分析工程师更加关注支持数据分析师和数据科学家，他们负责向业务利益相关方提供见解。

### 分析工程师职位比较

| **数据分析师** | **数据工程师** | **数据科学家** | **机器学习工程师** | **分析工程师** |
| 分析数据并向利益相关者报告见解
构建供更广泛的组织使用的仪表板
利用主题专业知识和领域知识进行推荐 | 使用供应商和专有 API 构建和维护 ETL 数据管道
优化并集中数据湖或数据仓库中的数据
将数据传送到机器学习管道
在云端处理数据 | 分析数据并向利益相关者报告见解
设计实验，如 A/B 测试
部署由更广泛的组织使用的仪表板
开发监督和非监督学习工作流程
分析非标准数据类型，如时间序列、文本、地理空间或图像数据 | 训练和部署机器学习模型
监控和改进机器学习模型在生产中的性能
将软件工程最佳实践应用于机器学习工作流程(CI/CD) | 优化数据工程师构建的消费管道
将工程最佳实践应用于数据分析师和科学家使用的数据模型
开发、标准化和改进数据文档

 |
| r 或 Python
SQL，Power BI，Tableau | r 或 Python
结构化查询语言
Git、Shell 和命令行工具
气流或火花等大数据工具
基于云的工具，如 AWS、Azure、GCP 或雪花 | r 或 Python
结构化查询语言
Git、Shell 和命令行工具
气流或火花等大数据工具 | r 或 Python
结构化查询语言
Git、Shell 和命令行工具
气流或火花等大数据工具 | r 或 Python
结构化查询语言
Git、Shell 和命令行工具
基于云的工具，如 AWS、Azure、GCP、雪花或 dbt

 |

## 分析工程师工资

分析工程师的角色是新生的，这意味着市场上很少有人具备成功完成这一角色所需的工程和分析技能。从薪酬的角度来看，这使得分析工程师的角色极具吸引力。以下是在美国，分析工程职位的薪资范围。

*   **Glassdoor** :根据 [Glassdoor](https://web.archive.org/web/20221212135912/https://www.glassdoor.com/Salaries/us-analytics-engineer-salary-SRCH_IL.0,2_IN1_KO3,21.htm) 的数据，分析工程师的平均年薪为 91080 美元。此外，根据工作地点和公司的不同，年薪最高可达 208，000 美元。
*   **可比**:根据[可比](https://web.archive.org/web/20221212135912/https://www.comparably.com/salaries/salaries-for-analytics-engineer)的数据，分析工程师的平均年薪为 100，305 美元。此外，根据工作地点和公司的不同，年薪最高可达 185，000 美元。

虽然上述数字代表了整个范围内分析工程职位的大致范围，但也值得注意的是，像网飞这样高度成熟的数据公司支付的数据职位薪酬处于个人薪酬市场的高端。比如按[级。仅供参考](https://web.archive.org/web/20221212135912/https://www.levels.fyi/company/Netflix/salaries/Analytics-Engineer)，网飞分析公司的工程师年薪高达 37.5 万美元。

## 如何成为一名分析工程师？

随着数据角色变得专业化，提升技能的途径也变得越来越狭窄。与上面讨论的许多数据角色不同，分析工程师需要广泛的技能，这要求他们学习如下工具和概念

### 结构化查询语言

可以说，SQL 是任何数据角色的所有分析中使用最广泛的工具之一。幸运的是，它也是最容易学习和掌握的语言之一。查看这篇文章，了解如何[学习 SQL](https://web.archive.org/web/20221212135912/https://www.datacamp.com/learn/sql) ，以及这里的其他学习资源。

*   **职业轨迹**:[SQL 中的数据分析师](https://web.archive.org/web/20221212135912/https://www.datacamp.com/tracks/data-analyst-in-sql)
*   **技能轨迹:** [SQL 基础](https://web.archive.org/web/20221212135912/https://www.datacamp.com/tracks/sql-fundamentals)

### 计算机编程语言

Python 实际上是目前最流行的编程语言。无论是进入分析工程角色，还是上面提到的任何数据角色，Python 肯定都是有用的。[在这里找到学习 Python 所需的所有资源](https://web.archive.org/web/20221212135912/https://www.datacamp.com/learn/python)并开始学习以下课程。

*   **职业轨迹:** [Python 程序员](https://web.archive.org/web/20221212135912/https://www.datacamp.com/tracks/python-programmer)
*   **职业轨迹:** [用 Python 做数据分析师](https://web.archive.org/web/20221212135912/https://www.datacamp.com/tracks/data-analyst-with-python)

### ETL 工具

ETL 代表“提取、转换和加载”。这些工具允许工程师建立数据管道，从不同来源提取数据，将其转换为可消费的数据，并将其加载到数据库中。最流行的开源 ETL 工具之一是 Airflow。检查在[过程中](https://web.archive.org/web/20221212135912/https://www.datacamp.com/courses/introduction-to-airflow-in-python)的气流。

### 云计算工具

虽然术语“云计算工具”绝对是一个总括术语，但分析工程师和其他数据角色都定期利用云计算服务，如 AWS、Azure、Google Cloud 或雪花。这些工具允许数据团队在云中存储、处理和部署数据和数据解决方案。最流行的云计算工具是 AWS。你可以通过查看下面列出的 DataCamp 的 AWS 课程来了解更多关于 AWS 的知识。

*   **课程:** [AWS 云概念](https://web.archive.org/web/20221212135912/https://www.datacamp.com/courses/aws-cloud-concepts)
*   **课程:**[Python 中的 AWS Boto 介绍](https://web.archive.org/web/20221212135912/https://www.datacamp.com/courses/introduction-to-aws-boto-in-python)

### 版本控制

版本控制可以说是软件工程最佳实践的支柱。简而言之，它允许实践者跟踪他们什么时候做了什么，撤销他们不想要的任何改变，并与他人大规模合作。

像 Git 这样的命令行工具允许您应用版本控制最佳实践。通过查看这个[备忘单](https://web.archive.org/web/20221212135912/https://www.datacamp.com/cheat-sheet/git-cheat-sheet)了解更多关于 Git 的信息！

### 通讯技能

虽然每个数据角色都需要某种程度的沟通技能，但分析工程需要与数据分析师角色相同水平的沟通技能，以及数据工程师的技术才能。成为更好的沟通者是一种技能，而不是天赋。[查看数据通信概念](https://web.archive.org/web/20221212135912/https://www.datacamp.com/courses/data-communication-concepts)以提高您的技术通信技能。

## 成为一名分析工程师

分析工程，就像 [MLOps](https://web.archive.org/web/20221212135912/https://www.datacamp.com/podcast/operationalizing-machine-learning-with-mlops) 一样，是非常新生的。要保持领先，请查看下面的参考资料。