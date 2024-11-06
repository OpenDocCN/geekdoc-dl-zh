# 什么是关系数据库？

> 原文：<https://www.algolia.com/blog/product/what-is-a-relational-database/>

如果说数据是世界的 [新石油](https://www.economist.com/leaders/2017/05/06/the-worlds-most-valuable-resource-is-no-longer-oil-but-data) ，那么数据库就是数据桶，尽管是相当复杂的桶。最先进的容器可以帮助您的公司专业地处理现实世界中的数据管理工作负载-存储数据、对数据进行分类、共享数据，并提供问题的答案，所有这些都不会弄脏您的手和鞋子。

所有关于你是一个组织以及你的公司做什么的信息都存储在数据库中。他们对一个组织的成功运作至关重要；它们是智能的坚实基石，在网站和应用程序中满足 [用户体验](https://www.algolia.com/blog/ai/a-gentle-introduction-to-intelligent-journeys-with-user-intent-graphs/) 。在这个机器学习辅助的大数据时代，随着企业日常创建、分析和使用的数据量(无论是在内部还是外部)的增加，考虑各种类型的数据库及其功能是值得的。

关系数据库是使您的企业所拥有的数据类型对需要信息的人更有用的关键。但是这些“桶”是如何工作的呢？

## [](#so-what-is-a-relational-database%c2%a0)**那么什么是关系数据库呢？**

关系数据库模型的结构是什么？

关系数据库是一种专门组织的数据库，用于识别不同信息项之间的联系，即“关系数据”的类型它是由 IBM 的程序员 E.F. Codd 在 1970 年发明的。

与层次数据库相比，数据库设计更加灵活，它由不同的表格组成，这些表格可以混合和匹配，以各种方式放在一起，使用户能够轻松地准确定位他们正在寻找的数据点，以便能够立即获得他们需要的所有细节，进行关联，并得出结论。

关系数据库中的每个表包括:

*   列的类别
*   行(也称为元组或记录)

行是存储唯一数据的地方，每个表都有一个“主键”来定义它的信息。例如，为了存储日常采购订单的信息，主键列可能被标记为 Customer，其中包含诸如姓名、联系方式和日期等类别的列。旁边可能是一个“订单号”主键，包含客户、产品、价格和其他详细信息。

在设置了他们需要轻松解释纵横交错的信息的组织视图后，员工可以搜索数据库以找到适合他们需求的特定数据子集。例如，他们可能需要找到:

*   欠账
*   订单完成
*   在特定时间段内下达的订单

## [](#the-structure-of-a-relational-database%c2%a0)**关系数据库的结构**

关系数据库使用 [结构化查询语言(SQL)](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/in-depth/prepare-data-in-depth/) 进行结构化，这使得用户能够对其进行查询，以汇集他们所需的信息。

最初设计时，关系数据库模型与 IBM 以前的层次结构不同，后者要求用户通过具有多个“分支”的“树状”设计来检索所需信息。

## [](#how-a-hierarchical-database-works)**分层数据库如何工作**

在分层数据库中，相同的数据需要重复存储在多个位置，这使得接近实时地搜索特定信息以及查看某些数据如何相互关联变得更加困难。要找到需要的信息，您必须从上到下搜索整个模型。在数据量不断增加的现代工作场所，建立联系并从数据中获得洞察力几乎是不可能的。

分级数据库的一个很好的例子是大多数人在某个时候都见过的:组织梯形图。信息是基于基本元素(如表、记录和字段)存储的。

*   位于层级顶端的是首席执行官或创始人
*   下一级是部门领导或团队领导
*   这一层之下是领导团队的成员

例如，人力资源部职员 John Smith 坐在人力资源部助理经理 Megumi Yee 的下面，Megumi Yee 坐在人力资源部经理 Adita Gupta 的下面，以此类推。

数据库的元素是设定的，信息是静态的。你得到了组织的一个特定视图，这个视图有助于一个单一的目的——容易地知道员工的职位和他们的工作关系。

但是这个设计有一些相当大的问题。例如，如果您想查看 John 加入公司的时间与 Megumi 被提升为助理经理的时间之间的关系，您会发现很难找出该细节。此外，如果您将人员详细信息存储在多个位置，并且需要进行一些更改，那么您需要在每个单独的数据存储位置进行相同的更改，而不是简单地更新一个主实例。

相比之下，关系数据库更加灵活。

## [](#how-relational-databases-work)**关系数据库如何工作**

这里有三个用例——关系数据库示例，说明如何处理数据复杂性。

关系数据库允许一个关系表中的记录链接到其他表中的信息。例如，每个雇员都可以用一个主键(也许是他们唯一的用户 ID)来标记，所有与他们的雇佣相关的信息都可以在这个唯一的标识符下连接起来。这意味着冗余的数据集信息被删除(不会在多个地方重复)，您可以更容易地找到复杂问题的答案。

比方说，你拥有一家电子商务公司，正准备出售。如果您使用的是关系数据库，只需更新商品的销售价格就可以了。使用非关系数据库，您可以为销售价格的商品版本创建单独的记录。

创建越来越多的数据还会使数据输入过程变得非常繁琐——还可能导致你在输入信息时出现一些错误——所以数据库管理爱好者会告诉你避免这种选择。

如果您使用的是非关系型数据库，那么每当客户购买某样东西时，就会为交易处理创建一个单独的客户名称实例以及他们的产品购买信息。这为错误留下了空间。例如，对于每笔交易，客户姓名的记录可能略有不同(例如，拼写错误或在另一个实例中未使用中间名首字母)，这取决于是谁输入的。如果您还添加了打字错误的可能性，您就会看到信息混乱的潜在可能性(更不用说需要配备更多的数据库管理员)。

## [](#what-are-the-benefits-of-the-relational-database-model)**关系数据库模型有什么好处？**

关系数据模型背后的主要思想是能够让用户通过操纵数据库表来阐明和理解各种数据类型之间的关系，从而得出有意义的见解。

使用关系数据库存储和排序数据有几个好处:

### [](#it%e2%80%99s-easy-to-use%c2%a0)**很好用**

使用关系数据库，您可以从多个数据存储位置快速收集信息，然后对其进行过滤，以满足您的特定信息需求。

### [](#excellent-accuracy-and-data-integrity)**卓越的准确性和数据完整性**

关系数据库结构紧密，数据使用约束、检查和其他数据库软件“规则”精心组织这种先进的结构有助于高水平的准确性和完整性。

### [](#less-data-redundancy)**数据冗余少**

由于强结构，在关系数据库中，不是复制信息并存储在多个地方，而是只保留唯一信息的单个记录。这可以防止事情变得过于复杂和多余，并确保您的员工和客户都使用相同的最新信息。

### [](#strong-security)**强安全**

关系数据库系统可以被配置为只允许授权用户进入，它们通常提供基于角色的安全性等特性来保护敏感数据。您还可以在用户级别设置字段访问控制和控制权限。

## [](#upgrade-your-relational-database-search%c2%a0)**升级你的关系数据库搜索**

需要更好地搜索您的数据库吗？Algolia 是 [数据库不可知论者](https://support.algolia.com/hc/en-us/articles/4406981924241-How-to-index-NoSQL-databases-like-MongoDB-in-Algolia-) (包括 NoSQL 数据库)。此外，在索引关系数据时，我们的搜索 API 是 [无模式](https://www.algolia.com/doc/guides/sending-and-managing-data/prepare-your-data/how-to/handling-data-relationships/) ，因此您可以非常灵活地组织数据，并确保您的员工和客户可以在不断增长的内容海洋中即时找到他们需要的内容。

好奇？通过与我们的团队联系[](https://www.algolia.com/contactus/)或试用我们备受推崇的 SaaS 搜索解决方案[](https://www.algolia.com/users/sign_up)，了解智能数据组织和管理对您的业务的潜在价值。