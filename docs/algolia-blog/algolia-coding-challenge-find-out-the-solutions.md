# Algolia 编码挑战:找出解决方案🔎

> 原文：<https://www.algolia.com/blog/engineering/algolia-coding-challenge-find-out-the-solutions/>

感谢所有在节日期间参与❄️ 2021 编码挑战❄️来帮助圣诞老人的人！

祝贺所有提交答案的人，尤其是我们的获奖者——**鲁本·金特罗** ！🥳

在这篇文章的剩余部分，我将使用 Ruben 的存储库作为例子来介绍每个挑战的解决方案。

我们开始吧！

# [](#challenge-1-%f0%9f%8e%a4)**挑战 1🎤**

圣诞老人是《星际迷航》的超级粉丝。他记得他最喜欢的《星际迷航》演员有一次做了一个 Ted 演讲，评分在 800 到 900 之间，被称为讲故事。

*那个圣诞老人要去拜访的说话人叫什么名字，(他也是著名的《星际迷航》演员)？*

[**Ted 演讲**](https://github.com/algolia/datasets/blob/master/tedtalks/talks.json) 数据集有几个有趣的属性可供分面。这里的诀窍是选择哪些要刻面( *标签* 和*inspiring _ rating*)。

对于这个 [解决方案](https://github.com/rbnquintero/algolia-coding-challenge/tree/main/challenge-1) ，Ruben 使用了 vanilla JS 的 InstantSearch 库。一个可搜索的*refinement list*允许 Santa 找到“讲故事的” *标签* ，而一个*range input*过滤器允许他为*insipuring _ rating 设置适当的范围。*

我们将搜索范围缩小到两个记录，其中一个是竹井乔治，他曾是《星际迷航》系列电影的演员。).

### [](#bonus-question)**加分题**

*关于合作和社会，最受关注的 TED 演讲是什么？*

**《如何识别骗子》**

# [](#challenge-2-%f0%9f%93%bd)**挑战#2📽**

当圣诞老人准备他的旅行时，他想起了他的“好”名单上的一个孩子，小布莱恩，要求一部由他计划拜访的男演员和女演员艾米·波勒主演的电影。圣诞老人需要另一个搜索界面来查找这部电影的片名(以及何时上映)。

*布莱恩圣诞节想要的电影片名和上映年份是什么？(通过* [*表单*](https://docs.google.com/forms/d/e/1FAIpQLSfU5XNTarCngITo5p4d-7VMnVc3a4IVRkK46XU5JADGfUjLow/viewform?usp=sf_link)*)*

[**电影**](https://github.com/algolia/datasets/blob/master/movies/) 数据集对于 Algolia 免费层来说太大了，所以我们需要使用托管版本。重要的是要记住，很多关于分面和相关性的配置发生在索引本身，而不是在查询中。因为我们不管理这个索引，所以我们需要使用配置的方面和可搜索属性。

对于这个 [的解决方案](https://github.com/rbnquintero/algolia-coding-challenge/tree/main/challenge-2) ，鲁本再次使用了一个可搜索的为 *演员* 。这就很容易找到 2013 年上映的电影 **【自由的小鸟】** ***。*** 鲁本为流派增加了一个*refinement list*和一个 *菜单* 的发布年份来解决奖金问题。

### [](#bonus-question)**加分题**

*2008 年上映的类型犯罪最高评分电影的片名和评分是多少？*

**7.8854034451496 为《黑暗骑士》**

# [](#challenge-3-%f0%9f%8d%b7)**挑战#3🍷**

*与《挑战#2》电影同年上映的 Fronsac 域波尔多葡萄酒叫什么名字？*

我们从 [**葡萄酒**](https://github.com/algolia/datasets/tree/master/wine/) 数据集开始，该数据集存在一些重复记录的问题。每种酒在数据集中出现 16 次！正如布莱恩和我在做这个 [挑战 live](https://discord.com/channels/906250704581177365/912436381152866305/918173239228903474) 时发现的那样，我们需要在将数据导入索引时对其进行重复数据删除。

一旦数据正常化，我们只需添加几个用于 *年份* 和 *领域* 就能发现 2013 年唯一一款来自弗龙萨克产区的葡萄酒是 **酒庄 Vrai Canon Bouche。**

### [](#bonus-question)**加分题**

*2010 年 30 美元以下最高评价的葡萄酒是什么？*

**费朗酒庄**

# [](#challenge-4-%f0%9f%9b%a9)**挑战#4 🛩**

手里拿着酒，圣诞老人准备开始他的旅程。但是在哪里？他知道他最喜欢的《星际迷航》演员最近进行了以下飞行:

距离上述三个航班最近的美国主要城市是哪个？

对于这个挑战，我们使用了 [**机场**](https://github.com/algolia/datasets/tree/master/airports) 数据集，但是我们需要结合地图数据来回答这个问题。Ruben 选择使用即时搜索 [地理搜索](https://www.algolia.com/doc/api-reference/widgets/geo-search/js/) 小部件。这个小部件使用 Google Maps APIs 来呈现地图数据。但是我们也可以为 [其他映射 API](https://www.algolia.com/doc/api-reference/widgets/geo-search/js/#connector)使用自定义连接器。

根据上面的机场代码过滤，我们发现虽然纽约市附近有两个著名的机场(LGA 和 JFK)，但实际上在亚特兰大、佐治亚州、AHN 和 MCN 附近有三个机场。

### [](#bonus-question)**加分题**

*离美国卡森国家森林最近的机场是哪个？*

**洛斯阿拉莫斯机场**

# [](#challenge-5-%f0%9f%8e%b8)**挑战#5🎸**

圣诞老人记得他还需要为他的“好”名单上的一个女孩挑选另一份礼物。小朱莉想要的圣诞礼物就是 2019 年 8 月 14 日至 8 月 18 日期间在同一座城市演出的音乐家的门票。

朱莉想在圣诞节见到的艺术家叫什么名字？

[**演唱会**](https://github.com/algolia/datasets/blob/master/concerts) 数据集对于 Algolia 免费层来说又太大了，所以我们需要使用托管版本。这个挑战是关于日期的。正如 Bryan 和我在我们的 [现场编码会议](https://youtu.be/h89gZei5fvo) 中发现的那样，日期很难确定。

Algolia 索引将日期存储为 Unix 时间戳，用于过滤和排名。我们需要建立一个界面，让人类更容易处理这些日期。我们选择使用 [Algolia 代码交换](https://www.algolia.com/developers/code-exchange/?query=date%20range%20picker&page=1) 中的日期范围选择器。从那里，只需几个过滤器就能查出这位艺术家是正在佐治亚州莱克伍德切莱里斯圆形剧场演奏的 **格莱姆斯** 。

### [](#bonus-question)**加分题**

*酷玩乐队 2018 年最后一次演出的城市是哪里？*

**法国巴黎**

# [](#conclusion-%f0%9f%8e%84)**结论🎄**

感谢冬季编码挑战中所有出色的参与者。我们将保持不和谐的服务器，所以让我们知道你是否想访问。尽管比赛已经结束，但这些仍然是你自己尝试建立的有趣挑战。

Algolia 全体员工祝您节日快乐，新年再见！🎊

在我们的开源代码交换平台上查看相关解决方案。

[T51](https://www.algolia.com/developers/code-exchange/?page=1&refinementList%5Bproduct_features%5D%5B0%5D=InstantSearch&refinementList%5Bproduct_features%5D%5B1%5D=Geo%20Search)